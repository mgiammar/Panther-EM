"""Single-stage compressed SVD-2DTM search.

A "compressed" search evaluates one :class:`FeatureTiling` -- a single selection
of contiguous ``(k, m)`` rectangles -- against an image and reduces the result to
per-pixel statistics. It is the building block of the multi-stage
:func:`panther_em.inference.search.incremental.incremental_search`: one stage of
the incremental driver is exactly one compressed search that reuses an existing
feature store. The shared engine is :func:`_run_stage`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from tqdm import tqdm

from panther_em.inference.search.statistics import PixelStats
from panther_em.inference.search.tiling import (
    FeatureTiling,
    FeaturizedImageStore,
    Rectangle,
)
from panther_em.inference.search.utils import build_layout_weights, featurize_cells

if TYPE_CHECKING:
    from panther_em.inference.projection_reconstruction import ProjectionReconstructor


def _search_output_shape(
    image: torch.Tensor, reconstructor: ProjectionReconstructor
) -> tuple[int, int, int]:
    """``(B, out_h, out_w)`` valid cross-corr grid for a ``(..., H, W)`` image."""
    leading_dims = image.shape[:-2] if image.dim() > 2 else None

    k_h, k_w = reconstructor.image_shape
    out_h = int(image.shape[-2] - k_h + 1)
    out_w = int(image.shape[-1] - k_w + 1)
    b = math.prod(leading_dims) if leading_dims is not None else 1

    return int(b), out_h, out_w


def _resolve_pixel_layout(
    pixel_index: torch.Tensor | None,
    leading: tuple[int, ...],
    batch: int,
    n_px_per_image: int,
    out_h: int,
    out_w: int,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, ...] | None]:
    """Map the caller's pixel selection to a flat ``batch * P`` index + output reshape.

    Parameters
    ----------
    pixel_index : torch.Tensor | None
        ``None`` for the whole search. A 1D tensor is a single per-image pixel
        selection (indices into ``[0, P)``) applied identically to every batch element.
        A tensor with **more than one dimension** is treated as caller-owned flat
        indices into the ``batch * P`` axis.
    leading : tuple[int, ...]
        The image's leading batch dimensions (``()`` for a single ``(H, W)`` image).
    batch : int
        ``math.prod(leading)`` -- the flattened batch size.
    n_px_per_image : int
        ``P = out_h * out_w`` valid-correlation pixels per image.
    out_h, out_w : int
        Valid-correlation grid dimensions.
    device : torch.device
        Device for the built index tensors.

    Returns
    -------
    flat_pixel_index : torch.Tensor
        Long indices into the flattened ``batch * P`` pixel axis for :func:`_run_stage`.
    reshape_shape : tuple[int, ...] | None
        Target shape for every returned map, or ``None`` to leave the maps flat (the
        caller is responsible for placing the values).
    """
    # All pixels selected
    if pixel_index is None:
        flat = torch.arange(batch * n_px_per_image, device=device)
        return flat, (*leading, out_h, out_w)

    pixel_index = pixel_index.to(device=device, dtype=torch.long)

    # Consistent mask across all batches, reshape to (*leading, n_mask) for the maps
    if pixel_index.dim() == 1:
        n_mask = int(pixel_index.numel())
        offsets = (torch.arange(batch, device=device) * n_px_per_image).view(batch, 1)
        flat = (offsets + pixel_index.view(1, n_mask)).reshape(-1)
        return flat, (*leading, n_mask)

    # A higher-dimensional (inconsistent / per-batch) selection
    return pixel_index.reshape(-1), None


def resolve_search_args(
    image: torch.Tensor,
    reconstructor: ProjectionReconstructor,
    hypothesis_indexes: torch.Tensor | None,
    pixel_index: torch.Tensor | None,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Fill in the whole-search defaults shared by compressed / incremental search.

    Parameters
    ----------
    image : torch.Tensor
        Real image of shape ``(H, W)`` or a batch ``(B, H, W)``.
    reconstructor : ProjectionReconstructor
        Holds the SVD tensors, polar transform, and device.
    hypothesis_indexes : torch.Tensor | None
        Long indices into the flattened hypothesis space. By default is ``None`` which
        includes all hypotheses. Set to a non-None long tensor to restrict the search to
        a subset of hypotheses.
    pixel_index : torch.Tensor | None
        Long indices into the ``B * P`` valid-correlation pixels (batch flattened into
        the pixel axis, batch-major). By default is ``None`` which includes all pixels.
        Set to a non-None long tensor to restrict the search to a subset of pixels.

    Returns
    -------
    n_px : int
        Total number of valid cross-correlation pixels ``B * out_h * out_w``.
    hypothesis_indexes : torch.Tensor
        Long indices into the flattened hypothesis space.
    pixel_index : torch.Tensor
        Long indices into the ``P`` valid-correlation pixels.
    n_psi : int
        Number of in-plane samples.
    """
    result = reconstructor.result
    device = reconstructor.device

    b, out_h, out_w = _search_output_shape(image, reconstructor)
    n_px = b * out_h * out_w

    if hypothesis_indexes is None:
        hypothesis_indexes = torch.arange(
            result.num_fourier_filters * result.num_orientations, device=device
        )

    if pixel_index is None:
        pixel_index = torch.arange(n_px, device=device)

    return n_px, hypothesis_indexes, pixel_index


@torch.no_grad()
def _run_stage(
    image: torch.Tensor,
    reconstructor: ProjectionReconstructor,
    tiling: FeatureTiling,
    store: FeaturizedImageStore,
    *,
    pixel_batch: int,
    hyp_batch: int,
    n_psi: int,
    feature_chunk: int,
    hypothesis_indexes: torch.Tensor,
    pixel_index: torch.Tensor,
    compute_device: torch.device | str | None = None,
    show_progress: bool = True,
    **polar_to_cart_kwargs: Any,
) -> dict[str, torch.Tensor]:
    r"""Run one tiling end-to-end and reduce it to per-pixel statistics.

    The nested loops, outermost to innermost, are:

    1.  **Pixel batches** -- bound the transient ``(P_b, N, N_\psi)`` correlogram.
    2.  **Hypothesis batches** -- ``W`` is sliced per batch; statistics accumulate
        online via :class:`PixelStats`.
    3.  **Regions (innermost)** -- :meth:`FeatureTiling.run` sweeps the rectangular
        regions, each a batched ``bmm`` accumulating into the spectrum ``C``.

    Parameters
    ----------
    image : torch.Tensor
        Real image of shape ``(..., H, W)``. Compute device is drawn from image device.
    reconstructor : ProjectionReconstructor
        Holds the SVD tensors, polar transform.
    tiling : FeatureTiling
        The stage's feature selection (built with :meth:`FeatureTiling.from_extents`).
    store : FeaturizedImageStore
        Persistent feature store. Only cells missing from it are featurized.
    pixel_batch : int
        Batch size for the pixel loop.
    hyp_batch : int
        Batch size for the hypothesis loop.
    n_psi : int
        Number of in-plane samples.
    feature_chunk : int
        Kernel-correlation chunk size for featurizing missing cells.
    hypothesis_indexes : torch.Tensor
        Long indices into the flattened hypothesis space.
    pixel_index : torch.Tensor
        Long indices into the ``P`` valid-correlation pixels to search.
    compute_device : torch.device or str, optional
        Device for the contraction (``bmm``), ``irfft``, and online statistics. The
        contraction weights and each pixel batch are moved here. Defaults to
        ``reconstructor.device``.
    show_progress : bool, optional
        Show tqdm progress bars over the pixel-batch and hypothesis-batch loops.
        Defaults to ``True``.
    **polar_to_cart_kwargs
        Forwarded to kernel construction when featurizing cells.

    Returns
    -------
    dict[str, torch.Tensor]
        The finalized :meth:`PixelStats.finalize` maps (``"mip"``, ``"zscore"``,
        ``"mean"``, ``"variance"``, ``"best_index"``, ``"best_psi"``) each of shape
        ``(P_stage,)``, indexed by ``pixel_index``.
    """
    compute_device = (
        torch.device(compute_device)
        if compute_device is not None
        else reconstructor.device
    )

    # image feature reuse to avoid re-computation
    features_to_compute = store.missing_cells(tiling)
    if features_to_compute.numel():
        feats = featurize_cells(
            image,
            reconstructor,
            features_to_compute,
            feature_chunk=feature_chunk,
            feature_store_device=store.device,
            **polar_to_cart_kwargs,
        )
    else:
        feats = torch.empty(
            (0, store.num_pixels), dtype=store.dtype, device=store.device
        )
    store.relayout(tiling, feats)

    stage_stats: list[dict[str, torch.Tensor]] = []
    w_layout = build_layout_weights(reconstructor, tiling).to(compute_device)
    hypothesis_indexes = hypothesis_indexes.to(compute_device)
    pixel_index = pixel_index.to(compute_device)
    pixel_stats = PixelStats(pixel_batch, device=compute_device)

    n_pixels = int(pixel_index.numel())
    n_hypotheses = int(hypothesis_indexes.numel())

    # Progress bars advance by the number of pixels / hypotheses actually processed
    # each batch, so their rate reads in pixels/s and hypotheses/s (not batches/s).
    pixel_bar = tqdm(
        total=n_pixels,
        desc="search pixels",
        unit="pixel",
        unit_scale=True,
        disable=not show_progress,
    )

    for p0 in range(0, n_pixels, pixel_batch):
        px_b = pixel_index[p0 : p0 + pixel_batch]
        Y_flat = store.image_view(px_b)
        if Y_flat.device != compute_device:
            Y_flat = Y_flat.to(compute_device, non_blocking=True)

        # If not 'pixel_batch' shape, create new 'pixel_stats'
        if Y_flat.shape[0] != pixel_batch:
            pixel_stats = PixelStats(int(px_b.numel()), device=compute_device)
        else:
            pixel_stats.clear()

        hyp_bar = tqdm(
            total=n_hypotheses,
            desc="hypotheses",
            unit="hypothesis",
            unit_scale=True,
            disable=not show_progress,
            leave=False,
        )
        for h0 in range(0, n_hypotheses, hyp_batch):
            hyp_b = hypothesis_indexes[h0 : h0 + hyp_batch]
            W_flat = w_layout[hyp_b]  # (N_b, r)

            # Accumulate correlogram rotational frequency spectrum (C)
            C = tiling.run(Y_flat, W_flat, result.k_max)
            corr = torch.fft.irfft(C.conj(), n=n_psi, dim=-1, norm="forward")
            pixel_stats.update(corr, hyp_b)

            hyp_bar.update(int(hyp_b.numel()))

        hyp_bar.close()

        stage_stats.append(pixel_stats.finalize())
        pixel_bar.update(int(px_b.numel()))

    pixel_bar.close()

    # Stitch the pixel batches back into stage-level maps.
    # NOTE: This return may change into a different dict or helper class in future...
    return {
        key: torch.cat([s[key] for s in stage_stats], dim=0) for key in stage_stats[0]
    }


@torch.no_grad()
def compressed_search(
    image: torch.Tensor,
    reconstructor: ProjectionReconstructor,
    rectangles: list[Rectangle],
    *,
    pixel_batch: int,
    hyp_batch: int,
    n_psi: int,
    feature_chunk: int,
    hypothesis_indexes: torch.Tensor | None = None,
    pixel_index: torch.Tensor | None = None,
    compute_device: torch.device | str | None = None,
    feature_store_device: torch.device | str | None = None,
    show_progress: bool = True,
    **polar_to_cart_kwargs: Any,
) -> dict[str, torch.Tensor]:
    r"""Single-stage SVD-2DTM search over one selection of feature rectangles.

    Builds a fresh feature store, featurizes the image for the selected ``(k, m)``
    cells, and reduces the search over hypotheses and in-plane angles to per-pixel
    statistics. For a multi-stage (multi-precision) search that reuses features across
    selections, see
    :func:`panther_em.inference.search.incremental.incremental_search`.

    Parameters
    ----------
    image : torch.Tensor
        Real image of shape ``(H, W)`` or a batch ``(*leading, H, W)`` with any number
        of leading batch dimensions. All batch images must share the same valid
        correlation grid; the leading dims are flattened into the pixel axis (batch
        major) and reduced independently per pixel.
    reconstructor : ProjectionReconstructor
        Holds the SVD tensors, polar transform, and device.
    rectangles : list[Rectangle]
        ``(k_start, m_start, k_extent, m_extent)`` blocks of feature space. Must be
        pairwise cell-disjoint; overshoot (extra low-value cells) is fine and is not
        zeroed.
    pixel_batch : int
        Batch size for the pixel loop.
    hyp_batch : int
        Batch size for the hypothesis loop.
    n_psi : int, optional
        Number of in-plane samples.
    feature_chunk : int
        Kernel-correlation chunk size for featurizing cells.
    hypothesis_indexes : torch.Tensor, optional
        Long indices into the flattened ``(FF * O)`` template space to search.
        Defaults to all templates.
    pixel_index : torch.Tensor, optional
        Pixel selection (mask). ``None`` (default) searches all pixels.
    compute_device : torch.device or str, optional
        Device for the contraction, ``irfft``, and per-pixel statistics. Defaults to
        ``reconstructor.device``.
    feature_store_device : torch.device or str, optional
        Device the featurized image stack is held on between the featurization and
        contraction stages. Defaults to ``compute_device``. Set to ``"cpu"`` for large
        images or image stacks to keep the ``(B * P, r)``.
    show_progress : bool, optional
        Show tqdm progress bars over the pixel-batch and hypothesis-batch loops.
        Defaults to ``True``.
    **polar_to_cart_kwargs
        Forwarded to kernel construction.

    Returns
    -------
    dict[str, torch.Tensor]
        The finalized per-pixel statistic maps (``"mip"``, ``"zscore"``, ``"mean"``,
        ``"variance"``, ``"best_index"``, ``"best_psi"``). Each map's shape depends on
        the image rank and the ``pixel_index`` layout:

        * no mask -> ``(*leading, out_h, out_w)`` (a single ``(H, W)`` image gives the
          plain ``(out_h, out_w)`` grid);
        * a 1D mask of ``n_mask`` pixels -> ``(*leading, n_mask)`` (a single image
          gives ``(n_mask,)``);
        * a multi-dimensional mask -> flat ``(n_selected,)``. Caller responsible for any
          reshaping.
    """
    compute_device = (
        torch.device(compute_device)
        if compute_device is not None
        else reconstructor.device
    )
    feature_store_device = (
        torch.device(feature_store_device)
        if feature_store_device is not None
        else compute_device
    )

    # Flatten any leading batch dimensions into (B, H, W)
    leading = tuple(image.shape[:-2])
    h, w = int(image.shape[-2]), int(image.shape[-1])
    batch = math.prod(leading) if leading else 1
    image_bhw = image.reshape(batch, h, w)

    _, out_h, out_w = _search_output_shape(image_bhw, reconstructor)
    n_px_per_image = out_h * out_w
    n_px = batch * n_px_per_image

    if hypothesis_indexes is None:
        meta = reconstructor.result
        hypothesis_indexes = torch.arange(
            meta.num_fourier_filters * meta.num_orientations, device=compute_device
        )

    flat_pixel_index, reshape_shape = _resolve_pixel_layout(
        pixel_index, leading, batch, n_px_per_image, out_h, out_w, compute_device
    )

    store = FeaturizedImageStore(n_px, device=feature_store_device)
    tiling = FeatureTiling.from_extents(rectangles, device=compute_device)

    result: dict[str, torch.Tensor] = _run_stage(
        image_bhw,
        reconstructor,
        tiling,
        store,
        hypothesis_indexes=hypothesis_indexes,
        pixel_index=flat_pixel_index,
        pixel_batch=pixel_batch,
        hyp_batch=hyp_batch,
        n_psi=n_psi,
        feature_chunk=feature_chunk,
        compute_device=compute_device,
        show_progress=show_progress,
        **polar_to_cart_kwargs,
    )

    # Restore leading batch dims + the valid-correlation grid where determinable.
    if reshape_shape is not None:
        result = {key: value.reshape(reshape_shape) for key, value in result.items()}

    return result
