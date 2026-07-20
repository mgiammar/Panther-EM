"""Single-stage compressed SVD-2DTM search.

A "compressed" search evaluates one :class:`FeatureTiling` -- a single selection
of contiguous ``(k, m)`` rectangles -- against an image and reduces the result to
per-pixel statistics. It is the building block of the multi-stage
:func:`panther_em.inference.search.incremental.incremental_search`: one stage of
the incremental driver is exactly one compressed search that reuses an existing
feature store. The shared engine is :func:`_run_stage`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from panther_em.inference.search.statistics import PixelStats
from panther_em.inference.search.tiling import (
    FeatureTiling,
    FeaturizedImageStore,
    Rectangle,
)
from panther_em.inference.search.utils import build_layout_weights, featurize_cells

if TYPE_CHECKING:
    from panther_em.inference.projection_reconstruction import ProjectionReconstructor


def _searchable_pixels(
    image: torch.Tensor, reconstructor: ProjectionReconstructor
) -> int:
    """Number of valid cross-correlation output pixels ``P`` for ``image``."""
    k_h, k_w = reconstructor.image_shape
    out_h = image.shape[-2] - k_h + 1
    out_w = image.shape[-1] - k_w + 1
    return int(out_h * out_w)


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
        Real image of shape ``(..., H, W)``.
    reconstructor : ProjectionReconstructor
        Holds the SVD tensors, polar transform, and device.
    hypothesis_indexes : torch.Tensor | None
        Long indices into the flattened hypothesis space. By default is ``None`` which
        includes all hypotheses. Set to a non-None long tensor to restrict the search to
        a subset of hypotheses.
    pixel_index : torch.Tensor | None
        Long indices into the ``P`` valid-correlation pixels. By default is ``None``
        which includes all pixels. Set to a non-None long tensor to restrict the search
        to a subset of pixels.

    Returns
    -------
    n_px : int
        Number of valid cross-correlation pixels.
    hypothesis_indexes : torch.Tensor
        Long indices into the flattened hypothesis space.
    pixel_index : torch.Tensor
        Long indices into the ``P`` valid-correlation pixels.
    n_psi : int
        Number of in-plane samples.
    """
    result = reconstructor.result
    device = reconstructor.device

    n_px = _searchable_pixels(image, reconstructor)

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
    **polar_to_cart_kwargs
        Forwarded to kernel construction when featurizing cells.

    Returns
    -------
    dict[str, torch.Tensor]
        The finalized :meth:`PixelStats.finalize` maps (``"mip"``, ``"zscore"``,
        ``"mean"``, ``"variance"``, ``"best_index"``, ``"best_psi"``) each of shape
        ``(P_stage,)``, indexed by ``pixel_index``.
    """
    device = image.device
    result = reconstructor.result

    # image feature reuse to avoid re-computation
    features_to_compute = store.missing_cells(tiling)
    if features_to_compute.numel():
        feats = featurize_cells(
            image,
            reconstructor,
            features_to_compute,
            feature_chunk=feature_chunk,
            **polar_to_cart_kwargs,
        )
    else:
        feats = torch.empty(
            (0, store.num_pixels), dtype=store.dtype, device=store.device
        )
    store.relayout(tiling, feats)

    stage_stats: list[dict[str, torch.Tensor]] = []
    w_layout = build_layout_weights(reconstructor, tiling)
    pixel_stats = PixelStats(pixel_batch, device=device)

    for p0 in range(0, int(pixel_index.numel()), pixel_batch):
        px_b = pixel_index[p0 : p0 + pixel_batch]
        Y_flat = store.image_view(px_b)

        # If not 'pixel_batch' shape, create new 'pixel_stats'
        if Y_flat.shape[0] != pixel_batch:
            pixel_stats = PixelStats(int(px_b.numel()), device=device)
        else:
            pixel_stats.clear()

        for h0 in range(0, int(hypothesis_indexes.numel()), hyp_batch):
            hyp_b = hypothesis_indexes[h0 : h0 + hyp_batch]
            W_flat = w_layout[hyp_b]  # (N_b, r)

            # Accumulate correlogram rotational frequency spectrum (C)
            C = tiling.run(Y_flat, W_flat, result.k_max)
            corr = torch.fft.irfft(C.conj(), n=n_psi, dim=-1, norm="forward")
            pixel_stats.update(corr, hyp_b)

        stage_stats.append(pixel_stats.finalize())

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
    **polar_to_cart_kwargs: Any,
) -> dict[str, torch.Tensor]:
    r"""Single-stage SVD-2DTM search over one selection of feature rectangles.

    Builds a fresh feature store, featurizes the image for the selected ``(k, m)``
    cells, and reduces the search over hypotheses and in-plane angles to per-pixel
    statistics. For a multi-stage (multi-precision) search that reuses features
    across selections, see
    :func:`panther_em.inference.search.incremental.incremental_search`.

    Parameters
    ----------
    image : torch.Tensor
        Real image of shape ``(H, W)``.
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
        Long indices into the ``P`` valid-correlation pixels (e.g. a mask). Defaults
        to all pixels.
    **polar_to_cart_kwargs
        Forwarded to kernel construction.

    Returns
    -------
    dict[str, torch.Tensor]
        The finalized per-pixel statistic maps (``"mip"``, ``"zscore"``, ``"mean"``,
        ``"variance"``, ``"best_index"``, ``"best_psi"``), each of shape
        ``(P_stage,)`` indexed by ``pixel_index``.
    """
    device = reconstructor.device
    n_px, hypothesis_indexes, pixel_index = resolve_search_args(
        image, reconstructor, hypothesis_indexes, pixel_index
    )

    store = FeaturizedImageStore(n_px, device=device)
    tiling = FeatureTiling.from_extents(rectangles, device=device)

    result: dict[str, torch.Tensor] = _run_stage(
        image,
        reconstructor,
        tiling,
        store,
        hypothesis_indexes=hypothesis_indexes,
        pixel_index=pixel_index,
        pixel_batch=pixel_batch,
        hyp_batch=hyp_batch,
        n_psi=n_psi,
        feature_chunk=feature_chunk,
        **polar_to_cart_kwargs,
    )

    return result
