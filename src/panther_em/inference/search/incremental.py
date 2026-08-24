"""Multi-stage (multi-precision) incremental SVD-2DTM search."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from panther_em.inference.search.compressed import _run_stage, resolve_search_args
from panther_em.inference.search.staging import DEFAULT_STAGE_BYTES
from panther_em.inference.search.tiling import (
    FeatureTiling,
    FeaturizedImageStore,
    Rectangle,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from panther_em.inference.projection_reconstruction import ProjectionReconstructor


def compute_zscore_error_intervals(
    error: float,
    corr: torch.Tensor,  # (P,)
    mu: torch.Tensor,  # (P,)
    sigma: torch.Tensor,  # (P,)
) -> torch.Tensor:  # (P, 2)
    r"""Compute worst-case intervals for z-scores for a fixed reconstruction error.

    Parameters
    ----------
    error : float
        Estimated reconstruction error.
    corr : torch.Tensor
        Per-pixel correlation values, shape ``(P,)``, estimated from low-rank SVD.
    mu : torch.Tensor
        Per-pixel mean values, shape ``(P,)``, estimated from low-rank SVD.
    sigma : torch.Tensor
        Per-pixel standard deviation values, shape ``(P,)``, estimated from low-rank
        SVD.

    Returns
    -------
    torch.Tensor
        Per-pixel error intervals, shape ``(P, 2)``, where the second axis is
        ``[lower_bound, upper_bound]``. If error is correct, then interval guaranteed to
        contain the true z-score for each pixel.
    """
    sigma_min = sigma - error
    sigma_max = torch.sqrt(sigma**2 + 2 * error**2) + error
    sigma_min = torch.clamp(sigma_min, min=1e-6)
    sigma_max = torch.clamp(sigma_max, min=1e-6)

    a_max = 2 * error / sigma_min
    b_max = error * (2 * sigma_max + error) / (sigma * (sigma_min + sigma))

    zscore = (corr - mu) / sigma
    delta_z = (a_max + b_max * torch.abs(zscore)) / (1 - b_max)

    return torch.stack([zscore - delta_z, zscore + delta_z], dim=-1)


@torch.no_grad()
def incremental_search(
    image: torch.Tensor,
    reconstructor: ProjectionReconstructor,
    stage_rectangles: Sequence[Sequence[Rectangle]],
    *,
    pixel_batch: int,
    hyp_batch: int,
    n_psi: int,
    feature_chunk: int,
    store: FeaturizedImageStore | None = None,
    hypothesis_indexes: torch.Tensor | None = None,
    pixel_index: torch.Tensor | None = None,
    compute_device: torch.device | str | None = None,
    feature_store_device: torch.device | str | None = None,
    follow_up_fn: (
        Callable[[dict[str, torch.Tensor], torch.Tensor], torch.Tensor | None] | None
    ) = None,
    use_fused_kernel: bool = True,
    stage_bytes: int = DEFAULT_STAGE_BYTES,
    **polar_to_cart_kwargs: Any,
) -> Iterator[dict[str, torch.Tensor]]:
    r"""Incremental SVD-2DTM search; yields per-pixel statistics per stage.

    Implements the multi-precision strategy: each stage selects a (typically higher-
    rank) set of contiguous feature-space rectangles and runs one compressed search
    (:func:`_run_stage`), reusing the image features computed in earlier stages. Between
    stages an external caller narrows the pixel set through ``follow_up_fn``.

    See :class:`panther_em.inference.search.tracking.MultiPrecisionPixelTracker` for a
    ready-made error-aware ``follow_up_fn`` implementing pixel accept/reject/absorb
    partitioning across stages, built on :func:`compute_zscore_error_intervals` above.

    Parameters
    ----------
    image : torch.Tensor
        Real image of shape ``(H, W)``.
    reconstructor : ProjectionReconstructor
        Holds the SVD tensors, polar transform, and device.
    stage_rectangles : Sequence[Sequence[Rectangle]]
        One list of ``(k_start, m_start, k_extent, m_extent)`` rectangles per stage.
        Rectangles within a stage must be pairwise cell-disjoint.
    pixel_batch : int
        Batch size for the pixel loop.
    hyp_batch : int
        Batch size for the hypothesis loop.
    n_psi : int
        Number of in-plane samples.
    feature_chunk : int
        Kernel-correlation chunk size for featurizing missing cells.
    store : FeaturizedImageStore, optional
        Persistent feature store to reuse across calls. A fresh one sized to the image's
        valid-correlation grid is created when omitted.
    hypothesis_indexes : torch.Tensor, optional
        Long indices into the flattened ``(FF * O)`` template space to search. Defaults
        to all templates.
    pixel_index : torch.Tensor, optional
        Long indices into the ``P`` valid-correlation pixels for the *first* stage (e.g.
        an initial mask). Defaults to all pixels. Subsequent stages use the mask
        returned by ``follow_up_fn``.
    compute_device : torch.device or str, optional
        Device for the contraction, ``irfft``, and per-pixel statistics. Defaults to
        ``reconstructor.device``.
    feature_store_device : torch.device or str, optional
        Device the persistent feature store is held on. Defaults to ``compute_device``.
        Set to ``"cpu"`` to keep large feature stacks off the compute device; each
        pixel batch is streamed back to ``compute_device`` as it is processed. Ignored
        when an explicit ``store`` is supplied.
    follow_up_fn : callable, optional
        ``(stats, pixel_index) -> next_pixel_index | None`` mapping a stage's finalized
        statistics and the pixels it ran on to the follow-up pixel mask for the next
        stage. Returning ``None`` (the default behaviour when omitted) keeps the same
        pixel set.
    use_fused_kernel : bool, optional
        Attempt the fused CUDA iRFFT+stats kernel (see
        :class:`~panther_em.inference.search.fused_statistics.FusedPixelStats`),
        falling back to the pure-torch reduction whenever the kernel is unavailable
        or ``(n_psi, k_stop)`` is unsupported. Defaults to ``True``.
    stage_bytes : int, optional
        Byte budget per staging buffer for the pinned, double-buffered,
        asynchronous transfer of the feature store's rows to ``compute_device``
        (see :class:`~panther_em.inference.search.staging.PixelStager`). Only
        relevant when ``feature_store_device`` differs from ``compute_device``.
    **polar_to_cart_kwargs
        Forwarded to kernel construction when featurizing cells.

    Yields
    ------
    dict[str, torch.Tensor]
        Per stage, the finalized :meth:`PixelStats.finalize` maps (``"mip"``,
        ``"zscore"``, ``"mean"``, ``"variance"``, ``"best_index"``, ``"best_psi"``)
        each of shape ``(P_stage,)``, indexed by the stage's pixel set.
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
    n_px, hypothesis_indexes, pixel_index = resolve_search_args(
        image, reconstructor, hypothesis_indexes, pixel_index
    )

    if store is None:
        store = FeaturizedImageStore(n_px, device=feature_store_device)

    px = pixel_index
    for rects in stage_rectangles:  # stages -- increasing rank
        if px.numel() == 0:
            return
        tiling = FeatureTiling.from_extents(rects, device=compute_device)
        stage_maps = _run_stage(
            image,
            reconstructor,
            tiling,
            store,
            hypothesis_indexes=hypothesis_indexes,
            pixel_index=px,
            pixel_batch=pixel_batch,
            hyp_batch=hyp_batch,
            n_psi=n_psi,
            feature_chunk=feature_chunk,
            compute_device=compute_device,
            use_fused_kernel=use_fused_kernel,
            stage_bytes=stage_bytes,
            **polar_to_cart_kwargs,
        )

        yield stage_maps

        # Hand the follow-up mask (selection logic) back to the next stage.
        if follow_up_fn is not None:
            nxt = follow_up_fn(stage_maps, px)
            if nxt is not None:
                px = nxt
