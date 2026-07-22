"""Reconstructor-facing helpers: kernels, image featurization, and weights."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from tqdm import tqdm

from panther_em.inference.correlation import (
    build_cartesian_kernels,
    compute_feature_stack,
)

if TYPE_CHECKING:
    from panther_em.inference.projection_reconstruction import ProjectionReconstructor
    from panther_em.inference.search.tiling import FeatureTiling


def _ensure_bhw(image: torch.Tensor) -> torch.Tensor:
    """Coerce ``(H, W)`` or ``(B, H, W)`` to ``(B, H, W)``, error otherwise."""
    if image.dim() == 2:
        return image.unsqueeze(0)
    if image.dim() == 3:
        return image
    raise ValueError(
        f"image must have shape (H, W) or (B, H, W), got {tuple(image.shape)}"
    )


def psi_degrees(n_psi: int, device: torch.device | str) -> torch.Tensor:
    """Uniformly spaced in-plane angles (degrees) from 0 to 360."""
    return 360.0 * torch.arange(n_psi, device=device, dtype=torch.float32) / n_psi


def build_block_kernels(
    reconstructor: ProjectionReconstructor,
    indices: torch.Tensor,
    **polar_to_cart_kwargs: dict[str, Any],
) -> torch.Tensor:
    r"""Build Cartesian feature kernels for arbitrary (k_idx, eig_idx) pairs.

    Parameters
    ----------
    reconstructor : ProjectionReconstructor
        Supplies the polar transform and stored SVD tensors.
    indices : torch.Tensor
        Shape ``(r, 2)`` integer array; each row is ``(k_idx, eig_idx)``.
        For real-valued decompositions, ``k_idx`` must be in ``[0, k_max]``.
    **polar_to_cart_kwargs
        Forwarded to
        :meth:`ProjectionReconstructor.construct_cartesian_feature` (e.g. ``order``,
        ``mode``, ``preserve_energy``).

    Returns
    -------
    torch.Tensor
        Complex64 kernels of shape ``(r, kH, kW)`` on the device, indexed by
        the input ``indices`` array.

    Raises
    ------
    NotImplementedError
        If the result is a complex-valued projection.
    """
    if reconstructor.result.is_complex_projection:
        raise NotImplementedError("Complex-valued projections not yet supported")

    return build_cartesian_kernels(reconstructor, indices, **polar_to_cart_kwargs)


def build_multichannel_correlogram(
    image: torch.Tensor,  # shape (B, H, W)
    kernels: torch.Tensor,  # shape (r, kH, kW)
    *,
    out: torch.Tensor | None = None,  # shape (B, r, out_H, out_W)
    feature_chunk: int | None = None,
    show_progress: bool = True,
) -> torch.Tensor:  # shape (B, r, out_H, out_W)
    r"""Cross-correlate an image against a Cartesian feature stack.

    Parameters
    ----------
    image : torch.Tensor
        Real image of shape ``(B, H, W)``.
    kernels : torch.Tensor
        Cartesian feature kernels of shape ``(r, kH, kW)`` as returned by
        :func:`build_block_kernels`.
    out : torch.Tensor, optional
        Optional pre-allocated output tensor of shape ``(B, r, out_H, out_W)``.
        NOTE: May be a different device than ``image`` or ``kernels`` to manage large
        tensors.
    feature_chunk : int, optional
        Number of kernels to correlate per FFT batch, bounding transient FFT memory.
        Defaults to all kernels at once.
    show_progress : bool, optional
        Show a progress bar for the feature correlation loop. Defaults to True.

    Returns
    -------
    torch.Tensor
        Complex64 stack ``Z`` of shape ``(B, r, out_H, out_W)`` on the kernels'
        device, where ``out_H = H - kH + 1`` and ``out_W = W - kW + 1``.

    Raises
    ------
    ValueError
        If ``image`` is smaller than the kernel box (empty valid correlation).
    """
    compute_device = kernels.device

    if kernels.dim() != 3:
        raise ValueError(
            f"kernels must have shape (r, kH, kW), got {tuple(kernels.shape)}"
        )

    r, k_h, k_w = kernels.shape

    # Compute expected shapes output shapes
    image = _ensure_bhw(image).to(compute_device)
    b = image.shape[0]
    out_h = image.shape[-2] - k_h + 1
    out_w = image.shape[-1] - k_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"image {tuple(image.shape[-2:])} smaller than kernel {(k_h, k_w)}; "
            "valid cross-correlation is empty."
        )

    # Check if out is provided and compatible
    if out is not None:
        if out.shape != (b, r, out_h, out_w):
            raise ValueError(
                f"out has shape {tuple(out.shape)}, expected {(b, r, out_h, out_w)}"
            )
        if out.dtype != torch.complex64:
            raise ValueError(f"out must have dtype complex64, got {out.dtype}")
        z_flat = out
    else:
        z_flat = torch.empty(
            (b, r, out_h, out_w), dtype=torch.complex64, device=compute_device
        )

    chunk = feature_chunk if feature_chunk is not None else r
    feat_bar = tqdm(
        range(0, r, chunk),
        desc="correlating features",
        unit="features",
        unit_scale=True,
        disable=not show_progress,
        total=r
    )
    for start in range(0, r, chunk):
        stop = min(start + chunk, r)
        tmp = compute_feature_stack(image, kernels[start:stop])
        z_flat[:, start:stop] = tmp.to(z_flat.device)  # NOTE: may be different device

        feat_bar.update(stop - start)

    return z_flat


def build_block_feature_stack(
    image: torch.Tensor,
    reconstructor: ProjectionReconstructor,
    *,
    indices: torch.Tensor,
    out: torch.Tensor | None = None,
    feature_chunk: int | None = None,
    **polar_to_cart_kwargs: dict[str, Any],
) -> torch.Tensor:
    r"""Convenience: build kernels and the multi-channel correlogram in one call.

    Equivalent to :func:`build_block_kernels` followed by
    :func:`build_multichannel_correlogram`. Use the two-step form when the kernels
    should be cached across images.

    Parameters
    ----------
    image : torch.Tensor
        Real image of shape ``(H, W)`` or ``(B, H, W)``.
    reconstructor : ProjectionReconstructor
        Builds the Cartesian kernels and supplies the polar transform / device.
    indices : torch.Tensor
        Shape ``(r, 2)`` integer array; each row is ``(k_idx, eig_idx)``.
    out : torch.Tensor, optional
        Optional pre-allocated output tensor of shape ``(B, r, out_H, out_W)``. May
        be on different device than the kernels or image to manage large tensors.
    feature_chunk : int, optional
        Kernel correlation chunk size (see
        :func:`build_multichannel_correlogram`).
    **polar_to_cart_kwargs
        Forwarded to kernel construction.

    Returns
    -------
    torch.Tensor
        Complex64 stack ``(B, r, out_H, out_W)``, can be different device than input
        image if ``out`` is provided and on a different device.
    """
    kernels = build_block_kernels(reconstructor, indices, **polar_to_cart_kwargs)
    return build_multichannel_correlogram(
        image, kernels, feature_chunk=feature_chunk, out=out
    )


def featurize_cells(
    image: torch.Tensor,
    reconstructor: ProjectionReconstructor,
    cells: torch.Tensor,
    *,
    feature_chunk: int | None = None,
    feature_store_device: torch.device | str | None = None,
    **polar_to_cart_kwargs: Any,
) -> torch.Tensor:
    """Cross-correlate an image against the kernels for a set of feature cells.

    Parameters
    ----------
    image : torch.Tensor
        Real image of shape ``(H, W)`` (single image) or ``(B, H, W)`` (a batch of
        images sharing the same valid-correlation grid ``P = out_h * out_w``).
    reconstructor : ProjectionReconstructor
        Builds the kernels and supplies the polar transform / device.
    cells : torch.Tensor
        ``(n_cells, 2)`` long tensor of ``(k, m)`` cells to featurize, e.g. from
        :meth:`FeaturizedImageStore.missing_cells`.
    feature_chunk : int, optional
        Kernel-correlation chunk size (see :func:`build_multichannel_correlogram`).
    feature_store_device : torch.device or str, optional
        Device the returned features should land on. When set (e.g. ``"cpu"`` while
        featurizing on the GPU), the output stack is allocated there and each
        ``feature_chunk`` is copied over as it is computed. When omitted the features
        stay on the kernels' (reconstructor's) device.
    **polar_to_cart_kwargs
        Forwarded to kernel construction.

    Returns
    -------
    torch.Tensor
        Complex features of shape ``(n_cells, B * P)`` on ``feature_store_device``
        (or the reconstructor's device when that is ``None``).
    """
    out = None
    if feature_store_device is not None:
        image_bhw = _ensure_bhw(image)
        k_h, k_w = reconstructor.image_shape
        b = int(image_bhw.shape[0])
        out_h = int(image_bhw.shape[-2] - k_h + 1)
        out_w = int(image_bhw.shape[-1] - k_w + 1)
        out = torch.empty(
            (b, int(cells.shape[0]), out_h, out_w),
            dtype=torch.complex64,
            device=torch.device(feature_store_device),
        )

    z = build_block_feature_stack(
        image,
        reconstructor,
        indices=cells.to(torch.long),
        feature_chunk=feature_chunk,
        out=out,
        **polar_to_cart_kwargs,
    )

    b, n_cells, out_h, out_w = z.shape

    # (B, n_cells, out_h, out_w) -> (n_cells, B * out_h * out_w), batch-major pixels.
    return z.permute(1, 0, 2, 3).reshape(n_cells, b * out_h * out_w)


def build_block_weights(
    reconstructor: ProjectionReconstructor,
    indices: torch.Tensor,
) -> torch.Tensor:
    r"""Build contraction weights ``W = U * S`` for arbitrary (k_idx, eig_idx) pairs.

    Parameters
    ----------
    reconstructor : ProjectionReconstructor
        Provides the on-device ``U``/``S`` tensors and the result metadata.
    indices : torch.Tensor
        Shape ``(r, 2)`` integer array; each row is ``(k_idx, eig_idx)``.
        For real-valued decompositions, ``k_idx`` must be in ``[0, k_max]``.

    Returns
    -------
    torch.Tensor
        Complex64 weights of shape ``(FF, O, r)`` on ``reconstructor.device``.

    Raises
    ------
    NotImplementedError
        If the result is a complex-valued projection.
    """
    result = reconstructor.result

    if result.is_complex_projection:
        raise NotImplementedError("Complex-valued projections not yet supported")

    U, S, _ = result.get_svd_tensors(indices.cpu().numpy(), reconstructor.device)
    # U: (FF, O, r),  S: (r,)
    return U * S[None, None, :]


def build_layout_weights(
    reconstructor: ProjectionReconstructor,
    tiling: FeatureTiling,
    hypothesis_indexes: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Build ``W = U * S`` weights laid out to match a tiling's feature dimension.

    Parameters
    ----------
    reconstructor : ProjectionReconstructor
        Supplies the on-device ``U`` / ``S`` tensors.
    tiling : FeatureTiling
        Defines the compacted feature dimension via :attr:`FeatureTiling.layout`.
    hypothesis_indexes : torch.Tensor, optional
        Long indices selecting a subset of the flattened ``(FF * O)`` template rows
        (the hypothesis batch). Defaults to all rows.

    Returns
    -------
    torch.Tensor
        Complex64 weights of shape ``(N, r)`` where ``r = tiling.num_features`` and
        ``N`` is the number of selected hypotheses.
    """
    weights = build_block_weights(reconstructor, tiling.layout)  # (FF, O, r)

    ff, o, r = weights.shape
    w_flat = weights.reshape(ff * o, r)

    if hypothesis_indexes is not None:
        w_flat = w_flat[hypothesis_indexes.to(w_flat.device)]

    return w_flat
