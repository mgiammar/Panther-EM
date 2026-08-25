"""Pipelined generation and transforms of projections in polar coordinates.

For the on-the-fly projection generation, groups of projections (row elements) can be
generated and transformed independently before the SVD stage. This module defines the
pipeline for this stage of computation:

1. Transform batch of (B, 3) ZYZ Euler angles (phi, theta, psi) into rotation matrices.
2. Take Fourier slices from a RFFT'd 3D volume using the rotation matrices. Produces
   (B, h, w // 2 + 1) Fourier-space projections.
3. Apply defocus offsets filters (of shape (f, h, w // 2 + 1)) in Fourier space to get
    (f, B, h, w // 2 + 1) defocus-offset Fourier-slices projections.
4. Inverse FFT into set of (f, B, h, w) CTF convolved cartesian-space projections.
5. Warp cartesian-space projections in the batch into polar coordinates, producing
   (f, B, num_angle, num_radius) polar projections.
6. Fourier transform along the angular dimension (axis=-2) to get
   (f, B, num_angular_mode, num_radius) angular-frequency-space polar projections.
   NOTE: If projections are real-valued, can leverage RFFT here.
7. Move batch of angular-frequency-space polar projections to CPU for further processing
   in the SVD pipeline.
8. Repeat for next batch until all projections are processed.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np
import roma
import torch
import torch.nn.functional as F
import tqdm
from torch_fourier_slice.slice_extraction import extract_central_slices_rfft_3d
from torch_fourier_slice.volume_utils import (
    compute_cube_face_averages,
    separable_sinc2_correction,
)

from panther_em.coordinates.transform_base import CoordinateTransform


def precompute_volume_dft(
    volume: torch.Tensor,
    pad_factor: float = 2.0,
) -> tuple[torch.Tensor, float, int]:
    """Precompute the 3D RFFT of a volume with padding.

    Parameters
    ----------
    volume : torch.Tensor
        `(d, d, d)` cubic volume.
    pad_factor : float
        Padding factor for the volume. Default is 2.0.

    Returns
    -------
    dft : torch.Tensor
        The fftshifted 3D RFFT of the sinc^2-corrected, padded volume.
    edge_value_scaled : float
        `edge_value * d`, the constant to add back to projections after IFFT, where
        `edge_value` is the cube-face average of the volume subtracted before padding.
    pad_width : int
        Number of pixels padded on each side (0 if pad_factor <= 1.0).
    """
    d = volume.shape[-1]

    edge_value = compute_cube_face_averages(volume, n=4)
    volume = volume - edge_value

    pad_width = 0
    if pad_factor > 1.0:
        pad_width = int((d * (pad_factor - 1.0)) // 2)
        volume = F.pad(volume, pad=[pad_width] * 6, mode="constant", value=0.0)

    # Divide by sinc^2 in real space to correct for the blur introduced by
    # trilinear interpolation during central-slice extraction (see
    # torch_fourier_slice issue #65).
    sinc2 = separable_sinc2_correction(volume.shape[-3:], device=volume.device)
    volume = volume / sinc2

    # Center-to-origin shift, then 3D RFFT
    dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))
    dft = torch.fft.rfftn(dft, dim=(-3, -2, -1))
    dft = torch.fft.fftshift(dft, dim=(-3, -2))  # shift so DC is at center

    edge_value_scaled = edge_value * d

    return dft, edge_value_scaled, pad_width


def project_from_precomputed_dft(
    dft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    edge_value_scaled: float,
    pad_width: int,
    fftfreq_max: float | None = 0.5,
    zyx_matrices: bool = False,
) -> torch.Tensor:
    """Extract central slices from a precomputed DFT and transform to real space.

    Parameters
    ----------
    dft : torch.Tensor
        Precomputed fftshifted 3D RFFT from `precompute_volume_dft`.
    rotation_matrices : torch.Tensor
        `(..., 3, 3)` rotation matrices for slice extraction.
    edge_value_scaled : float
        Constant to add back after IFFT (edge_value * d), from `precompute_volume_dft`.
    pad_width : int
        Padding width to remove from projections.
    fftfreq_max : float | None
        Maximum frequency in cycles per pixel. Default is 0.5.
    zyx_matrices : bool
        Whether matrices operate on zyx coordinates. Default is False.

    Returns
    -------
    projections : torch.Tensor
        `(..., d, d)` real-space projections.
    """
    projections = extract_central_slices_rfft_3d(
        volume_rfft=dft,
        rotation_matrices=rotation_matrices,
        fftfreq_max=fftfreq_max,
        zyx_matrices=zyx_matrices,
    )

    # Transform back to real space
    projections = torch.fft.ifftshift(projections, dim=(-2,))
    projections = torch.fft.irfftn(projections, dim=(-2, -1))
    projections = torch.fft.ifftshift(projections, dim=(-2, -1))

    # Unpad
    if pad_width > 0:
        projections = F.pad(projections, pad=[-pad_width] * 4)

    # Add back the edge value subtracted in precompute_volume_dft.
    projections += edge_value_scaled

    return projections


def generate_projection_batch(
    dft: torch.Tensor,
    edge_value_scaled: float,
    pad_width: int,
    phi: torch.Tensor,  # (B,)
    theta: torch.Tensor,  # (B,)
    psi: torch.Tensor,  # (B,)
    fftfreq_max: float = 0.5,
) -> torch.Tensor:  # (B, h, w)
    """Generate a batch of 2D projections from a precomputed volume DFT."""
    rot_matrix = roma.euler_to_rotmat("ZYZ", angles=(phi, theta, psi), degrees=True)

    return project_from_precomputed_dft(
        dft=dft,
        rotation_matrices=rot_matrix,
        edge_value_scaled=edge_value_scaled,
        pad_width=pad_width,
        fftfreq_max=fftfreq_max,
    )


def apply_fourier_filters(
    projections: torch.Tensor,  # (B, h, w)
    fourier_filters: torch.Tensor,  # (f, h, w // 2 + 1) OR (f, h, w)
) -> torch.Tensor:  # (f, B, h, w)
    """Apply pre-calculated Fourier filters to a batch of real-space projections."""
    _is_complex = (
        torch.is_complex(projections)
        or fourier_filters.shape[-2] == fourier_filters.shape[-1]  # not rfft'd
    )

    # Pathing for complex vs real valued resultant projections
    if _is_complex:
        projections_fft = torch.fft.fft2(projections)
        projections_fft = projections_fft[None, ...] * fourier_filters[:, None, :, :]
        return torch.fft.ifft2(projections_fft)
    else:
        projections_fft = torch.fft.rfft2(projections)
        projections_fft = projections_fft[None, ...] * fourier_filters[:, None, :, :]
        return torch.fft.irfft2(projections_fft)


def process_batch(
    dft: torch.Tensor,
    edge_value_scaled: float,
    pad_width: int,
    phi: torch.Tensor,
    theta: torch.Tensor,
    psi: torch.Tensor,
    fourier_filters: torch.Tensor,
    transformer: CoordinateTransform,
    warp_polar_kwargs: dict,
    fftfreq_max: float = 0.5,
) -> tuple[torch.Tensor, bool, int]:
    """Generate a batch of polar proj. from a DFT, FFT along angular dim.

    Parameters
    ----------
    dft : torch.Tensor
        Precomputed fftshifted 3D RFFT from `precompute_volume_dft`.
    edge_value_scaled : float
        Constant to add back after IFFT (edge_value * d), from `precompute_volume_dft`.
    pad_width : int
        Padding width to remove from projections.
    phi, theta, psi : torch.Tensor
        `(B,)` Euler angles in degrees for the batch.
    fourier_filters : torch.Tensor
        Fourier space filters to apply. If shape (f, h, w // 2 + 1), then will be
        applied in real-mode. Can support complex-valued filters in real-space if
        shape is (f, h, w).
    transformer : CoordinateTransform
        Pre-initialized transformer for warping to polar coordinates.
    warp_polar_kwargs : dict
        Additional kwargs forwarded to ``transformer.to_transform_space``.
    fftfreq_max : float
        Maximum frequency in cycles per pixel for Fourier slicing. Default is 0.5.

    Returns
    -------
    projections_polar_fft : torch.Tensor
        `(f, B, num_angular_mode, num_radius)` tensor of polar projections in angular
        frequency space where `f` is the number of Fourier filters and `B` is the
        projection batch size.
    is_complex : bool
        Whether the projections are complex-valued (True) or real-valued (False).
    num_angular_mode : int
        Number of angular modes in the polar projection. Will either be
        `num_angle` (if complex-valued) or `num_angle // 2 + 1` (if real-valued).
    """
    projections = generate_projection_batch(
        dft=dft,
        edge_value_scaled=edge_value_scaled,
        pad_width=pad_width,
        phi=phi,
        theta=theta,
        psi=psi,
        fftfreq_max=fftfreq_max,
    )

    projections_filtered = apply_fourier_filters(projections, fourier_filters)

    # to_transform_space expects a single batch dimension; flatten the two outer dims.
    num_defocus = fourier_filters.shape[0]
    batch_size = phi.shape[0]
    projections_filtered_view = projections_filtered.view(
        -1,
        projections_filtered.shape[-2],
        projections_filtered.shape[-1],
    )

    projections_polar = transformer.to_transform_space(
        projections_filtered_view,
        **warp_polar_kwargs,
    )

    # NOTE: Pathing here for handling both real- and complex-valued projections
    # NOTE: Orthonormal transformation to preserve forward-backward scaling of features
    is_complex = torch.is_complex(projections_polar)
    if is_complex:
        _fft_method = torch.fft.fft
    else:
        _fft_method = torch.fft.rfft

    projections_polar_fft = _fft_method(projections_polar, dim=-2, norm="ortho")
    num_angular_mode = projections_polar_fft.shape[-2]

    # Reshape back to (num_defocus, batch_size, num_angle, num_radius)
    num_radius = transformer.polar_shape[1]
    projections_polar_fft = projections_polar_fft.view(
        num_defocus, batch_size, num_angular_mode, num_radius
    )

    return projections_polar_fft, is_complex, num_angular_mode


def _compute_freq_crop(
    is_complex: bool, num_angular_mode: int, k_max: int | None
) -> tuple[int, int, int]:
    """Compute angular-frequency crop indices for the given k_max.

    Parameters
    ----------
    is_complex : bool
        Whether the projections are complex-valued.
    num_angular_mode : int
        Total number of angular frequency modes from the FFT.
    k_max : int | None
        Maximum angular frequency index to retain. None means keep all.

    Returns
    -------
    block_index_min : int
        Start index (inclusive) along the angular-mode axis.
    block_index_max : int
        Stop index (exclusive) along the angular-mode axis.
    k_max_actual : int
        The resolved k_max value (after applying the None default).
    """
    if is_complex:
        if num_angular_mode % 2 != 0:
            raise ValueError(
                "num_angular_mode must be even for complex projections in fftshifted "
                f"storage; got num_angular_mode={num_angular_mode}"
            )
        allowable = num_angular_mode // 2
        if k_max is None:
            k_max_actual = allowable
        elif k_max < 1 or k_max > allowable:
            raise ValueError(
                f"k_max must be in [1, {allowable}] for complex projection data "
                f"with num_angular_mode={num_angular_mode}, got k_max={k_max}"
            )
        else:
            k_max_actual = k_max
        dc = num_angular_mode // 2
        block_index_min = dc - k_max_actual
        block_index_max = dc + k_max_actual
    else:
        allowable = num_angular_mode
        if k_max is None:
            k_max_actual = allowable
        elif k_max < 1 or k_max > allowable:
            raise ValueError(
                f"k_max must be in [1, {allowable}] for real projection data "
                f"with num_angular_mode={num_angular_mode}, got k_max={k_max}"
            )
        else:
            k_max_actual = k_max
        block_index_min = 0
        block_index_max = k_max_actual

    return block_index_min, block_index_max, k_max_actual


class _ProjectionBuffer(ABC):
    """Abstract accumulation buffer for staged polar-projection results."""

    @abstractmethod
    def write_batch(
        self,
        gpu_result: torch.Tensor,
        vol_idx: int,
        dest_slice: slice,
    ) -> None:
        """Accept one batch of GPU results and write to the output buffer.

        Parameters
        ----------
        gpu_result : torch.Tensor
            `(F, B, num_angular_mode, num_radius)` complex64 tensor on the compute
            device.
        vol_idx : int
            Volume-axis index into the full output buffer.
        dest_slice : slice
            Orientation-axis slice into the full output buffer.
        """

    @abstractmethod
    def finalize(self) -> torch.Tensor:
        """Block until all writes are complete and return the filled buffer."""


class _OnDeviceBuffer(_ProjectionBuffer):
    """Keep accumulation tensor on the compute device; no D2H transfer."""

    def __init__(
        self,
        buffer_shape: tuple[int, ...],
        block_index_min: int,
        block_index_max: int,
        device: torch.device,
    ) -> None:
        self._output = torch.empty(buffer_shape, dtype=torch.complex64, device=device)
        self._block_index_min = block_index_min
        self._block_index_max = block_index_max

    def write_batch(
        self,
        gpu_result: torch.Tensor,
        vol_idx: int,
        dest_slice: slice,
    ) -> None:
        bmin, bmax = self._block_index_min, self._block_index_max
        self._output[vol_idx, :, dest_slice] = gpu_result[:, :, bmin:bmax, :]

    def finalize(self) -> torch.Tensor:
        if self._output.is_cuda:
            torch.cuda.synchronize(self._output.device)
        return self._output


class _HostStagingBuffer(_ProjectionBuffer):
    """Synchronous D2H copy: GPU result → host destination (pageable or mmap).

    Parameters
    ----------
    output_tensor : torch.Tensor
        Pre-allocated destination tensor (pageable CPU or ``torch.from_numpy``
        view of a ``np.memmap``).
    block_index_min, block_index_max : int
        Angular-frequency crop indices (fixed for the lifetime of the buffer).
    """

    def __init__(
        self,
        output_tensor: torch.Tensor,
        block_index_min: int,
        block_index_max: int,
    ) -> None:
        self._output = output_tensor
        self._block_index_min = block_index_min
        self._block_index_max = block_index_max

    def write_batch(
        self,
        gpu_result: torch.Tensor,
        vol_idx: int,
        dest_slice: slice,
    ) -> None:
        bmin, bmax = self._block_index_min, self._block_index_max
        self._output[vol_idx, :, dest_slice].copy_(gpu_result[:, :, bmin:bmax, :])

    def finalize(self) -> torch.Tensor:
        return self._output


def do_pipelined_projection_and_transforms(
    volume: torch.Tensor,
    phi: torch.Tensor,
    theta: torch.Tensor,
    psi: torch.Tensor,
    fourier_filters: torch.Tensor | None,
    transformer: CoordinateTransform,
    warp_polar_kwargs: dict,
    projection_batch_size: int = 128,
    pad_factor: float = 2.0,
    fftfreq_max: float = 0.5,
    show_progress: bool = True,
    k_max: int | None = None,
    storage_backend: Literal["on_device", "cpu", "mmap"] = "cpu",
    mmap_path: Path | str | None = None,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, bool, int]:
    """Pipelined generation and transforms of projections in polar coordinates.

    Parameters
    ----------
    volume : torch.Tensor
        `(num_volumes, d, d, d)` stack of cubic volumes.
    phi, theta, psi : torch.Tensor
        `(N,)` Euler angles in degrees, shared across all volumes.
    fourier_filters : torch.Tensor | None
        `(f, h, w // 2 + 1)` Fourier-space filters, or None for identity.
    transformer : CoordinateTransform
        Pre-initialised coordinate transform on the same device as ``volume``.
        Provides the polar-space geometry (``polar_shape``) and the
        ``to_transform_space`` warp method.
    warp_polar_kwargs : dict
        Additional kwargs forwarded to ``transformer.to_transform_space``.
    projection_batch_size : int
        Orientations per GPU batch. Default is 128.
    pad_factor : float
        Volume padding factor for Fourier slicing. Default is 2.0.
    fftfreq_max : float
        Maximum frequency in cycles per pixel. Default is 0.5.
    show_progress : bool
        Whether to show a tqdm progress bar. Default is True.
    k_max : int | None
        Maximum angular frequency index to store. Only the ``k_max`` lowest-index
        blocks (real-valued projections) or the ``2 * k_max`` blocks centred on DC
        (complex-valued) are written to the output buffer. None stores all blocks.
        Default is None.
    storage_backend : {"on_device", "cpu", "mmap"}
        Where to accumulate results:

        * ``"on_device"`` — keep the full buffer on the compute device (GPU).
          Fastest when VRAM is sufficient to hold all projections.
        * ``"cpu"`` — accumulate to a pageable CPU tensor via synchronous D2H
          copies after each batch.  Default.
        * ``"mmap"`` — same as ``"cpu"`` but the destination is a
          ``numpy.memmap``-backed file; use when the buffer exceeds RAM.
          Requires ``mmap_path``.
    mmap_path : Path | str | None
        Path for the memory-mapped intermediate file.  Required when
        ``storage_backend="mmap"``, ignored otherwise.  The file is created (or
        overwritten) in ``'w+'`` mode as a raw complex64 array.
    device : str | torch.device | None
        Compute device that each volume is transferred to before projecting. If None,
        uses ``volume.device``. Default is None.

    Returns
    -------
    torch.Tensor
        `(num_volumes, num_fourier_filters, num_projections, num_freq_block,
        num_radius)` complex64 tensor.  Resides on the compute device when
        ``storage_backend="on_device"``, on CPU otherwise.  When underlying projections
        are complex, the angular dim has been fft-shifted so that indices correspond to
        ``[-k_max, ..., 0, ..., k_max - 1]``; for real projections it corresponds to
        ``[0, 1, ..., k_max - 1]``.
    bool
        Whether the projections are complex-valued.
    int
        The resolved ``k_max`` value actually used.
    """
    if storage_backend == "mmap" and mmap_path is None:
        raise ValueError("mmap_path must be provided when storage_backend='mmap'")
    if volume.ndim != 4:
        raise ValueError(
            f"volume must have shape (num_volumes, d, d, d), got {tuple(volume.shape)}"
        )

    num_radius = transformer.polar_shape[1]
    num_volumes = volume.shape[0]
    compute_device = torch.device(device) if device is not None else volume.device

    if fourier_filters is None:
        filter_shape = (
            (1, volume.shape[-2], volume.shape[-1])
            if torch.is_complex(volume)
            else (1, volume.shape[-2], volume.shape[-1] // 2 + 1)
        )
        fourier_filters = torch.ones(
            filter_shape, device=compute_device, dtype=torch.complex64
        )
    else:
        fourier_filters = fourier_filters.to(compute_device)

    # NOTE: Doing a dummy batch on the first volume before allocating memory to get
    #       the `is_complex` flag and the number of angular modes which need stored.
    #       Orientations/filters/k_max are assumed identical across all volumes.
    dft0, edge_value_scaled0, pad_width0 = precompute_volume_dft(
        volume[0].to(compute_device, non_blocking=True), pad_factor=pad_factor
    )
    _, is_complex, num_angular_mode = process_batch(
        dft=dft0,
        edge_value_scaled=edge_value_scaled0,
        pad_width=pad_width0,
        phi=phi[:2],
        theta=theta[:2],
        psi=psi[:2],
        fourier_filters=fourier_filters,
        transformer=transformer,
        warp_polar_kwargs=warp_polar_kwargs,
        fftfreq_max=fftfreq_max,
    )

    block_index_min, block_index_max, k_max_actual = _compute_freq_crop(
        is_complex, num_angular_mode, k_max
    )
    num_freq_block = block_index_max - block_index_min

    num_orientations = phi.shape[0]
    num_filters = fourier_filters.shape[0]
    buffer_shape = (
        num_volumes,
        num_filters,
        num_orientations,
        num_freq_block,
        num_radius,
    )

    if storage_backend == "on_device":
        buffer: _ProjectionBuffer = _OnDeviceBuffer(
            buffer_shape, block_index_min, block_index_max, compute_device
        )
    elif storage_backend == "cpu":
        output_tensor = torch.empty(buffer_shape, dtype=torch.complex64, device="cpu")
        buffer = _HostStagingBuffer(output_tensor, block_index_min, block_index_max)
    else:  # "mmap"
        _mm = np.memmap(mmap_path, dtype=np.complex64, mode="w+", shape=buffer_shape)
        output_tensor = torch.from_numpy(_mm)
        buffer = _HostStagingBuffer(output_tensor, block_index_min, block_index_max)

    pbar = None
    if show_progress:
        pbar = tqdm.tqdm(
            total=num_volumes * num_orientations,
            desc="calc. projections",
            unit="proj",
            unit_scale=num_filters,
        )

    for vol_idx in range(num_volumes):
        if vol_idx == 0:
            dft, edge_value_scaled, pad_width = dft0, edge_value_scaled0, pad_width0
        else:
            vol_i = volume[vol_idx].to(compute_device, non_blocking=True)
            dft, edge_value_scaled, pad_width = precompute_volume_dft(
                vol_i, pad_factor=pad_factor
            )
            del vol_i

        for start_idx in range(0, num_orientations, projection_batch_size):
            end_idx = min(start_idx + projection_batch_size, num_orientations)
            slice_ = slice(start_idx, end_idx)
            batch_size = end_idx - start_idx

            result, is_complex, _ = process_batch(
                dft=dft,
                edge_value_scaled=edge_value_scaled,
                pad_width=pad_width,
                phi=phi[slice_],
                theta=theta[slice_],
                psi=psi[slice_],
                fourier_filters=fourier_filters,
                transformer=transformer,
                warp_polar_kwargs=warp_polar_kwargs,
                fftfreq_max=fftfreq_max,
            )

            if is_complex:
                result = torch.fft.fftshift(result, dim=-3)

            buffer.write_batch(result, vol_idx, slice_)
            del result

            if pbar is not None:
                pbar.update(batch_size)

        del dft

    if pbar is not None:
        pbar.close()

    output = buffer.finalize()
    return output, is_complex, k_max_actual
