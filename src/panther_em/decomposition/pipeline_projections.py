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

import roma
import torch
import torch.nn.functional as F
import tqdm
from torch_fourier_slice.slice_extraction import extract_central_slices_rfft_3d
from torch_fourier_slice.volume_utils import compute_cube_face_averages

from panther_em.coordinates.transform_base import CoordinateTransform


def precompute_volume_dft(
    volume: torch.Tensor,
    pad_factor: float = 2.0,
    zero_background: bool = True,
) -> tuple[torch.Tensor, float, int]:
    """Precompute the 3D RFFT of a volume with padding.

    Parameters
    ----------
    volume : torch.Tensor
        `(d, d, d)` cubic volume.
    pad_factor : float
        Padding factor for the volume. Default is 2.0.
    zero_background : bool
        When True, zero out the background by subtracting an average edge value from
        the volume.

    Returns
    -------
    dft : torch.Tensor
        The fftshifted 3D RFFT of the padded volume, with DC zeroed out.
    volume_mean_scaled : float
        `volume.mean() * d`, the constant to add back to projections after IFFT.
    pad_width : int
        Number of pixels padded on each side (0 if pad_factor <= 1.0).
    """
    d = volume.shape[-1]
    edge_value = compute_cube_face_averages(volume, n=4)

    if zero_background:
        volume = volume - edge_value
        edge_value = 0.0  # for pad_factor case

    pad_width = 0
    if pad_factor > 1.0:
        pad_width = int((d * (pad_factor - 1.0)) // 2)
        volume = F.pad(volume, pad=[pad_width] * 6, mode="constant", value=edge_value)

    volume_mean_scaled = volume.mean() * d

    # Center-to-origin shift, then 3D RFFT
    dft = torch.fft.fftshift(volume, dim=(-3, -2, -1))
    dft = torch.fft.rfftn(dft, dim=(-3, -2, -1))
    dft[..., 0, 0, 0] = 0.0  # zero DC to avoid low-res artifacts
    dft = torch.fft.fftshift(dft, dim=(-3, -2))  # shift so DC is at center

    return dft, volume_mean_scaled, pad_width


def project_from_precomputed_dft(
    dft: torch.Tensor,
    rotation_matrices: torch.Tensor,
    volume_mean_scaled: float,
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
    volume_mean_scaled : float
        Constant to add back after IFFT (volume_mean * d).
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

    # TEST: Try keeping zero-mean projections
    # Add back the mean
    projections += volume_mean_scaled

    return projections


def generate_projection_batch(
    dft: torch.Tensor,
    volume_mean_scaled: float,
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
        volume_mean_scaled=volume_mean_scaled,
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
    volume_mean_scaled: float,
    pad_width: int,
    phi: torch.Tensor,
    theta: torch.Tensor,
    psi: torch.Tensor,
    fourier_filters: torch.Tensor,
    transformer: CoordinateTransform,
    warp_polar_kwargs: dict,
    fftfreq_max: float = 0.5,
    normalize_projections: bool = False,
) -> tuple[torch.Tensor, bool, int]:
    """Generate a batch of polar proj. from a DFT, FFT along angular dim.

    Parameters
    ----------
    dft : torch.Tensor
        Precomputed fftshifted 3D RFFT from `precompute_volume_dft`.
    volume_mean_scaled : float
        Constant to add back after IFFT (volume_mean * d).
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
    normalize_projections : bool
        If True, normalize projections to have mean zero and unit variance. Default is
        False so projections are un-normalized.

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
        volume_mean_scaled=volume_mean_scaled,
        pad_width=pad_width,
        phi=phi,
        theta=theta,
        psi=psi,
        fftfreq_max=fftfreq_max,
    )

    projections_filtered = apply_fourier_filters(projections, fourier_filters)

    if normalize_projections:
        var, mean = torch.var_mean(projections_filtered, dim=(-2, -1), keepdim=True)
        projections_filtered = (projections_filtered - mean) / torch.sqrt(var + 1e-8)

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
                "num_angular_mode must be even for complex projections in fftshifted storage; "
                f"got num_angular_mode={num_angular_mode}"
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
    normalize_projections: bool = False,
    show_progress: bool = True,
    k_max: int | None = None,
) -> tuple[torch.Tensor, bool, int]:
    """Pipelined generation and transforms of projections in polar coordinates.

    Parameters
    ----------
    volume : torch.Tensor
        `(d, d, d)` cubic volume on GPU.
    phi, theta, psi : torch.Tensor
        `(N,)` Euler angles in degrees.
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
    normalize_projections : bool
        If True, normalize projections to have mean zero and unit variance. Default is
        False so projections are un-normalized.
    show_progress : bool
        Whether to show a tqdm progress bar. Default is True.
    k_max : int | None
        Maximum angular frequency index to store. Only the ``k_max`` lowest-index
        blocks (real-valued projections) or the ``2 * k_max`` blocks centred on DC
        (complex-valued) are written to the CPU buffer. None stores all blocks.
        Default is None.

    Returns
    -------
    torch.Tensor
        `(num_fourier_filters, num_projections, num_freq_block, num_radius)`
        complex64 tensor on CPU. When underlying projections are complex, angular dim
        has been fft-shifted so that indices correspond to
        ``[-k_max, ..., 0, ..., k_max - 1]``. When real valued, corresponds to
        ``[0, 1, ..., k_max - 1]`` frequencies.
    bool
        Whether the projections are complex-valued.
    int
        The resolved ``k_max`` value actually used.
    """
    # TODO: check input shapes/types

    num_radius = transformer.polar_shape[1]

    if fourier_filters is None:
        filter_shape = (
            (1, volume.shape[1], volume.shape[2])
            if torch.is_complex(volume)
            else (1, volume.shape[1], volume.shape[2] // 2 + 1)
        )
        fourier_filters = torch.ones(
            filter_shape, device=volume.device, dtype=torch.complex64
        )

    # Precompute the 3D RFFT once — stays on GPU for the entire pipeline
    dft, volume_mean_scaled, pad_width = precompute_volume_dft(
        volume, pad_factor=pad_factor
    )

    # Free the original volume from GPU since we only need the DFT now
    del volume

    # NOTE: Doing a dummy batch before allocating memory to get the `is_complex` flag
    #       and the number of angular modes which need stored
    _, is_complex, num_angular_mode = process_batch(
        dft=dft,
        volume_mean_scaled=volume_mean_scaled,
        pad_width=pad_width,
        phi=phi[:2],
        theta=theta[:2],
        psi=psi[:2],
        fourier_filters=fourier_filters,
        transformer=transformer,
        warp_polar_kwargs=warp_polar_kwargs,
        fftfreq_max=fftfreq_max,
        normalize_projections=normalize_projections,
    )

    block_index_min, block_index_max, k_max_actual = _compute_freq_crop(
        is_complex, num_angular_mode, k_max
    )
    num_freq_block = block_index_max - block_index_min

    # Allocate memory on CPU for cropped storage
    num_orientations = phi.shape[0]
    num_filters = fourier_filters.shape[0]
    polar_projections_transformed_cpu = torch.empty(
        (num_filters, num_orientations, num_freq_block, num_radius),
        dtype=torch.complex64,
        device="cpu",
        pin_memory=False,
    )

    pbar = None
    if show_progress:
        pbar = tqdm.tqdm(
            total=num_orientations,
            desc="calc. projections",
            unit="proj",
        )

    for start_idx in range(0, num_orientations, projection_batch_size):
        end_idx = min(start_idx + projection_batch_size, num_orientations)
        slice_ = slice(start_idx, end_idx)
        batch_size = end_idx - start_idx

        projections_polar_fft, is_complex, _ = process_batch(
            dft=dft,
            volume_mean_scaled=volume_mean_scaled,
            pad_width=pad_width,
            phi=phi[slice_],
            theta=theta[slice_],
            psi=psi[slice_],
            fourier_filters=fourier_filters,
            transformer=transformer,
            warp_polar_kwargs=warp_polar_kwargs,
            fftfreq_max=fftfreq_max,
            normalize_projections=normalize_projections,
        )

        if is_complex:
            projections_polar_fft = torch.fft.fftshift(projections_polar_fft, dim=-3)

        # Copy to CPU, cropping to the requested k_max window
        polar_projections_transformed_cpu[:, slice_].copy_(
            projections_polar_fft[:, :, block_index_min:block_index_max, :],
            non_blocking=True,
        )
        del projections_polar_fft

        if pbar is not None:
            pbar.update(batch_size)

    if pbar is not None:
        pbar.close()

    # Ensure all async copies are complete before returning
    if dft.is_cuda:
        torch.cuda.synchronize(dft.device)

    del dft

    return polar_projections_transformed_cpu, is_complex, k_max_actual
