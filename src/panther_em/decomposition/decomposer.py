"""Polar projection decomposition for cryo-EM volumes."""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tqdm

from panther_em.coordinates.transform_base import CoordinateTransform, GridTransform
from panther_em.coordinates.warp_config import WarpConfig
from panther_em.decomposition.projections import do_pipelined_projection_and_transforms
from panther_em.decomposition.result import DecompositionResult

_LOW_RANK_OVERSAMPLE = 20  # roughly 10 percent margin
_LOW_RANK_NITER = 2  # PyTorch default


def _dense_svd_block(
    freq_blocks: torch.Tensor, eig_max: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a full SVD and truncate to the top ``eig_max`` components.

    Parameters
    ----------
    freq_blocks : torch.Tensor
        Batched matrices of shape ``(num_k_batch, rows, num_radius)``.
    eig_max : int
        Number of leading singular components to keep.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Truncated ``(u, s, vh)`` each with ``eig_max`` leading components.
    """
    u, s, vh = torch.linalg.svd(freq_blocks, full_matrices=False)
    return u[..., :eig_max], s[..., :eig_max], vh[..., :eig_max, :]


def _lowrank_svd_block(
    freq_blocks: torch.Tensor, eig_max: int, num_radius: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a randomized low-rank SVD truncated to the top ``eig_max`` components.

    Parameters
    ----------
    freq_blocks : torch.Tensor
        Batched matrices of shape ``(num_k_batch, rows, num_radius)``.
    eig_max : int
        Number of leading singular components to keep.
    num_radius : int
        Number of columns in ``freq_blocks`` (upper bound for the oversampled rank).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Truncated ``(u, s, vh)`` each with ``eig_max`` leading components.
    """
    q = min(eig_max + _LOW_RANK_OVERSAMPLE, num_radius)
    u, s, v = torch.svd_lowrank(freq_blocks, q=q, niter=_LOW_RANK_NITER)
    return u[..., :eig_max], s[..., :eig_max], v[..., :eig_max].mH


class PolarProjectionDecomposer:
    """Manages projection generation and block-circulant data decomposition.

    NOTE: The Euler angles (phi, theta) represent the first two angles in ZYZ format
    where the final in-plane rotation angle (psi) is assumed to be zero based on the
    polar projection model.

    Parameters
    ----------
    volume : np.ndarray | torch.Tensor
        The 3D volume(s) to decompose. Either a single `(d, d, d)` volume or a stack of
        `(num_volumes, d, d, d)` volumes. A single volume is treated as `num_volumes=1`.
    phi_values : np.ndarray | torch.Tensor
        Phi values, in degrees of ZYZ Euler angles, for projection orientations. Shared
        across all volumes.
    theta_values : np.ndarray | torch.Tensor
        Theta values, in degrees of ZYZ Euler angles, for projection orientations.
        Shared across all volumes.
    coordinate_transform : CoordinateTransform
        Pre-built coordinate transform that defines the polar-space geometry.
    fourier_filters : np.ndarray | torch.Tensor | None, optional
        Pre-computed Fourier-space filters to apply during projection. Default is None
        (identity filter).
    volume_labels : np.ndarray | torch.Tensor | None, optional
        Optional per-volume labels/coordinates (e.g. a conformation/state coordinate
        value), shape `(num_volumes,)`. Default is None.
    warp : WarpConfig | None, optional
        Interpolation settings of the warp onto the polar grid, recorded in the result
        so every consumer warps polar images back the same way. Default is
        ``WarpConfig()``.
    device : str | torch.device, optional
        Compute device ('cpu', 'cuda', 'cuda:0', etc.). Default is 'cpu'.

    Attributes
    ----------
    result : DecompositionResult
        The decomposition result after calling :meth:`do_decomposition`.
    """

    def __init__(
        self,
        volume: np.ndarray | torch.Tensor,
        phi_values: np.ndarray | torch.Tensor,
        theta_values: np.ndarray | torch.Tensor,
        coordinate_transform: CoordinateTransform,
        fourier_filters: np.ndarray | torch.Tensor | None = None,
        volume_labels: np.ndarray | torch.Tensor | None = None,
        warp: WarpConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize the polar projection decomposer."""
        if phi_values.shape != theta_values.shape:
            raise ValueError(
                "phi_values and theta_values must have the same shape, "
                f"got {phi_values.shape} and {theta_values.shape}"
            )

        self.device = torch.device(device)
        self.warp = warp or WarpConfig()

        # Convert inputs to tensors on the correct device
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume.copy())

        if isinstance(phi_values, np.ndarray):
            phi_values = torch.from_numpy(phi_values.copy())

        if isinstance(theta_values, np.ndarray):
            theta_values = torch.from_numpy(theta_values.copy())

        if isinstance(fourier_filters, np.ndarray):
            fourier_filters = torch.from_numpy(fourier_filters.copy())

        if isinstance(volume_labels, np.ndarray):
            volume_labels = torch.from_numpy(volume_labels.copy())

        if volume.ndim == 3:
            volume = volume.unsqueeze(0)
        elif volume.ndim != 4:
            raise ValueError(
                "volume must have shape (d, d, d) or (num_volumes, d, d, d), "
                f"got {tuple(volume.shape)}"
            )
        self.volume = volume
        self.num_volumes = volume.shape[0]

        if volume_labels is not None and volume_labels.shape != (self.num_volumes,):
            raise ValueError(
                f"volume_labels shape {tuple(volume_labels.shape)} does not match "
                f"(num_volumes,) = ({self.num_volumes},)"
            )
        self.volume_labels = volume_labels

        # Send to the target device
        self.phi_values = phi_values.to(self.device)
        self.theta_values = theta_values.to(self.device)
        self.fourier_filters = (
            fourier_filters.to(self.device) if fourier_filters is not None else None
        )

        if coordinate_transform.cartesian_shape != tuple(self.volume.shape[-2:]):
            raise ValueError(
                "coordinate_transform.cartesian_shape must match volume image shape "
                f"{tuple(self.volume.shape[-2:])}, "
                f"got {coordinate_transform.cartesian_shape}."
            )

        self._coordinate_transform: CoordinateTransform = coordinate_transform
        self._result: DecompositionResult | None = None

    @property
    def result(self) -> DecompositionResult:
        """Access decomposition results.

        Returns
        -------
        DecompositionResult
            The stored decomposition result.

        Raises
        ------
        ValueError
            If decomposition has not been performed yet.
        """
        if self._result is None:
            raise ValueError(
                "Decomposition not yet performed. Call do_decomposition() first."
            )
        return self._result

    @property
    def is_decomposed(self) -> bool:
        """Check if decomposition has been performed."""
        return self._result is not None

    def do_decomposition(
        self,
        k_max: int | None = None,
        eig_max: int | None = None,
        projection_batch_size: int = 128,
        block_batch_size: int = 8,
        normalize_projections: bool = False,
        storage_backend: Literal["on_device", "cpu", "mmap"] = "cpu",
        mmap_path: Path | str | None = None,
        svd_method: Literal["dense", "low_rank"] = "dense",
    ) -> DecompositionResult:
        """Run the block-circulant decomposition using the held orientations.

        NOTE: Computation is split into two stages: 1) on-the-fly projection generation
        plus FFT transformation 2) block-wise SVD decomposition. A barrier exists
        between these two stages, and intermediate results from (1) are stored
        according to ``storage_backend``.

        Parameters
        ----------
        k_max : int | None, optional
            Maximum angular frequency block index to compute. If None, uses all
            available angular frequencies. Valid range depends on number of angular
            components (maximum is `num_angle // 2`). For example, if `num_angle=360`,
            the valid range is `[1, 180]`. Default is None.
        eig_max : int | None, optional
            Maximum radial eigenvalue index to compute. If None, sets eig_max to the
            number of radial components. Default is None.
        projection_batch_size : int, optional
            Number of projections to process at a time for memory efficiency.
            Default is 128.
        block_batch_size : int, optional
            Number of frequency blocks to process at a time on GPU for SVD.
            Default is 8.
        normalize_projections : bool, optional
            If True, normalize projections to have mean zero and unit variance before
            decomposition. Default is False.
        storage_backend : {"on_device", "cpu", "mmap"}, optional
            Where to accumulate Stage 1 results:

            * ``"on_device"`` — keep on the compute device; fastest when VRAM allows.
            * ``"cpu"`` — pageable CPU RAM via synchronous D2H copies (default).
            * ``"mmap"`` — memory-mapped file on disk; use when buffer exceeds RAM.
              Requires ``mmap_path``.
        mmap_path : Path | str | None, optional
            Path for the memory-mapped intermediate file. Required when
            ``storage_backend="mmap"``. Default is None.
        svd_method : Literal["dense", "low_rank"], optional
            Algorithm used for the per-frequency-block SVD. ``"dense"`` computes the
            full SVD via `torch.linalg.svd` and truncates to `eig_max` components.
            ``"low_rank"`` uses a randomized low-rank SVD (`torch.svd_lowrank`) that
            targets `eig_max` components directly, which is substantially faster when
            `num_radius` (the number of radial components) is much larger than
            `eig_max`.

        Returns
        -------
        DecompositionResult
            The decomposition result containing singular values and vectors.

        Raises
        ------
        ValueError
            If `svd_method` is not one of "dense" or "low_rank".
        """
        if svd_method not in ("dense", "low_rank"):
            raise ValueError(
                f"svd_method must be 'dense' or 'low_rank', got {svd_method!r}"
            )
        ### Stage 1: GPU projection generation and transform.

        # Transforms are device-agnostic; GPU dispatch is handled internally.
        transformer = self._coordinate_transform

        polar_projections_transformed_cpu, is_complex, k_max = (
            do_pipelined_projection_and_transforms(
                volume=self.volume,
                phi=self.phi_values,
                theta=self.theta_values,
                psi=torch.zeros_like(self.phi_values),
                fourier_filters=self.fourier_filters,
                transformer=transformer,
                warp=self.warp,
                projection_batch_size=projection_batch_size,
                normalize_projections=normalize_projections,
                k_max=k_max,
                storage_backend=storage_backend,
                mmap_path=mmap_path,
                device=self.device,
            )
        )

        ### Stage 2: Decompose each frequency block with SVD
        batch_shape = polar_projections_transformed_cpu.shape[:-2]
        num_angular_mode, num_radius = polar_projections_transformed_cpu.shape[-2:]

        # Flatten all outer dimensions:
        # from (..., num_angular_mode, num_radius)
        # ---> (batch_size, num_angular_mode, num_radius)
        batch_size = int(np.prod(batch_shape))
        polar_projections_reshaped = polar_projections_transformed_cpu.reshape(
            batch_size, num_angular_mode, num_radius
        )

        # Stored tensor is already cropped to k_max by the pipeline; iterate all blocks.
        num_freq_block = num_angular_mode

        if eig_max is None:
            eig_max = num_radius
        if eig_max < 1 or eig_max > num_radius:
            raise ValueError(
                f"eig_max must be in the range [1, {num_radius}] "
                f"for num_radius={num_radius}, got eig_max={eig_max}"
            )

        # Allocate tensors for the SVD results on CPU
        U = torch.zeros(
            (*batch_shape, num_freq_block, eig_max), dtype=torch.complex64, device="cpu"
        )
        S = torch.zeros((num_freq_block, eig_max), dtype=torch.float32, device="cpu")
        Vh = torch.zeros(
            (num_freq_block, eig_max, num_radius),
            dtype=torch.complex64,
            device="cpu",
        )

        block_index_range = range(0, num_freq_block, block_batch_size)
        for k_result_start in tqdm.tqdm(block_index_range, desc="decomp freq blocks"):
            k_result_end = min(k_result_start + block_batch_size, num_freq_block)
            num_k_batch = k_result_end - k_result_start
            k_indices = torch.arange(k_result_start, k_result_end, device="cpu")

            freq_blocks = polar_projections_reshaped[:, k_indices, :]
            freq_blocks = freq_blocks.to(device=self.device, non_blocking=True)

            # Reshape from (rows, num_k_B, num_r) --> (num_k_B, rows, num_r)
            # since we want SVD to operate on (rows, num_r)
            freq_blocks = freq_blocks.permute(1, 0, 2)

            if svd_method == "dense":
                u, s, vh = _dense_svd_block(freq_blocks, eig_max)
            else:
                u, s, vh = _lowrank_svd_block(freq_blocks, eig_max, num_radius)

            # Reshape outer indices for storage
            u_reshaped = u.permute(1, 0, 2)
            u_reshaped = u_reshaped.reshape(*batch_shape, num_k_batch, eig_max)

            U[..., k_indices, :] = u_reshaped.cpu()
            S[k_indices] = s.cpu()
            Vh[k_indices, :, :] = vh.cpu()

        # Eagerly materialize all coord grids into a self-contained GridTransform.
        result_transform = GridTransform.from_transform(transformer)

        # Convert results back to numpy for storage
        fourier_filters_np = (
            self.fourier_filters.cpu().numpy()
            if self.fourier_filters is not None
            else None
        )
        num_angle = result_transform.polar_shape[0]
        volume_labels_np = (
            self.volume_labels.cpu().numpy() if self.volume_labels is not None else None
        )
        self._result = DecompositionResult(
            S=S.cpu().numpy().astype(np.float32),
            U=U.cpu().numpy(),
            Vh=Vh.cpu().numpy(),
            k_max=k_max,
            eig_max=eig_max,
            is_complex_projection=is_complex,
            num_volumes=batch_shape[0],
            num_fourier_filters=batch_shape[1],
            num_orientations=batch_shape[2],
            num_angular_components=num_angle,
            num_radial_components=num_radius,
            phi_values=self.phi_values.cpu().numpy(),
            theta_values=self.theta_values.cpu().numpy(),
            fourier_filters=fourier_filters_np,
            volume_labels=volume_labels_np,
            coordinate_transform=result_transform,
            warp_config=self.warp,
        )

        return self._result
