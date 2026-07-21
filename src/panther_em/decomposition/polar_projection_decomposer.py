"""Polar projection decomposition for cryo-EM volumes."""

from pathlib import Path

import numpy as np
import torch
import tqdm

from panther_em.coordinates.transform_base import CoordinateTransform, GridTransform
from panther_em.decomposition.result import DecompositionResult
from panther_em.inference.projection_reconstruction import ProjectionReconstructor

from .pipeline_projections import do_pipelined_projection_and_transforms


class PolarProjectionDecomposer:
    """Manages projection generation and block-circulant data decomposition.

    NOTE: The Euler angles (phi, theta) represent the first two angles in ZYZ format
    where the final in-plane rotation angle (psi) is assumed to be zero based on the
    polar projection model.

    Parameters
    ----------
    volume : np.ndarray | torch.Tensor
        The 3D volume to decompose.
    phi_values : np.ndarray | torch.Tensor
        Phi values, in degrees of ZYZ Euler angles, for projection orientations.
    theta_values : np.ndarray | torch.Tensor
        Theta values, in degrees of ZYZ Euler angles, for projection orientations.
    coordinate_transform : CoordinateTransform
        Pre-built coordinate transform that defines the polar-space geometry.
    fourier_filters : np.ndarray | torch.Tensor | None, optional
        Pre-computed Fourier-space filters to apply during projection. Default is None
        (identity filter).
    device : str | torch.device, optional
        Compute device ('cpu', 'cuda', 'cuda:0', etc.). Default is 'cpu'.

    Attributes
    ----------
    result : DecompositionResult
        The decomposition result after calling :meth:`do_decomposition`.
    reconstructor : ProjectionReconstructor
        An object to perform reconstruction of decomposed features back into the input
        projections.
    """

    def __init__(
        self,
        volume: np.ndarray | torch.Tensor,
        phi_values: np.ndarray | torch.Tensor,
        theta_values: np.ndarray | torch.Tensor,
        coordinate_transform: CoordinateTransform,
        fourier_filters: np.ndarray | torch.Tensor | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize the polar projection decomposer."""
        if phi_values.shape != theta_values.shape:
            raise ValueError(
                "phi_values and theta_values must have the same shape, "
                f"got {phi_values.shape} and {theta_values.shape}"
            )

        self.device = torch.device(device)

        # Convert inputs to tensors on the correct device
        if isinstance(volume, np.ndarray):
            volume = torch.from_numpy(volume.copy())

        if isinstance(phi_values, np.ndarray):
            phi_values = torch.from_numpy(phi_values.copy())

        if isinstance(theta_values, np.ndarray):
            theta_values = torch.from_numpy(theta_values.copy())

        if isinstance(fourier_filters, np.ndarray):
            fourier_filters = torch.from_numpy(fourier_filters.copy())

        # Send to the target device
        self.volume = volume.to(self.device)
        self.phi_values = phi_values.to(self.device)
        self.theta_values = theta_values.to(self.device)
        self.fourier_filters = (
            fourier_filters.to(self.device) if fourier_filters is not None else None
        )

        if coordinate_transform.cartesian_shape != tuple(self.volume.shape[-2:]):
            raise ValueError(
                "coordinate_transform.cartesian_shape must match volume image shape "
                f"{tuple(self.volume.shape[-2:])}, got "
                f"{coordinate_transform.cartesian_shape}"
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
    def reconstructor(self) -> ProjectionReconstructor:
        """Get a ProjectionReconstructor from the decomposition result.

        Returns
        -------
        ProjectionReconstructor
            A reconstructor initialized with the current decomposition result
            and volume image shape.

        Raises
        ------
        ValueError
            If decomposition has not been performed yet.
        """
        return ProjectionReconstructor(
            result=self.result,
            device=self.device,
        )

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
    ) -> DecompositionResult:
        """Run the block-circulant decomposition using the held orientations.

        NOTE: Computation is split into two stages: 1) on-the-fly projection generation
        plus FFT transformation 2) block-wise SVD decomposition. A barrier exists
        between these two stages, and intermediate results from (1) are stored on CPU
        memory to support problem size scaling.

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

        Returns
        -------
        DecompositionResult
            The decomposition result containing singular values and vectors.
        """
        ### Stage 1: GPU projection generation and transform.

        # Transforms are device-agnostic; GPU dispatch is handled internally.
        transformer = self._coordinate_transform

        # Results generally too large to fit in GPU memory, so stored on CPU
        polar_projections_transformed_cpu, is_complex = (
            do_pipelined_projection_and_transforms(
                volume=self.volume,
                phi=self.phi_values,
                theta=self.theta_values,
                psi=torch.zeros_like(self.phi_values),
                fourier_filters=self.fourier_filters,
                transformer=transformer,
                warp_polar_kwargs={"preserve_energy": True},
                projection_batch_size=projection_batch_size,
                normalize_projections=normalize_projections,
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

        # Validate and select frequency block range
        # TODO: bring this logic into its own helper function...
        # NOTE: For complex projection data (when `is_complex=True`) frequency blocks
        #       are stored in fftshifted order ranging from -k_max to +k_max. Select
        #       block indices accordingly
        if is_complex:
            allowable_k_max = num_angular_mode // 2
            if k_max is None:
                k_max = allowable_k_max
            elif k_max < 1 or k_max > allowable_k_max:
                raise ValueError(
                    f"k_max must be in [1, {allowable_k_max}] for complex projection "
                    f"data with num_angular_mode={num_angular_mode}, got k_max={k_max}"
                )
            dc_data_index = num_angular_mode // 2
            block_index_min = dc_data_index - k_max
            block_index_max = dc_data_index + k_max
        else:
            allowable_k_max = num_angular_mode
            if k_max is None:
                k_max = allowable_k_max
            elif k_max < 1 or k_max > allowable_k_max:
                raise ValueError(
                    f"k_max must be in [1, {allowable_k_max}] for real projection "
                    f"data with num_angular_mode={num_angular_mode}, got k_max={k_max}"
                )
            block_index_min = 0
            block_index_max = k_max

        if eig_max is None:
            eig_max = num_radius
        if eig_max < 1 or eig_max > num_radius:
            raise ValueError(
                f"eig_max must be in the range [1, {num_radius}] "
                f"for num_radius={num_radius}, got eig_max={eig_max}"
            )

        num_freq_block = block_index_max - block_index_min

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

        block_index_range = range(block_index_min, block_index_max, block_batch_size)
        for k_result_start in tqdm.tqdm(block_index_range, desc="decomp freq blocks"):
            k_result_end = min(k_result_start + block_batch_size, num_freq_block)
            num_k_batch = k_result_end - k_result_start
            k_indices = torch.arange(k_result_start, k_result_end, device="cpu")

            freq_blocks = polar_projections_reshaped[:, k_indices, :]
            freq_blocks = freq_blocks.to(device=self.device, non_blocking=True)

            # Reshape from (rows, num_k_B, num_r) --> (num_k_B, rows, num_r)
            # since we want SVD to operate on (rows, num_r)
            freq_blocks = freq_blocks.permute(1, 0, 2)

            u, s, vh = torch.linalg.svd(freq_blocks, full_matrices=False)

            u = u[..., :eig_max]
            s = s[:, :eig_max]
            vh = vh[:, :eig_max, :]

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
        self._result = DecompositionResult(
            S=S.cpu().numpy().astype(np.float32),
            U=U.cpu().numpy(),
            Vh=Vh.cpu().numpy(),
            k_max=k_max,
            eig_max=eig_max,
            is_complex_projection=is_complex,
            num_fourier_filters=batch_shape[0],
            num_orientations=batch_shape[1],
            num_angular_components=num_angle,
            num_radial_components=num_radius,
            phi_values=self.phi_values.cpu().numpy(),
            theta_values=self.theta_values.cpu().numpy(),
            fourier_filters=fourier_filters_np,
            coordinate_transform=result_transform,
        )

        return self._result

    @classmethod
    def from_result(
        cls,
        result: DecompositionResult | str | Path,
        volume: np.ndarray | torch.Tensor,
        device: str | torch.device = "cpu",
    ) -> "PolarProjectionDecomposer":
        """Create a decomposer from a pre-computed (or loaded) result.

        Parameters
        ----------
        result : DecompositionResult | str | Path
            A :class:`~panther_em.decomposition.result.DecompositionResult`
            instance or a path to a saved ``.h5`` file.
        volume : np.ndarray | torch.Tensor
            The original 3D volume (needed if further decomposition will be run).
        device : str | torch.device, optional
            Compute device. Default is 'cpu'.

        Returns
        -------
        PolarProjectionDecomposer
            Decomposer instance pre-loaded with the given result.
        """
        if isinstance(result, (str, Path)):
            result = DecompositionResult.load(result)

        if result.coordinate_transform is None:
            raise ValueError(
                "The loaded DecompositionResult has no embedded coordinate_transform. "
                "Re-run decomposition with the current code to produce a result that "
                "contains the transform."
            )

        device_obj = torch.device(device)

        phi_values = torch.zeros(result.num_orientations, device=device_obj)
        theta_values = torch.zeros(result.num_orientations, device=device_obj)

        instance = cls(
            volume,
            phi_values,
            theta_values,
            coordinate_transform=result.coordinate_transform,
            device=device_obj,
        )
        instance._result = result

        return instance
