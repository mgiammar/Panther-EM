"""Inference helpers for projection/feature reconstruction from decomp. results."""

import numpy as np
import torch

from panther_em.decomposition.result import DecompositionResult


class ProjectionReconstructor:
    """Helper for reconstructing polar/cartesian features from a result."""

    def __init__(
        self,
        result: DecompositionResult,
        device: str | torch.device = "cpu",
    ) -> None:
        if result.coordinate_transform is None:
            raise ValueError("DecompositionResult must include a coordinate transform.")

        self.result = result
        self.device = torch.device(device)

        # GridTransform is device-agnostic; GPU dispatch is handled internally.
        self._coordinate_transform = result.coordinate_transform
        self.image_shape: tuple[int, int] = self._coordinate_transform.cartesian_shape

    def clear_coordinate_transform_cache(self) -> None:
        """Remove any cached interpolation grids."""
        self._coordinate_transform.clear_cache()

    def _resolve_num_components(self, num_components: int | None) -> int:
        """Parse possible 'None' value for num_components and enforce max limit."""
        r = self.result
        num_blocks = r.k_max * 2 if r.is_complex_projection else r.k_max
        max_components = num_blocks * r.eig_max

        if num_components is None:
            return max_components

        return min(num_components, max_components)

    def _build_k_idx_groups(self, num_components: int) -> dict[int, list[int]]:
        """Group eig_indices by their angular-frequency k_idx for the top-n components.

        Keys are actual angular frequencies:
            - Real-valued decompositions: ``0 ... k_max``
            - Complex decompositions:     ``-k_max ... k_max``
        """
        k_range = range(-self.result.k_max, self.result.k_max + 1)
        groups: dict[int, list[int]] = {k: [] for k in k_range}

        for k_idx, eig_idx in self.result.get_top_n(
            num_components,
            include_negative=False,  # NOTE: leveraging conj symmetry
        ):
            groups[k_idx].append(eig_idx)

        return groups

    def _get_angular_phase_component(
        self, k_idx: int, phase_shift: float
    ) -> torch.Tensor:
        """Compute the k-th discrete angular Fourier basis vector.

        Parameters
        ----------
        k_idx : int
            Actual angular frequency.  Real-valued: ``[0, k_max]``;
            complex: ``[-k_max, k_max]``.
        phase_shift : float
            Constant phase offset in degrees added to all angular positions.
            Used to analytically apply in-plane rotation.
        """
        phase_shift_rad = np.deg2rad(phase_shift)

        N = self.result.num_angular_components
        n = torch.arange(N, device=self.device, dtype=torch.float32)
        base_angles = 2.0 * np.pi * k_idx * n / N
        angles = base_angles + k_idx * phase_shift_rad

        return torch.exp(1j * angles) / np.sqrt(N)

    def _get_angular_phase_components_batch(
        self, k_idx: int, phase_shifts: torch.Tensor
    ) -> torch.Tensor:
        """Batched angular phase component for a vector of phase shifts.

        Parameters
        ----------
        k_idx : int
            Actual angular frequency.  Real-valued: ``[0, k_max]``;
            complex: ``[-k_max, k_max]``.
        phase_shifts : torch.Tensor
            Phase shifts in degrees, shape (B,).

        Returns
        -------
        torch.Tensor
            Complex angular components, shape (B, num_angular_components).
        """
        phase_shifts_rad = phase_shifts * (np.pi / 180.0)

        N = self.result.num_angular_components
        n = torch.arange(N, device=self.device, dtype=torch.float32)
        base_angles = 2.0 * np.pi * k_idx * n / N
        angles = base_angles[None, :] + k_idx * phase_shifts_rad[:, None]

        return torch.exp(1j * angles) / np.sqrt(N)

    def construct_polar_feature(
        self,
        k_idx: int,
        eig_idx: int,
        in_plane_rotation: float = 0.0,
        return_torch: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """Construct a polar feature for a given k_idx and eig_idx.

        Parameters
        ----------
        k_idx : int
            Actual angular frequency.  Real-valued: ``[0, k_max]``;
            complex: ``[-k_max, k_max]``.
        eig_idx : int
            Index of the radial eigenvector (component).
        in_plane_rotation : float, optional
            In-plane rotation angle in degrees, by default 0.0.
        return_torch : bool, optional
            Whether to return a PyTorch tensor instead of a NumPy array, by default
            False.

        Returns
        -------
        np.ndarray or torch.Tensor
            The constructed polar feature as a 2D array (angular x radial) with shape
            (num_angular_components, num_radial_components).
        """
        angular_component = self._get_angular_phase_component(k_idx, in_plane_rotation)
        _, _, Vh = self.result.get_svd_tensors(
            np.array([[k_idx, eig_idx]]), self.device
        )
        polar_feature = torch.outer(angular_component, Vh[0])

        return polar_feature if return_torch else polar_feature.cpu().numpy()

    def construct_cartesian_feature(
        self,
        k_idx: int,
        eig_idx: int,
        in_plane_rotation: float = 0.0,
        return_torch: bool = False,
        order: int = 5,
        mode: str = "constant",
        cval: float = 0.0,
        preserve_energy: bool = True,
        wrap_angular_axis: bool = True,
    ) -> np.ndarray | torch.Tensor:
        """Construct the cartesian feature for a given k_idx and eig_idx.

        Parameters
        ----------
        k_idx : int
            Index of the angular component (circular mode).
        eig_idx : int
            Index of the radial eigenvector (component).
        in_plane_rotation : float, optional
            In-plane rotation angle in degrees, by default 0.0.
        return_torch : bool, optional
            Whether to return a PyTorch tensor instead of a NumPy array, by default
            False.
        order : int, optional
            The order of the interpolation used in the polar to cartesian
            transformation, by default 5.
        mode : str, optional
            The mode parameter determines how the input array is extended when the
            transformation requires values outside of the input boundaries. By default
            "constant".
        cval : float, optional
            The value to fill past edges of input if mode is "constant", by default 0.0.
        preserve_energy : bool, optional
            Whether to preserve the energy of the feature during the polar to cartesian
            transformation, by default True.
        wrap_angular_axis : bool, optional
            Whether to wrap the angular axis during the polar to cartesian
            transformation, by default True.

        Returns
        -------
        np.ndarray or torch.Tensor
            The constructed cartesian feature as a 2D array with shape corresponding to
            the original image dimensions.
        """
        polar_feature = self.construct_polar_feature(
            k_idx, eig_idx, in_plane_rotation, return_torch=True
        )

        cartesian_feature = self._coordinate_transform.to_cartesian(
            polar_feature,
            order=order,
            mode=mode,
            cval=cval,
            preserve_energy=preserve_energy,
            wrap_angular_axis=wrap_angular_axis,
        )

        return cartesian_feature if return_torch else cartesian_feature.cpu().numpy()

    def reconstruct_projection(
        self,
        orientation_idx: int,
        fourier_filter_idx: int = 0,
        in_plane_rotation: float = 0.0,
        num_components: int | None = None,
        return_polar: bool = False,
        return_torch: bool = False,
        order: int = 5,
        mode: str = "constant",
        cval: float = 0.0,
        preserve_energy: bool = True,
        wrap_angular_axis: bool = True,
    ) -> np.ndarray | torch.Tensor:
        """Reconstruct a projection for a given number of top components.

        Parameters
        ----------
        orientation_idx : int
            Index of the orientation (in-plane rotation) to reconstruct.
        fourier_filter_idx : int, optional
            Index of the Fourier filter to reconstruct, by default 0.
        in_plane_rotation : float, optional
            The in-plane rotation angle for the reconstruction, in degrees. Analytically
            applies a phase-shift to the constructed angular components. By default 0.0.
        num_components : int | None, optional
            Number of top components (by singular value magnitude) to use for
            reconstruction. If None, uses all available components, by default None.
        return_polar : bool, optional
            Whether to return the reconstructed projection in polar coordinates instead
            of cartesian, by default False.
        return_torch : bool, optional
            Whether to return a PyTorch tensor instead of a NumPy array, by default
            False.
        order : int, optional
            The order of the interpolation used in the polar to cartesian
            transformation, by default 5.
        mode : str, optional
            The mode parameter determines how the input array is extended when the
            transformation requires values outside of the input boundaries. By default
            "constant".
        cval : float, optional
            The value to fill past edges of input if mode is "constant", by default 0.0.
        preserve_energy : bool, optional
            Whether to preserve the energy of the feature during the polar to cartesian
            transformation, by default True.
        wrap_angular_axis : bool, optional
            Whether to wrap the angular axis during the polar to cartesian
            transformation, by default True.

        Returns
        -------
        np.ndarray or torch.Tensor
            The reconstructed projection as a 2D array in either polar or cartesian
            coordinates, depending on the value of `return_polar`. The shape will be
            (num_angular_components, num_radial_components) for polar coordinates or the
            original image dimensions for cartesian coordinates.
        """
        if not (0 <= fourier_filter_idx < self.result.num_fourier_filters):
            raise ValueError(
                f"fourier_filter_idx {fourier_filter_idx} out of bounds "
                f"(max: {self.result.num_fourier_filters - 1})"
            )
        if not (0 <= orientation_idx < self.result.num_orientations):
            raise ValueError(
                f"orientation_idx {orientation_idx} out of bounds "
                f"(max: {self.result.num_orientations - 1})"
            )

        num_components = self._resolve_num_components(num_components)
        k_idx_groups = self._build_k_idx_groups(num_components)

        polar_projection = torch.zeros(
            (self.result.num_angular_components, self.result.num_radial_components),
            dtype=torch.complex64,
            device=self.device,
        )

        N = self.result.num_angular_components

        for k_idx, eig_indices in sorted(k_idx_groups.items()):
            # If no components to process
            if not eig_indices:
                continue

            indices_np = np.column_stack(
                [np.full(len(eig_indices), k_idx), eig_indices]
            )
            U_t, S_t, Vh_t = self.result.get_svd_tensors(indices_np, self.device)
            u_ki = U_t[fourier_filter_idx, orientation_idx]  # (E,)
            weighted_features = u_ki * S_t
            radial_contribution = weighted_features @ Vh_t

            angular_component = self._get_angular_phase_component(
                k_idx, in_plane_rotation
            )

            tmp_contribution = torch.outer(angular_component, radial_contribution)
            polar_projection += tmp_contribution

            # For real-valued decompositions the -k block equals conj(+k block).
            # DC (k=0) and Nyquist are self-conjugate; all other stored positive
            # modes must contribute their -k mirror.
            if not self.result.is_complex_projection and k_idx > 0:
                is_nyquist = (N % 2 == 0) and (k_idx == N // 2)
                if not is_nyquist:
                    polar_projection += tmp_contribution.conj()

        if return_polar:
            return polar_projection if return_torch else polar_projection.cpu().numpy()

        cartesian_projection = self._coordinate_transform.to_cartesian(
            polar_projection,
            order=order,
            mode=mode,
            cval=cval,
            preserve_energy=preserve_energy,
            wrap_angular_axis=wrap_angular_axis,
        )

        if return_torch:
            return cartesian_projection
        return (
            cartesian_projection.cpu().numpy()
            if isinstance(cartesian_projection, torch.Tensor)
            else cartesian_projection
        )

    def reconstruct_projection_batch(
        self,
        queries: list[tuple[int, int, float]],
        num_components: int | None = None,
        return_polar: bool = False,
        return_torch: bool = False,
        order: int = 5,
        mode: str = "constant",
        cval: float = 0.0,
        preserve_energy: bool = True,
        wrap_angular_axis: bool = True,
    ) -> np.ndarray | torch.Tensor:
        """Reconstruct projections for a batch of queries in one pass.

        Parameters
        ----------
        queries : list[tuple[int, int, float]]
            Each entry is
            (orientation_idx, fourier_filter_idx, in_plane_rotation_degrees).
        num_components : int | None, optional
            Number of top components to use. None uses all, by default None.
        return_polar : bool, optional
            Return in polar coordinates (B, A, R) instead of cartesian (B, H, W),
            by default False.
        return_torch : bool, optional
            Return a torch.Tensor instead of np.ndarray, by default False.
        order : int, optional
            Interpolation order for polar-to-cartesian warp, by default 5.
        mode : str, optional
            Boundary mode for warp, by default "constant".
        cval : float, optional
            Fill value when mode is "constant", by default 0.0.
        preserve_energy : bool, optional
            Apply Jacobian correction during warp, by default True.
        wrap_angular_axis : bool, optional
            Wrap angular axis during warp, by default True.

        Returns
        -------
        np.ndarray or torch.Tensor
            Shape (B, H, W) in cartesian or (B, A, R) in polar coordinates.
        """
        orientation_indices, ff_indices, in_plane_rotations = zip(
            *queries, strict=False
        )

        for ff_idx in ff_indices:
            if not (0 <= ff_idx < self.result.num_fourier_filters):
                raise ValueError(
                    f"fourier_filter_idx {ff_idx} out of bounds "
                    f"(max: {self.result.num_fourier_filters - 1})"
                )
        for ori_idx in orientation_indices:
            if not (0 <= ori_idx < self.result.num_orientations):
                raise ValueError(
                    f"orientation_idx {ori_idx} out of bounds "
                    f"(max: {self.result.num_orientations - 1})"
                )

        B = len(queries)
        num_components = self._resolve_num_components(num_components)
        k_idx_groups = self._build_k_idx_groups(num_components)

        orient_batch = torch.tensor(list(orientation_indices), device=self.device)
        ff_batch = torch.tensor(list(ff_indices), device=self.device)
        phase_shifts = torch.tensor(
            list(in_plane_rotations), device=self.device, dtype=torch.float32
        )

        polar_projections = torch.zeros(
            (B, self.result.num_angular_components, self.result.num_radial_components),
            dtype=torch.complex64,
            device=self.device,
        )

        N = self.result.num_angular_components

        for k_idx in sorted(k_idx_groups.keys()):
            eig_indices = k_idx_groups[k_idx]
            if not eig_indices:
                continue

            indices_np = np.column_stack(
                [np.full(len(eig_indices), k_idx), eig_indices]
            )
            # angular_batch: (B, A)
            angular_batch = self._get_angular_phase_components_batch(
                k_idx, phase_shifts
            )

            U_t, S_t, Vh_t = self.result.get_svd_tensors(indices_np, self.device)
            # U_t: (FF, O, E); select one (ff, or) pair per batch element
            u_batch = U_t[ff_batch, orient_batch, :]  # (B, E)
            radial = (u_batch * S_t[None, :]) @ Vh_t  # (B, R)
            tmp_contribution = angular_batch[:, :, None] * radial[:, None, :]
            polar_projections += tmp_contribution

            # For real-valued decompositions the -k block equals conj(+k block).
            if not self.result.is_complex_projection and k_idx > 0:
                is_nyquist = (N % 2 == 0) and (k_idx == N // 2)
                if not is_nyquist:
                    polar_projections += tmp_contribution.conj()

        if return_polar:
            return (
                polar_projections if return_torch else polar_projections.cpu().numpy()
            )

        cartesian_projections = self._coordinate_transform.to_cartesian(
            polar_projections,
            order=order,
            mode=mode,
            cval=cval,
            preserve_energy=preserve_energy,
            wrap_angular_axis=wrap_angular_axis,
        )

        if return_torch:
            return cartesian_projections
        return (
            cartesian_projections.cpu().numpy()
            if isinstance(cartesian_projections, torch.Tensor)
            else cartesian_projections
        )
