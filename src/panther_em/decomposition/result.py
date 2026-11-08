"""Dataclass for storing decomposition results."""

import json
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

import panther_em.coordinates  # noqa: F401  # registers built-in transforms
from panther_em.coordinates.transform_base import (
    CoordinateTransform,
    GridTransform,
    reconstruct_transform,
)


def _read_paired_last_two_axes(
    source: np.ndarray | h5py.Dataset,
    idx0: np.ndarray,
    idx1: np.ndarray,
) -> np.ndarray:
    """Read ``source[..., idx0, idx1]``, pairing ``idx0``/``idx1`` elementwise."""
    if isinstance(source, h5py.Dataset):
        return np.stack(
            [source[..., int(i0), int(i1)] for i0, i1 in zip(idx0, idx1, strict=False)],
            axis=-1,
        )
    return source[..., idx0, idx1]


def _read_fancy_last_axis(
    source: np.ndarray | h5py.Dataset, idx: np.ndarray
) -> np.ndarray:
    """Read ``source[..., idx]`` for an arbitrary (unsorted/duplicated) ``idx``."""
    if isinstance(source, h5py.Dataset):
        idx = np.asarray(idx)
        unique_idx, inverse = np.unique(idx, return_inverse=True)
        data = source[..., unique_idx]
        return data[..., inverse]
    return source[..., idx]


@dataclass
class DecompositionResult:
    """Stores the full results of a polar projection decomposition.

    Note
    ----
    The structure of the block-circulant SVD imposes a structure on the singular values
    where each singular value is associated with a specific angular frequency index
    `k_idx` in the range of `[0, k_max)` and a radial eigenvalue index `eig_idx` in the
    range of `[0, eig_max)`.

    For complex-valued underlying projection data there will be
    `num_freq_blocks = k_max * 2 + 1` total frequency blocks corresponding to the
    positive and negative frequencies including the DC (k=0) component. When projection
    data is real-valued, only the non-negative frequencies are stored, so
    `num_freq_blocks = k_max + 1`, but negative frequencies can still be indexed.

    The left-singular-vectors (`U`) have shape (..., num_freq_blocks, eig_idx), the
    singular-values (`S`) have shape (num_freq_blocks, eig_idx), and the
    right-singular-vectors (`Vh`) have shape (num_freq_blocks, eig_idx,
    num_radial_components). Helper methods exist for querying the top L pairs based on
    singular value magnitude; similar methods exist for retrieving the corresponding
    singular vectors and values.

    Note on multiple volumes
    -------------------------
    When decomposition is run across multiple volumes (e.g. sampled across a
    conformation/state coordinate), `U` gains a leading `num_volumes` axis while `S` and
    `Vh` remain unchanged. The SVD basis is a single joint fit across all volumes's
    `(fourier_filter, orientation)` samples simultaneously.

    Results are saved and loaded as HDF5 files (``.h5`` / ``.hdf5``) via
    :meth:`save` and :meth:`load`. For results whose ``U`` array is too large to
    fit in memory (e.g. >100GB), pass ``mmap=True`` to :meth:`load` to keep the
    HDF5 file open on disk and read ``U`` lazily on demand; call :meth:`close`
    (or use the result as a context manager) to release the file handle when
    done.

    Attributes
    ----------
    U : np.ndarray | h5py.Dataset
        Left singular vectors with shape
        (num_volumes, num_filters, num_orient, num_freq_blocks, num_radial_components).
        A disk-backed ``h5py.Dataset`` rather than an in-memory array when the
        result was loaded with ``mmap=True``.
    S : np.ndarray
        Singular values with shape (num_freq_blocks, num_radial_components).
    Vh : np.ndarray
        Right singular vectors (conjugate transpose) with shape
        (num_freq_blocks, num_radial_components, num_radial_components).
    num_volumes : int
        Number of volumes used in decomposition. Default is 1 (single-volume).
    num_fourier_filters : int
        Number of Fourier filters (defocus channels) used in decomposition.
    num_orientations : int
        Number of orientations used in the decomposition.
    num_angular_components : int
        Number of angular components in polar space.
    num_radial_components : int
        Number of radial components in polar space. Also the number of radial
        eigenvectors stored per angular frequency component
    k_max : int
        Maximum angular frequency index used.
    eig_max : int
        Maximum radial eigenvalue index used.
    is_complex_projection : bool
        Whether the underlying projection data is complex-valued.
    created_at : str
        ISO format timestamp of when the result was created.
    phi_values : np.ndarray | None
        Phi (azimuthal) ZYZ Euler angles in degrees used during decomposition, shape
        ``(num_orientations,)``. ``None`` if not recorded.
    theta_values : np.ndarray | None
        Theta (polar) ZYZ Euler angles in degrees used during decomposition, shape
        ``(num_orientations,)``. ``None`` if not recorded.
    fourier_filters : np.ndarray | None
        Fourier-space filters (e.g. CTF envelopes) applied during decomposition, shape
        ``(num_fourier_filters, ...)``. ``None`` if no filters were applied.
    volume_labels : np.ndarray | None
        Optional per-volume labels/coordinates (e.g. a conformation/state
        coordinate value), shape ``(num_volumes,)``. ``None`` if not recorded.
    """

    # Core SVD components
    U: np.ndarray | h5py.Dataset
    S: np.ndarray
    Vh: np.ndarray

    # Shapes for singular values
    k_max: int
    eig_max: int
    is_complex_projection: bool

    # Decomposition metadata
    num_fourier_filters: int
    num_orientations: int
    num_angular_components: int
    num_radial_components: int
    num_volumes: int = 1

    # Timestamp
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Optional orientation recorded at decomposition time
    phi_values: np.ndarray | None = None
    theta_values: np.ndarray | None = None

    # Optional Fourier filters recorded at decomposition time
    fourier_filters: np.ndarray | None = None

    # Optional per-volume labels/coordinates recorded at decomposition time
    volume_labels: np.ndarray | None = None

    # Optional coordinate transform used during decomposition
    # When present, DecompositionResult.save() embeds the transform's geometric
    # parameters so the file is self-contained for inference.
    coordinate_transform: CoordinateTransform | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate shapes after initialization."""
        if self.is_complex_projection:
            num_freq_blocks = self.k_max * 2  # NOTE: need plus 1?
        else:
            num_freq_blocks = self.k_max  # NOTE: need plus 1?

        expected_U_shape = (
            self.num_volumes,
            self.num_fourier_filters,
            self.num_orientations,
            num_freq_blocks,
            self.eig_max,
        )
        expected_S_shape = (num_freq_blocks, self.eig_max)
        expected_Vh_shape = (
            num_freq_blocks,
            self.eig_max,
            self.num_radial_components,
        )

        if self.S.shape != expected_S_shape:
            raise ValueError(
                f"S shape {self.S.shape} does not match expected {expected_S_shape}"
            )
        if self.U.shape != expected_U_shape:
            raise ValueError(
                f"U shape {self.U.shape} does not match expected {expected_U_shape}"
            )
        if self.Vh.shape != expected_Vh_shape:
            raise ValueError(
                f"Vh shape {self.Vh.shape} does not match expected {expected_Vh_shape}"
            )

        # Validate optional angle arrays
        if self.phi_values is not None:
            if self.phi_values.shape != (self.num_orientations,):
                raise ValueError(
                    f"phi_values shape {self.phi_values.shape} does not match "
                    f"expected ({self.num_orientations},)"
                )
        if self.theta_values is not None:
            if self.theta_values.shape != (self.num_orientations,):
                raise ValueError(
                    f"theta_values shape {self.theta_values.shape} does not match "
                    f"expected ({self.num_orientations},)"
                )

        # Validate optional Fourier filters
        if self.fourier_filters is not None:
            if self.fourier_filters.shape[0] != self.num_fourier_filters:
                raise ValueError(
                    f"fourier_filters leading dimension "
                    f"{self.fourier_filters.shape[0]} does not match "
                    f"num_fourier_filters={self.num_fourier_filters}"
                )

        # Validate optional volume labels
        if self.volume_labels is not None:
            if self.volume_labels.shape != (self.num_volumes,):
                raise ValueError(
                    f"volume_labels shape {self.volume_labels.shape} does not match "
                    f"expected ({self.num_volumes},)"
                )

        # Per-instance cache for top-n queries
        self._top_n_cache: dict[tuple[int, bool], np.ndarray] = {}

        # Open HDF5 file handle backing lazily-loaded arrays (set by load(mmap=True)).
        if not hasattr(self, "_h5_file"):
            self._h5_file: h5py.File | None = None

    def close(self) -> None:
        """Close the HDF5 file handle opened by ``load(..., mmap=True)``.

        No-op if this result was not loaded with ``mmap=True``, or the file handle has
        already been closed. Any arrays not yet read from disk (e.g. ``U``) become
        unusable after this call.
        """
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __enter__(self) -> "DecompositionResult":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        """String representation of the DecompositionResult."""
        transform_name = (
            getattr(self.coordinate_transform, "transform_name", None)
            if self.coordinate_transform is not None
            else None
        )
        s = textwrap.dedent(f"""
            DecompositionResult(
                k_max={self.k_max},
                eig_max={self.eig_max},
                is_complex_projection={self.is_complex_projection},
                num_volumes={self.num_volumes},
                num_fourier_filters={self.num_fourier_filters},
                num_orientations={self.num_orientations},
                num_angular_components={self.num_angular_components},
                num_radial_components={self.num_radial_components},
                has_euler_angles={self.phi_values is not None},
                has_fourier_filters={self.fourier_filters is not None},
                coordinate_transform='{transform_name}',
                created_at='{self.created_at}'
            )
            """)

        return s.strip()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _write_hdf5_metadata(self, f: h5py.File) -> None:
        """Write scalar metadata as root-level HDF5 attributes."""
        f.attrs["k_max"] = self.k_max
        f.attrs["eig_max"] = self.eig_max
        f.attrs["is_complex_projection"] = self.is_complex_projection
        f.attrs["num_volumes"] = self.num_volumes
        f.attrs["num_fourier_filters"] = self.num_fourier_filters
        f.attrs["num_orientations"] = self.num_orientations
        f.attrs["num_angular_components"] = self.num_angular_components
        f.attrs["num_radial_components"] = self.num_radial_components
        f.attrs["created_at"] = self.created_at

        # Embed coordinate transform geometry so the file is self-contained.
        # A result without a transform cannot be saved — inference depends on it.
        if self.coordinate_transform is None:
            raise ValueError(
                "DecompositionResult.coordinate_transform is None. "
                "A coordinate transform must be set before saving. "
                "Run do_decomposition() first, or assign a CoordinateTransform "
                "instance to result.coordinate_transform."
            )
        f.attrs["transform_config"] = json.dumps(self.coordinate_transform.to_dict())

    def _write_hdf5_optional_arrays(self, f: h5py.File) -> None:
        """Write optional orientation angles and Fourier filters if present."""
        if self.phi_values is not None:
            f.create_dataset("phi_values", data=self.phi_values)
        if self.theta_values is not None:
            f.create_dataset("theta_values", data=self.theta_values)
        if self.fourier_filters is not None:
            f.create_dataset("fourier_filters", data=self.fourier_filters)
        if self.volume_labels is not None:
            f.create_dataset("volume_labels", data=self.volume_labels)

    def _write_hdf5_transform_arrays(self, f: h5py.File) -> None:
        """Write coordinate grid arrays for a GridTransform into an HDF5 group."""
        if not isinstance(self.coordinate_transform, GridTransform):
            return

        grp = f.create_group("transform")
        grp.create_dataset(
            "transform_coords", data=self.coordinate_transform.transform_coords
        )
        grp.create_dataset(
            "cartesian_coords", data=self.coordinate_transform.cartesian_coords
        )
        jac = self.coordinate_transform.jacobian_grid
        if jac is not None:
            grp.create_dataset("jacobian", data=jac)

    def save(self, path: str | Path) -> None:
        """Save the full decomposition result to an HDF5 file.

        Scalar metadata is stored as root-level HDF5 attributes.  To save only the most
        significant components and reduce file size, use :meth:`save_top_n` instead.

        Parameters
        ----------
        path : str | Path
            Destination path.  The extension is replaced with ``.h5`` if it
            is not already ``.h5`` or ``.hdf5``.
        """
        path = Path(path)
        if path.suffix not in {".h5", ".hdf5"}:
            path = path.with_suffix(".h5")

        with h5py.File(path, "w") as f:
            self._write_hdf5_metadata(f)
            f.create_dataset("U", data=self.U)
            f.create_dataset("S", data=self.S)
            f.create_dataset("Vh", data=self.Vh)
            self._write_hdf5_optional_arrays(f)
            self._write_hdf5_transform_arrays(f)

    def save_top_n(self, path: str | Path, top_k: int) -> None:
        r"""Save only the ``top_k`` most significant components to HDF5.

        Convenience wrapper for ``self.to_sparse(top_k).save(path)``.

        Parameters
        ----------
        path : str | Path
            Destination path.  Extension is normalized to ``.h5``.
        top_k : int
            Number of unique ``(k_stored, eig_idx)`` pairs to store.

        Raises
        ------
        ValueError
            If ``top_k < 1``.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.to_sparse(top_k).save(path)

    def to_dense(self) -> "DecompositionResult":
        """Return self — already a dense result.

        Returns
        -------
        DecompositionResult
            This instance unchanged.
        """
        return self

    def to_sparse(self, n: int) -> "SparseDecompositionResult":
        """Create a SparseDecompositionResult keeping only the top-n components.

        Parameters
        ----------
        n : int
            Number of unique ``(k_stored, eig_idx)`` pairs to retain, ranked
            by ``|S|``.

        Returns
        -------
        SparseDecompositionResult
            Compact representation storing only the n most significant
            components.

        Raises
        ------
        ValueError
            If ``n < 1``.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        pairs = self.get_top_n(n, include_negative=False)
        pairs = pairs[:n]
        k_stored_arr = pairs[:, 0].astype(int)
        eig_idx_arr = pairs[:, 1].astype(int)

        U_data = _read_paired_last_two_axes(self.U, k_stored_arr, eig_idx_arr)
        S_data = self.S[k_stored_arr, eig_idx_arr]
        Vh_data = self.Vh[k_stored_arr, eig_idx_arr, :]
        selected_indices = np.stack([k_stored_arr, eig_idx_arr], axis=1)

        return SparseDecompositionResult.from_sparse_arrays(
            U_data=U_data,
            S_data=S_data,
            Vh_data=Vh_data,
            selected_indices=selected_indices,
            k_max=self.k_max,
            eig_max=self.eig_max,
            is_complex_projection=self.is_complex_projection,
            num_volumes=self.num_volumes,
            num_fourier_filters=self.num_fourier_filters,
            num_orientations=self.num_orientations,
            num_angular_components=self.num_angular_components,
            num_radial_components=self.num_radial_components,
            coordinate_transform=self.coordinate_transform,
            created_at=self.created_at,
            phi_values=self.phi_values,
            theta_values=self.theta_values,
            fourier_filters=self.fourier_filters,
            volume_labels=self.volume_labels,
        )

    @classmethod
    def load(cls, path: str | Path, mmap: bool = False) -> "DecompositionResult":
        """Load a decomposition result from an HDF5 file written by :meth:`save`.

        Parameters
        ----------
        path : str | Path
            Path to the ``.h5`` or ``.hdf5`` file.
        mmap : bool, optional
            If True, keep the HDF5 file open on disk and read ``U`` lazily on demand.
            ``S``, ``Vh``, and other metadata/arrays are always loaded eagerly since
            they are typically tiny compared to ``U``. Useful when ``U`` is larger than
            available RAM. Call :meth:`close` (or use the result as a context manager)
            to release the file handle when done. Default is False.

        Returns
        -------
        DecompositionResult
            The loaded decomposition result.

        Raises
        ------
        ValueError
            If the file extension is not ``.h5`` or ``.hdf5``, or if the file
            contains a sparse result (use :meth:`SparseDecompositionResult.load`
            instead).
        """
        path = Path(path)
        if path.suffix not in {".h5", ".hdf5"}:
            raise ValueError(
                f"Unknown file extension '{path.suffix}'. Expected '.h5' or '.hdf5'."
            )
        with h5py.File(path, "r") as f:
            if bool(f.attrs.get("is_sparse", False)):
                raise ValueError(
                    f"File '{path}' contains a sparse result. "
                    "Use SparseDecompositionResult.load() instead."
                )
        return cls._load_hdf5(path, mmap=mmap)

    @classmethod
    def _load_hdf5(cls, path: Path, mmap: bool = False) -> "DecompositionResult":
        """Load from an HDF5 file produced by :meth:`save`.

        Parameters
        ----------
        path : Path
            Path to the HDF5 file.
        mmap : bool, optional
            If True, ``U`` is left as a disk-backed ``h5py.Dataset`` and the
            underlying file handle is kept open on the returned instance
            (``result._h5_file``) rather than closed. Default is False.
        """
        f = h5py.File(path, "r")
        try:
            k_max = int(f.attrs["k_max"])
            eig_max = int(f.attrs["eig_max"])
            is_complex = bool(f.attrs["is_complex_projection"])
            # .get() with a default of 1 for backward compatibility with files
            # saved before the multi-volume axis was introduced.
            num_vol = int(f.attrs.get("num_volumes", 1))
            num_ff = int(f.attrs["num_fourier_filters"])
            num_or = int(f.attrs["num_orientations"])
            num_ang = int(f.attrs["num_angular_components"])
            num_r = int(f.attrs["num_radial_components"])

            # h5py >= 3 returns str; older versions may return bytes
            created_at = f.attrs["created_at"]
            if isinstance(created_at, bytes):
                created_at = created_at.decode("utf-8")

            phi_values = f["phi_values"][()] if "phi_values" in f else None
            theta_values = f["theta_values"][()] if "theta_values" in f else None
            fourier_filters = (
                f["fourier_filters"][()] if "fourier_filters" in f else None
            )
            volume_labels = f["volume_labels"][()] if "volume_labels" in f else None

            # Load coordinate transform — required; raise if absent or unreadable.
            if "transform_config" not in f.attrs:
                raise KeyError(
                    f"The file '{path}' has no 'transform_config' attribute. "
                )

            transform_json = f.attrs["transform_config"]
            if isinstance(transform_json, bytes):
                transform_json = transform_json.decode("utf-8")
            transform_params = json.loads(transform_json)
            coordinate_transform: CoordinateTransform

            if transform_params.get("transform_name") == "grid":
                transform_coords = f["transform/transform_coords"][()]
                cartesian_coords = f["transform/cartesian_coords"][()]
                jacobian = (
                    f["transform/jacobian"][()] if "transform/jacobian" in f else None
                )
                polar_shape = (
                    int(transform_params["polar_shape"][0]),
                    int(transform_params["polar_shape"][1]),
                )
                cartesian_shape = (
                    int(transform_params["cartesian_shape"][0]),
                    int(transform_params["cartesian_shape"][1]),
                )
                coordinate_transform = GridTransform.from_arrays(
                    transform_coords=transform_coords,
                    cartesian_coords=cartesian_coords,
                    jacobian=jacobian,
                    polar_shape=polar_shape,
                    cartesian_shape=cartesian_shape,
                    source_params=transform_params.get("source_params"),
                    has_periodic_axis=bool(
                        transform_params.get("has_periodic_axis", True)
                    ),
                    periodic_axis=int(transform_params.get("periodic_axis", 0)),
                )
            else:
                coordinate_transform = reconstruct_transform(transform_params)

            U: np.ndarray | h5py.Dataset = f["U"] if mmap else f["U"][()]
            S = f["S"][()]
            Vh = f["Vh"][()]

            # Legacy files (pre multi-volume support) stored U without the
            # leading volume axis; insert it so num_volumes=1 is self-consistent.
            if U.ndim == 4:
                U = U[np.newaxis, ...]

            result = cls(
                U=U,
                S=S,
                Vh=Vh,
                k_max=k_max,
                eig_max=eig_max,
                is_complex_projection=is_complex,
                num_volumes=num_vol,
                num_fourier_filters=num_ff,
                num_orientations=num_or,
                num_angular_components=num_ang,
                num_radial_components=num_r,
                created_at=created_at,
                phi_values=phi_values,
                theta_values=theta_values,
                fourier_filters=fourier_filters,
                volume_labels=volume_labels,
                coordinate_transform=coordinate_transform,
            )
        except Exception:
            f.close()
            raise

        if mmap:
            result._h5_file = f
        else:
            f.close()
        return result

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def k_to_stored(self, k_idx: int | np.ndarray) -> int | np.ndarray:
        """Convert an angular-frequency index to its storage-array row index."""
        if self.is_complex_projection:
            return k_idx + self.k_max

        # Since singular values are real-valued, complex conjugate returns same
        return np.abs(k_idx) if isinstance(k_idx, np.ndarray) else abs(k_idx)

    def is_conjugate_mode(self, k_idx: int | np.ndarray) -> bool | np.ndarray:
        """Helper for determining when conj is needed across real/complex modes."""
        # Always False for complex-valued decompositions
        if self.is_complex_projection:
            return (
                np.zeros_like(k_idx, dtype=bool)
                if isinstance(k_idx, np.ndarray)
                else False
            )

        # True when k_idx is negative for real-valued decompositions
        return (
            (k_idx < 0)
            if isinstance(k_idx, (int, np.integer))
            else (np.asarray(k_idx) < 0)
        )

    def get_top_n(self, top_k: int, include_negative: bool = True) -> np.ndarray:
        """Get the top-n singular values across all (frequency, radial) pairs.

        Parameters
        ----------
        top_k : int
            The number of top singular values to retrieve.
        include_negative : bool, optional
            If this is a real-valued decomposition, this flag chooses whether to include
            negative frequency indices (default True) or to only include non-negative
            angular frequencies (False). No effect when is_complex_projection is True.

        Returns
        -------
        np.ndarray
            An array of shape (l, 2) where l <= top_k containing the (k_idx, eig_idx)
            pairs corresponding to the top singular values sorted by decreasing
            magnitude.
        """
        cache_key = (top_k, include_negative)
        if cache_key not in self._top_n_cache:
            all_svs = self.S.flatten()
            all_svs_sorted = np.argsort(np.abs(all_svs))[::-1]
            top_indices = all_svs_sorted[:top_k]
            k_indices, eig_indices = np.unravel_index(top_indices, self.S.shape)

            # Handle real-valued decomposition case where must also include negative k
            # indices for each positive k index
            if not self.is_complex_projection and include_negative:
                tmp_k_indices = []
                tmp_eig_indices = []

                for k_idx, eig_idx in zip(k_indices, eig_indices, strict=False):
                    if k_idx == 0:
                        tmp_k_indices.append(k_idx)
                        tmp_eig_indices.append(eig_idx)
                    else:
                        tmp_k_indices.extend([k_idx, -k_idx])
                        tmp_eig_indices.extend([eig_idx, eig_idx])

                    # Exit condition since could be 2x number of entries
                    if len(tmp_k_indices) >= top_k:
                        break

                k_indices = np.array(tmp_k_indices)[:top_k]
                eig_indices = np.array(tmp_eig_indices)[:top_k]

            self._top_n_cache[cache_key] = np.stack((k_indices, eig_indices), axis=-1)

        return self._top_n_cache[cache_key]

    def get_component(
        self,
        k_idx: int | np.ndarray,  # shape (l,)
        eig_idx: int | np.ndarray,  # shape (l,)
        return_u: bool = True,
        return_s: bool = True,
        return_vh: bool = True,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Helper method for retrieving specific components of the SVD.

        Parameters
        ----------
        k_idx : int | np.ndarray
            Actual angular-frequency index or indices.  Valid range is
            ``[0, k_max]`` for real-valued and ``[-k_max, k_max]`` for complex
            decompositions.
        eig_idx : int | np.ndarray
            Eigenvalue index or indices.
        return_u : bool, optional
            Whether to return the left singular vectors. Default is True.
        return_s : bool, optional
            Whether to return the singular values. Default is True.
        return_vh : bool, optional
            Whether to return the right singular vectors. Default is True.

        Returns
        -------
        tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]
            A tuple containing the requested components. Elements not selected by
            (return_u, return_s, return_vh) will be None.
        """
        # Must be returning at least one component
        if not (return_u or return_s or return_vh):
            raise ValueError(
                "At least one of return_u, return_s, return_vh must be True."
            )

        k_idx_arr = np.atleast_1d(k_idx)
        k_stored = np.atleast_1d(self.k_to_stored(k_idx_arr))
        eig_idx = np.atleast_1d(eig_idx)
        conj_mask = np.atleast_1d(self.is_conjugate_mode(k_idx_arr))

        u = None
        s = None
        vh = None

        if return_u:
            u = _read_paired_last_two_axes(self.U, k_stored, eig_idx)
            if conj_mask.any():
                u = u.copy()
                u[..., conj_mask] = u[..., conj_mask].conj()
        if return_s:
            s = self.S[k_stored, eig_idx]
        if return_vh:
            vh = self.Vh[k_stored, eig_idx]
            if conj_mask.any():
                vh = vh.copy()
                vh[conj_mask] = vh[conj_mask].conj()

        return u, s, vh

    def get_svd_tensors(
        self,
        indices: np.ndarray,
        device: str | torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get device-allocated SVD tensors for the given ``(k_idx, eig_idx)`` pairs.

        Delegates to :meth:`get_component` so dense and sparse results both work
        correctly via polymorphism, including automatic conjugation for negative
        ``k_idx`` in real-valued decompositions.

        Parameters
        ----------
        indices : np.ndarray
            Shape ``(L, 2)`` integer array; each row is ``(k_idx, eig_idx)``.
        device : str or torch.device
            Target device for the returned tensors.

        Returns
        -------
        U : torch.Tensor
            Left singular vectors, shape ``(V, FF, O, L)``, complex64.
        S : torch.Tensor
            Singular values, shape ``(L,)``, float32.
        Vh : torch.Tensor
            Right singular vectors, shape ``(L, R)``, complex64.
        """
        k_idx = indices[:, 0]
        eig_idx = indices[:, 1]
        U_np, S_np, Vh_np = self.get_component(k_idx, eig_idx)
        return (
            torch.tensor(U_np, dtype=torch.complex64, device=device),
            torch.tensor(S_np, dtype=torch.float32, device=device),
            torch.tensor(Vh_np, dtype=torch.complex64, device=device),
        )

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def scree_plot(
        self, k_idx: int | None = None, **kwargs: dict[str, Any]
    ) -> tuple[plt.Figure, plt.Axes]:
        """Helper function for plotting sorted singular values vs index.

        Parameters
        ----------
        k_idx : int | None, optional
            Actual angular frequency to plot. Valid range is ``[0, k_max]``
            for real-valued and ``[-k_max, k_max]`` for complex decompositions.
            If None, plot all singular values across all frequencies, sorted by
            decreasing magnitude. Default is None.
        **kwargs : dict[str, Any]
            Additional keyword arguments to pass to plt.subplots()

        Returns
        -------
        plt.Figure, plt.Axes
            The matplotlib figure object and axes containing the scree plot.
        """
        fig, ax = plt.subplots(**kwargs)
        svs = (
            self.S[self.k_to_stored(k_idx), :]
            if k_idx is not None
            else self.S.flatten()
        )
        title = (
            f"Scree Plot for k={k_idx}"
            if k_idx is not None
            else "Scree Plot for All Singular Values"
        )

        sorted_svs = svs[np.argsort(np.abs(svs))[::-1]]
        ax.plot(sorted_svs)
        ax.set_title(title)
        ax.set_xlabel("Index (sorted by magnitude)")
        ax.set_ylabel("Singular Value (magnitude)")
        return fig, ax

    def variance_explained_plot(
        self,
        k_idx: int | None = None,
        inverted: bool = True,
        **kwargs: dict[str, Any],
    ) -> tuple[plt.Figure, plt.Axes]:
        """Helper function for plotting cumulative variance explained.

        Parameters
        ----------
        k_idx : int | None, optional
            Actual angular frequency to plot.  Valid range is ``[0, k_max]``
            for real-valued and ``[-k_max, k_max]`` for complex decompositions.
            If None, plot for all frequencies. Default is None.
        inverted : bool, optional
            If True, plot (1 - var exp) rather than (var exp). Default is True.
        **kwargs : dict[str, Any]
            Additional keyword arguments to pass to plt.subplots()

        Returns
        -------
        plt.Figure, plt.Axes
            The matplotlib figure object and axes containing the variance explained
            plot.
        """
        fig, ax = plt.subplots(**kwargs)
        svs = (
            self.S[self.k_to_stored(k_idx), :]
            if k_idx is not None
            else self.S.flatten()
        )
        title = (
            f"Cumulative Variance Explained for k={k_idx}"
            if k_idx is not None
            else "Cumulative Variance Explained for All Singular Values"
        )

        sorted_svs = svs[np.argsort(np.abs(svs))[::-1]]
        denom = np.sum(np.abs(sorted_svs) ** 2)
        variance_explained = np.cumsum(np.abs(sorted_svs) ** 2) / denom
        if inverted:
            variance_explained = 1 - variance_explained

        ax.plot(variance_explained)
        ax.set_title(title)
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Variance Explained")
        return fig, ax


class SparseDecompositionResult(DecompositionResult):
    """Sparse SVD result storing only the top-L most significant components.

    Unlike :class:`DecompositionResult`, this class holds compact arrays of shape
    ``(L,)`` / ``(num_ff, num_or, L)`` / ``(L, num_r)`` rather than full zero-padded
    dense tensors. Memory scales with L, not with ``num_freq_blocks * eig_max``.

    Attributes
    ----------
    num_stored : int
        Number of ``(k_stored, eig_idx)`` pairs retained.
    """

    # Class-level annotations for private compact arrays (set by from_sparse_arrays).
    _U_data: np.ndarray | h5py.Dataset
    _S_data: np.ndarray
    _Vh_data: np.ndarray
    _selected_indices: np.ndarray
    _index_map: dict[tuple[int, int], int]
    _top_n_cache: dict[tuple[int, bool], np.ndarray]

    # Properties that raise rather than returning non-existent dense arrays.

    @property
    def U(self) -> np.ndarray:
        """Not available — use get_component() or to_dense()."""
        raise AttributeError(
            "SparseDecompositionResult does not store the full U array. "
            "Use get_component() to retrieve specific components, or "
            "call to_dense() to materialize the full result."
        )

    @property
    def S(self) -> np.ndarray:
        """Not available — use get_component() or to_dense()."""
        raise AttributeError(
            "SparseDecompositionResult does not store the full S array. "
            "Use get_component() or to_dense()."
        )

    @property
    def Vh(self) -> np.ndarray:
        """Not available — use get_component() or to_dense()."""
        raise AttributeError(
            "SparseDecompositionResult does not store the full Vh array. "
            "Use get_component() to retrieve specific components, or "
            "call to_dense() to materialize the full result."
        )

    @property
    def num_stored(self) -> int:
        """Number of (k_stored, eig_idx) pairs stored."""
        return len(self._selected_indices)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_sparse_arrays(
        cls,
        U_data: np.ndarray,
        S_data: np.ndarray,
        Vh_data: np.ndarray,
        selected_indices: np.ndarray,
        k_max: int,
        eig_max: int,
        is_complex_projection: bool,
        num_fourier_filters: int,
        num_orientations: int,
        num_angular_components: int,
        num_radial_components: int,
        num_volumes: int = 1,
        coordinate_transform: CoordinateTransform | None = None,
        created_at: str | None = None,
        phi_values: np.ndarray | None = None,
        theta_values: np.ndarray | None = None,
        fourier_filters: np.ndarray | None = None,
        volume_labels: np.ndarray | None = None,
        _h5_file: h5py.File | None = None,
    ) -> "SparseDecompositionResult":
        """Construct from compact arrays.

        Parameters
        ----------
        U_data : np.ndarray
            Left singular vectors, shape ``(num_vol, num_ff, num_or, L)``.
        S_data : np.ndarray
            Singular values, shape ``(L,)``.
        Vh_data : np.ndarray
            Right singular vectors, shape ``(L, num_r)``.
        selected_indices : np.ndarray
            ``(L, 2)`` integer array; column 0 = ``k_stored``,
            column 1 = ``eig_idx``.
        k_max : int
            Maximum angular frequency index.
        eig_max : int
            Maximum radial eigenvalue index.
        is_complex_projection : bool
            Whether the underlying projection data is complex-valued.
        num_fourier_filters : int
            Number of Fourier filter channels.
        num_orientations : int
            Number of projection orientations.
        num_angular_components : int
            Number of angular components in polar space.
        num_radial_components : int
            Number of radial components in polar space.
        num_volumes : int, optional
            Number of volumes used in decomposition. Default is 1.
        coordinate_transform : CoordinateTransform | None, optional
            Coordinate transform embedded at decomposition time.
        created_at : str | None, optional
            ISO timestamp; defaults to now if None.
        phi_values : np.ndarray | None, optional
            Phi Euler angles used during decomposition.
        theta_values : np.ndarray | None, optional
            Theta Euler angles used during decomposition.
        fourier_filters : np.ndarray | None, optional
            Fourier-space filters applied during decomposition.
        volume_labels : np.ndarray | None, optional
            Per-volume labels/coordinates, shape ``(num_volumes,)``.
        _h5_file : h5py.File | None, optional
            Open HDF5 file handle backing ``U_data`` when it is a disk-backed
            ``h5py.Dataset`` (set internally by ``load(..., mmap=True)``).
        """
        L = int(selected_indices.shape[0])
        if selected_indices.shape != (L, 2):
            raise ValueError(
                f"selected_indices must have shape (L, 2), got {selected_indices.shape}"
            )
        if S_data.shape != (L,) or Vh_data.shape[0] != L or U_data.shape[-1] != L:
            raise ValueError(
                "Sparse arrays have inconsistent leading dimension L; expected "
                "S=(L,), Vh=(L, R), and U[..., L] to match selected_indices."
            )

        obj: SparseDecompositionResult = object.__new__(cls)
        obj.k_max = k_max
        obj.eig_max = eig_max
        obj.is_complex_projection = is_complex_projection
        obj.num_volumes = num_volumes
        obj.num_fourier_filters = num_fourier_filters
        obj.num_orientations = num_orientations
        obj.num_angular_components = num_angular_components
        obj.num_radial_components = num_radial_components
        obj.coordinate_transform = coordinate_transform
        obj.created_at = created_at or datetime.now().isoformat()
        obj.phi_values = phi_values
        obj.theta_values = theta_values
        obj.fourier_filters = fourier_filters
        obj.volume_labels = volume_labels

        obj._U_data = U_data
        obj._S_data = S_data
        obj._Vh_data = Vh_data
        obj._selected_indices = selected_indices.astype(int)
        obj._index_map = {
            (int(k), int(e)): i for i, (k, e) in enumerate(obj._selected_indices)
        }
        obj._top_n_cache = {}
        obj._h5_file = _h5_file
        return obj

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_dense(self) -> DecompositionResult:
        """Scatter compact arrays into a full zero-padded DecompositionResult.

        Returns
        -------
        DecompositionResult
            Dense result with U/S/Vh of full shape; non-stored entries are
            zero.
        """
        num_freq_blocks = self.k_max * 2 if self.is_complex_projection else self.k_max

        k_stored_arr = self._selected_indices[:, 0]
        eig_idx_arr = self._selected_indices[:, 1]

        U_dense = np.zeros(
            (
                self.num_volumes,
                self.num_fourier_filters,
                self.num_orientations,
                num_freq_blocks,
                self.eig_max,
            ),
            dtype=np.complex64,
        )
        S_dense = np.zeros((num_freq_blocks, self.eig_max), dtype=np.float32)
        Vh_dense = np.zeros(
            (num_freq_blocks, self.eig_max, self.num_radial_components),
            dtype=np.complex64,
        )

        U_dense[..., k_stored_arr, eig_idx_arr] = self._U_data
        S_dense[k_stored_arr, eig_idx_arr] = self._S_data
        Vh_dense[k_stored_arr, eig_idx_arr, :] = self._Vh_data

        return DecompositionResult(
            U=U_dense,
            S=S_dense,
            Vh=Vh_dense,
            k_max=self.k_max,
            eig_max=self.eig_max,
            is_complex_projection=self.is_complex_projection,
            num_volumes=self.num_volumes,
            num_fourier_filters=self.num_fourier_filters,
            num_orientations=self.num_orientations,
            num_angular_components=self.num_angular_components,
            num_radial_components=self.num_radial_components,
            created_at=self.created_at,
            phi_values=self.phi_values,
            theta_values=self.theta_values,
            fourier_filters=self.fourier_filters,
            volume_labels=self.volume_labels,
            coordinate_transform=self.coordinate_transform,
        )

    def to_sparse(self, n: int) -> "SparseDecompositionResult":
        """Return a new SparseDecompositionResult keeping only the top-n components.

        Parameters
        ----------
        n : int
            Number of ``(k_stored, eig_idx)`` pairs to retain. If ``n >= num_stored``
            the current instance is returned unchanged.

        Returns
        -------
        SparseDecompositionResult
            Sub-selected sparse result.

        Raises
        ------
        ValueError
            If ``n < 1``.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if n >= self.num_stored:
            return self

        pairs = self.get_top_n(n, include_negative=False)
        pairs = pairs[:n]
        k_stored_arr = pairs[:, 0].astype(int)
        eig_idx_arr = pairs[:, 1].astype(int)

        compact_indices = np.array(
            [
                self._index_map[(int(k), int(e))]
                for k, e in zip(k_stored_arr, eig_idx_arr, strict=False)
            ]
        )

        return SparseDecompositionResult.from_sparse_arrays(
            U_data=_read_fancy_last_axis(self._U_data, compact_indices),
            S_data=self._S_data[compact_indices],
            Vh_data=self._Vh_data[compact_indices],
            selected_indices=self._selected_indices[compact_indices],
            k_max=self.k_max,
            eig_max=self.eig_max,
            is_complex_projection=self.is_complex_projection,
            num_volumes=self.num_volumes,
            num_fourier_filters=self.num_fourier_filters,
            num_orientations=self.num_orientations,
            num_angular_components=self.num_angular_components,
            num_radial_components=self.num_radial_components,
            coordinate_transform=self.coordinate_transform,
            created_at=self.created_at,
            phi_values=self.phi_values,
            theta_values=self.theta_values,
            fourier_filters=self.fourier_filters,
            volume_labels=self.volume_labels,
        )

    # ------------------------------------------------------------------
    # Index helpers (override parent to work on compact arrays)
    # ------------------------------------------------------------------

    def get_component(
        self,
        k_idx: int | np.ndarray,
        eig_idx: int | np.ndarray,
        return_u: bool = True,
        return_s: bool = True,
        return_vh: bool = True,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Retrieve specific SVD components from compact storage.

        Pairs not present in the stored top-L set are returned as zeros, matching the
        behavior of the dense result (those entries were truncated to zero when the
        sparse result was created).

        Parameters
        ----------
        k_idx : int | np.ndarray
            Actual angular-frequency index or indices.
        eig_idx : int | np.ndarray
            Eigenvalue index or indices.
        return_u, return_s, return_vh : bool
            Which arrays to return; others are ``None``.

        Returns
        -------
        tuple[np.ndarray | None, ...]
            Same shape contract as :meth:`DecompositionResult.get_component`.
        """
        if not (return_u or return_s or return_vh):
            raise ValueError(
                "At least one of return_u, return_s, return_vh must be True."
            )

        k_idx_arr = np.atleast_1d(k_idx)
        k_stored = np.atleast_1d(self.k_to_stored(k_idx_arr))
        eig_idx_arr = np.atleast_1d(eig_idx)
        conj_mask = np.atleast_1d(self.is_conjugate_mode(k_idx_arr))
        L_query = len(k_stored)

        compact_indices = np.array(
            [
                self._index_map.get((int(k), int(e)), -1)
                for k, e in zip(k_stored, eig_idx_arr, strict=False)
            ]
        )
        found = compact_indices >= 0

        u = None
        s = None
        vh = None

        if return_u:
            u = np.zeros(
                (
                    self.num_volumes,
                    self.num_fourier_filters,
                    self.num_orientations,
                    L_query,
                ),
                dtype=np.complex64,
            )
            if found.any():
                u[..., found] = _read_fancy_last_axis(
                    self._U_data, compact_indices[found]
                )
            if conj_mask.any():
                u[..., conj_mask] = u[..., conj_mask].conj()

        if return_s:
            s = np.zeros(L_query, dtype=np.float32)
            if found.any():
                s[found] = self._S_data[compact_indices[found]]

        if return_vh:
            vh = np.zeros((L_query, self.num_radial_components), dtype=np.complex64)
            if found.any():
                vh[found] = self._Vh_data[compact_indices[found]]
            if conj_mask.any():
                vh[conj_mask] = vh[conj_mask].conj()

        return u, s, vh

    def get_top_n(self, top_k: int, include_negative: bool = True) -> np.ndarray:
        """Get the top-n pairs from compact storage, ranked by ``|S|``.

        When ``top_k`` exceeds the number of stored components all stored
        pairs are returned without error.

        Parameters
        ----------
        top_k : int
            Number of pairs to return.
        include_negative : bool, optional
            For real-valued decompositions, whether to include negative-k
            mirror pairs. Default is True.

        Returns
        -------
        np.ndarray
            Shape ``(l, 2)`` array of ``(k_idx, eig_idx)`` pairs.
        """
        cache_key = (top_k, include_negative)
        if cache_key not in self._top_n_cache:
            sorted_l = np.argsort(np.abs(self._S_data))[::-1]
            k_stored_sorted = self._selected_indices[sorted_l, 0]
            eig_idx_sorted = self._selected_indices[sorted_l, 1]

            if not self.is_complex_projection and include_negative:
                tmp_k: list[int] = []
                tmp_e: list[int] = []
                for k_s, e in zip(k_stored_sorted, eig_idx_sorted, strict=False):
                    if int(k_s) == 0:
                        tmp_k.append(int(k_s))
                        tmp_e.append(int(e))
                    else:
                        tmp_k.extend([int(k_s), -int(k_s)])
                        tmp_e.extend([int(e), int(e)])
                    if len(tmp_k) >= top_k:
                        break
                k_indices = np.array(tmp_k)[:top_k]
                eig_indices = np.array(tmp_e)[:top_k]
            else:
                k_indices = k_stored_sorted[:top_k]
                eig_indices = eig_idx_sorted[:top_k]

            self._top_n_cache[cache_key] = np.stack((k_indices, eig_indices), axis=-1)

        return self._top_n_cache[cache_key]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the sparse result to an HDF5 file.

        The file is tagged ``is_sparse=True``. To load it back use
        :meth:`SparseDecompositionResult.load`.

        Parameters
        ----------
        path : str | Path
            Destination path.  Extension is normalised to ``.h5``.
        """
        path = Path(path)
        if path.suffix not in {".h5", ".hdf5"}:
            path = path.with_suffix(".h5")

        with h5py.File(path, "w") as f:
            f.attrs["is_sparse"] = True
            self._write_hdf5_metadata(f)
            f.create_dataset("selected_indices", data=self._selected_indices)
            f.create_dataset("U", data=self._U_data)
            f.create_dataset("S", data=self._S_data)
            f.create_dataset("Vh", data=self._Vh_data)
            self._write_hdf5_optional_arrays(f)
            self._write_hdf5_transform_arrays(f)

    @classmethod
    def load(cls, path: str | Path, mmap: bool = False) -> "SparseDecompositionResult":
        """Load a sparse result from an HDF5 file written by :meth:`save`.

        Parameters
        ----------
        path : str | Path
            Path to the ``.h5`` or ``.hdf5`` file.
        mmap : bool, optional
            If True, keep the HDF5 file open on disk to read ``U`` lazily. Other data
            and metadata loaded eagerly. Default is False.

        Returns
        -------
        SparseDecompositionResult
            The loaded sparse result.

        Raises
        ------
        ValueError
            If the file extension is wrong, or the file does not contain a
            sparse result (use :meth:`DecompositionResult.load` instead).
        """
        path = Path(path)
        if path.suffix not in {".h5", ".hdf5"}:
            raise ValueError(
                f"Unknown file extension '{path.suffix}'. Expected '.h5' or '.hdf5'."
            )

        f = h5py.File(path, "r")
        try:
            if not bool(f.attrs.get("is_sparse", False)):
                raise ValueError(
                    f"File '{path}' contains a dense result. "
                    "Use DecompositionResult.load() instead."
                )

            k_max = int(f.attrs["k_max"])
            eig_max = int(f.attrs["eig_max"])
            is_complex = bool(f.attrs["is_complex_projection"])
            # .get() with a default of 1 for backward compatibility with files
            # saved before the multi-volume axis was introduced.
            num_vol = int(f.attrs.get("num_volumes", 1))
            num_ff = int(f.attrs["num_fourier_filters"])
            num_or = int(f.attrs["num_orientations"])
            num_ang = int(f.attrs["num_angular_components"])
            num_r = int(f.attrs["num_radial_components"])

            created_at = f.attrs["created_at"]
            if isinstance(created_at, bytes):
                created_at = created_at.decode("utf-8")

            phi_values = f["phi_values"][()] if "phi_values" in f else None
            theta_values = f["theta_values"][()] if "theta_values" in f else None
            fourier_filters = (
                f["fourier_filters"][()] if "fourier_filters" in f else None
            )
            volume_labels = f["volume_labels"][()] if "volume_labels" in f else None

            if "transform_config" not in f.attrs:
                raise KeyError(
                    f"The file '{path}' has no 'transform_config' attribute."
                )

            transform_json = f.attrs["transform_config"]
            if isinstance(transform_json, bytes):
                transform_json = transform_json.decode("utf-8")
            transform_params = json.loads(transform_json)

            if transform_params.get("transform_name") == "grid":
                transform_coords = f["transform/transform_coords"][()]
                cartesian_coords = f["transform/cartesian_coords"][()]
                jacobian = (
                    f["transform/jacobian"][()] if "transform/jacobian" in f else None
                )
                polar_shape = (
                    int(transform_params["polar_shape"][0]),
                    int(transform_params["polar_shape"][1]),
                )
                cartesian_shape = (
                    int(transform_params["cartesian_shape"][0]),
                    int(transform_params["cartesian_shape"][1]),
                )
                coordinate_transform: CoordinateTransform = GridTransform.from_arrays(
                    transform_coords=transform_coords,
                    cartesian_coords=cartesian_coords,
                    jacobian=jacobian,
                    polar_shape=polar_shape,
                    cartesian_shape=cartesian_shape,
                    source_params=transform_params.get("source_params"),
                    has_periodic_axis=bool(
                        transform_params.get("has_periodic_axis", True)
                    ),
                    periodic_axis=int(transform_params.get("periodic_axis", 0)),
                )
            else:
                coordinate_transform = reconstruct_transform(transform_params)

            selected_indices = f["selected_indices"][()]
            U_data: np.ndarray | h5py.Dataset = f["U"] if mmap else f["U"][()]
            S_data = f["S"][()]
            Vh_data = f["Vh"][()]

            # Legacy files (pre multi-volume support) stored U_data without the
            # leading volume axis; insert it so num_volumes=1 is self-consistent.
            if U_data.ndim == 3:
                U_data = U_data[np.newaxis, ...]

            result = cls.from_sparse_arrays(
                U_data=U_data,
                S_data=S_data,
                Vh_data=Vh_data,
                selected_indices=selected_indices,
                k_max=k_max,
                eig_max=eig_max,
                is_complex_projection=is_complex,
                num_fourier_filters=num_ff,
                num_orientations=num_or,
                num_angular_components=num_ang,
                num_radial_components=num_r,
                num_volumes=num_vol,
                coordinate_transform=coordinate_transform,
                created_at=created_at,
                phi_values=phi_values,
                theta_values=theta_values,
                fourier_filters=fourier_filters,
                volume_labels=volume_labels,
                _h5_file=f if mmap else None,
            )
        except Exception as e:
            f.close()
            raise e

        if not mmap:
            f.close()
        return result

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a string representation showing sparsity info."""
        num_freq_blocks = self.k_max * 2 if self.is_complex_projection else self.k_max
        total = num_freq_blocks * self.eig_max
        sparsity = f"{self.num_stored} / {total} ({100 * self.num_stored / total:.1f}%)"
        transform_name = (
            getattr(self.coordinate_transform, "transform_name", None)
            if self.coordinate_transform is not None
            else None
        )
        s = textwrap.dedent(f"""
            SparseDecompositionResult(
                k_max={self.k_max},
                eig_max={self.eig_max},
                is_complex_projection={self.is_complex_projection},
                num_volumes={self.num_volumes},
                num_fourier_filters={self.num_fourier_filters},
                num_orientations={self.num_orientations},
                num_angular_components={self.num_angular_components},
                num_radial_components={self.num_radial_components},
                num_stored={sparsity},
                has_euler_angles={self.phi_values is not None},
                has_fourier_filters={self.fourier_filters is not None},
                coordinate_transform='{transform_name}',
                created_at='{self.created_at}'
            )
            """)
        return s.strip()
