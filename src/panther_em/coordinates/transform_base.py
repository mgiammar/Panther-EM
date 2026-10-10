"""Abstract base class, registry, and grid-based transform for coordinate transforms.

Registering a custom transform
-------------------------------
Decorate your subclass with :func:`register_transform` and set the class
attribute ``transform_name`` to a unique string key::

    from panther_em.coordinates.transform_base import (
        CoordinateTransform,
        register_transform,
    )


    @register_transform
    class MyPolarTransform(CoordinateTransform):
        transform_name = "my_polar"
        ...

The transform will then be available via :func:`reconstruct_transform` when
loading a :class:`~panther_em.decomposition.result.DecompositionResult` that
was saved with this transform embedded.

NOTE: Built-in transforms (``"offset_polar"``, ``"nonuniform_polar"``) are
registered automatically when the corresponding modules are imported. Custom
transforms must be imported by the user *before* loading a result that
references them.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from panther_em.utils.warp_backends import (
    detect_device,
    ensure_device,
    get_warp_function,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[CoordinateTransform]] = {}
_INSTANCE_CACHE: dict[tuple, CoordinateTransform] = {}


def _make_cache_key(cls: type[CoordinateTransform], **kwargs: Any) -> tuple:
    """Build a hashable cache key from a class and its constructor kwargs."""

    def _freeze(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return tuple(v.tolist())
        if isinstance(v, (list, tuple)):
            return tuple(_freeze(x) for x in v)
        return v

    return (cls.transform_name, *sorted((k, _freeze(v)) for k, v in kwargs.items()))


def register_transform(
    cls: type[CoordinateTransform] | None = None,
    *,
    name: str | None = None,
) -> type[CoordinateTransform]:
    """Class decorator to register a :class:`CoordinateTransform` subclass.

    Parameters
    ----------
    cls : type[CoordinateTransform] | None
        The class being decorated. When calling ``@register_transform``
        without arguments this is supplied automatically by Python's decorator
        machinery; when calling ``@register_transform(name="foo")`` it is None
        and the inner decorator is returned instead.
    name : str | None, optional
        Override the registry key. Defaults to ``cls.transform_name``.

    Returns
    -------
    type[CoordinateTransform]
        The unmodified class (decorator is transparent).
    """

    def decorator(klass: type[CoordinateTransform]) -> type[CoordinateTransform]:
        key = name if name is not None else klass.transform_name
        if key in _REGISTRY and _REGISTRY[key] is not klass:
            warnings.warn(
                f"A transform named '{key}' is already registered "
                f"({_REGISTRY[key].__qualname__}). Overwriting with "
                f"{klass.__qualname__}.",
                UserWarning,
                stacklevel=2,
            )
        _REGISTRY[key] = klass
        return klass

    if cls is not None:
        return decorator(cls)
    return decorator  # type: ignore[return-value]


def get_transform_class(name: str) -> type[CoordinateTransform]:
    """Look up a registered transform class by name.

    Parameters
    ----------
    name : str
        Registry key (``transform_name`` of the desired class).

    Returns
    -------
    type[CoordinateTransform]
        The registered class.

    Raises
    ------
    KeyError
        If no transform with *name* has been registered.
    """
    if name not in _REGISTRY:
        known = ", ".join(f"'{k}'" for k in sorted(_REGISTRY))
        raise KeyError(
            f"No coordinate transform named '{name}' found in registry. "
            f"Known transforms: {known or '(none)'}. "
            "Make sure the module that registers your custom transform has "
            "been imported before loading the result."
        )
    return _REGISTRY[name]


def get_transform(cls: type[CoordinateTransform], **kwargs: Any) -> CoordinateTransform:
    """Return a cached transform instance, creating one on a cache miss.

    Parameters
    ----------
    cls : type[CoordinateTransform]
        The concrete transform class to instantiate.
    **kwargs
        Constructor arguments (geometric parameters only — no ``device``).

    Returns
    -------
    CoordinateTransform
        A cached or newly created instance.
    """
    key = _make_cache_key(cls, **kwargs)
    if key not in _INSTANCE_CACHE:
        _INSTANCE_CACHE[key] = cls(**kwargs)
    return _INSTANCE_CACHE[key]


def reconstruct_transform(
    params: dict[str, Any],
    device: str | Any = "numpy",
) -> CoordinateTransform:
    """Reconstruct a :class:`CoordinateTransform` from its serialized parameters.

    Parameters
    ----------
    params : dict[str, Any]
        Dictionary produced by :meth:`CoordinateTransform.to_dict`. Must
        contain a ``"transform_name"`` key.
    device : str | torch.device, optional
        Passed to :meth:`CoordinateTransform.from_dict` for compatibility; all
        built-in transforms are device-agnostic so this argument is ignored.

    Returns
    -------
    CoordinateTransform
        A fully initialised transform instance.

    Raises
    ------
    NotImplementedError
        If ``params["transform_name"]`` is ``"grid"`` — :class:`GridTransform`
        must be reconstructed from array data via
        :meth:`GridTransform.from_arrays` or :func:`DecompositionResult.load`.
    """
    if "transform_name" not in params:
        raise KeyError(
            "Serialized transform parameters are missing 'transform_name'. "
            "The dictionary must have been produced by CoordinateTransform.to_dict()."
        )
    cls = get_transform_class(params["transform_name"])
    return cls.from_dict(params, device=device)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class CoordinateTransform(ABC):
    """Abstract base class for all coordinate transforms in Panther-EM.

    Subclasses must implement :meth:`compute_transform_coords`,
    :meth:`compute_cartesian_coords`, the ``polar_shape`` and
    ``cartesian_shape`` properties, and the serialization pair
    :meth:`to_dict` / :meth:`from_dict`.

    The base class provides:

    * Lazy caching of coordinate grids (computed once per instance).
    * Per-device caching of GPU tensor copies (populated on first use).
    * Concrete :meth:`to_transform_space` and :meth:`to_cartesian` methods
      that dispatch to the appropriate warp backend.
    * :meth:`clear_cache` to release all cached arrays.

    Transforms are *device-agnostic*: coordinate grids are always stored as
    NumPy arrays; device-specific copies are produced on demand and cached.
    """

    # --- Unique registry key (set as class attribute in subclasses) ---
    transform_name: str

    # --- Optional capability flags ---
    supports_energy_preservation: bool = False
    has_periodic_axis: bool = False
    periodic_axis: int | None = None

    def __init__(self) -> None:
        """Initialise the base-class coordinate cache."""
        # Numpy arrays storing the pre-computed coordinate grids.
        self._transform_coords: np.ndarray | None = None
        self._cartesian_coords: np.ndarray | None = None
        self._jacobian: np.ndarray | None = None

        # Caches device-specific (e.g. CUDA) copies of the grids.
        self._device_cache: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Required shape properties (abstract)
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def polar_shape(self) -> tuple[int, int]:
        """Shape of the transformed image: ``(num_angle, num_radius)``."""

    @property
    @abstractmethod
    def cartesian_shape(self) -> tuple[int, int]:
        """Shape of the Cartesian image: ``(height, width)``."""

    # ------------------------------------------------------------------ #
    # Required serialization (abstract)
    # ------------------------------------------------------------------ #

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serializes geometric parameters to a JSON dict.

        Note
        ----
        Must contain a ``"transform_name"`` key. Device information is NOT
        included.
        """

    @classmethod
    @abstractmethod
    def from_dict(
        cls,
        params: dict[str, Any],
        device: str | torch.device = "numpy",
    ) -> CoordinateTransform:
        """Reconstruct an instance from serialized parameters.

        Parameters
        ----------
        params : dict[str, Any]
            Dictionary produced by :meth:`to_dict`.
        device : str | torch.device, optional
            Accepted for API compatibility; all built-in transforms are
            device-agnostic so this argument is ignored.
        """

    # ------------------------------------------------------------------ #
    # Overridable coord-computation hooks (raise by default)
    # ------------------------------------------------------------------ #

    def compute_transform_coords(self) -> np.ndarray:
        """Compute the coordinate grid for the Cartesian -> polar warp.

        Returns
        -------
        np.ndarray
            Shape ``(2, num_angle, num_radius)``, dtype float64.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement compute_transform_coords()."
        )

    def compute_cartesian_coords(self) -> np.ndarray:
        """Compute the coordinate grid for the polar -> Cartesian warp.

        Returns
        -------
        np.ndarray
            Shape ``(2, num_row, num_col)`` matching the warp backend's convention.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement compute_cartesian_coords()."
        )

    def compute_jacobian(self) -> np.ndarray:
        """Compute the Jacobian correction array.

        Returns
        -------
        np.ndarray
            Shape ``(num_angle, num_radius)``, dtype float32. Each element is
            ``sqrt(Cartesian area)`` of the corresponding polar cell.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement energy-preserving transforms "
            "(supports_energy_preservation=False)."
        )

    # ------------------------------------------------------------------ #
    # Lazy cached properties
    # ------------------------------------------------------------------ #

    @property
    def transform_coords(self) -> np.ndarray:
        """Cached coord grid for cartesian -> polar, ``(2, num_angle, num_radius)``."""
        if self._transform_coords is None:
            self._transform_coords = self.compute_transform_coords()
        return self._transform_coords

    @property
    def cartesian_coords(self) -> np.ndarray:
        """Cached coord grid for polar -> cartesian, ``(2, num_row, num_col)``."""
        if self._cartesian_coords is None:
            self._cartesian_coords = self.compute_cartesian_coords()
        return self._cartesian_coords

    @property
    def jacobian_grid(self) -> np.ndarray | None:
        """Cached Jacobian array ``(num_angle, num_radius)``, or ``None``."""
        if self._jacobian is None:
            if not self.supports_energy_preservation:
                return None
            self._jacobian = self.compute_jacobian()
        return self._jacobian

    # ------------------------------------------------------------------ #
    # Device-dispatch coord cache
    # ------------------------------------------------------------------ #

    def _get_device_coords(self, device: str | torch.device) -> dict[str, Any]:
        """Return device-appropriate coord arrays, caching CUDA copies.

        Parameters
        ----------
        device : str | torch.device
            ``"numpy"`` for CPU or a ``torch.device("cuda:N")`` for GPU.

        Returns
        -------
        dict with keys ``"transform"``, ``"cartesian"``, ``"jacobian"``
            Values are NumPy arrays (for ``"numpy"``) or CUDA tensors.
            ``"jacobian"`` is ``None`` when the transform does not support
            energy preservation.
        """
        key = str(device)
        if key not in self._device_cache:
            jac = self.jacobian_grid
            self._device_cache[key] = {
                "transform": ensure_device(self.transform_coords, device),
                "cartesian": ensure_device(self.cartesian_coords, device),
                "jacobian": ensure_device(jac, device) if jac is not None else None,
            }
        return self._device_cache[key]

    # ------------------------------------------------------------------ #
    # Cache management
    # ------------------------------------------------------------------ #

    def clear_cache(self) -> None:
        """Release all cached coordinate grids and device copies."""
        self._transform_coords = None
        self._cartesian_coords = None
        self._jacobian = None
        self._device_cache.clear()

    # ------------------------------------------------------------------ #
    # Radial boundary hooks for the polar -> Cartesian warp
    # ------------------------------------------------------------------ #

    def _pad_radial_axis(
        self, image: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray | torch.Tensor, int]:
        """Extend a polar image along its radial axis before the inverse warp."""
        return image, 0

    def _outer_radius_index(self) -> float | None:
        """Largest radial coordinate inside the transform's support."""
        return None

    # ------------------------------------------------------------------ #
    # Concrete warp methods
    # ------------------------------------------------------------------ #

    def to_transform_space(
        self,
        image: np.ndarray | torch.Tensor,
        preserve_energy: bool = True,
        order: int = 5,
        mode: str = "constant",
        cval: float = 0.0,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Warp a Cartesian image to this transform's coordinate space.

        Note
        ----
        Function signature is designed to match ``skimage.transform.warp`` signature.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input 2-D image ``(H, W)`` or batched 3-D array ``(B, H, W)``.
        preserve_energy : bool, optional
            Scale the output by the Jacobian so total energy is preserved.
            Silently skipped when the transform has no Jacobian. Default True.
        order : int, optional
            Spline interpolation order. Default 5.
        mode : str, optional
            Boundary fill mode. Default ``"constant"``.
        cval : float, optional
            Fill value for ``mode="constant"``. Default 0.0.
        **kwargs
            Forwarded to the warp backend.

        Returns
        -------
        np.ndarray or torch.Tensor
            Warped image with shape :attr:`polar_shape` (or batched equivalent).
        """
        device = detect_device(image)

        # --- batched (3-D) routing ---
        if image.ndim == 3:
            results = [
                self.to_transform_space(
                    im,
                    preserve_energy=preserve_energy,
                    order=order,
                    mode=mode,
                    cval=cval,
                    **kwargs,
                )
                for im in image
            ]
            return np.stack(results) if device == "numpy" else torch.stack(results)

        if image.ndim != 2:
            raise ValueError(f"Image must be 2-D or 3-D (batched), got {image.ndim}D.")

        # --- complex routing (independent real/imag channels) ---
        is_complex = (
            np.iscomplexobj(image)
            if device == "numpy"
            else (isinstance(image, torch.Tensor) and image.is_complex())
        )
        if is_complex:
            real_part = self.to_transform_space(
                image.real,
                preserve_energy=preserve_energy,
                order=order,
                mode=mode,
                cval=cval,
                **kwargs,
            )
            imag_part = self.to_transform_space(
                image.imag,
                preserve_energy=preserve_energy,
                order=order,
                mode=mode,
                cval=cval,
                **kwargs,
            )
            if device == "numpy":
                return real_part + 1j * imag_part
            return torch.complex(real_part, imag_part)

        # --- base case: single real 2-D image ---
        dc = self._get_device_coords(device)
        warp_fn = get_warp_function(device)
        warped = warp_fn(
            image,
            dc["transform"],
            output_shape=self.polar_shape,
            order=order,
            mode=mode,
            cval=cval,
            **kwargs,
        )

        if preserve_energy and dc["jacobian"] is not None:
            warped = warped * dc["jacobian"]

        return warped

    def to_cartesian(
        self,
        image: np.ndarray | torch.Tensor,
        preserve_energy: bool = True,
        wrap_angular_axis: bool = True,
        pad_radial_axis: bool = True,
        order: int = 5,
        mode: str = "constant",
        cval: float = 0.0,
        **kwargs: Any,
    ) -> np.ndarray | torch.Tensor:
        """Warp an image from this transform's space back to Cartesian.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input 2-D image ``(num_angle, num_radius)`` or batched 3-D array.
        preserve_energy : bool, optional
            Remove the Jacobian scaling applied during the forward transform.
            Default True.
        wrap_angular_axis : bool, optional
            Apply circular padding at the 0°/360° boundary before warping.
            Only applied when :attr:`has_periodic_axis` is ``True``. Default True.
        pad_radial_axis : bool, optional
            Continue the radial axis past its first and last samples before warping and
            zero the pixels outside the support afterwards.
        order : int, optional
            Spline interpolation order. Default 5.
        mode : str, optional
            Boundary fill mode. Default ``"constant"``.
        cval : float, optional
            Fill value for ``mode="constant"``. Default 0.0.
        **kwargs
            Forwarded to the warp backend.

        Returns
        -------
        np.ndarray or torch.Tensor
            Warped image with shape :attr:`cartesian_shape` (or batched).
        """
        device = detect_device(image)

        # --- batched (3-D) routing ---
        if image.ndim == 3:
            results = [
                self.to_cartesian(
                    im,
                    preserve_energy=preserve_energy,
                    wrap_angular_axis=wrap_angular_axis,
                    pad_radial_axis=pad_radial_axis,
                    order=order,
                    mode=mode,
                    cval=cval,
                    **kwargs,
                )
                for im in image
            ]
            return np.stack(results) if device == "numpy" else torch.stack(results)

        if image.ndim != 2:
            raise ValueError(f"Image must be 2-D or 3-D (batched), got {image.ndim}D.")

        # --- complex routing ---
        is_complex = (
            np.iscomplexobj(image)
            if device == "numpy"
            else (isinstance(image, torch.Tensor) and image.is_complex())
        )
        if is_complex:
            real_part = self.to_cartesian(
                image.real,
                preserve_energy=preserve_energy,
                wrap_angular_axis=wrap_angular_axis,
                pad_radial_axis=pad_radial_axis,
                order=order,
                mode=mode,
                cval=cval,
                **kwargs,
            )
            imag_part = self.to_cartesian(
                image.imag,
                preserve_energy=preserve_energy,
                wrap_angular_axis=wrap_angular_axis,
                pad_radial_axis=pad_radial_axis,
                order=order,
                mode=mode,
                cval=cval,
                **kwargs,
            )
            if device == "numpy":
                return real_part + 1j * imag_part
            return torch.complex(real_part, imag_part)

        # --- base case: single real 2-D image ---
        dc = self._get_device_coords(device)

        if preserve_energy and dc["jacobian"] is not None:
            image = image / dc["jacobian"]

        coords = dc["cartesian"]

        # Continue the radial axis though origin for spline interpolation
        if pad_radial_axis:
            image, num_prepended = self._pad_radial_axis(image)
            if num_prepended:
                coords = coords.copy() if device == "numpy" else coords.clone()
                coords[1, ...] += num_prepended

        # Handle periodic angular axis to prevent boundary artifacts by wrap/circular
        # padding around the boundary before interpolation
        if wrap_angular_axis and self.has_periodic_axis:
            if self.periodic_axis not in (None, 0):
                raise NotImplementedError(
                    "Periodic padding is only implemented for periodic_axis=0."
                )
            # The spline prefilter's boundary error decays like |pole|**rows (~0.43
            # for order 5), so `order` rows left ~4e-3 at the 0/360 seam
            pad_size = 4 * order
            if device == "numpy":
                image = np.pad(image, ((pad_size, pad_size), (0, 0)), mode="wrap")
            else:
                image = image.unsqueeze(0)
                image = F.pad(image, (0, 0, pad_size, pad_size), mode="circular")
                image = image.squeeze(0)
            # Offset the angular coordinate to account for the padding rows
            coords = coords.copy() if device == "numpy" else coords.clone()
            coords[0, ...] += pad_size

        warp_fn = get_warp_function(device)
        warped = warp_fn(
            image,
            coords,
            output_shape=self.cartesian_shape,
            order=order,
            mode=mode,
            cval=cval,
            **kwargs,
        )

        # Zero the pixels outside the support, which padded rows could otherwise reach
        outer_index = self._outer_radius_index() if pad_radial_axis else None
        if outer_index is not None:
            warped = warped * (dc["cartesian"][1] <= outer_index)
        return warped


# ---------------------------------------------------------------------------
# GridTransform — concrete grid-holding transform
# ---------------------------------------------------------------------------


@register_transform
class GridTransform(CoordinateTransform):
    """A coordinate transform defined entirely by pre-computed coordinate grids.

    Rather than storing geometric parameters and computing grids on demand,
    :class:`GridTransform` holds the grids directly. It is the canonical
    form stored inside a :class:`~panther_em.decomposition.result.DecompositionResult`
    so that loading a result never requires recomputing any coordinate mappings or
    depends on a particular transform class's implementation.

    Registered under the key ``"grid"``.

    Construction
    ------------
    * :meth:`from_transform` — eagerly materialize grids from a parameterized
      transform (e.g.
      :class:`~panther_em.coordinates.offset_polar.OffsetPolarTransform`).
    * :meth:`from_arrays` — construct directly from NumPy arrays (e.g. when
      loading from HDF5).

    Serialization
    -------------
    :meth:`to_dict` produces a compact JSON dict with shape metadata and
    provenance (the original transform's parameter dict). The grid arrays
    themselves are written as separate HDF5 datasets by
    :meth:`~panther_em.decomposition.result.DecompositionResult.save`.

    :meth:`from_dict` intentionally raises :class:`NotImplementedError`; use
    :meth:`from_arrays` with arrays loaded from the HDF5 file instead.
    """

    transform_name: str = "grid"

    def __init__(
        self,
        transform_coords: np.ndarray,
        cartesian_coords: np.ndarray,
        jacobian: np.ndarray | None,
        polar_shape: tuple[int, int],
        cartesian_shape: tuple[int, int],
        source_params: dict[str, Any] | None = None,
        has_periodic_axis: bool = True,
        periodic_axis: int = 0,
    ) -> None:
        super().__init__()
        # Pre-fill the base-class caches directly (no lazy computation needed)
        self._transform_coords = transform_coords
        self._cartesian_coords = cartesian_coords
        self._jacobian = jacobian
        self._polar_shape = polar_shape
        self._cartesian_shape = cartesian_shape
        self.source_params = source_params
        self._source_transform: CoordinateTransform | None = None
        self._source_resolved = False

        # Instance-level overrides of class-level flags
        self.has_periodic_axis = has_periodic_axis
        self.periodic_axis = periodic_axis
        self.supports_energy_preservation = jacobian is not None

    @property
    def source_transform(self) -> CoordinateTransform | None:
        """The parameterized transform these grids were built from, or ``None``.

        Raises
        ------
        ValueError
            If the rebuilt transform's shapes do not match the stored grids.
        """
        if not self._source_resolved:
            self._source_resolved = True
            name = (self.source_params or {}).get("transform_name")
            if name is not None and name != self.transform_name and name in _REGISTRY:
                source = reconstruct_transform(self.source_params)  # type: ignore[arg-type]
                if (source.polar_shape, source.cartesian_shape) != (
                    self.polar_shape,
                    self.cartesian_shape,
                ):
                    raise ValueError(
                        f"source_params describe a {name} transform with polar shape "
                        f"{source.polar_shape} and Cartesian shape "
                        f"{source.cartesian_shape}, but the grids have "
                        f"{self.polar_shape} and {self.cartesian_shape}."
                    )
                self._source_transform = source
        return self._source_transform

    def _pad_radial_axis(
        self, image: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray | torch.Tensor, int]:
        """Delegate to :attr:`source_transform` (no padding without one)."""
        source = self.source_transform
        if source is None:
            return image, 0
        return source._pad_radial_axis(image)

    def _outer_radius_index(self) -> float | None:
        """Delegate to :attr:`source_transform` (no masking without one)."""
        source = self.source_transform
        return None if source is None else source._outer_radius_index()

    @classmethod
    def from_transform(cls, transform: CoordinateTransform) -> GridTransform:
        """Eagerly materialize all coord grids from a parameterized transform.

        Accesses :attr:`~CoordinateTransform.transform_coords`,
        :attr:`~CoordinateTransform.cartesian_coords`, and
        :attr:`~CoordinateTransform.jacobian_grid` — triggering computation
        if the grids have not yet been built.

        Parameters
        ----------
        transform : CoordinateTransform
            Source transform (e.g. an
            :class:`~panther_em.coordinates.offset_polar.OffsetPolarTransform`).

        Returns
        -------
        GridTransform
        """
        return cls(
            transform_coords=transform.transform_coords,
            cartesian_coords=transform.cartesian_coords,
            jacobian=transform.jacobian_grid,
            polar_shape=transform.polar_shape,
            cartesian_shape=transform.cartesian_shape,
            source_params=transform.to_dict(),
            has_periodic_axis=transform.has_periodic_axis,
            periodic_axis=(
                transform.periodic_axis if transform.periodic_axis is not None else 0
            ),
        )

    @classmethod
    def from_arrays(
        cls,
        transform_coords: np.ndarray,
        cartesian_coords: np.ndarray,
        jacobian: np.ndarray | None,
        polar_shape: tuple[int, int],
        cartesian_shape: tuple[int, int],
        source_params: dict[str, Any] | None = None,
        has_periodic_axis: bool = True,
        periodic_axis: int = 0,
    ) -> GridTransform:
        """Construct a :class:`GridTransform` directly from pre-computed arrays.

        Parameters
        ----------
        transform_coords : np.ndarray
            Coordinate grid for Cartesian -> polar warp.
        cartesian_coords : np.ndarray
            Coordinate grid for polar -> Cartesian warp.
        jacobian : np.ndarray or None
            Jacobian correction array ``(num_angle, num_radius)``, or ``None``.
        polar_shape : tuple[int, int]
            ``(num_angle, num_radius)`` shape of the polar image.
        cartesian_shape : tuple[int, int]
            ``(height, width)`` shape of the Cartesian image.
        source_params : dict or None, optional
            Provenance dict from the original parameterized transform.
        has_periodic_axis : bool, optional
            Whether the angular axis is periodic. Default True.
        periodic_axis : int, optional
            Index of the periodic axis. Default 0.

        Returns
        -------
        GridTransform
        """
        return cls(
            transform_coords=transform_coords,
            cartesian_coords=cartesian_coords,
            jacobian=jacobian,
            polar_shape=polar_shape,
            cartesian_shape=cartesian_shape,
            source_params=source_params,
            has_periodic_axis=has_periodic_axis,
            periodic_axis=periodic_axis,
        )

    @property
    def polar_shape(self) -> tuple[int, int]:
        """Shape of image in transform (polar) space: ``(num_angle, num_radius)``."""
        return self._polar_shape

    @property
    def cartesian_shape(self) -> tuple[int, int]:
        """Shape of image in Cartesian space: ``(height, width)``."""
        return self._cartesian_shape

    def to_dict(self) -> dict[str, Any]:
        """Serialize shape metadata and provenance to a JSON dict.

        The coordinate arrays themselves are NOT included; they are written as HDF5
        datasets by :meth:`~panther_em.decomposition.result.DecompositionResult.save`.
        """
        return {
            "transform_name": self.transform_name,
            "polar_shape": list(self._polar_shape),
            "cartesian_shape": list(self._cartesian_shape),
            "has_jacobian": self._jacobian is not None,
            "has_periodic_axis": self.has_periodic_axis,
            "periodic_axis": self.periodic_axis,
            "source_params": self.source_params,
        }

    @classmethod
    def from_dict(
        cls,
        params: dict[str, Any],
        device: str | torch.device = "numpy",
    ) -> GridTransform:
        """Implementation to intentionally raise an error for fixed grid transforms."""
        raise NotImplementedError(
            "GridTransform cannot be reconstructed from scalar parameters alone. "
            "Load a DecompositionResult via DecompositionResult.load() to obtain "
            "a GridTransform with its coordinate arrays, or use "
            "GridTransform.from_arrays() directly."
        )
