"""Standard-polar coordinate transform.

Provides :class:`StandardPolarTransform`, a concrete
:class:`~panther_em.coordinates.transform_base.CoordinateTransform` that maps
between Cartesian and standard polar coordinates. Features:

- Per-instance lazy caching of coordinate grids (inherited from the base
  class); grids are computed once and reused across all calls.
- Per-device caching of GPU tensor copies (populated on first use).
- Optional energy-preserving transforms via Jacobian correction.
- Circular padding at the 0°/360° boundary in the inverse transform.

The *standard polar* coordinate system maps each Cartesian pixel to a
``(radius, angle)`` pair with uniform angular spacing at every radius, i.e.
the ordinary polar coordinate system (no per-ring angular offset).
"""

from typing import Any

import numpy as np

from .transform_base import CoordinateTransform, get_transform, register_transform

# ---------------------------------------------------------------------------
# Coordinate mapping functions
# ---------------------------------------------------------------------------


def forward_cartesian_to_polar_mapping(
    input_coords: np.ndarray,  # (M, 2) - (row, col)
    num_angle: int,
    num_radius: int,
    max_radius: float,
    center: tuple[float, float],
) -> np.ndarray:
    """The forward mapping function to take cartesian coords to polar coords.

    Parameters
    ----------
    input_coords : np.ndarray
        (M, 2) array of (row, col) coordinates in cartesian space.
    num_angle : int
        Number of angular samples in polar space.
    num_radius : int
        Number of radial samples in polar space.
    max_radius : float
        Maximum radius for the polar coordinate system.
    center : tuple[float, float]
        Center point (row, col) of the polar coordinate system.

    Returns
    -------
    polar_coords : np.ndarray
        (M, 2) array of (radius_idx, angle_idx) in polar space.
    """
    row = input_coords[:, 0] - center[0]
    col = input_coords[:, 1] - center[1]

    radius = np.sqrt(row**2 + col**2)
    angle = np.arctan2(row, col)
    angle = angle % (2 * np.pi)

    radius_idx = (radius / max_radius) * num_radius
    angle_idx = (angle / (2 * np.pi)) * num_angle

    return np.column_stack((radius_idx, angle_idx))


def forward_polar_to_cartesian_mapping(
    input_coords: np.ndarray,  # (M, 2) - (radius_idx, angle_idx)
    num_angle: int,
    num_radius: int,
    max_radius: float,
    center: tuple[float, float],
) -> np.ndarray:
    """Forward mapping: Polar coordinates → Cartesian coordinates.

    Parameters
    ----------
    input_coords : np.ndarray
        (M, 2) array of (radius_idx, angle_idx) in polar space.
    num_angle : int
        Number of angular samples in polar space.
    num_radius : int
        Number of radial samples in polar space.
    max_radius : float
        Maximum radius for the polar coordinate system.
    center : tuple[float, float]
        Center point (row, col) of the polar coordinate system.

    Returns
    -------
    cartesian_coords : np.ndarray
        (M, 2) array of (row, col) coordinates in cartesian space.
    """
    radius_idx = input_coords[:, 0]
    angle_idx = input_coords[:, 1]

    angle = (angle_idx / num_angle) * (2 * np.pi)
    radius = (radius_idx / num_radius) * max_radius

    row = radius * np.sin(angle) + center[0]
    col = radius * np.cos(angle) + center[1]

    return np.column_stack((row, col))


def inverse_polar_to_cartesian_mapping(
    output_coords: np.ndarray,
    num_angle: int,
    num_radius: int,
    max_radius: float,
    center: tuple[float, float],
) -> np.ndarray:
    """Inverse mapping for warping cartesian image to polar space.

    Given output coordinates in polar space, returns the corresponding
    input coordinates in cartesian space. Used with skimage.transform.warp.

    Parameters
    ----------
    output_coords : np.ndarray
        Array of (col, row) coordinates in the output (polar) image.
        col corresponds to radius, row corresponds to angle.
    num_angle : int
        Number of angular samples in polar space.
    num_radius : int
        Number of radial samples in polar space.
    max_radius : float
        Maximum radius for the polar coordinate system.
    center : tuple[float, float]
        Center point (row, col) of the polar coordinate system in the input cartesian
        image.

    Returns
    -------
    coords : np.ndarray
        Array of (col, row) coordinates in the input (cartesian) image.
    """
    radius_idx = output_coords[:, 0]
    angle_idx = output_coords[:, 1]

    angle = (angle_idx / num_angle) * (2 * np.pi)
    radius = (radius_idx / num_radius) * max_radius

    row = radius * np.sin(angle) + center[0]
    col = radius * np.cos(angle) + center[1]

    return np.column_stack((col, row))


def inverse_cartesian_to_polar_mapping(
    output_coords: np.ndarray,
    num_angle: int,
    num_radius: int,
    max_radius: float,
    center: tuple[float, float],
) -> np.ndarray:
    """Inverse mapping for warping polar image back to cartesian space.

    Given output coordinates in cartesian space, returns the corresponding
    input coordinates in polar space. Used with skimage.transform.warp.

    Parameters
    ----------
    output_coords : np.ndarray
        Array of (col, row) coordinates in the output (cartesian) image.
    num_angle : int
        Number of angular samples in polar space.
    num_radius : int
        Number of radial samples in polar space.
    max_radius : float
        Maximum radius for the polar coordinate system.
    center : tuple[float, float]
        Center point (row, col) of the polar coordinate system.

    Returns
    -------
    coords : np.ndarray
        Array of (col, row) coordinates in the input (polar) image.
        col corresponds to radius, row corresponds to angle.
    """
    col = output_coords[:, 0]
    row = output_coords[:, 1]

    dc = col - center[1]
    dr = row - center[0]

    radius = np.sqrt(dr**2 + dc**2)
    angle = np.arctan2(dr, dc)

    angle = angle % (2 * np.pi)

    radius_idx = (radius / max_radius) * num_radius
    angle_idx = (angle / (2 * np.pi)) * num_angle

    return np.column_stack((radius_idx, angle_idx))


def jacobian_correction_polar(
    num_angle: int,
    num_radius: int,
    max_radius: float,
) -> np.ndarray:
    """Correction factor for area element in polar coordinates.

    Calculates the exact Cartesian area covered by each polar grid cell.
    (Area of annular sector: 0.5 * (r_outer^2 - r_inner^2) * dtheta)

    Parameters
    ----------
    num_angle : int
        Number of angular samples in polar space.
    num_radius : int
        Number of radial samples in polar space.
    max_radius : float
        Maximum radius for the polar coordinate system.

    Returns
    -------
    area_elements : np.ndarray
        (num_radius,) array containing the Cartesian area of each polar pixel.
    """
    dr = max_radius / num_radius
    dtheta = (2 * np.pi) / num_angle

    r_inner = np.arange(num_radius) * dr
    r_outer = r_inner + dr

    area_elements = 0.5 * (r_outer**2 - r_inner**2) * dtheta
    return area_elements


# ---------------------------------------------------------------------------
# StandardPolarTransform
# ---------------------------------------------------------------------------


@register_transform
class StandardPolarTransform(CoordinateTransform):
    """Standard-polar coordinate transform with lazy grid caching.

    The *standard polar* coordinate system maps Cartesian pixels to ``(radius, angle)``
    pairs with uniform angular spacing at every radius.

    Implements :class:`~panther_em.coordinates.transform_base.CoordinateTransform`
    and is registered under the key ``"standard_polar"``.

    Parameters
    ----------
    center : tuple[float, float]
        Center of the transformation (row, col) in Cartesian coordinates.
    radius : float
        Maximum radius, in pixels, for the transformation.
    num_angle : int
        Number of angular samples in the polar image.
    num_radius : int
        Number of radial samples in the polar image.
    height : int
        Height of the Cartesian image.
    width : int
        Width of the Cartesian image.
    """

    transform_name: str = "standard_polar"
    supports_energy_preservation: bool = True
    has_periodic_axis: bool = True
    periodic_axis: int = 0  # angle axis is periodic

    def __init__(
        self,
        center: tuple[float, float],
        radius: float,
        num_angle: int,
        num_radius: int,
        height: int,
        width: int,
    ) -> None:
        super().__init__()
        self.center = center
        self.radius = radius
        self.num_angle = num_angle
        self.num_radius = num_radius
        self.height = height
        self.width = width

    @classmethod
    def from_image(
        cls,
        image_shape: tuple[int, int],
        num_angle: int = 360,
        num_radius: int | None = None,
        center: tuple[float, float] | None = None,
        radius: float | None = None,
    ) -> "StandardPolarTransform":
        """Convenience constructor from image shape and polar geometry.

        Parameters
        ----------
        image_shape : tuple[int, int]
            ``(height, width)`` of the Cartesian image.
        num_angle : int, optional
            Number of angular samples. Default 360.
        num_radius : int | None, optional
            Number of radial samples. Defaults to ``ceil(radius)``.
        center : tuple[float, float] | None, optional
            ``(row, col)`` centre. Defaults to image centre.
        radius : float | None, optional
            Maximum radius in pixels. Defaults to ``height / 2``.

        Returns
        -------
        StandardPolarTransform
        """
        height, width = image_shape

        if center is None:
            center = (height / 2 - 0.5, width / 2 - 0.5)

        if radius is None:
            radius = height / 2  # Assuming square images, radius is half the height

        if num_radius is None:
            num_radius = int(np.ceil(radius))

        return get_transform(  # type: ignore[return-value]
            cls,
            center=center,
            radius=radius,
            num_angle=num_angle,
            num_radius=num_radius,
            height=height,
            width=width,
        )

    # ------------------------------------------------------------------ #
    # Shape properties
    # ------------------------------------------------------------------ #

    @property
    def polar_shape(self) -> tuple[int, int]:
        """Shape of the transformed image: ``(num_angle, num_radius)``."""
        return (self.num_angle, self.num_radius)

    @property
    def cartesian_shape(self) -> tuple[int, int]:
        """Shape of the Cartesian image: ``(height, width)``."""
        return (self.height, self.width)

    # ------------------------------------------------------------------ #
    # Coordinate grid computation (called lazily by base-class properties)
    # ------------------------------------------------------------------ #

    def compute_transform_coords(self) -> np.ndarray:
        """Inverse mapping for the Cartesian→polar warp.

        For each output pixel in polar space (radius_idx, angle_idx) returns
        the Cartesian source coordinates to sample.

        Returns
        -------
        np.ndarray
            Shape ``(2, num_angle, num_radius)``, dtype float64.
        """
        t = np.arange(self.num_angle)
        r = np.arange(self.num_radius)
        tt, rr = np.meshgrid(t, r, indexing="ij")
        output_coords = np.stack([rr.flatten(), tt.flatten()], axis=-1)

        source_coords = inverse_polar_to_cartesian_mapping(
            output_coords,
            num_angle=self.num_angle,
            num_radius=self.num_radius,
            max_radius=self.radius,
            center=self.center,
        )

        source_coords = source_coords[:, [1, 0]]  # swap to (row, col) ordering
        return source_coords.T.reshape(2, self.num_angle, self.num_radius)

    def compute_cartesian_coords(self) -> np.ndarray:
        """Inverse mapping for the polar→Cartesian warp.

        For each output pixel in Cartesian space (row, col) returns the polar
        source coordinates to sample.

        Returns
        -------
        np.ndarray
            Shape ``(2, width, height)``.
        """
        r = np.arange(self.height)
        c = np.arange(self.width)
        rr, cc = np.meshgrid(r, c, indexing="ij")
        output_coords = np.stack([cc.flatten(), rr.flatten()], axis=-1)

        source_coords = inverse_cartesian_to_polar_mapping(
            output_coords,
            num_angle=self.num_angle,
            num_radius=self.num_radius,
            max_radius=self.radius,
            center=self.center,
        )

        source_coords = source_coords[:, [1, 0]]  # swap columns
        return source_coords.T.reshape(2, self.width, self.height)

    def compute_jacobian(self) -> np.ndarray:
        """Compute the Jacobian (area-correction) array.

        Each row is identical (the Jacobian depends only on radius for
        uniform angular spacing), so the 1-D radial correction is broadcast
        to the full ``(num_angle, num_radius)`` shape.

        Returns
        -------
        np.ndarray
            Shape ``(num_angle, num_radius)``, dtype float32.
        """
        jac_1d = jacobian_correction_polar(self.num_angle, self.num_radius, self.radius)
        jac_sqrt = np.sqrt(jac_1d).astype(np.float32)
        return np.broadcast_to(
            jac_sqrt[np.newaxis, :], (self.num_angle, self.num_radius)
        ).copy()

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Serialize geometric parameters to a JSON-compatible dict."""
        return {
            "transform_name": self.transform_name,
            "center": list(self.center),
            "radius": float(self.radius),
            "num_angle": int(self.num_angle),
            "num_radius": int(self.num_radius),
            "height": int(self.height),
            "width": int(self.width),
        }

    @classmethod
    def from_dict(
        cls,
        params: dict[str, Any],
        device: Any = None,  # accepted but ignored; transforms are device-agnostic
    ) -> "StandardPolarTransform":
        """Reconstruct from serialized parameters, returning a cached instance."""
        return get_transform(  # type: ignore[return-value]
            cls,
            center=tuple(params["center"]),
            radius=float(params["radius"]),
            num_angle=int(params["num_angle"]),
            num_radius=int(params["num_radius"]),
            height=int(params["height"]),
            width=int(params["width"]),
        )
