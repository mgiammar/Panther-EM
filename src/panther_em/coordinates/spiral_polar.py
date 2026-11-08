"""Spiral polar coordinate system for interpolation and transforms.

Provides :class:`SpiralPolarTransform`, which implements a polar coordinate system where
radial nodes are non-uniformly spaced and angular nodes are offset from each other to
produce a spiral pattern for sampling. This samples the underlying Cartesian grid more
uniformly near the center and edges than either a standard polar grid or the offset
polar grid.

The spiral coordinates are defined as:

Symbols
-------
``r_max``
    Outer radius of the transform, in pixels (``radius`` constructor argument).
``n_a`` = ``num_angle``
    Number of angular samples -- equivalently, the number of spiral "arms".
``n_r`` = ``num_radius``
    Number of radial samples per arm.
``c``
    Free scalar parameter (dimensionless, defined relative to ``r_max`` -- see the
    radial profile below) controlling the transition from linear to square-root
    radial growth. Small ``c`` (near 0) gives a nearly pure square-root radial
    profile (denser sampling near the center, matching equal-area polar sampling);
    large ``c`` approaches uniform (linear) radial spacing, as in a standard polar
    grid. Must be ``> 0``. Default 0.3.
``p_o`` = ``percent_arc_offset``
    Fraction of a full turn (``2*pi / n_a``) that the spiral rotates for each
    radial step outward. Default 0.322.

Derived constants
------------------
::

    t_max = 0.5 * (1 + 2 * c)
    dt    = t_max / n_r

``dt`` is the step size of a uniform parameter ``t`` from which physical radii are
derived.

Radial profile
---------------
Physical radius as a function of the uniform parameter ``t``::

    r(t) = r_max * (sqrt(c**2 + 2*t) - c)

Since ``dr/dt = r_max / (r/r_max + c)``, for a fixed step ``dt`` the radial spacing
``dr`` shrinks like ``1 / (c*r_max + r)``, so the number of nodes per annulus stays
roughly constant once ``r >> c*r_max``. This is what gives the spiral its more uniform
sampling density relative to a plain polar grid (whose annuli grow linearly in area with
radius, but whose node count per ring is fixed).

Grid sample coordinates
------------------------
Each output pixel of the polar-space image is indexed by an integer
``(angle_idx, radius_idx)`` with ``angle_idx in [0, n_a)`` and
``radius_idx in [0, n_r)``, corresponding to a specific arm and a specific
radial step along it::

    t(radius_idx)     = (radius_idx + 1) * dt
    r(radius_idx)     = r_max * (sqrt(c**2 + 2*t(radius_idx)) - c)
    twist(radius_idx) = p_o * 2*pi / (n_a * dt) * t(radius_idx)
    theta(radius_idx, angle_idx) = 2*pi * angle_idx / n_a + twist(radius_idx)

    row = r(radius_idx) * sin(theta) + center_row
    col = r(radius_idx) * cos(theta) + center_col

Radial boundaries of the inverse warp
--------------------------------------
The first ring sits at ``t = dt`` so the Cartesian pixels nearest the centre map to
``radius_idx`` in ``[-1, 0)``, outside the polar image. Before the polar -> Cartesian
warp, :meth:`SpiralPolarTransform._pad_radial_axis` extends both axes.
"""

from typing import Any

import numpy as np
import torch

from .transform_base import CoordinateTransform, get_transform, register_transform

# ---------------------------------------------------------------------------
# Shared geometry derivation
# ---------------------------------------------------------------------------


def _spiral_geometry(
    num_angle: int,
    num_radius: int,
    c: float,
    percent_arc_offset: float,
) -> tuple[float, float]:
    """Derive the scalar geometry parameters shared by all spiral mapping functions.

    Parameters
    ----------
    num_angle : int
        Number of angular samples (spiral arms).
    num_radius : int
        Number of radial samples per arm.
    c : float
        Free linear-to-sqrt growth transition parameter; see module docstring.
        Must be ``> 0``.
    percent_arc_offset : float
        Fraction of a full turn the spiral rotates per outward radial step.

    Returns
    -------
    dt : float
        Step size of the uniform radial parameter ``t``.
    twist_rate : float
        Angular twist (radians) accumulated per unit of ``t``.

    Raises
    ------
    ValueError
        If ``c`` is not strictly positive.
    """
    if c <= 0:
        raise ValueError(f"c must be > 0 for a valid radial profile; got c={c}.")

    t_max = 0.5 * (1 + 2 * c)
    dt = t_max / num_radius

    twist_rate = percent_arc_offset * 2 * np.pi / (num_angle * dt)

    return dt, twist_rate


def _radius_from_t(t: np.ndarray, max_radius: float, c: float) -> np.ndarray:
    """Physical radius ``r(t) = max_radius * (sqrt(c**2 + 2t) - c)``."""
    return max_radius * (np.sqrt(c**2 + 2 * t) - c)


def _t_from_radius(radius: np.ndarray, max_radius: float, c: float) -> np.ndarray:
    """Inverse of :func:`_radius_from_t`: solve for ``t`` given physical radius."""
    return ((radius / max_radius + c) ** 2 - c**2) / 2


# ---------------------------------------------------------------------------
# Coordinate mapping functions
# ---------------------------------------------------------------------------


def forward_cartesian_to_spiral_polar_mapping(
    input_coords: np.ndarray,  # (M, 2) - (row, col)
    num_angle: int,
    num_radius: int,
    max_radius: float,
    center: tuple[float, float],
    c: float = 0.3,
    percent_arc_offset: float = 0.322,
) -> np.ndarray:
    """The forward mapping function to take cartesian coords to spiral polar coords.

    Parameters
    ----------
    input_coords : np.ndarray
        (M, 2) array of (row, col) coordinates in cartesian space.
    num_angle : int
        Number of angular samples (spiral arms) in polar space.
    num_radius : int
        Number of radial samples per arm in polar space.
    max_radius : float
        Maximum radius for the polar coordinate system.
    center : tuple[float, float]
        Center point (row, col) of the polar coordinate system.
    c : float, optional
        Free linear-to-sqrt growth transition parameter; see module docstring.
        Must be ``> 0``. Default 0.3.
    percent_arc_offset : float, optional
        Fraction of a full turn rotated per radial step; see module docstring.
        Default 0.322.

    Returns
    -------
    polar_coords : np.ndarray
        (M, 2) array of (radius_idx, angle_idx) in spiral polar space.
    """
    row = input_coords[:, 0] - center[0]
    col = input_coords[:, 1] - center[1]

    radius = np.sqrt(row**2 + col**2)
    angle = np.arctan2(row, col)
    angle = angle % (2 * np.pi)

    dt, twist_rate = _spiral_geometry(num_angle, num_radius, c, percent_arc_offset)

    t = _t_from_radius(radius, max_radius, c)
    radius_idx = t / dt - 1

    twist = twist_rate * t
    angle_idx = ((angle - twist) % (2 * np.pi)) / (2 * np.pi) * num_angle

    return np.column_stack((radius_idx, angle_idx))


def forward_spiral_polar_to_cartesian_mapping(
    input_coords: np.ndarray,  # (M, 2) - (radius_idx, angle_idx)
    num_angle: int,
    num_radius: int,
    max_radius: float,
    center: tuple[float, float],
    c: float = 0.3,
    percent_arc_offset: float = 0.322,
) -> np.ndarray:
    """Forward mapping: Spiral polar coordinates -> Cartesian coordinates.

    Parameters
    ----------
    input_coords : np.ndarray
        (M, 2) array of (radius_idx, angle_idx) in spiral polar space.
    num_angle : int
        Number of angular samples (spiral arms) in polar space.
    num_radius : int
        Number of radial samples per arm in polar space.
    max_radius : float
        Maximum radius for the polar coordinate system.
    center : tuple[float, float]
        Center point (row, col) of the polar coordinate system.
    c : float, optional
        Free linear-to-sqrt growth transition parameter; see module docstring.
        Must be ``> 0``. Default 0.3.
    percent_arc_offset : float, optional
        Fraction of a full turn rotated per radial step; see module docstring.
        Default 0.322.

    Returns
    -------
    cartesian_coords : np.ndarray
        (M, 2) array of (row, col) coordinates in cartesian space.
    """
    radius_idx = input_coords[:, 0]
    angle_idx = input_coords[:, 1]

    dt, twist_rate = _spiral_geometry(num_angle, num_radius, c, percent_arc_offset)

    t = (radius_idx + 1) * dt
    radius = _radius_from_t(t, max_radius, c)

    twist = twist_rate * t
    angle = (angle_idx / num_angle) * (2 * np.pi) + twist

    row = radius * np.sin(angle) + center[0]
    col = radius * np.cos(angle) + center[1]

    return np.column_stack((row, col))


def inverse_spiral_polar_to_cartesian_mapping(
    output_coords: np.ndarray,
    num_angle: int,
    num_radius: int,
    max_radius: float,
    center: tuple[float, float],
    c: float = 0.3,
    percent_arc_offset: float = 0.322,
) -> np.ndarray:
    """Inverse mapping for warping cartesian image to spiral polar space.

    Given output coordinates in spiral polar space, returns the corresponding
    input coordinates in cartesian space. Used with skimage.transform.warp.

    Parameters
    ----------
    output_coords : np.ndarray
        Array of (col, row) coordinates in the output (spiral polar) image.
        col corresponds to radius, row corresponds to angle.
    num_angle : int
        Number of angular samples (spiral arms) in polar space.
    num_radius : int
        Number of radial samples per arm in polar space.
    max_radius : float
        Maximum radius for the polar coordinate system.
    center : tuple[float, float]
        Center point (row, col) of the polar coordinate system in the input cartesian
        image.
    c : float, optional
        Free linear-to-sqrt growth transition parameter; see module docstring.
        Must be ``> 0``. Default 0.3.
    percent_arc_offset : float, optional
        Fraction of a full turn rotated per radial step; see module docstring.
        Default 0.322.

    Returns
    -------
    coords : np.ndarray
        Array of (col, row) coordinates in the input (cartesian) image.
    """
    radius_idx = output_coords[:, 0]
    angle_idx = output_coords[:, 1]

    dt, twist_rate = _spiral_geometry(num_angle, num_radius, c, percent_arc_offset)

    t = (radius_idx + 1) * dt
    radius = _radius_from_t(t, max_radius, c)

    twist = twist_rate * t
    angle = (angle_idx / num_angle) * (2 * np.pi) + twist

    row = radius * np.sin(angle) + center[0]
    col = radius * np.cos(angle) + center[1]

    return np.column_stack((col, row))


def inverse_cartesian_to_spiral_polar_mapping(
    output_coords: np.ndarray,
    num_angle: int,
    num_radius: int,
    max_radius: float,
    center: tuple[float, float],
    c: float = 0.3,
    percent_arc_offset: float = 0.322,
) -> np.ndarray:
    """Inverse mapping for warping spiral polar image back to cartesian space.

    Given output coordinates in cartesian space, returns the corresponding
    input coordinates in spiral polar space. Used with skimage.transform.warp.

    Parameters
    ----------
    output_coords : np.ndarray
        Array of (col, row) coordinates in the output (cartesian) image.
    num_angle : int
        Number of angular samples (spiral arms) in polar space.
    num_radius : int
        Number of radial samples per arm in polar space.
    max_radius : float
        Maximum radius for the polar coordinate system.
    center : tuple[float, float]
        Center point (row, col) of the polar coordinate system.
    c : float, optional
        Free linear-to-sqrt growth transition parameter; see module docstring.
        Must be ``> 0``. Default 0.3.
    percent_arc_offset : float, optional
        Fraction of a full turn rotated per radial step; see module docstring.
        Default 0.322.

    Returns
    -------
    coords : np.ndarray
        Array of (col, row) coordinates in the input (spiral polar) image.
        col corresponds to radius, row corresponds to angle.
    """
    col = output_coords[:, 0]
    row = output_coords[:, 1]

    dc = col - center[1]
    dr = row - center[0]

    radius = np.sqrt(dr**2 + dc**2)
    angle = np.arctan2(dr, dc)
    angle = angle % (2 * np.pi)

    dt, twist_rate = _spiral_geometry(num_angle, num_radius, c, percent_arc_offset)

    t = _t_from_radius(radius, max_radius, c)
    radius_idx = t / dt - 1

    twist = twist_rate * t
    angle_idx = ((angle - twist) % (2 * np.pi)) / (2 * np.pi) * num_angle

    return np.column_stack((radius_idx, angle_idx))


def jacobian_correction_spiral_polar(
    num_angle: int,
    num_radius: int,
    max_radius: float,
    c: float = 0.3,
    percent_arc_offset: float = 0.322,
) -> np.ndarray:
    """Correction factor for area element in spiral polar coordinates.

    Calculates the exact Cartesian area covered by each spiral polar grid cell.
    Radial bin boundaries are the midpoints between consecutive sample radii
    (Voronoi cells along the radial axis); the innermost bin extends down to
    ``r=0`` and the outermost bin extends out to the last sample radius.

    Parameters
    ----------
    num_angle : int
        Number of angular samples (spiral arms) in polar space.
    num_radius : int
        Number of radial samples per arm in polar space.
    max_radius : float
        Maximum radius for the polar coordinate system.
    c : float, optional
        Free linear-to-sqrt growth transition parameter; see module docstring.
        Must be ``> 0``. Default 0.3.
    percent_arc_offset : float, optional
        Fraction of a full turn rotated per radial step; see module docstring.
        Default 0.322.

    Returns
    -------
    area_elements : np.ndarray
        (num_radius,) array containing the Cartesian area of each spiral polar pixel.
    """
    dt, _twist_rate = _spiral_geometry(num_angle, num_radius, c, percent_arc_offset)

    radius_idx = np.arange(num_radius)
    t = (radius_idx + 1) * dt
    radii = _radius_from_t(t, max_radius, c)

    midpoints = (radii[:-1] + radii[1:]) / 2.0
    r_inner = np.empty(num_radius)
    r_outer = np.empty(num_radius)
    r_inner[0] = 0.0
    r_inner[1:] = midpoints
    r_outer[:-1] = midpoints
    r_outer[-1] = radii[-1]

    dtheta = (2 * np.pi) / num_angle
    area_elements = 0.5 * (r_outer**2 - r_inner**2) * dtheta
    return area_elements


# ---------------------------------------------------------------------------
# Radial boundaries of the inverse warp
# ---------------------------------------------------------------------------


RADIAL_PAD = 10


def _lagrange_weights(nodes: np.ndarray, x: float) -> np.ndarray:
    """Weights of the Lagrange polynomial through ``nodes``, evaluated at ``x``."""
    weights = np.ones(len(nodes))
    for j, node in enumerate(nodes):
        others = np.delete(nodes, j)
        weights[j] = np.prod((x - others) / (node - others))
    return weights


def origin_continuation(
    num_angle: int,
    num_radius: int,
    max_radius: float,
    c: float = 0.3,
    percent_arc_offset: float = 0.322,
    num_rows: int = RADIAL_PAD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linear map from the innermost rings to rows continued through the origin.

    Notes
    -----
    Padded row ``m`` sits at ``radius_idx = -1 - m`` (``t = -m * dt``). With
    ``A = rfft(f[:, :num_nodes], axis=0) * untwist`` (angular Fourier coefficients of
    the rings, twist removed), the row is::

        irfft(sum_j weights[m, j] * A[:, nodes[m, j]] * phases[:, m], n=num_angle)

    Parameters
    ----------
    num_angle : int
        Number of angular samples (spiral arms).
    num_radius : int
        Number of radial samples per arm (at least 4).
    max_radius : float
        Maximum radius of the transform, in pixels.
    c : float, optional
        Radial growth parameter; see module docstring. Default 0.3.
    percent_arc_offset : float, optional
        Fraction of a full turn rotated per radial step. Default 0.322.
    num_rows : int, optional
        Requested number of rows. Fewer are returned if the spiral map does not
        extend that far below ``t = 0`` (it needs ``c**2 + 2*t > 0``). Default
        :data:`RADIAL_PAD`.

    Returns
    -------
    untwist : np.ndarray
        ``(num_angle // 2 + 1, num_nodes)`` complex128 ``exp(-i k twist(t_i))``.
    nodes : np.ndarray
        ``(rows, 4)`` int64 ring indices interpolated from, all ``< num_nodes``.
    weights : np.ndarray
        ``(rows, 4)`` float64 interpolation weights.
    phases : np.ndarray
        ``(num_angle // 2 + 1, rows)`` complex128 ``(-1)**k exp(i k twist(t_m))``
        (only ``k = 0`` for the row at the origin).
    """
    if num_radius < 4:
        raise ValueError(f"num_radius must be at least 4; got {num_radius}.")
    dt, twist_rate = _spiral_geometry(num_angle, num_radius, c, percent_arc_offset)
    num_rows = min(num_rows, int(np.ceil(c**2 / (2 * dt))))

    k = np.arange(num_angle // 2 + 1)
    radii = _radius_from_t((np.arange(num_radius) + 1) * dt, max_radius, c)
    nodes = np.zeros((num_rows, 4), dtype=np.int64)
    weights = np.zeros((num_rows, 4))
    phases = np.zeros((k.size, num_rows), dtype=np.complex128)
    for m in range(num_rows):
        if m == 0:
            # Origin: a_0 is even in r, so a quadratic in r**2 through three rings
            nodes[m] = np.arange(4)
            weights[m, :3] = _lagrange_weights(radii[:3] ** 2, 0.0)
            phases[0, m] = 1.0
            continue
        t_m = -m * dt
        rho = _radius_from_t(t_m, max_radius, c)  # negative
        s = _t_from_radius(-rho, max_radius, c) / dt - 1  # radius_idx of |rho|
        start = min(max(int(np.floor(s)) - 1, 0), num_radius - 4)
        nodes[m] = start + np.arange(4)
        weights[m] = _lagrange_weights(nodes[m].astype(np.float64), s)
        phases[:, m] = (-1.0) ** k * np.exp(1j * k * twist_rate * t_m)

    num_nodes = int(nodes.max()) + 1
    ring_t = (np.arange(num_nodes) + 1) * dt
    untwist = np.exp(-1j * k[:, None] * twist_rate * ring_t[None, :])
    return untwist, nodes, weights, phases


# ---------------------------------------------------------------------------
# SpiralPolarTransform
# ---------------------------------------------------------------------------


@register_transform
class SpiralPolarTransform(CoordinateTransform):
    """Spiral-polar coordinate transform with lazy grid caching.

    The *spiral polar* coordinate system is a polar grid where the radial sample spacing
    grows non-uniformly (near-linear close to the origin, transitioning to square-root
    growth further out, so that the number of samples per annulus stays constant) and
    each successive radial step is rotated by a fixed angular "twist". The result is an
    ensemble of ``num_angle`` rotated copies of a single spiral arm, which can sample
    the underlying Cartesian grid more uniformly than either
    :class:`~panther_em.coordinates.standard_polar.StandardPolarTransform`
    or :class:`~panther_em.coordinates.offset_polar.OffsetPolarTransform`.
    See the module docstring for the exact governing equations.

    Implements :class:`~panther_em.coordinates.transform_base.CoordinateTransform`
    and is registered under the key ``"spiral_polar"``.

    Parameters
    ----------
    center : tuple[float, float]
        Center of the transformation (row, col) in Cartesian coordinates.
    radius : float
        Maximum radius, in pixels, for the transformation.
    num_angle : int
        Number of angular samples (spiral arms) in the spiral polar image.
    num_radius : int
        Number of radial samples per arm in the spiral polar image.
    height : int
        Height of the Cartesian image.
    width : int
        Width of the Cartesian image.
    c : float, optional
        Free scalar parameter (dimensionless, relative to ``radius``) controlling
        the transition from linear to square-root radial growth; see module
        docstring. Must be ``> 0``. Unlike a spacing-ratio-style parameterization,
        any positive value is valid regardless of ``num_radius`` or ``radius``,
        which makes ``c`` suitable for direct sweeping/optimization. Default 0.3.
    percent_arc_offset : float, optional
        Fraction of a full turn (``2*pi / num_angle``) the spiral rotates for
        each radial step outward. Default 0.322.
    """

    transform_name: str = "spiral_polar"
    supports_energy_preservation: bool = True
    has_periodic_axis: bool = True
    periodic_axis: int = 0  # angle axis is periodic (twist is shared across angle_idx)

    def __init__(
        self,
        center: tuple[float, float],
        radius: float,
        num_angle: int,
        num_radius: int,
        height: int,
        width: int,
        c: float = 0.3,
        percent_arc_offset: float = 0.322,
    ) -> None:
        super().__init__()
        self.center = center
        self.radius = radius
        self.num_angle = num_angle
        self.num_radius = num_radius
        self.height = height
        self.width = width
        self.c = c
        self.percent_arc_offset = percent_arc_offset

        # Precompute derived scalar geometry once (cheap; not a coordinate grid).
        self._dt, self._twist_rate = _spiral_geometry(
            num_angle,
            num_radius,
            c,
            percent_arc_offset,
        )
        # Per-device tensors of `origin_continuation`, built on first use
        self._continuation_cache: dict[str, tuple[torch.Tensor, ...]] = {}

    @classmethod
    def from_image(
        cls,
        image_shape: tuple[int, int],
        num_radius: int | None = None,
        num_angle: int | None = None,
        center: tuple[float, float] | None = None,
        radius: float | None = None,
        c: float = 0.3,
        percent_arc_offset: float = 0.322,
    ) -> "SpiralPolarTransform":
        """Convenience constructor from image shape and spiral polar geometry.

        Parameters
        ----------
        image_shape : tuple[int, int]
            ``(height, width)`` of the Cartesian image.
        num_radius : int | None, optional
            Number of radial samples per arm. Defaults to ``ceil(radius)``.
        num_angle : int | None, optional
            Number of angular samples (spiral arms). Defaults to
            ``round((h * w / num_radius) / spacing_ratio_relative_to_cartesian)``.
        center : tuple[float, float] | None, optional
            ``(row, col)`` centre. Defaults to image centre.
        radius : float | None, optional
            Maximum radius in pixels. Defaults to ``height / 2``.
        c : float, optional
            Free linear-to-sqrt radial growth transition parameter passed through
            to the transform; see module docstring and the class docstring.
            Must be ``> 0``. Default 0.3.
        percent_arc_offset : float, optional
            Fraction of a full turn rotated per radial step; see module
            docstring. Default 0.322.

        Returns
        -------
        SpiralPolarTransform
        """
        height, width = image_shape

        if center is None:
            center = (height / 2 - 0.5, width / 2 - 0.5)

        if radius is None:
            radius = height / 2  # Assuming square images, radius is half the height

        if num_radius is None:
            num_radius = int(np.ceil(radius))

        if num_angle is None:
            num_angle = round(height * width / num_radius)

        return get_transform(  # type: ignore[return-value]
            cls,
            center=center,
            radius=radius,
            num_angle=num_angle,
            num_radius=num_radius,
            height=height,
            width=width,
            c=c,
            percent_arc_offset=percent_arc_offset,
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
        """Inverse mapping for the Cartesian->polar warp.

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

        source_coords = inverse_spiral_polar_to_cartesian_mapping(
            output_coords,
            num_angle=self.num_angle,
            num_radius=self.num_radius,
            max_radius=self.radius,
            center=self.center,
            c=self.c,
            percent_arc_offset=self.percent_arc_offset,
        )

        source_coords = source_coords[:, [1, 0]]  # swap to (row, col) ordering
        return source_coords.T.reshape(2, self.num_angle, self.num_radius)

    def compute_cartesian_coords(self) -> np.ndarray:
        """Inverse mapping for the polar->Cartesian warp.

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

        source_coords = inverse_cartesian_to_spiral_polar_mapping(
            output_coords,
            num_angle=self.num_angle,
            num_radius=self.num_radius,
            max_radius=self.radius,
            center=self.center,
            c=self.c,
            percent_arc_offset=self.percent_arc_offset,
        )

        source_coords = source_coords[:, [1, 0]]  # swap columns
        return source_coords.T.reshape(2, self.width, self.height)

    def compute_jacobian(self) -> np.ndarray:
        """Compute the Jacobian (area-correction) array.

        Each row is identical (the Jacobian depends only on radius, not on
        the angular twist), so the 1-D radial correction is broadcast to the
        full ``(num_angle, num_radius)`` shape.

        Returns
        -------
        np.ndarray
            Shape ``(num_angle, num_radius)``, dtype float32.
        """
        jac_1d = jacobian_correction_spiral_polar(
            self.num_angle,
            self.num_radius,
            self.radius,
            self.c,
            self.percent_arc_offset,
        )
        jac_sqrt = np.sqrt(jac_1d).astype(np.float32)
        return np.broadcast_to(
            jac_sqrt[np.newaxis, :], (self.num_angle, self.num_radius)
        ).copy()

    # ------------------------------------------------------------------ #
    # Radial boundaries of the inverse warp
    # ------------------------------------------------------------------ #

    def clear_cache(self) -> None:
        """Release cached grids, device copies and origin-continuation tensors."""
        super().clear_cache()
        self._continuation_cache.clear()

    def _continuation_tensors(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        """:func:`origin_continuation` as tensors on ``device`` (cached)."""
        key = str(device)
        if key not in self._continuation_cache:
            arrays = origin_continuation(
                self.num_angle,
                self.num_radius,
                self.radius,
                self.c,
                self.percent_arc_offset,
            )
            self._continuation_cache[key] = tuple(
                torch.as_tensor(a, device=device) for a in arrays
            )
        return self._continuation_cache[key]

    def _pad_radial_axis(
        self, image: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray | torch.Tensor, int]:
        """Continue the radial axis through the origin and mirror it past ``radius``.

        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            ``(num_angle, num_radius)`` real polar image, Jacobian already removed.

        Returns
        -------
        tuple[np.ndarray | torch.Tensor, int]
            The padded image and the number of rows prepended.
        """
        is_numpy = isinstance(image, np.ndarray)
        f = torch.from_numpy(image) if is_numpy else image
        untwist, nodes, weights, phases = self._continuation_tensors(f.device)

        rings = torch.fft.rfft(f[:, : untwist.shape[1]].double(), dim=0) * untwist
        spectrum = (rings[:, nodes] * weights).sum(dim=-1) * phases
        inner = torch.fft.irfft(spectrum, n=self.num_angle, dim=0)
        inner = inner.flip(-1).to(f.dtype)  # radius_idx -rows ... -1
        outer = f[:, -RADIAL_PAD:].flip(-1)  # radius_idx n_r ... n_r + RADIAL_PAD - 1

        padded = torch.cat([inner, f, outer], dim=1)
        return (padded.numpy() if is_numpy else padded), inner.shape[1]

    def _outer_radius_index(self) -> float:
        """``radius_idx`` of ``r = radius``: the last ring (plus rounding slack)."""
        return self.num_radius - 1 + 1e-6

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
            "c": float(self.c),
            "percent_arc_offset": float(self.percent_arc_offset),
        }

    @classmethod
    def from_dict(
        cls,
        params: dict[str, Any],
        device: Any = None,  # accepted but ignored; transforms are device-agnostic
    ) -> "SpiralPolarTransform":
        """Reconstruct from serialized parameters, returning a cached instance."""
        return get_transform(  # type: ignore[return-value]
            cls,
            center=tuple(params["center"]),
            radius=float(params["radius"]),
            num_angle=int(params["num_angle"]),
            num_radius=int(params["num_radius"]),
            height=int(params["height"]),
            width=int(params["width"]),
            c=float(params["c"]),
            percent_arc_offset=float(params["percent_arc_offset"]),
        )
