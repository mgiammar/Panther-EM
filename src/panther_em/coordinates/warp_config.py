"""The interpolation settings of a warp between polar and Cartesian grids."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class WarpConfig:
    """Interpolation settings of a polar <-> Cartesian warp.

    One instance names the warp of a decomposition: the decomposer warps projections
    onto the polar grid with it, and every consumer of the decomposition (kernels,
    their adjoint, reconstructions, error tables) warps polar images back with it.

    Attributes
    ----------
    order : int
        B-spline interpolation order.
    mode, cval : str, float
        Boundary handling outside the source grid (``"constant"`` with ``cval``).
    preserve_energy : bool
        Fold the transform's Jacobian into the warp.
    wrap_angular_axis : bool
        Treat the angular axis as periodic (polar -> Cartesian only).
    """

    order: int = 5
    mode: str = "constant"
    cval: float = 0.0
    preserve_energy: bool = True
    wrap_angular_axis: bool = True

    def cartesian_kwargs(self) -> dict[str, Any]:
        """Keyword arguments of ``CoordinateTransform.to_cartesian``."""
        return self.to_dict()

    def polar_kwargs(self) -> dict[str, Any]:
        """Keyword arguments of ``CoordinateTransform.to_transform_space``."""
        values = self.to_dict()
        del values["wrap_angular_axis"]
        return values

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable fields."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> WarpConfig:
        """Inverse of :meth:`to_dict`."""
        return cls(**values)
