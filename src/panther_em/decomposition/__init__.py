"""Decomposition of a volume's polar projections into angular-frequency SVD blocks.

* :class:`PolarProjectionDecomposer` -- projects a volume at the given orientations,
  warps the projections onto a polar grid and decomposes every angular-frequency block.
* :class:`DecompositionResult` / :class:`SparseDecompositionResult` -- the dense and
  top-``n`` storage layouts (common interface :class:`DecompositionBase`), saved to and
  loaded from HDF5 with their coordinate transform and warp settings.
* :data:`Rectangle` -- a ``(k_start, m_start, k_extent, m_extent)`` block of components.
* :mod:`~panther_em.decomposition.projections` -- the projection generator.
"""

from panther_em.decomposition.decomposer import PolarProjectionDecomposer
from panther_em.decomposition.rectangles import Rectangle
from panther_em.decomposition.result import (
    DecompositionBase,
    DecompositionResult,
    SparseDecompositionResult,
)

__all__ = [
    "DecompositionBase",
    "DecompositionResult",
    "PolarProjectionDecomposer",
    "Rectangle",
    "SparseDecompositionResult",
]
