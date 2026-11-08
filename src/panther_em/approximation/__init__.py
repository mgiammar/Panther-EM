"""The low-rank approximation of a decomposition's templates and its errors.

* :class:`Reconstructor` -- the forward model: polar and Cartesian basis features,
  template projections from their leading components, the search's feature kernels
  and their adjoint, all under the decomposition's recorded
  :class:`~panther_em.coordinates.WarpConfig`.
* :class:`PerSpatialFrequencyErrorTable` (:func:`compute_error_tables`) -- ring
  profiles of every template row's reconstruction, residual and signal against the
  exact projections of a reference volume.
"""

from panther_em.approximation.errors import (
    PerSpatialFrequencyErrorTable,
    RadialBins,
    compute_error_tables,
)
from panther_em.approximation.reconstructor import Reconstructor

__all__ = [
    "PerSpatialFrequencyErrorTable",
    "RadialBins",
    "Reconstructor",
    "compute_error_tables",
]
