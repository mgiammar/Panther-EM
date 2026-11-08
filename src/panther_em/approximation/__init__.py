"""The low-rank approximation of a decomposition's templates and its errors.

* :class:`Reconstructor` -- the forward model: polar and Cartesian basis features,
  template projections from their leading components, the search's feature kernels
  and their adjoint, all under the decomposition's recorded
  :class:`~panther_em.coordinates.WarpConfig`.

Errors of a selection of feature rectangles, in three tiers:

* cheap (:mod:`.energies`) -- :func:`truncation_error`: every row's polar truncation
  error from ``U``, ``S`` and the recorded block energies alone, split into the
  angular part ``rho_k`` and the eigenvector part ``rho_m``; :func:`energy_spectrum`
  pools it for every leading-corner rectangle, which :func:`suggest_rectangles`
  reads to suggest ranks for an error target or a budget;
* exact (:func:`compute_errors`) -- :class:`ApproximationErrors`: the Cartesian
  reconstructions against the reference volume's projections, reduced by frequency,
  radius, orientation or in total (for publication; includes the warp's own error);
* noise model (:func:`compute_error_tables`) -- the
  :class:`PerSpatialFrequencyErrorTable` the search's threshold test reads.
"""

from panther_em.approximation.energies import (
    EnergySpectrum,
    TruncationError,
    energy_spectrum,
    truncation_error,
)
from panther_em.approximation.errors import (
    ApproximationErrors,
    PerSpatialFrequencyErrorTable,
    RadialBins,
    compute_error_tables,
    compute_errors,
)
from panther_em.approximation.reconstructor import Reconstructor
from panther_em.approximation.selection import RectangleCandidate, suggest_rectangles

__all__ = [
    "ApproximationErrors",
    "EnergySpectrum",
    "PerSpatialFrequencyErrorTable",
    "RadialBins",
    "Reconstructor",
    "RectangleCandidate",
    "TruncationError",
    "compute_error_tables",
    "compute_errors",
    "energy_spectrum",
    "suggest_rectangles",
    "truncation_error",
]
