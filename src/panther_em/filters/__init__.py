"""Fourier filters of a dataset, shared by the decomposition and the search.

* :class:`WhiteningFilter` -- the averaged whitening ``W_avg``, every micrograph's
  correction ``C_i`` and the band: the one source of the decomposition's Fourier
  filter, the search image's filter and the noise model's signal weight, with guards
  on the links between them.
* :mod:`~panther_em.filters.curves` -- 1-D spectra on ``torch_fourier_filter``'s
  frequency axis and their DFT grids.
"""

from panther_em.filters.curves import (
    amplitude_spectrum_1d,
    curve_to_grid,
    frequency_axis,
    whitening_curve,
)
from panther_em.filters.whitening import WhiteningFilter

__all__ = [
    "WhiteningFilter",
    "amplitude_spectrum_1d",
    "curve_to_grid",
    "frequency_axis",
    "whitening_curve",
]
