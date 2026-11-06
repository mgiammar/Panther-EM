"""The coloured-noise model of a search: whitening, per-micrograph curves, error tables.

Built alongside the current ``search.noise`` / ``panther_em.spectra`` code, which it
will replace:

- :class:`WhiteningFilter`: ``W_avg``, every micrograph's ``C_i`` and the template band,
  the one source of the decomposition filter, the search-image filter and the signal
  correction, with guards on the links between them;
- :class:`ImageSet`: one micrograph's search image, CTF planes, noise / signal curves
  and level map;
- :class:`PerSpatialFrequencyErrorTable` (:func:`compute_error_tables`): the template's
  ring profiles of reconstruction, residual and signal.
"""

from panther_em.inference.search.noise_model.image_set import ImageSet
from panther_em.inference.search.noise_model.spatial_frequency_error import (
    PerSpatialFrequencyErrorTable,
    RadialBins,
    compute_error_tables,
)
from panther_em.inference.search.noise_model.whitening import WhiteningFilter

__all__ = [
    "ImageSet",
    "PerSpatialFrequencyErrorTable",
    "RadialBins",
    "WhiteningFilter",
    "compute_error_tables",
]
