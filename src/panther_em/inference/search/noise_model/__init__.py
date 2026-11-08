"""The coloured-noise model of a search: the prepared image, error tables, the test.

The whitening it relies on lives in :mod:`panther_em.filters`.

- :class:`ImageSet`: one micrograph's search image, CTF planes, noise / signal curves
  and level map;
- :class:`PerSpatialFrequencyErrorTable` (:func:`compute_error_tables`): the template's
  ring profiles of reconstruction, residual and signal;
- :class:`ThresholdTest` (:class:`Calibration`, :class:`NoiseScales`): the per-pixel
  rule that discards pixels whose exact correlation cannot reach the threshold.

The package imports nothing from the rest of the search package at module level (the
search imports it).
"""

from panther_em.inference.search.noise_model.image_set import ImageSet
from panther_em.inference.search.noise_model.spatial_frequency_error import (
    PerSpatialFrequencyErrorTable,
    RadialBins,
    compute_error_tables,
)
from panther_em.inference.search.noise_model.threshold_test import (
    Calibration,
    NoiseScales,
    ThresholdTest,
    zeta_quantile,
    zscore_threshold,
)

__all__ = [
    "Calibration",
    "ImageSet",
    "NoiseScales",
    "PerSpatialFrequencyErrorTable",
    "RadialBins",
    "ThresholdTest",
    "compute_error_tables",
    "zeta_quantile",
    "zscore_threshold",
]
