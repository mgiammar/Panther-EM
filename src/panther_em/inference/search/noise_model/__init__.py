"""The coloured-noise model of a search: the prepared image and the threshold test.

The whitening it relies on lives in :mod:`panther_em.filters`, the template's error
tables in :mod:`panther_em.approximation`.

- :class:`ImageSet`: one micrograph's search image, CTF planes, noise / signal curves
  and level map;
- :class:`ThresholdTest` (:class:`Calibration`, :class:`NoiseScales`): the per-pixel
  rule that discards pixels whose exact correlation cannot reach the threshold.

The package imports nothing from the rest of the search package at module level (the
search imports it).
"""

from panther_em.inference.search.noise_model.image_set import ImageSet
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
    "ThresholdTest",
    "zeta_quantile",
    "zscore_threshold",
]
