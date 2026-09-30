"""Dense per-pixel label state for the multi-stage search."""

from __future__ import annotations

import enum


class PixelLabel(enum.IntEnum):
    """Per-pixel status across search stages.

    - PENDING: not yet searched.
    - UNCERTAIN: searched, but undecided (still eligible for the next stage).
    - REJECTED: decided below the detection threshold.
    - ACCEPTED: decided above the detection threshold.
    - ABSORBED: was UNCERTAIN but lies near an ACCEPTED pixel and was absorbed by
      morphological dilation.
    - MASKED: excluded by the caller's initial mask; never searched.
    """

    PENDING = 0
    UNCERTAIN = 1
    REJECTED = 2
    ACCEPTED = 3
    ABSORBED = 4
    MASKED = 5
