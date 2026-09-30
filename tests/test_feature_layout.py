"""FeatureLayout k-run construction vs the legacy FeatureTiling k-intervals."""

from __future__ import annotations

import pytest
import torch

from panther_em.inference.search.tiling import (
    FeatureLayout,
    FeatureTiling,
    rectangles_disjoint,
)


@pytest.mark.parametrize(
    "rects",
    [
        [(0, 0, 8, 4)],
        [(0, 0, 8, 4), (8, 0, 4, 4)],
        [(0, 0, 4, 2), (0, 2, 4, 2), (4, 0, 2, 4)],
        [(2, 1, 3, 3), (0, 0, 2, 5), (5, 0, 1, 1)],
    ],
)
def test_layout_k_runs_match_legacy_tiling(rects):
    layout = FeatureLayout(rects)
    legacy = FeatureTiling.from_extents(rects, device="cpu")
    assert layout.num_features == legacy.num_features
    assert layout.k_stop == legacy.k_stop
    assert [(r.k_start, r.k_stop, r.num_m) for r in layout.k_runs] == [
        (iv.k_start, iv.k_stop, iv.num_m) for iv in legacy.k_intervals
    ]
    assert torch.equal(layout.cells(), legacy.layout)
    # parts cover disjoint m ranges in the same order the legacy interval concatenates
    for run in layout.k_runs:
        ms = [(p.m_start, p.m_stop) for p in run.parts]
        assert len(set(ms)) == len(ms)
    covered = layout.covered_k()
    assert len(covered) == layout.k_stop


def test_layout_validation_and_slots():
    with pytest.raises(ValueError):
        FeatureLayout([(0, 0, 4, 4), (2, 2, 4, 4)])
    with pytest.raises(ValueError):
        FeatureLayout([])
    assert not rectangles_disjoint([(0, 0, 4, 4), (3, 3, 1, 1)])
    assert rectangles_disjoint([(0, 0, 4, 4), (4, 0, 1, 1), (0, 4, 4, 1)])
    layout = FeatureLayout([(0, 0, 2, 3), (2, 0, 1, 3)])
    assert layout.rect_slots(0) == (0, 6) and layout.rect_slots(1) == (6, 9)
    assert hash(layout) == hash(FeatureLayout([(0, 0, 2, 3), (2, 0, 1, 3)]))
    gap = FeatureLayout([(0, 0, 2, 3), (4, 0, 1, 3)])
    assert gap.covered_k() == (True, True, False, False, True)
