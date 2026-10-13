"""FeatureLayout k-run construction."""

from __future__ import annotations

import pytest

from panther_em.inference.search.tiling import FeatureLayout, rectangles_disjoint


@pytest.mark.parametrize(
    ("rects", "expected_runs"),
    [
        ([(0, 0, 8, 4)], [(0, 8, 4)]),
        ([(0, 0, 8, 4), (8, 0, 4, 4)], [(0, 8, 4), (8, 12, 4)]),
        ([(0, 0, 4, 2), (0, 2, 4, 2), (4, 0, 2, 4)], [(0, 4, 4), (4, 6, 4)]),
        # k in [0, 2): rect 1 (m 0..5); [2, 5): rect 0 (m 1..4); [5, 6): rect 2 (m 0..1)
        ([(2, 1, 3, 3), (0, 0, 2, 5), (5, 0, 1, 1)], [(0, 2, 5), (2, 5, 3), (5, 6, 1)]),
    ],
)
def test_layout_k_runs(rects, expected_runs):
    layout = FeatureLayout(rects)
    assert layout.num_features == sum(r[2] * r[3] for r in rects)
    assert layout.k_stop == max(r[0] + r[2] for r in rects)
    assert [(r.k_start, r.k_stop, r.num_m) for r in layout.k_runs] == expected_runs
    for run in layout.k_runs:
        ms = [(p.m_start, p.m_stop) for p in run.parts]
        assert len(set(ms)) == len(ms)  # disjoint m ranges within a run
        for part in run.parts:
            k0, m0, ke, me = rects[part.rect]
            assert k0 <= run.k_start and run.k_stop <= k0 + ke
            assert (part.m_start, part.m_stop) == (m0, m0 + me)
    assert len(layout.covered_k()) == layout.k_stop


def test_layout_validation_and_slots():
    with pytest.raises(ValueError):
        FeatureLayout([(0, 0, 4, 4), (2, 2, 4, 4)])
    with pytest.raises(ValueError):
        FeatureLayout([])
    assert not rectangles_disjoint([(0, 0, 4, 4), (3, 3, 1, 1)])
    assert rectangles_disjoint([(0, 0, 4, 4), (4, 0, 1, 1), (0, 4, 4, 1)])
    layout = FeatureLayout([(0, 0, 2, 3), (2, 0, 1, 3)])
    assert layout.num_features == 9
    assert hash(layout) == hash(FeatureLayout([(0, 0, 2, 3), (2, 0, 1, 3)]))
    gap = FeatureLayout([(0, 0, 2, 3), (4, 0, 1, 3)])
    assert gap.covered_k() == (True, True, False, False, True)
