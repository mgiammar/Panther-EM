"""HEALPix orientation neighbourhoods on top of ``HierarchicalS2Grid``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from panther_em.inference.search.hypotheses import HypothesisSpace
from panther_em.inference.search.orientations import (
    S2Neighborhood,
    healpix_orientations,
)

hp = pytest.importorskip("healpy")
grid_mod = pytest.importorskip("torch_so3.hierarchical_s2_grid")


def _space(phi, theta, n_filters=1):
    o = len(phi)
    return HypothesisSpace(
        1, n_filters, o, torch.as_tensor(phi).float(), torch.as_tensor(theta).float()
    )


def test_healpix_orientations_round_trip_through_the_grid():
    phi, theta = healpix_orientations(2)
    assert phi.shape == theta.shape == (48,) and phi.dtype == np.float32
    grid = grid_mod.HierarchicalS2Grid(2, 1)
    assert np.array_equal(grid.index_from_angle(theta, phi, level=0), np.arange(48))
    s2 = S2Neighborhood.from_space(_space(phi, theta))
    assert s2 is not None and s2.grid.nside_finest == 2 and s2.n_orientations == 48
    assert np.array_equal(s2.ipix, np.arange(48))


def test_from_space_recognises_subsets_in_any_order_and_rejects_other_angles():
    nside, ring_ids = 4, np.arange(0, 192, 3)  # a RING-ordered subset of the sphere
    theta, phi = (np.degrees(a) for a in hp.pix2ang(nside, ring_ids, nest=False))
    s2 = S2Neighborhood.from_space(_space(phi, theta))
    assert s2 is not None and s2.grid.nside_finest == nside
    assert np.array_equal(s2.ipix, hp.ring2nest(nside, ring_ids))

    assert (
        S2Neighborhood.from_space(_space(np.linspace(0, 300, 6), np.linspace(0, 90, 6)))
        is None
    )
    assert S2Neighborhood.from_space(HypothesisSpace(1, 1, 6)) is None  # no angles


@pytest.mark.parametrize("rings", [0, 1, 2])
def test_neighbor_table_matches_grid_neighbor_indices(rings):
    nside = 4
    ipix = np.random.default_rng(0).permutation(
        np.unique(np.concatenate([np.arange(0, 192, 3), np.arange(100, 110)]))
    )
    grid = grid_mod.HierarchicalS2Grid(nside, 1)
    theta, phi = grid.angle_from_index(ipix, level=0)
    s2 = S2Neighborhood.from_space(_space(phi, theta))
    assert s2 is not None and np.array_equal(s2.ipix, ipix)
    table = s2.neighbor_table(rings)
    assert table.dtype == torch.long and table.shape[0] == len(ipix)
    present = {int(p): o for o, p in enumerate(ipix)}
    for o, pix in enumerate(ipix):
        ref = sorted(
            present[j]
            for j in grid.neighbor_indices(int(pix), 0, rings)
            if j in present
        )
        row = table[o]
        assert row[row >= 0].tolist() == ref and o in ref
        assert bool(torch.all(row[len(ref) :] == -1))
    assert s2.neighbor_table(rings) is table  # cached
    assert s2.neighbor_table(rings, "cpu").device.type == "cpu"
