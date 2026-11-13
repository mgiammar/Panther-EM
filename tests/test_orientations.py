"""orientations: HEALPix grids, their recognition, levels and ring neighborhoods."""

from __future__ import annotations

import healpy as hp
import numpy as np
import pytest
import torch

from panther_em.orientations import (
    healpix_ipix,
    healpix_level_ids,
    healpix_orientations,
    psi_degrees,
    ring_neighborhoods,
)


def test_psi_degrees_are_uniform():
    assert torch.equal(psi_degrees(4), torch.tensor([0.0, 90.0, 180.0, 270.0]))


@pytest.mark.parametrize("nside", [1, 2, 4])
def test_healpix_ipix_recognizes_its_own_grid(nside):
    phi, theta = healpix_orientations(nside)
    assert phi.dtype == np.float32 and phi.shape == (12 * nside**2,)
    found = healpix_ipix(phi, theta)
    assert found is not None
    assert found[0] == nside
    assert np.array_equal(found[1], np.arange(12 * nside**2))
    # torch input, a permuted subset: the ids follow the given order
    order = np.random.default_rng(0).permutation(phi.size)[: phi.size // 2]
    sub = healpix_ipix(torch.from_numpy(phi[order]), torch.from_numpy(theta[order]))
    assert sub is not None and np.array_equal(sub[1], order)


def test_healpix_ipix_rejects_off_grid_orientations():
    phi, theta = healpix_orientations(2)
    assert healpix_ipix(phi + 1.0, theta, max_nside=64) is None
    with pytest.raises(ValueError, match="shape"):
        healpix_ipix(phi, theta[:-1])


def test_healpix_level_ids_keep_one_child_per_coarser_pixel():
    ipix = np.arange(48)  # nside = 2, NESTED: parents at nside = 1 own 4 children
    assert healpix_level_ids(ipix, 1).tolist() == list(range(0, 48, 4))
    assert healpix_level_ids(ipix, 2).tolist() == list(range(0, 48, 16))
    assert healpix_level_ids(ipix, 0).tolist() == list(range(48))
    # orientation ids index the template's own orientation order, not the sphere
    sub = np.array([5, 4, 8, 0])
    assert healpix_level_ids(sub, 1).tolist() == [1, 2, 3]
    with pytest.raises(ValueError):
        healpix_level_ids(ipix, -1)


@pytest.mark.parametrize("nside", [2, 4])
def test_ring_neighborhoods_match_healpy(nside):
    npix = 12 * nside**2
    assert np.array_equal(ring_neighborhoods(nside, 0), np.arange(npix)[:, None])

    one = ring_neighborhoods(nside, 1)
    two = ring_neighborhoods(nside, 2)
    for pix in range(npix):
        nb = hp.get_all_neighbours(nside, pix, nest=True)
        expected = np.unique(np.append(nb[nb >= 0], pix))
        assert np.array_equal(np.unique(one[pix]), expected)
        # the second ring is the first ring of the first ring
        rows = one[np.unique(one[pix])]
        assert np.array_equal(np.unique(two[pix]), np.unique(rows))
    with pytest.raises(ValueError):
        ring_neighborhoods(nside, -1)
