"""Self-seeded stages: seed templates, lattices of seeded pixels, local entries."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from panther_em.orientations import healpix_ipix, healpix_level_ids
from panther_em.search.correlation import StageMaps
from panther_em.search.correlation.layout import FeatureLayout
from panther_em.search.correlation.reduction import hermitian_spectrum
from panther_em.search.hypotheses import HypothesisSet, HypothesisSpace
from panther_em.search.images import PixelSelection, SearchGeometry, SearchImages
from panther_em.search.seeding import (
    CellMeanWeights,
    LatticeSelection,
    Seeding,
    SeedResources,
    cell_mean_templates,
    seed_work,
)
from panther_em.search.template import Template

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array_wrap__ must accept.*:DeprecationWarning",
    "ignore:.*__array__ implementation doesn't accept.*:DeprecationWarning",
)


def test_seeding_validation():
    assert Seeding().n_candidates == 1
    assert Seeding(lattice=2).n_candidates == 4
    for bad in (
        {"level": 0},
        {"rings": -1},
        {"templates": "mean"},
        {"n_psi": 1},
        {"lattice": 0},
        {"top_k": 0},
    ):
        with pytest.raises(ValueError):
            Seeding(**bad)


# --------------------------------------------------------------------------- #
# Cell-mean templates
# --------------------------------------------------------------------------- #


def _correlogram(a, b, n_psi):
    """``corr(psi)`` of ``(K, M)`` weights ``a`` against ``b`` (kernel convention)."""
    g = torch.einsum("km,km->k", a, b.conj())
    return torch.fft.irfft(hermitian_spectrum(g, n_psi)[0], n=n_psi, norm="forward")


def test_cell_mean_of_in_plane_rotations_is_the_reference_member():
    """Members that are in-plane rotations of one template align back onto it."""
    gen = torch.Generator().manual_seed(1)
    k_count, m_count, n_psi = 6, 4, 32
    base = torch.complex(
        torch.randn(2, k_count, m_count, generator=gen),
        torch.randn(2, k_count, m_count, generator=gen),
    )  # one base template per cell
    k = torch.arange(k_count, dtype=torch.float64)
    psi = 2 * np.pi * torch.tensor([0, 5, 11, 20, 3, 30, 9, 17]) / n_psi
    cell_of = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    us = base[cell_of] * torch.exp(-1j * psi[:, None, None] * k[None, :, None]).to(
        torch.complex64
    )
    templates, cells = cell_mean_templates(us, cell_of, 2, n_psi)
    assert cells.tolist() == [0, 1]
    # cell c's template is its first member (every member aligns onto it exactly)
    torch.testing.assert_close(templates, us[[0, 4]], rtol=1e-5, atol=1e-5)
    # and every member's correlogram with it peaks at its rotation relative to the
    # cell's first member (the template's frame)
    first = {0: 0, 1: 3}
    for n in range(8):
        corr = _correlogram(us[n], templates[cell_of[n]], n_psi)
        want = round(float(psi[n]) * n_psi / (2 * np.pi)) - first[int(cell_of[n])]
        assert int(corr.argmax()) == want % n_psi


def test_cell_mean_handles_uneven_cells_filters_and_masks():
    gen = torch.Generator().manual_seed(2)
    o, k_count, m_count, n_psi = 6, 4, 5, 16
    us = torch.complex(
        torch.randn(2 * o, k_count, m_count, generator=gen),
        torch.randn(2 * o, k_count, m_count, generator=gen),
    )
    cell_of = torch.tensor([0, 0, 0, 1, 1, 2])  # cell 3 of 4 is empty
    mask = torch.ones(k_count, m_count, dtype=torch.bool)
    mask[:, 3:] = False
    templates, cells = cell_mean_templates(
        us, cell_of, 4, n_psi, n_filters=2, cell_mask=mask
    )
    assert cells.tolist() == [0, 1, 2]
    assert templates.shape == (6, k_count, m_count)
    assert torch.all(templates[..., 3:] == 0)
    # a one-member cell's template is the (masked) member, for either filter
    for f in range(2):
        member = torch.where(mask, us[f * o + 5], torch.zeros_like(us[0]))
        torch.testing.assert_close(templates[f * 3 + 2], member, rtol=1e-5, atol=1e-6)

    # energies: a template carries its members' mean self-energy
    def energy(w):
        p = (w.abs() ** 2).sum(-1)
        return p[..., 0] + 2 * p[..., 1:].sum(-1)

    masked = torch.where(mask, us[:o], torch.zeros_like(us[:o]))
    torch.testing.assert_close(
        energy(templates[0]), energy(masked[:3]).mean(), rtol=1e-4, atol=1e-4
    )
    with pytest.raises(ValueError, match="every orientation"):
        cell_mean_templates(us, cell_of, 4, n_psi, n_filters=1)


def test_cell_mean_weights_serve_blocks_in_their_box():
    templates = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5) * (1 + 0j)
    w = CellMeanWeights(
        templates,
        cells=torch.tensor([0, 2, 5]),
        n_filters=1,
        k_start=2,
        m_start=1,
        key=("test",),
    )
    assert w.n_templates == w.n_orientations == 3
    assert w.key == ("test",)
    block = w.block((3, 2, 2, 3), torch.tensor([2, 0]))
    assert torch.equal(block, templates[[2, 0], 1:3, 1:4])
    with pytest.raises(ValueError, match="outside"):
        w.block((0, 0, 2, 2))


# --------------------------------------------------------------------------- #
# Resources of a seeded stage
# --------------------------------------------------------------------------- #


def _healpix(synthetic_result, n_filters=2, n_defocus=2):
    template = Template(synthetic_result(s2_nside=2, n_filters=n_filters), device="cpu")
    space = HypothesisSpace.from_result(template.result, n_defocus=n_defocus)
    nside, ipix = healpix_ipix(space.phi_deg, space.theta_deg)
    return template, space, nside, ipix


@pytest.mark.parametrize("templates", ["cell_mean", "subset"])
def test_seed_resources_name_a_cell_per_seed_hypothesis(synthetic_result, templates):
    template, space, nside, ipix = _healpix(synthetic_result)
    layout = FeatureLayout(((0, 0, 3, 3), (1, 3, 3, 2)))
    spec = Seeding(level=1, rings=1, templates=templates)
    res = SeedResources.build(
        template.weights,
        layout,
        spec,
        space,
        nside,
        ipix,
        level=0,
        template_ids=None,
        n_psi=8,
    )
    assert res.n_cells == 12 and res.local.base_level == 1
    if templates == "cell_mean":
        assert res.weights is not None and res.template_ids is None
        assert res.space.n_orientations == 12 and res.space.n_filters == 2
        assert res.weights.n_templates == 24
        assert res.weights.k_start == 0 and res.weights.m_start == 0
        assert res.weights.k_stop == 4 and res.weights.m_stop == 5
        assert torch.equal(res.cells, torch.arange(12))
    else:
        assert res.weights is None and res.space is space
        assert torch.equal(
            res.template_ids,
            torch.cat([healpix_level_ids(ipix, 1), 48 + healpix_level_ids(ipix, 1)]),
        )
        assert torch.equal(res.cells, res.local.cell_of)
    # every seed hypothesis (plane, filter, orientation) names its cell
    hyps = HypothesisSet.create(res.space, res.template_ids, None, hyp_batch=8, n_psi=8)
    best = hyps.decode(torch.arange(hyps.n_virtual))
    o = res.space.decode(best)[2]
    cells = res.seed_cells(best)
    want = res.cells[o]
    assert torch.equal(cells, want)
    assert int(cells.max()) == 11 and int(cells.min()) == 0
    with pytest.raises(ValueError, match="exceed"):
        SeedResources.build(
            template.weights,
            layout,
            Seeding(level=1),
            space,
            nside,
            ipix,
            level=1,
            template_ids=healpix_level_ids(ipix, 1),
            n_psi=8,
        )


def test_cell_mean_resources_respect_the_layout_mask(synthetic_result):
    template, space, nside, ipix = _healpix(synthetic_result, n_filters=1)
    layout = FeatureLayout(((0, 0, 2, 2),))  # a corner of the 4 x 5 grid
    res = SeedResources.build(
        template.weights,
        layout,
        Seeding(level=1),
        space,
        nside,
        ipix,
        level=0,
        template_ids=None,
        n_psi=8,
    )
    assert res.weights is not None
    assert res.weights.us.shape == (12, 2, 2)  # only the bounding box is held
    assert res.weights.block((0, 0, 2, 2)).shape == (12, 2, 2)


# --------------------------------------------------------------------------- #
# Lattices
# --------------------------------------------------------------------------- #


def _geometry(h=26, w=28, box=20, n_images=1):
    images = SearchImages.from_stack(torch.zeros(n_images, h, w))
    return SearchGeometry.from_images(images, (box, box))  # grid (7, 9)


def test_lattice_candidates_are_the_bracketing_lattice_pixels():
    geometry = _geometry()
    pixels = PixelSelection.all(geometry)
    lat = LatticeSelection.build(geometry, pixels, 2)
    gh, gw = geometry.grid
    rows, cols = (
        torch.arange(gh)[:, None].expand(gh, gw),
        torch.arange(gw).expand(gh, gw),
    )
    on = ((rows % 2 == 0) & (cols % 2 == 0)).reshape(-1)
    assert torch.equal(lat.columns, torch.nonzero(on).squeeze(-1))
    assert lat.candidates.shape == (4, gh * gw)
    for p in range(gh * gw):
        r, c = divmod(p, gw)
        want = {
            (r2, c2)
            for r2 in (r - r % 2, r - r % 2 + 2)
            for c2 in (c - c % 2, c - c % 2 + 2)
            if r2 < gh and c2 < gw and r2 % 2 == 0 and c2 % 2 == 0
        }
        got = {divmod(int(lat.columns[i]), gw) for i in lat.candidates[:, p] if i >= 0}
        assert got == want, (r, c)
    assert int((lat.candidates >= 0).sum(0).min()) >= 1


def test_lattice_falls_back_to_the_pixel_itself():
    geometry = _geometry()
    gh, gw = geometry.grid
    mask = torch.ones(gh, gw, dtype=torch.bool)
    mask[0::2, 0::2] = False  # no lattice pixel is selected at all
    mask[3, 3] = True
    pixels = PixelSelection.from_mask(geometry, mask)
    lat = LatticeSelection.build(geometry, pixels, 2)
    assert torch.equal(lat.columns, torch.arange(len(pixels)))  # everyone seeds itself
    assert torch.equal(lat.candidates[0], torch.arange(len(pixels)))
    assert bool((lat.candidates[1:] < 0).all())
    # lattice 1 and center mode: identity
    for geo in (
        geometry,
        SearchGeometry.from_images(
            SearchImages.from_stack(torch.zeros(3, 20, 20)), (20, 20)
        ),
    ):
        px = PixelSelection.all(geo)
        one = LatticeSelection.build(geo, px, 1 if geo is geometry else 2)
        assert torch.equal(one.columns, torch.arange(len(px)))
        assert one.candidates.shape == (1, len(px))


def test_lattice_covers_several_images():
    geometry = _geometry(n_images=2)
    lat = LatticeSelection.build(geometry, PixelSelection.all(geometry), 2)
    per_image = geometry.n_pixels_per_image
    image_of_column = lat.columns // per_image
    for p in range(geometry.n_pixels):
        cands = lat.candidates[:, p]
        assert all(int(image_of_column[i]) == p // per_image for i in cands if i >= 0)


# --------------------------------------------------------------------------- #
# Entries of the local pass
# --------------------------------------------------------------------------- #


def test_seed_work_builds_distinct_entries_and_keeps_the_best_candidates(
    synthetic_result,
):
    template, space, nside, ipix = _healpix(synthetic_result, n_filters=1)
    layout = FeatureLayout(((0, 0, 3, 3),))
    geometry = _geometry()
    pixels = PixelSelection.all(geometry)
    for top_k in (None, 2, 1):
        spec = Seeding(level=1, rings=0, lattice=2, top_k=top_k)
        res = SeedResources.build(
            template.weights,
            layout,
            spec,
            space,
            nside,
            ipix,
            level=0,
            template_ids=None,
            n_psi=8,
        )
        lat = LatticeSelection.build(geometry, pixels, 2)
        hyps = HypothesisSet.create(res.space, None, None, hyp_batch=8, n_psi=8)
        gen = torch.Generator().manual_seed(int(top_k or 0))
        s = lat.n_seeded
        maps = StageMaps.empty(s, "cpu", hyps.n_evaluated)
        maps.best_virtual = torch.randint(0, hyps.n_virtual, (s,), generator=gen)
        maps.zscore = torch.randn(s, generator=gen)
        work = seed_work(res, lat, hyps, maps)
        assert work.columns is not None
        assert work.rows is res.local.rows
        # expected: per pixel, the cells of its valid candidates (top_k by z)
        cell_sub = res.seed_cells(hyps.decode(maps.best_virtual))
        want = set()
        for p in range(len(pixels)):
            cands = [int(i) for i in lat.candidates[:, p] if i >= 0]
            if top_k is not None:
                cands = sorted(cands, key=lambda i: -float(maps.zscore[i]))[:top_k]
            want |= {(p, int(cell_sub[i])) for i in cands}
        got = set(zip(work.columns.tolist(), work.cells.tolist(), strict=True))
        assert got == want
        assert len(got) == work.n_entries  # distinct
        assert set(work.columns.tolist()) == set(range(len(pixels)))
