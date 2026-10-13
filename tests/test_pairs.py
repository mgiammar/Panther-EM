"""PairList operations, dilation and the block collector."""

from __future__ import annotations

import pytest
import torch

from panther_em.inference.search.hypotheses import HypothesisPlan, HypothesisSpace
from panther_em.inference.search.ledger import PixelLabel, PixelLedger
from panther_em.inference.search.orientations import (
    S2Neighborhood,
    healpix_orientations,
)
from panther_em.inference.search.pairs import (
    PairCollector,
    PairEmissionSpec,
    PairList,
    dilate_pairs,
)
from panther_em.inference.search.types import ImageSet, PixelSelection, SearchGeometry

pytest.importorskip("healpy")


def _geometry(n_images=2, h=24, w=23, box=20):
    images = ImageSet.from_stack(torch.zeros(n_images, h, w))
    return SearchGeometry.from_images(images, (box, box))  # grid (5, 4) per image


def _pairs(pixel, template, n_templates, score=None):
    pixel, template = torch.tensor(pixel), torch.tensor(template)
    score = torch.ones(len(pixel)) if score is None else torch.tensor(score)
    return PairList(pixel, template, n_templates, score)


def _s2(nside=1):
    phi, theta = healpix_orientations(nside)
    s2 = S2Neighborhood.from_space(
        HypothesisSpace(1, 1, len(phi), torch.tensor(phi), torch.tensor(theta))
    )
    assert s2 is not None
    return s2


def test_pairlist_key_unique_concat_select_and_io(tmp_path):
    pairs = _pairs([3, 3, 1, 3], [7, 7, 2, 1], 10, [0.5, 0.9, 0.1, 0.2])
    assert len(pairs) == 4 and pairs.key().tolist() == [37, 37, 12, 31]
    u = pairs.unique()
    assert u.key().tolist() == [12, 31, 37]
    assert u.score.tolist() == pytest.approx([0.1, 0.2, 0.9])
    assert u.pixels().tolist() == [1, 3]
    other = PairList.from_key(torch.tensor([99, 12]), 10, torch.tensor([1.0, 5.0]))
    both = PairList.concat([u, other]).unique()
    assert both.key().tolist() == [12, 31, 37, 99]
    assert both.score.tolist() == pytest.approx([5.0, 0.2, 0.9, 1.0])
    assert both.select(both.pixel == 3).key().tolist() == [31, 37]
    both.save(tmp_path / "pairs.h5")
    loaded = PairList.load(tmp_path / "pairs.h5")
    assert torch.equal(loaded.key(), both.key()) and torch.equal(
        loaded.score, both.score
    )
    assert loaded.n_templates == 10 and len(loaded.to("cpu")) == 4
    with pytest.raises(ValueError):
        PairList(torch.tensor([1]), torch.tensor([1, 2]), 10, torch.ones(1))
    with pytest.raises(ValueError):
        PairList.concat([u, _pairs([0], [0], 3)])
    with pytest.raises(ValueError):
        PairEmissionSpec(1.0, pair_dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        PairEmissionSpec(1.0, pixel_radius=-1)


def test_pixel_dilation_clips_borders_and_never_crosses_images():
    geo = _geometry()  # (B=2, 5, 4): 20 pixels per image
    per = geo.n_pixels_per_image
    # image-0 corner, image-1 far corner, image-0 interior (r=2, c=1)
    seeds = _pairs([0, per + 19, 9], [4, 4, 4], 6, [1.0, 2.0, 3.0])
    out = dilate_pairs(seeds, geo, 1, None)
    assert torch.all(out.template == 4)
    corner0 = {0, 1, 4, 5}
    corner1 = {per + 19, per + 18, per + 15, per + 14}
    interior = {r * 4 + c for r in (1, 2, 3) for c in (0, 1, 2)}
    assert set(out.pixel.tolist()) == corner0 | corner1 | interior
    score_of = dict(zip(out.pixel.tolist(), out.score.tolist(), strict=True))
    assert all(score_of[p] == 3.0 for p in interior - corner0)
    assert score_of[0] == 1.0 and score_of[4] == 3.0  # overlap keeps the best seed
    assert torch.equal(dilate_pairs(seeds, geo, 0, None).key(), seeds.unique().key())
    wide = dilate_pairs(seeds, geo, 2, None)
    for p in wide.pixel.tolist():
        if p < per:  # within Chebyshev radius 2 of (0, 0) or (2, 1)
            r, c = divmod(p, 4)
            assert max(r, c) <= 2 or (abs(r - 2) <= 2 and abs(c - 1) <= 2)
    center = SearchGeometry.from_images(
        ImageSet.from_boxes(torch.zeros(3, 20, 20)), (20, 20)
    )
    assert dilate_pairs(_pairs([1], [0], 6), center, 2, None).pixel.tolist() == [1]


def test_orientation_dilation_follows_the_ring_table():
    geo = _geometry(n_images=1)
    s2 = _s2(1)  # 12 orientations; two filters -> n_templates = 24
    table = s2.neighbor_table(1)
    seeds = _pairs([6, 6], [2, 12 + 9], 24)
    out = dilate_pairs(seeds, geo, 0, table)
    expected = {
        (6, ff * 12 + o2)
        for ff, o in ((0, 2), (1, 9))
        for o2 in table[o][table[o] >= 0].tolist()
    }
    assert set(zip(out.pixel.tolist(), out.template.tolist(), strict=True)) == expected
    for o in (2, 9):  # every dilated orientation is a grid neighbour
        for o2 in table[o][table[o] >= 0].tolist():
            assert s2.ipix[o2] in s2.grid.neighbor_indices(int(s2.ipix[o]), 0, 1)
    # both dilations at once: 9 pixels x ring, same filter
    both = dilate_pairs(_pairs([5], [2], 24), geo, 1, table)
    assert len(both) == 9 * int((table[2] >= 0).sum())


def test_collector_thresholds_decodes_dilates_and_respects_the_mask():
    geo = _geometry(n_images=1)  # 20 pixels
    space = HypothesisSpace(n_defocus=2, n_filters=1, n_orientations=12)
    plan = HypothesisPlan(space, None, hyp_batch=4, n_psi=8)
    assert plan.n_virtual == 24  # 2 planes x 12 templates
    mask = torch.ones(geo.dense_shape, dtype=torch.bool)
    mask[0, 0, 2] = False  # pixel 2 is MASKED: dilation must not reach it
    ledger = PixelLedger(
        geo,
        "cpu",
        initial=PixelSelection.from_mask(geo, mask),
        aux={"image_norm": torch.ones(20), "noise_variance": torch.ones(20)},
    )
    column_index = torch.tensor([19, 7, 3, 0, 12])  # compact order, not sorted
    spec = PairEmissionSpec(2.0, pair_dtype=torch.float32)
    s2 = _s2(1)
    col = PairCollector(spec, ledger, column_index, plan, rank=20, s2=s2, device="cpu")
    slab = torch.zeros(5, 24)
    mean, var = torch.zeros(5), torch.ones(5)
    slab[1, 5] = 3.0  # pixel 7, plane 0, template 5 (z = 3 > 2)
    slab[1, 12 + 5] = 2.5  # pixel 7, plane 1, same template: same pair, lower score
    slab[2, 12 + 2] = 10.0  # pixel 3, plane 1, template 2 (neighbour of masked pixel 2)
    slab[0, 0] = 1.0  # below threshold
    col.consume_block(slab[:3], col_start=0, n_valid=3, mean=mean[:3], variance=var[:3])
    col.consume_block(slab[3:], col_start=3, n_valid=2, mean=mean[3:], variance=var[3:])
    assert col.n_raw == 3
    pairs = col.finish()
    seeds = _pairs([7, 3], [5, 2], 12, [3.0, 10.0])
    ref = dilate_pairs(seeds, geo, 1, s2.neighbor_table(1))
    ref = ref.select(ref.pixel != 2)
    assert torch.equal(pairs.key(), ref.key())
    torch.testing.assert_close(pairs.score, ref.score)
    assert pairs.score[pairs.key() == 7 * 12 + 5].tolist() == [3.0]  # max over planes
    assert 2 not in pairs.pixels().tolist()
    assert ledger.label[2] == int(PixelLabel.MASKED)

    bare = PairCollector(
        PairEmissionSpec(2.0, pixel_radius=0, orientation_rings=0),
        ledger,
        column_index,
        plan,
        rank=20,
        s2=None,
        device="cpu",
    )
    bare.consume_block(slab, col_start=0, n_valid=5, mean=mean, variance=var)
    assert bare.finish().key().tolist() == sorted([7 * 12 + 5, 3 * 12 + 2])
    assert len(bare.finish()) == 0  # nothing pending after a finish
    with pytest.raises(ValueError, match="image_norm"):
        PairCollector(
            spec,
            PixelLedger(geo, "cpu"),
            column_index,
            plan,
            rank=20,
            s2=None,
            device="cpu",
        )
