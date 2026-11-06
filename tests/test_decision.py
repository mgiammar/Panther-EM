"""The pixel ledger and the decision rule (threshold test and plain threshold)."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from noise_case import BOX, noise_case
from skimage.morphology import isotropic_dilation

from panther_em.inference.search.correlation import StageMaps
from panther_em.inference.search.decision import Decision, PixelLabel
from panther_em.inference.search.hypotheses import HypothesisSet, HypothesisSpace
from panther_em.inference.search.noise_model import Calibration, ThresholdTest
from panther_em.inference.search.results import PixelLedger
from panther_em.inference.search.types import (
    PixelSelection,
    SearchGeometry,
    SearchImages,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array_wrap__ must accept.*:DeprecationWarning",
    "ignore:.*__array__ implementation doesn't accept.*:DeprecationWarning",
)

RECTS = [(0, 0, 3, 4)]
AMBIGUOUS, REJECTED = int(PixelLabel.AMBIGUOUS), int(PixelLabel.REJECTED)
ACCEPTED, ABSORBED = int(PixelLabel.ACCEPTED), int(PixelLabel.ABSORBED)


def _geometry(b=2, h=24, w=24, box=20):
    images = SearchImages.from_stack(torch.zeros(b, h, w))
    return SearchGeometry.from_images(images, (box, box))


def _maps(n, *, mip=None, zscore=None, seed=0):
    gen = torch.Generator().manual_seed(seed)
    maps = StageMaps.empty(n, "cpu", n_evaluated=10)
    maps.mip = (
        torch.randn(n, generator=gen)
        if mip is None
        else torch.as_tensor(mip, dtype=torch.float32)
    )
    maps.mean = torch.zeros(n)
    maps.variance = torch.ones(n)
    maps.zscore = (
        maps.mip.clone()
        if zscore is None
        else torch.as_tensor(zscore, dtype=torch.float32)
    )
    maps.best_virtual = torch.arange(n) % 6
    maps.best_psi = torch.arange(n) % 16
    return maps


def _hyps(n_psi=16, template_ids=None):
    return HypothesisSet.create(
        HypothesisSpace(1, 1, 6), template_ids, None, hyp_batch=6, n_psi=n_psi
    )


# --------------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------------- #


def test_ledger_write_masks_scores_and_overwrite():
    geo = _geometry()
    mask = torch.zeros(geo.dense_shape, dtype=torch.bool)
    mask[:, :3, :] = True
    initial = PixelSelection.from_mask(geo, mask)
    ledger = PixelLedger(geo, "cpu", initial=initial)
    masked = int((ledger.label == int(PixelLabel.MASKED)).sum())
    pending = int((ledger.label == int(PixelLabel.PENDING)).sum())
    assert masked == geo.n_pixels - len(initial) and pending == len(initial)

    idx = initial.index
    ledger.write(_maps(len(idx)), idx, 0, _hyps())
    assert torch.equal(ledger.pixels_with(PixelLabel.AMBIGUOUS), idx)
    assert ledger.n_psi_of_stage == [16] and torch.all(ledger.last_stage[idx] == 0)
    assert torch.isnan(ledger.mip[ledger.label == int(PixelLabel.MASKED)]).all()
    assert ledger.best_index[idx[1]] == 1  # virtual == global here
    assert torch.isnan(ledger.score).all()

    ledger.set_scores(idx[:4], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert ledger.score[idx[:4]].tolist() == [1.0, 2.0, 3.0, 4.0]
    sub = idx[:5]
    ledger.write(_maps(5, mip=[9.0] * 5), sub, 1, _hyps(n_psi=32))
    assert torch.all(ledger.last_stage[sub] == 1) and ledger.n_psi_of_stage == [16, 32]
    assert ledger.last_stage[idx[5]] == 0 and ledger.mip[sub].tolist() == [9.0] * 5
    assert torch.isnan(ledger.score[sub]).all()  # a re-scored pixel loses its score
    # the next stage re-scores AMBIGUOUS and ACCEPTED pixels, never the others
    ledger.set_labels(idx[:2], torch.tensor([3, 2], dtype=torch.int8))
    nxt = ledger.next_stage_pixels().index
    assert nxt.numel() == len(idx) - 1 and idx[0] in nxt and idx[1] not in nxt
    torch.testing.assert_close(
        ledger.std(), ledger.variance.clamp_min(0).sqrt(), equal_nan=True
    )


@pytest.mark.parametrize("radius", [1, 2, 3])
def test_absorb_matches_skimage_disk_and_respects_image_boundaries(radius):
    geo = _geometry(b=3, h=30, w=28)
    ledger = PixelLedger(geo, "cpu")
    gen = torch.Generator().manual_seed(radius)
    labels = torch.where(
        torch.rand(geo.n_pixels, generator=gen) < 0.05,
        torch.tensor(ACCEPTED),
        torch.tensor(AMBIGUOUS),
    ).to(torch.int8)
    labels[::7] = REJECTED
    ledger.label[:] = labels
    ledger.absorb_near_accepted(radius)

    grid = labels.reshape(geo.dense_shape).numpy()
    expected = labels.clone().reshape(geo.dense_shape)
    for b in range(geo.n_images):
        accepted = grid[b] == ACCEPTED
        if not accepted.any():  # skimage misbehaves on an empty plane (legacy guard)
            continue
        dilated = isotropic_dilation(accepted, radius=radius)
        absorbed = torch.from_numpy(dilated) & (expected[b] == AMBIGUOUS)
        expected[b][absorbed] = ABSORBED
    assert torch.equal(ledger.label.reshape(geo.dense_shape), expected)

    boxes = SearchImages.from_boxes(torch.zeros(4, 20, 20))
    led = PixelLedger(SearchGeometry.from_images(boxes, (20, 20)), "cpu")
    led.label[:] = torch.tensor([3, 1, 1, 1], dtype=torch.int8)
    led.absorb_near_accepted(radius)
    assert led.label.tolist() == [3, 1, 1, 1]


# --------------------------------------------------------------------------- #
# The plain threshold
# --------------------------------------------------------------------------- #


def test_plain_threshold_with_and_without_acceptance():
    geo = _geometry(b=1)
    ledger = PixelLedger(geo, "cpu")
    idx = torch.arange(geo.n_pixels)
    z = torch.linspace(-2.0, 12.0, geo.n_pixels)
    z[5] = 7.0  # exactly on the threshold
    ledger.write(_maps(geo.n_pixels, zscore=z), idx, 0, _hyps())

    Decision(7.0).apply(ledger, idx, _hyps())
    assert torch.equal(ledger.label == REJECTED, z < 7.0)
    assert torch.equal(ledger.label == ACCEPTED, z > 7.0)
    assert ledger.label[5] == AMBIGUOUS and int((ledger.label == AMBIGUOUS).sum()) == 1
    assert torch.isnan(ledger.score).all()  # no test ran

    Decision(7.0, accept=False).apply(ledger, idx, _hyps())
    assert torch.equal(ledger.label == REJECTED, z < 7.0)
    assert torch.equal(ledger.label == AMBIGUOUS, z >= 7.0)
    assert not (ledger.label == ACCEPTED).any()

    Decision(10.5).apply(ledger, idx, _hyps())
    assert int((ledger.label == ACCEPTED).sum()) == int((z > 10.5).sum())
    with pytest.raises(ValueError):
        Decision(1.0, absorb_radius=-1)
    with pytest.raises(ValueError):
        Decision(1.0, test=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="without a threshold test"):
        Decision(1.0).score(ledger, idx, _hyps(), None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The threshold test on the ledger
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def micrograph():
    """The noise case's table of ``RECTS``, its micrograph and the search geometry."""
    case = noise_case()
    table = case.table(RECTS)
    images = case.images(table)
    search = SearchImages.single(images.image, filters=images.ctf)
    return table, images, SearchGeometry.from_images(search, (BOX, BOX))


def _case_hyps():
    """3 planes x 6 template rows (the noise case's space), 2 rows, 2 planes."""
    return HypothesisSet.create(
        HypothesisSpace(3, 1, 6),
        torch.tensor([0, 2, 5]),
        torch.tensor([0, 2]),
        hyp_batch=6,
        n_psi=16,
    )


def _case_ledger(geo, hyps, *, z=None, scale=10.0, seed=1):
    """Ledger with random maps whose spread is on the scale of the noise scales."""
    n = geo.n_pixels
    gen = torch.Generator().manual_seed(seed)
    maps = StageMaps.empty(n, "cpu", n_evaluated=10)
    maps.mean = torch.randn(n, generator=gen)
    maps.variance = scale**2 * (0.5 + torch.rand(n, generator=gen))
    if z is None:
        z = 3.0 * torch.rand(n, generator=gen)
    maps.zscore = torch.as_tensor(z, dtype=torch.float32)
    maps.mip = maps.mean + maps.variance.sqrt() * maps.zscore
    maps.best_virtual = torch.randint(0, hyps.n_virtual, (n,), generator=gen)
    maps.best_psi = torch.zeros(n, dtype=torch.long)
    ledger = PixelLedger(geo, "cpu")
    idx = torch.arange(n)
    ledger.write(maps, idx, 0, hyps)
    return ledger, idx


def test_score_gathers_level_plane_and_row(micrograph):
    table, images, geo = micrograph
    hyps = _case_hyps()
    ledger, idx = _case_ledger(geo, hyps)
    ledger.best_index[idx[7]] = -1  # nothing found at this pixel
    test = ThresholdTest(table, Calibration(alpha=1e-2, zeta=0.9, eps_rho=0.2))
    decision = Decision(2.5, test, accept=True)
    score = decision.score(ledger, idx, hyps, images)

    out_w = geo.grid[1]
    r, c = idx // out_w + (BOX - 1) // 2, idx % out_w + (BOX - 1) // 2
    best = ledger.best_index[idx].clamp_min(0)
    scales = test.scales(images, rows=hyps.template_rows(), planes=(0, 2))
    expected = test.distance(
        scales,
        2.5,
        level=images.level[r, c],
        plane=best // 6,
        row=best % 6,
        mip=ledger.mip[idx],
        mean=ledger.mean[idx],
        variance=ledger.variance[idx],
    )
    assert torch.isnan(score[7]) and torch.isfinite(score[6])
    keep = torch.arange(idx.numel()) != 7
    assert torch.equal(score[keep], expected[keep])
    assert set((best // 6).unique().tolist()) == {0, 2}  # the stage's planes only

    decision.apply(ledger, idx, hyps, images)
    torch.testing.assert_close(ledger.score[idx], score, equal_nan=True)
    rejected = score >= test.calibration.z_alpha
    assert rejected.any() and not rejected.all()
    assert torch.equal(ledger.label[idx] == REJECTED, rejected)
    accepted = ~rejected & (ledger.zscore[idx] > 2.5)
    assert torch.equal(ledger.label[idx] == ACCEPTED, accepted)
    assert ledger.label[idx[7]] != REJECTED  # NaN never rejects

    with pytest.raises(ValueError, match="noise model"):
        decision.apply(ledger, idx, hyps)
    stack = SearchGeometry.from_images(
        SearchImages.from_stack(torch.zeros(2, *images.image_shape)), (BOX, BOX)
    )
    with pytest.raises(ValueError, match="not the noise model"):
        decision.score(PixelLedger(stack, "cpu"), idx[:3], hyps, images)


def _full_rank(table):
    """The table with no missed part: ``q_perp = lam = rho = 0``."""
    zeros = torch.zeros_like(table.residual_autocorr)
    return dataclasses.replace(
        table,
        residual_autocorr=zeros,
        reconstruction_residual_corr=zeros,
        signal_residual_corr=zeros,
    )


def test_full_rank_test_is_the_lowered_threshold(micrograph):
    table, images, geo = micrograph
    hyps = _case_hyps()
    z = torch.linspace(-2.0, 12.0, geo.n_pixels)
    ledger, idx = _case_ledger(geo, hyps, z=z)
    test = ThresholdTest(_full_rank(table), Calibration(zeta=0.9))

    Decision(7.0, test).apply(ledger, idx, hyps, images)
    clear = (z - 6.3).abs() > 1e-3  # away from the lowered threshold
    assert torch.equal((ledger.label == REJECTED)[clear], (z < 6.3)[clear])
    assert torch.equal(ledger.label == ACCEPTED, z > 7.0)
    between = (z > 6.3 + 1e-3) & (z < 7.0)
    assert between.any() and torch.all(ledger.label[between] == AMBIGUOUS)
    assert torch.isfinite(ledger.score).all()

    Decision(7.0, test, accept=False).apply(ledger, idx, hyps, images)
    assert torch.equal((ledger.label == REJECTED)[clear], (z < 6.3)[clear])
    assert not (ledger.label == ACCEPTED).any()


def test_decision_scores_and_absorbs_near_accepted_pixels(micrograph):
    table, images, geo = micrograph
    hyps = _case_hyps()
    z = torch.full((geo.n_pixels,), 6.8)  # between the lowered 6.3 and the threshold
    z[0] = 20.0  # one clear detection
    ledger, idx = _case_ledger(geo, hyps, z=z)
    test = ThresholdTest(_full_rank(table), Calibration(zeta=0.9))
    n = 2 * geo.grid[1]  # the first two rows of the grid
    Decision(7.0, test, absorb_radius=1).apply(ledger, idx[:n], hyps, images)
    labels = ledger.label.reshape(geo.dense_shape)[0]
    assert labels[0, 0] == ACCEPTED
    assert labels[0, 1] == labels[1, 0] == ABSORBED
    assert labels[1, 1] == AMBIGUOUS  # outside the unit disk
    assert labels[2, 0] == AMBIGUOUS and labels[5, 4] == AMBIGUOUS  # untested
    assert torch.isfinite(ledger.score[idx[:n]]).all()
    assert torch.isnan(ledger.score[idx[n:]]).all()
