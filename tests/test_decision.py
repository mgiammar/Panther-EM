"""The pixel ledger and the decision rule."""

from __future__ import annotations

import pytest
import torch
from skimage.morphology import isotropic_dilation

from panther_em.inference.search.decision import (
    Decision,
    PixelLabel,
    classify_zscore_interval,
    zscore_bound_is_vacuous,
    zscore_error_intervals,
)
from panther_em.inference.search.executor import StageMaps
from panther_em.inference.search.hypotheses import HypothesisSet, HypothesisSpace
from panther_em.inference.search.results import PixelLedger
from panther_em.inference.search.stage import FixedError, Stage
from panther_em.inference.search.types import ImageSet, PixelSelection, SearchGeometry


def _geometry(b=2, h=24, w=24, box=20):
    return SearchGeometry.from_images(ImageSet.from_stack(torch.zeros(b, h, w)), (box, box))


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
        maps.mip.clone() if zscore is None else torch.as_tensor(zscore, dtype=torch.float32)
    )
    maps.best_virtual = torch.arange(n) % 6
    maps.best_psi = torch.arange(n) % 16
    return maps


def _hyps(n_psi=16):
    return HypothesisSet.create(HypothesisSpace(1, 1, 6), None, None, hyp_batch=6, n_psi=n_psi)


STAGE = Stage.create([(0, 0, 4, 4)], n_psi=16)


def test_ledger_write_masks_and_overwrite():
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

    sub = idx[:5]
    ledger.write(_maps(5, mip=[9.0] * 5), sub, 1, _hyps(n_psi=32))
    assert torch.all(ledger.last_stage[sub] == 1) and ledger.n_psi_of_stage == [16, 32]
    assert ledger.last_stage[idx[5]] == 0 and ledger.mip[sub].tolist() == [9.0] * 5
    # the next stage re-scores AMBIGUOUS and ACCEPTED pixels, never the others
    ledger.set_labels(idx[:2], torch.tensor([3, 2], dtype=torch.int8))
    nxt = ledger.next_stage_pixels().index
    assert nxt.numel() == len(idx) - 1 and idx[0] in nxt and idx[1] not in nxt
    torch.testing.assert_close(ledger.std(), ledger.variance.clamp_min(0).sqrt(), equal_nan=True)


@pytest.mark.parametrize("radius", [1, 2, 3])
def test_absorb_matches_skimage_disk_and_respects_image_boundaries(radius):
    geo = _geometry(b=3, h=30, w=28)
    ledger = PixelLedger(geo, "cpu")
    gen = torch.Generator().manual_seed(radius)
    labels = torch.where(
        torch.rand(geo.n_pixels, generator=gen) < 0.05,
        torch.tensor(int(PixelLabel.ACCEPTED)),
        torch.tensor(int(PixelLabel.AMBIGUOUS)),
    ).to(torch.int8)
    labels[::7] = int(PixelLabel.REJECTED)
    ledger.label[:] = labels
    ledger.absorb_near_accepted(radius)

    grid = labels.reshape(geo.dense_shape).numpy()
    expected = labels.clone().reshape(geo.dense_shape)
    for b in range(geo.n_images):
        accepted = grid[b] == int(PixelLabel.ACCEPTED)
        if not accepted.any():  # skimage misbehaves on an empty plane (legacy guard)
            continue
        dilated = isotropic_dilation(accepted, radius=radius)
        absorbed = torch.from_numpy(dilated) & (expected[b] == int(PixelLabel.AMBIGUOUS))
        expected[b][absorbed] = int(PixelLabel.ABSORBED)
    assert torch.equal(ledger.label.reshape(geo.dense_shape), expected)

    center = SearchGeometry.from_images(ImageSet.from_boxes(torch.zeros(4, 20, 20)), (20, 20))
    led = PixelLedger(center, "cpu")
    led.label[:] = torch.tensor([3, 1, 1, 1], dtype=torch.int8)
    led.absorb_near_accepted(radius)
    assert led.label.tolist() == [3, 1, 1, 1]


def test_decision_intervals_vacuous_guard_and_plain_threshold():
    geo = _geometry(b=1)
    ledger = PixelLedger(geo, "cpu")
    idx = torch.arange(geo.n_pixels)
    z = torch.linspace(-2.0, 12.0, geo.n_pixels)
    ledger.write(_maps(geo.n_pixels, zscore=z), idx, 0, _hyps())

    Decision(7.0, FixedError(0.05)).apply(ledger, idx, STAGE)
    expected = classify_zscore_interval(
        zscore_error_intervals(0.05, z, torch.zeros_like(z), torch.ones_like(z)), 7.0
    )
    assert torch.equal(ledger.label[idx], expected)
    for label in (PixelLabel.ACCEPTED, PixelLabel.REJECTED, PixelLabel.AMBIGUOUS):
        assert (ledger.label == int(label)).any()

    assert zscore_bound_is_vacuous(0.5) and not zscore_bound_is_vacuous(0.3)
    with pytest.warns(UserWarning, match="vacuous"):
        Decision(7.0, FixedError(0.6)).apply(ledger, idx, STAGE)
    assert torch.all(ledger.label[idx] == int(PixelLabel.AMBIGUOUS))

    # no bound: the interval collapses to the z-score and every pixel is decided
    Decision(7.0).apply(ledger, idx, STAGE)
    assert not (ledger.label == int(PixelLabel.AMBIGUOUS)).any()
    assert int((ledger.label == int(PixelLabel.ACCEPTED)).sum()) == int((z > 7.0).sum())
    Decision(10.5).apply(ledger, idx, STAGE)
    assert int((ledger.label == int(PixelLabel.ACCEPTED)).sum()) == int((z > 10.5).sum())
    with pytest.raises(ValueError):
        Decision(1.0, absorb_radius=-1)


def test_decision_absorbs_ambiguous_pixels_near_accepted_ones():
    geo = _geometry(b=1, h=30, w=30)
    ledger = PixelLedger(geo, "cpu")
    idx = torch.arange(geo.n_pixels)
    z = torch.full((geo.n_pixels,), 6.8)  # interval [6.3, 7.3] straddles 7
    z[0] = 20.0  # one clear detection; everything else sits inside the interval
    ledger.write(_maps(geo.n_pixels, zscore=z), idx, 0, _hyps())
    Decision(7.0, FixedError(0.05), absorb_radius=1).apply(ledger, idx, STAGE)
    labels = ledger.label.reshape(geo.dense_shape)[0]
    assert labels[0, 0] == int(PixelLabel.ACCEPTED)
    assert labels[0, 1] == labels[1, 0] == int(PixelLabel.ABSORBED)
    assert labels[1, 1] == int(PixelLabel.AMBIGUOUS)  # outside the unit disk
    assert labels[5, 5] == int(PixelLabel.AMBIGUOUS)
