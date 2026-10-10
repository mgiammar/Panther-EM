"""PixelLedger and the concrete decisions."""

from __future__ import annotations

import pytest
import torch
from skimage.morphology import isotropic_dilation

from panther_em.inference.search.decisions import (
    DecisionChain,
    DilationAbsorb,
    NoDecision,
    ThresholdClassifier,
    TopFractionSelector,
    ZScoreIntervalClassifier,
    classify_zscore_interval,
    zscore_bound_is_vacuous,
    zscore_error_intervals,
)
from panther_em.inference.search.executor import StageMaps
from panther_em.inference.search.hypotheses import HypothesisPlan, HypothesisSpace
from panther_em.inference.search.ledger import PixelLabel, PixelLedger
from panther_em.inference.search.schedule import StageSpec
from panther_em.inference.search.types import ImageSet, PixelSelection, SearchGeometry


def _geometry(b=2, h=24, w=24, box=20):
    return SearchGeometry.from_images(
        ImageSet.from_stack(torch.zeros(b, h, w)), (box, box)
    )


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


def _plan(n_psi=16):
    return HypothesisPlan(HypothesisSpace(1, 1, 6), None, hyp_batch=6, n_psi=n_psi)


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
    ledger.write(_maps(len(idx)), idx, 0, _plan())
    assert torch.equal(ledger.pixels_with(PixelLabel.UNCERTAIN), idx)
    assert ledger.n_psi_of_stage == [16] and torch.all(ledger.last_stage[idx] == 0)
    assert torch.isnan(ledger.mip[ledger.label == int(PixelLabel.MASKED)]).all()
    assert ledger.best_index[idx[1]] == 1  # virtual == global here

    sub = idx[:5]
    ledger.write(_maps(5, mip=[9.0] * 5), sub, 1, _plan(n_psi=32))
    assert torch.all(ledger.last_stage[sub] == 1) and ledger.n_psi_of_stage == [16, 32]
    assert ledger.last_stage[idx[5]] == 0 and ledger.mip[sub].tolist() == [9.0] * 5
    assert ledger.grid("mip").shape == geo.dense_shape
    assert ledger.uncertain().index.numel() == len(idx)
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
        torch.tensor(int(PixelLabel.ACCEPTED)),
        torch.tensor(int(PixelLabel.UNCERTAIN)),
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
        absorbed = torch.from_numpy(dilated) & (
            expected[b] == int(PixelLabel.UNCERTAIN)
        )
        expected[b][absorbed] = int(PixelLabel.ABSORBED)
    assert torch.equal(ledger.label.reshape(geo.dense_shape), expected)

    center = SearchGeometry.from_images(
        ImageSet.from_boxes(torch.zeros(4, 20, 20)), (20, 20)
    )
    led = PixelLedger(center, "cpu")
    led.label[:] = torch.tensor([3, 1, 1, 1], dtype=torch.int8)
    led.absorb_near_accepted(radius)
    assert led.label.tolist() == [3, 1, 1, 1]


def test_zscore_interval_classifier_relative_units_and_vacuous_guard():
    geo = _geometry(b=1)
    ledger = PixelLedger(geo, "cpu")
    idx = torch.arange(geo.n_pixels)
    z = torch.linspace(-2.0, 12.0, geo.n_pixels)
    ledger.write(_maps(geo.n_pixels, zscore=z), idx, 0, _plan())
    spec = StageSpec([(0, 0, 4, 4)], n_psi=16, error=0.05)

    ZScoreIntervalClassifier(7.0)(ledger, idx, spec, 0)
    expected = classify_zscore_interval(
        zscore_error_intervals(0.05, z, torch.zeros_like(z), torch.ones_like(z)), 7.0
    )
    assert torch.equal(ledger.label[idx], expected)
    assert (ledger.label == int(PixelLabel.ACCEPTED)).any()
    assert (ledger.label == int(PixelLabel.REJECTED)).any()
    assert (ledger.label == int(PixelLabel.UNCERTAIN)).any()

    assert zscore_bound_is_vacuous(0.5) and not zscore_bound_is_vacuous(0.3)
    with pytest.warns(UserWarning, match="vacuous"):
        ZScoreIntervalClassifier(7.0, error=0.6)(ledger, idx, spec, 0)
    assert torch.all(ledger.label[idx] == int(PixelLabel.UNCERTAIN))

    # a zero error collapses every interval to a point: every pixel is decided
    ZScoreIntervalClassifier(7.0, error=0.0)(ledger, idx, spec, 0)
    assert not (ledger.label == int(PixelLabel.UNCERTAIN)).any()

    with pytest.raises(ValueError):
        ZScoreIntervalClassifier(7.0)(
            ledger, idx, StageSpec([(0, 0, 4, 4)], n_psi=16), 0
        )
    with pytest.raises(ValueError):
        ZScoreIntervalClassifier(7.0, units="bogus")  # type: ignore[arg-type]


def test_absolute_units_reproduce_legacy_tracker_math():
    geo = _geometry(b=1)
    ledger = PixelLedger(geo, "cpu")
    idx = torch.arange(geo.n_pixels)
    maps = _maps(geo.n_pixels)
    maps.mean = torch.full((geo.n_pixels,), 0.1)
    maps.variance = torch.full((geo.n_pixels,), 0.25)
    ledger.write(maps, idx, 0, _plan())
    spec = StageSpec([(0, 0, 4, 4)], n_psi=16, error=0.02)
    ZScoreIntervalClassifier(1.0, units="absolute")(ledger, idx, spec, 0)
    expected = classify_zscore_interval(
        zscore_error_intervals(0.02, maps.mip, maps.mean, maps.variance.sqrt()), 1.0
    )
    assert torch.equal(ledger.label[idx], expected)


def test_threshold_top_fraction_chain_and_no_decision():
    geo = _geometry(b=2)
    ledger = PixelLedger(geo, "cpu")
    idx = torch.arange(geo.n_pixels)
    z = torch.arange(geo.n_pixels, dtype=torch.float32)
    ledger.write(_maps(geo.n_pixels, zscore=z), idx, 0, _plan())
    spec = StageSpec([(0, 0, 4, 4)], n_psi=16)

    NoDecision()(ledger, idx, spec, 0)
    assert torch.all(ledger.label == int(PixelLabel.UNCERTAIN))

    TopFractionSelector(fraction=0.1)(ledger, idx, spec, 0)
    kept = ledger.pixels_with(PixelLabel.UNCERTAIN)
    assert (
        kept.numel() == round(0.1 * geo.n_pixels)
        and kept.min() >= geo.n_pixels - kept.numel()
    )

    ledger.label[:] = int(PixelLabel.UNCERTAIN)
    TopFractionSelector(top_k=3, per_image=True)(ledger, idx, spec, 0)
    kept = ledger.pixels_with(PixelLabel.UNCERTAIN)
    per = geo.n_pixels_per_image
    assert sorted(kept.tolist()) == [
        per - 3,
        per - 2,
        per - 1,
        2 * per - 3,
        2 * per - 2,
        2 * per - 1,
    ]

    ledger.label[:] = int(PixelLabel.UNCERTAIN)
    ThresholdClassifier(10.5)(ledger, idx, spec, 0)
    assert int((ledger.label == int(PixelLabel.ACCEPTED)).sum()) == geo.n_pixels - 11
    assert not (ledger.label == int(PixelLabel.UNCERTAIN)).any()

    chain = DecisionChain(ZScoreIntervalClassifier(7.0), DilationAbsorb(1))
    assert chain.pointwise is False and chain.requires_error is True
    assert DecisionChain(ThresholdClassifier(1.0)).pointwise is True
    ledger.label[:] = int(PixelLabel.UNCERTAIN)
    chain(ledger, idx, StageSpec([(0, 0, 4, 4)], n_psi=16, error=0.01), 0)
    assert (ledger.label == int(PixelLabel.ABSORBED)).any()
    with pytest.raises(ValueError):
        TopFractionSelector()
    with pytest.raises(ValueError):
        TopFractionSelector(fraction=0.5, top_k=2)
    with pytest.raises(ValueError):
        DilationAbsorb(-1)
    assert "ThresholdClassifier" in repr(chain) or "ZScoreIntervalClassifier" in repr(
        chain
    )


def test_surpass_threshold_broadcasting_and_validation():
    from panther_em.inference.search.decisions import surpass_threshold

    corr = torch.tensor([1.0, 5.0, 9.0])
    mean = torch.zeros(3)
    var = torch.full((3,), 4.0)
    out = surpass_threshold(
        corr,
        mean,
        var,
        threshold=2.0,
        image_norm=torch.ones(3),
        noise_variance=1.0,
        rank=64,
    )
    assert out.dtype == torch.bool and out.tolist() == [
        False,
        True,
        True,
    ]  # z = 0.5, 2.5, 4.5

    pairs = torch.tensor([[1.0, 5.0], [9.0, 3.0], [0.0, 0.0]])
    out2 = surpass_threshold(
        pairs,
        mean,
        var,
        threshold=2.0,
        image_norm=1.0,
        noise_variance=torch.ones(3),
        rank=64,
    )
    assert out2.shape == (3, 2) and out2.tolist() == [
        [False, True],
        [True, False],
        [False, False],
    ]
    with pytest.raises(ValueError):
        surpass_threshold(
            corr,
            mean,
            var,
            threshold=2.0,
            image_norm=torch.ones(4),
            noise_variance=1.0,
            rank=64,
        )
    with pytest.raises(ValueError):
        surpass_threshold(
            corr, mean, var, threshold=2.0, image_norm=1.0, noise_variance=1.0, rank=0
        )
    with pytest.raises(ValueError):
        surpass_threshold(
            corr[None, None],
            mean,
            var,
            threshold=2.0,
            image_norm=1.0,
            noise_variance=1.0,
            rank=1,
        )


def test_surpass_threshold_decision_reads_aux_maps_and_labels():
    from panther_em.inference.search.decisions import SurpassThresholdDecision

    geo = _geometry(b=1)
    aux = {
        "image_norm": torch.ones(geo.dense_shape),
        "noise_variance": torch.full(geo.dense_shape, 0.5),
    }
    ledger = PixelLedger(geo, "cpu", aux=aux)
    assert ledger.aux["noise_variance"].shape == (geo.n_pixels,)
    assert torch.equal(ledger.grid("image_norm"), aux["image_norm"])
    idx = torch.arange(geo.n_pixels)
    z = torch.linspace(-1.0, 9.0, geo.n_pixels)
    ledger.write(_maps(geo.n_pixels, mip=z), idx, 0, _plan())  # mean 0, variance 1
    spec = StageSpec([(0, 0, 4, 4)], n_psi=16)

    SurpassThresholdDecision(4.0)(ledger, idx, spec, 0)
    assert torch.equal(ledger.label[idx] == int(PixelLabel.ACCEPTED), z > 4.0)
    assert torch.equal(ledger.label[idx] == int(PixelLabel.REJECTED), z <= 4.0)

    follow = SurpassThresholdDecision(4.0, pass_label=PixelLabel.UNCERTAIN)
    follow(ledger, idx, spec, 0)
    assert torch.equal(ledger.uncertain().index, idx[z > 4.0])
    assert "SurpassThresholdDecision" in repr(follow)

    bare = PixelLedger(geo, "cpu")
    with pytest.raises(KeyError, match="image_norm"):
        SurpassThresholdDecision(4.0)(bare, idx, spec, 0)
    with pytest.raises(ValueError):
        PixelLedger(geo, "cpu", aux={"image_norm": torch.ones(3)})
