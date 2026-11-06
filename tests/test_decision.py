"""The pixel ledger and the decision rule (threshold test and plain threshold)."""

from __future__ import annotations

import pytest
import torch
from scipy.special import ndtri
from skimage.morphology import isotropic_dilation

from panther_em.inference.search.decision import (
    Calibration,
    Decision,
    PixelLabel,
    ThresholdTest,
)
from panther_em.inference.search.executor import StageMaps
from panther_em.inference.search.hypotheses import HypothesisSet, HypothesisSpace
from panther_em.inference.search.noise import NoiseScales, RadialTables
from panther_em.inference.search.results import PixelLedger
from panther_em.inference.template import Template
from panther_em.inference.search.types import (
    ImageSet,
    ImageSpectra,
    PixelSelection,
    SearchGeometry,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array_wrap__ must accept.*:DeprecationWarning",
    "ignore:.*__array__ implementation doesn't accept.*:DeprecationWarning",
)

RECTS = [(0, 0, 2, 3)]  # a low-rank selection of the synthetic (k 4, e 5) template
FULL = [(0, 0, 4, 5)]
AMBIGUOUS, REJECTED = int(PixelLabel.AMBIGUOUS), int(PixelLabel.REJECTED)
ACCEPTED, ABSORBED = int(PixelLabel.ACCEPTED), int(PixelLabel.ABSORBED)


def _geometry(b=2, h=24, w=24, box=20):
    images = ImageSet.from_stack(torch.zeros(b, h, w))
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


@pytest.fixture
def template(synthetic_result):
    return Template(synthetic_result(), device="cpu")


def _spectra(tables, geometry=None, *, noise=1.5, seed=0):
    """White noise curves at power ``noise``; with a geometry, a random level map."""
    level = None
    if geometry is not None:
        gen = torch.Generator().manual_seed(seed)
        b, (h, w) = geometry.n_images, geometry.image_shape
        level = 0.5 + torch.rand(b, h, w, generator=gen)
    curve = noise * torch.ones(tables.n_bins)
    return ImageSpectra.create(tables.freq, curve, level=level)


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

    boxes = ImageSet.from_boxes(torch.zeros(4, 20, 20))
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


def test_calibration_validation_and_z_alpha():
    cal = Calibration()
    assert cal.alpha == 5e-3 and cal.zeta == 1.0 and cal.eps_rho == 0.1
    assert cal.z_alpha == pytest.approx(-ndtri(5e-3))
    assert Calibration(alpha=0.1).z_alpha == pytest.approx(1.2815515655446004)
    for bad in (
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"zeta": 0.0},
        {"zeta": 1.1},
        {"eps_rho": -0.1},
        {"b_factor": -1.0},
    ):
        with pytest.raises(ValueError):
            Calibration(**bad)
    with pytest.raises(ValueError):
        ThresholdTest(object(), cal)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The threshold test
# --------------------------------------------------------------------------- #


def test_distance_matches_the_hand_computed_rule(template):
    tables = RadialTables.build(template, RECTS)
    geo = _geometry(b=2, h=24, w=24, box=20)  # (2, 5, 5): 50 pixels
    spectra = _spectra(tables, geo)
    cal = Calibration(alpha=1e-2, zeta=0.95, eps_rho=0.2)
    test = ThresholdTest(tables, cal)
    gen = torch.Generator().manual_seed(1)
    n = geo.n_pixels
    idx = torch.arange(n)
    mip = 3.0 * torch.rand(n, generator=gen) - 0.5
    mean = 0.1 * torch.randn(n, generator=gen)
    variance = 0.5 + torch.rand(n, generator=gen)
    best = torch.randint(0, 6, (n,), generator=gen)
    hyps = _hyps(template_ids=torch.tensor([0, 2, 3]))
    threshold = 2.5
    dist = test.distance(
        threshold, spectra, geo, idx, mip, mean, variance, best, hyps=hyps
    )
    assert dist.shape == (n,) and dist.dtype is torch.float32

    scales = tables.scales(spectra, hyps=hyps, b_factor=0.0)
    b = idx // 25
    r, c = (idx % 25) // 5 + 9, (idx % 25) % 5 + 9  # window centre of a 20-box
    ell = spectra.level[b, r, c].double()
    q_kept, q_perp = scales.q_kept[0, 0, best], scales.q_perp[0, 0, best]
    lam, rho = scales.lam[0, 0, best], scales.rho[0, 0, best]
    x = mip.double() - mean.double()
    a = rho / (1 - rho)
    sigma_hat = (variance.double() + ell * scales.qbar_perp[0]).sqrt()
    t_lo = mean.double() + threshold * cal.zeta * sigma_hat
    mean_p = mean.double() + x / (1 - rho)
    sd2 = (a - lam) ** 2 * ell * q_kept + ell * q_perp + (cal.eps_rho * a * x) ** 2
    expected = (t_lo - mean_p) / sd2.sqrt()
    torch.testing.assert_close(dist.double(), expected, rtol=1e-5, atol=1e-5)

    rejected, same = test.reject(
        threshold, spectra, geo, idx, mip, mean, variance, best, hyps=hyps
    )
    assert torch.equal(same, dist) and torch.equal(rejected, dist >= cal.z_alpha)
    assert rejected.any() and not rejected.all()
    # precomputed scales give the same distances; the hypotheses set qbar_perp
    again = test.distance(
        threshold, spectra, geo, idx, mip, mean, variance, best, scales=scales
    )
    torch.testing.assert_close(again, dist)
    assert test.scales(spectra).qbar_perp != scales.qbar_perp
    # an unsearched pixel has no distance
    best[3] = -1
    nan = test.distance(threshold, spectra, geo, idx, mip, mean, variance, best)
    assert torch.isnan(nan[3]) and torch.isfinite(nan[4])
    with pytest.raises(ValueError, match="template rows"):
        test.distance(
            threshold,
            spectra,
            geo,
            idx,
            mip,
            mean,
            variance,
            best,
            scales=_subset_scales(scales),
        )


def _subset_scales(scales):
    """Scales with one template row fewer (a mismatch)."""
    return NoiseScales(
        scales.q_kept[..., :-1].contiguous(),
        scales.q_residual[..., :-1].contiguous(),
        scales.q_cross[..., :-1].contiguous(),
        scales.lam[..., :-1].contiguous(),
        scales.q_perp[..., :-1].contiguous(),
        scales.rho[..., :-1].contiguous(),
        scales.qbar_perp,
        scales.rectangles,
    )


def test_full_rank_test_is_the_lowered_threshold(template):
    tables = RadialTables.build(template, FULL)
    geo = _geometry(b=1, h=30, w=30)
    ledger = PixelLedger(geo, "cpu")
    idx = torch.arange(geo.n_pixels)
    z = torch.linspace(-2.0, 12.0, geo.n_pixels)
    ledger.write(_maps(geo.n_pixels, zscore=z, mip=z), idx, 0, _hyps())
    spectra = _spectra(tables, geo)
    test = ThresholdTest(tables, Calibration(zeta=0.9))

    Decision(7.0, test).apply(ledger, idx, _hyps(), spectra)
    clear = (z - 6.3).abs() > 1e-3  # away from the lowered threshold
    assert torch.equal((ledger.label == REJECTED)[clear], (z < 6.3)[clear])
    assert torch.equal(ledger.label == ACCEPTED, z > 7.0)
    between = (z > 6.3 + 1e-3) & (z < 7.0)
    assert between.any() and torch.all(ledger.label[between] == AMBIGUOUS)
    assert torch.isfinite(ledger.score).all()

    Decision(7.0, test, accept=False).apply(ledger, idx, _hyps(), spectra)
    assert torch.equal((ledger.label == REJECTED)[clear], (z < 6.3)[clear])
    assert not (ledger.label == ACCEPTED).any()
    with pytest.raises(ValueError, match="spectra"):
        Decision(7.0, test).apply(ledger, idx, _hyps())


def test_low_rank_test_monotone_in_mip_and_calibration_directions(template):
    tables = RadialTables.build(template, RECTS)
    geo = _geometry(b=1, h=30, w=30)  # 121 pixels
    spectra = _spectra(tables)  # no level map: every pixel at level 1
    n = geo.n_pixels
    idx = torch.arange(n)
    mip = torch.linspace(0.0, 4.0, n)
    mean, variance = torch.zeros(n), torch.ones(n)
    best = torch.full((n,), 2)
    threshold = 20.0  # far above every mip: the predicted mean stays below T_lo

    def distance(**kwargs):
        test = ThresholdTest(tables, Calibration(**kwargs))
        return test.distance(threshold, spectra, geo, idx, mip, mean, variance, best)

    base = distance()
    assert torch.all(base > 0) and torch.all(base[1:] < base[:-1])  # falls with mip

    # the calibration constants move the rejections the way the note describes,
    # on random pixels around a threshold the test can reach
    gen = torch.Generator().manual_seed(2)
    mixed = {
        "mip": 3.0 * torch.rand(n, generator=gen) - 0.5,
        "mean": 0.1 * torch.randn(n, generator=gen),
        "variance": 0.5 + torch.rand(n, generator=gen),
        "best_index": torch.randint(0, 6, (n,), generator=gen),
    }

    def n_rejected(**kwargs):
        cal = Calibration(**kwargs)
        d = ThresholdTest(tables, cal).distance(
            2.5, spectra, geo, idx, **mixed
        )
        return int((d >= cal.z_alpha).sum())

    n_base = n_rejected()
    assert 0 < n_base < n  # some pixels rejected, some not
    assert n_rejected(alpha=1e-4) <= n_base <= n_rejected(alpha=1e-1)
    assert n_rejected(zeta=0.8) <= n_base
    assert n_rejected(eps_rho=0.5) <= n_base <= n_rejected(eps_rho=0.0)
    # the B-factor envelope changes the signal loss, hence the distances
    envelope = ThresholdTest(tables, Calibration(b_factor=30.0))
    s0, s30 = envelope.scales(spectra), ThresholdTest(tables).scales(spectra)
    assert not torch.allclose(s0.rho, s30.rho) and s0.b_factor == 30.0
    torch.testing.assert_close(
        s0.rho, tables.scales(spectra, b_factor=30.0).rho
    )
    d30 = envelope.distance(threshold, spectra, geo, idx, mip, mean, variance, best)
    assert not torch.allclose(d30, base)


def test_decision_scores_and_absorbs_near_accepted_pixels(template):
    tables = RadialTables.build(template, FULL)
    geo = _geometry(b=1, h=30, w=30)
    ledger = PixelLedger(geo, "cpu")
    idx = torch.arange(geo.n_pixels)
    z = torch.full((geo.n_pixels,), 6.8)  # between the lowered 6.3 and the threshold
    z[0] = 20.0  # one clear detection
    ledger.write(_maps(geo.n_pixels, zscore=z, mip=z), idx, 0, _hyps())
    spectra = _spectra(tables, geo)
    test = ThresholdTest(tables, Calibration(zeta=0.9))
    Decision(7.0, test, absorb_radius=1).apply(ledger, idx[:60], _hyps(), spectra)
    labels = ledger.label.reshape(geo.dense_shape)[0]
    assert labels[0, 0] == ACCEPTED
    assert labels[0, 1] == labels[1, 0] == ABSORBED
    assert labels[1, 1] == AMBIGUOUS  # outside the unit disk
    assert labels[5, 4] == AMBIGUOUS and labels[6, 0] == AMBIGUOUS  # untouched: 60+
    assert torch.isfinite(ledger.score[idx[:60]]).all()
    assert torch.isnan(ledger.score[idx[60:]]).all()
