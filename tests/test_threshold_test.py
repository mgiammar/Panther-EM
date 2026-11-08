"""noise_model.ThresholdTest: the scales, the per-pixel rule and the helpers."""

from __future__ import annotations

import pytest
import torch
from noise_case import PIXEL_SIZE, noise_case
from scipy.special import ndtri

from panther_em.inference.search.noise_model import (
    Calibration,
    NoiseScales,
    ThresholdTest,
    zeta_quantile,
    zscore_threshold,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array_wrap__ must accept.*:DeprecationWarning",
    "ignore:.*__array__ implementation doesn't accept.*:DeprecationWarning",
)

RECTS = [(0, 0, 3, 4)]


@pytest.fixture(scope="module")
def case():
    case = noise_case()
    table = case.table(RECTS)
    return case, table, case.images(table)


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
    with pytest.raises(ValueError, match="PerSpatialFrequencyErrorTable"):
        ThresholdTest(object(), cal)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Calibration"):
        ThresholdTest(noise_case().table(RECTS), object())  # type: ignore[arg-type]


def test_scales_are_the_curves_weighted_by_the_tables(case):
    _, table, images = case
    for b_factor in (0.0, 40.0):
        scales = ThresholdTest(table, Calibration(b_factor=b_factor)).scales(images)
        assert scales.q_kept.shape == (3, 6) and scales.n_templates == 6

        def weigh(curves, name):
            return curves @ getattr(table, name).double().T

        noise = images.noise
        envelope = torch.exp(
            -b_factor * (table.frequencies.double() / PIXEL_SIZE) ** 2 / 4
        )
        q_kept = weigh(noise, "reconstruction_autocorr")
        lam = weigh(noise, "reconstruction_residual_corr") / q_kept
        q_perp = weigh(noise, "residual_autocorr") - lam**2 * q_kept
        g_kept = weigh(images.signal * envelope, "signal_reconstruction_corr")
        g_res = weigh(images.signal * envelope, "signal_residual_corr")
        rho = g_res / (g_kept + g_res)
        torch.testing.assert_close(scales.q_kept, q_kept)
        torch.testing.assert_close(scales.lam, lam)
        torch.testing.assert_close(scales.q_perp, q_perp.clamp_min(0))
        torch.testing.assert_close(scales.rho, rho.clamp(0, 1 - 1e-6))
        assert scales.qbar_perp == pytest.approx(float(scales.q_perp.mean()))
    # the envelope moves only the signal loss
    plain = ThresholdTest(table).scales(images)
    assert torch.equal(plain.q_kept, scales.q_kept)
    assert not torch.allclose(plain.rho, scales.rho)
    # qbar_perp averages over the stage's rows and planes
    sub = ThresholdTest(table).scales(images, rows=torch.tensor([0, 2]), planes=(1,))
    assert sub.qbar_perp == pytest.approx(float(sub.q_perp[1, [0, 2]].mean()))


def test_scales_need_the_tables_axis(case):
    noise, _, images = case
    padded = noise.table(RECTS, pad=3)
    with pytest.raises(ValueError, match="another frequency axis"):
        ThresholdTest(padded).scales(images)


def _pixels(n, scales, seed=1):
    gen = torch.Generator().manual_seed(seed)
    q = float(scales.q_kept.mean())
    return {
        "level": 0.5 + torch.rand(n, generator=gen),
        "plane": torch.randint(0, scales.q_kept.shape[0], (n,), generator=gen),
        "row": torch.randint(0, scales.n_templates, (n,), generator=gen),
        "mean": torch.randn(n, generator=gen),
        "variance": q * (0.5 + torch.rand(n, generator=gen)),
        "mip": 3.0 * q**0.5 * torch.rand(n, generator=gen),
    }


def test_distance_matches_the_hand_computed_rule(case):
    _, table, images = case
    cal = Calibration(alpha=1e-2, zeta=0.95, eps_rho=0.2)
    test = ThresholdTest(table, cal)
    scales = test.scales(images)
    px = _pixels(400, scales)
    threshold = 2.5
    dist = test.distance(scales, threshold, **px)
    assert dist.shape == (400,) and dist.dtype is torch.float32

    d, n = px["plane"], px["row"]
    q_kept, q_perp = scales.q_kept[d, n], scales.q_perp[d, n]
    lam, rho = scales.lam[d, n], scales.rho[d, n]
    ell, mu = px["level"].double(), px["mean"].double()
    x = px["mip"].double() - mu
    a = rho / (1 - rho)
    sigma_hat = (px["variance"].double() + ell * scales.qbar_perp).sqrt()
    t_lo = mu + threshold * cal.zeta * sigma_hat
    mean_p = mu + x / (1 - rho)
    sd2 = (a - lam) ** 2 * ell * q_kept + ell * q_perp + (cal.eps_rho * a * x) ** 2
    expected = (t_lo - mean_p) / sd2.sqrt()
    torch.testing.assert_close(dist.double(), expected, rtol=1e-5, atol=1e-5)
    rejected = dist >= cal.z_alpha
    assert rejected.any() and not rejected.all()

    fewer = NoiseScales(
        *(getattr(scales, k)[:, :-1] for k in ("q_kept", "q_perp", "lam", "rho")),
        scales.qbar_perp,
    )
    with pytest.raises(ValueError, match="template rows"):
        test.distance(fewer, threshold, **px)


def test_full_rank_distance_is_the_lowered_threshold():
    """No missed part (``q_perp = lam = rho = 0``): reject exactly below ``T zeta``."""
    zeros = torch.zeros(2, 6, dtype=torch.float64)  # the table's 6 rows
    scales = NoiseScales(zeros + 4.0, zeros, zeros, zeros, 0.0)
    test = ThresholdTest(noise_case().table(RECTS), Calibration(zeta=0.9))
    z = torch.linspace(-2.0, 12.0, 141)
    n = z.numel()
    dist = test.distance(
        scales,
        7.0,
        level=torch.ones(n),
        plane=torch.zeros(n, dtype=torch.long),
        row=torch.zeros(n, dtype=torch.long),
        mip=z,
        mean=torch.zeros(n),
        variance=torch.ones(n),
    )
    clear = (z - 6.3).abs() > 1e-3
    assert torch.equal((dist >= test.calibration.z_alpha)[clear], (z < 6.3)[clear])


def test_distance_moves_with_mip_and_the_calibration(case):
    _, table, images = case
    scales = ThresholdTest(table).scales(images)
    n = 200
    px = _pixels(n, scales, seed=2)

    def n_rejected(**kwargs):
        test = ThresholdTest(table, Calibration(**kwargs))
        d = test.distance(scales, 2.5, **px)
        return int((d >= test.calibration.z_alpha).sum())

    n_base = n_rejected()
    assert 0 < n_base < n
    assert n_rejected(alpha=1e-4) <= n_base <= n_rejected(alpha=1e-1)
    assert n_rejected(zeta=0.8) <= n_base
    assert n_rejected(eps_rho=0.5) <= n_base <= n_rejected(eps_rho=0.0)

    rising = {**px, "mip": torch.linspace(0.0, 4.0, n), "plane": px["plane"] * 0}
    rising["row"] = rising["row"] * 0 + 2
    rising["level"] = torch.ones(n)
    rising["mean"], rising["variance"] = torch.zeros(n), torch.ones(n)
    d = ThresholdTest(table).distance(scales, 20.0, **rising)
    assert torch.all(d > 0) and torch.all(d[1:] < d[:-1])  # falls with mip


def test_noise_scales_validation():
    zeros = torch.zeros(2, 3, dtype=torch.float64)
    NoiseScales(zeros, zeros, zeros, zeros, 0.0)
    with pytest.raises(ValueError, match="float64"):
        NoiseScales(zeros.float(), zeros, zeros, zeros, 0.0)
    with pytest.raises(ValueError, match="float64"):
        NoiseScales(zeros, zeros[:, :2], zeros, zeros, 0.0)
    with pytest.raises(ValueError, match="qbar_perp"):
        NoiseScales(zeros, zeros, zeros, zeros, torch.tensor(0.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"\(D, N\)"):
        NoiseScales(zeros[0], zeros[0], zeros[0], zeros[0], 0.0)


def test_threshold_helpers():
    assert zscore_threshold(1, 2) == pytest.approx(float(-ndtri(0.5)))
    assert zscore_threshold(4096**2, 10**8) == pytest.approx(
        float(-ndtri(1.0 / (4096**2 * 1e8)))
    )
    with pytest.raises(ValueError):
        zscore_threshold(0, 10)
    ratio = torch.linspace(0.5, 1.5, 101)
    assert zeta_quantile(ratio, torch.ones(101), 0.1) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        zeta_quantile(ratio, torch.ones(101), 1.0)
    with pytest.raises(ValueError, match="finite"):
        zeta_quantile(torch.full((3,), float("nan")), torch.ones(3), 0.5)
