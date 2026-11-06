"""Radial tables, scales, image spectra and level maps of the coloured-noise model."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter
from scipy.special import ndtri
from torch_fourier_filter.bandpass import low_pass_filter

from panther_em.coordinates import OffsetPolarTransform
from panther_em.decomposition import PolarProjectionDecomposer
from panther_em.inference.search.hypotheses import HypothesisSet
from panther_em.inference.search.noise import (
    NoiseScales,
    ProjectedTemplates,
    RadialTables,
    SpectraConfig,
    SvdTemplates,
    estimate_spectra,
    level_map,
    ring_sums,
    support_mask,
    zeta_quantile,
    zscore_threshold,
)
from panther_em.inference.template import Template
from panther_em.inference.search.types import ImageSet, ImageSpectra
from panther_em.spectra import WhiteningFilter, curve_to_grid, frequency_axis

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array_wrap__ must accept.*:DeprecationWarning",
    "ignore:.*__array__ implementation doesn't accept.*:DeprecationWarning",
)
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

RECTS = [(0, 0, 2, 3)]  # a low-rank selection of the synthetic (k 4, e 5) template
FULL = [(0, 0, 4, 5)]
TOL = {"rtol": 1e-4, "atol": 1e-6}


@pytest.fixture
def template(synthetic_result):
    return Template(synthetic_result(), device="cpu")


def _rows(n):
    return torch.arange(n)


def test_ring_sums_are_parseval_exact():
    for shape in [(16, 16), (15, 17), (16, 15)]:
        gen = torch.Generator().manual_seed(shape[0] * shape[1])
        a = torch.randn(3, *shape, generator=gen, dtype=torch.float64)
        b = torch.randn(3, *shape, generator=gen, dtype=torch.float64)
        fa, fb = torch.fft.rfft2(a), torch.fft.rfft2(b)
        freq = frequency_axis(40)
        energy = ring_sums(fa.abs().square(), freq, shape)
        assert energy.shape == (3, 40)
        torch.testing.assert_close(energy.sum(1), a.square().sum((1, 2)))
        cross = ring_sums((fa.conj() * fb).real, freq, shape)
        torch.testing.assert_close(cross.sum(1), (a * b).sum((1, 2)))
    with pytest.raises(ValueError):
        ring_sums(fa, freq, shape)


def test_support_mask_is_the_polar_disk(template):
    transform = template.warp.transform
    mask = support_mask(transform, template.kernel_config)
    assert mask.shape == (20, 20) and mask.dtype is torch.bool
    assert mask[10, 10] and not mask[0, 0]
    # the disk of the polar grid (radius height / 2 about the image centre) up to
    # the ring the spline warp cannot fill from the outermost polar sample
    rows, cols = torch.meshgrid(torch.arange(20.0), torch.arange(20.0), indexing="ij")
    radius = torch.hypot(rows - 9.5, cols - 9.5)
    assert torch.all(mask[radius < 8.5]) and not mask[radius >= 10.0].any()
    assert mask.float().mean().item() > 0.6


def test_tables_parseval_full_rank_and_reconstructor_convention(template):
    svd = SvdTemplates(template, RECTS)
    kept, full = svd(_rows(svd.n_templates))
    assert kept.shape == full.shape == (6, 20, 20)
    t = RadialTables.build(template, RECTS)
    assert t.exact == "reconstruction" and t.signal_source == "template"
    assert t.n_templates == 6 and t.template_shape == (20, 20) and t.freq[0] == 0.0
    assert t.rectangles == ((0, 0, 2, 3),)
    delta = full - kept
    tol = TOL
    torch.testing.assert_close(t.kept.sum(1), kept.square().sum((1, 2)), **tol)
    torch.testing.assert_close(t.residual.sum(1), delta.square().sum((1, 2)), **tol)
    torch.testing.assert_close(t.cross.sum(1), (kept * delta).sum((1, 2)), **tol)
    # with p = m the signal tables are the cross-spectra with the full template
    torch.testing.assert_close(t.kept_signal.sum(1), (kept * full).sum((1, 2)), **tol)
    torch.testing.assert_close(
        t.residual_signal.sum(1), (delta * full).sum((1, 2)), **tol
    )
    assert torch.all(t.support == svd.support)

    # the full selection leaves no residual, and the full reconstruction is the
    # dense reconstructor's projection (real part), i.e. the same polar convention
    t_full = RadialTables.build(template, FULL)
    assert t_full.residual.abs().max() < 1e-6 * t_full.kept.sum(1).max()
    assert t_full.cross.abs().max() < 1e-6 * t_full.kept.sum(1).max()
    kept_full, _ = SvdTemplates(template, FULL)(_rows(6))
    for o in range(6):
        ref = template.warp.reconstructor.reconstruct_projection(
            o, 0, return_torch=True
        )
        ref = torch.as_tensor(np.asarray(ref)).real.to(torch.float32)  # NumPy on CPU
        torch.testing.assert_close(kept_full[o], ref, rtol=1e-4, atol=1e-5)

    with pytest.raises(ValueError):
        RadialTables.build(template, [(0, 0, 2, 3), (1, 1, 2, 2)])  # overlap
    with pytest.raises(ValueError):
        RadialTables.build(template, [(0, 0, 8, 3)])  # beyond the stored grid
    with pytest.raises(ValueError):
        RadialTables.build(template, RECTS, whitening=torch.ones(20, 11))  # no volume


def test_tables_predict_the_correlation_variance(template):
    """``q = sum_f S Gamma`` is the variance of the kept correlation under ``S``."""
    kept, _ = SvdTemplates(template, RECTS)(_rows(6))
    t = RadialTables.build(template, RECTS)
    gen = torch.Generator().manual_seed(7)
    n_samples = 8000
    white = torch.randn(n_samples, 20, 20, generator=gen)
    corr = torch.einsum("ohw,shw->os", kept, white)
    torch.testing.assert_close(corr.var(dim=1), t.kept.sum(1), rtol=0.06, atol=0)
    # radially coloured noise of power P(f): filter white noise by sqrt(P)
    power = 1.0 + 3.0 * t.freq.double() ** 2
    root = curve_to_grid(power.sqrt().float(), t.freq, (20, 20))
    coloured = torch.fft.irfft2(torch.fft.rfft2(white) * root, s=(20, 20))
    corr = torch.einsum("ohw,shw->os", kept, coloured)
    predicted = t.kept.double() @ power
    torch.testing.assert_close(corr.var(dim=1).double(), predicted, rtol=0.08, atol=0)


def _small_decomposition(box=24, filtered=True):
    rng = np.random.default_rng(0)
    volume = gaussian_filter(rng.standard_normal((box, box, box)), 2.0).astype(
        np.float32
    )
    phi = np.linspace(0.0, 300.0, 6, dtype=np.float32)
    theta = np.linspace(10.0, 80.0, 6, dtype=np.float32)
    transform = OffsetPolarTransform.from_image(
        image_shape=(box, box), num_angle=32, num_radius=12
    )
    filters = None
    if filtered:
        filters = low_pass_filter(0.35, 0.05, (box, box), rfft=True, fftshift=False)[
            None
        ]
    result = PolarProjectionDecomposer(
        volume, phi, theta, coordinate_transform=transform, fourier_filters=filters
    ).do_decomposition(eig_max=6, normalize_projections=True, svd_method="dense")
    return volume, result  # 6 rows: eig_max 6 keeps every component of each block


def test_projection_route_on_a_real_decomposition(tmp_path):
    volume, result = _small_decomposition()
    template = Template(result, device="cpu")
    k_max, eig_max = result.k_max, result.eig_max
    full = [(0, 0, k_max, eig_max)]
    projected = ProjectedTemplates(template, volume)
    exact, signal = projected(_rows(6))
    assert exact.shape == (6, 24, 24) and torch.equal(exact, signal)
    assert projected.signal_source == "template"
    kept, _ = SvdTemplates(template, full)(_rows(6))
    # every stored component: the reconstruction is the projection up to the polar
    # warp round trip, so the residual holds a small share of the energy
    residual = (exact - kept).square().sum((1, 2)) / exact.square().sum((1, 2))
    assert residual.max() < 0.25
    t = RadialTables.build(template, full, volume=volume)
    assert t.exact == "projection" and t.signal_source == "template"
    tol = TOL
    torch.testing.assert_close(
        (t.kept + 2 * t.cross + t.residual).sum(1), exact.square().sum((1, 2)), **tol
    )
    # the low-rank selection loses more
    t_low = RadialTables.build(template, [(0, 0, 3, 4)], volume=volume)
    assert t_low.residual.sum(1).min() > t.residual.sum(1).max()

    # a whitening filter makes p the W_avg-filtered projection
    wf = WhiteningFilter.from_images(
        [torch.randn(64, 64, generator=torch.Generator().manual_seed(1))]
    )
    t_w = RadialTables.build(template, full, volume=volume, whitening=wf)
    assert t_w.signal_source == "projection"
    exact_w, signal_w = ProjectedTemplates(template, volume, whitening=wf)(_rows(6))
    torch.testing.assert_close(exact_w, exact)
    assert not torch.equal(signal_w, exact_w)
    torch.testing.assert_close(
        (t_w.kept_signal + t_w.residual_signal).sum(1),
        (exact * signal_w).sum((1, 2)),
        **tol,
    )
    torch.testing.assert_close(t_w.kept, t.kept)
    # an unmasked exact template keeps the box corners
    unmasked = ProjectedTemplates(template, volume, mask_exact=False)(_rows(6))[0]
    assert unmasked[:, 0, 0].abs().max() > 0 and exact[:, 0, 0].abs().max() == 0

    path = tmp_path / "tables.h5"
    t_w.save(path)
    back = RadialTables.load(path)
    for name in ("freq", "kept", "residual", "cross", "kept_signal", "residual_signal"):
        torch.testing.assert_close(getattr(back, name), getattr(t_w, name))
    assert torch.equal(back.support, t_w.support) and back.rectangles == t_w.rectangles
    assert (
        back.exact == "projection"
        and back.signal_source == "projection"
        and back.pad == 2
    )
    assert back.summary()["rectangles"] == [[0, 0, k_max, eig_max]]
    with pytest.raises(ValueError):
        ProjectedTemplates(template, volume[:-1])


def test_scales_lambda_rho_qbar_and_io(template, tmp_path):
    t = RadialTables.build(template, RECTS)
    spectra = ImageSpectra.create(t.freq, 2.0 * torch.ones(t.n_bins))  # white, level 2
    s = t.scales(spectra)
    assert s.q_kept.shape == (1, 1, 6) and s.n_templates == 6
    torch.testing.assert_close(s.q_kept[0, 0], 2.0 * t.kept.double().sum(1))
    torch.testing.assert_close(s.q_residual[0, 0], 2.0 * t.residual.double().sum(1))
    torch.testing.assert_close(s.lam, s.q_cross / s.q_kept)
    torch.testing.assert_close(
        s.q_perp, (s.q_residual - s.lam**2 * s.q_kept).clamp_min(0)
    )
    assert torch.all(s.q_perp >= 0) and torch.all((s.rho >= 0) & (s.rho < 1))
    g_k, g_d = t.kept_signal.double().sum(1), t.residual_signal.double().sum(1)
    torch.testing.assert_close(s.rho[0, 0], (g_d / (g_k + g_d)).clamp(0, 1 - 1e-6))
    torch.testing.assert_close(s.qbar_perp, s.q_perp[0, 0].mean()[None])
    assert s.rectangles == t.rectangles and s.b_factor == 0.0

    # the B-factor envelope weights the signal tables
    b = t.scales(spectra, b_factor=40.0)
    env = torch.exp(-40.0 * t.freq.double() ** 2 / 4)
    g_k, g_d = t.kept_signal.double() @ env, t.residual_signal.double() @ env
    torch.testing.assert_close(b.rho[0, 0], (g_d / (g_k + g_d)).clamp(0, 1 - 1e-6))
    torch.testing.assert_close(b.q_kept, s.q_kept)
    # a pixel size scales the spatial frequency of the envelope
    px = ImageSpectra.create(t.freq, 2.0 * torch.ones(t.n_bins), pixel_size=2.0)
    env2 = torch.exp(-40.0 * (t.freq.double() / 2.0) ** 2 / 4)
    g_k2, g_d2 = t.kept_signal.double() @ env2, t.residual_signal.double() @ env2
    torch.testing.assert_close(
        t.scales(px, b_factor=40.0).rho[0, 0], (g_d2 / (g_k2 + g_d2)).clamp(0, 1 - 1e-6)
    )

    # qbar_perp follows the stage's hypotheses; a shared noise curve serves every plane
    hyps = HypothesisSet.create(
        template.space, torch.tensor([0, 2]), None, hyp_batch=4, n_psi=16
    )
    sub = t.scales(spectra, hyps=hyps)
    torch.testing.assert_close(sub.qbar_perp, s.q_perp[0, 0, [0, 2]].mean()[None])
    three = ImageSpectra.create(
        t.freq, (torch.tensor([1.0, 2.0, 3.0])[:, None] * torch.ones(3, t.n_bins))[None]
    )
    assert three.noise.shape == (1, 3, t.n_bins)
    per_plane = t.scales(three)
    torch.testing.assert_close(per_plane.q_kept[0, 1], 2.0 * t.kept.double().sum(1))
    # a non-white curve: the dot product with the table
    curve = 1.0 + t.freq.double() * 3.0
    coloured = t.scales(ImageSpectra.create(t.freq, curve))
    torch.testing.assert_close(coloured.q_kept[0, 0], t.kept.double() @ curve)
    # curves on another axis are resampled
    other = frequency_axis(10)
    resampled = t.scales(ImageSpectra.create(other, torch.ones(10)))
    torch.testing.assert_close(resampled.q_kept, s.q_kept / 2.0)

    path = tmp_path / "scales.h5"
    b.save(path)
    back = NoiseScales.load(path)
    for name in (
        "q_kept",
        "q_residual",
        "q_cross",
        "lam",
        "q_perp",
        "rho",
        "qbar_perp",
    ):
        torch.testing.assert_close(getattr(back, name), getattr(b, name))
    assert back.rectangles == b.rectangles and back.b_factor == 40.0
    assert back.to("cpu").q_kept.device.type == "cpu"
    with pytest.raises(ValueError):
        NoiseScales(
            s.q_kept,
            s.q_residual,
            s.q_cross,
            s.lam,
            s.q_perp,
            s.rho[0],
            s.qbar_perp,
            t.rectangles,
        )


def test_estimate_spectra_shapes_filters_corrections_and_levels(template):
    t = RadialTables.build(template, RECTS)
    gen = torch.Generator().manual_seed(3)
    images = ImageSet.from_stack(3.0 * torch.randn(2, 40, 40, generator=gen))
    spectra = estimate_spectra(images, t, SpectraConfig(level="none"))
    assert spectra.noise.shape == (2, 1, t.n_bins) and spectra.signal.shape == (
        1,
        1,
        t.n_bins,
    )
    assert spectra.level.shape == (1, 1, 1) and torch.equal(spectra.freq, t.freq)
    assert spectra.noise[:, 0, 2:-8].mean().item() == pytest.approx(9.0, rel=0.1)
    assert torch.all(spectra.signal == 1.0)
    assert images.with_spectra(spectra).spectra is spectra

    # shared CTF planes: the noise follows the plane's filter, the signal is |filter|^2
    filters = torch.ones(2, 40, 21)
    filters[1] = 0.5
    with_planes = ImageSet.from_stack(images.images, filters=filters)
    sp = estimate_spectra(with_planes, t, SpectraConfig(level="none"))
    assert sp.noise.shape == (2, 2, t.n_bins) and sp.signal.shape == (1, 2, t.n_bins)
    torch.testing.assert_close(sp.noise[:, 1], 0.25 * sp.noise[:, 0])
    torch.testing.assert_close(
        sp.signal[0, 1], 0.25 * torch.ones(t.n_bins, dtype=torch.float64)
    )
    per_image = ImageSet.from_stack(
        images.images, filters=filters[None].expand(2, -1, -1, -1)
    )
    assert estimate_spectra(per_image, t, SpectraConfig(level="none")).signal.shape == (
        2,
        2,
        t.n_bins,
    )
    # whitening corrections enter the signal weight squared
    corrected = estimate_spectra(
        with_planes,
        t,
        SpectraConfig(level="none"),
        correction=(frequency_axis(8), 2.0 * torch.ones(8)),
    )
    torch.testing.assert_close(corrected.signal, 4.0 * sp.signal)
    with pytest.raises(ValueError):
        estimate_spectra(
            with_planes, t, correction=(frequency_axis(8), torch.ones(3, 8))
        )

    # the level map: ~1 on white noise, raised where the noise is stronger
    noisy = images.images.clone()
    noisy[0, :20, :20] *= 3.0
    cfg = SpectraConfig(level="disk", band_cutoff=0.4, band_falloff=0.05)
    lv = estimate_spectra(ImageSet.from_stack(noisy), t, cfg).level
    assert lv.shape == (2, 40, 40)
    torch.testing.assert_close(lv.mean((1, 2)), torch.ones(2), rtol=1e-4, atol=1e-4)
    assert (lv[1] - 1).abs().max() < 0.5
    assert lv[0, 5:15, 5:15].mean() > 2 * lv[0, 25:35, 25:35].mean()
    noisy_spectra = estimate_spectra(ImageSet.from_stack(noisy), t, cfg)
    direct = level_map(noisy, noisy_spectra.noise, noisy_spectra.freq, t.support, cfg)
    torch.testing.assert_close(direct, lv)
    tiles = level_map(
        noisy,
        spectra.noise,
        spectra.freq,
        t.support,
        SpectraConfig(level="tiles", tile=10, margin=1),
    )
    assert (
        tiles.shape == (2, 40, 40)
        and tiles[0, 5:15, 5:15].mean() > tiles[0, 25:35, 25:35].mean()
    )
    torch.testing.assert_close(tiles.mean((1, 2)), torch.ones(2), rtol=1e-4, atol=1e-4)
    with pytest.raises(ValueError):
        level_map(
            noisy,
            spectra.noise,
            spectra.freq,
            t.support,
            SpectraConfig(level="tiles", tile=64),
        )
    with pytest.raises(ValueError):
        level_map(
            noisy, spectra.noise, spectra.freq, t.support, SpectraConfig(level="none")
        )
    with pytest.raises(ValueError):
        SpectraConfig(level="median")
    with pytest.raises(ValueError):
        SpectraConfig(floor=0.0)


def test_image_spectra_validation_and_io(tmp_path):
    freq = frequency_axis(6)
    sp = ImageSpectra.create(freq, torch.ones(6), level=torch.full((2, 8, 8), 1.5))
    assert sp.noise.shape == (1, 1, 6) and sp.signal.shape == (1, 1, 6)
    assert (
        sp.level.shape == (2, 8, 8)
        and sp.level.dtype is torch.float32
        and sp.n_bins == 6
    )
    sp.validate_for(2, 3, (8, 8))
    with pytest.raises(ValueError, match="level axis 0"):
        sp.validate_for(3, 3, (8, 8))
    with pytest.raises(ValueError, match="level axis 1"):
        sp.validate_for(2, 3, (9, 8))
    with pytest.raises(ValueError, match="noise axis 1"):
        ImageSpectra.create(freq, torch.ones(1, 2, 6)).validate_for(1, 3, (8, 8))
    with pytest.raises(ValueError):
        ImageSet.from_stack(
            torch.zeros(3, 8, 8), spectra=sp
        )  # 2 level maps for 3 images
    ok = ImageSet.from_stack(torch.zeros(2, 8, 8), spectra=sp)
    assert ok.spectra is sp and ok.with_spectra(None).spectra is None
    with pytest.raises(ValueError):
        ImageSpectra.create(freq, -torch.ones(6))
    with pytest.raises(ValueError):
        ImageSpectra.create(freq, torch.ones(6), level=torch.zeros(1, 1, 1))
    with pytest.raises(ValueError):
        ImageSpectra.create(freq, torch.ones(5))
    with pytest.raises(ValueError):
        ImageSpectra.create(freq, torch.ones(6), pixel_size=0.0)
    with pytest.raises(ValueError):
        ImageSpectra(freq.double(), sp.noise, sp.signal, sp.level)

    path = tmp_path / "spectra.h5"
    sp.save(path)
    back = ImageSpectra.load(path)
    for name in ("freq", "noise", "signal", "level"):
        torch.testing.assert_close(getattr(back, name), getattr(sp, name))
    assert back.pixel_size == sp.pixel_size


def test_threshold_and_zeta_helpers():
    assert zscore_threshold(10_000, 1_000) == pytest.approx(-ndtri(1e-7))
    with pytest.raises(ValueError):
        zscore_threshold(0, 10)
    sigma = torch.tensor([1.0, 1.0, 1.0, 1.0, float("nan")])
    sigma_hat = torch.tensor([1.0, 1.25, 2.0, 0.5, 1.0])
    assert zeta_quantile(sigma, sigma_hat, 0.5) == pytest.approx(
        0.9
    )  # ratios 1, .8, .5, 2
    with pytest.raises(ValueError):
        zeta_quantile(sigma, sigma_hat, 1.0)


@requires_cuda
def test_estimate_spectra_on_cuda_matches_cpu(template):
    """A CUDA ``rfft2`` is not contiguous; the curves and level map must not care."""
    t = RadialTables.build(template, RECTS)
    gen = torch.Generator().manual_seed(11)
    images = torch.randn(2, 48, 40, generator=gen)
    filters = torch.rand(2, 48, 21, generator=gen) + 0.5
    config = SpectraConfig(level="disk", band_cutoff=0.4)
    cpu = estimate_spectra(ImageSet.from_stack(images, filters=filters), t, config)
    gpu = estimate_spectra(
        ImageSet.from_stack(images.cuda(), filters=filters.cuda()), t, config
    )
    for name in ("noise", "signal", "level"):
        got, want = getattr(gpu, name), getattr(cpu, name)
        assert got.device.type == "cpu"
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
