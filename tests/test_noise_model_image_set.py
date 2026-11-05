"""noise_model.ImageSet: parity with estimate_spectra and the per-micrograph checks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch_fourier_filter.utils import bin_1dim_with_lerp
from torch_fourier_filter.whitening import calculate_num_freq_bins
from torch_grid_utils.fftfreq_grid import fftfreq_grid

from panther_em.inference.search import ImageSet as SearchImageSet
from panther_em.inference.search import SpectraConfig, estimate_spectra
from panther_em.inference.search.noise_model.image_set import ImageSet
from panther_em.inference.search.noise_model.whitening import WhiteningFilter
from panther_em.spectra import WhiteningFilter as LegacyWhiteningFilter
from panther_em.spectra import frequency_axis

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

SHAPE = (96, 96)
BOX = 16
BAND = (0.35, 0.05)
PIXEL_SIZE = 1.5
FREQ = frequency_axis(calculate_num_freq_bins((2 * BOX, 2 * BOX)))  # the tables' axis


def coloured(shape, seed, scale=1.0):
    """Gaussian noise under a smooth low-pass envelope (a non-flat spectrum)."""
    x = torch.randn(shape, generator=torch.Generator().manual_seed(seed))
    r = fftfreq_grid(image_shape=shape, rfft=True, fftshift=False, norm=True)
    envelope = 0.2 + torch.exp(-((r / 0.15) ** 2))
    return scale * torch.fft.irfft2(torch.fft.rfft2(x) * envelope, s=shape)


def micrographs():
    raws = [coloured(SHAPE, 10 + i, s) for i, s in enumerate((1, 0.7, 1.4))]
    raws[2][:48, :48] *= 3.0  # one bright quadrant: a non-uniform noise level
    return [(f"m{i}", raw) for i, raw in enumerate(raws)]


def ctf_planes(shape, n=3):
    """Real CTF-like planes ``cos(a r^2 + 0.1)``."""
    r = fftfreq_grid(image_shape=shape, rfft=True, fftshift=False, norm=True)
    return torch.stack([torch.cos((60 + 40 * d) * r**2 + 0.1) for d in range(n)])


def support_disk(box=BOX, radius=7):
    yy, xx = torch.meshgrid(torch.arange(box), torch.arange(box), indexing="ij")
    c = (box - 1) / 2
    return (yy - c) ** 2 + (xx - c) ** 2 <= radius**2


@pytest.fixture(scope="module")
def whitening():
    return WhiteningFilter.from_images(
        micrographs(), band_cutoff=BAND[0], band_falloff=BAND[1]
    )


def prepare(whitening, name="m2", device="cpu"):
    raw = dict(micrographs())[name].to(device)
    return ImageSet.from_micrograph(
        raw,
        name,
        whitening,
        ctf_planes(SHAPE),
        frequencies=FREQ,
        support=support_disk(),
        pixel_size=PIXEL_SIZE,
    )


def full_grid_ring_mean(values):
    """Reference ring mean over the full DFT grid (rfft columns weighted 1, 2, 1)."""
    h, w = SHAPE
    radius = fftfreq_grid(image_shape=SHAPE, rfft=True, fftshift=False, norm=True)
    weight = torch.full((w // 2 + 1,), 2.0, dtype=torch.float64)
    weight[0], weight[-1] = 1.0, 1.0
    weight = weight.expand(h, -1).contiguous()
    axis, radius = FREQ.double(), radius.double()
    total = bin_1dim_with_lerp(radius, values.double() * weight, axis, False)
    return total / bin_1dim_with_lerp(radius, weight, axis, False)


def test_curves_are_full_grid_ring_means(whitening):
    """noise and signal against an independent binning of the defining grids."""
    images = prepare(whitening)
    h, w = SHAPE
    power = torch.fft.rfft2(images.image).abs().square()
    correction = whitening.correction_grid("m2", SHAPE).square()
    for d, plane in enumerate(ctf_planes(SHAPE)):
        torch.testing.assert_close(
            images.noise[d], full_grid_ring_mean(power * plane**2) / (h * w)
        )
        torch.testing.assert_close(
            images.signal[d], full_grid_ring_mean(plane**2 * correction)
        )


def test_matches_the_legacy_pipeline(whitening):
    """Same search image and level map as the caller + estimate_spectra path.

    The legacy curves are half-grid ring means (``power_spectral_density`` counts each
    rfft column once); the new ones weight the columns as the full grid does, like the
    error tables. They differ by a few percent per ring (most at DC and the corner),
    which moves the level map by under 1 %.
    """
    images = prepare(whitening)
    legacy = LegacyWhiteningFilter.from_images(micrographs())
    raw = dict(micrographs())["m2"]
    x = torch.fft.irfft2(torch.fft.rfft2(raw) * legacy.image_grid("m2", SHAPE), s=SHAPE)
    x = (x - x.mean()) / x.std()
    torch.testing.assert_close(images.image, x)

    tables = SimpleNamespace(n_bins=FREQ.numel(), support=support_disk())
    spectra = estimate_spectra(
        SearchImageSet.single(x, filters=ctf_planes(SHAPE)),
        tables,
        SpectraConfig(pixel_size=PIXEL_SIZE, band_cutoff=BAND[0], band_falloff=BAND[1]),
        correction=(legacy.freq, legacy.correction_1d("m2")),
    )
    torch.testing.assert_close(images.frequencies, spectra.freq)
    interior = slice(1, -1)
    torch.testing.assert_close(
        images.noise[:, interior], spectra.noise[0][:, interior], rtol=0.06, atol=0.0
    )
    torch.testing.assert_close(images.level, spectra.level[0], rtol=0.02, atol=0.0)


def test_curves_and_level(whitening):
    images = prepare(whitening)
    assert images.n_defocus == 3 and images.n_bins == FREQ.numel()
    assert images.image_shape == SHAPE
    assert abs(float(images.image.mean())) < 1e-5
    assert float(images.image.std()) == pytest.approx(1.0, rel=1e-5)
    assert float(images.level.mean()) == pytest.approx(1.0, rel=1e-5)
    # the bright quadrant is noisier than the rest
    inner = images.level[8:40, 8:40].mean() / images.level[56:88, 56:88].mean()
    assert inner > 2.0


def test_wrong_micrograph_or_inputs_raise(whitening):
    raw = dict(micrographs())["m1"]
    kwargs = {"frequencies": FREQ, "support": support_disk(), "pixel_size": PIXEL_SIZE}
    with pytest.raises(ValueError, match="another micrograph"):
        ImageSet.from_micrograph(raw, "m2", whitening, ctf_planes(SHAPE), **kwargs)
    with pytest.raises(ValueError, match="ctf planes"):
        ImageSet.from_micrograph(
            raw, "m1", whitening, ctf_planes((SHAPE[0], SHAPE[1] - 2)), **kwargs
        )
    with pytest.raises(ValueError, match="real"):
        ImageSet.from_micrograph(
            raw, "m1", whitening, ctf_planes(SHAPE).to(torch.complex64), **kwargs
        )
    with pytest.raises(ValueError, match="bool"):
        ImageSet.from_micrograph(
            raw,
            "m1",
            whitening,
            ctf_planes(SHAPE),
            **{**kwargs, "support": 1.0 * support_disk()},
        )


def test_as_search_images(whitening):
    images = prepare(whitening)
    search = images.as_search_images()
    assert search.n_images == 1 and search.n_defocus == 3
    assert search.spectra is not None
    assert tuple(search.spectra.noise.shape) == (1, 3, FREQ.numel())
    assert tuple(search.spectra.level.shape) == (1, *SHAPE)
    torch.testing.assert_close(search.images[0], images.image)


@requires_cuda
def test_cuda_matches_cpu(whitening):
    cpu = prepare(whitening)
    gpu = prepare(whitening, device="cuda")
    assert gpu.image.is_cuda and gpu.ctf.is_cuda and not gpu.noise.is_cuda
    torch.testing.assert_close(gpu.image.cpu(), cpu.image, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(gpu.noise, cpu.noise, rtol=1e-4, atol=1e-8)
    torch.testing.assert_close(gpu.signal, cpu.signal, rtol=1e-4, atol=1e-8)
    torch.testing.assert_close(gpu.level, cpu.level, rtol=1e-3, atol=1e-4)
