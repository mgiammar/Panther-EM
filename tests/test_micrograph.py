"""search.Micrograph: the curves, the level map and the per-micrograph checks."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch_fourier_filter.bandpass import low_pass_filter
from torch_fourier_filter.utils import bin_1dim_with_lerp
from torch_fourier_filter.whitening import calculate_num_freq_bins
from torch_grid_utils.fftfreq_grid import fftfreq_grid

from panther_em.approximation.errors import (
    PerSpatialFrequencyErrorTable,
)
from panther_em.coordinates import WarpConfig
from panther_em.filters.curves import (
    curve_to_grid,
    frequency_axis,
)
from panther_em.filters.whitening import WhiteningFilter
from panther_em.search.micrograph import FLATTEN_FLOOR, Micrograph

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


def error_table(whitening):
    """A table carrying only what the image set reads: axis, support, filter."""
    zeros = torch.zeros(1, FREQ.numel())
    return PerSpatialFrequencyErrorTable(
        FREQ,
        *(zeros,) * 5,
        support=support_disk(),
        template_filter=whitening.template_filter((BOX, BOX)),
        rectangles=((0, 0, 1, 1),),
        warp=WarpConfig(),
    )


@pytest.fixture(scope="module")
def whitening():
    return WhiteningFilter.from_images(
        micrographs(), band_cutoff=BAND[0], band_falloff=BAND[1]
    )


def prepare(whitening, name="m2", device="cpu"):
    raw = dict(micrographs())[name].to(device)
    return Micrograph.prepare(
        raw,
        name,
        whitening,
        ctf_planes(SHAPE),
        error_table(whitening),
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


def windowed_disk_mean(values, support):
    """Reference disk mean by explicit windows (zero outside the image)."""
    h, w = values.shape
    k_h, k_w = support.shape
    r0, c0 = (k_h - 1) // 2, (k_w - 1) // 2
    pad = (k_w - 1 - c0, c0, k_h - 1 - r0, r0)
    kernel = support.flip(0, 1).flatten().to(values.dtype)

    def windows(x):
        return F.unfold(F.pad(x[None, None], pad), (k_h, k_w))[0]  # (kH kW, H W)

    total = (kernel[:, None] * windows(values)).sum(0)
    count = (kernel[:, None] * windows(torch.ones_like(values))).sum(0)
    return (total / count).reshape(h, w)


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


def test_level_flattens_by_the_image_curve(whitening):
    """The level map against an independent flattening and windowed disk mean."""
    images = prepare(whitening)
    h, w = SHAPE
    spectrum = torch.fft.rfft2(images.image)
    curve = full_grid_ring_mean(spectrum.abs().square()) / (h * w)
    s_grid = curve_to_grid(curve.float(), FREQ, SHAPE)
    band = low_pass_filter(*BAND, SHAPE, rfft=True, fftshift=False)
    floor = FLATTEN_FLOOR * s_grid[band > 1e-3].mean()
    flat = torch.fft.irfft2(spectrum * band / s_grid.clamp_min(floor).sqrt(), s=SHAPE)
    local = windowed_disk_mean(flat.square(), support_disk())
    torch.testing.assert_close(images.level, local / local.mean(), rtol=1e-4, atol=1e-5)

    assert float(images.level.mean()) == pytest.approx(1.0, rel=1e-5)
    # the bright quadrant is noisier than the rest
    inner = images.level[8:40, 8:40].mean() / images.level[56:88, 56:88].mean()
    assert inner > 2.0


def test_search_image_is_the_micrograph_under_its_image_filter(whitening):
    """``W_avg * C_i^2`` applied, then zero mean and unit variance."""
    images = prepare(whitening)
    raw = dict(micrographs())["m2"]
    grid = whitening.image_filter("m2", SHAPE)
    x = torch.fft.irfft2(torch.fft.rfft2(raw) * grid, s=SHAPE)
    x = (x - x.mean()) / x.std()
    torch.testing.assert_close(images.image, x)


def test_shapes_and_normalization(whitening):
    images = prepare(whitening)
    assert images.n_defocus == 3 and images.n_bins == FREQ.numel()
    assert images.image_shape == SHAPE
    assert abs(float(images.image.mean())) < 1e-5
    assert float(images.image.std()) == pytest.approx(1.0, rel=1e-5)


def test_wrong_micrograph_table_or_inputs_raise(whitening):
    raw = dict(micrographs())["m1"]
    table = error_table(whitening)
    with pytest.raises(ValueError, match="another micrograph"):
        Micrograph.prepare(
            raw, "m2", whitening, ctf_planes(SHAPE), table, pixel_size=PIXEL_SIZE
        )
    other = WhiteningFilter.from_images(micrographs()[:2], band_cutoff=BAND[0])
    with pytest.raises(ValueError, match="another W_avg or band"):
        Micrograph.prepare(
            raw,
            "m1",
            whitening,
            ctf_planes(SHAPE),
            error_table(other),
            pixel_size=PIXEL_SIZE,
        )
    with pytest.raises(ValueError, match="ctf planes"):
        Micrograph.prepare(
            raw,
            "m1",
            whitening,
            ctf_planes((SHAPE[0], SHAPE[1] - 2)),
            table,
            pixel_size=PIXEL_SIZE,
        )
    with pytest.raises(ValueError, match="real"):
        Micrograph.prepare(
            raw,
            "m1",
            whitening,
            ctf_planes(SHAPE).to(torch.complex64),
            table,
            pixel_size=PIXEL_SIZE,
        )


@requires_cuda
def test_cuda_matches_cpu(whitening):
    cpu = prepare(whitening)
    gpu = prepare(whitening, device="cuda")
    assert gpu.image.is_cuda and gpu.ctf.is_cuda and not gpu.noise.is_cuda
    torch.testing.assert_close(gpu.image.cpu(), cpu.image, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(gpu.noise, cpu.noise, rtol=1e-4, atol=1e-8)
    torch.testing.assert_close(gpu.signal, cpu.signal, rtol=1e-4, atol=1e-8)
    torch.testing.assert_close(gpu.level, cpu.level, rtol=1e-3, atol=1e-4)


def test_save_and_load_round_trip(whitening, tmp_path):
    prepared = prepare(whitening)
    prepared.save(tmp_path / "m2.h5")
    loaded = Micrograph.load(tmp_path / "m2.h5")
    assert loaded.name == prepared.name and loaded.pixel_size == prepared.pixel_size
    for field in ("image", "ctf", "frequencies", "noise", "signal", "level"):
        got, want = getattr(loaded, field), getattr(prepared, field)
        assert got.dtype == want.dtype and torch.equal(got, want.cpu())
