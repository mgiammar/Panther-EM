"""noise_model.WhiteningFilter: parity with panther_em.spectra and its guards."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_fourier_filter.bandpass import low_pass_filter
from torch_grid_utils.fftfreq_grid import fftfreq_grid

from panther_em.inference.search.noise_model.whitening import WhiteningFilter
from panther_em.spectra import WhiteningFilter as LegacyWhiteningFilter

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

SHAPE = (96, 96)
BOX = (16, 16)
BAND = (0.35, 0.05)


def coloured(shape, seed, scale=1.0):
    """Gaussian noise under a smooth low-pass envelope (a non-flat spectrum)."""
    x = torch.randn(shape, generator=torch.Generator().manual_seed(seed))
    r = fftfreq_grid(image_shape=shape, rfft=True, fftshift=False, norm=True)
    envelope = 0.2 + torch.exp(-((r / 0.15) ** 2))
    return scale * torch.fft.irfft2(torch.fft.rfft2(x) * envelope, s=shape)


def micrographs():
    return [(f"m{i}", coloured(SHAPE, 10 + i, s)) for i, s in enumerate((1, 0.7, 1.4))]


@pytest.fixture(scope="module")
def whitening():
    return WhiteningFilter.from_images(
        micrographs(), band_cutoff=BAND[0], band_falloff=BAND[1]
    )


def test_matches_the_legacy_filter(whitening):
    legacy = LegacyWhiteningFilter.from_images(micrographs(), max_freq=0.5)
    torch.testing.assert_close(whitening.frequencies, legacy.freq)
    torch.testing.assert_close(whitening.spectra, legacy.spectra)
    torch.testing.assert_close(whitening.average_curve, legacy.average_1d)
    torch.testing.assert_close(
        whitening.correction_curve("m1"), legacy.correction_1d(1)
    )
    torch.testing.assert_close(
        whitening.image_filter("m1", SHAPE), legacy.image_grid("m1", SHAPE)
    )
    torch.testing.assert_close(
        whitening.correction_grid("m2", SHAPE), legacy.correction_grid("m2", SHAPE)
    )
    band = low_pass_filter(*BAND, BOX, rfft=True, fftshift=False)
    torch.testing.assert_close(
        whitening.template_filter(BOX), legacy.average_grid(BOX) * band
    )


def test_from_images_needs_named_pairs():
    with pytest.raises(TypeError):
        WhiteningFilter.from_images([coloured(SHAPE, 0)], band_cutoff=0.3)
    with pytest.raises(TypeError):  # a (2, W) tensor must not unpack as a pair
        WhiteningFilter.from_images([torch.randn(2, 8)], band_cutoff=0.3)
    with pytest.raises(ValueError):
        WhiteningFilter.from_images([], band_cutoff=0.3)
    with pytest.raises(ValueError):
        WhiteningFilter.from_images(
            [("a", coloured(SHAPE, 0)), ("a", coloured(SHAPE, 1))], band_cutoff=0.3
        )


def test_check_image(whitening):
    images = dict(micrographs())
    whitening.check_image("m1", images["m1"])
    with pytest.raises(ValueError, match="another micrograph"):
        whitening.check_image("m1", images["m2"])  # the wrong name
    with pytest.raises(ValueError, match="pre-processed differently"):
        whitening.check_image("m1", 1.01 * images["m1"])
    with pytest.raises(KeyError):
        whitening.check_image("z", images["m1"])


def test_check_template_filter(whitening):
    grid = whitening.template_filter(BOX)
    whitening.check_template_filter(grid)
    whitening.check_template_filter(grid[None].numpy())  # a decomposition's layout
    whitening.check_template_filter(grid.to(torch.complex64))
    with pytest.raises(ValueError, match="another W_avg or band"):
        whitening.check_template_filter(
            whitening.template_filter(BOX) / whitening.band_grid(BOX).clamp_min(1e-3)
        )
    other = WhiteningFilter.from_images(micrographs()[:2], band_cutoff=BAND[0])
    with pytest.raises(ValueError, match="another W_avg or band"):
        whitening.check_template_filter(other.template_filter(BOX))
    with pytest.raises(ValueError, match="exactly one"):
        whitening.check_template_filter(torch.stack([grid, grid]))
    with pytest.raises(ValueError, match="square"):
        whitening.check_template_filter(grid[:, :-1])


def test_round_trip_and_validation(whitening, tmp_path):
    path = tmp_path / "whitening.h5"
    whitening.save(path)
    back = WhiteningFilter.load(path)
    assert back.names == whitening.names
    assert (back.band_cutoff, back.band_falloff, back.max_freq) == (
        whitening.band_cutoff,
        whitening.band_falloff,
        whitening.max_freq,
    )
    torch.testing.assert_close(back.spectra, whitening.spectra)
    torch.testing.assert_close(
        back.template_filter(BOX), whitening.template_filter(BOX)
    )

    f, s, n = whitening.frequencies, whitening.spectra, whitening.names
    for bad in (
        {"frequencies": f.double()},
        {"spectra": -s},
        {"spectra": s[:, :-1]},
        {"names": n[:-1]},
        {"names": ("a", "a", "b")},
        {"band_cutoff": 0.0},
        {"band_cutoff": 0.8},
        {"band_falloff": -0.1},
        {"max_freq": 0.0},
    ):
        fields = {
            "frequencies": f,
            "spectra": s,
            "names": n,
            "band_cutoff": 0.3,
            "band_falloff": 0.0,
        }
        with pytest.raises(ValueError):
            WhiteningFilter(**{**fields, **bad})


@requires_cuda
def test_check_image_accepts_a_cuda_micrograph(whitening):
    image = dict(micrographs())["m0"].cuda()
    whitening.check_image("m0", image)
    assert np.isclose(
        float(whitening.image_filter("m0", SHAPE, "cuda").cpu().sum()),
        float(whitening.image_filter("m0", SHAPE).sum()),
    )
