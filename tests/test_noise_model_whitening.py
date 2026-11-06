"""noise_model.WhiteningFilter: the filters, the corrections and the guards."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_fourier_filter.bandpass import low_pass_filter
from torch_fourier_filter.whitening import calculate_num_freq_bins, whitening_filter
from torch_grid_utils.fftfreq_grid import fftfreq_grid

from panther_em.inference.search.noise_model import whitening as whitening_module
from panther_em.inference.search.noise_model.whitening import (
    WhiteningFilter,
    canonical_device,
)

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


def test_single_micrograph_filter_is_torch_fourier_filters():
    """One micrograph: ``W_avg`` is torch_fourier_filter's whitening filter of it,
    its correction is 1, and the template filter adds the band."""
    # a large mean: whitening_filter keeps the DC term (1 / huge ~ 0), we zero it
    x = coloured(SHAPE, 3) + 1.0e4
    single = WhiteningFilter.from_images(
        [("x", x)], band_cutoff=BAND[0], band_falloff=BAND[1], max_freq=0.5
    )
    n_bins = calculate_num_freq_bins(SHAPE)
    reference = whitening_filter(
        torch.fft.rfft2(x), rfft=True, num_freq_bins=n_bins, max_freq=0.5
    )
    torch.testing.assert_close(single.average_filter(SHAPE), reference)
    torch.testing.assert_close(single.image_filter("x", SHAPE), reference)
    assert torch.all(single.correction_grid("x", SHAPE) == 1.0)
    on_box = whitening_filter(
        torch.fft.rfft2(x),
        rfft=True,
        num_freq_bins=n_bins,
        max_freq=0.5,
        output_shape=BOX,
    )
    band = low_pass_filter(*BAND, BOX, rfft=True, fftshift=False)
    torch.testing.assert_close(single.template_filter(BOX), on_box * band)


def test_corrections_relate_each_micrograph_to_the_average(whitening):
    """``C_i = W_i / W_avg`` and the search-image filter ``W_avg * C_i^2``."""
    torch.testing.assert_close(
        whitening.average_spectrum, whitening.spectra.square().mean(0).sqrt()
    )
    for name in ("m0", "m1", "m2"):
        c = whitening.correction_curve(name)
        ok = whitening.average_curve > 0  # 0 / 0 at the zeroed DC: C_i is 1 there
        torch.testing.assert_close(
            c[ok], (whitening.image_curve(name) / whitening.average_curve)[ok]
        )
        assert torch.all(c[~ok] == 1.0)
        torch.testing.assert_close(
            whitening.image_filter(name, SHAPE),
            whitening.average_filter(SHAPE)
            * whitening.correction_grid(name, SHAPE) ** 2,
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


def test_shared_grids_are_built_once_and_safe_to_modify(monkeypatch):
    whitening = WhiteningFilter.from_images(micrographs(), band_cutoff=BAND[0])
    calls = {"curve_to_grid": 0, "low_pass_filter": 0}

    def counting(name, fn):
        def wrapped(*args, **kwargs):
            calls[name] += 1
            return fn(*args, **kwargs)

        return wrapped

    for name in calls:
        monkeypatch.setattr(
            whitening_module, name, counting(name, getattr(whitening_module, name))
        )

    first = whitening.image_filter("m0", SHAPE)
    whitening.image_filter("m1", SHAPE)
    whitening.template_filter(SHAPE)
    # W_avg once, the band once, each micrograph's own C_i every time
    assert calls == {"curve_to_grid": 3, "low_pass_filter": 1}
    whitening.template_filter(BOX)  # another shape: its own grids
    assert calls == {"curve_to_grid": 4, "low_pass_filter": 2}

    average = whitening.average_filter(SHAPE)
    average.zero_()  # the caller's copy, not the cache
    whitening.band_grid(SHAPE).zero_()
    torch.testing.assert_close(whitening.image_filter("m0", SHAPE), first)
    assert float(whitening.template_filter(SHAPE).abs().sum()) > 0


@requires_cuda
def test_check_image_accepts_a_cuda_micrograph(whitening):
    image = dict(micrographs())["m0"].cuda()
    whitening.check_image("m0", image)
    assert np.isclose(
        float(whitening.image_filter("m0", SHAPE, "cuda").cpu().sum()),
        float(whitening.image_filter("m0", SHAPE).sum()),
    )


@requires_cuda
def test_cuda_and_its_index_share_one_cache_entry():
    assert canonical_device("cuda") == canonical_device(torch.cuda.current_device())
    assert canonical_device(None) == canonical_device("cpu") == torch.device("cpu")
    whitening = WhiteningFilter.from_images(micrographs(), band_cutoff=BAND[0])
    whitening.template_filter(SHAPE, "cuda")
    whitening.template_filter(SHAPE, f"cuda:{torch.cuda.current_device()}")
    assert len(whitening._shared_grids) == 2  # W_avg and the band, once each
