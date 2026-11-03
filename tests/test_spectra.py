"""The averaged whitening filter against torch_fourier_filter's single-image one."""

from __future__ import annotations

import gc
import weakref

import pytest
import torch
from torch_fourier_filter.whitening import calculate_num_freq_bins, whitening_filter
from torch_grid_utils.fftfreq_grid import fftfreq_grid

from panther_em.spectra import (
    WhiteningFilter,
    amplitude_spectrum_1d,
    curve_to_grid,
    frequency_axis,
    whitening_curve,
)


def _image(h, w, seed=0, scale=1.0):
    return scale * torch.randn(h, w, generator=torch.Generator().manual_seed(seed))


def test_amplitude_spectrum_matches_torch_fourier_filter():
    x = _image(64, 64, seed=1, scale=3.0) + 7.0  # the mean (DC) is dropped
    freq, spectrum = amplitude_spectrum_1d(x)
    n_bins = calculate_num_freq_bins((64, 64))
    assert freq.shape == spectrum.shape == (n_bins,)
    torch.testing.assert_close(freq, frequency_axis(n_bins))
    assert spectrum[0] == 0.0  # DC zeroed
    # white noise: |X|^2 ~ N s^2 so the RMS amplitude is sqrt(N) * s
    assert spectrum[2:-6].mean().item() == pytest.approx(64 * 3.0, rel=0.05)
    _, coarse = amplitude_spectrum_1d(x, num_freq_bins=12)
    assert coarse.shape == (12,)
    with pytest.raises(ValueError):
        amplitude_spectrum_1d(torch.fft.rfft2(x))
    with pytest.raises(ValueError):
        amplitude_spectrum_1d(torch.zeros(2, 8, 8))


def test_single_image_filter_equals_whitening_filter():
    """One micrograph: W_avg is torch_fourier_filter's whitening filter of it."""
    # a large mean: whitening_filter keeps the DC term (1 / huge ~ 0), we zero it
    x = _image(48, 40, seed=2) + 1.0e4
    wf = WhiteningFilter.from_images([x], max_freq=0.5)
    n_bins = calculate_num_freq_bins((48, 40))
    reference = whitening_filter(
        torch.fft.rfft2(x), rfft=True, num_freq_bins=n_bins, max_freq=0.5
    )
    torch.testing.assert_close(wf.average_grid((48, 40)), reference)
    torch.testing.assert_close(wf.image_grid(0, (48, 40)), reference)  # C_0 = 1
    assert torch.all(wf.correction_grid(0, (48, 40)) == 1.0)
    # on another grid, and shifted / full layouts
    other = whitening_filter(
        torch.fft.rfft2(x),
        rfft=True,
        num_freq_bins=n_bins,
        max_freq=0.5,
        output_shape=(32, 32),
        output_rfft=False,
        output_fftshift=True,
    )
    torch.testing.assert_close(
        wf.average_grid((32, 32), rfft=False, fftshift=True), other
    )


def test_whitening_curve_normalisation_and_fill():
    freq = frequency_axis(36)
    spectrum = 1.0 + 10.0 * freq**2
    curve = whitening_curve(spectrum, freq, max_freq=0.5)
    below = freq <= 0.5
    assert curve[below].max().item() == pytest.approx(1.0)
    assert torch.all(curve[~below] == 1.0)
    torch.testing.assert_close(
        curve[below], (1 / spectrum)[below] / (1 / spectrum)[below].max()
    )
    unlimited = whitening_curve(spectrum, freq, max_freq=None)
    assert unlimited.max().item() == pytest.approx(1.0) and unlimited[-1] < 1.0
    spectrum[3] = 0.0
    assert whitening_curve(spectrum, freq)[3] == 0.0
    with pytest.raises(ValueError):
        whitening_curve(spectrum[:-1], freq)
    grid = curve_to_grid(curve, freq, (20, 20))
    beyond = fftfreq_grid(image_shape=(20, 20), rfft=True, norm=True) > freq[-1]
    assert grid.shape == (20, 11) and not beyond.any()  # the axis reaches the corner


def _micrographs(records):
    """Generator of (name, image) pairs asserting the previous image was released."""
    previous = None
    for name, shape, scale in records:
        if previous is not None:
            gc.collect()
            assert previous() is None, "the filter held the previous micrograph"
        image = _image(*shape, seed=len(name), scale=scale)
        previous = weakref.ref(image)
        yield name, image
        del image
    gc.collect()
    assert previous() is None


def test_whitening_filter_streams_names_corrections_and_round_trip(tmp_path):
    records = [("a", (64, 64), 1.0), ("b", (64, 64), 2.0), ("c", (48, 48), 0.5)]
    wf = WhiteningFilter.from_images(_micrographs(records), max_freq=0.4)
    n_bins = calculate_num_freq_bins((64, 64))
    assert wf.names == ("a", "b", "c") and wf.n_images == 3
    assert wf.spectra.shape == (3, n_bins)  # one shared axis, set by the first image
    assert wf.spectra[1, 2:-6].mean().item() == pytest.approx(2 * 64, rel=0.1)
    assert wf.spectra[2, 2:-6].mean().item() == pytest.approx(0.5 * 48, rel=0.1)
    assert wf.index("b") == 1 and wf.index(-1) == 2
    with pytest.raises(KeyError):
        wf.index("z")
    with pytest.raises(IndexError):
        wf.index(3)
    torch.testing.assert_close(wf.average_spectrum, wf.spectra.square().mean(0).sqrt())
    torch.testing.assert_close(
        wf.average_1d, whitening_curve(wf.average_spectrum, wf.freq, 0.4)
    )
    c = wf.correction_1d("b")
    ok = wf.average_1d > 0  # the zeroed DC bin divides 0 / 0: C_i is 1 there
    torch.testing.assert_close(c[ok], (wf.filter_1d("b") / wf.average_1d)[ok])
    assert torch.all(c[~ok] == 1.0) and not ok[0]
    assert torch.all(c[wf.freq > 0.4] == 1.0)
    shape = (40, 40)
    torch.testing.assert_close(
        wf.image_grid("b", shape),
        wf.average_grid(shape) * wf.correction_grid("b", shape) ** 2,
    )
    assert wf.average_grid(shape, device="cpu").shape == (40, 21)
    # unnamed images are numbered; a plain list works too
    plain = WhiteningFilter.from_images([_image(32, 32), _image(32, 32, seed=1)])
    assert plain.names == ("0", "1")
    with pytest.raises(ValueError):
        WhiteningFilter.from_images([])

    path = tmp_path / "whitening.h5"
    wf.save(path)
    back = WhiteningFilter.load(path)
    assert back.names == wf.names and back.max_freq == wf.max_freq
    torch.testing.assert_close(back.freq, wf.freq)
    torch.testing.assert_close(back.spectra, wf.spectra)
    torch.testing.assert_close(back.average_1d, wf.average_1d)
    unlimited = WhiteningFilter(wf.freq, wf.spectra, wf.names, None)
    unlimited.save(path)
    assert WhiteningFilter.load(path).max_freq is None


def test_whitening_filter_validation():
    freq = frequency_axis(5)
    spectra = torch.ones(2, 5)
    WhiteningFilter(freq, spectra, ("x", "y"))
    with pytest.raises(ValueError):
        WhiteningFilter(freq, spectra, ("x", "x"))
    with pytest.raises(ValueError):
        WhiteningFilter(freq, spectra, ("x",))
    with pytest.raises(ValueError):
        WhiteningFilter(freq.double(), spectra, ("x", "y"))
    with pytest.raises(ValueError):
        WhiteningFilter(freq, -spectra, ("x", "y"))
    with pytest.raises(ValueError):
        WhiteningFilter(freq, spectra[:, :-1], ("x", "y"))
    with pytest.raises(ValueError):
        WhiteningFilter(freq, spectra, ("x", "y"), max_freq=0.0)
