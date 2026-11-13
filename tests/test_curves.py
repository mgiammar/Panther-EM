"""filters.curves: the frequency axis, amplitude spectra, whitening curves."""

from __future__ import annotations

import pytest
import torch
from torch_fourier_filter.whitening import calculate_num_freq_bins
from torch_grid_utils.fftfreq_grid import fftfreq_grid

from panther_em.filters.curves import (
    amplitude_spectrum_1d,
    curve_to_grid,
    frequency_axis,
    whitening_curve,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


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


@requires_cuda
def test_cuda_images_give_the_cpu_spectra():
    """A CUDA ``rfft2`` is not contiguous; the spectra must still bin it."""
    x = _image(96, 80, seed=4, scale=2.0)
    freq, cpu = amplitude_spectrum_1d(x)
    freq_gpu, gpu = amplitude_spectrum_1d(x.cuda())
    torch.testing.assert_close(gpu.cpu(), cpu, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(freq_gpu.cpu(), freq)
