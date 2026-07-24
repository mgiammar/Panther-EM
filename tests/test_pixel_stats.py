"""Unit tests for PixelStats and its fused-CUDA-kernel specialization.

Covers
------
- PixelStats.update / finalize against a hand-derived reference.
- FusedPixelStats falling back to the pure-torch reduction: on a CPU spectrum
  (no CUDA at all), when the fused kernel fails to compile, and when
  (n_psi, NumFreq) isn't one of the kernel's supported configs.
- The fused-kernel-unavailable warning fires exactly once per process.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from panther_em.inference.search import fused_kernel_loader
from panther_em.inference.search.fused_statistics import FusedPixelStats
from panther_em.inference.search.statistics import PixelStats, _reduce_stats


def _make_spectrum(
    num_pixels: int, hyp_batch: int, n_freq: int, device: str = "cpu", seed: int = 0
) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    real = torch.randn(num_pixels, hyp_batch, n_freq, generator=gen, device=device)
    imag = torch.randn(num_pixels, hyp_batch, n_freq, generator=gen, device=device)
    return torch.complex(real, imag).to(torch.complex64)


def test_pixel_stats_update_matches_reference_reduce_stats():
    num_pixels, hyp_batch, n_freq, num_psi = 4, 6, 9, 16
    spectrum = _make_spectrum(num_pixels, hyp_batch, n_freq)
    hyp_global_idx = torch.arange(100, 100 + hyp_batch)

    stats = PixelStats(num_pixels, device=torch.device("cpu"))
    stats.update(spectrum, hyp_global_idx, num_psi=num_psi)
    result = stats.finalize()

    corr = torch.fft.irfft(spectrum, n=num_psi, dim=-1, norm="forward")
    s1, s2, vmax, amax = _reduce_stats(corr, torch.view_as_real(spectrum))
    mean = s1 / (hyp_batch * num_psi)
    variance = (s2 / (hyp_batch * num_psi) - mean**2).clamp_min(0.0)
    zscore = (vmax - mean) / variance.sqrt().clamp_min(1e-12)
    n_local = torch.div(amax, num_psi, rounding_mode="floor")
    expected_best_hyp = hyp_global_idx[n_local]
    expected_psi = (num_psi - amax % num_psi) % num_psi

    assert torch.allclose(result["mip"], vmax)
    assert torch.allclose(result["mean"], mean)
    assert torch.allclose(result["variance"], variance)
    assert torch.allclose(result["zscore"], zscore)
    assert torch.equal(result["best_index"], expected_best_hyp)
    assert torch.equal(result["best_psi"], expected_psi)


def test_pixel_stats_clear_resets_accumulators():
    stats = PixelStats(3, device=torch.device("cpu"))
    spectrum = _make_spectrum(3, 4, 5)
    stats.update(spectrum, torch.arange(4), num_psi=8)
    stats.clear()

    assert stats.hypothesis_count == 0
    assert torch.all(stats.corr_sum == 0)
    assert torch.all(stats.corr_sum2 == 0)
    assert torch.all(stats.best_corr == float("-inf"))
    assert torch.all(stats.best_hypothesis == -1)
    assert torch.all(stats.best_psi_angle == -1)


def test_fused_pixel_stats_matches_plain_on_cpu_spectrum():
    """A CPU spectrum can never take the CUDA-kernel branch (spectrum.is_cuda is
    always False), so this needs no GPU and always runs the same on any machine."""
    num_pixels, hyp_batch, n_freq, num_psi = 5, 7, 12, 32
    spectrum = _make_spectrum(num_pixels, hyp_batch, n_freq)
    hyp_global_idx = torch.arange(hyp_batch)

    plain = PixelStats(num_pixels, device=torch.device("cpu"))
    plain.update(spectrum, hyp_global_idx, num_psi=num_psi)

    fused = FusedPixelStats(num_pixels, device=torch.device("cpu"))
    fused.update(spectrum, hyp_global_idx, num_psi=num_psi)

    plain_result, fused_result = plain.finalize(), fused.finalize()
    for key in plain_result:
        assert torch.allclose(plain_result[key].float(), fused_result[key].float()), key


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_fused_pixel_stats_falls_back_when_compile_unavailable(monkeypatch):
    monkeypatch.setattr(fused_kernel_loader, "_try_compile", lambda: None)

    num_pixels, hyp_batch, n_freq, num_psi = 4, 6, 9, 16
    spectrum = _make_spectrum(num_pixels, hyp_batch, n_freq, device="cuda")
    hyp_global_idx = torch.arange(hyp_batch, device="cuda")

    plain = PixelStats(num_pixels, device=torch.device("cuda"))
    plain.update(spectrum, hyp_global_idx, num_psi=num_psi)

    fused = FusedPixelStats(num_pixels, device=torch.device("cuda"))
    fused.update(spectrum, hyp_global_idx, num_psi=num_psi)

    plain_result, fused_result = plain.finalize(), fused.finalize()
    for key in plain_result:
        assert torch.allclose(plain_result[key].float(), fused_result[key].float()), key


def test_fused_kernel_loader_returns_none_for_unsupported_config(monkeypatch):
    """A non-power-of-two n_psi is a normal, expected condition (not every caller's
    n_psi is restricted to the kernel's supported grid) -- must return None quietly,
    not raise, regardless of whether a compiled kernel is even available here."""
    monkeypatch.setattr(
        fused_kernel_loader,
        "_try_compile",
        lambda: type(
            "_FakeModule",
            (),
            {"get_supported_configs": staticmethod(lambda: [(64, 33, 8, 8)])},
        )(),
    )
    spectrum = _make_spectrum(2, 3, 9)  # (n_psi=100, n_freq=9) not in fake config list
    result = fused_kernel_loader.fused_irfft_stats(spectrum, n_psi=100)
    assert result is None


def test_fused_kernel_loader_warns_once_on_compile_failure(monkeypatch):
    monkeypatch.setattr(fused_kernel_loader, "_warned_compile_failure", False)
    fused_kernel_loader._try_compile.cache_clear()

    def _boom():
        raise RuntimeError("simulated header-discovery failure")

    monkeypatch.setattr(fused_kernel_loader, "_find_cufftdx_includes", _boom)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.warns(UserWarning, match="Fused iRFFT\\+stats CUDA kernel unavailable"):
        assert fused_kernel_loader._try_compile() is None

    # Second call is served from the lru_cache -- must not warn again.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert fused_kernel_loader._try_compile() is None
    assert len(record) == 0, [str(r.message) for r in record]

    fused_kernel_loader._try_compile.cache_clear()
