"""Correctness for fused iRFFT + Parseval-moments + argmax CUDA kernel against torch.

Notes
-----
Requires a CUDA device AND a working cuFFTDx/nvidia-mathdx toolchain able to JIT-compile
the extension (`pip install panther-em[fused-kernels]`). Will be skipped entirely
otherwise (e.g. on CI runners which lack GPUs or Nvidia libraries).
"""

from __future__ import annotations

import warnings

import pytest
import torch

from panther_em.inference.search import fused_kernel_loader
from panther_em.inference.search.statistics import _reduce_stats

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


@pytest.fixture(scope="module", autouse=True)
def _require_compiled_kernel():
    """Probing may raise 'kernel unavailable' UserWarning, so skip test if warned."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        configs = fused_kernel_loader.get_supported_configs()
    if configs is None:
        pytest.skip("fused kernel did not compile (toolchain/headers unavailable)")


def _reference(spectrum: torch.Tensor, n_psi: int):
    corr = torch.fft.irfft(spectrum, n=n_psi, dim=-1, norm="forward")
    return _reduce_stats(corr, torch.view_as_real(spectrum))


def _make_spectrum(p: int, q: int, n_freq: int) -> torch.Tensor:
    """Random complex64 spectrum with a real-valued DC bin to match iRFFT semantics."""
    c = torch.randn(p, q, n_freq, dtype=torch.complex64, device="cuda")
    c[..., 0] = c[..., 0].real.to(torch.complex64)
    return c


@pytest.mark.parametrize(
    ("p", "q"),
    [(1, 1), (1, 17), (8, 8), (16, 100), (64, 512)],
)
@pytest.mark.parametrize("n_psi", [64, 128, 256])
def test_fused_matches_reference_across_shapes(n_psi, p, q):
    all_configs = fused_kernel_loader.get_supported_configs()
    configs = {cfg[1] for cfg in all_configs if cfg[0] == n_psi}
    n_freq = min(configs)  # smallest supported NumFreq for this n_psi

    torch.manual_seed(p * 1000 + q * 7 + n_psi)
    spectrum = _make_spectrum(p, q, n_freq)

    s1, s2, vmax, amax = fused_kernel_loader.fused_irfft_stats(spectrum, n_psi)
    ref_s1, ref_s2, ref_vmax, ref_amax = _reference(spectrum, n_psi)

    torch.testing.assert_close(s1, ref_s1, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(s2, ref_s2, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(vmax, ref_vmax, rtol=1e-4, atol=1e-4)
    assert torch.equal(amax, ref_amax), "argmax must match exactly (see kernel caveats)"


def test_fused_matches_reference_nyquist_config():
    """n_psi=64, NumFreq=33 (== n_psi/2+1) exercises Nyquist-bin correction branch."""
    n_psi, n_freq = 64, 33
    configs = {(cfg[0], cfg[1]) for cfg in fused_kernel_loader.get_supported_configs()}
    if (n_psi, n_freq) not in configs:
        pytest.skip("Nyquist config not present in this build's SUPPORTED_CONFIGS")

    torch.manual_seed(64033)
    spectrum = _make_spectrum(50, 30, n_freq)

    s1, s2, vmax, amax = fused_kernel_loader.fused_irfft_stats(spectrum, n_psi)
    ref_s1, ref_s2, ref_vmax, ref_amax = _reference(spectrum, n_psi)

    torch.testing.assert_close(s1, ref_s1, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(s2, ref_s2, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(vmax, ref_vmax, rtol=1e-4, atol=1e-4)
    assert torch.equal(amax, ref_amax)


def test_unsupported_config_returns_none():
    spectrum = torch.randn(2, 3, 9, dtype=torch.complex64, device="cuda")
    assert fused_kernel_loader.fused_irfft_stats(spectrum, n_psi=100) is None


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires a second CUDA device"
)
def test_fused_matches_reference_on_non_default_device():
    """Regression test: the kernel launch must target `c`'s own device."""
    n_psi, n_freq = 256, 64
    device = torch.device("cuda:1")

    # Generated directly on `device` rather than on the default device and
    # then `.to(device)`-transferred. This environment has been
    # NOTE: Observed some silently corrupt (zero out) of rows of a tensor device-copied
    # to any CUDA kernel that later reads it on the destination device (even after a
    # full `torch.cuda.synchronize()`), although a `.cpu()` readback of the same tensor
    # shows the correct values. Unsure where this is coming from...
    torch.manual_seed(12345)
    spectrum = torch.randn(8, 40, n_freq, dtype=torch.complex64, device=device)
    spectrum[..., 0] = spectrum[..., 0].real.to(torch.complex64)

    s1, s2, vmax, amax = fused_kernel_loader.fused_irfft_stats(spectrum, n_psi)
    torch.cuda.synchronize(device)

    # Reference computed via plain (uncompiled) torch ops
    corr = torch.fft.irfft(spectrum, n=n_psi, dim=-1, norm="forward")
    spectrum_ri = torch.view_as_real(spectrum)
    re, im = spectrum_ri[..., 0], spectrum_ri[..., 1]
    dc_real = re[..., 0]
    power = re * re + im * im
    ref_s1 = n_psi * dc_real.sum(dim=1)
    per_hyp = n_psi * (dc_real * dc_real + 2.0 * power[..., 1:].sum(dim=2))
    ref_s2 = per_hyp.sum(dim=1)
    ref_vmax, ref_amax = corr.reshape(corr.shape[0], -1).max(dim=1)

    torch.testing.assert_close(s1, ref_s1, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(s2, ref_s2, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(vmax, ref_vmax, rtol=1e-4, atol=1e-4)
    assert torch.equal(amax, ref_amax)
