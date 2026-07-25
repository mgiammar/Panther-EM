"""Correctness for the zero-copy (NumFreq, P, Q) fused kernel entry point.

Mirrors ``test_fused_kernel_loader.py``'s coverage for
``fused_kernel_loader.fused_irfft_stats_transposed``, the kernel path that
consumes frequency-outermost input directly (no permute/copy) instead of the
``(P, Q, NumFreq)`` layout ``fused_irfft_stats`` expects.

Notes
-----
Requires a CUDA device AND a working cuFFTDx/nvidia-mathdx toolchain able to
JIT-compile the extension (`pip install panther-em[fused-kernels]`). Will be
skipped entirely otherwise (e.g. on CI runners which lack GPUs or Nvidia
libraries).
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


def _reference(spectrum_pqn: torch.Tensor, n_psi: int):
    """Reference computed from a (P, Q, NumFreq)-shaped view (logical values)."""
    corr = torch.fft.irfft(spectrum_pqn, n=n_psi, dim=-1, norm="forward")
    return _reduce_stats(corr, torch.view_as_real(spectrum_pqn))


def _make_spectrum_transposed(p: int, q: int, n_freq: int) -> torch.Tensor:
    """Random complex64 spectrum, (NumFreq, P, Q)-contiguous, real-valued DC bin."""
    c = torch.randn(n_freq, p, q, dtype=torch.complex64, device="cuda")
    c[0, ...] = c[0, ...].real.to(torch.complex64)  # DC bin (k=0) is real-valued
    return c


@pytest.mark.parametrize(
    ("p", "q"),
    [(1, 1), (1, 17), (8, 8), (16, 100), (64, 512)],
)
@pytest.mark.parametrize("n_psi", [64, 128, 256])
def test_transposed_matches_reference_across_shapes(n_psi, p, q):
    all_configs = fused_kernel_loader.get_supported_configs()
    configs = {cfg[1] for cfg in all_configs if cfg[0] == n_psi}
    n_freq = min(configs)  # smallest supported NumFreq for this n_psi

    torch.manual_seed(p * 1000 + q * 7 + n_psi)
    spectrum_kpq = _make_spectrum_transposed(p, q, n_freq)
    assert spectrum_kpq.is_contiguous()

    result = fused_kernel_loader.fused_irfft_stats_transposed(spectrum_kpq, n_psi)
    assert result is not None
    s1, s2, vmax, amax = result

    spectrum_pqn = spectrum_kpq.permute(1, 2, 0)  # (P, Q, NumFreq) view, non-contig
    ref_s1, ref_s2, ref_vmax, ref_amax = _reference(spectrum_pqn, n_psi)

    torch.testing.assert_close(s1, ref_s1, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(s2, ref_s2, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(vmax, ref_vmax, rtol=1e-4, atol=1e-4)
    assert torch.equal(amax, ref_amax), "argmax must match exactly (see kernel caveats)"


@pytest.mark.parametrize("n_freq", [128, 129])
def test_transposed_matches_reference_bank_conflict_boundary(n_freq):
    """n_freq=128 (multiple of 16, exercises the +1 shared-tile padding) vs 129 (not).

    (p, q) = (16, 100) matches the existing test suite's proven-safe shape
    range (see test_fused_kernel_loader.py's own (16, 100) case) and still
    exercises the trailing partial-Q-tile masking (100 % fpb=8 != 0). Larger
    (p, q) with n_freq=129 (Nyquist) was observed to trip a pre-existing
    precision quirk in the `_reduce_stats` reference itself -- confirmed
    present identically against the original (untouched) fused_irfft_stats
    kernel at that scale, so it's unrelated to this new code path and out of
    scope here.
    """
    n_psi = 256
    configs = {(cfg[0], cfg[1]) for cfg in fused_kernel_loader.get_supported_configs()}
    if (n_psi, n_freq) not in configs:
        pytest.skip(
            f"({n_psi}, {n_freq}) not present in this build's SUPPORTED_CONFIGS"
        )

    torch.manual_seed(256000 + n_freq)
    p, q = 16, 100
    spectrum_kpq = _make_spectrum_transposed(p, q, n_freq)

    s1, s2, vmax, amax = fused_kernel_loader.fused_irfft_stats_transposed(
        spectrum_kpq, n_psi
    )
    spectrum_pqn = spectrum_kpq.permute(1, 2, 0)
    ref_s1, ref_s2, ref_vmax, ref_amax = _reference(spectrum_pqn, n_psi)

    torch.testing.assert_close(s1, ref_s1, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(s2, ref_s2, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(vmax, ref_vmax, rtol=1e-4, atol=1e-4)
    assert torch.equal(amax, ref_amax)


def test_transposed_matches_reference_nyquist_config():
    """n_psi=64, NumFreq=33 (== n_psi/2+1) exercises Nyquist-bin correction branch."""
    n_psi, n_freq = 64, 33
    configs = {(cfg[0], cfg[1]) for cfg in fused_kernel_loader.get_supported_configs()}
    if (n_psi, n_freq) not in configs:
        pytest.skip("Nyquist config not present in this build's SUPPORTED_CONFIGS")

    torch.manual_seed(64033)
    spectrum_kpq = _make_spectrum_transposed(50, 30, n_freq)

    s1, s2, vmax, amax = fused_kernel_loader.fused_irfft_stats_transposed(
        spectrum_kpq, n_psi
    )
    spectrum_pqn = spectrum_kpq.permute(1, 2, 0)
    ref_s1, ref_s2, ref_vmax, ref_amax = _reference(spectrum_pqn, n_psi)

    torch.testing.assert_close(s1, ref_s1, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(s2, ref_s2, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(vmax, ref_vmax, rtol=1e-4, atol=1e-4)
    assert torch.equal(amax, ref_amax)


def test_transposed_matches_reference_via_bmm_native_output():
    """End-to-end sanity: feed the *actual* torch.bmm-native output shape/strides.

    ``_contract_region``'s single-region fast path returns
    ``torch.bmm(y_b, w_b).permute(1,2,0)`` un-materialized -- i.e. a
    ``(P, N, num_k)``-shaped VIEW whose underlying storage is genuinely
    ``(num_k, P, N)`` contiguous. ``spectrum.permute(2,0,1)`` recovers that
    contiguous storage without a copy. This test builds data the same way to
    confirm the new kernel is compatible with what production code will
    actually hand it once wired in (out of scope for this iteration, but the
    shape/stride shape should already work).
    """
    n_psi, n_freq, p, n = 256, 64, 12, 40
    torch.manual_seed(9001)

    y = torch.randn(p, n_freq, 5, dtype=torch.complex64, device="cuda")
    w = torch.randn(n, n_freq, 5, dtype=torch.complex64, device="cuda")
    bmm_out = torch.bmm(y.permute(1, 0, 2), w.conj().permute(1, 2, 0))  # (num_k,P,N)
    spectrum_pqn = bmm_out.permute(
        1, 2, 0
    )  # (P, N, num_k) view -- what tiling.py returns
    spectrum_pqn[..., 0] = spectrum_pqn[..., 0].real.to(torch.complex64)

    spectrum_kpq = spectrum_pqn.permute(2, 0, 1)  # recovers bmm_out's own storage
    assert spectrum_kpq.is_contiguous()
    assert spectrum_kpq.data_ptr() == bmm_out.data_ptr()

    s1, s2, vmax, amax = fused_kernel_loader.fused_irfft_stats_transposed(
        spectrum_kpq, n_psi
    )
    ref_s1, ref_s2, ref_vmax, ref_amax = _reference(spectrum_pqn, n_psi)

    torch.testing.assert_close(s1, ref_s1, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(s2, ref_s2, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(vmax, ref_vmax, rtol=1e-4, atol=1e-4)
    assert torch.equal(amax, ref_amax)


def test_transposed_rejects_non_contiguous_input_instead_of_copying():
    """The whole point of this entry point is to avoid a hidden copy.

    Passing a non-contiguous (NumFreq,P,Q)-shaped tensor must NOT silently
    succeed via an internal .contiguous() call -- it should fail (returning
    None through the Python wrapper), so callers fall back to
    fused_irfft_stats() instead.
    """
    n_psi, n_freq = 256, 64
    configs = {(cfg[0], cfg[1]) for cfg in fused_kernel_loader.get_supported_configs()}
    assert (n_psi, n_freq) in configs

    # (P, Q, NumFreq) contiguous, then permuted to (NumFreq, P, Q) -- same
    # logical shape as the fast path expects, but NOT contiguous in that layout.
    spectrum_pqn = torch.randn(12, 40, n_freq, dtype=torch.complex64, device="cuda")
    spectrum_kpq = spectrum_pqn.permute(2, 0, 1)
    assert not spectrum_kpq.is_contiguous()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fused_kernel_loader.fused_irfft_stats_transposed(spectrum_kpq, n_psi)
    assert result is None


def test_unsupported_config_returns_none():
    spectrum = torch.randn(9, 2, 3, dtype=torch.complex64, device="cuda")
    assert fused_kernel_loader.fused_irfft_stats_transposed(spectrum, n_psi=100) is None


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="requires a second CUDA device"
)
def test_transposed_matches_reference_on_non_default_device():
    """Regression test: the kernel launch must target `c`'s own device."""
    n_psi, n_freq = 256, 64
    device = torch.device("cuda:1")

    torch.manual_seed(12345)
    spectrum_kpq = torch.randn(n_freq, 8, 40, dtype=torch.complex64, device=device)
    spectrum_kpq[0, ...] = spectrum_kpq[0, ...].real.to(torch.complex64)

    s1, s2, vmax, amax = fused_kernel_loader.fused_irfft_stats_transposed(
        spectrum_kpq, n_psi
    )
    torch.cuda.synchronize(device)

    # Reference computed via plain (uncompiled) torch ops on the (P,Q,NumFreq) view.
    spectrum_pqn = spectrum_kpq.permute(1, 2, 0)
    corr = torch.fft.irfft(spectrum_pqn, n=n_psi, dim=-1, norm="forward")
    spectrum_ri = torch.view_as_real(spectrum_pqn)
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
