"""The reduction: the torch reference, the lean kernel and their block statistics.

- ``reduce_spectrum`` against the explicit correlogram (moments by Parseval, the
  Nyquist bin, the argmax);
- ``TorchBlockStats`` over several hypothesis batches and GEMM sub-batches against a
  dense reference, on CPU;
- the lean kernel against ``reduce_spectrum`` (shapes, the Nyquist bin, the
  contraction's native layout, rejected inputs) and ``LeanBlockStats`` against
  ``TorchBlockStats`` at every supported ``n_psi`` and spectrum precision, on CUDA;
- the packed ``(value, index)`` encoding, the build's warn-once failure path and the
  CUDA-graph probe.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from panther_em.search.correlation import Spectrum
from panther_em.search.correlation.kernels import fused_kernel_loader
from panther_em.search.correlation.kernels.packed import (
    decode_argmax_packed,
    encode_argmax_packed,
)
from panther_em.search.correlation.loop import cuda_graph_capture_supported
from panther_em.search.correlation.reduction import (
    LeanBlockStats,
    SpectrumSpec,
    TorchBlockStats,
    fused_stats_supported,
    make_block_stats,
    reduce_spectrum,
)
from panther_em.search.hypotheses import HypBatch

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.filterwarnings("ignore:ComplexHalf support is experimental")
requires_cuda = pytest.mark.skipif(not CUDA, reason="requires a CUDA device")
MAP_KEYS = ("mip", "mean", "variance", "zscore", "best_index", "best_psi")


def native_spectrum(n_freq, p, q, *, device="cpu", seed=0, dtype=torch.complex64):
    """Random ``(F, P, Q)``-contiguous spectrum with a real DC bin."""
    gen = torch.Generator(device=device).manual_seed(seed)
    c = torch.complex(
        torch.randn(n_freq, p, q, generator=gen, device=device),
        torch.randn(n_freq, p, q, generator=gen, device=device),
    )
    c[0] = c[0].real.to(c.dtype)
    return c.to(dtype).contiguous()


def correlogram(spectrum_pnf, n_psi):
    """``(P, N, n_psi)`` explicit float64 correlations: the real part of
    ``sum_k a_k C_k e^{2 pi i k psi / n_psi}`` with ``a = (1, 2, ..., 2, 1 | 2)``, the
    DC and a Nyquist bin by their real parts (no FFT library's convention involved)."""
    c = spectrum_pnf.to(torch.complex128).cpu()
    n_freq = c.shape[-1]
    k = torch.arange(n_freq, dtype=torch.float64)
    a = torch.full((n_freq,), 2.0, dtype=torch.float64)
    a[0] = 1.0
    if n_psi % 2 == 0 and n_freq == n_psi // 2 + 1:
        a[-1] = 1.0
    c = c * a
    c[..., 0] = c[..., 0].real.clone()
    if n_psi % 2 == 0 and n_freq == n_psi // 2 + 1:
        c[..., -1] = c[..., -1].real.clone()
    psi = torch.arange(n_psi, dtype=torch.float64)
    phase = torch.exp(2j * torch.pi * k[:, None] * psi[None, :] / n_psi)  # (F, n_psi)
    return (c @ phase).real


# --------------------------------------------------------------------------- #
# The reference reduction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("n_psi", "n_freq"), [(64, 17), (64, 33), (32, 5)])
def test_reduce_spectrum_matches_the_correlogram(n_psi, n_freq):
    """Parseval moments (incl. a Nyquist bin at n_freq = n_psi/2 + 1) and the max."""
    spectrum = native_spectrum(n_freq, 6, 9, seed=n_freq).permute(1, 2, 0)
    s1, s2, vmax, amax = reduce_spectrum(spectrum, n_psi)
    corr = correlogram(spectrum, n_psi).reshape(6, -1)
    torch.testing.assert_close(s1.double(), corr.sum(1), rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(s2.double(), (corr**2).sum(1), rtol=1e-4, atol=1e-3)
    want_v, want_a = corr.max(1)
    torch.testing.assert_close(vmax.double(), want_v, rtol=1e-5, atol=1e-5)
    assert torch.equal(amax, want_a)


def _batches(n_virtual, sizes):
    out, j0 = [], 0
    for size in sizes:
        out.append(HypBatch(0, j0, j0 + size, j0))
        j0 += size
    assert j0 == n_virtual
    return out


def _run_block(stats, spectra, batches, n_sub):
    stats.clear()
    for sub in range(n_sub):
        for batch, native in zip(batches, spectra[sub], strict=True):
            stats.consume(Spectrum(native), batch, sub)
    n = stats.n_sub * stats.gemm_pixels
    outs = {
        "mip": torch.empty(n),
        "mean": torch.empty(n),
        "variance": torch.empty(n),
        "zscore": torch.empty(n),
        "best_index": torch.empty(n, dtype=torch.long),
        "best_psi": torch.empty(n, dtype=torch.long),
    }
    stats.finalize_into(outs, 0, n)
    return outs


def _reference(spectra, n_psi):
    """Dense per-pixel maps over every batch, rows sub-batch major."""
    rows = []
    for per_sub in spectra:
        corr = torch.cat(
            [correlogram(native.permute(1, 2, 0).cpu(), n_psi) for native in per_sub],
            dim=1,
        )  # (P_g, N, n_psi)
        rows.append(corr)
    corr = torch.cat(rows)  # (n_sub * P_g, N, n_psi)
    flat = corr.reshape(corr.shape[0], -1)
    mip, amax = flat.max(1)
    mean = flat.mean(1)
    variance = (flat**2).mean(1) - mean**2
    unique = (flat == mip[:, None]).sum(1) == 1
    return {
        "mip": mip,
        "mean": mean,
        "variance": variance,
        "zscore": (mip - mean) / variance.sqrt(),
        "best_index": amax // n_psi,
        "best_psi": (n_psi - amax % n_psi) % n_psi,
        "unique": unique,
    }


def _assert_maps(got, want, *, rtol=1e-4, atol=1e-4):
    for key in ("mip", "mean", "variance", "zscore"):
        torch.testing.assert_close(
            got[key].cpu().double(), want[key], rtol=rtol, atol=atol
        )
    u = want["unique"]
    assert torch.equal(got["best_index"].cpu()[u], want["best_index"][u])
    assert torch.equal(got["best_psi"].cpu()[u], want["best_psi"][u])


def test_torch_block_stats_over_batches_and_sub_batches():
    n_sub, gemm, n_psi, n_freq, sizes = 2, 5, 32, 9, (4, 4, 3)
    batches = _batches(sum(sizes), sizes)
    spectra = [
        [
            native_spectrum(n_freq, gemm, b.size, seed=10 * s + i)
            for i, b in enumerate(batches)
        ]
        for s in range(n_sub)
    ]
    stats = TorchBlockStats(n_sub=n_sub, gemm_pixels=gemm, n_psi=n_psi, device="cpu")
    got = _run_block(stats, spectra, batches, n_sub)
    assert stats.n_evaluated == sum(sizes) * n_psi and not stats.fused
    _assert_maps(got, _reference(spectra, n_psi))
    # a second block starts from a clean state
    again = _run_block(stats, spectra, batches, n_sub)
    for key in MAP_KEYS:
        torch.testing.assert_close(again[key], got[key])


def test_block_stats_selection():
    cpu = SpectrumSpec(torch.complex64, 17, 64, 8, 1, 4, 12, torch.device("cpu"))
    assert not fused_stats_supported(cpu)
    assert isinstance(make_block_stats(cpu, fused=False), TorchBlockStats)
    if CUDA:
        cuda = torch.device("cuda")
        n_psi_64 = SpectrumSpec(torch.complex64, 17, 64, 8, 1, 4, 12, cuda)
        assert not fused_stats_supported(n_psi_64)  # the lean kernel needs 128+


# --------------------------------------------------------------------------- #
# The lean kernel
# --------------------------------------------------------------------------- #


@pytest.fixture
def lean():
    if not CUDA:
        pytest.skip("requires a CUDA device")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fused_kernel_loader._try_compile() is None:
            pytest.skip("the lean kernel did not compile")


@requires_cuda
@pytest.mark.parametrize(("p", "q"), [(1, 1), (1, 17), (16, 100), (64, 512)])
@pytest.mark.parametrize(
    ("n_psi", "n_freq"), [(128, 33), (256, 64), (256, 129), (512, 200)]
)
def test_lean_kernel_matches_reduce_spectrum(lean, n_psi, n_freq, p, q):
    """Every shape, incl. the Nyquist bin (n_freq = n_psi/2 + 1) and bins past 64."""
    native = native_spectrum(n_freq, p, q, device="cuda", seed=n_psi + n_freq + q)
    s1, s2, vmax, amax = fused_kernel_loader.lean_irfft_stats_transposed(native, n_psi)
    ref = reduce_spectrum(native.permute(1, 2, 0), n_psi)
    torch.testing.assert_close(s1, ref[0], rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(s2, ref[1], rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(vmax, ref[2], rtol=1e-4, atol=1e-4)
    corr = correlogram(native.permute(1, 2, 0), n_psi).reshape(p, -1)
    unique = (corr == corr.max(1, keepdim=True).values).sum(1) == 1
    assert torch.equal(amax[unique], ref[3][unique])


@requires_cuda
def test_lean_kernel_reads_the_contraction_output_in_place(lean):
    """``torch.bmm``'s ``(F, P, N)`` output is consumed without a copy."""
    n_psi, n_freq, p, n = 256, 64, 12, 40
    gen = torch.Generator(device="cuda").manual_seed(9001)
    y = torch.randn(p, n_freq, 5, dtype=torch.complex64, device="cuda", generator=gen)
    w = torch.randn(n, n_freq, 5, dtype=torch.complex64, device="cuda", generator=gen)
    native = torch.bmm(y.permute(1, 0, 2), w.conj().permute(1, 2, 0))  # (F, P, N)
    native[0] = native[0].real.to(native.dtype)
    _, s2, vmax, _ = fused_kernel_loader.lean_irfft_stats_transposed(native, n_psi)
    ref = reduce_spectrum(native.permute(1, 2, 0), n_psi)
    torch.testing.assert_close(s2, ref[1], rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(vmax, ref[2], rtol=1e-4, atol=1e-4)


@requires_cuda
def test_lean_kernel_rejects_what_it_does_not_cover(lean):
    native = native_spectrum(64, 12, 40, device="cuda")
    assert fused_kernel_loader.lean_irfft_stats_transposed(native, 100) is None
    assert fused_kernel_loader.lean_irfft_stats_transposed(native, 64) is None
    too_many_bins = native_spectrum(66, 4, 4, device="cuda")
    assert fused_kernel_loader.lean_irfft_stats_transposed(too_many_bins, 128) is None
    # a non-contiguous (F, P, Q) view is refused, not silently copied
    view = native_spectrum(12, 40, 64, device="cuda").permute(2, 0, 1)
    assert not view.is_contiguous()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert fused_kernel_loader.lean_irfft_stats_transposed(view, 256) is None


@requires_cuda
@pytest.mark.parametrize("n", [64, 128, 256])
def test_lean_debug_ifft_is_the_inverse_dft(lean, n):
    gen = torch.Generator(device="cuda").manual_seed(n)
    x = torch.randn(9, n, dtype=torch.complex64, device="cuda", generator=gen)
    got = fused_kernel_loader.lean_debug_ifft(x)
    torch.testing.assert_close(
        got, torch.fft.ifft(x, norm="forward"), rtol=1e-4, atol=1e-4
    )


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex32])
@pytest.mark.parametrize("n_psi", [128, 256, 512])
def test_lean_block_stats_match_torch(lean, n_psi, dtype):
    """In-kernel accumulation over batches with virtual offsets and sub-batches equals
    the torch reference (n_freq past the 64-bin base transform: the fold pass)."""
    n_sub, gemm, n_freq, sizes = 2, 8, n_psi // 4 + 1, (12, 12, 5)
    batches = _batches(sum(sizes), sizes)
    spectra = [
        [
            native_spectrum(
                n_freq, gemm, b.size, device="cuda", seed=7 * s + i, dtype=dtype
            )
            for i, b in enumerate(batches)
        ]
        for s in range(n_sub)
    ]
    device = torch.device("cuda")
    lean_stats = LeanBlockStats(
        n_sub=n_sub, gemm_pixels=gemm, n_psi=n_psi, device=device
    )
    torch_stats = TorchBlockStats(
        n_sub=n_sub, gemm_pixels=gemm, n_psi=n_psi, device=device
    )
    got = _run_block(lean_stats, spectra, batches, n_sub)
    want = _run_block(torch_stats, spectra, batches, n_sub)
    assert lean_stats.fused and lean_stats.n_evaluated == torch_stats.n_evaluated
    tol = 1e-4 if dtype is torch.complex64 else 2e-3
    ref = _reference(spectra, n_psi)
    _assert_maps(got, ref, rtol=tol, atol=tol)
    _assert_maps(want, ref, rtol=tol, atol=tol)


# --------------------------------------------------------------------------- #
# Packing, the build and the graph probe
# --------------------------------------------------------------------------- #


def test_encode_decode_argmax_packed_round_trip():
    torch.manual_seed(0)
    n = 4096
    val = torch.randn(n, dtype=torch.float32) * 1e3
    val[:4] = torch.tensor([0.0, -0.0, float("inf"), float("-inf")])
    idx = torch.randint(0, 2**31 - 1, (n,), dtype=torch.int64)
    decoded_val, decoded_idx = decode_argmax_packed(encode_argmax_packed(val, idx))
    assert torch.equal(decoded_idx, idx) and torch.equal(decoded_val, val)


def test_encode_argmax_packed_orders_like_the_values():
    """Unsigned order of the packed words is the order of the floats (atomicMax)."""
    torch.manual_seed(1)
    val = torch.randn(2048, dtype=torch.float32) * 1e6
    packed = encode_argmax_packed(val, torch.arange(2048, dtype=torch.int64))
    order = np.argsort(packed.numpy().astype(np.uint64))
    assert np.array_equal(order, np.argsort(val.numpy()))


def test_build_failure_warns_once(monkeypatch):
    monkeypatch.setattr(fused_kernel_loader, "_warned_compile_failure", False)
    fused_kernel_loader._try_compile.cache_clear()

    def boom(*args, **kwargs):
        raise RuntimeError("simulated build failure")

    monkeypatch.setattr(fused_kernel_loader, "load", boom)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 9))
    with pytest.warns(UserWarning, match="kernel unavailable"):
        assert fused_kernel_loader._try_compile() is None
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert fused_kernel_loader._try_compile() is None  # cached: no second warning
    assert not record
    fused_kernel_loader._try_compile.cache_clear()


@requires_cuda
def test_cuda_graph_capture_probe():
    assert cuda_graph_capture_supported(torch.cuda.current_device())
