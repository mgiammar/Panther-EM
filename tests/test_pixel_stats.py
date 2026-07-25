"""Unit tests for PixelStats and its fused-CUDA-kernel specialization.

Covers
------
- PixelStats.update / finalize against a hand-derived reference.
- FusedPixelStats falling back to the pure-torch reduction: on a CPU spectrum
  (no CUDA at all), when the fused kernel fails to compile, and when
  (n_psi, NumFreq) isn't one of the kernel's supported configs.
- The fused-kernel-unavailable warning fires exactly once per process.
- encode_argmax_packed / decode_argmax_packed round-trip correctness.
- The batch-then-reduce accumulation path (begin_hypothesis_batches /
  accumulate_batch / reduce_hypothesis_batches) matches repeated update() calls,
  for both the contiguous-arange offset fast path and the gather fallback.
- The CUDA-graph-captured streaming accumulate path (accumulate_graphed /
  update_graphed) matches repeated update() calls, reuses one captured graph
  across varying hypothesis-batch sizes, and falls back cleanly on CPU.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from panther_em.inference.search import fused_kernel_loader
from panther_em.inference.search.fused_statistics import FusedPixelStats
from panther_em.inference.search.statistics import (
    PixelStats,
    _cuda_graph_capture_supported,
    _reduce_stats,
    decode_argmax_packed,
    encode_argmax_packed,
)


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


# --------------------------------------------------------------------------- #
# encode_argmax_packed / decode_argmax_packed
# --------------------------------------------------------------------------- #
def test_encode_decode_argmax_packed_round_trip():
    torch.manual_seed(0)
    n = 4096
    val = torch.randn(n, dtype=torch.float32) * 1e3
    val[0] = 0.0
    val[1] = -0.0
    val[2] = float("inf")
    val[3] = float("-inf")
    idx = torch.randint(0, 2**31 - 1, (n,), dtype=torch.int64)

    packed = encode_argmax_packed(val, idx)
    decoded_val, decoded_idx = decode_argmax_packed(packed)

    assert torch.equal(decoded_idx, idx)
    assert torch.equal(decoded_val, val)


def test_encode_argmax_packed_preserves_order_via_unsigned_comparison():
    """The packing's whole point: sorting PACKED values by *unsigned* integer order
    must match sorting the original floats by value -- this is the property
    decode_argmax_packed's docstring warns not to rely on via plain (signed) torch
    comparison. Verified here via numpy's uint64 view, independent of torch."""
    import numpy as np

    torch.manual_seed(1)
    val = torch.randn(2048, dtype=torch.float32) * 1e6
    idx = torch.arange(2048, dtype=torch.int64)
    packed = encode_argmax_packed(val, idx)

    packed_u64 = packed.numpy().astype(np.uint64)
    order_by_packed = np.argsort(packed_u64)
    order_by_val = np.argsort(val.numpy())
    assert np.array_equal(order_by_packed, order_by_val)


# --------------------------------------------------------------------------- #
# Batch-then-reduce accumulation path
# --------------------------------------------------------------------------- #
def _run_streaming(
    stats_cls, spectra, hyp_idx_batches, num_psi, reverse_psi_axis=True, device="cpu"
):
    stats = stats_cls(spectra[0].shape[0], device=torch.device(device))
    for spectrum, hyp_idx in zip(spectra, hyp_idx_batches, strict=False):
        stats.update(
            spectrum, hyp_idx, num_psi=num_psi, reverse_psi_axis=reverse_psi_axis
        )
    return stats.finalize()


def _run_batched(
    stats_cls,
    spectra,
    hyp_idx_batches,
    num_psi,
    hyp_offsets=None,
    reverse_psi_axis=True,
    device="cpu",
):
    stats = stats_cls(spectra[0].shape[0], device=torch.device(device))
    stats.begin_hypothesis_batches(len(spectra))
    offsets = hyp_offsets if hyp_offsets is not None else [None] * len(spectra)
    for spectrum, hyp_idx, offset in zip(
        spectra, hyp_idx_batches, offsets, strict=False
    ):
        stats.accumulate_batch(spectrum, hyp_idx, num_psi=num_psi, hyp_offset=offset)
    stats.reduce_hypothesis_batches(num_psi=num_psi, reverse_psi_axis=reverse_psi_axis)
    return stats.finalize()


def _assert_results_match(expected, actual):
    for key in expected:
        if key in ("best_index", "best_psi"):
            assert torch.equal(expected[key], actual[key]), key
        else:
            assert torch.allclose(expected[key], actual[key], atol=1e-4), key


@pytest.mark.parametrize("stats_cls", [PixelStats, FusedPixelStats])
@pytest.mark.parametrize("reverse_psi_axis", [True, False])
def test_batched_accumulation_matches_streaming_offset_fast_path(
    stats_cls, reverse_psi_axis
):
    """Contiguous-arange hypothesis ids per batch -- exercises the additive
    (no-gather) reconstruction of the winning global hypothesis id."""
    num_pixels, n_freq, num_psi = 5, 9, 16
    hyp_per_batch, n_batches = 6, 4

    spectra = [
        _make_spectrum(num_pixels, hyp_per_batch, n_freq, seed=100 + b)
        for b in range(n_batches)
    ]
    hyp_idx_batches = [
        torch.arange(b * hyp_per_batch, (b + 1) * hyp_per_batch)
        for b in range(n_batches)
    ]
    hyp_offsets = [b * hyp_per_batch for b in range(n_batches)]

    expected = _run_streaming(
        stats_cls, spectra, hyp_idx_batches, num_psi, reverse_psi_axis
    )
    actual = _run_batched(
        stats_cls,
        spectra,
        hyp_idx_batches,
        num_psi,
        hyp_offsets=hyp_offsets,
        reverse_psi_axis=reverse_psi_axis,
    )
    _assert_results_match(expected, actual)


@pytest.mark.parametrize("stats_cls", [PixelStats, FusedPixelStats])
def test_batched_accumulation_matches_streaming_gather_fallback(stats_cls):
    """Shuffled/non-contiguous hypothesis ids per batch (hyp_offset=None throughout)
    -- exercises the stored-hyp_idx-stack gather reconstruction path."""
    num_pixels, n_freq, num_psi = 5, 9, 16
    hyp_per_batch, n_batches = 6, 4

    torch.manual_seed(7)
    spectra = [
        _make_spectrum(num_pixels, hyp_per_batch, n_freq, seed=200 + b)
        for b in range(n_batches)
    ]
    hyp_idx_batches = [torch.randperm(1000)[:hyp_per_batch] for _ in range(n_batches)]

    expected = _run_streaming(stats_cls, spectra, hyp_idx_batches, num_psi)
    actual = _run_batched(stats_cls, spectra, hyp_idx_batches, num_psi)
    _assert_results_match(expected, actual)


def test_reduce_hypothesis_batches_is_noop_without_staged_batches():
    stats = PixelStats(3, device=torch.device("cpu"))
    stats.begin_hypothesis_batches(2)
    before = stats.finalize()
    stats.reduce_hypothesis_batches(num_psi=16)
    after = stats.finalize()
    for key in before:
        assert torch.equal(
            torch.nan_to_num(before[key]), torch.nan_to_num(after[key])
        ), key
    assert stats.hypothesis_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_fused_batched_accumulation_matches_streaming_on_cuda():
    """End-to-end on an actual CUDA device: if the fused kernel is unavailable here,
    both paths transparently fall back to the same pure-torch reduction, so this
    still validates the batched-accumulation control flow either way."""
    num_pixels, n_freq, num_psi = 8, 33, 64
    hyp_per_batch, n_batches = 12, 5

    spectra = [
        _make_spectrum(num_pixels, hyp_per_batch, n_freq, device="cuda", seed=300 + b)
        for b in range(n_batches)
    ]
    hyp_idx_batches = [
        torch.arange(b * hyp_per_batch, (b + 1) * hyp_per_batch, device="cuda")
        for b in range(n_batches)
    ]
    hyp_offsets = [b * hyp_per_batch for b in range(n_batches)]

    expected = _run_streaming(
        FusedPixelStats, spectra, hyp_idx_batches, num_psi, device="cuda"
    )
    actual = _run_batched(
        FusedPixelStats,
        spectra,
        hyp_idx_batches,
        num_psi,
        hyp_offsets=hyp_offsets,
        device="cuda",
    )
    for key in expected:
        if key in ("best_index", "best_psi"):
            assert torch.equal(expected[key], actual[key]), key
        else:
            assert torch.allclose(
                expected[key].float(), actual[key].float(), atol=1e-3
            ), key


# --------------------------------------------------------------------------- #
# CUDA-graph-captured streaming accumulate (accumulate_graphed / update_graphed)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_cuda_graph_capture_supported_probe_is_true_on_this_device():
    assert _cuda_graph_capture_supported(torch.device("cuda").index or 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("stats_cls", [PixelStats, FusedPixelStats])
def test_update_graphed_matches_streaming_update_varying_hyp_batch_size(stats_cls):
    """Feed batches of DIFFERENT hypothesis-batch sizes through the same captured
    graph -- accumulate's own cost (and thus its graph) doesn't depend on
    hyp_batch size, so one capture must stay correct across all of them."""
    num_pixels, n_freq, num_psi = 6, 9, 16
    hyp_sizes = [6, 6, 3, 6, 2, 6]  # varying sizes, including a small tail-like batch

    torch.manual_seed(11)
    spectra = []
    offsets = []
    h0 = 0
    for i, m in enumerate(hyp_sizes):
        spectra.append(
            _make_spectrum(num_pixels, m, n_freq, device="cuda", seed=500 + i)
        )
        offsets.append(h0)
        h0 += m
    hyp_idx_batches = [
        torch.arange(off, off + m, device="cuda")
        for off, m in zip(offsets, hyp_sizes, strict=False)
    ]

    expected = _run_streaming(
        stats_cls, spectra, hyp_idx_batches, num_psi, device="cuda"
    )

    graphed = stats_cls(num_pixels, device=torch.device("cuda"))
    for spectrum, off in zip(spectra, offsets, strict=False):
        graphed.update_graphed(spectrum, off, num_psi=num_psi, reverse_psi_axis=True)
    actual = graphed.finalize()

    _assert_results_match(expected, actual)
    assert graphed.hypothesis_count == sum(hyp_sizes) * num_psi


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_update_graphed_rejects_mismatched_num_psi_after_capture():
    num_pixels, n_freq = 4, 9
    stats = PixelStats(num_pixels, device=torch.device("cuda"))
    spectrum = _make_spectrum(num_pixels, 5, n_freq, device="cuda", seed=1)
    stats.update_graphed(spectrum, hyp_offset=0, num_psi=16, reverse_psi_axis=True)

    other = _make_spectrum(num_pixels, 5, n_freq, device="cuda", seed=2)
    with pytest.raises(ValueError, match="specialized to"):
        stats.update_graphed(other, hyp_offset=5, num_psi=32, reverse_psi_axis=True)


def test_update_graphed_falls_back_to_update_on_cpu_spectrum():
    """A CPU spectrum can never satisfy `spectrum.is_cuda`, so this must transparently
    behave exactly like update() -- no GPU required, always runs the same anywhere."""
    num_pixels, hyp_batch, n_freq, num_psi = 5, 7, 12, 32
    spectrum = _make_spectrum(num_pixels, hyp_batch, n_freq)

    streaming = PixelStats(num_pixels, device=torch.device("cpu"))
    streaming.update(spectrum, torch.arange(hyp_batch), num_psi=num_psi)

    graphed = PixelStats(num_pixels, device=torch.device("cpu"))
    graphed.update_graphed(
        spectrum, hyp_offset=0, num_psi=num_psi, reverse_psi_axis=True
    )

    _assert_results_match(streaming.finalize(), graphed.finalize())
    assert graphed.hypothesis_count == streaming.hypothesis_count
