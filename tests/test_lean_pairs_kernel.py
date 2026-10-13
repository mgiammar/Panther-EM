"""Per-pair emission variant of the lean iRFFT + stats kernel vs torch.

Requires a CUDA device and the ``panther_em_lean_pairs`` extension (JIT-built from
``csrc/lean_pairs_1d.cu``); skipped entirely otherwise.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from panther_em.inference.search import fused_kernel_loader

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device"),
    pytest.mark.filterwarnings("ignore:ComplexHalf support is experimental"),
]


@pytest.fixture(scope="module", autouse=True)
def _require_compiled_kernels():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pairs = fused_kernel_loader._try_compile_pairs()
        dense = fused_kernel_loader._try_compile()
    if pairs is None or dense is None:
        pytest.skip("lean kernels did not compile (toolchain/headers unavailable)")


def _reference(spectrum_pqn: torch.Tensor, n_psi: int):
    """``(s1, s2, vmax, amax, pair_max)`` from a logical ``(P, Q, NumFreq)`` view.

    Eager torch (no ``torch.compile``): the moments are plain sums over the
    correlogram, so tiny shapes cannot trip the inductor.
    """
    corr = torch.fft.irfft(spectrum_pqn, n=n_psi, dim=-1, norm="forward")
    flat = corr.reshape(corr.shape[0], -1)
    s1 = flat.double().sum(dim=1).float()
    s2 = (flat.double() ** 2).sum(dim=1).float()
    vmax, amax = flat.max(dim=1)
    return s1, s2, vmax, amax, corr.amax(dim=-1)


def _make_spectrum(
    p: int, q: int, n_freq: int, seed: int, n_psi: int = 0
) -> torch.Tensor:
    """Random complex64 ``(NumFreq, P, Q)``-contiguous spectrum with a real DC bin.

    The Nyquist bin (``n_freq == n_psi // 2 + 1``) is made real as well: the kernel
    drops its imaginary part like the CPU FFT does, but cuFFT's C2R transform uses it
    for large batches, so a complex Nyquist bin would make the reference itself
    batch-size dependent.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed)
    c = torch.randn(n_freq, p, q, dtype=torch.complex64, device="cuda", generator=gen)
    c[0] = c[0].real.to(torch.complex64)
    if n_psi and n_freq == n_psi // 2 + 1:
        c[-1] = c[-1].real.to(torch.complex64)
    return c


def _outs(p: int) -> list[torch.Tensor]:
    sentinel = fused_kernel_loader.lean_sentinel_packed()
    assert sentinel is not None
    return [
        torch.zeros(p, dtype=torch.float32, device="cuda"),
        torch.zeros(p, dtype=torch.float32, device="cuda"),
        torch.full((p,), sentinel, dtype=torch.int64, device="cuda"),
    ]


def _run_pairs(c, n_psi, *, hyp_offset=0, pair_dtype=torch.float32, pair_out=None):
    p, q = c.shape[1], c.shape[2]
    if c.dtype == torch.float16:
        q //= 2
    outs = _outs(p)
    if pair_out is None:
        pair_out = torch.empty(p, q, dtype=pair_dtype, device="cuda")
    ran = fused_kernel_loader.lean_irfft_stats_pairs_transposed(
        c, n_psi, hyp_offset=hyp_offset, outs=outs, pair_out=pair_out
    )
    assert ran
    return outs, pair_out


def _tolerance(dtype: torch.dtype) -> dict[str, float]:
    return (
        {"rtol": 2e-3, "atol": 2e-2}
        if dtype == torch.float16
        else {
            "rtol": 1e-4,
            "atol": 1e-4,
        }
    )


@pytest.mark.parametrize("n_psi", [128, 256])
@pytest.mark.parametrize("n_freq", [16, 60, 64, 65, 72, 100, 129])
def test_pair_max_matches_reference_across_shapes(n_psi, n_freq):
    if not fused_kernel_loader.lean_pairs_supported(n_psi, n_freq):
        pytest.skip(f"({n_psi}, {n_freq}) is not a lean config")
    for p in (1, 8, 37):
        for q in (1, 17, 100, 130, 200):
            c = _make_spectrum(
                p, q, n_freq, seed=p * 1009 + q * 7 + n_psi + n_freq, n_psi=n_psi
            )
            s1, s2, vmax, amax, pair_ref = _reference(c.permute(1, 2, 0), n_psi)
            outs, pair_out = _run_pairs(c, n_psi)
            torch.testing.assert_close(pair_out, pair_ref, rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(outs[0], s1, rtol=1e-3, atol=1e-2)
            torch.testing.assert_close(outs[1], s2, rtol=1e-3, atol=1e-2)
            got_vmax, got_amax = fused_kernel_loader.decode_argmax_packed(outs[2])
            torch.testing.assert_close(got_vmax, vmax, rtol=1e-4, atol=1e-4)
            assert torch.equal(got_amax, amax), (p, q)
            # the per-pair maxima dominate the per-pixel maximum exactly
            assert torch.equal(pair_out.amax(dim=1), got_vmax)


@pytest.mark.parametrize("n_psi", [128, 256])
@pytest.mark.parametrize(
    "input_dtype", [torch.complex64, torch.complex32, torch.float16]
)
@pytest.mark.parametrize("pair_dtype", [torch.float32, torch.float16])
def test_input_and_output_dtypes(n_psi, input_dtype, pair_dtype):
    p, q, n_freq = 13, 70, 40
    c64 = _make_spectrum(p, q, n_freq, seed=42 + n_psi)
    if input_dtype == torch.complex64:
        c, logical = c64, c64
    elif input_dtype == torch.complex32:
        c = c64.to(torch.complex32)
        logical = c.to(torch.complex64)
    else:  # interleaved [re, im] halves: (NumFreq, P, 2Q) float16
        c = torch.view_as_real(c64).to(torch.float16).reshape(n_freq, p, 2 * q)
        logical = torch.view_as_complex(c.to(torch.float32).reshape(n_freq, p, q, 2))
    c = c.contiguous()
    _, _, vmax, amax, pair_ref = _reference(logical.permute(1, 2, 0), n_psi)
    outs, pair_out = _run_pairs(c, n_psi, pair_dtype=pair_dtype)
    assert pair_out.dtype == pair_dtype
    torch.testing.assert_close(pair_out.float(), pair_ref, **_tolerance(pair_dtype))
    got_vmax, got_amax = fused_kernel_loader.decode_argmax_packed(outs[2])
    torch.testing.assert_close(got_vmax, vmax, rtol=1e-4, atol=1e-4)
    assert torch.equal(got_amax, amax)


@pytest.mark.parametrize("n_psi", [128, 256])
def test_stats_match_dense_kernel_with_hyp_offset(n_psi):
    p, q, n_freq = 29, 150, 48
    hyp_offset = 1000
    c = _make_spectrum(p, q, n_freq, seed=7 + n_psi)
    dense = _outs(p)
    got = fused_kernel_loader.lean_irfft_stats_transposed(
        c, n_psi, decode=False, hyp_offset=hyp_offset, outs=dense
    )
    assert got is not None
    pairs, pair_out = _run_pairs(c, n_psi, hyp_offset=hyp_offset)
    # the packed best is an atomicMax (order-independent): bit-identical; the moment
    # sums are fp32 atomicAdds across hypothesis tiles, so only summation order differs
    assert torch.equal(dense[2], pairs[2])
    torch.testing.assert_close(dense[0], pairs[0], rtol=1e-6, atol=1e-3)
    torch.testing.assert_close(dense[1], pairs[1], rtol=1e-6, atol=1e-2)
    # the packed best decodes to ids offset by hyp_offset * n_psi
    _, amax = fused_kernel_loader.decode_argmax_packed(pairs[2])
    assert int(amax.min()) >= hyp_offset * n_psi
    _, _, _, amax_ref, _ = _reference(c.permute(1, 2, 0), n_psi)
    assert torch.equal(amax - hyp_offset * n_psi, amax_ref)
    # accumulation across two batches keeps the running max over both
    c2 = _make_spectrum(p, q, n_freq, seed=99 + n_psi)
    assert fused_kernel_loader.lean_irfft_stats_pairs_transposed(
        c2, n_psi, hyp_offset=hyp_offset + q, outs=pairs, pair_out=pair_out
    )
    both = torch.cat([c, c2], dim=2)  # (NumFreq, P, 2Q)
    s1, s2, vmax, _, _ = _reference(both.permute(1, 2, 0), n_psi)
    torch.testing.assert_close(pairs[0], s1, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(pairs[1], s2, rtol=1e-3, atol=1e-2)
    got_vmax, _ = fused_kernel_loader.decode_argmax_packed(pairs[2])
    torch.testing.assert_close(got_vmax, vmax, rtol=1e-4, atol=1e-4)


def test_pair_out_column_window_of_a_wider_slab():
    """A slab window ``slab[:, v0:v0+Q]`` (``stride(0) > Q``) is written in place."""
    p, q, n_freq, n_psi = 11, 33, 20, 128
    n_virtual, v0 = 90, 41
    c = _make_spectrum(p, q, n_freq, seed=3)
    _, _, _, _, pair_ref = _reference(c.permute(1, 2, 0), n_psi)
    for dtype in (torch.float32, torch.float16):
        slab = torch.full((p, n_virtual), -7.0, dtype=dtype, device="cuda")
        window = slab[:, v0 : v0 + q]
        assert window.stride(0) == n_virtual and window.stride(1) == 1
        _run_pairs(c, n_psi, pair_out=window)
        torch.testing.assert_close(window.float(), pair_ref, **_tolerance(dtype))
        untouched = torch.cat([slab[:, :v0], slab[:, v0 + q :]], dim=1)
        assert torch.all(untouched == -7.0)


def test_graph_capture_and_replay():
    p, q, n_freq, n_psi = 16, 64, 32, 256
    c_static = _make_spectrum(p, q, n_freq, seed=11)
    outs = _outs(p)
    pair_out = torch.zeros(p, q, dtype=torch.float16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):  # warm-up on a side stream, as capture requires
        _run_pairs(c_static, n_psi, pair_out=pair_out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ran = fused_kernel_loader.lean_irfft_stats_pairs_transposed(
            c_static, n_psi, hyp_offset=0, outs=outs, pair_out=pair_out
        )
    assert ran
    for seed in (12, 13):
        c_static.copy_(_make_spectrum(p, q, n_freq, seed=seed))
        outs[0].zero_()
        outs[1].zero_()
        outs[2].fill_(fused_kernel_loader.lean_sentinel_packed())
        pair_out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        s1, _, vmax, amax, pair_ref = _reference(c_static.permute(1, 2, 0), n_psi)
        torch.testing.assert_close(pair_out.float(), pair_ref, rtol=2e-3, atol=2e-2)
        torch.testing.assert_close(outs[0], s1, rtol=1e-3, atol=1e-2)
        got_vmax, got_amax = fused_kernel_loader.decode_argmax_packed(outs[2])
        torch.testing.assert_close(got_vmax, vmax, rtol=1e-4, atol=1e-4)
        assert torch.equal(got_amax, amax)


def test_binding_rejects_bad_arguments():
    module = fused_kernel_loader._try_compile_pairs()
    p, q, n_freq, n_psi = 4, 10, 16, 128
    c = _make_spectrum(p, q, n_freq, seed=1)
    outs = _outs(p)
    ok = torch.empty(p, q, dtype=torch.float32, device="cuda")

    def call(**kw):
        args = {"c": c, "num_psi": n_psi, "hyp_offset": 0, "outs": outs, "pair_out": ok}
        args.update(kw)
        module.lean_irfft_stats_pairs_transposed(*args.values())

    call()  # the baseline is accepted
    with pytest.raises(RuntimeError):
        call(pair_out=torch.empty(p, q + 1, dtype=torch.float32, device="cuda"))
    with pytest.raises(RuntimeError):
        call(pair_out=torch.empty(p, q, dtype=torch.int32, device="cuda"))
    with pytest.raises(RuntimeError):  # unit stride must be along Q
        call(pair_out=torch.empty(q, p, dtype=torch.float32, device="cuda").T)
    with pytest.raises(RuntimeError):
        call(pair_out=torch.empty(p, q, dtype=torch.float32))
    with pytest.raises(RuntimeError):
        call(outs=outs[:2])
    with pytest.raises(RuntimeError):
        call(c=c.permute(0, 2, 1))
    with pytest.raises(RuntimeError):
        call(num_psi=64)
    # the Python wrapper turns an unsupported config into a fallback, not a raise
    assert not fused_kernel_loader.lean_irfft_stats_pairs_transposed(
        c, 64, hyp_offset=0, outs=outs, pair_out=ok
    )
    assert not fused_kernel_loader.lean_pairs_supported(64, 16)
    assert fused_kernel_loader.lean_pairs_supported(128, 16)
