"""Contraction engines vs the dense einsum reference."""

from __future__ import annotations

import pytest
import torch
from conftest import read_engine_block, reference_spectrum

from panther_em.inference.search.contraction import (
    ComplexGemmEngine,
    Fp16GemmEngine,
    Spectrum,
    make_engine,
)
from panther_em.inference.search.tiling import FeatureLayout

CUDA = torch.cuda.is_available()


def _random_features(n_d, n_p, rects, seed=0):
    """Per-rectangle complex features (D, P, num_k, num_m) and the flat (P, r) view."""
    gen = torch.Generator().manual_seed(seed)
    per_rect = []
    for _, _, ke, me in rects:
        re = torch.randn(n_d, n_p, ke, me, generator=gen)
        im = torch.randn(n_d, n_p, ke, me, generator=gen)
        per_rect.append(torch.complex(re, im))
    return per_rect


def _random_weights(n_hyp, rects, seed=1):
    gen = torch.Generator().manual_seed(seed)
    out = []
    for _, _, ke, me in rects:
        re = torch.randn(n_hyp, ke, me, generator=gen)
        im = torch.randn(n_hyp, ke, me, generator=gen)
        out.append(torch.complex(re, im))
    return out


def _dense(rects, per_rect, w_per_rect, d):
    """Dense (P, N, k_stop) reference over all rectangles, zero where uncovered."""
    k_stop = max(k0 + ke for k0, _, ke, _ in rects)
    n_p = per_rect[0].shape[1]
    n_hyp = w_per_rect[0].shape[0]
    ref = torch.zeros(n_p, n_hyp, k_stop, dtype=torch.complex64)
    for (k0, _, ke, _), y, w in zip(rects, per_rect, w_per_rect, strict=True):
        ref[:, :, k0 : k0 + ke] += reference_spectrum(y[d], w)
    return ref


def _run_engine(
    engine, rects, per_rect, w_per_rect, n_d, n_p, n_hyp, device, p_batch=None
):
    layout = FeatureLayout(rects)
    blocks = []
    for i, (_, _, ke, me) in enumerate(rects):
        blk = torch.zeros(
            engine.block_shape(n_d, ke, me, n_p),
            dtype=engine.block_dtype,
            device=device,
        )
        data = (
            per_rect[i].reshape(n_d, n_p, ke * me).permute(0, 2, 1).to(device)
        )  # (D, cells, cols)
        # write in awkward cell chunks and column halves to exercise the splitting
        half = n_p // 2
        chunk = max(1, (ke * me) // 3)
        for c0 in range(0, ke * me, chunk):
            c1 = min(c0 + chunk, ke * me)
            engine.write_block(
                blk, slice(c0, c1), slice(0, half), data[:, c0:c1, :half]
            )
            engine.write_block(
                blk, slice(c0, c1), slice(half, n_p), data[:, c0:c1, half:]
            )
        blocks.append(blk)
    prepared = [engine.prepare_weights(w.to(device)) for w in w_per_rect]

    p0, p1 = (0, n_p) if p_batch is None else p_batch
    outs = []
    for d in range(n_d):
        spec = Spectrum.empty(
            layout.k_stop, p1 - p0, n_hyp, engine.spectrum_dtype, device
        )
        spec.zero_uncovered(layout.covered_k())
        for run in layout.k_runs:
            a_parts, b_parts = [], []
            for part in run.parts:
                k0 = rects[part.rect][0]
                a_parts.append(
                    engine.operand(
                        blocks[part.rect], d, run.k_start - k0, run.k_stop - k0, p0, p1
                    )
                )
                b_parts.append(prepared[part.rect][run.k_start - k0 : run.k_stop - k0])
            engine.contract(a_parts, b_parts, run, spec)
        outs.append(spec)
    return blocks, outs


@pytest.mark.parametrize(
    "rects",
    [
        [(0, 0, 5, 3)],
        [(0, 0, 4, 2), (0, 2, 4, 3), (4, 0, 2, 5)],
        [(1, 0, 3, 2), (5, 1, 2, 2)],  # gaps at k = 0 and k = 4
    ],
)
def test_complex_engine_matches_reference_cpu(rects):
    n_d, n_p, n_hyp = 2, 7, 5
    per_rect = _random_features(n_d, n_p, rects)
    w_per_rect = _random_weights(n_hyp, rects)
    engine = ComplexGemmEngine("fp32")
    blocks, outs = _run_engine(
        engine, rects, per_rect, w_per_rect, n_d, n_p, n_hyp, "cpu"
    )
    for d in range(n_d):
        ref = _dense(rects, per_rect, w_per_rect, d)
        torch.testing.assert_close(outs[d].pnf(), ref, rtol=1e-5, atol=1e-5)
        # read_block round-trips what write_block stored
        for i in range(len(rects)):
            torch.testing.assert_close(
                read_engine_block(engine, blocks[i], d), per_rect[i][d]
            )


def test_complex_engine_pixel_sub_batch_is_a_view():
    rects = [(0, 0, 3, 4)]
    n_d, n_p, n_hyp = 1, 10, 3
    per_rect = _random_features(n_d, n_p, rects)
    w_per_rect = _random_weights(n_hyp, rects)
    engine = ComplexGemmEngine()
    _, outs = _run_engine(
        engine, rects, per_rect, w_per_rect, n_d, n_p, n_hyp, "cpu", p_batch=(4, 9)
    )
    ref = _dense(rects, per_rect, w_per_rect, 0)[4:9]
    torch.testing.assert_close(outs[0].pnf(), ref, rtol=1e-5, atol=1e-5)


def test_make_engine_and_spectrum_validation():
    assert isinstance(make_engine("fp16"), Fp16GemmEngine)
    assert make_engine("tf32").precision.value == "tf32"
    with pytest.raises(ValueError):
        ComplexGemmEngine("fp16")
    with pytest.raises(ValueError):
        Spectrum(torch.zeros(3, 4, 5, dtype=torch.complex64).permute(1, 2, 0))


@pytest.mark.skipif(not CUDA, reason="fp16 engine needs CUDA")
@pytest.mark.filterwarnings("ignore:ComplexHalf support is experimental")
@pytest.mark.parametrize(
    "rects",
    [
        [(0, 0, 5, 3)],
        [(0, 0, 4, 2), (0, 2, 4, 3), (4, 0, 2, 5)],
        [(1, 0, 3, 2), (5, 1, 2, 2)],
    ],
)
def test_fp16_engine_matches_reference_cuda(rects):
    device = torch.device("cuda")
    n_d, n_p, n_hyp = 2, 8, 6
    per_rect = _random_features(n_d, n_p, rects)
    w_per_rect = _random_weights(n_hyp, rects)
    engine = Fp16GemmEngine()
    blocks, outs = _run_engine(
        engine, rects, per_rect, w_per_rect, n_d, n_p, n_hyp, device
    )
    for d in range(n_d):
        ref = _dense(rects, per_rect, w_per_rect, d)
        got = outs[d].pnf().to(torch.complex64).cpu()
        assert outs[d].dtype == torch.complex32
        # fp16 inputs + fp16 accumulate: a few 1e-3 absolute on O(sqrt(r)) values
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=5e-2)
        for i in range(len(rects)):
            torch.testing.assert_close(
                read_engine_block(engine, blocks[i], d).cpu(),
                per_rect[i][d],
                rtol=2e-3,
                atol=2e-3,
            )
