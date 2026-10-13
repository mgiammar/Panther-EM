"""Feature layouts, contraction engines and the feature store (features.py)."""

from __future__ import annotations

import pytest
import torch
from conftest import read_engine_block, read_group, reference_spectrum

from panther_em.inference.search.features import (
    ComplexGemmEngine,
    FeatureLayout,
    FeatureStore,
    Fp16GemmEngine,
    ReadyBlock,
    Spectrum,
    make_engine,
    rectangles_disjoint,
)
from panther_em.inference.search.types import ImageSet, PixelSelection, SearchGeometry


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("rects", "expected_runs"),
    [
        ([(0, 0, 8, 4)], [(0, 8, 4)]),
        ([(0, 0, 8, 4), (8, 0, 4, 4)], [(0, 8, 4), (8, 12, 4)]),
        ([(0, 0, 4, 2), (0, 2, 4, 2), (4, 0, 2, 4)], [(0, 4, 4), (4, 6, 4)]),
        # k in [0, 2): rect 1 (m 0..5); [2, 5): rect 0 (m 1..4); [5, 6): rect 2 (m 0..1)
        ([(2, 1, 3, 3), (0, 0, 2, 5), (5, 0, 1, 1)], [(0, 2, 5), (2, 5, 3), (5, 6, 1)]),
    ],
)
def test_layout_k_runs(rects, expected_runs):
    layout = FeatureLayout.create(rects)
    assert layout.num_features == sum(r[2] * r[3] for r in rects)
    assert layout.k_stop == max(r[0] + r[2] for r in rects)
    assert [(r.k_start, r.k_stop, r.num_m) for r in layout.k_runs] == expected_runs
    for run in layout.k_runs:
        ms = [(p.m_start, p.m_stop) for p in run.parts]
        assert len(set(ms)) == len(ms)  # disjoint m ranges within a run
        for part in run.parts:
            k0, m0, ke, me = rects[part.rect]
            assert k0 <= run.k_start and run.k_stop <= k0 + ke
            assert (part.m_start, part.m_stop) == (m0, m0 + me)
    assert len(layout.covered_k()) == layout.k_stop


def test_layout_validation_and_slots():
    with pytest.raises(ValueError):
        FeatureLayout.create([(0, 0, 4, 4), (2, 2, 4, 4)])
    with pytest.raises(ValueError):
        FeatureLayout.create([])
    assert not rectangles_disjoint([(0, 0, 4, 4), (3, 3, 1, 1)])
    assert rectangles_disjoint([(0, 0, 4, 4), (4, 0, 1, 1), (0, 4, 4, 1)])
    layout = FeatureLayout.create([(0, 0, 2, 3), (2, 0, 1, 3)])
    assert layout.num_features == 9
    assert hash(layout) == hash(FeatureLayout.create([(0, 0, 2, 3), (2, 0, 1, 3)]))
    gap = FeatureLayout.create([(0, 0, 2, 3), (4, 0, 1, 3)])
    assert gap.covered_k() == (True, True, False, False, True)



# --------------------------------------------------------------------------- #
# contraction
# --------------------------------------------------------------------------- #


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
    layout = FeatureLayout.create(rects)
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



# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #


CUDA = torch.cuda.is_available()


def _store(n_px=10, block_pixels=4, n_defocus=2, device="cpu", mask_every=None):
    geo = SearchGeometry.from_images(
        ImageSet.from_stack(torch.zeros(1, 21, 21)), (20, 20)
    )
    assert geo.n_pixels == 4  # 2x2 grid -> make a bigger one below
    geo = SearchGeometry.from_images(
        ImageSet.from_stack(torch.zeros(1, 24, 24)), (20, 20)
    )
    assert geo.n_pixels == 25
    index = torch.arange(geo.n_pixels)
    if mask_every:
        index = index[::mask_every]
    index = index[:n_px]
    pixels = PixelSelection(index, geo)
    column_index = index.flip(0)  # arbitrary permutation: compact order != sorted order
    return FeatureStore(
        geo,
        pixels,
        column_index,
        block_pixels=block_pixels,
        n_defocus=n_defocus,
        device=device,
    )


def _random_group_data(group, n_cols, seed=0):
    gen = torch.Generator().manual_seed(seed)
    re = torch.randn(group.n_defocus, group.num_cells, n_cols, generator=gen)
    im = torch.randn(group.n_defocus, group.num_cells, n_cols, generator=gen)
    return torch.complex(re, im)


def test_add_group_blocks_and_tail():
    store = _store(n_px=10, block_pixels=4)
    assert store.n_blocks == 3 and store.block_range(2) == (8, 10)
    g = store.add_group((1, 0, 3, 2), "fp32")
    assert len(g.blocks) == 3 and g.blocks[0].shape == (2, 3, 4, 2)
    assert g.blocks[0].dtype == torch.complex64 and g.id == 0
    with pytest.raises(ValueError):
        store.add_group((1, 0, 3, 2), "fp32")
    assert store.groups_covering((1, 0, 3, 2), "fp32", (0,)) == [(g, 0)]
    assert store.missing_planes((0, 0, 1, 1), "fp32", (0,)) == (0,)


def test_writer_splits_across_blocks_and_requires_completeness():
    store = _store(n_px=10, block_pixels=4)
    g = store.add_group((0, 0, 2, 3), "fp32")
    data = _random_group_data(g, 10)
    w = store.writer(g)
    with pytest.raises(ValueError):
        store.blocks([g])  # incomplete
    # write in awkward pieces: columns [2, 9) straddle blocks 0/1/2; cells in two chunks
    w.write(slice(0, 4), (2, 9), data[:, 0:4, 2:9])
    w.write(slice(4, 6), (2, 9), data[:, 4:6, 2:9])
    with pytest.raises(ValueError):
        w.close()
    w.write(slice(0, 6), (0, 2), data[:, :, 0:2])
    w.write(slice(0, 6), (9, 10), data[0, :, 9:10], plane=0)
    w.write(slice(0, 6), (9, 10), data[1, :, 9:10], plane=1)
    w.close()
    assert g.complete
    for d in range(2):
        got = read_group(store, g, d)  # (P, num_k, num_m)
        expected = data[d].T.reshape(10, 2, 3)
        torch.testing.assert_close(got, expected)
    with pytest.raises(ValueError):
        w.write(slice(0, 6), (0, 2), data[:, :, 0:3])


def test_direct_block_source_yields_views_in_order():
    store = _store(n_px=10, block_pixels=4)
    g = store.add_group((0, 0, 2, 3), "fp32")
    data = _random_group_data(g, 10)
    w = store.writer(g)
    w.write(slice(0, 6), (0, 10), data)
    w.close()
    blocks = list(store.blocks([g]))
    assert [b.index for b in blocks] == [0, 1, 2]
    assert [b.n_valid for b in blocks] == [4, 4, 2]
    assert all(isinstance(b, ReadyBlock) and b.ready is None for b in blocks)
    assert blocks[1].operands[g.id] is g.blocks[1]
    blocks[0].release(None)  # no-op for device residency


def test_subset_compacts_columns_block_wise():
    store = _store(n_px=10, block_pixels=4)
    g1 = store.add_group((0, 0, 2, 3), "fp32")
    g2 = store.add_group((2, 1, 1, 2), "fp32")
    d1, d2 = _random_group_data(g1, 10, seed=1), _random_group_data(g2, 10, seed=2)
    for g, d in ((g1, d1), (g2, d2)):
        w = store.writer(g)
        w.write(slice(0, g.num_cells), (0, 10), d)
        w.close()
    keep = torch.tensor([1, 0, 0, 1, 1, 0, 1, 0, 0, 1], dtype=torch.bool)
    sub = store.subset(keep, block_pixels=3)
    assert sub.n_pixels == 5 and sub.n_blocks == 2 and sub.block_pixels == 3
    assert torch.equal(sub.column_index, store.column_index[keep])
    assert torch.equal(sub.pixels.index, torch.sort(sub.column_index).values)
    assert len(sub.groups) == 2 and all(g.complete for g in sub.groups)
    for g_old, g_new in ((g1, sub.groups[0]), (g2, sub.groups[1])):
        assert g_new.rect == g_old.rect
        for plane in range(2):
            torch.testing.assert_close(
                read_group(sub, g_new, plane), read_group(store, g_old, plane)[keep]
            )
    with pytest.raises(ValueError):
        store.subset(keep[:3])


@pytest.mark.skipif(not CUDA, reason="fp16 blocks need CUDA")
def test_fp16_group_round_trip_cuda():
    store = _store(n_px=10, block_pixels=4, device="cuda")
    g = store.add_group((0, 0, 2, 3), "fp16")
    assert g.blocks[0].dtype == torch.float16 and g.blocks[0].shape == (2, 2, 4, 6)
    data = _random_group_data(g, 10)
    w = store.writer(g)
    w.write(slice(0, 6), (0, 10), data)
    w.close()
    torch.testing.assert_close(
        read_group(store, g, 1).cpu(), data[1].T.reshape(10, 2, 3), rtol=2e-3, atol=2e-3
    )
    assert store.nbytes() == 3 * 2 * 2 * 4 * 6 * 2


def test_store_validation():
    geo = SearchGeometry.from_images(
        ImageSet.from_stack(torch.zeros(1, 24, 24)), (20, 20)
    )
    pixels = PixelSelection.all(geo)
    with pytest.raises(ValueError):
        FeatureStore(
            geo, pixels, torch.arange(3), block_pixels=4, n_defocus=1, device="cpu"
        )
    with pytest.raises(ValueError, match="residency"):
        FeatureStore(
            geo,
            pixels,
            pixels.index,
            block_pixels=4,
            n_defocus=1,
            device="cpu",
            residency="transient",
        )


def test_plane_restricted_groups_cover_and_reject_overlap():
    store = _store(n_px=10, block_pixels=4, n_defocus=3)
    rect = (0, 0, 2, 3)
    g_a = store.add_group(rect, "fp32", planes=(2,))
    assert g_a.planes == (2,) and g_a.blocks[0].shape == (1, 2, 4, 3)
    assert store.missing_planes(rect, "fp32", (0, 1, 2)) == (0, 1)
    with pytest.raises(ValueError, match="already holds planes"):
        store.add_group(rect, "fp32", planes=(1, 2))
    with pytest.raises(ValueError):
        store.add_group(rect, "fp32", planes=(3,))
    g_b = store.add_group(rect, "fp32", planes=(0, 1))
    assert store.missing_planes(rect, "fp32", (0, 1, 2)) == ()
    covering = store.groups_covering(rect, "fp32", (0, 2, 1))
    assert covering == [(g_b, 0), (g_a, 0), (g_b, 1)]
    with pytest.raises(KeyError):
        store.groups_covering(rect, "fp16", (0,))
    assert g_a.local_plane(2) == 0 and g_b.local_plane(1) == 1 and not g_a.holds(0)
    with pytest.raises(ValueError):
        g_a.local_plane(0)

    # writes address planes globally and completeness is tracked per plane
    data = _random_group_data(g_b, 10)  # (2, cells, 10) in group plane order
    w = store.writer(g_b)
    w.write(slice(0, 6), (0, 10), data[1], plane=1)
    with pytest.raises(ValueError, match="incomplete"):
        w.close()
    w.write(slice(0, 6), (0, 10), data[0], plane=0)
    w.close()
    torch.testing.assert_close(read_group(store, g_b, 1), data[1].T.reshape(10, 2, 3))
    with pytest.raises(ValueError):
        store.writer(g_a).write(
            slice(0, 6), (0, 10), data[0], plane=0
        )  # g_a lacks plane 0

    # compaction preserves plane sets and ids
    wa = store.writer(g_a)
    wa.write(slice(0, 6), (0, 10), data[1], plane=2)
    wa.close()
    keep = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=torch.bool)
    sub = store.subset(keep, block_pixels=3)
    assert [(g.id, g.planes) for g in sub.groups] == [(0, (2,)), (1, (0, 1))]
