"""StageExecutor end to end vs a dense reference (CPU torch path; CUDA fused/graph)."""

from __future__ import annotations

import pytest
import torch

from panther_em.inference.search.basis import WeightSource
from panther_em.inference.search.context import (
    DeviceCaps,
    EagerHypLoop,
    GraphedHypLoop,
    Planner,
    SearchContext,
)
from panther_em.inference.search.executor import (
    StageExecutor,
    plan_memory,
    plan_stage,
)
from panther_em.inference.search.hypotheses import (
    HypothesisPlan,
    HypothesisSpace,
    HypothesisSubset,
)
from panther_em.inference.search.schedule import StageSpec
from panther_em.inference.search.store import FeatureStore
from panther_em.inference.search.types import (
    BatchShape,
    ImageSet,
    PixelSelection,
    SearchGeometry,
)

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.filterwarnings("ignore:ComplexHalf support is experimental")


def _geometry(n_images=1):
    images = ImageSet.from_stack(torch.zeros(n_images, 24, 24))
    return SearchGeometry.from_images(images, (20, 20))  # 5 x 5 grid per image


def _fill_store(result, spec, n_defocus, block_pixels, device, n_images=1, seed=0):
    """A device store with random features for every rectangle of ``spec``."""
    geo = _geometry(n_images)
    pixels = PixelSelection.all(geo)
    store = FeatureStore(
        geo,
        pixels,
        pixels.index,
        block_pixels=block_pixels,
        n_defocus=n_defocus,
        device=device,
    )
    gen = torch.Generator().manual_seed(seed)
    features = {}
    for rect in spec.rectangles:
        group = store.add_group(rect, spec.precision)
        n_cells = rect[2] * rect[3]
        re = torch.randn(n_defocus, n_cells, store.n_pixels, generator=gen)
        im = torch.randn(n_defocus, n_cells, store.n_pixels, generator=gen)
        data = torch.complex(re, im)
        w = store.writer(group)
        w.write(slice(0, n_cells), (0, store.n_pixels), data)
        w.close()
        features[rect] = data  # (D, cells, P)
    return store, features


def _reference_corr(result, spec, features, hyp_plan, n_psi):
    """Dense correlogram ``(P, D_sel, N_sel, psi)`` from the store's features."""
    src = WeightSource(result, "cpu")
    rows = hyp_plan.template_rows()
    planes = hyp_plan.defocus_planes()
    k_stop = spec.k_stop
    P = next(iter(features.values())).shape[-1]
    n_sel = int(rows.numel())
    C = torch.zeros(P, len(planes), n_sel, k_stop, dtype=torch.complex64)
    for rect, data in features.items():
        k0, _m0, ke, me = rect
        w = src.block(rect, rows)  # (N_sel, ke, me)
        y = data.permute(0, 2, 1).reshape(
            len(planes) if data.shape[0] == len(planes) else data.shape[0], P, ke, me
        )
        y = y[list(planes)]  # (D_sel, P, ke, me)
        C[:, :, :, k0 : k0 + ke] += torch.einsum("dpkm,nkm->pdnk", y, w.conj())
    return torch.fft.irfft(C, n=n_psi, dim=-1, norm="forward")  # (P, D, N, psi)


def _reference_pair_max(result, spec, features, hyp_plan, n_psi):
    """``(P, n_virtual)`` max over psi; columns are virtual ids ``d*N_sel + j``."""
    corr = _reference_corr(result, spec, features, hyp_plan, n_psi)
    return corr.amax(dim=-1).reshape(corr.shape[0], -1)


def _reference(result, spec, features, hyp_plan, n_psi):
    """Dense per-pixel statistics over (defocus, template, psi)."""
    corr = _reference_corr(result, spec, features, hyp_plan, n_psi)
    P = corr.shape[0]
    flat = corr.reshape(P, -1)
    mip, amax = flat.max(dim=1)
    mean = flat.mean(dim=1)
    var = (flat**2).mean(dim=1) - mean**2
    psi_raw = amax % n_psi
    best_virtual = amax // n_psi  # d_local * n_sel + j
    best_psi = (n_psi - psi_raw) % n_psi
    unique = (flat == mip[:, None]).sum(dim=1) == 1
    return mip, mean, var, best_virtual, best_psi, unique


def _run_stage(
    ctx,
    result,
    spec,
    store,
    space,
    *,
    subset=None,
    batch=None,
    use_cuda_graph=False,
    use_fused_kernel=True,
    pair_dtype=None,
    collector=None,
):
    memory, backend, engine, hyp_plan = plan_memory(
        ctx,
        spec,
        n_pixels=store.n_pixels,
        space=space,
        subset=subset,
        batch=batch,
        block_pixels=store.block_pixels,
        use_cuda_graph=use_cuda_graph,
        use_fused_kernel=use_fused_kernel,
        store=store,
        pair_dtype=pair_dtype,
    )
    plan = plan_stage(ctx, 0, spec, store, memory, backend, engine, hyp_plan)
    maps = StageExecutor(ctx).run(store, plan, collector=collector)
    return plan, maps


class _SlabRecorder:
    """A ``BlockPairSink`` that stitches the block slabs back into one matrix."""

    def __init__(self, n_pixels, n_virtual, dtype, device):
        self.pair_max = torch.full(
            (n_pixels, n_virtual), float("nan"), dtype=dtype, device=device
        )
        self.mean = torch.full((n_pixels,), float("nan"), device=device)
        self.variance = torch.full((n_pixels,), float("nan"), device=device)
        self.blocks = []

    def consume_block(self, pair_max, *, col_start, n_valid, mean, variance):
        assert pair_max.shape == (n_valid, self.pair_max.shape[1])
        assert mean.shape == variance.shape == (n_valid,)
        self.pair_max[col_start : col_start + n_valid] = pair_max
        self.mean[col_start : col_start + n_valid] = mean
        self.variance[col_start : col_start + n_valid] = variance
        self.blocks.append((col_start, n_valid))


def _assert_matches(maps, ref, hyp_plan, *, rtol=1e-4, atol=1e-4):
    mip, mean, var, best_virtual, best_psi, unique = ref
    torch.testing.assert_close(maps.mip.cpu(), mip, rtol=rtol, atol=atol)
    torch.testing.assert_close(maps.mean.cpu(), mean, rtol=rtol, atol=atol)
    torch.testing.assert_close(maps.variance.cpu(), var, rtol=rtol, atol=atol)
    assert torch.equal(maps.best_virtual.cpu()[unique], best_virtual[unique])
    assert torch.equal(maps.best_psi.cpu()[unique], best_psi[unique])
    assert maps.n_evaluated == hyp_plan.n_evaluated


@pytest.mark.parametrize("block_pixels", [24, 8, 4])
def test_cpu_eager_matches_reference_any_block_size(synthetic_result, block_pixels):
    result = synthetic_result()
    space = HypothesisSpace.from_result(result)
    spec = StageSpec([(0, 0, 4, 3), (0, 3, 2, 2)], precision="fp32", n_psi=16)
    store, features = _fill_store(result, spec, 1, block_pixels, "cpu")
    ctx = SearchContext("cpu", WeightSource(result, "cpu"))
    plan, maps = _run_stage(ctx, result, spec, store, space, batch=BatchShape(4, 4))
    assert plan.memory.block_pixels == block_pixels and plan.memory.loop == "eager"
    assert plan.stats_backend.name == "torch"
    _assert_matches(
        maps, _reference(result, spec, features, plan.hyp_plan, 16), plan.hyp_plan
    )


def test_defocus_axis_reduces_over_planes(synthetic_result):
    result = synthetic_result()
    space = HypothesisSpace.from_result(result, n_defocus=3)
    spec = StageSpec([(0, 0, 4, 5)], n_psi=16)
    store, features = _fill_store(result, spec, 3, 8, "cpu")
    ctx = SearchContext("cpu", WeightSource(result, "cpu"))
    plan, maps = _run_stage(ctx, result, spec, store, space, batch=BatchShape(4, 6))
    ref = _reference(result, spec, features, plan.hyp_plan, 16)
    _assert_matches(maps, ref, plan.hyp_plan)
    # the best virtual id decodes to a (defocus, filter, orientation) triple
    decoded = space.decode(plan.hyp_plan.decode(maps.best_virtual))
    assert decoded.defocus.max() <= 2 and decoded.orientation.max() <= 5
    # equals the plane-wise maximum of three single-plane searches
    per_plane = []
    for d in range(3):
        single_space = HypothesisSpace.from_result(result)
        single_store, _ = _fill_store(result, spec, 1, 8, "cpu")
        single_store.groups[0].blocks = [
            b[d : d + 1].clone() for b in store.groups[0].blocks
        ]
        _, m = _run_stage(
            ctx, result, spec, single_store, single_space, batch=BatchShape(4, 6)
        )
        per_plane.append(m.mip)
    torch.testing.assert_close(
        maps.mip, torch.stack(per_plane).max(dim=0).values, rtol=1e-5, atol=1e-5
    )


def test_strided_subset_matches_restricted_reference_and_decodes_globally(
    synthetic_result,
):
    result = synthetic_result(n_filters=2, n_orientations=5)
    space = HypothesisSpace.from_result(result, n_defocus=2)
    spec = StageSpec([(0, 0, 3, 4)], n_psi=16)
    store, features = _fill_store(result, spec, 2, 25, "cpu")
    ctx = SearchContext("cpu", WeightSource(result, "cpu"))
    subset = HypothesisSubset(
        template_ids=torch.arange(1, 10, 3), defocus_ids=torch.tensor([1])
    )
    plan, maps = _run_stage(
        ctx, result, spec, store, space, subset=subset, batch=BatchShape(5, 2)
    )
    assert plan.hyp_plan.n_virtual == 3 and not plan.hyp_plan.is_contiguous
    ref = _reference(result, spec, features, plan.hyp_plan, 16)
    _assert_matches(maps, ref, plan.hyp_plan)
    global_ids = plan.hyp_plan.decode(maps.best_virtual)
    decoded = space.decode(global_ids)
    assert torch.all(decoded.defocus == 1)
    assert set(global_ids.tolist()) <= {1 * 10 + t for t in (1, 4, 7)}


@pytest.mark.parametrize("block_pixels", [24, 8, 4])
def test_cpu_pair_slab_matches_reference_max_over_psi(synthetic_result, block_pixels):
    result = synthetic_result(n_orientations=6)
    space = HypothesisSpace.from_result(result, n_defocus=2)
    spec = StageSpec([(0, 0, 4, 3), (0, 3, 2, 2)], precision="fp32", n_psi=16)
    store, features = _fill_store(result, spec, 2, block_pixels, "cpu")
    ctx = SearchContext("cpu", WeightSource(result, "cpu"))
    n_virtual = 2 * result.num_fourier_filters * 6
    rec = _SlabRecorder(store.n_pixels, n_virtual, torch.float32, "cpu")
    plan, maps = _run_stage(
        ctx,
        result,
        spec,
        store,
        space,
        batch=BatchShape(4, 4),
        pair_dtype=torch.float32,
        collector=rec,
    )
    assert plan.memory.pair_dtype is torch.float32
    assert plan.spectrum.pair_dtype is torch.float32
    assert plan.hyp_plan.n_virtual == n_virtual
    assert plan.memory.pair_slab_bytes(n_virtual) == block_pixels * n_virtual * 4
    assert [n for _, n in rec.blocks] == [
        min(block_pixels, store.n_pixels - s) for s, _ in rec.blocks
    ]
    assert sum(n for _, n in rec.blocks) == store.n_pixels
    ref = _reference_pair_max(result, spec, features, plan.hyp_plan, 16)
    torch.testing.assert_close(rec.pair_max, ref, rtol=1e-4, atol=1e-4)
    # the slab's row maximum is the per-pixel mip; moments are the finalized maps
    torch.testing.assert_close(rec.pair_max.amax(dim=1), maps.mip)
    assert torch.equal(rec.mean, maps.mean) and torch.equal(rec.variance, maps.variance)
    _assert_matches(
        maps, _reference(result, spec, features, plan.hyp_plan, 16), plan.hyp_plan
    )
    # the loop cache keys on the pair dtype: the dense plan is a different loop
    plan_dense, _ = _run_stage(ctx, result, spec, store, space, batch=BatchShape(4, 4))
    assert (
        plan_dense.signature != plan.signature and plan_dense.memory.pair_dtype is None
    )
    with pytest.raises(ValueError, match="pair_dtype"):
        _run_stage(
            ctx, result, spec, store, space, batch=BatchShape(4, 4), collector=rec
        )


def test_cpu_pair_slab_strided_subset_and_fp16(synthetic_result):
    result = synthetic_result(n_filters=2, n_orientations=5)
    space = HypothesisSpace.from_result(result, n_defocus=2)
    spec = StageSpec([(0, 0, 3, 4)], n_psi=16)
    store, features = _fill_store(result, spec, 2, 25, "cpu")
    ctx = SearchContext("cpu", WeightSource(result, "cpu"))
    subset = HypothesisSubset(
        template_ids=torch.arange(1, 10, 3), defocus_ids=torch.tensor([1])
    )
    rec = _SlabRecorder(store.n_pixels, 3, torch.float16, "cpu")
    plan, _maps = _run_stage(
        ctx,
        result,
        spec,
        store,
        space,
        subset=subset,
        batch=BatchShape(5, 2),
        pair_dtype=torch.float16,
        collector=rec,
    )
    assert plan.hyp_plan.n_virtual == 3
    ref = _reference_pair_max(result, spec, features, plan.hyp_plan, 16)
    torch.testing.assert_close(rec.pair_max.float(), ref, rtol=2e-3, atol=2e-2)
    # virtual columns decode to the restricted global ids, plane 1 only
    global_ids = plan.hyp_plan.decode(torch.arange(3))
    assert global_ids.tolist() == [1 * 10 + t for t in (1, 4, 7)]
    assert torch.all(space.decode(global_ids).defocus == 1)


def test_planner_defaults_and_overrides():
    caps = DeviceCaps.probe("cpu")
    planner = Planner(l2_target=64 * 1024**2, block_bytes_max=1024)
    from panther_em.inference.search.statistics import TorchStatsBackend

    plan = planner.plan(
        caps,
        n_pixels=1000,
        n_freq=64,
        n_psi=256,
        n_virtual=5000,
        n_defocus=1,
        feature_bytes_per_pixel=16,
        spectrum_dtype=torch.complex64,
        precision="fp32",
        stats_backend=TorchStatsBackend(),
    )
    assert plan.hyp_batch % 64 == 0 and plan.block_pixels % plan.gemm_pixels == 0
    assert plan.loop == "eager" and plan.n_sub >= 1
    explicit = planner.plan(
        caps,
        n_pixels=10,
        n_freq=4,
        n_psi=16,
        n_virtual=6,
        n_defocus=1,
        feature_bytes_per_pixel=16,
        spectrum_dtype=torch.complex64,
        precision="fp32",
        stats_backend=TorchStatsBackend(),
        batch=BatchShape(4, 3),
        block_pixels=9,
    )
    assert (explicit.gemm_pixels, explicit.hyp_batch, explicit.block_pixels) == (
        4,
        3,
        8,
    )
    assert (
        Planner.hyp_batch_multiple(256) == 64 and Planner.hyp_batch_multiple(128) == 128
    )


def test_planner_pair_slab_joins_block_budget():
    from panther_em.inference.search.statistics import TorchStatsBackend

    caps = DeviceCaps.probe("cpu")
    planner = Planner(l2_target=64 * 1024**2, block_bytes_max=64 << 20)
    common = {
        "n_pixels": 100_000,
        "n_freq": 64,
        "n_psi": 256,
        "n_virtual": 4096,
        "n_defocus": 1,
        "feature_bytes_per_pixel": 16,
        "spectrum_dtype": torch.complex64,
        "precision": "fp32",
        "stats_backend": TorchStatsBackend(),
        "batch": BatchShape(64, 256),
    }
    dense = planner.plan(caps, **common)
    half = planner.plan(caps, **common, pair_dtype=torch.float16)
    full = planner.plan(caps, **common, pair_dtype=torch.float32)
    assert dense.pair_dtype is None and dense.pair_slab_bytes(4096) == 0
    assert half.pair_dtype is torch.float16 and full.pair_dtype is torch.float32
    # per pixel: 16 B dense, 16 + 8192 B fp16, 16 + 16384 B fp32 -> blocks shrink
    assert dense.block_pixels > half.block_pixels > full.block_pixels >= 64
    assert half.block_pixels % 64 == 0 and full.block_pixels % 64 == 0
    assert half.pair_slab_bytes(4096) == half.block_pixels * 4096 * 2
    # one GEMM batch of slab must fit in a quarter of free memory
    tiny = DeviceCaps(
        torch.device("cpu"), False, False, False, None, free_bytes=1 << 20
    )
    with pytest.raises(ValueError, match="pair slab"):
        planner.plan(tiny, **common, pair_dtype=torch.float32)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_cuda_fused_eager_and_graph_match_reference(synthetic_result, precision):
    device = torch.device("cuda")
    result = synthetic_result(n_orientations=8)
    space = HypothesisSpace.from_result(result, n_defocus=2)
    spec = StageSpec([(0, 0, 4, 3), (0, 3, 2, 2)], precision=precision, n_psi=128)
    store, features = _fill_store(result, spec, 2, 8, device)
    ctx = SearchContext(device, WeightSource(result, device))
    tol = (
        {"rtol": 1e-4, "atol": 1e-4}
        if precision == "fp32"
        else {"rtol": 2e-2, "atol": 5e-2}
    )

    plan_e, eager = _run_stage(
        ctx, result, spec, store, space, batch=BatchShape(4, 4), use_cuda_graph=False
    )
    assert plan_e.stats_backend.name == "fused" and plan_e.memory.loop == "eager"
    assert isinstance(ctx.loops[plan_e.signature], EagerHypLoop)
    ref = _reference(result, spec, features, plan_e.hyp_plan, 128)
    _assert_matches(eager, ref, plan_e.hyp_plan, **tol)

    plan_g, graphed = _run_stage(
        ctx, result, spec, store, space, batch=BatchShape(4, 4), use_cuda_graph=True
    )
    assert plan_g.memory.loop == "graph"
    assert isinstance(ctx.loops[plan_g.signature], GraphedHypLoop)
    torch.testing.assert_close(graphed.mip, eager.mip, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(graphed.mean, eager.mean, rtol=1e-5, atol=1e-5)
    assert torch.equal(graphed.best_virtual, eager.best_virtual)
    assert torch.equal(graphed.best_psi, eager.best_psi)
    # a second stage with the same signature reuses the captured loop
    n_loops = len(ctx.loops)
    _run_stage(
        ctx, result, spec, store, space, batch=BatchShape(4, 4), use_cuda_graph=True
    )
    assert len(ctx.loops) == n_loops


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
# the torch-backend fallback compiles _reduce_stats for complex32 input; under
# Python 3.13 triton's AST building emits a DeprecationWarning that pytest's
# warnings-as-errors would otherwise turn into an inductor CompilationError
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    ("precision", "pair_dtype"),
    [("fp32", torch.float32), ("fp32", torch.float16), ("fp16", torch.float16)],
)
def test_cuda_pair_slab_eager_graph_and_fallback(
    synthetic_result, precision, pair_dtype
):
    device = torch.device("cuda")
    result = synthetic_result(n_orientations=8)
    space = HypothesisSpace.from_result(result, n_defocus=2)
    spec = StageSpec([(0, 0, 4, 3), (0, 3, 2, 2)], precision=precision, n_psi=128)
    store, features = _fill_store(result, spec, 2, 8, device)
    ctx = SearchContext(device, WeightSource(result, device), want_pairs=True)
    n_virtual = 2 * result.num_fourier_filters * 8
    tol = (
        {"rtol": 1e-4, "atol": 1e-4}
        if precision == "fp32" and pair_dtype == torch.float32
        else {"rtol": 2e-2, "atol": 5e-2}
    )

    def run(use_cuda_graph, use_fused_kernel=True):
        rec = _SlabRecorder(store.n_pixels, n_virtual, pair_dtype, device)
        plan, maps = _run_stage(
            ctx,
            result,
            spec,
            store,
            space,
            batch=BatchShape(4, 4),
            use_cuda_graph=use_cuda_graph,
            use_fused_kernel=use_fused_kernel,
            pair_dtype=pair_dtype,
            collector=rec,
        )
        torch.cuda.synchronize()
        return plan, maps, rec

    plan_e, eager, rec_e = run(use_cuda_graph=False)
    if ctx.caps.pair_kernel:
        assert plan_e.stats_backend.name == "fused"
    assert plan_e.memory.loop == "eager"
    ref = _reference_pair_max(result, spec, features, plan_e.hyp_plan, 128)
    torch.testing.assert_close(rec_e.pair_max.float().cpu(), ref, **tol)
    _assert_matches(
        eager,
        _reference(result, spec, features, plan_e.hyp_plan, 128),
        plan_e.hyp_plan,
        **(
            {"rtol": 1e-4, "atol": 1e-4}
            if precision == "fp32"
            else {"rtol": 2e-2, "atol": 5e-2}
        ),
    )
    assert not torch.isnan(rec_e.pair_max.float()).any()

    plan_g, graphed, rec_g = run(use_cuda_graph=True)
    if ctx.caps.pair_kernel:
        assert plan_g.memory.loop == "graph"
        assert isinstance(ctx.loops[plan_g.signature], GraphedHypLoop)
    assert torch.equal(rec_g.pair_max, rec_e.pair_max)
    assert torch.equal(graphed.best_virtual, eager.best_virtual)

    # the torch backend (no fused kernel) emits the same slab from corr.amax
    plan_t, _torch_maps, rec_t = run(use_cuda_graph=False, use_fused_kernel=False)
    assert plan_t.stats_backend.name == "torch"
    torch.testing.assert_close(rec_t.pair_max.float(), rec_e.pair_max.float(), **tol)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_cuda_epochs_merge_with_base(synthetic_result, monkeypatch):
    """Force several epochs through a narrow packed-index budget on the fused path."""
    device = torch.device("cuda")
    result = synthetic_result(n_orientations=12)
    space = HypothesisSpace.from_result(result, n_defocus=2)
    spec = StageSpec([(0, 0, 4, 5)], n_psi=128)
    store, features = _fill_store(result, spec, 2, 8, device)
    ctx = SearchContext(device, WeightSource(result, device))
    import panther_em.inference.search.executor as executor_mod

    original = executor_mod.HypothesisPlan

    def narrow(space, subset, hyp_batch, n_psi):
        return original(
            space, subset, hyp_batch=hyp_batch, n_psi=n_psi, packed_index_bits=10
        )

    monkeypatch.setattr(executor_mod, "HypothesisPlan", narrow)
    plan, maps = _run_stage(
        ctx, result, spec, store, space, batch=BatchShape(4, 2), use_cuda_graph=True
    )
    assert len(plan.hyp_plan.epochs) > 1
    ref = _reference(
        result, spec, features, HypothesisPlan(space, None, hyp_batch=2, n_psi=128), 128
    )
    _assert_matches(maps, ref, plan.hyp_plan)


def _fill_store_split(result, spec, plane_groups, block_pixels, device, seed=0):
    """Like ``_fill_store`` but each rectangle's planes live in several groups."""
    n_defocus = sum(len(pg) for pg in plane_groups)
    geo = _geometry(1)
    pixels = PixelSelection.all(geo)
    store = FeatureStore(
        geo,
        pixels,
        pixels.index,
        block_pixels=block_pixels,
        n_defocus=n_defocus,
        device=device,
    )
    gen = torch.Generator().manual_seed(seed)
    features = {}
    for rect in spec.rectangles:
        n_cells = rect[2] * rect[3]
        re = torch.randn(n_defocus, n_cells, store.n_pixels, generator=gen)
        im = torch.randn(n_defocus, n_cells, store.n_pixels, generator=gen)
        data = torch.complex(re, im)
        for planes in plane_groups:
            group = store.add_group(rect, spec.precision, planes)
            w = store.writer(group)
            for plane in planes:
                w.write(
                    slice(0, n_cells), (0, store.n_pixels), data[plane], plane=plane
                )
            w.close()
        features[rect] = data
    return store, features


def test_planes_split_across_groups_match_dense_reference(synthetic_result):
    result = synthetic_result()
    space = HypothesisSpace.from_result(result, n_defocus=3)
    spec = StageSpec([(0, 0, 4, 3), (0, 3, 2, 2)], n_psi=16)
    store, features = _fill_store_split(result, spec, ((0, 2), (1,)), 8, "cpu")
    assert [g.planes for g in store.groups] == [(0, 2), (1,), (0, 2), (1,)]
    ctx = SearchContext("cpu", WeightSource(result, "cpu"))
    plan, maps = _run_stage(ctx, result, spec, store, space, batch=BatchShape(4, 6))
    assert len(plan.groups) == 4
    # rect 0: planes 0,1,2 -> (group 0 local 0), (group 1 local 0), (group 0 local 1)
    assert plan.plane_lookup[0] == ((0, 0), (1, 0), (0, 1))
    assert plan.plane_lookup[1] == ((2, 0), (3, 0), (2, 1))
    _assert_matches(
        maps, _reference(result, spec, features, plan.hyp_plan, 16), plan.hyp_plan
    )

    # a stage restricted to plane 2 reads only the first group's local slab 1
    subset = HypothesisSubset.defocus(space, torch.tensor([2]))
    plan2, maps2 = _run_stage(
        ctx, result, spec, store, space, subset=subset, batch=BatchShape(4, 6)
    )
    assert plan2.plane_lookup[0] == ((0, 1),) and len(plan2.groups) == 2
    _assert_matches(
        maps2, _reference(result, spec, features, plan2.hyp_plan, 16), plan2.hyp_plan
    )
    assert plan2.signature != plan.signature


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_cuda_pinned_host_store_streams_blocks_and_matches_device(synthetic_result):
    """Blocks written through device staging, flushed to pinned memory and streamed
    back through slots give the same maps and pair slab as a device-resident store."""
    device = torch.device("cuda")
    result = synthetic_result(n_orientations=8)
    space = HypothesisSpace.from_result(result, n_defocus=2)
    spec = StageSpec([(0, 0, 4, 3), (0, 3, 2, 2)], precision="fp16", n_psi=128)
    ctx = SearchContext(device, WeightSource(result, device), want_pairs=True)
    runs = {}
    for residency in ("device", "pinned_host"):
        geo = _geometry(1)
        pixels = PixelSelection.all(geo)
        store = FeatureStore(
            geo, pixels, pixels.index, block_pixels=8, n_defocus=2, device=device,
            residency=residency,
        )
        store.copy_stream = ctx.copy
        gen = torch.Generator().manual_seed(0)
        for rect in spec.rectangles:
            group = store.add_group(rect, spec.precision)
            n_cells = rect[2] * rect[3]
            data = torch.complex(
                torch.randn(2, n_cells, store.n_pixels, generator=gen),
                torch.randn(2, n_cells, store.n_pixels, generator=gen),
            )
            w = store.writer(group)
            # several writes so staging sees partial blocks before completion
            for c0 in range(0, n_cells, 3):
                w.write(slice(c0, min(n_cells, c0 + 3)), (0, store.n_pixels), data[:, c0 : c0 + 3])
            w.close()
            assert not store._staging
        if residency == "pinned_host":
            assert all(not b.is_cuda and b.is_pinned() for g in store.groups for b in g.blocks)
        rec = _SlabRecorder(store.n_pixels, 2 * 8, torch.float32, device)
        plan, maps = _run_stage(
            ctx, result, spec, store, space, batch=BatchShape(4, 4),
            use_cuda_graph=True, pair_dtype=torch.float32, collector=rec,
        )
        torch.cuda.synchronize()
        runs[residency] = (maps, rec.pair_max.clone())
    torch.testing.assert_close(runs["pinned_host"][0].mip, runs["device"][0].mip)
    torch.testing.assert_close(runs["pinned_host"][0].mean, runs["device"][0].mean)
    assert torch.equal(runs["pinned_host"][0].best_virtual, runs["device"][0].best_virtual)
    torch.testing.assert_close(runs["pinned_host"][1], runs["device"][1])

