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
    )
    plan = plan_stage(ctx, 0, spec, store, memory, backend, engine, hyp_plan)
    maps = StageExecutor(ctx).run(store, plan)
    return plan, maps


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
    back through slots give the same maps as a device-resident store."""
    device = torch.device("cuda")
    result = synthetic_result(n_orientations=8)
    space = HypothesisSpace.from_result(result, n_defocus=2)
    spec = StageSpec([(0, 0, 4, 3), (0, 3, 2, 2)], precision="fp16", n_psi=128)
    ctx = SearchContext(device, WeightSource(result, device))
    runs = {}
    for residency in ("device", "pinned_host"):
        geo = _geometry(1)
        pixels = PixelSelection.all(geo)
        store = FeatureStore(
            geo,
            pixels,
            pixels.index,
            block_pixels=8,
            n_defocus=2,
            device=device,
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
                w.write(
                    slice(c0, min(n_cells, c0 + 3)),
                    (0, store.n_pixels),
                    data[:, c0 : c0 + 3],
                )
            w.close()
            assert not store._staging
        if residency == "pinned_host":
            assert all(
                not b.is_cuda and b.is_pinned() for g in store.groups for b in g.blocks
            )
        plan, maps = _run_stage(
            ctx, result, spec, store, space, batch=BatchShape(4, 4), use_cuda_graph=True
        )
        torch.cuda.synchronize()
        runs[residency] = maps
    torch.testing.assert_close(runs["pinned_host"].mip, runs["device"].mip)
    torch.testing.assert_close(runs["pinned_host"].mean, runs["device"].mean)
    assert torch.equal(runs["pinned_host"].best_virtual, runs["device"].best_virtual)
