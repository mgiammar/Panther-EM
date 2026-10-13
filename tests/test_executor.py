"""Executor end to end vs a dense reference (CPU torch path; CUDA fused/graph)."""

from __future__ import annotations

import pytest
import torch

from panther_em.inference.search.basis import WeightSource
from panther_em.inference.search.contraction import ComplexGemmEngine
from panther_em.inference.search.executor import (
    Executor,
    hyp_batch_multiple,
    plan_memory,
    plan_stage,
)
from panther_em.inference.search.hypotheses import HypBatch
from panther_em.inference.search.tiling import FeatureLayout
from panther_em.inference.search.hypotheses import HypothesisSet, HypothesisSpace
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
    template_ids=None,
    defocus_ids=None,
    batch=None,
    use_cuda_graph=False,
    use_fused_kernel=True,
):
    memory = plan_memory(
        ctx,
        spec,
        n_pixels=store.n_pixels,
        space=space,
        template_ids=template_ids,
        defocus_ids=defocus_ids,
        batch=batch,
        block_pixels=store.block_pixels,
        use_cuda_graph=use_cuda_graph,
        use_fused_kernel=use_fused_kernel,
        store=store,
    )
    plan = plan_stage(ctx, 0, spec, store, memory)
    maps = ctx.run(store, plan)
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
    spec = StageSpec.create([(0, 0, 4, 3), (0, 3, 2, 2)], precision="fp32", n_psi=16)
    store, features = _fill_store(result, spec, 1, block_pixels, "cpu")
    ctx = Executor("cpu", WeightSource(result, "cpu"))
    plan, maps = _run_stage(ctx, result, spec, store, space, batch=BatchShape(4, 4))
    assert plan.memory.block_pixels == block_pixels and plan.memory.loop == "eager"
    assert plan.fused is False
    _assert_matches(
        maps, _reference(result, spec, features, plan.hyps, 16), plan.hyps
    )


def test_defocus_axis_reduces_over_planes(synthetic_result):
    result = synthetic_result()
    space = HypothesisSpace.from_result(result, n_defocus=3)
    spec = StageSpec.create([(0, 0, 4, 5)], n_psi=16)
    store, features = _fill_store(result, spec, 3, 8, "cpu")
    ctx = Executor("cpu", WeightSource(result, "cpu"))
    plan, maps = _run_stage(ctx, result, spec, store, space, batch=BatchShape(4, 6))
    ref = _reference(result, spec, features, plan.hyps, 16)
    _assert_matches(maps, ref, plan.hyps)
    # the best virtual id decodes to a (defocus, filter, orientation) triple
    decoded_d, _, decoded_o = space.decode(plan.hyps.decode(maps.best_virtual))
    assert decoded_d.max() <= 2 and decoded_o.max() <= 5
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
    spec = StageSpec.create([(0, 0, 3, 4)], n_psi=16)
    store, features = _fill_store(result, spec, 2, 25, "cpu")
    ctx = Executor("cpu", WeightSource(result, "cpu"))
    plan, maps = _run_stage(
        ctx,
        result,
        spec,
        store,
        space,
        template_ids=torch.arange(1, 10, 3),
        defocus_ids=torch.tensor([1]),
        batch=BatchShape(5, 2),
    )
    assert plan.hyps.n_virtual == 3 and not plan.hyps.is_contiguous
    ref = _reference(result, spec, features, plan.hyps, 16)
    _assert_matches(maps, ref, plan.hyps)
    global_ids = plan.hyps.decode(maps.best_virtual)
    decoded_d, _, _ = space.decode(global_ids)
    assert torch.all(decoded_d == 1)
    assert set(global_ids.tolist()) <= {1 * 10 + t for t in (1, 4, 7)}


def test_plan_memory_defaults_and_overrides(synthetic_result):
    result = synthetic_result(n_orientations=40)
    space = HypothesisSpace.from_result(result, n_defocus=3)
    ctx = Executor("cpu", WeightSource(result, "cpu"))
    spec = StageSpec.create([(0, 0, 4, 5)], n_psi=256)
    plan = plan_memory(
        ctx, spec, n_pixels=1000, space=space, l2_target=64 * 1024**2, block_bytes_max=1024
    )
    # the L2 budget allows more than the 120 hypotheses, so the batch is clamped to them
    assert plan.hyp_batch == 120 and plan.block_pixels % plan.gemm_pixels == 0
    assert plan.loop == "eager" and plan.n_sub >= 1 and not plan.fused
    assert plan.residency == "device" and plan.hyps.hyp_batch == plan.hyp_batch
    assert plan.hyps.n_virtual == 3 * 40
    explicit = plan_memory(
        ctx,
        spec,
        n_pixels=10,
        space=space,
        defocus_ids=torch.tensor([1]),
        batch=BatchShape(4, 3),
        block_pixels=9,
    )
    assert (explicit.gemm_pixels, explicit.hyp_batch, explicit.block_pixels) == (4, 3, 8)
    assert explicit.hyps.defocus_planes() == (1,)
    assert hyp_batch_multiple(256) == 64 and hyp_batch_multiple(128) == 128


def test_executor_prepared_weights_hit_and_evict(synthetic_result):
    result = synthetic_result()
    src = WeightSource(result, "cpu")
    layout = FeatureLayout([(0, 0, 3, 2), (0, 2, 3, 3), (3, 0, 1, 5)])
    engine = ComplexGemmEngine()
    run = layout.k_runs[0]  # k in [0, 3): both first rectangles
    batch = HypBatch(0, 1, 5, 1)

    ctx = Executor("cpu", src)
    got = ctx.prepared(layout, run, batch, engine)
    manual = engine.prepare_weights(
        torch.cat([src.block((0, 0, 3, 2))[1:5], src.block((0, 2, 3, 3))[1:5]], dim=-1)
    )
    torch.testing.assert_close(got, manual)
    assert got.shape == (3, 5, 4)
    assert ctx.prepared(layout, run, batch, engine) is got
    assert ctx.prepared_bytes == got.numel() * got.element_size()

    rows = torch.tensor([0, 2, 4, 5])
    sub = ctx.prepared(layout, run, HypBatch(0, 0, 2, 0), engine, rows, rows_key="r")
    torch.testing.assert_close(
        sub,
        engine.prepare_weights(
            torch.cat(
                [src.block((0, 0, 3, 2))[[0, 2]], src.block((0, 2, 3, 3))[[0, 2]]],
                dim=-1,
            )
        ),
    )

    tiny = Executor("cpu", src, prepared_budget=got.numel() * got.element_size())
    a = tiny.prepared(layout, run, batch, engine)
    b = tiny.prepared(layout, layout.k_runs[1], batch, engine)  # over budget: newest kept
    assert tiny.prepared(layout, layout.k_runs[1], batch, engine) is b
    assert tiny.prepared(layout, run, batch, engine) is not a  # the older entry went
    tiny.release()
    assert tiny.prepared_bytes == 0


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_cuda_fused_eager_and_graph_match_reference(synthetic_result, precision):
    device = torch.device("cuda")
    result = synthetic_result(n_orientations=8)
    space = HypothesisSpace.from_result(result, n_defocus=2)
    spec = StageSpec.create([(0, 0, 4, 3), (0, 3, 2, 2)], precision=precision, n_psi=128)
    store, features = _fill_store(result, spec, 2, 8, device)
    ctx = Executor(device, WeightSource(result, device))
    tol = (
        {"rtol": 1e-4, "atol": 1e-4}
        if precision == "fp32"
        else {"rtol": 2e-2, "atol": 5e-2}
    )

    plan_e, eager = _run_stage(
        ctx, result, spec, store, space, batch=BatchShape(4, 4), use_cuda_graph=False
    )
    assert plan_e.fused and plan_e.memory.loop == "eager"
    assert ctx.loops[plan_e.signature].graph is None
    ref = _reference(result, spec, features, plan_e.hyps, 128)
    _assert_matches(eager, ref, plan_e.hyps, **tol)

    plan_g, graphed = _run_stage(
        ctx, result, spec, store, space, batch=BatchShape(4, 4), use_cuda_graph=True
    )
    assert plan_g.memory.loop == "graph"
    assert ctx.loops[plan_g.signature].graph is not None
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
    spec = StageSpec.create([(0, 0, 4, 3), (0, 3, 2, 2)], n_psi=16)
    store, features = _fill_store_split(result, spec, ((0, 2), (1,)), 8, "cpu")
    assert [g.planes for g in store.groups] == [(0, 2), (1,), (0, 2), (1,)]
    ctx = Executor("cpu", WeightSource(result, "cpu"))
    plan, maps = _run_stage(ctx, result, spec, store, space, batch=BatchShape(4, 6))
    assert len(plan.groups) == 4
    # rect 0: planes 0,1,2 -> (group 0 local 0), (group 1 local 0), (group 0 local 1)
    assert plan.plane_lookup[0] == ((0, 0), (1, 0), (0, 1))
    assert plan.plane_lookup[1] == ((2, 0), (3, 0), (2, 1))
    _assert_matches(
        maps, _reference(result, spec, features, plan.hyps, 16), plan.hyps
    )

    # a stage restricted to plane 2 reads only the first group's local slab 1
    plan2, maps2 = _run_stage(
        ctx, result, spec, store, space, defocus_ids=torch.tensor([2]), batch=BatchShape(4, 6)
    )
    assert plan2.plane_lookup[0] == ((0, 1),) and len(plan2.groups) == 2
    _assert_matches(
        maps2, _reference(result, spec, features, plan2.hyps, 16), plan2.hyps
    )
    assert plan2.signature != plan.signature


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_cuda_pinned_host_store_streams_blocks_and_matches_device(synthetic_result):
    """Blocks written through device staging, flushed to pinned memory and streamed
    back through slots give the same maps as a device-resident store."""
    device = torch.device("cuda")
    result = synthetic_result(n_orientations=8)
    space = HypothesisSpace.from_result(result, n_defocus=2)
    spec = StageSpec.create([(0, 0, 4, 3), (0, 3, 2, 2)], precision="fp16", n_psi=128)
    ctx = Executor(device, WeightSource(result, device))
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
