"""Local stages: HEALPix neighborhoods, closed-form moments, local correlation."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import read_group, requires_cuda
from test_search import EXEC, EXEC_JOINT, StubFeaturizer

from panther_em.inference.search.config import Tuning
from panther_em.inference.search.correlation import (
    Correlator,
    LocalWork,
    LoopPlan,
    MomentForms,
)
from panther_em.inference.search.correlation.contraction import BatchShape
from panther_em.inference.search.correlation.local import (
    choose_tile,
    correlate_local,
    make_tiles,
)
from panther_em.inference.search.correlation.moments import (
    block_moments,
    moments_to_stats,
)
from panther_em.inference.search.featurization import FeatureChunk, FeatureStore
from panther_em.inference.search.hypotheses import HypothesisSpace, LocalHypotheses
from panther_em.inference.search.images import PixelSelection, SearchImages
from panther_em.inference.search.planning import DeviceInfo, plan_memory
from panther_em.inference.search.search import Searcher
from panther_em.inference.search.stage import Stage, stage_from_json, stage_to_json
from panther_em.inference.search.template import Template

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array_wrap__ must accept.*:DeprecationWarning",
    "ignore:.*__array__ implementation doesn't accept.*:DeprecationWarning",
    "ignore:ComplexHalf support is experimental",
)


# --------------------------------------------------------------------------- #
# neighborhoods
# --------------------------------------------------------------------------- #


def _ring_reference(nside: int, pixels, rings: int) -> set[int]:
    """Pixels within ``rings`` hops of ``pixels`` (torch_so3's breadth-first search)."""
    from torch_so3.hierarchical_s2_grid import HierarchicalS2Grid

    grid = HierarchicalS2Grid(nside, 1)
    return set(grid.neighbor_indices(np.asarray(pixels), level=0, rings=rings).tolist())


@pytest.mark.parametrize("rings", [0, 1, 2])
def test_neighborhoods_are_rings_on_the_grid(rings):
    nside = 4
    o = 12 * nside**2
    rng = np.random.default_rng(rings)
    ipix = rng.permutation(o)  # orientation i sits at NESTED pixel ipix[i]
    space = SimpleNamespace(n_orientations=o, n_filters=2)
    local = LocalHypotheses.build(
        space, nside, ipix, level=0, seed_level=0, rings=rings, template_ids=None
    )
    assert local.base_level == 0
    for seed in rng.choice(o, 12, replace=False):
        cell = int(local.groups(torch.tensor([seed]))[0])
        n = int(local.counts[cell])
        rows = local.rows[cell, :n].tolist()
        want = {
            int(np.nonzero(ipix == p)[0][0])
            for p in _ring_reference(nside, [ipix[seed]], rings)
        }
        assert set(rows) == want | {o + w for w in want}  # every filter
        assert len(rows) == n == 2 * len(want)
        assert set(local.rows[cell, n:].tolist()) <= {rows[0]}  # padding repeats
    if rings == 0:
        assert local.width == 2


def test_coarse_seeds_keep_their_whole_cell():
    """Seeds from a level-1 stage: the cell's four orientations, widened by a ring."""
    nside = 4
    ipix = np.arange(12 * nside**2)
    space = SimpleNamespace(n_orientations=ipix.size, n_filters=1)
    local = LocalHypotheses.build(
        space, nside, ipix, level=0, seed_level=1, rings=1, template_ids=None
    )
    assert local.base_level == 1
    for seed in (0, 4, 37 * 4, 191 - 3):
        cell = int(local.groups(torch.tensor([seed]))[0])
        assert cell == seed // 4
        children = [4 * cell + c for c in range(4)]
        n = int(local.counts[cell])
        assert set(local.rows[cell, :n].tolist()) == _ring_reference(nside, children, 1)


def test_level_stage_rows_index_its_selected_orientations():
    """A level-1 local stage: positions into the stage's own (level-1) template rows."""
    nside = 4
    ipix = np.arange(12 * nside**2)
    space = SimpleNamespace(n_orientations=ipix.size, n_filters=1)
    selected = torch.arange(0, ipix.size, 4)  # healpix_level_ids(ipix, 1)
    local = LocalHypotheses.build(
        space, nside, ipix, level=1, seed_level=0, rings=1, template_ids=selected
    )
    seed = 4 * 17 + 2  # a fine seed inside level-1 pixel 17
    cell = int(local.groups(torch.tensor([seed]))[0])
    n = int(local.counts[cell])
    orientations = {int(selected[j]) for j in local.rows[cell, :n].tolist()}
    assert orientations == {4 * p for p in _ring_reference(nside // 2, [17], 1)}


def test_partial_template_drops_missing_orientations():
    nside = 2
    full = np.arange(12 * nside**2)
    ipix = full[full % 7 != 3]  # orientations on 41 of the 48 pixels
    space = SimpleNamespace(n_orientations=ipix.size, n_filters=1)
    local = LocalHypotheses.build(
        space, nside, ipix, level=0, seed_level=0, rings=1, template_ids=None
    )
    for seed in range(ipix.size):
        cell = int(local.groups(torch.tensor([seed]))[0])
        n = int(local.counts[cell])
        got = {int(ipix[j]) for j in local.rows[cell, :n].tolist()}
        assert got == _ring_reference(nside, [ipix[seed]], 1) & set(ipix.tolist())


# --------------------------------------------------------------------------- #
# Tiles
# --------------------------------------------------------------------------- #


def test_tiles_cover_every_column_once_within_one_group():
    cells = torch.tensor([3, 1, 3, 3, 0, 1, 3, 3, 3, 0, 2])
    columns, valid, tile_cell = make_tiles(cells, 2)
    real = columns[valid]
    assert sorted(real.tolist()) == list(range(cells.numel()))
    for t in range(columns.shape[0]):
        assert set(cells[columns[t]].tolist()) == {int(tile_cell[t])}
    assert columns.shape[0] == 1 + 1 + 1 + 3  # cells 0, 1, 2, 3 have 2, 2, 1, 6
    assert make_tiles(torch.empty(0, dtype=torch.long), 4)[0].shape == (0, 4)


def test_tile_choice_follows_the_group_sizes():
    common = {"n_planes": 1, "width": 49, "cells": 5760, "n_freq": 80, "n_psi": 256}
    assert choose_tile(np.ones(1000, dtype=np.int64), **common) == 8
    assert choose_tile(np.full(100, 128), **common) >= 64


# --------------------------------------------------------------------------- #
# A store and a pass, built by hand
# --------------------------------------------------------------------------- #


def _pass(
    synthetic_result,
    rects,
    n_psi,
    *,
    precision="fp32",
    orientations=None,
    n_planes=2,
    n_pixels=37,
    device="cpu",
):
    template = Template(
        synthetic_result(n_orientations=6, n_filters=2, k_max=6, eig_max=6),
        device=device,
    )
    space = HypothesisSpace.from_result(template.result, n_defocus=n_planes)
    stage = Stage.create(
        rects, precision=precision, n_psi=n_psi, orientations=orientations
    )
    template_ids, defocus_ids = stage.hypothesis_ids(space)
    tuning = Tuning(
        batch=BatchShape(8, 5),
        block_pixels=16,
        use_cuda_graph=False,
        use_fused_kernel=False,
    )
    memory = plan_memory(
        DeviceInfo(torch.device(device)),
        stage,
        n_pixels=n_pixels,
        space=space,
        template_ids=template_ids,
        defocus_ids=defocus_ids,
        tuning=tuning,
    )
    planes = memory.hyps.defocus_planes()
    store = FeatureStore(
        stage.rectangles, stage.precision, planes, n_pixels, 16, device
    )
    gen = torch.Generator().manual_seed(7)
    for rect in stage.rectangles:
        n_cells = rect[2] * rect[3]
        for d in range(len(planes)):
            values = torch.complex(
                torch.randn(n_cells, n_pixels, generator=gen),
                torch.randn(n_cells, n_pixels, generator=gen),
            ).to(device)
            store.write(
                rect, FeatureChunk(d, slice(0, n_cells), slice(0, n_pixels), values)
            )
    store.check_complete()
    return template, LoopPlan.create(stage, store, memory), store


CASES = [
    ([(0, 0, 4, 5)], 8, None),  # no Nyquist bin
    ([(0, 0, 4, 5)], 6, None),  # k_stop = n_psi / 2 + 1: a Nyquist bin
    ([(0, 0, 3, 3), (1, 3, 4, 2)], 8, [0, 2, 3, 5]),  # three k-runs, two parts
]


def _check_moments(template, plan, store, device, rtol=2e-4):
    maps = Correlator(device, template.weights).run(store, plan)
    forms = MomentForms.build(
        template.weights,
        plan.layout,
        plan.engine,
        plan.stage.n_psi,
        template_rows=None if plan.hyps.is_contiguous else plan.hyps.template_rows(),
    )
    for block in store.blocks(plan.groups):
        start, n = block.col_range[0], block.n_valid
        s1, s2 = block_moments(plan, forms, block.operands, n)
        mean, var = moments_to_stats(s1, s2, plan.hyps.n_evaluated)
        ref_var = maps.variance[start : start + n]
        torch.testing.assert_close(var, ref_var, rtol=rtol, atol=0)
        scale = ref_var.sqrt()
        assert ((mean - maps.mean[start : start + n]).abs() <= rtol * scale).all()


@pytest.mark.parametrize(("rects", "n_psi", "orientations"), CASES)
def test_closed_form_moments_equal_the_loop(
    synthetic_result, rects, n_psi, orientations
):
    template, plan, store = _pass(
        synthetic_result, rects, n_psi, orientations=orientations
    )
    _check_moments(template, plan, store, "cpu")


@requires_cuda
@pytest.mark.parametrize(("rects", "n_psi", "orientations"), CASES)
def test_closed_form_moments_equal_the_loop_fp16(
    synthetic_result, rects, n_psi, orientations
):
    template, plan, store = _pass(
        synthetic_result,
        rects,
        n_psi,
        precision="fp16",
        orientations=orientations,
        device="cuda",
    )
    # the loop's spectrum is fp16-rounded; the forms are exact in fp32
    _check_moments(template, plan, store, "cuda", rtol=2e-3)


# --------------------------------------------------------------------------- #
# Local correlation vs brute force
# --------------------------------------------------------------------------- #


def _local_reference(template, plan, store, work):
    """Per column: the maximum over its group's hypotheses by an explicit irfft."""
    hyps = plan.hyps
    n_psi, f = plan.stage.n_psi, plan.n_freq
    us = template.weights.us.to(torch.complex64)  # (N, K, M) from (0, 0)
    rows = hyps.template_rows(us.device)
    out = []
    for p in range(store.n_pixels):
        j = work.rows[int(work.cells[p])].to(us.device)
        w = us[rows[j]]  # (W, K, M)
        c = torch.zeros(
            hyps.n_defocus_selected,
            j.numel(),
            f,
            dtype=torch.complex64,
            device=us.device,
        )
        for rect in plan.layout.rectangles:
            k0, m0, ke, me = rect
            group = store.group(rect)
            for d in range(hyps.n_defocus_selected):
                y = read_group(store, group, d)[p]  # (ke, me)
                c[d, :, k0 : k0 + ke] += torch.einsum(
                    "km,wkm->wk", y, w[:, k0 : k0 + ke, m0 : m0 + me].conj()
                )
        c[..., 0] = c[..., 0].real
        if n_psi % 2 == 0 and f == n_psi // 2 + 1:
            c[..., -1] = c[..., -1].real
        corr = torch.fft.irfft(c, n=n_psi, dim=-1, norm="forward")
        value, flat = corr.reshape(-1).max(dim=0)
        d, rest = divmod(int(flat), j.numel() * n_psi)
        slot, raw = divmod(rest, n_psi)
        out.append(
            (float(value), d * hyps.n_selected + int(j[slot]), (n_psi - raw) % n_psi)
        )
    return out


@pytest.mark.parametrize(("rects", "n_psi", "orientations"), CASES)
@pytest.mark.parametrize("chunk_bytes", [1, 1 << 30])
def test_local_correlation_matches_brute_force(
    synthetic_result, rects, n_psi, orientations, chunk_bytes
):
    template, plan, store = _pass(
        synthetic_result, rects, n_psi, orientations=orientations
    )
    gen = torch.Generator().manual_seed(3)
    n_sel = plan.hyps.n_selected
    work = LocalWork(
        cells=torch.randint(0, 4, (store.n_pixels,), generator=gen),
        rows=torch.randint(0, n_sel, (4, 3), generator=gen),
    )
    correlator = Correlator("cpu", template.weights)
    forms = correlator.moment_forms(plan)
    maps = correlate_local(
        store, plan, work, template.weights, forms, chunk_bytes=chunk_bytes
    )
    ref = _local_reference(template, plan, store, work)
    mip = torch.tensor([r[0] for r in ref])
    torch.testing.assert_close(maps.mip, mip, rtol=1e-5, atol=1e-5)
    assert maps.best_virtual.tolist() == [r[1] for r in ref]
    assert maps.best_psi.tolist() == [r[2] for r in ref]
    # the whole set's moments, not the neighborhood's
    full = correlator.run(store, plan)
    torch.testing.assert_close(maps.variance, full.variance, rtol=2e-4, atol=0)
    assert maps.n_evaluated == full.n_evaluated
    torch.testing.assert_close(
        maps.zscore, (maps.mip - maps.mean) / maps.variance.sqrt()
    )


# --------------------------------------------------------------------------- #
# Local stages in a search
# --------------------------------------------------------------------------- #


def _healpix_template(synthetic_result):
    return Template(synthetic_result(s2_nside=2, n_filters=2), device="cpu")


def _images():
    gen = torch.Generator().manual_seed(5)
    return SearchImages.from_stack(
        torch.randn(1, 30, 30, generator=gen),
        filters=1.0 + torch.rand(2, 30, 16, generator=gen),
    )


def _two_stages(stage1_kwargs, level0=None):
    return [
        Stage.create([(0, 0, 3, 3)], n_psi=8, orientations=level0, name="coarse"),
        Stage.create([(0, 0, 4, 5)], n_psi=8, name="fine", **stage1_kwargs),
    ]


def _run(template, stages, execution=EXEC_JOINT):
    searcher = Searcher(
        template,
        stages,
        execution=execution,
        featurization=StubFeaturizer(),
        n_defocus=2,
        keep_stage_results=True,
    )
    session = searcher.session(_images())
    results = list(session)
    return searcher, session, results


def test_local_stage_covering_everything_equals_the_full_stage(synthetic_result):
    template = _healpix_template(synthetic_result)
    _, _, full = _run(template, _two_stages({}))
    searcher, _, local = _run(template, _two_stages({"local_rings": 8}))
    assert int(searcher.local[1].counts.min()) == 2 * 48  # every orientation, filter
    a, b = full[1].maps, local[1].maps
    torch.testing.assert_close(b.mip, a.mip, rtol=1e-5, atol=1e-5)
    assert torch.equal(full[1].best_index, local[1].best_index)
    assert torch.equal(a.best_psi, b.best_psi)
    torch.testing.assert_close(b.variance, a.variance, rtol=2e-4, atol=0)
    torch.testing.assert_close(b.zscore, a.zscore, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("level0", [None, 1])
def test_local_stage_scores_each_pixel_over_its_neighborhood(synthetic_result, level0):
    template = _healpix_template(synthetic_result)
    searcher, session, results = _run(
        template, _two_stages({"local_rings": 1}, level0=level0)
    )
    local = searcher.local[1]
    assert local.base_level == (1 if level0 else 0)
    seeds = searcher.space.decode(results[0].best_index)[2]
    cells = local.groups(seeds)
    # the reference reads the features the local pass correlated
    plan = searcher.plan(1, session.pixels, session.geometry, seeds)
    assert torch.equal(plan.bands[0].cells, cells)
    store = session.store
    loop_plan = LoopPlan.create(plan.stage, store, plan.bands[0].passes[0].memory)
    ref = _local_reference(template, loop_plan, store, LocalWork(cells, local.rows))
    maps = results[1].maps
    torch.testing.assert_close(
        maps.mip, torch.tensor([r[0] for r in ref]), rtol=1e-5, atol=1e-5
    )
    assert maps.best_virtual.tolist() == [r[1] for r in ref]
    # the best orientation lies in the seed's widened cell
    o = searcher.space.decode(results[1].best_index)[2]
    assert all(
        int(o_p) in (local.rows[c] % searcher.space.n_orientations).tolist()
        for o_p, c in zip(o, cells, strict=True)
    )


def test_local_stage_plane_passes_and_bands_match_the_joint_pass(
    synthetic_result, monkeypatch
):
    template = _healpix_template(synthetic_result)
    stages = _two_stages({"local_rings": 1}, level0=1)
    _, _, joint = _run(template, stages, EXEC_JOINT)
    _, _, planes = _run(template, stages, EXEC)  # one plane per pass, merged
    monkeypatch.setattr(DeviceInfo, "free_bytes", lambda self: 40_000)
    searcher, session, banded = _run(template, stages, EXEC_JOINT)
    pixels = PixelSelection.all(session.geometry)
    plan = searcher.plan(1, pixels, session.geometry, torch.zeros(len(pixels)).long())
    assert len(plan.bands) > 1
    assert sum(len(b.cells) for b in plan.bands) == len(pixels)
    for other in (planes, banded):
        a, b = joint[1].maps, other[1].maps
        torch.testing.assert_close(b.mip, a.mip, rtol=1e-5, atol=1e-5)
        assert torch.equal(joint[1].best_index, other[1].best_index)
        torch.testing.assert_close(b.variance, a.variance, rtol=2e-4, atol=0)


def test_local_stage_validation(synthetic_result):
    template = _healpix_template(synthetic_result)
    with pytest.raises(ValueError, match="stage 0 cannot be local"):
        Searcher(template, [Stage.create([(0, 0, 2, 2)], n_psi=8, local_rings=1)])
    with pytest.raises(ValueError, match="HEALPix grid"):
        Stage.create([(0, 0, 2, 2)], n_psi=8, orientations=[0, 1], local_rings=1)
    with pytest.raises(ValueError, match="local_rings"):
        Stage.create([(0, 0, 2, 2)], n_psi=8, local_rings=-1)
    plain = Template(synthetic_result(n_orientations=6), device="cpu")
    with pytest.raises(ValueError, match="HEALPix pixel centers"):
        Searcher(plain, _two_stages({"local_rings": 1}))
    stage = _two_stages({"local_rings": 2})[1]
    assert stage_from_json(stage_to_json(stage)).local_rings == 2
    assert replace(stage, local_rings=None).is_local is False


@requires_cuda
@pytest.mark.parametrize(("rects", "n_psi", "orientations"), CASES)
def test_local_correlation_fp16_store_matches_brute_force(
    synthetic_result, rects, n_psi, orientations
):
    template, plan, store = _pass(
        synthetic_result,
        rects,
        n_psi,
        precision="fp16",
        orientations=orientations,
        device="cuda",
    )
    gen = torch.Generator().manual_seed(4)
    work = LocalWork(
        cells=torch.randint(0, 3, (store.n_pixels,), generator=gen),
        rows=torch.randint(0, plan.hyps.n_selected, (3, 5), generator=gen),
    )
    correlator = Correlator("cuda", template.weights)
    maps = correlator.run_local(store, plan, work, chunk_bytes=1 << 20)
    ref = _local_reference(template, plan, store, work)
    torch.testing.assert_close(
        maps.mip.cpu(), torch.tensor([r[0] for r in ref]), rtol=1e-5, atol=1e-5
    )
    assert maps.best_virtual.tolist() == [r[1] for r in ref]
    assert maps.best_psi.tolist() == [r[2] for r in ref]


@requires_cuda
@pytest.mark.parametrize("local", [False, True])
def test_correlator_waits_for_the_callers_stream(synthetic_result, local):
    """Work queued on the caller's stream (features, forms) precedes the correlation.

    The caller's stream is held up by a spin kernel before it doubles the features;
    a correlation that does not wait for it reads the old features.
    """
    template, plan, store = _pass(synthetic_result, [(0, 0, 4, 5)], 8, device="cuda")
    work = LocalWork(
        cells=torch.arange(store.n_pixels) % 3,
        rows=torch.arange(3 * 4).reshape(3, 4) % plan.hyps.n_selected,
    )

    def correlate(correlator):
        if local:
            return correlator.run_local(store, plan, work, chunk_bytes=1 << 20)
        return correlator.run(store, plan)

    before = correlate(Correlator("cuda", template.weights))
    torch.cuda.synchronize()
    torch.cuda._sleep(int(2e9))  # ~1 s on the caller's stream
    for group in store.groups:
        group.data.mul_(2)
    after = correlate(Correlator("cuda", template.weights))  # forms built after it
    torch.cuda.synchronize()
    torch.testing.assert_close(after.mip, 2 * before.mip)
    torch.testing.assert_close(after.variance, 4 * before.variance, rtol=1e-4, atol=0)
