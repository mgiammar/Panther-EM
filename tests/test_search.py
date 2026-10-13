"""Searcher / SearchSession end to end: single stage and two stages vs dense references."""

from __future__ import annotations

import warnings

import pytest
import torch

from panther_em.inference.correlation import compute_feature_stack
from panther_em.inference.search.decision import Decision, ErrorModel, PixelLabel
from panther_em.inference.search.results import SearchResult, StageResult
from panther_em.inference.search.search import ExecutionConfig, Searcher
from panther_em.inference.search.stage import FixedError, Stage
from panther_em.inference.search.template import Template
from panther_em.inference.search.types import BatchShape, ImageSet

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array_wrap__ must accept.*:DeprecationWarning",
    "ignore:.*__array__ implementation doesn't accept.*:DeprecationWarning",
)

EXEC = ExecutionConfig(batch=BatchShape(8, 5), cell_chunk=4, use_cuda_graph=False)


class StubFeaturizer:
    """Deterministic random features per cell; a stand-in for ``featurize``."""

    def __init__(self, seed: int = 1) -> None:
        self.seed = seed
        self.calls = 0
        self.planes_produced = 0

    def __call__(
        self,
        bank,
        images,
        group,
        writer,
        schedule,
        *,
        chunk=64,
        order="pixel_major",
        stream=None,
    ):
        self.calls += 1
        self.planes_produced += len(group.planes)
        n = int(schedule.column_index.numel())
        k0, m0, ke, me = group.rect
        for c, (k, m) in enumerate((k0 + i // me, m0 + i % me) for i in range(ke * me)):
            gen = torch.Generator().manual_seed(self.seed * 100003 + k * 1009 + m)
            full = torch.complex(
                torch.randn(images.n_defocus, 10_000, generator=gen),
                torch.randn(images.n_defocus, 10_000, generator=gen),
            )
            # pixel-identity features, per global plane; the group holds a subset
            data = full[list(group.planes)][:, schedule.column_index]
            writer.write(slice(c, c + 1), (0, n), data[:, None, :])
        writer.close()


@pytest.fixture
def template(synthetic_result):
    return Template(synthetic_result(), device="cpu")


def _image(h=30, w=30, seed=0):
    return torch.randn(h, w, generator=torch.Generator().manual_seed(seed))


def _dense_reference(template, image, rects, n_psi, planes=None, filters=None):
    """Per-pixel statistics from the dense correlogram (kernels, FFT, einsum, irfft)."""
    recon = template.reconstructor
    us = torch.as_tensor(recon.result.U) * torch.as_tensor(recon.result.S)[None, None]
    us = us.reshape(-1, *us.shape[2:]).to(torch.complex64)  # (N, K, E)
    k_stop = max(k0 + ke for k0, _, ke, _ in rects)
    images = [image] if filters is None else [
        torch.fft.irfft2(torch.fft.rfft2(image) * filters[d], s=image.shape) for d in planes
    ]
    C = None
    for d, img in enumerate(images):
        for k0, m0, ke, me in rects:
            cells = torch.stack(
                torch.meshgrid(
                    torch.arange(k0, k0 + ke), torch.arange(m0, m0 + me), indexing="ij"
                ),
                dim=-1,
            ).reshape(-1, 2)
            kernels = template.kernels.spatial(cells)
            feats = compute_feature_stack(img[None], kernels)[0]  # (cells, oh, ow)
            oh, ow = feats.shape[-2:]
            y = feats.reshape(ke, me, oh * ow).permute(2, 0, 1)  # (P, ke, me)
            w = us[:, k0 : k0 + ke, m0 : m0 + me]
            if C is None:
                C = torch.zeros(
                    oh * ow, len(images), us.shape[0], k_stop, dtype=torch.complex64
                )
            C[:, d, :, k0 : k0 + ke] += torch.einsum("pkm,nkm->pnk", y, w.conj())
    corr = torch.fft.irfft(C, n=n_psi, dim=-1, norm="forward")  # (P, D, N, psi)
    flat = corr.reshape(corr.shape[0], -1)
    mip, amax = flat.max(dim=1)
    mean = flat.mean(dim=1)
    var = (flat**2).mean(dim=1) - mean**2
    best = amax // n_psi  # d_local * N + template
    psi = (n_psi - amax % n_psi) % n_psi
    unique = (flat == mip[:, None]).sum(dim=1) == 1
    shape = (oh, ow)
    return (
        mip.reshape(shape),
        mean.reshape(shape),
        var.reshape(shape),
        best.reshape(shape),
        psi.reshape(shape),
        unique.reshape(shape),
    )


# --------------------------------------------------------------------------- #
# Search type 1: a single stage
# --------------------------------------------------------------------------- #


def test_single_stage_matches_dense_reference_and_decides(template, tmp_path):
    image = _image()
    rects = [(0, 0, 4, 3), (0, 3, 2, 2)]
    mip, mean, var, best, psi, unique = _dense_reference(template, image, rects, 16)
    zscore = (mip - mean) / var.clamp_min(0).sqrt()
    ordered = zscore.flatten().sort().values
    threshold = float(0.5 * (ordered[60] + ordered[61]))  # strictly between two pixels

    sweep = Searcher(template, [Stage.create(rects, n_psi=16)], execution=EXEC)
    res = sweep.run(ImageSet.single(image))
    assert res.mip.shape == (1, 11, 11)
    torch.testing.assert_close(res.mip[0], mip, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(res.mean[0], mean, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(res.variance[0], var, rtol=1e-4, atol=1e-4)
    assert torch.equal(res.best_index[0][unique], best[unique])
    assert torch.equal(res.best_psi[0][unique], psi[unique])
    # no decision: every searched pixel stays AMBIGUOUS
    assert torch.all(res.label == int(PixelLabel.AMBIGUOUS)) and torch.all(res.last_stage == 0)

    # a plain threshold decides every pixel; the spectral bound leaves some ambiguous
    decided = Searcher(
        template, [Stage.create(rects, n_psi=16, decision=Decision(threshold))], execution=EXEC
    ).run(ImageSet.single(image))
    accepted = decided.label[0] == int(PixelLabel.ACCEPTED)
    assert torch.equal(accepted, zscore > threshold)
    assert not (decided.label == int(PixelLabel.AMBIGUOUS)).any()
    # the flat synthetic spectrum makes the spectral bound vacuous below full rank
    # (see test_full_rank_bound_decides_everything); a fixed bound exercises the rule
    assert ErrorModel.from_result(template.result).rho(tuple(rects)) > 0.43
    rho = 0.1
    bound = FixedError(rho)
    bounded = Searcher(
        template,
        [Stage.create(rects, n_psi=16, decision=Decision(threshold, bound))],
        execution=EXEC,
        keep_stage_results=True,
    ).run(ImageSet.single(image))
    labels = bounded.label[0]
    assert (labels == int(PixelLabel.AMBIGUOUS)).any()
    # the bound only widens the call: it never flips a side, it only abstains
    assert torch.all(accepted[labels == int(PixelLabel.ACCEPTED)])
    assert not accepted[labels == int(PixelLabel.REJECTED)].any()

    # the result file carries the final maps and the per-stage group
    path = tmp_path / "single.h5"
    bounded.save(path)
    loaded = SearchResult.load(path)
    torch.testing.assert_close(loaded.mip, bounded.mip, equal_nan=True)
    torch.testing.assert_close(loaded.label, bounded.label)
    assert len(loaded.stage_results) == 1
    stage = loaded.stage_results[0]
    torch.testing.assert_close(stage.maps.mip, bounded.stage_results[0].maps.mip)
    assert torch.equal(stage.labels_after, bounded.stage_results[0].labels_after)
    assert stage.stage.rectangles == tuple(rects) and stage.n_pixels == 121
    assert loaded.stages[0].decision.threshold == threshold
    assert loaded.stages[0].decision.bound.rho(tuple(rects)) == pytest.approx(rho)
    assert isinstance(loaded.stages[0].decision.bound, FixedError)


# --------------------------------------------------------------------------- #
# Search type 2: two stages
# --------------------------------------------------------------------------- #


def test_two_stage_rescored_pixels_match_fresh_searches(template):
    """Each stage equals a single-stage search of its rectangles on its pixels, and
    stage 1 searches exactly the pixels stage 0 accepted or left ambiguous."""
    image = _image(seed=3)
    rects0, rects1 = [(0, 0, 2, 2)], [(0, 0, 4, 5)]
    mip0, mean0, var0, *_ = _dense_reference(template, image, rects0, 16)
    z0 = (mip0 - mean0) / var0.clamp_min(0).sqrt()
    t0 = float(z0.flatten().quantile(0.6))
    stages = [
        Stage.create(rects0, n_psi=16, decision=Decision(t0, FixedError(0.1), absorb_radius=1)),
        Stage.create(rects1, n_psi=16, decision=Decision(1.5)),
    ]
    searcher = Searcher(template, stages, execution=EXEC, keep_stage_results=True)
    session = searcher.session(ImageSet.single(image))
    results = list(session)
    assert len(results) == 2 and results[0].n_pixels == 121
    after0 = results[0].labels_after
    survivors = results[0].pixel_index[
        (after0 == int(PixelLabel.ACCEPTED)) | (after0 == int(PixelLabel.AMBIGUOUS))
    ]
    assert 0 < survivors.numel() < 121
    assert torch.equal(torch.sort(results[1].pixel_index).values, torch.sort(survivors).values)
    assert (after0 == int(PixelLabel.ABSORBED)).any() or (after0 == int(PixelLabel.REJECTED)).any()

    for stage, rects in zip(results, (rects0, rects1), strict=True):
        mip, mean, var, best, _psi, unique = _dense_reference(template, image, rects, 16)
        rows, cols = stage.pixel_index // 11, stage.pixel_index % 11
        torch.testing.assert_close(stage.maps.mip, mip[rows, cols], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(stage.maps.mean, mean[rows, cols], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(stage.maps.variance, var[rows, cols], rtol=1e-4, atol=1e-4)
        u = unique[rows, cols]
        assert torch.equal(stage.best_index[u], best[rows, cols][u])

    final = session.snapshot()
    # rejected / absorbed pixels keep stage 0's maps; re-scored ones carry stage 1's
    flat_last = final.last_stage.reshape(-1)
    assert torch.all(flat_last[results[1].pixel_index] == 1)
    rejected = results[0].pixel_index[after0 == int(PixelLabel.REJECTED)]
    assert torch.all(flat_last[rejected] == 0)
    assert not (final.label == int(PixelLabel.AMBIGUOUS)).any()  # stage 1 decided all
    assert [g.rect for g in session.store.groups] == [(0, 0, 2, 2), (0, 0, 4, 5)]


def test_two_stage_healpix_level_and_planes(synthetic_result):
    """Stage 0 at a coarser HEALPix level on one CTF plane, stage 1 over everything."""
    result = synthetic_result(s2_nside=2, n_filters=2)  # 48 orientations x 2 filters
    template = Template(result, device="cpu")
    stub = StubFeaturizer()
    bound = ErrorModel.from_result(result)
    stages = [
        Stage.create(
            [(0, 0, 3, 3)],
            n_psi=16,
            orientations=1,
            planes=(1,),
            decision=Decision(0.0, FixedError(0.2), absorb_radius=1),
        ),
        Stage.create([(0, 0, 4, 5)], n_psi=16, decision=Decision(0.5, bound)),  # rho = 0
    ]
    searcher = Searcher(template, stages, execution=EXEC, featurize=stub, n_defocus=3)
    assert searcher.ipix is not None and searcher.ipix.tolist() == list(range(48))
    t_ids, d_ids = stages[0].hypothesis_ids(searcher.space, searcher.ipix)
    assert t_ids.numel() == 12 * 2 and d_ids.tolist() == [1]  # one child per parent pixel
    images = ImageSet.from_stack(torch.zeros(1, 30, 30), filters=torch.ones(3, 30, 16))
    session = searcher.session(images)
    results = list(session)
    # stage 1's larger rectangle is a new group, so it is produced on all three planes
    assert stub.calls == 2 and stub.planes_produced == 4
    assert [(g.rect, g.planes) for g in session.store.groups] == [
        ((0, 0, 3, 3), (1,)),
        ((0, 0, 4, 5), (0, 1, 2)),
    ]
    assert results[0].maps.n_evaluated == 24 * 16
    assert results[1].maps.n_evaluated == 3 * 96 * 16
    final = session.snapshot()
    assert not (final.label == int(PixelLabel.AMBIGUOUS)).any()  # full rank decides all
    decoded = final.decode()
    o = decoded["orientation_index"][final.last_stage == 0]
    assert set(o[o >= 0].tolist()) <= set(range(0, 48, 4))
    assert set(decoded["defocus_index"][final.last_stage == 0].tolist()) <= {1, -1}
    assert torch.isfinite(decoded["phi"][final.last_stage >= 0]).all()


def test_session_stops_when_no_pixel_is_left(template):
    stub = StubFeaturizer()
    stages = [
        Stage.create([(0, 0, 2, 2)], n_psi=16, decision=Decision(1e9)),  # rejects all
        Stage.create([(0, 0, 4, 3)], n_psi=16, decision=Decision(0.0)),
    ]
    searcher = Searcher(template, stages, execution=EXEC, featurize=stub, keep_stage_results=True)
    session = searcher.session(ImageSet.from_stack(torch.zeros(2, 30, 30)))
    results = list(session)
    assert [r.stage_index for r in results] == [0] and results[0].n_pixels == 242
    assert results[0].n_ambiguous_after == 0 and stub.calls == 1
    res = session.snapshot()
    assert len(res.stage_results) == 1 and isinstance(res.stage_results[0], StageResult)
    assert torch.all(res.label == int(PixelLabel.REJECTED)) and torch.all(res.last_stage == 0)

    # everything accepted: stage 1 re-scores every pixel
    stages = [
        Stage.create([(0, 0, 2, 2)], n_psi=16, decision=Decision(-1e9)),
        Stage.create([(0, 0, 4, 3)], n_psi=16),
    ]
    searcher = Searcher(template, stages, execution=EXEC, featurize=StubFeaturizer())
    res = searcher.run(ImageSet.from_stack(torch.zeros(2, 30, 30)))
    assert torch.all(res.last_stage == 1) and torch.all(res.label == int(PixelLabel.AMBIGUOUS))


def test_searcher_validation(template):
    with pytest.raises(ValueError, match="fp16"):
        Searcher(template, [Stage.create([(0, 0, 2, 2)], precision="fp16", n_psi=16)])
    with pytest.raises(ValueError, match="exceeds"):
        Searcher(template, [Stage.create([(0, 0, 8, 2)], n_psi=32)])
    with pytest.raises(ValueError, match="defocus"):
        Searcher(template, [Stage.create([(0, 0, 2, 2)], n_psi=16)], n_defocus=2).session(
            ImageSet.single(_image())
        )
    with pytest.raises(ValueError, match="planes"):
        Searcher(template, [Stage.create([(0, 0, 2, 2)], n_psi=16, planes=(3,))], n_defocus=2)
    with pytest.raises(ValueError, match="HEALPix"):
        Searcher(template, [Stage.create([(0, 0, 2, 2)], n_psi=16, orientations=1)])
    with pytest.raises(ValueError):
        Searcher(template, [])


def test_full_rank_bound_decides_everything(template):
    bound = ErrorModel.from_result(template.result)
    assert bound.rho(((0, 0, 4, 5),)) == pytest.approx(0.0, abs=1e-12)
    searcher = Searcher(
        template,
        [Stage.create([(0, 0, 4, 5)], n_psi=16, decision=Decision(3.0, bound))],
        execution=EXEC,
        featurize=StubFeaturizer(),
    )
    res = searcher.run(ImageSet.single(torch.zeros(30, 30)))
    assert not (res.label == int(PixelLabel.AMBIGUOUS)).any()  # zero error: all decided


def test_center_mode_orientation_subset_and_run_many(template):
    stub = StubFeaturizer()
    coarse = torch.arange(0, template.space.n_orientations, 2)
    stages = [Stage.create([(0, 0, 3, 3)], n_psi=16, orientations=coarse)]
    searcher = Searcher(
        template,
        stages,
        execution=EXEC,
        featurize=stub,
        n_defocus=2,
        defocus_values=torch.tensor([-500.0, 500.0]),
    )
    boxes = ImageSet.from_boxes(torch.zeros(7, 20, 20), filters=torch.ones(2, 20, 11))
    results = list(searcher.run_many([boxes, boxes]))
    assert len(results) == 2
    res = results[0]
    assert res.mip.shape == (7, 1) and res.geometry.mode == "center"
    dec = res.decode()
    assert set(dec["orientation_index"].unique().tolist()) <= {0, 2, 4}
    assert dec["defocus"] is not None and set(dec["defocus"].unique().tolist()) <= {
        -500.0,
        500.0,
    }
    assert dec["phi"] is not None and torch.isfinite(dec["phi"]).all()
    torch.testing.assert_close(results[0].mip, results[1].mip)
    assert len(searcher.executor.loops) == 1  # the loop was reused across image sets


def test_search_result_round_trip_and_masked_pixels(template, tmp_path):
    stub = StubFeaturizer()
    mask = torch.zeros(11, 11, dtype=torch.bool)
    mask[2:9, 2:9] = True
    probe = Searcher(
        template, [Stage.create([(0, 0, 2, 2)], n_psi=16)], execution=EXEC, featurize=stub
    ).run(ImageSet.single(torch.zeros(30, 30), mask=mask))
    z = probe.zscore[0][mask]
    ordered = z.sort().values
    threshold = float(0.5 * (ordered[-11] + ordered[-10]))  # exactly 10 pixels above
    stages = [
        Stage.create([(0, 0, 2, 2)], n_psi=16, decision=Decision(threshold)),
        Stage.create([(0, 0, 4, 5)], n_psi=32),
    ]
    searcher = Searcher(
        template, stages, execution=EXEC, featurize=StubFeaturizer(), keep_stage_results=True
    )
    res = searcher.run(ImageSet.single(torch.zeros(30, 30), mask=mask))
    assert (res.label[0][~mask] == int(PixelLabel.MASKED)).all()
    assert torch.isnan(res.mip[0][~mask]).all() and (res.last_stage[0][~mask] == -1).all()
    n_rescored = int((z > threshold).sum())
    assert n_rescored == 10
    assert (res.last_stage == 1).sum() == n_rescored
    assert (res.last_stage == 0).sum() == 49 - n_rescored
    psi = res.psi_degrees()
    assert torch.isnan(psi[0][~mask]).all()
    assert psi[0][res.last_stage[0] == 1].max() < 360 and res.n_psi_of_stage == (16, 32)

    path = tmp_path / "result.h5"
    res.save(path)
    loaded = SearchResult.load(path)
    for name in ("mip", "zscore", "mean", "variance", "best_index", "best_psi", "label", "last_stage"):
        torch.testing.assert_close(getattr(loaded, name), getattr(res, name), equal_nan=True)
    assert loaded.geometry == res.geometry and loaded.n_psi_of_stage == res.n_psi_of_stage
    assert [s.rectangles for s in loaded.stages] == [s.rectangles for s in res.stages]
    assert loaded.stages[0].decision.threshold == threshold and loaded.stages[1].decision is None
    torch.testing.assert_close(loaded.decode()["psi"], res.decode()["psi"], equal_nan=True)
    torch.testing.assert_close(loaded.decode()["phi"], res.decode()["phi"], equal_nan=True)
    assert [r.stage_index for r in loaded.stage_results] == [0, 1]
    for got, want in zip(loaded.stage_results, res.stage_results, strict=True):
        assert torch.equal(got.pixel_index, want.pixel_index)
        torch.testing.assert_close(got.maps.zscore, want.maps.zscore)
        assert torch.equal(got.best_index, want.best_index)
        assert got.n_ambiguous_after == want.n_ambiguous_after and got.elapsed_s == want.elapsed_s
    arrays = res.to_numpy()
    assert arrays["mip"].shape == (1, 11, 11)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res.to("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.filterwarnings("ignore:ComplexHalf support is experimental")
def test_cuda_mixed_precision_two_stage_matches_cpu(synthetic_result):
    result = synthetic_result(n_orientations=8)
    image = _image(h=32, w=30, seed=5)

    def stages(precision0):
        return [
            Stage.create([(0, 0, 2, 3)], precision=precision0, n_psi=128, decision=Decision(0.0)),
            Stage.create([(0, 0, 4, 5)], n_psi=128, decision=Decision(0.0)),
        ]

    cpu = Searcher(
        Template(result, device="cpu"),
        stages("fp32"),
        execution=ExecutionConfig(batch=BatchShape(8, 4), cell_chunk=4),
    ).run(ImageSet.single(image))
    gpu_searcher = Searcher(
        Template(result, device="cuda"),
        stages("fp16"),
        execution=ExecutionConfig(batch=BatchShape(8, 4), cell_chunk=4, use_cuda_graph=True),
        keep_stage_results=True,
    )
    gpu = gpu_searcher.run(ImageSet.single(image)).to("cpu")
    assert any(loop.graph is not None for loop in gpu_searcher.executor.loops.values())
    # stage 0 ran in fp16 on the GPU: the re-scored set may differ at near-ties, so
    # compare the final maps only where both searched the pixel in the same stage
    same = (gpu.last_stage == cpu.last_stage) & (cpu.last_stage >= 0)
    assert same.float().mean() > 0.9
    torch.testing.assert_close(gpu.mip[same], cpu.mip[same], rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(gpu.mean[same], cpu.mean[same], rtol=2e-2, atol=5e-2)
    final = (gpu.last_stage == 1) & (cpu.last_stage == 1)
    torch.testing.assert_close(gpu.mip[final], cpu.mip[final], rtol=1e-4, atol=1e-4)
    assert torch.equal(gpu.best_index[final], cpu.best_index[final])
    assert gpu.geometry == cpu.geometry and gpu.n_psi_of_stage == (128, 128)


def test_stage_planes_are_featurized_only_where_needed(template):
    stub = StubFeaturizer()
    rect = (0, 0, 3, 3)
    stages = [
        Stage.create([rect], n_psi=16, planes=(1,), decision=Decision(0.0)),
        Stage.create([rect], n_psi=16),
    ]
    searcher = Searcher(
        template, stages, execution=EXEC, featurize=stub, n_defocus=3, keep_stage_results=True
    )
    images = ImageSet.from_stack(torch.zeros(1, 30, 30), filters=torch.ones(3, 30, 16))
    session = searcher.session(images)
    results = list(session)
    assert stub.calls == 2 and stub.planes_produced == 3  # plane 1, then planes 0 and 2 only
    assert [g.planes for g in session.store.groups] == [(1,), (0, 2)]
    accepted = int((results[0].labels_after == int(PixelLabel.ACCEPTED)).sum())
    assert results[0].n_pixels == 121 and results[1].n_pixels == accepted > 0

    # stage 1 (all planes, survivors only) equals a fresh all-plane search there
    mask = torch.zeros(1, 11, 11, dtype=torch.bool)
    mask.reshape(-1)[results[1].pixel_index] = True
    fresh = Searcher(
        template, [Stage.create([rect], n_psi=16)], execution=EXEC, featurize=StubFeaturizer(),
        n_defocus=3,
    ).run(ImageSet.from_stack(torch.zeros(1, 30, 30), filters=torch.ones(3, 30, 16), mask=mask))
    order = torch.argsort(results[1].pixel_index)
    torch.testing.assert_close(
        results[1].maps.mip[order], fresh.mip.reshape(-1)[results[1].pixel_index[order]]
    )
    assert torch.equal(
        results[1].best_index[order], fresh.best_index.reshape(-1)[results[1].pixel_index[order]]
    )


def test_session_store_is_released_after_the_session(template):
    """The searcher's cached loops must not retain a finished session's blocks."""
    import gc
    import weakref

    searcher = Searcher(
        template, [Stage.create([(0, 0, 4, 5)], n_psi=16)], execution=EXEC, featurize=StubFeaturizer()
    )
    session = searcher.session(ImageSet.single(torch.zeros(30, 30)))
    session.run_all()
    store_ref = weakref.ref(session.store)
    del session
    gc.collect()
    assert store_ref() is None  # only the session owned the store
    assert searcher.executor.loops  # the loop cache survives for the next image set
