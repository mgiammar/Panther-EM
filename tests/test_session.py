"""Searcher / SearchSession end to end against a dense reference."""

from __future__ import annotations

import warnings

import pytest
import torch

from panther_em.inference.correlation import compute_feature_stack
from panther_em.inference.search.basis import Template
from panther_em.inference.search.decisions import (
    NoDecision,
    ThresholdClassifier,
    TopFractionSelector,
    ZScoreIntervalClassifier,
)
from panther_em.inference.search.featurize import ImageBatch, TileSchedule
from panther_em.inference.search.hypotheses import HypothesisSubset
from panther_em.inference.search.ledger import PixelLabel
from panther_em.inference.search.results import SearchResult, StageResult
from panther_em.inference.search.schedule import (
    ErrorModel,
    SearchSchedule,
    StageSpec,
    nested_schedule,
)
from panther_em.inference.search.searcher import ExecutionConfig, Searcher
from panther_em.inference.search.types import BatchShape, ImageSet

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array_wrap__ must accept.*:DeprecationWarning",
    "ignore:.*__array__ implementation doesn't accept.*:DeprecationWarning",
)

EXEC = ExecutionConfig(batch=BatchShape(8, 5), cell_chunk=4, use_cuda_graph=False)


class StubFeaturizer:
    """Deterministic random features per cell; mirrors the legacy test stub."""

    def __init__(self, seed: int = 1) -> None:
        self.seed = seed
        self.calls = 0
        self.planes_produced = 0

    def schedule(self, geometry, pixels) -> TileSchedule:
        idx = pixels.index.to("cpu")
        return TileSchedule(idx, (ImageBatch(0, idx, slice(0, idx.numel())),), None)

    def produce(
        self,
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


def _dense_reference(template, image, rects, n_psi):
    """Per-pixel statistics from the dense correlogram (kernels, FFT, einsum, irfft)."""
    recon = template.reconstructor
    us = torch.as_tensor(recon.result.U) * torch.as_tensor(recon.result.S)[None, None]
    us = us.reshape(-1, *us.shape[2:]).to(torch.complex64)  # (N, K, E)
    k_stop = max(k0 + ke for k0, _, ke, _ in rects)
    C = None
    for k0, m0, ke, me in rects:
        cells = torch.stack(
            torch.meshgrid(
                torch.arange(k0, k0 + ke), torch.arange(m0, m0 + me), indexing="ij"
            ),
            dim=-1,
        ).reshape(-1, 2)
        kernels = template.kernels.spatial(cells)
        feats = compute_feature_stack(image[None], kernels)[0]  # (cells, oh, ow)
        oh, ow = feats.shape[-2:]
        y = feats.reshape(ke, me, oh * ow).permute(2, 0, 1)  # (P, ke, me)
        w = us[:, k0 : k0 + ke, m0 : m0 + me]
        if C is None:
            C = torch.zeros(oh * ow, us.shape[0], k_stop, dtype=torch.complex64)
        C[:, :, k0 : k0 + ke] += torch.einsum("pkm,nkm->pnk", y, w.conj())
    corr = torch.fft.irfft(C, n=n_psi, dim=-1, norm="forward")
    flat = corr.reshape(corr.shape[0], -1)
    mip, amax = flat.max(dim=1)
    mean = flat.mean(dim=1)
    var = (flat**2).mean(dim=1) - mean**2
    best = amax // n_psi
    psi = (n_psi - amax % n_psi) % n_psi
    unique = (flat == mip[:, None]).sum(dim=1) == 1
    return (
        mip.reshape(oh, ow),
        mean.reshape(oh, ow),
        var.reshape(oh, ow),
        best.reshape(oh, ow),
        psi.reshape(oh, ow),
        unique.reshape(oh, ow),
    )


def test_single_stage_matches_dense_reference(template):
    image = _image()
    rects = [(0, 0, 4, 3), (0, 3, 2, 2)]
    mip, mean, var, best, psi, unique = _dense_reference(template, image, rects, 16)
    searcher = Searcher(
        template, SearchSchedule.single(StageSpec(rects, n_psi=16)), execution=EXEC
    )
    res = searcher.run(ImageSet.single(image))
    assert res.mip.shape == (1, 11, 11)
    torch.testing.assert_close(res.mip[0], mip, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(res.mean[0], mean, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(res.variance[0], var, rtol=1e-4, atol=1e-4)
    assert torch.equal(res.best_index[0][unique], best[unique])
    assert torch.equal(res.best_psi[0][unique], psi[unique])
    assert torch.all(res.label == int(PixelLabel.UNCERTAIN)) and torch.all(
        res.last_stage == 0
    )


def test_multi_stage_reuses_features_and_matches_single_stage_runs(template):
    """Each stage of a nested schedule equals a single-stage search of its rects."""
    image = _image(seed=3)
    stage_rects = [[(0, 0, 2, 2)], [(0, 0, 2, 2), (0, 2, 4, 3)], [(0, 0, 4, 5)]]
    schedule = SearchSchedule(
        tuple(StageSpec(r, n_psi=16) for r in stage_rects),
        (
            TopFractionSelector(fraction=0.5),
            TopFractionSelector(fraction=0.5),
            ThresholdClassifier(1.5),
        ),
        keep_stage_results=True,
    )
    searcher = Searcher(template, schedule, execution=EXEC)
    results = list(searcher.session(ImageSet.single(image)))
    assert len(results) == 3
    n_pixels = [r.n_pixels for r in results]
    assert n_pixels[0] == 121 and n_pixels[0] > n_pixels[1] > n_pixels[2] > 0
    for stage in results:
        mip, mean, var, best, _psi, unique = _dense_reference(
            template, image, list(stage.spec.rectangles), 16
        )
        rows, cols = stage.pixel_index // 11, stage.pixel_index % 11
        torch.testing.assert_close(
            stage.maps.mip, mip[rows, cols], rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            stage.maps.mean, mean[rows, cols], rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            stage.maps.variance, var[rows, cols], rtol=1e-4, atol=1e-4
        )
        u = unique[rows, cols]
        assert torch.equal(stage.best_index[u], best[rows, cols][u])
    # stage 0's (0,0,2,2) group was carried into stage 1 rather than re-featurized
    store = searcher.session(ImageSet.single(image))
    list(store)
    assert [g.rect for g in store.store.groups] == [
        (0, 0, 2, 2),
        (0, 2, 4, 3),
        (0, 0, 4, 5),
    ]
    final = searcher.run(ImageSet.single(image))
    assert set(torch.unique(final.label).tolist()) <= {int(lab) for lab in PixelLabel}
    assert (final.last_stage[0] >= 0).sum() == 121


def test_session_narrows_stops_and_keeps_stage_results(template):
    stub = StubFeaturizer()
    schedule = nested_schedule(
        [(2, 2), (4, 3), (4, 5)],
        n_psi=16,
        decision=[
            TopFractionSelector(fraction=0.2),
            ThresholdClassifier(1e9),
            NoDecision(),
        ],
        keep_stage_results=True,
    )
    searcher = Searcher(template, schedule, execution=EXEC, featurizer=stub)
    session = searcher.session(ImageSet.from_stack(torch.zeros(2, 30, 30)))
    stages = list(session)
    assert [s.stage_index for s in stages] == [
        0,
        1,
    ]  # ThresholdClassifier decides everything
    assert stages[0].n_pixels == 242 and stages[1].n_pixels == round(0.2 * 242)
    assert stages[1].n_uncertain_after == 0
    res = session.snapshot()
    assert len(res.stages) == 2 and isinstance(res.stages[0], StageResult)
    assert res.label.max() == int(PixelLabel.REJECTED)
    assert (res.last_stage == 1).sum() == stages[1].n_pixels
    assert stub.calls == 2  # one group per executed stage; stage 2 never ran
    assert set(torch.unique(res.last_stage).tolist()) == {0, 1}


def test_fp32_after_fp16_needs_cuda_and_validation(template):
    with pytest.raises(ValueError, match="fp16"):
        Searcher(
            template,
            SearchSchedule.single(
                StageSpec([(0, 0, 2, 2)], precision="fp16", n_psi=16)
            ),
        )
    with pytest.raises(ValueError, match="exceeds"):
        Searcher(template, SearchSchedule.single(StageSpec([(0, 0, 8, 2)], n_psi=32)))
    with pytest.raises(ValueError, match="defocus"):
        Searcher(
            template,
            SearchSchedule.single(StageSpec([(0, 0, 2, 2)], n_psi=16)),
            n_defocus=2,
        ).session(ImageSet.single(_image()))


def test_error_model_fills_missing_errors_and_decision_uses_them(template):
    schedule = nested_schedule(
        [(4, 5)], n_psi=16, decision=ZScoreIntervalClassifier(3.0)
    )
    assert schedule.requires_errors()
    searcher = Searcher(template, schedule, execution=EXEC, featurizer=StubFeaturizer())
    assert searcher.schedule.stages[0].error == pytest.approx(
        0.0, abs=1e-12
    )  # full rank
    res = searcher.run(ImageSet.single(torch.zeros(30, 30)))
    assert not (res.label == int(PixelLabel.UNCERTAIN)).any()  # zero error: all decided
    assert searcher.schedule.stages[0].error == ErrorModel.from_result(
        template.result
    ).rho((0, 0, 4, 5))


def test_center_mode_defocus_subset_and_run_many(template):
    stub = StubFeaturizer()
    scout = HypothesisSubset.strided(template.space, 2)
    schedule = nested_schedule([(3, 3)], n_psi=16, scout=scout)
    searcher = Searcher(
        template,
        schedule,
        execution=EXEC,
        featurizer=stub,
        n_defocus=2,
        defocus_values=torch.tensor([-500.0, 500.0]),
    )
    boxes = ImageSet.from_boxes(torch.zeros(7, 20, 20), filters=torch.ones(2, 20, 11))
    results = list(searcher.run_many([boxes, boxes]))
    assert len(results) == 2
    res = results[0].per_particle()
    assert res.mip.shape == (7,) and res.geometry.mode == "center"
    dec = results[0].decode()
    assert set(dec.orientation_index.unique().tolist()) <= {0, 2, 4}
    assert dec.defocus is not None and set(dec.defocus.unique().tolist()) <= {
        -500.0,
        500.0,
    }
    assert dec.phi is not None and torch.isfinite(dec.phi).all()
    torch.testing.assert_close(results[0].mip, results[1].mip)
    assert (
        len(searcher.ctx.loops) == 1
    )  # the captured/eager loop was reused across images


def test_search_result_round_trip_and_masked_pixels(template, tmp_path):
    stub = StubFeaturizer()
    schedule = nested_schedule(
        [(2, 2), (4, 5)],
        n_psi=[16, 32],
        decision=[TopFractionSelector(top_k=10), NoDecision()],
    )
    mask = torch.zeros(11, 11, dtype=torch.bool)
    mask[2:9, 2:9] = True
    searcher = Searcher(template, schedule, execution=EXEC, featurizer=stub)
    res = searcher.run(ImageSet.single(torch.zeros(30, 30), mask=mask))
    assert (res.label[0][~mask] == int(PixelLabel.MASKED)).all()
    assert (
        torch.isnan(res.mip[0][~mask]).all() and (res.last_stage[0][~mask] == -1).all()
    )
    assert (res.last_stage == 1).sum() == 10 and (res.last_stage == 0).sum() == 49 - 10
    psi = res.psi_degrees()
    assert torch.isnan(psi[0][~mask]).all()
    assert psi[0][res.last_stage[0] == 1].max() < 360 and res.n_psi_of_stage == (16, 32)

    path = tmp_path / "result.h5"
    res.save(path)
    loaded = SearchResult.load(path)
    for name in (
        "mip",
        "zscore",
        "mean",
        "variance",
        "best_index",
        "best_psi",
        "label",
        "last_stage",
    ):
        torch.testing.assert_close(
            getattr(loaded, name), getattr(res, name), equal_nan=True
        )
    assert (
        loaded.geometry == res.geometry and loaded.n_psi_of_stage == res.n_psi_of_stage
    )
    assert [s.rectangles for s in loaded.schedule.stages] == [
        s.rectangles for s in res.schedule.stages
    ]
    torch.testing.assert_close(loaded.decode().psi, res.decode().psi, equal_nan=True)
    torch.testing.assert_close(loaded.decode().phi, res.decode().phi, equal_nan=True)
    arrays = res.to_numpy()
    assert arrays["mip"].shape == (1, 11, 11)
    with pytest.raises(ValueError):
        res.per_particle()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res.to("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.filterwarnings("ignore:ComplexHalf support is experimental")
def test_cuda_mixed_precision_session_matches_cpu(synthetic_result):
    result = synthetic_result(n_orientations=8)
    image = _image(h=32, w=30, seed=5)
    schedule = nested_schedule(
        [(2, 3), (4, 5)],
        precision=["fp16", "fp32"],
        n_psi=128,
        decision=[TopFractionSelector(fraction=0.5), ThresholdClassifier(0.0)],
        keep_stage_results=True,
    )
    cpu = Searcher(
        Template(result, device="cpu"),
        nested_schedule(
            [(2, 3), (4, 5)],
            n_psi=128,
            decision=[TopFractionSelector(fraction=0.5), ThresholdClassifier(0.0)],
        ),
        execution=ExecutionConfig(batch=BatchShape(8, 4), cell_chunk=4),
    ).run(ImageSet.single(image))
    gpu_searcher = Searcher(
        Template(result, device="cuda"),
        schedule,
        execution=ExecutionConfig(
            batch=BatchShape(8, 4), cell_chunk=4, use_cuda_graph=True
        ),
    )
    gpu = gpu_searcher.run(ImageSet.single(image)).to("cpu")
    assert any(
        type(loop).__name__ == "GraphedHypLoop"
        for loop in gpu_searcher.ctx.loops.values()
    )
    # stage 0 ran in fp16 on the GPU: the follow-up set may differ at near-ties, so
    # compare the final maps only where both searched the pixel in the same stage
    same = (gpu.last_stage == cpu.last_stage) & cpu.searched()
    assert same.float().mean() > 0.9
    torch.testing.assert_close(gpu.mip[same], cpu.mip[same], rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(gpu.mean[same], cpu.mean[same], rtol=2e-2, atol=5e-2)
    final = (gpu.last_stage == 1) & (cpu.last_stage == 1)
    torch.testing.assert_close(gpu.mip[final], cpu.mip[final], rtol=1e-4, atol=1e-4)
    assert torch.equal(gpu.best_index[final], cpu.best_index[final])
    assert gpu.geometry == cpu.geometry and gpu.n_psi_of_stage == (128, 128)


def test_stage_planes_are_featurized_only_where_needed(template):
    stub = StubFeaturizer()
    space_planes = 3
    rect = (0, 0, 3, 3)
    schedule = SearchSchedule(
        (
            StageSpec(
                [rect],
                n_psi=16,
                hypotheses=HypothesisSubset(defocus_ids=torch.tensor([1])),
            ),
            StageSpec([rect], n_psi=16),
        ),
        (TopFractionSelector(fraction=0.5), NoDecision()),
        keep_stage_results=True,
    )
    searcher = Searcher(
        template, schedule, execution=EXEC, featurizer=stub, n_defocus=space_planes
    )
    images = ImageSet.from_stack(torch.zeros(1, 30, 30), filters=torch.ones(3, 30, 16))
    session = searcher.session(images)
    stages = list(session)
    assert (
        stub.calls == 2 and stub.planes_produced == 3
    )  # plane 1, then planes 0 and 2 only
    assert [g.planes for g in session.store.groups] == [(1,), (0, 2)]
    assert stages[0].n_pixels == 121 and stages[1].n_pixels == round(0.5 * 121)

    # stage 1 (all planes, survivors only) equals a fresh all-plane search there
    mask = torch.zeros(1, 11, 11, dtype=torch.bool)
    mask.reshape(-1)[stages[1].pixel_index] = True
    fresh = Searcher(
        template,
        SearchSchedule.single(StageSpec([rect], n_psi=16)),
        execution=EXEC,
        featurizer=StubFeaturizer(),
        n_defocus=space_planes,
    ).run(
        ImageSet.from_stack(
            torch.zeros(1, 30, 30), filters=torch.ones(3, 30, 16), mask=mask
        )
    )
    order = torch.argsort(stages[1].pixel_index)
    torch.testing.assert_close(
        stages[1].maps.mip[order], fresh.mip.reshape(-1)[stages[1].pixel_index[order]]
    )
    assert torch.equal(
        stages[1].best_index[order],
        fresh.best_index.reshape(-1)[stages[1].pixel_index[order]],
    )


def test_session_surpass_threshold_decision_round_trip(template):
    from panther_em.inference.search.decisions import SurpassThresholdDecision

    stub = StubFeaturizer()
    schedule = SearchSchedule.single(
        StageSpec([(0, 0, 4, 5)], n_psi=16), SurpassThresholdDecision(1.0)
    )
    searcher = Searcher(template, schedule, execution=EXEC, featurizer=stub)
    images = ImageSet.single(
        torch.zeros(30, 30),
        pixel_maps={
            "image_norm": torch.ones(11, 11),
            "noise_variance": torch.full((1, 11, 11), 2.0),
        },
    )
    session = searcher.session(images)
    assert session.ledger.aux["noise_variance"].shape == (121,)
    res = session.run_all()
    assert not (res.label == int(PixelLabel.UNCERTAIN)).any()
    z = res.zscore[0]
    assert torch.equal(res.label[0] == int(PixelLabel.ACCEPTED), z > 1.0)

    with pytest.raises(KeyError, match="image_norm"):
        searcher.run(ImageSet.single(torch.zeros(30, 30)))


# --------------------------------------------------------------------------- #
# Pair emission (Stage D)
# --------------------------------------------------------------------------- #


def _planted(result, orientation, *, seed=3, noise=0.02, at=(4, 6)):
    """A 30x30 noise image with the reconstructed projection planted at grid ``at``."""
    template = Template(result, device="cpu")
    proj = template.reconstructor.reconstruct_projection(
        orientation_idx=orientation, return_torch=True
    )
    proj = torch.as_tensor(proj)
    proj = (proj.real if proj.is_complex() else proj).to(torch.float32)
    proj = proj / proj.norm()
    image = noise * torch.randn(30, 30, generator=torch.Generator().manual_seed(seed))
    r, c = at
    image[r : r + 20, c : c + 20] += proj
    return template, image, r * 11 + c


def _pair_schedule(threshold=2.5, n_psi=16, **kw):
    """One emitting stage; 2.5 sigma admits the planted pair (z ~ 3.3 at rank 20)."""
    from panther_em.inference.search.pairs import PairEmissionSpec

    spec = PairEmissionSpec(threshold, **kw)
    return SearchSchedule.single(StageSpec([(0, 0, 4, 5)], n_psi=n_psi, pairs=spec))


def test_session_emits_dilated_pairs_around_a_planted_template(synthetic_result):
    pytest.importorskip("healpy")
    from panther_em.inference.search.decisions import PairEmissionDecision
    from panther_em.inference.search.orientations import S2Neighborhood

    result = synthetic_result(n_orientations=12, s2_nside=1)
    true_o = 7
    template, image, true_pixel = _planted(result, true_o)
    schedule = _pair_schedule(pair_dtype=torch.float32)
    assert isinstance(schedule.decisions[0], PairEmissionDecision)
    searcher = Searcher(template, schedule, execution=EXEC)
    assert searcher.s2 is not None and searcher.s2.grid.nside_finest == 1
    maps = {"image_norm": torch.ones(11, 11), "noise_variance": torch.ones(11, 11)}
    session = searcher.session(ImageSet.single(image, pixel_maps=maps))
    (stage,) = list(session)
    pairs = stage.pairs
    assert pairs is not None and len(pairs) > 0 and pairs.n_templates == 12
    assert session.ledger.pairs[0] is pairs
    keys = set(pairs.key().tolist())
    # the planted pair is emitted ...
    assert (true_pixel * 12 + true_o) in keys
    # ... with its 3x3 pixel neighbourhood x ring-1 orientation neighbourhood
    s2 = S2Neighborhood.from_space(searcher.space)
    ring = s2.neighbor_table(1)[true_o]
    ring = ring[ring >= 0].tolist()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            p = (4 + dr) * 11 + (6 + dc)
            for o in ring:
                assert (p * 12 + o) in keys, (dr, dc, o)
    # labels: UNCERTAIN exactly on the pairs' pixels, REJECTED elsewhere
    label = session.ledger.label
    uncertain = (label == int(PixelLabel.UNCERTAIN)).nonzero().squeeze(-1)
    assert torch.equal(uncertain, pairs.pixels())
    assert torch.all(
        label[label != int(PixelLabel.UNCERTAIN)] == int(PixelLabel.REJECTED)
    )
    assert stage.n_uncertain_after == int(pairs.pixels().numel())
    # the dense statistics are the ordinary ones (emission does not alter the maps)
    mip, _, _, best, _, _ = _dense_reference(template, image, [(0, 0, 4, 5)], 16)
    torch.testing.assert_close(
        stage.maps.mip.reshape(11, 11), mip, rtol=1e-4, atol=1e-4
    )
    assert int(best.reshape(-1)[true_pixel]) == true_o
    # every emitted seed pair's score is the stage's per-pair max: never above the mip
    assert torch.all(pairs.score <= stage.maps.mip[pairs.pixel].max() + 1e-6)

    # the result carries the pairs and round-trips them through HDF5
    res = session.snapshot()
    assert torch.equal(res.pairs[0].key(), pairs.key())
    assert res.schedule.stages[0].pairs is not None


def test_session_pairs_round_trip_and_requirements(synthetic_result, tmp_path):
    pytest.importorskip("healpy")
    from panther_em.inference.search.decisions import PairEmissionDecision
    from panther_em.inference.search.pairs import PairEmissionSpec, PairList

    result = synthetic_result(n_orientations=12, s2_nside=1)
    template, image, _ = _planted(result, 2)
    schedule = _pair_schedule(pair_dtype=torch.float32)
    searcher = Searcher(template, schedule, execution=EXEC)
    with pytest.raises(ValueError, match="image_norm"):
        searcher.run(ImageSet.single(image))
    maps = {"image_norm": torch.ones(11, 11), "noise_variance": torch.ones(11, 11)}
    res = searcher.run(ImageSet.single(image, pixel_maps=maps))
    path = tmp_path / "pairs_result.h5"
    res.save(path)
    loaded = SearchResult.load(path)
    assert torch.equal(loaded.pairs[0].key(), res.pairs[0].key())
    torch.testing.assert_close(loaded.pairs[0].score, res.pairs[0].score)
    assert loaded.schedule.stages[0].pairs == schedule.stages[0].pairs

    # an explicit decision on an emitting stage is kept as given
    kept = SearchSchedule((schedule.stages[0],), (ThresholdClassifier(1.0),))
    assert isinstance(kept.decisions[0], ThresholdClassifier)
    # a PairEmissionDecision without recorded pairs fails loudly ...
    bare = SearchSchedule(
        (StageSpec([(0, 0, 4, 5)], n_psi=16),), (PairEmissionDecision(),)
    )
    with pytest.raises(RuntimeError, match="emitted no pairs"):
        Searcher(template, bare, execution=EXEC).run(ImageSet.single(image))
    # ... unless a list is injected
    session = Searcher(template, bare, execution=EXEC).session(ImageSet.single(image))
    session.set_pairs(
        0, PairList(torch.tensor([5]), torch.tensor([1]), 12, torch.ones(1))
    )
    (stage,) = list(session)
    assert stage.pairs is None and session.ledger.uncertain().index.tolist() == [5]

    # a template whose angles are not HEALPix pixel centers: orientation rings are
    # rejected up front, pixel-only dilation still works
    plain = Template(synthetic_result(n_orientations=6), device="cpu")
    with pytest.raises(ValueError, match="orientation_rings"):
        Searcher(plain, _pair_schedule(2.0), execution=EXEC)
    pixel_only = SearchSchedule.single(
        StageSpec(
            [(0, 0, 4, 5)], n_psi=16, pairs=PairEmissionSpec(2.0, orientation_rings=0)
        )
    )
    plain_searcher = Searcher(plain, pixel_only, execution=EXEC)
    assert plain_searcher.s2 is None
    plain_res = plain_searcher.run(ImageSet.single(image, pixel_maps=maps))
    assert isinstance(plain_res.pairs[0], PairList)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.filterwarnings("ignore:ComplexHalf support is experimental")
def test_cuda_pair_emission_matches_cpu(synthetic_result):
    pytest.importorskip("healpy")
    result = synthetic_result(n_orientations=12, s2_nside=1)
    _, image, _ = _planted(result, 9)
    maps = {"image_norm": torch.ones(11, 11), "noise_variance": torch.ones(11, 11)}
    images = ImageSet.single(image, pixel_maps=maps)
    schedule = _pair_schedule(n_psi=128, pair_dtype=torch.float32)
    cpu = Searcher(Template(result, device="cpu"), schedule).run(images)
    ref = cpu.pairs[0]
    assert len(ref) > 0
    for use_graph in (False, True):
        gpu = Searcher(
            Template(result, device="cuda"),
            schedule,
            execution=ExecutionConfig(use_cuda_graph=use_graph),
        ).run(images)
        got = gpu.pairs[0].to("cpu")
        assert set(got.key().tolist()) == set(ref.key().tolist())
        assert gpu.label.device.type == "cuda"


def test_session_store_is_released_after_the_session(template):
    """The searcher's cached loops must not retain a finished session's blocks."""
    import gc
    import weakref

    stub = StubFeaturizer()
    searcher = Searcher(
        template,
        SearchSchedule.single(StageSpec([(0, 0, 4, 5)], n_psi=16)),
        execution=EXEC,
        featurizer=stub,
    )
    session = searcher.session(ImageSet.single(torch.zeros(30, 30)))
    session.run_all()
    store_ref = weakref.ref(session.store)
    del session
    gc.collect()
    assert store_ref() is None  # only the session owned the store
    assert searcher.ctx.loops  # the loop cache itself survives for the next image set
