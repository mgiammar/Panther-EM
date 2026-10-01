"""Searcher / SearchSession end to end, incl. the A/B gate against the legacy API."""

from __future__ import annotations

import warnings

import pytest
import torch

from panther_em.inference.search import compressed_search, incremental_search
from panther_em.inference.search.basis import Template
from panther_em.inference.search.decisions import (
    DecisionChain,
    DilationAbsorb,
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
from panther_em.inference.search.tracking import MultiPrecisionPixelTracker
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
        n = schedule.n_columns
        k0, m0, ke, me = group.rect
        for c, (k, m) in enumerate((k0 + i // me, m0 + i % me) for i in range(ke * me)):
            gen = torch.Generator().manual_seed(self.seed * 100003 + k * 1009 + m)
            full = torch.complex(
                torch.randn(group.n_defocus, 10_000, generator=gen),
                torch.randn(group.n_defocus, 10_000, generator=gen),
            )
            data = full[:, schedule.column_index]  # pixel-identity features
            writer.write(slice(c, c + 1), (0, n), data[:, None, :])
        writer.close()


@pytest.fixture
def template(synthetic_result):
    return Template(synthetic_result(), device="cpu")


def _image(h=30, w=30, seed=0):
    return torch.randn(h, w, generator=torch.Generator().manual_seed(seed))


def test_single_stage_matches_legacy_compressed_search(template):
    image = _image()
    rects = [(0, 0, 4, 3), (0, 3, 2, 2)]
    legacy = compressed_search(
        image,
        template.reconstructor,
        rects,
        pixel_batch=8,
        hyp_batch=5,
        n_psi=16,
        feature_chunk=4,
        show_progress=False,
        use_fused_kernel=False,
    )
    searcher = Searcher(
        template, SearchSchedule.single(StageSpec(rects, n_psi=16)), execution=EXEC
    )
    res = searcher.run(ImageSet.single(image))
    assert res.mip.shape == (1, 11, 11)
    for key in ("mip", "mean", "variance", "zscore"):
        torch.testing.assert_close(
            getattr(res, key)[0], legacy[key], rtol=1e-4, atol=1e-4
        )
    unique = (res.zscore[0] - legacy["zscore"]).abs() < 1e-3
    assert torch.equal(res.best_index[0][unique], legacy["best_index"][unique])
    assert torch.equal(res.best_psi[0][unique], legacy["best_psi"][unique])
    assert torch.all(res.label == int(PixelLabel.UNCERTAIN)) and torch.all(
        res.last_stage == 0
    )


def test_multi_stage_matches_legacy_incremental_search_with_tracker(template):
    image = _image(seed=3)
    stage_rects = [[(0, 0, 2, 2)], [(0, 0, 4, 3)], [(0, 0, 4, 5)]]
    errors = [0.3, 0.1, 0.02]
    threshold = 1.5
    recon = template.reconstructor
    tracker = MultiPrecisionPixelTracker(
        stage_errors=errors,
        zscore_threshold=threshold,
        dilation_radius=1,
        reconstructor=recon,
        image=image,
        device="cpu",
    )
    follow_ups: list[torch.Tensor] = []

    def follow(maps, px):
        nxt = tracker.step(maps, px)
        follow_ups.append(nxt)
        return nxt

    legacy_stages = list(
        incremental_search(
            image,
            recon,
            stage_rects,
            pixel_batch=8,
            hyp_batch=5,
            n_psi=16,
            feature_chunk=4,
            follow_up_fn=follow,
            use_fused_kernel=False,
        )
    )
    legacy_pixels = [torch.arange(121), *follow_ups[: len(legacy_stages) - 1]]
    legacy_final = tracker.finalize(reshape=True)

    schedule = SearchSchedule(
        tuple(
            StageSpec(r, n_psi=16, error=e)
            for r, e in zip(stage_rects, errors, strict=True)
        ),
        tuple(
            DecisionChain(
                ZScoreIntervalClassifier(threshold, units="absolute"), DilationAbsorb(1)
            )
            for _ in errors
        ),
        keep_stage_results=True,
    )
    searcher = Searcher(template, schedule, execution=EXEC)
    results = list(searcher.session(ImageSet.single(image)))
    assert len(results) == len(legacy_stages)
    for new, old, old_px in zip(results, legacy_stages, legacy_pixels, strict=True):
        order = torch.argsort(new.pixel_index)
        assert torch.equal(new.pixel_index[order], old_px)
        for key in ("mip", "mean", "variance"):
            torch.testing.assert_close(
                getattr(new.maps, key)[order], old[key], rtol=1e-4, atol=1e-4
            )
    final = searcher.session(ImageSet.single(image)).run_all()
    assert torch.equal(final.label[0], legacy_final["status"][0])
    torch.testing.assert_close(
        final.mip[0], legacy_final["mip"][0], rtol=1e-4, atol=1e-4, equal_nan=True
    )
    assert torch.equal(
        final.last_stage[0].to(torch.int64), legacy_final["last_stage"][0]
    )


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
