"""Searcher / SearchSession end to end: one and two stages vs dense references."""

from __future__ import annotations

import gc
import warnings
import weakref
from dataclasses import replace

import h5py
import pytest
import torch
from noise_case import BOX, noise_case

from panther_em.search import runner as runner_module
from panther_em.search.config import ExecutionConfig, Tuning
from panther_em.search.correlation.contraction import BatchShape
from panther_em.search.decision import Decision, PixelLabel
from panther_em.search.featurization import (
    BaseImageFeaturization,
    FeatureChunk,
)
from panther_em.search.hypotheses import HypothesisSet
from panther_em.search.images import SearchImages
from panther_em.search.ledger import PixelLedger
from panther_em.search.results import MAP_NAMES, SearchResult, StageResult
from panther_em.search.searcher import Searcher
from panther_em.search.stage import Stage
from panther_em.search.template import Template
from panther_em.search.threshold import Calibration, ThresholdTest

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array_wrap__ must accept.*:DeprecationWarning",
    "ignore:.*__array__ implementation doesn't accept.*:DeprecationWarning",
)

EXEC = ExecutionConfig(
    tuning=Tuning(batch=BatchShape(8, 5), cell_chunk=4, use_cuda_graph=False)
)
# every plane resident at once, one kernel sweep over all of them
EXEC_JOINT = replace(EXEC, tuning=replace(EXEC.tuning, plane_major=False))


class StubFeaturization(BaseImageFeaturization):
    """Deterministic random features per cell and pixel, plus the plane's value there.

    A stand-in for the FFT and direct engines: pixel ``(row, col)`` of a plane gets
    ``noise[cell, row * out_w + col] + planes[d, row, col]``, so features follow the
    pixel identity and differ between planes whose filtered images differ.
    """

    def __init__(self, warp, rect, owner):
        super().__init__(warp, rect)
        self.owner = owner

    def chunks(self, planes, rows, cols):
        self.validate(planes, rows, cols)
        self.owner.calls += 1
        self.owner.planes_produced += int(planes.shape[0])
        out_w = int(planes.shape[-1]) - self.box_shape[1] + 1
        flat = (rows * out_w + cols).cpu()
        noise = []
        for k, m in self.cells.tolist():
            gen = torch.Generator().manual_seed(self.owner.seed * 100003 + k * 1009 + m)
            full = torch.complex(
                torch.randn(10_000, generator=gen), torch.randn(10_000, generator=gen)
            )
            noise.append(full[flat])
        noise = torch.stack(noise)  # (cells, P)
        for d in range(int(planes.shape[0])):
            values = noise + planes[d].cpu()[rows.cpu(), cols.cpu()][None]
            yield FeatureChunk(
                d,
                slice(0, self.n_cells),
                slice(0, int(rows.numel())),
                values.to(torch.complex64).to(self.device),
            )


class StubFeaturizer:
    """A featurization factory for ``Searcher(featurization=...)`` that counts calls."""

    def __init__(self, seed: int = 1) -> None:
        self.seed = seed
        self.calls = 0  # chunks() calls: one per (rectangle, band, pass, image)
        self.planes_produced = 0

    def __call__(self, warp, rect, mode):
        return StubFeaturization(warp, rect, self)


@pytest.fixture
def template(synthetic_result):
    return Template(synthetic_result(), device="cpu")


def _image(h=30, w=30, seed=0):
    return torch.randn(h, w, generator=torch.Generator().manual_seed(seed))


def compute_feature_stack(image, kernels):
    """Valid cross-correlation ``(B, L, H - kH + 1, W - kW + 1)`` of ``image``.

    ``image (B, H, W)`` with ``kernels (L, kH, kW)`` by a full-size FFT (the test's
    independent reference).
    """
    _, h, w = image.shape
    _, kh, kw = kernels.shape
    spectrum = torch.fft.fft2(image)[:, None] * torch.fft.fft2(kernels, s=(h, w)).conj()
    return torch.fft.ifft2(spectrum)[..., : h - kh + 1, : w - kw + 1]


def _dense_reference(template, image, rects, n_psi, planes=None, filters=None):
    """Per-pixel statistics from the dense correlogram (kernels, FFT, einsum, irfft)."""
    recon = template.reconstructor
    u0 = torch.as_tensor(recon.result.U[0])  # drop the singleton volume axis
    us = u0 * torch.as_tensor(recon.result.S)[None, None]
    us = us.reshape(-1, *us.shape[2:]).to(torch.complex64)  # (N, K, E)
    k_stop = max(k0 + ke for k0, _, ke, _ in rects)
    images = (
        [image]
        if filters is None
        else [
            torch.fft.irfft2(torch.fft.rfft2(image) * filters[d], s=image.shape)
            for d in planes
        ]
    )
    C = None
    for d, img in enumerate(images):
        for k0, m0, ke, me in rects:
            cells = torch.stack(
                torch.meshgrid(
                    torch.arange(k0, k0 + ke), torch.arange(m0, m0 + me), indexing="ij"
                ),
                dim=-1,
            ).reshape(-1, 2)
            kernels = template.reconstructor.kernels(cells)
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


def test_single_stage_matches_dense_reference_and_decides(template):
    image = _image()
    rects = [(0, 0, 4, 3), (0, 3, 2, 2)]
    mip, mean, var, best, psi, unique = _dense_reference(template, image, rects, 16)
    zscore = (mip - mean) / var.clamp_min(0).sqrt()
    ordered = zscore.flatten().sort().values
    threshold = float(0.5 * (ordered[60] + ordered[61]))  # strictly between two pixels

    sweep = Searcher(template, [Stage.create(rects, n_psi=16)], execution=EXEC)
    res = sweep.run(SearchImages.single(image))
    assert res.mip.shape == (1, 11, 11)
    torch.testing.assert_close(res.mip[0], mip, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(res.mean[0], mean, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(res.variance[0], var, rtol=1e-4, atol=1e-4)
    assert torch.equal(res.best_index[0][unique], best[unique])
    assert torch.equal(res.best_psi[0][unique], psi[unique])
    # no decision: every searched pixel stays AMBIGUOUS
    assert torch.all(res.label == int(PixelLabel.AMBIGUOUS)) and torch.all(
        res.last_stage == 0
    )

    # a plain threshold decides every pixel and scores none
    decided = Searcher(
        template,
        [Stage.create(rects, n_psi=16, decision=Decision(threshold))],
        execution=EXEC,
    ).run(SearchImages.single(image))
    accepted = decided.label[0] == int(PixelLabel.ACCEPTED)
    assert torch.equal(accepted, zscore > threshold)
    assert not (decided.label == int(PixelLabel.AMBIGUOUS)).any()
    assert torch.isnan(decided.score).all()


# --------------------------------------------------------------------------- #
# Search type 2: two stages
# --------------------------------------------------------------------------- #


def test_two_stage_rescored_pixels_match_fresh_searches(template):
    """Each stage equals a single-stage search of its rectangles on its pixels, and
    stage 1 searches exactly the pixels stage 0 accepted or left ambiguous."""
    image = _image(seed=3)
    rects0, rects1 = [(0, 0, 3, 3)], [(0, 0, 4, 5)]
    mip0, mean0, var0, *_ = _dense_reference(template, image, rects0, 16)
    z0 = (mip0 - mean0) / var0.clamp_min(0).sqrt()
    t0 = float(z0.flatten().quantile(0.6))  # stage 0 rejects 60 %
    # stage 0 rejects or abstains; stage 1 decides
    stages = [
        Stage.create(rects0, n_psi=16, decision=Decision(t0, accept=False)),
        Stage.create(rects1, n_psi=16, decision=Decision(1.5)),
    ]
    searcher = Searcher(template, stages, execution=EXEC, keep_stage_results=True)
    session = searcher.session(SearchImages.single(image))
    results = list(session)
    assert len(results) == 2 and results[0].n_pixels == 121
    after0 = results[0].labels_after
    assert not (after0 == int(PixelLabel.ACCEPTED)).any()  # stage 0 never accepts
    survivors = results[0].pixel_index[after0 == int(PixelLabel.AMBIGUOUS)]
    assert 0 < survivors.numel() < 121
    assert torch.equal(
        torch.sort(results[1].pixel_index).values, torch.sort(survivors).values
    )
    assert (after0 == int(PixelLabel.REJECTED)).any()
    assert torch.isnan(results[0].score).all()  # plain thresholds score nothing

    for stage, rects in zip(results, (rects0, rects1), strict=True):
        mip, mean, var, best, _psi, unique = _dense_reference(
            template, image, rects, 16
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

    final = session.snapshot()
    # rejected / absorbed pixels keep stage 0's maps; re-scored ones carry stage 1's
    flat_last = final.last_stage.reshape(-1)
    assert torch.all(flat_last[results[1].pixel_index] == 1)
    rejected = results[0].pixel_index[after0 == int(PixelLabel.REJECTED)]
    assert torch.all(flat_last[rejected] == 0)
    assert not (final.label == int(PixelLabel.AMBIGUOUS)).any()  # stage 1 decided all
    # stage 1 reads a different rectangle, so stage 0's store was dropped, not compacted
    assert [g.rect for g in session.store.groups] == [(0, 0, 4, 5)]


def test_two_stage_healpix_level_and_planes(synthetic_result):
    """Stage 0 at a coarser HEALPix level on one CTF plane, stage 1 over everything."""
    result = synthetic_result(s2_nside=2, n_filters=2)  # 48 orientations x 2 filters
    template = Template(result, device="cpu")
    stub = StubFeaturizer()
    test0 = Decision(0.0, accept=False, absorb_radius=1)
    test1 = Decision(0.5)
    stages = [
        Stage.create(
            [(0, 0, 3, 3)], n_psi=16, orientations=1, planes=(1,), decision=test0
        ),
        Stage.create([(0, 0, 4, 5)], n_psi=16, decision=test1),
    ]
    searcher = Searcher(
        template, stages, execution=EXEC_JOINT, featurization=stub, n_defocus=3
    )
    assert searcher.ipix is not None and searcher.ipix.tolist() == list(range(48))
    t_ids, d_ids = stages[0].hypothesis_ids(searcher.space, searcher.ipix)
    assert t_ids.numel() == 12 * 2 and d_ids.tolist() == [
        1
    ]  # one child per parent pixel
    images = SearchImages.from_stack(
        torch.zeros(1, 30, 30), filters=torch.ones(3, 30, 16)
    )
    session = searcher.session(images)
    results = list(session)
    # stage 1's larger rectangle is a new group, so it is produced on all three planes
    assert stub.calls == 2 and stub.planes_produced == 4
    assert [(g.rect, g.planes) for g in session.store.groups] == [
        ((0, 0, 4, 5), (0, 1, 2))
    ]
    assert results[0].maps.n_evaluated == 24 * 16
    assert results[1].maps.n_evaluated == 3 * 96 * 16
    final = session.snapshot()
    assert not (final.label == int(PixelLabel.AMBIGUOUS)).any()  # stage 1 decides all
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
    searcher = Searcher(
        template, stages, execution=EXEC, featurization=stub, keep_stage_results=True
    )
    session = searcher.session(SearchImages.from_stack(torch.zeros(2, 30, 30)))
    results = list(session)
    assert [r.stage_index for r in results] == [0] and results[0].n_pixels == 242
    assert results[0].n_ambiguous_after == 0 and stub.calls == 2  # one per image
    res = session.snapshot()
    assert len(res.stage_results) == 1 and isinstance(res.stage_results[0], StageResult)
    assert torch.all(res.label == int(PixelLabel.REJECTED)) and torch.all(
        res.last_stage == 0
    )

    # everything accepted: stage 1 re-scores every pixel
    stages = [
        Stage.create([(0, 0, 2, 2)], n_psi=16, decision=Decision(-1e9)),
        Stage.create([(0, 0, 4, 3)], n_psi=16),
    ]
    searcher = Searcher(
        template, stages, execution=EXEC, featurization=StubFeaturizer()
    )
    res = searcher.run(SearchImages.from_stack(torch.zeros(2, 30, 30)))
    assert torch.all(res.last_stage == 1) and torch.all(
        res.label == int(PixelLabel.AMBIGUOUS)
    )


def test_searcher_validation(template):
    with pytest.raises(ValueError, match="fp16"):
        Searcher(template, [Stage.create([(0, 0, 2, 2)], precision="fp16", n_psi=16)])
    with pytest.raises(ValueError, match="exceeds"):
        Searcher(template, [Stage.create([(0, 0, 8, 2)], n_psi=32)])
    with pytest.raises(ValueError, match="defocus"):
        Searcher(
            template, [Stage.create([(0, 0, 2, 2)], n_psi=16)], n_defocus=2
        ).session(SearchImages.single(_image()))
    with pytest.raises(ValueError, match="planes"):
        Searcher(
            template, [Stage.create([(0, 0, 2, 2)], n_psi=16, planes=(3,))], n_defocus=2
        )
    with pytest.raises(ValueError, match="planes"):
        Searcher(
            template, [Stage.create([(0, 0, 2, 2)], n_psi=16, planes=(3,))], n_defocus=2
        )
    with pytest.raises(ValueError, match="HEALPix"):
        Searcher(template, [Stage.create([(0, 0, 2, 2)], n_psi=16, orientations=1)])
    with pytest.raises(ValueError):
        Searcher(template, [])


def test_searcher_validates_the_threshold_test(synthetic_result):
    case = noise_case()
    rect = (0, 0, 2, 2)
    table = case.table([rect])

    def searcher(t, template=case.template):
        stage = Stage.create([rect], n_psi=16, decision=Decision(3.0, ThresholdTest(t)))
        return Searcher(
            template,
            [stage],
            execution=EXEC,
            featurization=StubFeaturizer(),
            n_defocus=3,
        )

    # the table against the template, when the searcher is built
    with pytest.raises(ValueError, match=r"stage 0: .*rectangles"):
        searcher(case.table([(0, 0, 2, 3)]))
    with pytest.raises(ValueError, match=r"stage 0: .*box"):
        searcher(table, Template(synthetic_result(), device="cpu"))
    with pytest.raises(ValueError, match=r"stage 0: .*no Fourier filter"):
        searcher(table, Template(synthetic_result(box=BOX), device="cpu"))

    # the micrograph against the table, when the session starts
    images = case.images(table)
    plain = SearchImages.single(images.image, filters=images.ctf)
    with pytest.raises(ValueError, match=r"stage 0: .*noise model"):
        searcher(table).session(plain)
    with pytest.raises(ValueError, match=r"stage 0: .*frequency axis"):
        searcher(case.table([rect], pad=3)).session(images)

    session = searcher(table).session(images)
    assert session.noise is images and session.images.n_images == 1
    assert torch.equal(session.images.images[0], images.image)
    assert torch.equal(session.images.filters, images.ctf)
    res = searcher(table).run(images)
    assert torch.isfinite(res.score).all()


def test_two_stage_search_under_the_threshold_test(tmp_path):
    """A micrograph searched with the test in both stages: the scores are the
    decision's, the labels follow them, and the survivors are re-scored."""
    case = noise_case()
    rects0, rects1 = [(0, 0, 2, 3)], [(0, 0, 3, 4)]
    table0, table1 = case.table(rects0), case.table(rects1)
    images = case.images(table0)
    cal = Calibration(alpha=0.05)
    plain = Searcher(
        case.template, [Stage.create(rects0, n_psi=16)], execution=EXEC, n_defocus=3
    ).run(images)
    threshold = float(plain.zscore.flatten().quantile(0.9))
    stages = [
        Stage.create(
            rects0,
            n_psi=16,
            planes=(1,),
            decision=Decision(threshold, ThresholdTest(table0, cal), accept=False),
        ),
        Stage.create(
            rects1, n_psi=16, decision=Decision(threshold, ThresholdTest(table1, cal))
        ),
    ]
    searcher = Searcher(
        case.template, stages, execution=EXEC, n_defocus=3, keep_stage_results=True
    )
    session = searcher.session(images)
    results = list(session)
    assert len(results) == 2
    first, second = results
    after0 = first.labels_after
    assert (after0 == int(PixelLabel.REJECTED)).any()
    assert (after0 == int(PixelLabel.AMBIGUOUS)).any()
    assert not (after0 == int(PixelLabel.ACCEPTED)).any()
    assert torch.isfinite(first.score).all() and torch.isfinite(second.score).all()
    z_alpha = cal.z_alpha
    assert torch.equal(after0 == int(PixelLabel.REJECTED), first.score >= z_alpha)
    survivors = first.pixel_index[after0 == int(PixelLabel.AMBIGUOUS)]
    assert torch.equal(second.pixel_index.sort().values, survivors.sort().values)

    # each stage's score is its decision's distance on the stage's maps
    for index, (result, stage) in enumerate(zip(results, stages, strict=True)):
        hyps = HypothesisSet.create(
            searcher.space,
            *stage.hypothesis_ids(searcher.space, searcher.ipix),
            hyp_batch=6,
            n_psi=16,
        )
        ledger = PixelLedger(session.geometry, "cpu")
        ledger.write(result.maps, result.pixel_index, index, hyps)
        assert torch.equal(ledger.best_index[result.pixel_index], result.best_index)
        score = stage.decision.score(ledger, result.pixel_index, hyps, images)
        torch.testing.assert_close(score, result.score)

    final = session.snapshot()
    rejected = final.label == int(PixelLabel.REJECTED)
    accepted = final.label == int(PixelLabel.ACCEPTED)
    rescored = final.last_stage == 1
    assert torch.equal(accepted, rescored & ~rejected & (final.zscore > threshold))
    final.save(tmp_path / "tested.h5")
    loaded = SearchResult.load(tmp_path / "tested.h5")
    torch.testing.assert_close(loaded.score, final.score, equal_nan=True)
    assert loaded.stages[1].decision.test is None  # tests are not restored


def test_center_mode_orientation_subset_and_run_many(template):
    stub = StubFeaturizer()
    coarse = torch.arange(0, template.space.n_orientations, 2)
    stages = [Stage.create([(0, 0, 3, 3)], n_psi=16, orientations=coarse)]
    searcher = Searcher(
        template,
        stages,
        execution=EXEC_JOINT,
        featurization=stub,
        n_defocus=2,
        defocus_values=torch.tensor([-500.0, 500.0]),
    )
    boxes = SearchImages.from_boxes(
        torch.zeros(7, 20, 20), filters=torch.ones(2, 20, 11)
    )
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
    assert len(searcher.correlator.loops) == 1  # the loop was reused across image sets


def test_search_result_round_trip_and_masked_pixels(template, tmp_path):
    stub = StubFeaturizer()
    mask = torch.zeros(11, 11, dtype=torch.bool)
    mask[2:9, 2:9] = True
    probe = Searcher(
        template,
        [Stage.create([(0, 0, 2, 2)], n_psi=16)],
        execution=EXEC,
        featurization=stub,
    ).run(SearchImages.single(torch.zeros(30, 30), mask=mask))
    z = probe.zscore[0][mask]
    ordered = z.sort().values
    threshold = float(0.5 * (ordered[-11] + ordered[-10]))  # exactly 10 pixels above
    stages = [
        Stage.create([(0, 0, 2, 2)], n_psi=16, decision=Decision(threshold)),
        Stage.create([(0, 0, 4, 5)], n_psi=32),
    ]
    searcher = Searcher(
        template,
        stages,
        execution=EXEC,
        featurization=StubFeaturizer(),
        keep_stage_results=True,
    )
    res = searcher.run(SearchImages.single(torch.zeros(30, 30), mask=mask))
    assert (res.label[0][~mask] == int(PixelLabel.MASKED)).all()
    assert (
        torch.isnan(res.mip[0][~mask]).all() and (res.last_stage[0][~mask] == -1).all()
    )
    assert torch.isnan(res.score).all()  # plain thresholds score nothing
    n_rescored = int((z > threshold).sum())
    assert n_rescored == 10
    assert (res.last_stage == 1).sum() == n_rescored
    assert (res.last_stage == 0).sum() == 49 - n_rescored
    psi = res.psi_degrees()
    assert torch.isnan(psi[0][~mask]).all()
    assert psi[0][res.last_stage[0] == 1].max() < 360 and res.n_psi_of_stage == (16, 32)

    path = tmp_path / "result.h5"
    res.save(path)
    with h5py.File(path, "r") as f:  # integer maps compressed, float maps raw
        assert f["label"].compression == f["stages/stage0/pixel_index"].compression
        assert f["label"].compression == "gzip" and f["mip"].compression is None
    loaded = SearchResult.load(path)
    for name in MAP_NAMES:
        torch.testing.assert_close(
            getattr(loaded, name), getattr(res, name), equal_nan=True
        )
    assert (
        loaded.geometry == res.geometry and loaded.n_psi_of_stage == res.n_psi_of_stage
    )
    assert [s.rectangles for s in loaded.stages] == [s.rectangles for s in res.stages]
    assert (
        loaded.stages[0].decision.threshold == threshold
        and loaded.stages[1].decision is None
    )
    torch.testing.assert_close(
        loaded.decode()["psi"], res.decode()["psi"], equal_nan=True
    )
    torch.testing.assert_close(
        loaded.decode()["phi"], res.decode()["phi"], equal_nan=True
    )
    assert [r.stage_index for r in loaded.stage_results] == [0, 1]
    for got, want in zip(loaded.stage_results, res.stage_results, strict=True):
        assert torch.equal(got.pixel_index, want.pixel_index)
        torch.testing.assert_close(got.maps.zscore, want.maps.zscore)
        assert torch.equal(got.best_index, want.best_index)
        assert torch.equal(got.maps.best_virtual, want.maps.best_virtual.cpu())
        assert (
            got.n_ambiguous_after == want.n_ambiguous_after
            and got.elapsed_s == want.elapsed_s
        )
    arrays = res.to_numpy()
    assert arrays["mip"].shape == (1, 11, 11)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res.to("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_bands_match_one_band_and_never_coexist(synthetic_result, monkeypatch):
    """A stage cut into bands gives the maps of one band, and every band's store is
    released before the next one is allocated (two would not fit a real device)."""
    template = Template(synthetic_result(), device="cuda")
    image = _image(h=40, w=36, seed=7)  # 21 x 17 = 357 pixels
    stages = [
        Stage.create([(0, 0, 3, 4)], n_psi=16, decision=Decision(0.0, accept=False)),
        Stage.create([(0, 0, 4, 5)], n_psi=16),
    ]
    exe = replace(EXEC, tuning=replace(EXEC.tuning, batch=BatchShape(64, 5)))
    alive_at_allocation: list[int] = []
    stores: list[weakref.ref] = []

    class Tracked(runner_module.FeatureStore):
        def __init__(self, *args, **kwargs):
            alive_at_allocation.append(sum(r() is not None for r in stores))
            super().__init__(*args, **kwargs)
            stores.append(weakref.ref(self))

    monkeypatch.setattr(runner_module, "FeatureStore", Tracked)

    def run(fraction):
        searcher = Searcher(
            template,
            stages,
            execution=replace(exe, device_store_fraction=fraction),
            featurization=StubFeaturizer(),
            keep_stage_results=True,
        )
        return searcher.run(SearchImages.single(image.cuda()))

    one = run(0.5)
    assert len(stores) == 2  # one store per stage
    banded = run(1e-9)  # every band holds a single GEMM batch of pixels
    assert len(stores) > 2 + 5
    assert alive_at_allocation[2:] == [0] * (len(stores) - 2)
    for name in MAP_NAMES:
        torch.testing.assert_close(
            getattr(banded, name), getattr(one, name), equal_nan=True
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.filterwarnings("ignore:ComplexHalf support is experimental")
def test_cuda_mixed_precision_two_stage_matches_cpu(synthetic_result):
    result = synthetic_result(n_orientations=8)
    image = _image(h=32, w=30, seed=5)

    def stages(precision0):
        return [
            Stage.create(
                [(0, 0, 2, 3)], precision=precision0, n_psi=128, decision=Decision(0.0)
            ),
            Stage.create([(0, 0, 4, 5)], n_psi=128, decision=Decision(0.0)),
        ]

    cpu = Searcher(
        Template(result, device="cpu"),
        stages("fp32"),
        execution=ExecutionConfig(tuning=Tuning(batch=BatchShape(8, 4), cell_chunk=4)),
    ).run(SearchImages.single(image))
    gpu_searcher = Searcher(
        Template(result, device="cuda"),
        stages("fp16"),
        # fp32 direct contraction: the fp16 default is 5e-4 off on the features, and
        # stage 1 (fp32) is compared with the CPU at 1e-4 below
        execution=ExecutionConfig(
            direct_precision="fp32",
            tuning=Tuning(batch=BatchShape(8, 4), cell_chunk=4, use_cuda_graph=True),
        ),
        keep_stage_results=True,
    )
    gpu = gpu_searcher.run(SearchImages.single(image)).to("cpu")
    assert any(
        loop.graph is not None for loop in gpu_searcher.correlator.loops.values()
    )
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
        template,
        stages,
        execution=EXEC_JOINT,
        featurization=stub,
        n_defocus=3,
        keep_stage_results=True,
    )
    images = SearchImages.from_stack(
        torch.zeros(1, 30, 30), filters=torch.ones(3, 30, 16)
    )
    session = searcher.session(images)
    results = list(session)
    # plane 1, then all three: a stage never reuses an earlier stage's features
    assert stub.calls == 2 and stub.planes_produced == 4
    assert [g.planes for g in session.store.groups] == [(0, 1, 2)]
    accepted = int((results[0].labels_after == int(PixelLabel.ACCEPTED)).sum())
    assert results[0].n_pixels == 121 and results[1].n_pixels == accepted > 0

    # stage 1 (all planes, survivors only) equals a fresh all-plane search there
    mask = torch.zeros(1, 11, 11, dtype=torch.bool)
    mask.reshape(-1)[results[1].pixel_index] = True
    fresh = Searcher(
        template,
        [Stage.create([rect], n_psi=16)],
        execution=EXEC,
        featurization=StubFeaturizer(),
        n_defocus=3,
    ).run(
        SearchImages.from_stack(
            torch.zeros(1, 30, 30), filters=torch.ones(3, 30, 16), mask=mask
        )
    )
    order = torch.argsort(results[1].pixel_index)
    torch.testing.assert_close(
        results[1].maps.mip[order], fresh.mip.reshape(-1)[results[1].pixel_index[order]]
    )
    assert torch.equal(
        results[1].best_index[order],
        fresh.best_index.reshape(-1)[results[1].pixel_index[order]],
    )


def test_session_store_is_released_after_the_session(template):
    """The searcher's cached loops must not retain a finished session's blocks."""
    searcher = Searcher(
        template,
        [Stage.create([(0, 0, 4, 5)], n_psi=16)],
        execution=EXEC,
        featurization=StubFeaturizer(),
    )
    session = searcher.session(SearchImages.single(torch.zeros(30, 30)))
    session.run_all()
    store_ref = weakref.ref(session.store)
    block_ref = weakref.ref(session.store.groups[0].data)  # the bytes themselves
    gc.disable()  # freed by reference counting alone: no cycle holds the session
    try:
        del session
        assert store_ref() is None  # only the session owned the store
    finally:
        gc.enable()
    assert block_ref() is None  # and nothing (the cached loop's plan) kept its blocks
    assert searcher.correlator.loops  # the loop cache survives for the next image set


def test_plane_major_passes_match_the_joint_search(template):
    """Plane-by-plane passes (one plane resident at a time) reproduce the joint maps."""
    rect = (0, 0, 3, 4)
    image = SearchImages.single(_image(seed=11), filters=torch.rand(3, 30, 16))
    stage = Stage.create([rect], n_psi=16, planes=(0, 2), decision=Decision(0.5))

    def run(exe, stub):
        searcher = Searcher(
            template,
            [stage],
            execution=exe,
            featurization=stub,
            n_defocus=3,
            keep_stage_results=True,
        )
        session = searcher.session(image)
        result = session.run_all()
        return session, result

    joint_stub, major_stub = StubFeaturizer(), StubFeaturizer()
    _, joint = run(EXEC_JOINT, joint_stub)
    major_session, major = run(EXEC, major_stub)
    assert joint_stub.calls == 1 and joint_stub.planes_produced == 2
    assert (
        major_stub.calls == 2 and major_stub.planes_produced == 2
    )  # one pass per plane
    # each pass featurized its own plane; the session keeps the last pass's store
    assert [g.planes for g in major_session.store.groups] == [(2,)]
    for name in MAP_NAMES:
        torch.testing.assert_close(
            getattr(major, name),
            getattr(joint, name),
            rtol=1e-5,
            atol=1e-5,
            equal_nan=True,
        )
    torch.testing.assert_close(
        major.stage_results[0].best_index, joint.stage_results[0].best_index
    )
    assert (
        major.stage_results[0].maps.n_evaluated
        == joint.stage_results[0].maps.n_evaluated
    )
    # the two passes shared one loop and the prepared weights of the hypotheses
    assert len(major_session.searcher.correlator.loops) == 1
    # a later stage over the same planes starts from an empty store (featurizes again)
    stages = [
        stage,
        Stage.create([rect], n_psi=16, planes=(0, 2), decision=Decision(0.5)),
    ]
    stub = StubFeaturizer()
    searcher = Searcher(
        template,
        stages,
        execution=EXEC,
        featurization=stub,
        n_defocus=3,
        keep_stage_results=True,
    )
    two = searcher.run(image)
    assert stub.calls == 4
    torch.testing.assert_close(
        two.stage_results[1].maps.mip, joint.stage_results[0].maps.mip
    )
