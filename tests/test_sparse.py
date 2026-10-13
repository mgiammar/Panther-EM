"""Sparse pair evaluation vs the dense correlogram; the sparse stage in a session."""

from __future__ import annotations

import pytest
import torch

from panther_em.inference.correlation import compute_feature_stack
from panther_em.inference.search.basis import Template
from panther_em.inference.search.decisions import (
    ThresholdClassifier,
    TopFractionSelector,
)
from panther_em.inference.search.hypotheses import HypothesisSubset
from panther_em.inference.search.ledger import PixelLabel
from panther_em.inference.search.pairs import PairEmissionSpec, PairList
from panther_em.inference.search.results import SearchResult
from panther_em.inference.search.schedule import SearchSchedule, StageSpec
from panther_em.inference.search.searcher import ExecutionConfig, Searcher
from panther_em.inference.search.sparse import PairScores, SparsePairEvaluator
from panther_em.inference.search.types import BatchShape, ImageSet, SearchGeometry

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array_wrap__ must accept.*:DeprecationWarning",
    "ignore:.*__array__ implementation doesn't accept.*:DeprecationWarning",
)

RECTS = [(0, 0, 4, 3), (0, 3, 2, 2)]
N_PSI = 16


def _images(n_defocus=1, seed=0):
    gen = torch.Generator().manual_seed(seed)
    raw = torch.randn(2, 30, 30, generator=gen)
    filters = None if n_defocus == 1 else torch.rand(n_defocus, 30, 16, generator=gen)
    return ImageSet.from_stack(raw, filters=filters)


def _dense_corr(template, images, rects, n_psi):
    """``(P, D, N, psi)`` correlogram of every pixel against every template."""
    res = template.result
    us = (torch.as_tensor(res.U) * torch.as_tensor(res.S)[None, None]).to(
        torch.complex64
    )
    us = us.reshape(-1, *us.shape[2:])  # (N, K, E)
    k_stop = max(k0 + ke for k0, _, ke, _ in rects)
    d_planes = images.n_defocus
    out = None
    for d in range(d_planes):
        if images.filters is None:
            imgs = images.images
        else:
            imgs = torch.fft.irfft2(
                torch.fft.rfft2(images.images) * images.filters[d], s=(30, 30)
            )
        C = None
        for k0, m0, ke, me in rects:
            cells = torch.stack(
                torch.meshgrid(
                    torch.arange(k0, k0 + ke), torch.arange(m0, m0 + me), indexing="ij"
                ),
                dim=-1,
            ).reshape(-1, 2)
            feats = compute_feature_stack(
                imgs, template.kernels.spatial(cells)
            )  # (B, cells, oh, ow)
            b, _, oh, ow = feats.shape
            y = feats.permute(0, 2, 3, 1).reshape(b * oh * ow, ke, me)
            if C is None:
                C = torch.zeros(b * oh * ow, us.shape[0], k_stop, dtype=torch.complex64)
            C[:, :, k0 : k0 + ke] += torch.einsum(
                "pkm,nkm->pnk", y, us[:, k0 : k0 + ke, m0 : m0 + me].conj()
            )
        corr = torch.fft.irfft(C, n=n_psi, dim=-1, norm="forward")  # (P, N, psi)
        out = corr[:, None] if out is None else torch.cat([out, corr[:, None]], dim=1)
    return out


def _all_pairs(n_pixels, n_templates):
    pixel = torch.arange(n_pixels).repeat_interleave(n_templates)
    template = torch.arange(n_templates).repeat(n_pixels)
    return PairList(pixel, template, n_templates, torch.zeros(len(pixel)))


@pytest.mark.parametrize("n_defocus", [1, 2])
def test_all_pairs_match_the_dense_correlogram(synthetic_result, n_defocus):
    template = Template(synthetic_result(n_orientations=6), device="cpu")
    images = _images(n_defocus)
    geo = SearchGeometry.from_images(images, template.template_shape)  # 2 x 11 x 11
    spec = StageSpec(RECTS, n_psi=N_PSI)
    corr = _dense_corr(template, images, RECTS, N_PSI)  # (P, D, N, psi)
    pairs = _all_pairs(geo.n_pixels, 6)
    evaluator = SparsePairEvaluator(template)
    scores = evaluator.evaluate(images, pairs, spec)
    ref = corr.permute(0, 2, 1, 3).reshape(
        geo.n_pixels * 6, n_defocus * N_PSI
    )  # pair-major
    ref_value, ref_arg = ref.max(dim=1)
    torch.testing.assert_close(scores.value, ref_value, rtol=1e-4, atol=1e-4)
    unique = (ref == ref_value[:, None]).sum(1) == 1
    assert torch.equal(scores.plane[unique], (ref_arg // N_PSI)[unique])
    assert torch.equal(scores.psi[unique], ((N_PSI - ref_arg % N_PSI) % N_PSI)[unique])

    # a plane restriction equals that plane's own maximum
    if n_defocus == 2:
        one = evaluator.evaluate(images, pairs, spec, planes=[1])
        torch.testing.assert_close(
            one.value, ref[:, N_PSI:].max(dim=1).values, rtol=1e-4, atol=1e-4
        )
        assert torch.all(one.plane == 1)

    # chunking only changes GEMM blocking (rounding), not the result
    small = SparsePairEvaluator(template, pixel_chunk=5, pair_chunk=7).evaluate(
        images, pairs, spec
    )
    torch.testing.assert_close(small.value, scores.value, rtol=1e-5, atol=1e-5)
    assert (small.psi == scores.psi).float().mean() > 0.99
    # the patch budget caps the pixels per chunk across planes (every plane of a
    # chunk goes through the same GEMMs); a budget of three patches holds one pixel
    # per chunk at two planes, and the result is unchanged
    kh, kw = template.template_shape
    tight = SparsePairEvaluator(template, pixel_chunk=5, patch_budget=3 * kh * kw * 4)
    assert tight.chunk_pixels(n_defocus, torch.float32) == max(1, 3 // n_defocus)
    tight_scores = tight.evaluate(images, pairs, spec)
    torch.testing.assert_close(tight_scores.value, scores.value, rtol=1e-5, atol=1e-5)
    assert torch.equal(tight_scores.plane[unique], scores.plane[unique])

    # best_per_pixel picks each pixel's maximum and encodes global ids
    pixel, best = scores.best_per_pixel()
    assert torch.equal(pixel, torch.arange(geo.n_pixels))
    flat = corr.reshape(geo.n_pixels, -1)  # (P, D*N*psi)
    mip, amax = flat.max(dim=1)
    torch.testing.assert_close(scores.value[best], mip, rtol=1e-4, atol=1e-4)
    uniq = (flat == mip[:, None]).sum(1) == 1
    assert torch.equal(scores.global_ids()[best][uniq], (amax // N_PSI)[uniq])


def test_best_per_pixel_ties_take_the_first_pair():
    pairs = PairList(
        torch.tensor([4, 2, 4, 2]), torch.tensor([0, 1, 2, 3]), 5, torch.zeros(4)
    )
    scores = PairScores(
        pairs,
        torch.tensor([1.0, 3.0, 1.0, 3.0]),
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([0, 1, 0, 1]),
        8,
    )
    pixel, best = scores.best_per_pixel()
    assert pixel.tolist() == [2, 4] and best.tolist() == [1, 0]
    assert scores.global_ids().tolist() == [0, 6, 2, 8]


def test_session_sparse_stage_after_emission(synthetic_result, tmp_path):
    pytest.importorskip("healpy")
    result = synthetic_result(n_orientations=12, s2_nside=1)
    template = Template(result, device="cpu")
    images = _images(n_defocus=2, seed=3)
    images = ImageSet.from_stack(
        images.images,
        filters=images.filters,
        pixel_maps={
            "image_norm": torch.ones(11, 11),
            "noise_variance": torch.ones(11, 11),
        },
    )
    emit = StageSpec(
        [(0, 0, 2, 2)],
        n_psi=N_PSI,
        pairs=PairEmissionSpec(
            1.0, pixel_radius=1, orientation_rings=1, pair_dtype=torch.float32
        ),
        hypotheses=HypothesisSubset(defocus_ids=torch.tensor([0])),
    )
    sparse = StageSpec(RECTS, n_psi=N_PSI, mode="sparse_pairs")
    schedule = SearchSchedule(
        (emit, sparse), (TopFractionSelector(0.5), ThresholdClassifier(1.5))
    )
    exe = ExecutionConfig(batch=BatchShape(8, 6), cell_chunk=4, use_cuda_graph=False)
    searcher = Searcher(template, schedule, execution=exe, n_defocus=2)
    session = searcher.session(images)
    stage0, stage1 = list(session)
    assert (
        stage0.pairs is not None and stage1.scores is not None and stage1.pairs is None
    )
    # the sparse stage scored the emitted pairs on the pixels the selector kept
    assert torch.equal(stage1.pixel_index, torch.unique(stage1.scores.pairs.pixel))
    assert set(stage1.pixel_index.tolist()) <= set(stage0.pairs.pixels().tolist())
    assert len(stage1.scores.pairs) <= len(stage0.pairs)

    # survivors' mip = dense full-rank correlogram max over the pair's templates/planes
    corr = _dense_corr(template, images, RECTS, N_PSI)  # (P, D, N, psi)
    for i, p in enumerate(stage1.pixel_index.tolist()):
        own = stage1.scores.pairs.pixel == p
        tmpl = stage1.scores.pairs.template[own]
        ref = corr[p][:, tmpl].max()
        torch.testing.assert_close(stage1.maps.mip[i], ref, rtol=1e-4, atol=1e-4)
    # moments are stage 0's, the labels come from the threshold on the new z-score
    res = session.snapshot()
    px = stage1.pixel_index
    torch.testing.assert_close(
        res.mean.reshape(-1)[px],
        stage0.maps.mean[torch.searchsorted(stage0.pixel_index, px)],
    )
    assert torch.all(res.last_stage.reshape(-1)[px] == 1)
    z = res.zscore.reshape(-1)[px]
    labels = res.label.reshape(-1)[px]
    assert torch.equal(labels == int(PixelLabel.ACCEPTED), z > 1.5)
    assert not (res.label == int(PixelLabel.UNCERTAIN)).any()
    decoded = res.decode()
    assert torch.all(decoded.defocus_index.reshape(-1)[px] >= 0)

    # the stage mode survives HDF5
    path = tmp_path / "sparse_result.h5"
    res.save(path)
    assert SearchResult.load(path).schedule.stages[1].mode == "sparse_pairs"


def test_sparse_stage_validation(synthetic_result):
    emit = StageSpec([(0, 0, 2, 2)], n_psi=N_PSI, pairs=PairEmissionSpec(1.0))
    with pytest.raises(ValueError, match="no emitting stage"):
        SearchSchedule((StageSpec(RECTS, n_psi=N_PSI, mode="sparse_pairs"),))
    with pytest.raises(ValueError, match="cannot emit"):
        StageSpec(RECTS, n_psi=N_PSI, mode="sparse_pairs", pairs=PairEmissionSpec(1.0))
    with pytest.raises(ValueError, match="defocus_ids"):
        StageSpec(
            RECTS,
            n_psi=N_PSI,
            mode="sparse_pairs",
            hypotheses=HypothesisSubset(template_ids=torch.tensor([1])),
        )
    with pytest.raises(ValueError, match="mode"):
        StageSpec(RECTS, n_psi=N_PSI, mode="dense_pairs")
    SearchSchedule((emit, StageSpec(RECTS, n_psi=N_PSI, mode="sparse_pairs")))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.filterwarnings("ignore:ComplexHalf support is experimental")
def test_cuda_fp16_sparse_matches_cpu(synthetic_result):
    result = synthetic_result(n_orientations=6)
    images = _images(n_defocus=2)
    cpu = Template(result, device="cpu")
    geo = SearchGeometry.from_images(images, cpu.template_shape)
    pairs = _all_pairs(geo.n_pixels, 6)
    ref = SparsePairEvaluator(cpu).evaluate(
        images, pairs, StageSpec(RECTS, n_psi=N_PSI)
    )
    gpu = Template(result, device="cuda")
    for precision, tol in (("fp32", 1e-4), ("fp16", 5e-2)):
        got = SparsePairEvaluator(gpu).evaluate(
            images, pairs, StageSpec(RECTS, precision=precision, n_psi=N_PSI)
        )
        torch.testing.assert_close(got.value.cpu(), ref.value, rtol=tol, atol=tol)
        agree = (got.psi.cpu() == ref.psi).float().mean()
        assert agree > (0.99 if precision == "fp32" else 0.9)
