"""Entry emission (``emit_entries``): every correlation above a z cut, against a dense
reference correlogram, for both normalisations, and unchanged by cutting a stage into
bands."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from panther_em.search.correlation.reduction import emission_active, emit_entries
from panther_em.search.images import SearchImages
from panther_em.search.searcher import Searcher
from panther_em.search.stage import Stage
from panther_em.search.template import Template
from test_search import EXEC, _image

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array_wrap__ must accept.*:DeprecationWarning",
    "ignore:.*__array__ implementation doesn't accept.*:DeprecationWarning",
)

RECTS = [(0, 0, 4, 3)]
N_PSI = 16
CUT = 2.0
EDGE = 1e-3  # z this close to the cut may round either way between the two paths


def reference_correlogram(template: Template, image: torch.Tensor) -> torch.Tensor:
    """``(P, N, n_psi)`` dense correlogram of one plane (kernels, FFT, einsum, irfft),
    psi already in the maps' convention (index ``(n_psi - a) % n_psi`` of the irfft)."""
    recon = template.reconstructor
    us = torch.as_tensor(recon.result.U[0]) * torch.as_tensor(recon.result.S)[None, None]
    us = us.reshape(-1, *us.shape[2:]).to(torch.complex64)  # (N, K, E)
    (k0, m0, ke, me), = RECTS
    cells = torch.stack(
        torch.meshgrid(torch.arange(k0, k0 + ke), torch.arange(m0, m0 + me), indexing="ij"),
        dim=-1,
    ).reshape(-1, 2)
    kernels = recon.kernels(cells)
    h, w = image.shape
    kh, kw = kernels.shape[-2:]
    spec = torch.fft.fft2(image)[None] * torch.fft.fft2(kernels, s=(h, w)).conj()
    feats = torch.fft.ifft2(spec)[..., : h - kh + 1, : w - kw + 1]
    y = feats.reshape(ke, me, -1).permute(2, 0, 1)  # (P, ke, me)
    c = torch.einsum("pkm,nkm->pnk", y, us[:, k0 : k0 + ke, m0 : m0 + me].conj())
    corr = torch.fft.irfft(c, n=N_PSI, dim=-1, norm="forward")  # (P, N, psi) irfft order
    return corr[..., (N_PSI - torch.arange(N_PSI)) % N_PSI]  # index = the maps' psi


def expected_entries(corr: torch.Tensor, rows: torch.Tensor | None):
    """``{(pixel, template, psi)}`` with z >= CUT over every template, and the set
    within EDGE of the cut; z's moments over all templates, entries from ``rows``."""
    flat = corr.reshape(corr.shape[0], -1)
    mean = flat.mean(dim=1)
    sd = ((flat**2).mean(dim=1) - mean**2).clamp_min(0).sqrt()
    z = (corr - mean[:, None, None]) / sd[:, None, None]
    if rows is not None:
        keep = torch.zeros(corr.shape[1], dtype=torch.bool)
        keep[rows] = True
        z = torch.where(keep[None, :, None], z, torch.full_like(z, -torch.inf))
    sure = {tuple(t) for t in (z >= CUT + EDGE).nonzero().tolist()}
    edge = {tuple(t) for t in ((z - CUT).abs() < EDGE).nonzero().tolist()}
    return sure, edge


def emitted(sink, result_pixels: torch.Tensor, template_ids=None):
    e = sink.gather()
    pix = result_pixels.cpu()[e["column"].long()]
    tpl = e["virtual"].long()
    if template_ids is not None:
        tpl = template_ids[tpl]
    return {(int(p), int(t), int(s)) for p, t, s in zip(pix, tpl, e["psi"].long())}, e


def run_stage(template, image, stage, execution=EXEC, normalize="stage"):
    searcher = Searcher(template, [stage], execution=execution)
    session = searcher.session(SearchImages.single(image))
    with emit_entries(CUT, normalize=normalize) as sink:
        assert emission_active()
        result = next(iter(session))
    assert not emission_active()
    return sink, result


def test_stage_normalisation_matches_the_dense_correlogram(synthetic_result):
    template = Template(synthetic_result(), device="cpu")
    image = _image(seed=5)
    sink, result = run_stage(template, image, Stage.create(RECTS, n_psi=N_PSI))
    got, e = emitted(sink, result.pixel_index)
    sure, edge = expected_entries(reference_correlogram(template, image), None)
    assert sure, "the cut must select something"
    assert sure <= got and not (got - sure - edge)
    # z is (value - mean) / sd of the pixel, and the largest entry is the map maximum
    torch.testing.assert_close(e["z"], (e["value"] - e["mean"]) / e["std"])
    col = e["column"].long()
    top = torch.full((result.n_pixels,), -torch.inf).scatter_reduce(0, col, e["value"], "amax")
    reach = result.maps.zscore.cpu() >= CUT
    torch.testing.assert_close(top[reach], result.maps.mip.cpu()[reach])
    assert torch.isinf(top[~reach]).all()


def test_all_template_normalisation_with_an_orientation_subset(synthetic_result):
    """Correlate a subset, normalise over every template: z as if all were searched,
    entries only from the subset (an angular restriction gates detection only)."""
    template = Template(synthetic_result(n_orientations=6), device="cpu")
    image = _image(seed=6)
    subset = torch.tensor([1, 2, 4])
    stage = Stage.create(RECTS, n_psi=N_PSI, orientations=subset)
    sink, result = run_stage(template, image, stage, normalize="all")
    template_ids, _ = stage.hypothesis_ids(Searcher(template, [stage], execution=EXEC).space)
    got, _ = emitted(sink, result.pixel_index, template_ids)
    sure, edge = expected_entries(reference_correlogram(template, image), subset)
    assert sure <= got and not (got - sure - edge)
    # the maps' z is over every template too
    corr = reference_correlogram(template, image).reshape(result.n_pixels, -1)
    mean = corr.mean(dim=1)
    torch.testing.assert_close(result.maps.mean.cpu(), mean, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_bands_emit_what_one_band_emits(synthetic_result):
    template = Template(synthetic_result(), device="cuda")
    image = _image(h=40, w=36, seed=7).cuda()
    stage = Stage.create(RECTS, n_psi=N_PSI)
    results = []
    for fraction in (0.5, 1e-9):  # one band; a band per GEMM batch of pixels
        exe = replace(EXEC, device_store_fraction=fraction)
        sink, result = run_stage(template, image, stage, execution=exe)
        results.append(emitted(sink, result.pixel_index)[0])
    assert results[0] and results[0] == results[1]
