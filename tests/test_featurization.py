"""featurization: the FFT and direct engines against a dense float64 reference."""

from __future__ import annotations

import pytest
import torch
from conftest import count_kernel_builds, requires_cuda

from panther_em.inference.search.featurization import (
    AdjointContraction,
    DirectFeaturization,
    FeatureChunk,
    FFTFeaturization,
    PlanarContraction,
    rectangle_cells,
)
from panther_em.inference.search.featurization import (
    direct_featurization as direct_module,
)
from panther_em.inference.template import TemplateWarp

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array_wrap__ must accept.*:DeprecationWarning",
    "ignore:.*__array__ implementation doesn't accept.*:DeprecationWarning",
)

RECT = (1, 1, 2, 3)  # k in [1, 3), m in [1, 4): 6 cells
BOX = 20  # the synthetic template's box


def make_warp(synthetic_result, device="cpu"):
    return TemplateWarp(synthetic_result(), device=device)


@pytest.fixture
def warp(synthetic_result):
    return make_warp(synthetic_result)


def planes(d=2, h=34, w=31, seed=0):
    return torch.randn(d, h, w, generator=torch.Generator().manual_seed(seed))


def positions(n, h, w, seed=1):
    """Unsorted (with repeats) box origins inside ``(h, w)`` planes."""
    gen = torch.Generator().manual_seed(seed)
    rows = torch.randint(0, h - BOX + 1, (n,), generator=gen)
    cols = torch.randint(0, w - BOX + 1, (n,), generator=gen)
    return rows, cols


def reference(warp, x, rows, cols, rect=RECT):
    """``(D, n_cells, P)`` features in float64 from the dense Cartesian kernels."""
    kernels = warp.spatial(rectangle_cells(rect)).cpu().to(torch.complex128)
    boxes = torch.stack(
        [
            x[:, r : r + BOX, c : c + BOX]
            for r, c in zip(rows.tolist(), cols.tolist(), strict=True)
        ],
        dim=1,
    ).to(torch.complex128)  # (D, P, kH, kW)
    return torch.einsum("dpij,nij->dnp", boxes, kernels.conj()).to(torch.complex64)


def assert_features(actual, expected, tol=1e-5):
    """Max error relative to the features' scale (fp16 runs at ~5e-4 of it)."""
    assert actual.shape == expected.shape and actual.dtype is torch.complex64
    error = float((actual.cpu() - expected).abs().max() / expected.abs().max())
    assert error < tol, f"max error {error:.2e} of scale, allowed {tol:.0e}"


def coverage(chunks, d, n_cells, p):
    """How many times each (plane, cell, position) was yielded."""
    hits = torch.zeros((d, n_cells, p), dtype=torch.int64)
    for c in chunks:
        hits[c.plane, c.cells, c.columns] += 1
    return hits


# -- the base interface ---------------------------------------------------------


def test_feature_chunk_validates():
    values = torch.zeros((2, 3), dtype=torch.complex64)
    FeatureChunk(0, slice(0, 2), slice(4, 7), values)
    for bad in (
        {"plane": -1},
        {"cells": slice(0, 3)},
        {"cells": slice(None, 2)},
        {"columns": slice(4, 7, 2)},
        {"values": values.to(torch.complex128)},
    ):
        fields = {
            "plane": 0,
            "cells": slice(0, 2),
            "columns": slice(4, 7),
            "values": values,
        }
        with pytest.raises(ValueError):
            FeatureChunk(**{**fields, **bad})


@pytest.mark.parametrize("engine", [FFTFeaturization, DirectFeaturization])
def test_inputs_are_validated(warp, engine):
    with pytest.raises(ValueError, match="exceeds the stored"):
        engine(warp, (0, 0, 5, 1))  # k_max = 4
    with pytest.raises(ValueError, match="extents"):
        engine(warp, (0, 0, 0, 1))
    f = engine(warp, RECT)
    x = planes()
    rows, cols = positions(5, 34, 31)
    for args, match in (
        ((x[0], rows, cols), "real floating"),
        ((x.to(torch.complex64), rows, cols), "real floating"),
        ((x[:, : BOX - 1], rows, cols), "smaller than the box"),
        ((x, rows.int(), cols), "int64"),
        ((x, rows, cols[:-1]), "differ in length"),
        ((x, rows + 34, cols), "rows must lie"),
        ((x, rows, cols - 32), "cols must lie"),
    ):
        with pytest.raises(ValueError, match=match):
            list(f.chunks(*args))


@pytest.mark.parametrize("engine", [FFTFeaturization, DirectFeaturization])
def test_no_positions(warp, engine):
    f = engine(warp, RECT)
    empty = torch.empty(0, dtype=torch.long)
    assert list(f.chunks(planes(), empty, empty)) == []
    assert f.featurize(planes(), empty, empty).shape == (2, 6, 0)


# -- FFT ------------------------------------------------------------------------------


def test_fft_matches_reference_in_the_callers_order(warp):
    x = planes()
    rows, cols = positions(40, 34, 31)
    f = FFTFeaturization(warp, RECT, cell_chunk=4)
    chunks = list(f.chunks(x, rows, cols))
    # one chunk per (cell chunk, plane), every position
    assert [(c.plane, c.cells) for c in chunks] == [
        (0, slice(0, 4)),
        (1, slice(0, 4)),
        (0, slice(4, 6)),
        (1, slice(4, 6)),
    ]
    assert bool((coverage(chunks, 2, 6, 40) == 1).all())
    assert_features(f.featurize(x, rows, cols), reference(warp, x, rows, cols))


def test_fft_region_is_the_bounding_box(warp):
    f = FFTFeaturization(warp, RECT)
    rows, cols = torch.tensor([3, 9, 5]), torch.tensor([2, 4, 7])
    origin, out, fft_shape = f.region(rows, cols)
    assert origin == (3, 2) and out == (7, 6)
    assert fft_shape[0] >= 7 + BOX - 1 and fft_shape[1] >= 6 + BOX - 1


def test_fft_dense_band_skips_the_gather(warp):
    """Every position of a band, row-major: the same features as a gathered order."""
    x = planes(d=1, h=40, w=31)
    r, c = torch.meshgrid(torch.arange(4, 13), torch.arange(0, 12), indexing="ij")
    rows, cols = r.reshape(-1), c.reshape(-1)
    f = FFTFeaturization(warp, RECT)
    dense = f.featurize(x, rows, cols)
    assert_features(dense, reference(warp, x, rows, cols))
    flipped = f.featurize(x, rows.flip(0), cols.flip(0))
    torch.testing.assert_close(flipped, dense.flip(-1))


def test_fft_builds_its_kernels_once(warp):
    """The featurization holds its rectangle's kernels across calls (images)."""
    built = count_kernel_builds(warp)
    f = FFTFeaturization(warp, RECT)
    x = planes(d=1)
    rows, cols = positions(10, 34, 31)
    first = f.featurize(x, rows, cols)
    second = f.featurize(planes(d=1, seed=5), rows, cols)
    assert built[0] == 6 and f.kernels.shape == (6, BOX, BOX)
    assert_features(first, reference(warp, x, rows, cols))
    assert not torch.allclose(first, second)


# -- direct ---------------------------------------------------------------------------


def test_direct_adjoint_matches_reference_without_kernels(warp):
    x = planes()
    rows, cols = positions(23, 34, 31)
    built = count_kernel_builds(warp)
    f = DirectFeaturization(warp, RECT, batch=7)
    assert isinstance(f.contraction, AdjointContraction)
    chunks = list(f.chunks(x, rows, cols))
    assert [c.columns for c in chunks[:4]] == [
        slice(0, 7),
        slice(7, 14),
        slice(14, 21),
        slice(21, 23),
    ]
    assert bool((coverage(chunks, 2, 6, 23) == 1).all())
    result = f.featurize(x, rows, cols)
    assert built[0] == 0  # no Cartesian kernel was built
    assert_features(result, reference(warp, x, rows, cols))


@pytest.mark.parametrize("resident", [True, False])
def test_direct_planar_matches_reference(warp, monkeypatch, resident):
    if not resident:
        monkeypatch.setattr(direct_module, "PLANAR_FRACTION", 0.0)
        monkeypatch.setattr(direct_module, "PLANAR_CHUNK_BYTES", 2 * 4 * BOX * BOX * 4)
    x = planes()
    rows, cols = positions(23, 34, 31)
    f = DirectFeaturization(warp, RECT, batch=10, contraction="planar")
    assert isinstance(f.contraction, PlanarContraction)
    result = f.featurize(x, rows, cols)
    assert (f.contraction.block is not None) is resident
    if not resident:
        assert f.contraction.cell_chunk == 4  # two chunks over the 6 cells
    assert_features(result, reference(warp, x, rows, cols))


def test_direct_falls_back_to_planar_without_an_adjoint(warp, monkeypatch):
    monkeypatch.setitem(warp.__dict__, "adjoint", None)
    f = DirectFeaturization(warp, RECT)
    assert isinstance(f.contraction, PlanarContraction)
    with pytest.raises(ValueError, match="no adjoint"):
        AdjointContraction(warp, f.rect)


def test_direct_on_boxes_the_size_of_the_template(warp):
    """A particle-stack-like image: the box is the whole image (one position)."""
    x = planes(d=3, h=BOX, w=BOX)
    zero = torch.zeros(1, dtype=torch.long)
    direct = DirectFeaturization(warp, RECT).featurize(x, zero, zero)
    assert_features(direct, reference(warp, x, zero, zero))
    fft = FFTFeaturization(warp, RECT).featurize(x, zero, zero)
    assert_features(fft, direct.cpu())


def test_engines_agree(warp):
    x = planes(d=2, h=48, w=45)
    rows, cols = positions(60, 48, 45, seed=3)
    fft = FFTFeaturization(warp, RECT).featurize(x, rows, cols)
    direct = DirectFeaturization(warp, RECT, batch=16).featurize(x, rows, cols)
    assert_features(direct, fft)


# -- CUDA ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("precision, tol", [("fp32", 1e-5), ("fp16", 2e-3)])
def test_cuda_engines_match_reference(synthetic_result, precision, tol):
    warp = make_warp(synthetic_result, device="cuda")
    x = planes(d=2, h=48, w=45)
    rows, cols = positions(50, 48, 45, seed=4)
    expected = reference(warp, x, rows, cols)
    fft = FFTFeaturization(warp, RECT).featurize(x.cuda(), rows.cuda(), cols.cuda())
    assert fft.is_cuda
    assert_features(fft, expected)
    for contraction in ("adjoint", "planar"):
        direct = DirectFeaturization(
            warp, RECT, precision=precision, contraction=contraction
        ).featurize(x, rows, cols)
        assert direct.is_cuda
        assert_features(direct, expected, tol=tol)
