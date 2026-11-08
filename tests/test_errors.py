"""approximation.errors: the table fields, the decomposition guard and I/O."""

from __future__ import annotations

import pytest
import torch
from noise_case import BAND, BOX, SHAPE, decompose, noise_case, prepare
from torch_fourier_filter.bandpass import low_pass_filter

from panther_em.approximation import Reconstructor
from panther_em.approximation.errors import (
    TABLE_NAMES,
    PerSpatialFrequencyErrorTable,
    RadialBins,
    compute_error_tables,
)
from panther_em.coordinates import WarpConfig

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array_wrap__ must accept.*:DeprecationWarning",
    "ignore:.*__array__ implementation doesn't accept.*:DeprecationWarning",
)


@pytest.fixture(scope="module")
def whitening():
    return noise_case().whitening


@pytest.fixture(scope="module")
def decomposition():
    case = noise_case()
    return case.volume, case.template


@pytest.mark.parametrize("rects", [[(0, 0, 3, 4)], [(0, 0, 2, 3), (2, 0, 2, 2)]])
def test_table_fields(whitening, decomposition, rects):
    volume, template = decomposition
    table = compute_error_tables(volume, template.reconstructor, rects, whitening)
    assert table.n_templates == 6 and table.template_shape == (BOX, BOX)
    for name in TABLE_NAMES:
        t = getattr(table, name)
        assert t.shape == (6, table.n_bins) and bool(torch.isfinite(t).all())
    assert bool((table.reconstruction_autocorr > 0).any())
    assert bool((table.residual_autocorr >= 0).all())
    assert torch.equal(table.support, template.reconstructor.support)
    assert table.rectangles == tuple(rects)
    assert table.warp == template.warp
    torch.testing.assert_close(
        table.template_filter, whitening.template_filter((BOX, BOX))
    )
    summary = table.summary()
    assert summary["rectangles"] == [list(r) for r in rects]
    assert summary["warp"]["order"] == template.warp.order


def test_check(whitening, decomposition, synthetic_result):
    volume, template = decomposition
    table = compute_error_tables(
        volume, template.reconstructor, [(0, 0, 3, 4)], whitening
    )
    table.check(template.reconstructor)
    result = template.result
    with pytest.raises(ValueError, match="warp"):
        table.check(Reconstructor(result, WarpConfig(order=3)))
    with pytest.raises(ValueError, match="box"):
        table.check(Reconstructor(synthetic_result(box=20)))
    with pytest.raises(ValueError, match="template rows"):
        table.check(Reconstructor(synthetic_result(box=BOX, n_orientations=4)))
    with pytest.raises(ValueError, match="beyond"):
        table.check(Reconstructor(synthetic_result(box=BOX, k_max=2)))
    with pytest.raises(ValueError, match="no Fourier filter"):
        table.check(Reconstructor(synthetic_result(box=BOX)))
    band = low_pass_filter(*BAND, (BOX, BOX), rfft=True, fftshift=False)[None]
    _, band_only = decompose(band)
    with pytest.raises(ValueError, match="another Fourier filter"):
        table.check(band_only.reconstructor)


def test_guards(whitening, decomposition):
    volume, template = decomposition
    rects = [(0, 0, 3, 4)]
    _, unfiltered = decompose(None)
    with pytest.raises(ValueError, match="no Fourier filter"):
        compute_error_tables(volume, unfiltered.reconstructor, rects, whitening)
    band = low_pass_filter(*BAND, (BOX, BOX), rfft=True, fftshift=False)[None]
    _, band_only = decompose(band)  # the old pipeline's low-pass without W_avg
    with pytest.raises(ValueError, match="another W_avg or band"):
        compute_error_tables(volume, band_only.reconstructor, rects, whitening)
    with pytest.raises(ValueError, match="beyond the stored"):
        compute_error_tables(volume, template.reconstructor, [(0, 0, 3, 9)], whitening)
    with pytest.raises(ValueError, match="volume must be"):
        compute_error_tables(volume[:-1], template.reconstructor, rects, whitening)


def test_round_trip_and_validation(whitening, decomposition, tmp_path):
    volume, template = decomposition
    table = compute_error_tables(
        volume, template.reconstructor, [(0, 0, 3, 4)], whitening
    )
    path = tmp_path / "table.h5"
    table.save(path)
    back = PerSpatialFrequencyErrorTable.load(path)
    for name in ("frequencies", *TABLE_NAMES, "template_filter"):
        torch.testing.assert_close(getattr(back, name), getattr(table, name))
    assert torch.equal(back.support, table.support)
    assert back.rectangles == table.rectangles and back.pad == table.pad
    assert back.warp == table.warp

    fields = {
        name: getattr(table, name)
        for name in ("frequencies", *TABLE_NAMES, "support", "template_filter")
    }
    fields["rectangles"] = table.rectangles
    fields["warp"] = table.warp
    for bad in (
        {"frequencies": table.frequencies.double()},
        {"residual_autocorr": table.residual_autocorr[:-1]},
        {"support": table.support.float()},
        {"template_filter": table.template_filter[:, :-1]},
        {"rectangles": ()},
        {"rectangles": ((0, 0, 0, 4),)},
        {"warp": None},
        {"pad": 0},
    ):
        with pytest.raises(ValueError):
            PerSpatialFrequencyErrorTable(**{**fields, **bad})


def test_image_bins_are_shared_and_reused(whitening, decomposition, monkeypatch):
    volume, template = decomposition
    table = compute_error_tables(
        volume, template.reconstructor, [(0, 0, 3, 4)], whitening
    )
    bins = table.image_bins(SHAPE)
    assert table.image_bins(SHAPE, "cpu") is bins
    assert table.image_bins((SHAPE[0], SHAPE[1] - 2)) is not bins
    ones = torch.ones((2, SHAPE[0], SHAPE[1] // 2 + 1), dtype=torch.float64)
    torch.testing.assert_close(
        bins.ring_mean(ones), torch.ones(2, table.n_bins).double()
    )
    with pytest.raises(ValueError, match="no frequency"):
        table.image_bins((8, 8))  # coarser than the padded template box's axis

    # preparing more micrographs builds no further bins
    built = []
    create = RadialBins.create
    monkeypatch.setattr(
        RadialBins,
        "create",
        classmethod(lambda cls, *a, **k: built.append(a) or create(*a, **k)),
    )
    for name in ("m0", "m1", "m2"):
        prepare(whitening, table, name)
    assert built == []
