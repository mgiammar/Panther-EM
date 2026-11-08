"""approximation.errors: the noise tables (fields, guard, I/O) and exact errors."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from noise_case import BAND, BOX, SHAPE, decompose, noise_case, prepare
from scipy.ndimage import gaussian_filter
from torch_fourier_filter.bandpass import low_pass_filter

from panther_em.approximation import Reconstructor
from panther_em.approximation.errors import (
    TABLE_NAMES,
    PerSpatialFrequencyErrorTable,
    RadialBins,
    compute_error_tables,
    compute_errors,
)
from panther_em.coordinates import OffsetPolarTransform, WarpConfig
from panther_em.decomposition import PolarProjectionDecomposer

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


# --------------------------------------------------------------------------- #
# compute_errors: the exact tier
# --------------------------------------------------------------------------- #


def _decompose(filters=None, *, normalize=True, n_or=6):
    rng = np.random.default_rng(0)
    volume = gaussian_filter(rng.standard_normal((BOX,) * 3), 2.0).astype(np.float32)
    transform = OffsetPolarTransform.from_image(
        image_shape=(BOX, BOX), num_angle=32, num_radius=12
    )
    result = PolarProjectionDecomposer(
        volume,
        np.linspace(0.0, 300.0, n_or, dtype=np.float32),
        np.linspace(10.0, 80.0, n_or, dtype=np.float32),
        coordinate_transform=transform,
        fourier_filters=filters,
    ).do_decomposition(eig_max=6, normalize_projections=normalize)
    return volume, Reconstructor(result)


def test_errors_share_the_tables_pass(whitening, decomposition):
    """The residual spectra are the noise tables' residual autocorrelation."""
    volume, template = decomposition
    rects = [(0, 0, 3, 4)]
    table = compute_error_tables(volume, template.reconstructor, rects, whitening)
    errors = compute_errors(volume, template.reconstructor, rects)
    torch.testing.assert_close(errors.frequencies, table.frequencies)
    torch.testing.assert_close(
        errors.residual_spectrum.float(), table.residual_autocorr, rtol=1e-4, atol=1e-6
    )
    assert errors.rectangles == table.rectangles and errors.warp == table.warp


def test_error_reductions(decomposition):
    volume, template = decomposition
    errors = compute_errors(volume, template.reconstructor, [(0, 0, 3, 4)])
    n = template.result.num_orientations
    assert errors.by_orientation().shape == (n,)
    assert errors.by_frequency().shape == errors.frequencies.shape
    assert errors.radii.shape == (errors.template_radial.shape[1],)

    res, tmp = errors.residual_radial.sum(1), errors.template_radial.sum(1)
    torch.testing.assert_close(errors.by_orientation(), torch.sqrt(res / tmp).float())
    assert errors.total() == pytest.approx(float(torch.sqrt(res.sum() / tmp.sum())))
    # Parseval: the rings hold the box's energy but for the corners past Nyquist
    ratio = errors.template_spectrum.sum(1) / tmp
    assert bool(((ratio > 0.9) & (ratio <= 1.0 + 1e-6)).all())

    # annuli outside the support hold no template energy: no ratio there
    by_radius = errors.by_radius()
    inside = errors.template_radial.sum(0) > 0
    assert bool(torch.isfinite(by_radius[inside]).all())
    assert bool(torch.isnan(by_radius[~inside]).all()) and bool((~inside).any())


def test_errors_without_a_filter():
    """No whitening needed: an unfiltered decomposition has unfiltered errors."""
    volume, reconstructor = _decompose()
    assert reconstructor.result.fourier_filters is None
    coarse = compute_errors(volume, reconstructor, [(0, 0, 2, 2)]).total()
    fine = compute_errors(volume, reconstructor, [(0, 0, 17, 6)]).total()
    assert 0 < fine < coarse < 1


def test_errors_use_each_rows_filter():
    """Rows of a second filter twice the first scale with it: the same errors."""
    grid = low_pass_filter(0.3, 0.05, (BOX, BOX), rfft=True, fftshift=False)
    volume, reconstructor = _decompose(torch.stack([grid, 2 * grid]), normalize=False)
    assert reconstructor.result.normalized_projections is False
    rho = compute_errors(volume, reconstructor, [(0, 0, 5, 3)]).by_orientation()
    torch.testing.assert_close(rho[:6], rho[6:], rtol=1e-4, atol=1e-6)
    assert float(rho.max()) < 1
