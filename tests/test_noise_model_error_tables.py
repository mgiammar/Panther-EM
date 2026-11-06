"""noise_model error tables: parity with RadialTables, the guards and the scales."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter
from torch_fourier_filter.bandpass import low_pass_filter
from torch_grid_utils.fftfreq_grid import fftfreq_grid

from panther_em.coordinates import OffsetPolarTransform
from panther_em.decomposition import PolarProjectionDecomposer
from panther_em.inference.search.hypotheses import HypothesisSet, HypothesisSpace
from panther_em.inference.search.noise import RadialTables, support_mask
from panther_em.inference.search.noise_model.image_set import ImageSet
from panther_em.inference.search.noise_model.spatial_frequency_error import (
    TABLE_NAMES,
    PerSpatialFrequencyErrorTable,
    RadialBins,
    compute_error_tables,
)
from panther_em.inference.search.noise_model.whitening import WhiteningFilter
from panther_em.inference.template import KernelConfig, Template
from panther_em.spectra import WhiteningFilter as LegacyWhiteningFilter

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array_wrap__ must accept.*:DeprecationWarning",
    "ignore:.*__array__ implementation doesn't accept.*:DeprecationWarning",
)

SHAPE = (96, 96)
BOX = 24
BAND = (0.35, 0.05)
PIXEL_SIZE = 1.5
LEGACY = {  # new name -> RadialTables name
    "reconstruction_autocorr": "kept",
    "residual_autocorr": "residual",
    "reconstruction_residual_corr": "cross",
    "signal_reconstruction_corr": "kept_signal",
    "signal_residual_corr": "residual_signal",
}


def coloured(shape, seed, scale=1.0):
    x = torch.randn(shape, generator=torch.Generator().manual_seed(seed))
    r = fftfreq_grid(image_shape=shape, rfft=True, fftshift=False, norm=True)
    envelope = 0.2 + torch.exp(-((r / 0.15) ** 2))
    return scale * torch.fft.irfft2(torch.fft.rfft2(x) * envelope, s=shape)


def micrographs():
    return [(f"m{i}", coloured(SHAPE, 10 + i, s)) for i, s in enumerate((1, 0.7, 1.4))]


def ctf_planes(shape, n=3):
    r = fftfreq_grid(image_shape=shape, rfft=True, fftshift=False, norm=True)
    return torch.stack([torch.cos((60 + 40 * d) * r**2 + 0.1) for d in range(n)])


def decompose(filters):
    rng = np.random.default_rng(0)
    volume = gaussian_filter(rng.standard_normal((BOX,) * 3), 2.0).astype(np.float32)
    phi = np.linspace(0.0, 300.0, 6, dtype=np.float32)
    theta = np.linspace(10.0, 80.0, 6, dtype=np.float32)
    transform = OffsetPolarTransform.from_image(
        image_shape=(BOX, BOX), num_angle=32, num_radius=12
    )
    result = PolarProjectionDecomposer(
        volume, phi, theta, coordinate_transform=transform, fourier_filters=filters
    ).do_decomposition(eig_max=6, normalize_projections=True, svd_method="dense")
    return volume, Template(result, device="cpu")


@pytest.fixture(scope="module")
def whitening():
    return WhiteningFilter.from_images(
        micrographs(), band_cutoff=BAND[0], band_falloff=BAND[1]
    )


@pytest.fixture(scope="module")
def decomposition(whitening):
    return decompose(whitening.template_filter((BOX, BOX))[None])


@pytest.mark.parametrize("rects", [[(0, 0, 3, 4)], [(0, 0, 2, 3), (2, 0, 2, 2)]])
def test_matches_radial_tables(whitening, decomposition, rects):
    """Same tables as RadialTables.build(volume=, whitening=W_avg)."""
    volume, template = decomposition
    table = compute_error_tables(volume, template, rects, whitening)
    legacy_filter = LegacyWhiteningFilter.from_images(micrographs())
    legacy = RadialTables.build(template, rects, volume=volume, whitening=legacy_filter)
    torch.testing.assert_close(table.frequencies, legacy.freq)
    for name, old in LEGACY.items():
        new, ref = getattr(table, name), getattr(legacy, old)
        scale = ref.abs().max()
        torch.testing.assert_close(new, ref, rtol=1e-4, atol=1e-5 * float(scale))
    assert torch.equal(table.support, legacy.support)
    assert torch.equal(table.support, template.warp.support)
    assert table.rectangles == tuple(rects) and table.n_templates == 6
    assert table.kernel_config == template.kernel_config
    torch.testing.assert_close(
        table.template_filter, whitening.template_filter((BOX, BOX))
    )


def test_template_support_is_the_legacy_mask(decomposition):
    _, template = decomposition
    legacy = support_mask(template.warp.transform, template.kernel_config)
    assert torch.equal(template.warp.support, legacy)


def test_check_template(whitening, decomposition, synthetic_result):
    volume, template = decomposition
    table = compute_error_tables(volume, template, [(0, 0, 3, 4)], whitening)
    table.check_template(template)
    result = template.result
    with pytest.raises(ValueError, match="kernel warp"):
        table.check_template(
            Template(result, device="cpu", kernel=KernelConfig(order=3))
        )
    with pytest.raises(ValueError, match="box"):
        table.check_template(Template(synthetic_result(box=20), device="cpu"))
    with pytest.raises(ValueError, match="template rows"):
        table.check_template(
            Template(synthetic_result(box=BOX, n_orientations=4), device="cpu")
        )
    with pytest.raises(ValueError, match="beyond"):
        table.check_template(Template(synthetic_result(box=BOX, k_max=2), device="cpu"))
    with pytest.raises(ValueError, match="no Fourier filter"):
        table.check_template(Template(synthetic_result(box=BOX), device="cpu"))
    band = low_pass_filter(*BAND, (BOX, BOX), rfft=True, fftshift=False)[None]
    _, band_only = decompose(band)
    with pytest.raises(ValueError, match="another Fourier filter"):
        table.check_template(band_only)


def test_guards(whitening, decomposition):
    volume, template = decomposition
    rects = [(0, 0, 3, 4)]
    _, unfiltered = decompose(None)
    with pytest.raises(ValueError, match="no Fourier filter"):
        compute_error_tables(volume, unfiltered, rects, whitening)
    band = low_pass_filter(*BAND, (BOX, BOX), rfft=True, fftshift=False)[None]
    _, band_only = decompose(band)  # the old pipeline's low-pass without W_avg
    with pytest.raises(ValueError, match="another W_avg or band"):
        compute_error_tables(volume, band_only, rects, whitening)
    with pytest.raises(ValueError, match="beyond the stored"):
        compute_error_tables(volume, template, [(0, 0, 3, 9)], whitening)
    with pytest.raises(ValueError, match="volume must be"):
        compute_error_tables(volume[:-1], template, rects, whitening)


def prepare(whitening, table, name="m2"):
    return ImageSet.from_micrograph(
        dict(micrographs())[name],
        name,
        whitening,
        ctf_planes(SHAPE),
        table,
        pixel_size=PIXEL_SIZE,
    )


def test_scales_match_radial_tables(whitening, decomposition):
    """table.scales(images) == RadialTables.scales(images' spectra)."""
    volume, template = decomposition
    rects = [(0, 0, 3, 4)]
    table = compute_error_tables(volume, template, rects, whitening)
    legacy = RadialTables.build(
        template,
        rects,
        volume=volume,
        whitening=LegacyWhiteningFilter.from_images(micrographs()),
    )
    images = prepare(whitening, table)
    spectra = images.as_search_images().spectra
    hyps = HypothesisSet.create(
        HypothesisSpace(3, 1, 6),
        torch.tensor([0, 2]),
        torch.tensor([1]),
        hyp_batch=4,
        n_psi=16,
    )
    for kwargs in ({}, {"b_factor": 40.0}, {"hyps": hyps}):
        new = table.scales(images, **kwargs)
        old = legacy.scales(spectra, **kwargs)
        assert new.q_kept.shape == (1, 3, 6)
        for name in ("q_kept", "q_residual", "q_cross", "q_perp", "qbar_perp"):
            ref = getattr(old, name)
            torch.testing.assert_close(
                getattr(new, name), ref, rtol=1e-4, atol=1e-6 * float(ref.abs().max())
            )
        torch.testing.assert_close(new.lam, old.lam, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(new.rho, old.rho, rtol=1e-4, atol=1e-5)
        assert new.rectangles == old.rectangles and new.b_factor == old.b_factor
    sub = table.scales(images, hyps=hyps)
    torch.testing.assert_close(sub.qbar_perp, sub.q_perp[0, 1, [0, 2]].mean()[None])


def test_scales_need_the_tables_axis(whitening, decomposition):
    volume, template = decomposition
    rects = [(0, 0, 3, 4)]
    table = compute_error_tables(volume, template, rects, whitening)
    padded = compute_error_tables(volume, template, rects, whitening, pad=3)
    images = prepare(whitening, table)
    with pytest.raises(ValueError, match="another frequency axis"):
        padded.scales(images)


def test_round_trip_and_validation(whitening, decomposition, tmp_path):
    volume, template = decomposition
    table = compute_error_tables(volume, template, [(0, 0, 3, 4)], whitening)
    path = tmp_path / "table.h5"
    table.save(path)
    back = PerSpatialFrequencyErrorTable.load(path)
    for name in ("frequencies", *TABLE_NAMES, "template_filter"):
        torch.testing.assert_close(getattr(back, name), getattr(table, name))
    assert torch.equal(back.support, table.support)
    assert back.rectangles == table.rectangles and back.pad == table.pad
    assert back.kernel_config == table.kernel_config

    fields = {
        name: getattr(table, name)
        for name in ("frequencies", *TABLE_NAMES, "support", "template_filter")
    }
    fields["rectangles"] = table.rectangles
    fields["kernel_config"] = table.kernel_config
    for bad in (
        {"frequencies": table.frequencies.double()},
        {"residual_autocorr": table.residual_autocorr[:-1]},
        {"support": table.support.float()},
        {"template_filter": table.template_filter[:, :-1]},
        {"rectangles": ()},
        {"rectangles": ((0, 0, 0, 4),)},
        {"kernel_config": None},
        {"pad": 0},
    ):
        with pytest.raises(ValueError):
            PerSpatialFrequencyErrorTable(**{**fields, **bad})


def test_image_bins_are_shared_and_reused(whitening, decomposition, monkeypatch):
    volume, template = decomposition
    table = compute_error_tables(volume, template, [(0, 0, 3, 4)], whitening)
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
