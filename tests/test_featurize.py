"""Featurizers vs the dense FFT reference ``compute_feature_stack``."""

from __future__ import annotations

import pytest
import torch
from conftest import read_group

from panther_em.inference.correlation import compute_feature_stack
from panther_em.inference.projection_reconstruction import ProjectionReconstructor
from panther_em.inference.search.basis import KernelBank
from panther_em.inference.search.featurize import (
    FFTCorrelationFeaturizer,
    ImageBatch,
    InnerProductFeaturizer,
    Tile,
    make_featurizer,
)
from panther_em.inference.search.store import FeatureStore
from panther_em.inference.search.types import ImageSet, PixelSelection, SearchGeometry

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array_wrap__ must accept.*:DeprecationWarning",
    "ignore:.*__array__ implementation doesn't accept.*:DeprecationWarning",
)

RECT = (1, 0, 2, 3)  # k in [1, 3), m in [0, 3): 6 cells


@pytest.fixture
def bank(synthetic_result):
    recon = ProjectionReconstructor(synthetic_result(), device="cpu")
    return KernelBank(recon)


def _images(b=2, h=30, w=28, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(b, h, w, generator=gen)


def _cells(rect):
    k0, m0, ke, me = rect
    return torch.stack(
        torch.meshgrid(
            torch.arange(k0, k0 + ke), torch.arange(m0, m0 + me), indexing="ij"
        ),
        dim=-1,
    ).reshape(-1, 2)


def _reference(bank, images_bhw, rect):
    """``(B, n_cells, out_h, out_w)`` valid correlation from the dense reference."""
    kernels = bank.spatial(_cells(rect))
    return compute_feature_stack(images_bhw, kernels)


def _run(
    featurizer,
    images,
    geo,
    pixels,
    rect,
    *,
    block_pixels=7,
    chunk=4,
    order="pixel_major",
):
    sched = featurizer.schedule(geo, pixels)
    store = FeatureStore(
        geo,
        pixels,
        sched.column_index,
        block_pixels=block_pixels,
        n_defocus=images.n_defocus,
        device="cpu",
    )
    group = store.add_group(rect, "fp32")
    featurizer.produce(
        images, group, store.writer(group), sched, chunk=chunk, order=order
    )
    return sched, store, group


def _expected_columns(ref, geo, column_index, plane_ref=None):
    """Gather reference features ``(P, num_k, num_m)`` at the compact columns."""
    b_, n, oh, ow = ref.shape
    flat = ref.permute(0, 2, 3, 1).reshape(b_ * oh * ow, n)  # (B*P, cells)
    return flat[column_index]


def test_single_tile_matches_reference_all_pixels(bank):
    images = ImageSet.from_stack(_images())
    geo = SearchGeometry.from_images(images, bank.template_shape)
    pixels = PixelSelection.all(geo)
    feat = FFTCorrelationFeaturizer(bank, tile=1024)
    sched, store, group = _run(feat, images, geo, pixels, RECT)
    # tile side is clamped to min(out_h, out_w) = 9 on an 11 x 9 grid -> 2 tiles/image,
    # whose owned rows stay row-major, so the compact order equals the sorted order
    assert sched.tile == 9 and len(sched.items) == 4
    assert torch.equal(sched.column_index, pixels.index)
    ref = _reference(bank, images.images, RECT)
    expected = _expected_columns(ref, geo, sched.column_index).reshape(-1, 2, 3)
    torch.testing.assert_close(
        read_group(store, group, 0), expected, rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("order", ["pixel_major", "cell_major"])
def test_tiled_masked_selection_matches_reference(bank, order):
    images = ImageSet.from_stack(_images(b=2, h=31, w=29))
    geo = SearchGeometry.from_images(images, bank.template_shape)  # grid (12, 10)
    mask = torch.zeros(geo.dense_shape, dtype=torch.bool)
    mask[0, ::3, ::2] = True
    mask[1, 5:9, 3:8] = True
    mask[1, 11, 9] = True  # last pixel: lives in a clamped edge tile
    pixels = PixelSelection.from_mask(geo, mask)
    feat = FFTCorrelationFeaturizer(bank, tile=5)
    sched, store, group = _run(
        feat, images, geo, pixels, RECT, block_pixels=6, chunk=5, order=order
    )
    assert sched.tile == 5 and sched.fft_shape == (24, 24)
    assert torch.equal(torch.sort(sched.column_index).values, pixels.index)
    for item in sched.items:
        assert isinstance(item, Tile) and item.row0 <= geo.grid[0] - 5
    ref = _reference(bank, images.images, RECT)
    expected = _expected_columns(ref, geo, sched.column_index).reshape(-1, 2, 3)
    torch.testing.assert_close(
        read_group(store, group, 0), expected, rtol=1e-4, atol=1e-5
    )


def test_filter_planes_match_prefiltered_reference(bank):
    raw = _images(b=2, h=30, w=28)
    filters = torch.rand(3, 30, 15)  # D = 3 shared planes
    images = ImageSet.from_stack(raw, filters=filters)
    geo = SearchGeometry.from_images(images, bank.template_shape)
    pixels = PixelSelection.all(geo)
    feat = FFTCorrelationFeaturizer(bank, tile=6)
    sched, store, group = _run(feat, images, geo, pixels, RECT)
    for d in range(3):
        pre = torch.fft.irfft2(torch.fft.rfft2(raw) * filters[d], s=(30, 28))
        ref = _reference(bank, pre, RECT)
        expected = _expected_columns(ref, geo, sched.column_index).reshape(-1, 2, 3)
        torch.testing.assert_close(
            read_group(store, group, d), expected, rtol=1e-4, atol=1e-5
        )

    # a group holding only planes 2 and 0 (stored as (0, 2)) gets exactly those planes
    partial = store.add_group((0, 0, 1, 2), "fp32", planes=(2, 0))
    feat.produce(images, partial, store.writer(partial), sched, chunk=4)
    assert partial.planes == (0, 2)
    for local, d in enumerate((0, 2)):
        pre = torch.fft.irfft2(torch.fft.rfft2(raw) * filters[d], s=(30, 28))
        ref = _reference(bank, pre, (0, 0, 1, 2))
        expected = _expected_columns(ref, geo, sched.column_index).reshape(-1, 1, 2)
        torch.testing.assert_close(
            read_group(store, partial, local), expected, rtol=1e-4, atol=1e-5
        )

    per_image = ImageSet.from_stack(raw, filters=torch.rand(2, 2, 30, 15))
    sched, store, group = _run(feat, per_image, geo, pixels, RECT)
    for d in range(2):
        pre = torch.stack(
            [
                torch.fft.irfft2(
                    torch.fft.rfft2(raw[b]) * per_image.filters[b, d], s=(30, 28)
                )
                for b in range(2)
            ]
        )
        ref = _reference(bank, pre, RECT)
        expected = _expected_columns(ref, geo, sched.column_index).reshape(-1, 2, 3)
        torch.testing.assert_close(
            read_group(store, group, d), expected, rtol=1e-4, atol=1e-5
        )


def test_inner_product_matches_grid_center_pixel(bank):
    boxes = _images(b=5, h=20, w=20)
    images = ImageSet.from_boxes(boxes)
    geo = SearchGeometry.from_images(images, bank.template_shape)
    assert geo.mode == "center"
    feat = make_featurizer(geo, bank, image_batch=2)
    assert isinstance(feat, InnerProductFeaturizer)
    pixels = PixelSelection.all(geo)
    sched, store, group = _run(feat, images, geo, pixels, RECT, block_pixels=2)
    assert len(sched.items) == 3 and all(isinstance(i, ImageBatch) for i in sched.items)
    ref = _reference(bank, boxes, RECT)  # (B, cells, 1, 1)
    expected = ref[:, :, 0, 0].reshape(5, 2, 3)
    torch.testing.assert_close(
        read_group(store, group, 0), expected, rtol=1e-4, atol=1e-5
    )

    # a larger box with center_only=True equals the grid path's center pixel
    big = _images(b=3, h=25, w=24)
    centred = ImageSet.from_boxes(big, center_only=True)
    geo_c = SearchGeometry.from_images(centred, bank.template_shape)
    sched, store, group = _run(
        make_featurizer(geo_c, bank), centred, geo_c, PixelSelection.all(geo_c), RECT
    )
    ref = _reference(bank, big, RECT)  # (B, cells, 6, 5)
    expected = ref[:, :, (6 - 1) // 2, (5 - 1) // 2].reshape(3, 2, 3)
    torch.testing.assert_close(
        read_group(store, group, 0), expected, rtol=1e-4, atol=1e-5
    )

    # with a filter plane
    filt = torch.rand(1, 20, 11)
    filtered = ImageSet.from_boxes(boxes, filters=filt)
    sched, store, group = _run(make_featurizer(geo, bank), filtered, geo, pixels, RECT)
    pre = torch.fft.irfft2(torch.fft.rfft2(boxes) * filt[0], s=(20, 20))
    expected = _reference(bank, pre, RECT)[:, :, 0, 0].reshape(5, 2, 3)
    torch.testing.assert_close(
        read_group(store, group, 0), expected, rtol=1e-4, atol=1e-5
    )


def test_featurizer_mode_mismatch_raises(bank):
    images = ImageSet.from_boxes(_images(b=2, h=20, w=20))
    geo = SearchGeometry.from_images(images, bank.template_shape)
    with pytest.raises(ValueError):
        FFTCorrelationFeaturizer(bank).schedule(geo, PixelSelection.all(geo))
    grid = SearchGeometry.from_images(
        ImageSet.from_stack(_images()), bank.template_shape
    )
    with pytest.raises(ValueError):
        InnerProductFeaturizer(bank).schedule(grid, PixelSelection.all(grid))
