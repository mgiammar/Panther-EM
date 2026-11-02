"""Featurization functions vs the dense FFT reference ``compute_feature_stack``."""

from __future__ import annotations

import pytest
import torch
from conftest import read_group

from panther_em.inference.correlation import compute_feature_stack
from panther_em.inference.projection_reconstruction import ProjectionReconstructor
from panther_em.inference.search.template import KernelBank
from panther_em.inference.search.featurize import (
    DIRECT_MAX_BOX_EXCESS,
    DIRECT_MAX_DENSITY,
    choose_mode,
    featurize,
    featurize_direct,
    featurize_grid,
    schedule_tiles,
)
from panther_em.inference.search.features import FeatureStore
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
    bank,
    images,
    geo,
    pixels,
    rect,
    *,
    block_pixels=7,
    chunk=4,
    order="pixel_major",
    tile=1024,
    image_batch=1024,
    mode=None,
    column_order=None,
    planar_fraction=None,
):
    sched = schedule_tiles(
        geo, pixels, tile=tile, image_batch=image_batch, mode=mode, order=column_order
    )
    store = FeatureStore(
        geo,
        pixels,
        sched.column_index,
        block_pixels=block_pixels,
        n_defocus=images.n_defocus,
        device="cpu",
    )
    group = store.add_group(rect, "fp32")
    if planar_fraction is not None:
        featurize_direct(
            bank, images, group, store.writer(group), sched, chunk=chunk,
            planar_fraction=planar_fraction,
        )
    else:
        featurize(bank, images, group, store.writer(group), sched, chunk=chunk, order=order)
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
    sched, store, group = _run(bank, images, geo, pixels, RECT, tile=1024, mode="fft")
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
    sched, store, group = _run(
        bank, images, geo, pixels, RECT, block_pixels=6, chunk=5, order=order, tile=5,
        mode="fft",
    )
    assert sched.tile == 5 and sched.fft_shape == (24, 24)
    assert torch.equal(torch.sort(sched.column_index).values, pixels.index)
    assert sched.mode == "fft" and all(item.row0 <= geo.grid[0] - 5 for item in sched.items)
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
    sched, store, group = _run(bank, images, geo, pixels, RECT, tile=6, mode="fft")
    for d in range(3):
        pre = torch.fft.irfft2(torch.fft.rfft2(raw) * filters[d], s=(30, 28))
        ref = _reference(bank, pre, RECT)
        expected = _expected_columns(ref, geo, sched.column_index).reshape(-1, 2, 3)
        torch.testing.assert_close(
            read_group(store, group, d), expected, rtol=1e-4, atol=1e-5
        )

    # a group holding only planes 2 and 0 (stored as (0, 2)) gets exactly those planes
    partial = store.add_group((0, 0, 1, 2), "fp32", planes=(2, 0))
    featurize_grid(bank, images, partial, store.writer(partial), sched, chunk=4)
    assert partial.planes == (0, 2)
    for local, d in enumerate((0, 2)):
        pre = torch.fft.irfft2(torch.fft.rfft2(raw) * filters[d], s=(30, 28))
        ref = _reference(bank, pre, (0, 0, 1, 2))
        expected = _expected_columns(ref, geo, sched.column_index).reshape(-1, 1, 2)
        torch.testing.assert_close(
            read_group(store, partial, local), expected, rtol=1e-4, atol=1e-5
        )

    per_image = ImageSet.from_stack(raw, filters=torch.rand(2, 2, 30, 15))
    sched, store, group = _run(bank, per_image, geo, pixels, RECT, tile=6, mode="fft")
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
    pixels = PixelSelection.all(geo)
    sched, store, group = _run(bank, images, geo, pixels, RECT, block_pixels=2, image_batch=2)
    assert sched.mode == "direct" and len(sched.items) == 3
    assert all(i.images is not None and i.rows is not None for i in sched.items)
    ref = _reference(bank, boxes, RECT)  # (B, cells, 1, 1)
    expected = ref[:, :, 0, 0].reshape(5, 2, 3)
    torch.testing.assert_close(
        read_group(store, group, 0), expected, rtol=1e-4, atol=1e-5
    )

    # a larger box with center_only=True equals the grid path's center pixel
    big = _images(b=3, h=25, w=24)
    centred = ImageSet.from_boxes(big, center_only=True)
    geo_c = SearchGeometry.from_images(centred, bank.template_shape)
    sched, store, group = _run(bank, centred, geo_c, PixelSelection.all(geo_c), RECT)
    ref = _reference(bank, big, RECT)  # (B, cells, 6, 5)
    expected = ref[:, :, (6 - 1) // 2, (5 - 1) // 2].reshape(3, 2, 3)
    torch.testing.assert_close(
        read_group(store, group, 0), expected, rtol=1e-4, atol=1e-5
    )

    # with a filter plane
    filt = torch.rand(1, 20, 11)
    filtered = ImageSet.from_boxes(boxes, filters=filt)
    sched, store, group = _run(bank, filtered, geo, pixels, RECT)
    pre = torch.fft.irfft2(torch.fft.rfft2(boxes) * filt[0], s=(20, 20))
    expected = _reference(bank, pre, RECT)[:, :, 0, 0].reshape(5, 2, 3)
    torch.testing.assert_close(
        read_group(store, group, 0), expected, rtol=1e-4, atol=1e-5
    )


def test_featurize_mode_mismatch_raises(bank):
    images = ImageSet.from_boxes(_images(b=2, h=20, w=20))
    geo = SearchGeometry.from_images(images, bank.template_shape)
    center = schedule_tiles(geo, PixelSelection.all(geo))
    grid_images = ImageSet.from_stack(_images())
    grid_geo = SearchGeometry.from_images(grid_images, bank.template_shape)
    grid = schedule_tiles(grid_geo, PixelSelection.all(grid_geo), mode="fft")
    assert center.mode == "direct" and grid.mode == "fft" and center.fft_shape is None
    store = FeatureStore(
        geo, PixelSelection.all(geo), center.column_index, block_pixels=4, n_defocus=1, device="cpu"
    )
    group = store.add_group(RECT, "fp32")
    with pytest.raises(ValueError):
        featurize_grid(bank, images, group, store.writer(group), center)
    with pytest.raises(ValueError):
        featurize_direct(bank, grid_images, group, store.writer(group), grid)
    with pytest.raises(ValueError):  # the FFT engine has no center-geometry form
        schedule_tiles(geo, PixelSelection.all(geo), mode="fft")


# --------------------------------------------------------------------------- #
# Direct engine on grid images (sparse selections, slightly larger boxes)
# --------------------------------------------------------------------------- #


def _sparse_case(bank, b=2, h=60, w=58, n_selected=40, filters=None, seed=1):
    """Images well above the box-excess rule; ``n_selected`` random grid pixels."""
    images = ImageSet.from_stack(_images(b=b, h=h, w=w, seed=seed), filters=filters)
    geo = SearchGeometry.from_images(images, bank.template_shape)
    assert geo.image_shape[0] - geo.template_shape[0] > DIRECT_MAX_BOX_EXCESS
    gen = torch.Generator().manual_seed(seed)
    index = torch.randperm(geo.n_pixels, generator=gen)[:n_selected].sort().values
    return images, geo, PixelSelection.create(index, geo)


def test_choose_mode_rules(bank):
    images, geo, sparse = _sparse_case(bank, n_selected=40)  # 40 / 3198 = 1.25 %
    assert choose_mode(geo, sparse) == "direct"
    dense = PixelSelection.all(geo)
    assert choose_mode(geo, dense) == "fft"
    at_rule = PixelSelection.create(
        torch.arange(int(DIRECT_MAX_DENSITY * geo.n_pixels)), geo
    )
    assert choose_mode(geo, at_rule) == "direct"
    small = ImageSet.from_stack(_images(b=2, h=30, w=28))  # 10 px over the 20 box
    small_geo = SearchGeometry.from_images(small, bank.template_shape)
    assert choose_mode(small_geo, PixelSelection.all(small_geo)) == "direct"
    boxes = ImageSet.from_boxes(_images(b=3, h=20, w=20))
    box_geo = SearchGeometry.from_images(boxes, bank.template_shape)
    assert choose_mode(box_geo, PixelSelection.all(box_geo)) == "direct"
    assert schedule_tiles(geo, sparse).mode == "direct"
    assert schedule_tiles(geo, dense).mode == "fft"


@pytest.mark.parametrize("planar_fraction", [None, 0.0])
def test_direct_sparse_selection_matches_reference(bank, planar_fraction):
    images, geo, pixels = _sparse_case(bank, n_selected=37)
    sched, store, group = _run(
        bank, images, geo, pixels, RECT, block_pixels=8, image_batch=10,
        planar_fraction=planar_fraction,
    )
    assert sched.mode == "direct" and len(sched.items) == 4
    assert torch.equal(sched.column_index, pixels.index)
    ref = _reference(bank, images.images, RECT)
    expected = _expected_columns(ref, geo, sched.column_index).reshape(-1, 2, 3)
    torch.testing.assert_close(
        read_group(store, group, 0), expected, rtol=1e-4, atol=1e-5
    )


def test_direct_filter_planes_shared_and_per_image(bank):
    raw = _images(b=2, h=60, w=58, seed=2)
    shared = torch.rand(3, 60, 30)
    images, geo, pixels = _sparse_case(bank, n_selected=30, filters=shared, seed=2)
    sched, store, group = _run(bank, images, geo, pixels, RECT, block_pixels=8)
    assert sched.mode == "direct"
    for d in range(3):
        pre = torch.fft.irfft2(torch.fft.rfft2(raw) * shared[d], s=(60, 58))
        expected = _expected_columns(
            _reference(bank, pre, RECT), geo, sched.column_index
        ).reshape(-1, 2, 3)
        torch.testing.assert_close(
            read_group(store, group, d), expected, rtol=1e-4, atol=1e-5
        )
    per_image = torch.rand(2, 2, 60, 30)
    images, geo, pixels = _sparse_case(bank, n_selected=30, filters=per_image, seed=2)
    sched, store, group = _run(bank, images, geo, pixels, RECT, block_pixels=8)
    for d in range(2):
        pre = torch.stack(
            [
                torch.fft.irfft2(torch.fft.rfft2(raw[b]) * per_image[b, d], s=(60, 58))
                for b in range(2)
            ]
        )
        expected = _expected_columns(
            _reference(bank, pre, RECT), geo, sched.column_index
        ).reshape(-1, 2, 3)
        torch.testing.assert_close(
            read_group(store, group, d), expected, rtol=1e-4, atol=1e-5
        )


def test_direct_slightly_larger_boxes_every_offset(bank):
    # a stack 4 px over the box per side: every valid offset through the direct engine
    stack = ImageSet.from_stack(_images(b=3, h=24, w=24, seed=3))
    geo = SearchGeometry.from_images(stack, bank.template_shape)
    assert geo.mode == "grid" and geo.grid == (5, 5)
    pixels = PixelSelection.all(geo)
    sched, store, group = _run(bank, stack, geo, pixels, RECT, block_pixels=25, image_batch=40)
    assert sched.mode == "direct" and len(sched.items) == 2
    ref = _reference(bank, stack.images, RECT)
    expected = _expected_columns(ref, geo, sched.column_index).reshape(-1, 2, 3)
    torch.testing.assert_close(
        read_group(store, group, 0), expected, rtol=1e-4, atol=1e-5
    )


def test_direct_schedule_keeps_given_order(bank):
    images, geo, pixels = _sparse_case(bank, n_selected=12)
    order = pixels.index[torch.randperm(12, generator=torch.Generator().manual_seed(5))]
    sched, store, group = _run(
        bank, images, geo, pixels, RECT, block_pixels=6, column_order=order
    )
    assert torch.equal(sched.column_index, order)
    ref = _reference(bank, images.images, RECT)
    expected = _expected_columns(ref, geo, order).reshape(-1, 2, 3)
    torch.testing.assert_close(
        read_group(store, group, 0), expected, rtol=1e-4, atol=1e-5
    )
    with pytest.raises(ValueError):
        schedule_tiles(geo, pixels, order=order[:-1])


def test_direct_engine_reuses_the_banks_planar_block(bank):
    images, geo, pixels = _sparse_case(bank, n_selected=20)
    sched, store, group = _run(bank, images, geo, pixels, RECT, block_pixels=8)
    assert sched.mode == "direct"
    assert bank.has_planar_block(RECT, torch.float32)
    block = bank.planar_block(RECT, torch.float32)
    nbytes = bank.planar_bytes
    # a second featurization (another store) reuses the block: nothing rebuilt
    sched2, store2, group2 = _run(bank, images, geo, pixels, RECT, block_pixels=8)
    assert bank.planar_block(RECT, torch.float32) is block and bank.planar_bytes == nbytes
    torch.testing.assert_close(read_group(store2, group2, 0), read_group(store, group, 0))
    # planar_fraction=0 takes the chunked path and leaves the tier alone
    bank.clear()
    _run(bank, images, geo, pixels, RECT, block_pixels=8, planar_fraction=0.0)
    assert not bank.has_planar_block(RECT, torch.float32) and bank.planar_bytes == 0
