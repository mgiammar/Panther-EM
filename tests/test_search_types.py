"""Value types: ImageSet, SearchGeometry, PixelSelection, Precision."""

from __future__ import annotations

import pytest
import torch

from panther_em.inference.search.types import (
    BatchShape,
    ImageSet,
    KernelConfig,
    PixelSelection,
    Precision,
    SearchGeometry,
)


def test_precision_coerce():
    assert Precision.coerce("fp16") is Precision.FP16
    assert Precision.coerce(Precision.TF32) is Precision.TF32
    with pytest.raises(ValueError):
        Precision.coerce("fp8")


def test_batch_shape_validation():
    with pytest.raises(ValueError):
        BatchShape(0, 4)


def test_kernel_config_kwargs_match_construct_cartesian_feature_signature():
    import inspect

    from panther_em.inference.projection_reconstruction import ProjectionReconstructor

    params = inspect.signature(
        ProjectionReconstructor.construct_cartesian_feature
    ).parameters
    for key in KernelConfig().as_kwargs():
        assert key in params


def test_image_set_promotes_single_image_and_validates_filters():
    img = torch.zeros(30, 32)
    s = ImageSet.single(img)
    assert s.images.shape == (1, 30, 32)
    assert s.n_defocus == 1 and s.filter_plane(0) is None

    shared = ImageSet(img, filters=torch.ones(3, 30, 17))
    assert shared.n_defocus == 3 and not shared.per_image_filters
    assert shared.filter_plane(2, 0).shape == (30, 17)

    per_image = ImageSet(torch.zeros(4, 30, 32), filters=torch.ones(4, 2, 30, 17))
    assert per_image.n_defocus == 2 and per_image.per_image_filters

    with pytest.raises(ValueError):
        ImageSet(img, filters=torch.ones(3, 30, 16))
    with pytest.raises(ValueError):
        ImageSet(torch.zeros(4, 30, 32), filters=torch.ones(3, 2, 30, 17))
    with pytest.raises(ValueError):
        ImageSet(torch.zeros(2, 3, 30, 32))


def test_geometry_grid_mode_and_flat_indexing():
    images = ImageSet.from_stack(torch.zeros(2, 30, 30))
    geo = SearchGeometry.from_images(images, (20, 20))
    assert geo.mode == "grid" and geo.grid == (11, 11)
    assert geo.n_pixels == 2 * 121 and geo.dense_shape == (2, 11, 11)

    mask = torch.zeros(11, 11, dtype=torch.bool)
    mask[3, 4] = True
    flat = geo.flat_index(mask)
    # shared mask -> the same pixel in both images, batch-major
    assert flat.tolist() == [3 * 11 + 4, 121 + 3 * 11 + 4]

    per_image = torch.zeros(2, 11, 11, dtype=torch.bool)
    per_image[1, 0, 0] = True
    assert geo.flat_index(per_image).tolist() == [121]

    with pytest.raises(ValueError):
        geo.validate_mask(torch.zeros(5, 5, dtype=torch.bool))
    with pytest.raises(ValueError):
        geo.validate_mask(torch.zeros(11, 11))  # not bool


def test_geometry_center_mode_by_box_size_and_override():
    boxes = ImageSet.from_boxes(torch.zeros(5, 20, 20))
    geo = SearchGeometry.from_images(boxes, (20, 20))
    assert geo.mode == "center" and geo.grid == (1,) and geo.n_pixels == 5
    assert geo.validate_mask(torch.ones(5, dtype=torch.bool)).shape == (5, 1)

    larger = ImageSet.from_boxes(torch.zeros(5, 24, 24), center_only=True)
    assert SearchGeometry.from_images(larger, (20, 20)).mode == "center"

    with pytest.raises(ValueError):
        SearchGeometry.from_images(ImageSet.single(torch.zeros(10, 10)), (20, 20))


def test_pixel_selection_all_subset_scatter():
    geo = SearchGeometry.from_images(
        ImageSet.from_stack(torch.zeros(2, 30, 30)), (20, 20)
    )
    sel = PixelSelection.all(geo)
    assert len(sel) == geo.n_pixels and sel.is_full

    keep = torch.zeros(len(sel), dtype=torch.bool)
    keep[[0, 5, 121 + 7]] = True
    sub = sel.subset(keep)
    assert sub.index.tolist() == [0, 5, 128] and not sub.is_full

    dense = sub.scatter(torch.tensor([1.0, 2.0, 3.0]), fill=float("nan"))
    assert dense.shape == (2, 11, 11)
    assert dense[0, 0, 0] == 1.0 and dense[0, 0, 5] == 2.0 and dense[1, 0, 7] == 3.0
    assert torch.isnan(dense).sum() == geo.n_pixels - 3

    from_mask = PixelSelection.from_mask(geo, None)
    assert from_mask.is_full


def test_pixel_maps_validation_and_broadcast():
    images = ImageSet.from_stack(
        torch.zeros(2, 30, 30), pixel_maps={"image_norm": torch.ones(11, 11)}
    )
    geo = SearchGeometry.from_images(images, (20, 20))
    maps = geo.validate_pixel_maps(images.pixel_maps)
    assert (
        maps["image_norm"].shape == (2, 11, 11)
        and maps["image_norm"].dtype == torch.float32
    )
    per_image = geo.validate_pixel_map("noise_variance", torch.rand(2, 11, 11).double())
    assert per_image.shape == (2, 11, 11) and per_image.dtype == torch.float32
    assert geo.validate_pixel_maps(None) == {}
    with pytest.raises(ValueError, match="noise_variance"):
        geo.validate_pixel_map("noise_variance", torch.ones(5, 5))
    with pytest.raises(ValueError):
        SearchGeometry.from_images(
            ImageSet.from_stack(
                torch.zeros(2, 30, 30), pixel_maps={"x": torch.ones(5, 5)}
            ),
            (20, 20),
        )
    with pytest.raises(ValueError):
        ImageSet.single(torch.zeros(30, 30), pixel_maps={"x": torch.ones(2, 2, 2, 2)})
    with pytest.raises(ValueError):
        ImageSet.single(
            torch.zeros(30, 30),
            pixel_maps={"x": torch.ones(11, 11, dtype=torch.complex64)},
        )

    boxes = ImageSet.from_boxes(
        torch.zeros(5, 20, 20), pixel_maps={"image_norm": torch.arange(5.0)}
    )
    center = SearchGeometry.from_images(boxes, (20, 20))
    assert center.validate_pixel_maps(boxes.pixel_maps)["image_norm"].shape == (5, 1)
