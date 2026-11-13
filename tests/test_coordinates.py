"""Unit tests for the coordinate transform module.

Covers:
- Low-level mapping functions in offset_polar.py
- OffsetPolarTransform class (shapes, serialization, caching)
- transform_base registry (register, get_class, get_transform, reconstruct)
- GridTransform construction and serialization
"""

import numpy as np
import pytest
import torch

from panther_em.coordinates.offset_polar import (
    OffsetPolarTransform,
    forward_cartesian_to_offset_polar_mapping,
    forward_offset_polar_to_cartesian_mapping,
    jacobian_correction_offset_polar,
)
from panther_em.coordinates.transform_base import (
    _REGISTRY,
    CoordinateTransform,
    GridTransform,
    get_transform,
    get_transform_class,
    reconstruct_transform,
    register_transform,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_ANGLE = 64
NUM_RADIUS = 32
MAX_RADIUS = 32.0
CENTER = (50.0, 50.0)
IMAGE_SHAPE = (100, 100)


def make_transform() -> OffsetPolarTransform:
    return OffsetPolarTransform(
        center=CENTER,
        radius=MAX_RADIUS,
        num_angle=NUM_ANGLE,
        num_radius=NUM_RADIUS,
        height=IMAGE_SHAPE[0],
        width=IMAGE_SHAPE[1],
    )


# ===========================================================================
# Low-level mapping functions
# ===========================================================================


class TestForwardCartesianToOffsetPolar:
    """forward_cartesian_to_offset_polar_mapping"""

    def test_center_maps_to_zero_radius(self):
        coords = np.array([[CENTER[0], CENTER[1]]], dtype=float)
        result = forward_cartesian_to_offset_polar_mapping(
            coords, NUM_ANGLE, NUM_RADIUS, MAX_RADIUS, CENTER
        )
        assert result[0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_known_point_right_of_center(self):
        # A point directly to the right of center should have angle index ≈ 0
        r = 10.0
        coords = np.array([[CENTER[0], CENTER[1] + r]], dtype=float)
        result = forward_cartesian_to_offset_polar_mapping(
            coords, NUM_ANGLE, NUM_RADIUS, MAX_RADIUS, CENTER
        )
        radius_idx = result[0, 0]
        assert radius_idx == pytest.approx(r / MAX_RADIUS * NUM_RADIUS, rel=1e-6)

    def test_output_shape(self):
        M = 20
        coords = np.random.default_rng(0).uniform(0, 100, (M, 2))
        result = forward_cartesian_to_offset_polar_mapping(
            coords, NUM_ANGLE, NUM_RADIUS, MAX_RADIUS, CENTER
        )
        assert result.shape == (M, 2)

    def test_angle_index_in_range(self):
        rng = np.random.default_rng(42)
        coords = rng.uniform(20, 80, (100, 2))
        result = forward_cartesian_to_offset_polar_mapping(
            coords, NUM_ANGLE, NUM_RADIUS, MAX_RADIUS, CENTER
        )
        angle_idx = result[:, 1]
        assert np.all(angle_idx >= 0)
        assert np.all(angle_idx < NUM_ANGLE)


class TestForwardOffsetPolarToCartesian:
    """forward_offset_polar_to_cartesian_mapping"""

    def test_zero_radius_maps_to_center(self):
        coords = np.array([[0.0, 0.0]], dtype=float)
        result = forward_offset_polar_to_cartesian_mapping(
            coords, NUM_ANGLE, NUM_RADIUS, MAX_RADIUS, CENTER
        )
        assert result[0, 0] == pytest.approx(CENTER[0], abs=1e-10)
        assert result[0, 1] == pytest.approx(CENTER[1], abs=1e-10)

    def test_output_shape(self):
        M = 15
        coords = np.column_stack(
            [
                np.linspace(0, NUM_RADIUS - 1, M),
                np.linspace(0, NUM_ANGLE - 1, M),
            ]
        )
        result = forward_offset_polar_to_cartesian_mapping(
            coords, NUM_ANGLE, NUM_RADIUS, MAX_RADIUS, CENTER
        )
        assert result.shape == (M, 2)

    def test_max_radius_ring_approximately_at_max_radius(self):
        # Point at max radius index, angle 0 -> should be approx. MAX_RADIUS from center
        coords = np.array([[float(NUM_RADIUS), 0.0]])
        result = forward_offset_polar_to_cartesian_mapping(
            coords, NUM_ANGLE, NUM_RADIUS, MAX_RADIUS, CENTER
        )
        dr = result[0, 0] - CENTER[0]
        dc = result[0, 1] - CENTER[1]
        distance = np.sqrt(dr**2 + dc**2)
        assert distance == pytest.approx(MAX_RADIUS, rel=1e-6)


class TestRoundTrip:
    """Round-trip forward transforms return same coordinates."""

    def test_round_trip_random_points(self):
        rng = np.random.default_rng(7)
        # Restrict points to well within max_radius to avoid boundary issues
        offsets = rng.uniform(-MAX_RADIUS * 0.8, MAX_RADIUS * 0.8, (50, 2))
        cart_in = offsets + np.array(CENTER)

        polar = forward_cartesian_to_offset_polar_mapping(
            cart_in, NUM_ANGLE, NUM_RADIUS, MAX_RADIUS, CENTER
        )
        cart_out = forward_offset_polar_to_cartesian_mapping(
            polar, NUM_ANGLE, NUM_RADIUS, MAX_RADIUS, CENTER
        )
        np.testing.assert_allclose(cart_out, cart_in, atol=1e-8)

    def test_round_trip_cardinal_directions(self):
        r = 15.0
        # Right, below, left, above center (in image row/col coords)
        directions = [
            [CENTER[0], CENTER[1] + r],
            [CENTER[0] + r, CENTER[1]],
            [CENTER[0], CENTER[1] - r],
            [CENTER[0] - r, CENTER[1]],
        ]
        cart_in = np.array(directions)
        polar = forward_cartesian_to_offset_polar_mapping(
            cart_in, NUM_ANGLE, NUM_RADIUS, MAX_RADIUS, CENTER
        )
        cart_out = forward_offset_polar_to_cartesian_mapping(
            polar, NUM_ANGLE, NUM_RADIUS, MAX_RADIUS, CENTER
        )
        np.testing.assert_allclose(cart_out, cart_in, atol=1e-8)


class TestAngularOffset:
    """Every other ring should be offset by half delta_theta."""

    def test_even_odd_ring_angle_difference(self):
        # Place a point at angle=0, on ring 0 (even) vs ring 1 (odd).
        # The ring-1 adjusted angle should differ by delta_theta/2.
        _delta_theta = (2 * np.pi) / NUM_ANGLE

        # Ring 0: radius_idx in [0, 1) --> use 0.5
        # Ring 1: radius_idx in [1, 2) --> use 1.5
        # Same Cartesian angle (arctan2 = 0, i.e., point to the right)
        r0 = 0.5 / NUM_RADIUS * MAX_RADIUS
        r1 = 1.5 / NUM_RADIUS * MAX_RADIUS

        # Both points are directly to the right of center (angle=0)
        pts = np.array(
            [
                [CENTER[0], CENTER[1] + r0],
                [CENTER[0], CENTER[1] + r1],
            ]
        )
        result = forward_cartesian_to_offset_polar_mapping(
            pts, NUM_ANGLE, NUM_RADIUS, MAX_RADIUS, CENTER
        )
        angle_idx_ring0 = result[0, 1]
        angle_idx_ring1 = result[1, 1]

        # Ring 0 (even) --> no offset; ring 1 (odd) --> offset by delta_theta/2
        # subtracted from the angle. Since original angle=0, the adjusted angle goes
        # negative --> wrapped to ~2pi - delta_theta/2.
        expected_shift_idx = 0.5  # half a bin
        diff = (angle_idx_ring0 - angle_idx_ring1) % NUM_ANGLE

        # diff should equal expected_shift_idx mod NUM_ANGLE
        diff = min(diff, NUM_ANGLE - diff)

        assert diff == pytest.approx(expected_shift_idx, abs=1e-6)


# ===========================================================================
# Jacobian correction
# ===========================================================================


class TestJacobianCorrection:
    """jacobian_correction_offset_polar"""

    def test_output_shape(self):
        jac = jacobian_correction_offset_polar(NUM_ANGLE, NUM_RADIUS, MAX_RADIUS)
        assert jac.shape == (NUM_RADIUS,)

    def test_monotonically_increasing(self):
        jac = jacobian_correction_offset_polar(NUM_ANGLE, NUM_RADIUS, MAX_RADIUS)
        assert np.all(np.diff(jac) > 0)

    def test_area_formula(self):
        # Area of annular sector = 0.5 * (r_out^2 - r_in^2) * dtheta
        jac = jacobian_correction_offset_polar(NUM_ANGLE, NUM_RADIUS, MAX_RADIUS)
        dr = MAX_RADIUS / NUM_RADIUS
        dtheta = (2 * np.pi) / NUM_ANGLE
        ring = 5
        r_in = ring * dr
        r_out = r_in + dr
        expected = 0.5 * (r_out**2 - r_in**2) * dtheta
        assert jac[ring] == pytest.approx(expected, rel=1e-10)

    def test_all_positive(self):
        jac = jacobian_correction_offset_polar(NUM_ANGLE, NUM_RADIUS, MAX_RADIUS)
        assert np.all(jac > 0)


# ===========================================================================
# OffsetPolarTransform
# ===========================================================================


class TestOffsetPolarTransformProperties:
    def test_polar_shape(self):
        t = make_transform()
        assert t.polar_shape == (NUM_ANGLE, NUM_RADIUS)

    def test_cartesian_shape(self):
        t = make_transform()
        assert t.cartesian_shape == IMAGE_SHAPE

    def test_from_image_defaults(self):
        t = OffsetPolarTransform.from_image(IMAGE_SHAPE)
        h, w = IMAGE_SHAPE
        assert t.cartesian_shape == IMAGE_SHAPE
        assert t.center == (h / 2 - 0.5, w / 2 - 0.5)
        assert t.radius == h / 2

    def test_from_image_custom_params(self):
        t = OffsetPolarTransform.from_image(
            IMAGE_SHAPE, num_angle=128, num_radius=40, radius=40.0
        )
        assert t.polar_shape == (128, 40)
        assert t.radius == 40.0


class TestOffsetPolarTransformCoordGrids:
    def test_transform_coords_shape(self):
        t = make_transform()
        tc = t.transform_coords
        assert tc.shape == (2, NUM_ANGLE, NUM_RADIUS)

    def test_cartesian_coords_shape(self):
        t = make_transform()
        cc = t.cartesian_coords
        assert cc.shape == (2, IMAGE_SHAPE[0], IMAGE_SHAPE[1])

    def test_jacobian_shape(self):
        t = make_transform()
        jac = t.jacobian_grid
        assert jac is not None
        assert jac.shape == (NUM_ANGLE, NUM_RADIUS)

    def test_jacobian_is_float32(self):
        t = make_transform()
        assert t.jacobian_grid.dtype == np.float32

    def test_transform_coords_cached(self):
        t = make_transform()
        tc1 = t.transform_coords
        tc2 = t.transform_coords
        assert tc1 is tc2  # same object — not recomputed

    def test_clear_cache_resets_grids(self):
        t = make_transform()
        _ = t.transform_coords
        _ = t.jacobian_grid
        t.clear_cache()
        assert t._transform_coords is None
        assert t._cartesian_coords is None
        assert t._jacobian is None
        assert t._device_cache == {}


class TestOffsetPolarTransformSerialization:
    def test_to_dict_keys(self):
        t = make_transform()
        d = t.to_dict()
        for key in (
            "transform_name",
            "center",
            "radius",
            "num_angle",
            "num_radius",
            "height",
            "width",
        ):
            assert key in d

    def test_to_dict_values(self):
        t = make_transform()
        d = t.to_dict()
        assert d["transform_name"] == "offset_polar"
        assert d["num_angle"] == NUM_ANGLE
        assert d["num_radius"] == NUM_RADIUS
        assert d["radius"] == MAX_RADIUS
        assert d["center"] == list(CENTER)

    def test_from_dict_round_trip(self):
        t = make_transform()
        d = t.to_dict()
        t2 = OffsetPolarTransform.from_dict(d)
        assert t2.polar_shape == t.polar_shape
        assert t2.cartesian_shape == t.cartesian_shape
        assert t2.center == t.center
        assert t2.radius == t.radius

    def test_reconstruct_transform(self):
        t = make_transform()
        d = t.to_dict()
        t2 = reconstruct_transform(d)
        assert isinstance(t2, OffsetPolarTransform)
        assert t2.polar_shape == t.polar_shape


class TestOffsetPolarTransformWarp:
    """Smoke tests: warp an image to polar space and back; check shapes."""

    def test_to_transform_space_shape(self):
        t = make_transform()
        img = np.ones(IMAGE_SHAPE, dtype=np.float64)
        polar = t.to_transform_space(img, preserve_energy=False, order=1)
        assert polar.shape == (NUM_ANGLE, NUM_RADIUS)

    def test_to_cartesian_shape(self):
        t = make_transform()
        polar_img = np.ones((NUM_ANGLE, NUM_RADIUS), dtype=np.float64)
        cart = t.to_cartesian(polar_img, preserve_energy=False, order=1)
        assert cart.shape == IMAGE_SHAPE

    def test_batched_to_transform_space_shape(self):
        t = make_transform()
        imgs = np.ones((3, *IMAGE_SHAPE), dtype=np.float64)
        polars = t.to_transform_space(imgs, preserve_energy=False, order=1)
        assert polars.shape == (3, NUM_ANGLE, NUM_RADIUS)

    def test_energy_preservation_increases_values(self):
        # Jacobian > 1 for rings away from center, so energy-preserved output
        # should differ from non-preserved.
        t = make_transform()
        img = np.ones(IMAGE_SHAPE, dtype=np.float64)
        polar_no_ep = t.to_transform_space(img, preserve_energy=False, order=1)
        polar_ep = t.to_transform_space(img, preserve_energy=True, order=1)
        # They should differ (Jacobian is not identically 1)
        assert not np.allclose(polar_no_ep, polar_ep)


class TestArrayTypeIsPreserved:
    """numpy in gives numpy out, and a CPU tensor in gives a CPU tensor out."""

    @pytest.mark.parametrize("batch", [(), (3,)], ids=["single", "batched"])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64])
    @pytest.mark.parametrize("method", ["to_transform_space", "to_cartesian"])
    def test_cpu_tensor_round_trip(self, method, dtype, batch):
        t = make_transform()
        shape = IMAGE_SHAPE if method == "to_transform_space" else t.polar_shape
        rng = np.random.default_rng(0)
        image = rng.standard_normal((*batch, *shape))
        if np.issubdtype(dtype, np.complexfloating):
            image = image + 1j * rng.standard_normal(image.shape)
        image = image.astype(dtype)

        warp = getattr(t, method)
        expected = warp(image, preserve_energy=False, order=1)
        result = warp(torch.from_numpy(image), preserve_energy=False, order=1)

        assert isinstance(expected, np.ndarray)
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"
        np.testing.assert_array_equal(result.numpy(), expected)


# ===========================================================================
# Registry functions
# ===========================================================================


class TestRegistry:
    def test_offset_polar_is_registered(self):
        cls = get_transform_class("offset_polar")
        assert cls is OffsetPolarTransform

    def test_get_transform_class_unknown_raises(self):
        with pytest.raises(KeyError, match="No coordinate transform named"):
            get_transform_class("does_not_exist_xyz")

    def test_get_transform_returns_same_instance(self):
        kwargs = {
            "center": (40.0, 40.0),
            "radius": 30.0,
            "num_angle": 32,
            "num_radius": 16,
            "height": 80,
            "width": 80,
        }
        t1 = get_transform(OffsetPolarTransform, **kwargs)
        t2 = get_transform(OffsetPolarTransform, **kwargs)
        assert t1 is t2

    def test_register_transform_decorator(self):
        @register_transform
        class _TestTransform(CoordinateTransform):
            transform_name = "_test_tmp_coord"
            supports_energy_preservation = False
            has_periodic_axis = False

            @property
            def polar_shape(self):
                return (1, 1)

            @property
            def cartesian_shape(self):
                return (1, 1)

            def to_dict(self):
                return {"transform_name": self.transform_name}

            @classmethod
            def from_dict(cls, params, device="numpy"):
                return cls()

        assert get_transform_class("_test_tmp_coord") is _TestTransform
        # Cleanup to avoid polluting other tests
        del _REGISTRY["_test_tmp_coord"]

    def test_register_transform_duplicate_warns(self):
        @register_transform
        class _DupA(CoordinateTransform):
            transform_name = "_test_dup_abc"

            @property
            def polar_shape(self):
                return (1, 1)

            @property
            def cartesian_shape(self):
                return (1, 1)

            def to_dict(self):
                return {"transform_name": self.transform_name}

            @classmethod
            def from_dict(cls, params, device="numpy"):
                return cls()

        with pytest.warns(UserWarning, match="already registered"):

            @register_transform
            class _DupB(CoordinateTransform):
                transform_name = "_test_dup_abc"

                @property
                def polar_shape(self):
                    return (1, 1)

                @property
                def cartesian_shape(self):
                    return (1, 1)

                def to_dict(self):
                    return {"transform_name": self.transform_name}

                @classmethod
                def from_dict(cls, params, device="numpy"):
                    return cls()

        del _REGISTRY["_test_dup_abc"]

    def test_reconstruct_transform_missing_key_raises(self):
        with pytest.raises(KeyError, match="transform_name"):
            reconstruct_transform({"num_angle": 64})


# ===========================================================================
# GridTransform
# ===========================================================================


class TestGridTransform:
    def _source_transform(self) -> OffsetPolarTransform:
        return OffsetPolarTransform(
            center=(25.0, 25.0),
            radius=20.0,
            num_angle=32,
            num_radius=16,
            height=50,
            width=50,
        )

    def test_from_transform_shapes(self):
        src = self._source_transform()
        gt = GridTransform.from_transform(src)
        assert gt.polar_shape == src.polar_shape
        assert gt.cartesian_shape == src.cartesian_shape

    def test_from_transform_grids_match(self):
        src = self._source_transform()
        gt = GridTransform.from_transform(src)
        np.testing.assert_array_equal(gt.transform_coords, src.transform_coords)
        np.testing.assert_array_equal(gt.cartesian_coords, src.cartesian_coords)
        np.testing.assert_array_equal(gt.jacobian_grid, src.jacobian_grid)

    def test_from_arrays_construction(self):
        src = self._source_transform()
        gt = GridTransform.from_arrays(
            transform_coords=src.transform_coords,
            cartesian_coords=src.cartesian_coords,
            jacobian=src.jacobian_grid,
            polar_shape=src.polar_shape,
            cartesian_shape=src.cartesian_shape,
        )
        assert gt.polar_shape == src.polar_shape
        assert gt.cartesian_shape == src.cartesian_shape

    def test_to_dict_keys(self):
        src = self._source_transform()
        gt = GridTransform.from_transform(src)
        d = gt.to_dict()
        for key in (
            "transform_name",
            "polar_shape",
            "cartesian_shape",
            "has_jacobian",
            "has_periodic_axis",
            "periodic_axis",
            "source_params",
        ):
            assert key in d
        assert d["transform_name"] == "grid"
        assert d["has_jacobian"] is True

    def test_from_dict_raises(self):
        with pytest.raises(NotImplementedError, match="GridTransform cannot"):
            GridTransform.from_dict({})

    def test_warp_shape_matches_source(self):
        src = self._source_transform()
        gt = GridTransform.from_transform(src)
        img = np.ones((50, 50), dtype=np.float64)
        polar = gt.to_transform_space(img, preserve_energy=False, order=1)
        assert polar.shape == src.polar_shape
