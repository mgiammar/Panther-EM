"""Unit tests for the spiral polar transform's inverse-warp boundary handling.

Covers:
- origin_continuation (interpolation nodes and weights)
- Round trips near the origin, with and without radial padding
- Zeroing outside the support
- The angular seam of the wrap padding (constant-image interior round trip)
- GridTransform delegating the radial padding to its source transform
"""

import numpy as np
import pytest

from panther_em.coordinates.spiral_polar import (
    RADIAL_PAD,
    SpiralPolarTransform,
    origin_continuation,
)
from panther_em.coordinates.transform_base import GridTransform

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SHAPE = (128, 128)
CENTER = (63.5, 63.5)
RADIUS = 64.0
NUM_ANGLE = 240
NUM_RADIUS = 192


def make_transform() -> SpiralPolarTransform:
    return SpiralPolarTransform(
        center=CENTER,
        radius=RADIUS,
        num_angle=NUM_ANGLE,
        num_radius=NUM_RADIUS,
        height=SHAPE[0],
        width=SHAPE[1],
    )


def radius_map() -> np.ndarray:
    yy, xx = np.indices(SHAPE)
    return np.hypot(yy - CENTER[0], xx - CENTER[1])


def off_centre_gaussian(dy: float = 2.3, dx: float = -1.7, sigma: float = 3.0):
    """Compact blob near the origin; has odd angular modes about the centre."""
    yy, xx = np.indices(SHAPE)
    return np.exp(
        -((yy - CENTER[0] - dy) ** 2 + (xx - CENTER[1] - dx) ** 2) / (2 * sigma**2)
    )


def round_trip(transform, image: np.ndarray, **kwargs) -> np.ndarray:
    return transform.to_cartesian(transform.to_transform_space(image), **kwargs)


# ===========================================================================
# origin_continuation
# ===========================================================================


class TestOriginContinuation:
    def test_shapes(self):
        untwist, nodes, weights, phases = origin_continuation(
            NUM_ANGLE, NUM_RADIUS, RADIUS
        )
        num_rows = nodes.shape[0]
        assert 1 <= num_rows <= RADIAL_PAD
        assert weights.shape == (num_rows, 4)
        assert phases.shape == (NUM_ANGLE // 2 + 1, num_rows)
        assert untwist.shape == (NUM_ANGLE // 2 + 1, int(nodes.max()) + 1)

    def test_weights_reproduce_constants(self):
        _, _, weights, _ = origin_continuation(NUM_ANGLE, NUM_RADIUS, RADIUS)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-12)

    def test_origin_row_keeps_only_dc(self):
        _, _, _, phases = origin_continuation(NUM_ANGLE, NUM_RADIUS, RADIUS)
        assert phases[0, 0] == 1.0
        assert np.all(phases[1:, 0] == 0.0)

    def test_rows_limited_by_geometry(self):
        # A small c leaves little room below t = 0 (needs c**2 + 2t > 0)
        _, nodes, _, _ = origin_continuation(NUM_ANGLE, NUM_RADIUS, RADIUS, c=0.05)
        assert nodes.shape[0] < RADIAL_PAD

    def test_too_few_rings_raises(self):
        with pytest.raises(ValueError, match="num_radius"):
            origin_continuation(NUM_ANGLE, 3, RADIUS)


# ===========================================================================
# Inverse warp near the origin and outside the support
# ===========================================================================


class TestRadialBoundaries:
    @pytest.mark.parametrize(
        "image",
        [np.ones(SHAPE), off_centre_gaussian()],
        ids=["constant", "off_centre_gaussian"],
    )
    def test_centre_round_trip(self, image):
        r = radius_map()
        transform = make_transform()
        padded = np.abs(round_trip(transform, image) - image)[r < 3]
        unpadded = np.abs(round_trip(transform, image, pad_radial_axis=False) - image)
        assert padded.max() < 1e-4
        # Without the continuation the centre drops toward the zero padding
        assert unpadded[r < 3].max() > 0.05

    def test_interior_round_trip_of_constant(self):
        r = radius_map()
        error = np.abs(round_trip(make_transform(), np.ones(SHAPE)) - 1.0)
        assert error[(r >= 3) & (r < 56)].max() < 1e-5

    def test_zero_outside_support(self):
        r = radius_map()
        out = round_trip(make_transform(), np.ones(SHAPE))
        assert np.all(out[r > RADIUS + 1e-3] == 0.0)

    def test_batched_and_complex(self):
        transform = make_transform()
        polar = transform.to_transform_space(off_centre_gaussian())
        single = transform.to_cartesian(polar)
        batched = transform.to_cartesian(np.stack([polar, 2 * polar]))
        np.testing.assert_allclose(batched[0], single)
        np.testing.assert_allclose(batched[1], 2 * single)
        complex_out = transform.to_cartesian(polar + 1j * polar)
        np.testing.assert_allclose(complex_out.real, single)
        np.testing.assert_allclose(complex_out.imag, single)


# ===========================================================================
# GridTransform delegation
# ===========================================================================


class TestGridTransformDelegation:
    def test_source_transform_rebuilt(self):
        grid = GridTransform.from_transform(make_transform())
        assert isinstance(grid.source_transform, SpiralPolarTransform)

    def test_matches_source_transform(self):
        transform = make_transform()
        grid = GridTransform.from_transform(transform)
        polar = transform.to_transform_space(off_centre_gaussian())
        np.testing.assert_array_equal(
            grid.to_cartesian(polar), transform.to_cartesian(polar)
        )

    def test_without_source_params_is_unpadded(self):
        transform = make_transform()
        grid = GridTransform.from_arrays(
            transform_coords=transform.transform_coords,
            cartesian_coords=transform.cartesian_coords,
            jacobian=transform.jacobian_grid,
            polar_shape=transform.polar_shape,
            cartesian_shape=transform.cartesian_shape,
        )
        assert grid.source_transform is None
        polar = transform.to_transform_space(off_centre_gaussian())
        np.testing.assert_array_equal(
            grid.to_cartesian(polar),
            transform.to_cartesian(polar, pad_radial_axis=False),
        )

    def test_mismatched_source_params_raise(self):
        transform = make_transform()
        params = transform.to_dict()
        params["num_radius"] = NUM_RADIUS // 2
        grid = GridTransform.from_arrays(
            transform_coords=transform.transform_coords,
            cartesian_coords=transform.cartesian_coords,
            jacobian=transform.jacobian_grid,
            polar_shape=transform.polar_shape,
            cartesian_shape=transform.cartesian_shape,
            source_params=params,
        )
        with pytest.raises(ValueError, match="source_params"):
            _ = grid.source_transform
