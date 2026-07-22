"""Unit tests for the decomposition pipeline and result classes.

Covers:
- _compute_freq_crop: index math for k_max cropping (real and complex modes)
- precompute_volume_dft: shape, padding, DC zeroing
- apply_fourier_filters: output shape and identity behaviour
- DecompositionResult: shape validation, index helpers, get_component, get_top_n
- DecompositionResult save/load round-trip
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from panther_em.coordinates.offset_polar import OffsetPolarTransform
from panther_em.decomposition.pipeline_projections import (
    _compute_freq_crop,
    apply_fourier_filters,
    precompute_volume_dft,
)
from panther_em.decomposition.result import DecompositionResult

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NUM_FF = 2  # fourier filters
NUM_OR = 8  # orientations
K_MAX = 5
EIG_MAX = 4
NUM_R = 10  # radial components
NUM_ANG = 32  # angular components in polar space


def _make_result(is_complex: bool) -> DecompositionResult:
    """Minimal DecompositionResult with deterministic random arrays."""
    rng = np.random.default_rng(0)
    num_freq_blocks = K_MAX * 2 if is_complex else K_MAX

    # Use complex arrays even for real-valued decompositions so conjugation paths are tested.
    U = (
        rng.standard_normal((NUM_FF, NUM_OR, num_freq_blocks, EIG_MAX)).astype(np.float32)
        + 1j
        * rng.standard_normal((NUM_FF, NUM_OR, num_freq_blocks, EIG_MAX)).astype(np.float32)
    ).astype(np.complex64)
    S = rng.random((num_freq_blocks, EIG_MAX)).astype(np.float32)
    Vh = (
        rng.standard_normal((num_freq_blocks, EIG_MAX, NUM_R)).astype(np.float32)
        + 1j * rng.standard_normal((num_freq_blocks, EIG_MAX, NUM_R)).astype(np.float32)
    ).astype(np.complex64)
    transform = OffsetPolarTransform.from_image(
        image_shape=(64, 64),
        num_angle=NUM_ANG,
        num_radius=NUM_R,
    )
    return DecompositionResult(
        U=U,
        S=S,
        Vh=Vh,
        k_max=K_MAX,
        eig_max=EIG_MAX,
        is_complex_projection=is_complex,
        num_fourier_filters=NUM_FF,
        num_orientations=NUM_OR,
        num_angular_components=NUM_ANG,
        num_radial_components=NUM_R,
        coordinate_transform=transform,
    )


# ===========================================================================
# _compute_freq_crop
# ===========================================================================


class TestComputeFreqCrop:
    """_compute_freq_crop: index computation for angular-frequency cropping."""

    # --- real-valued projection mode ---

    def test_real_none_keeps_all(self):
        lo, hi, k = _compute_freq_crop(
            is_complex=False, num_angular_mode=20, k_max=None
        )
        assert lo == 0
        assert hi == 20
        assert k == 20

    def test_real_explicit_k_max(self):
        lo, hi, k = _compute_freq_crop(is_complex=False, num_angular_mode=20, k_max=8)
        assert lo == 0
        assert hi == 8
        assert k == 8

    def test_real_k_max_1(self):
        lo, hi, k = _compute_freq_crop(is_complex=False, num_angular_mode=20, k_max=1)
        assert lo == 0
        assert hi == 1
        assert k == 1

    def test_real_k_max_at_limit(self):
        lo, hi, k = _compute_freq_crop(is_complex=False, num_angular_mode=16, k_max=16)
        assert lo == 0
        assert hi == 16
        assert k == 16

    def test_real_k_max_zero_raises(self):
        with pytest.raises(ValueError, match="k_max must be in"):
            _compute_freq_crop(is_complex=False, num_angular_mode=16, k_max=0)

    def test_real_k_max_exceeds_mode_raises(self):
        with pytest.raises(ValueError, match="k_max must be in"):
            _compute_freq_crop(is_complex=False, num_angular_mode=16, k_max=17)

    def test_real_num_freq_block(self):
        lo, hi, _ = _compute_freq_crop(is_complex=False, num_angular_mode=20, k_max=7)
        assert hi - lo == 7

    # --- complex-valued projection mode ---

    def test_complex_none_keeps_all(self):
        lo, hi, k = _compute_freq_crop(is_complex=True, num_angular_mode=20, k_max=None)
        dc = 20 // 2
        assert lo == dc - k
        assert hi == dc + k
        assert k == dc

    def test_complex_explicit_k_max(self):
        lo, hi, k = _compute_freq_crop(is_complex=True, num_angular_mode=20, k_max=3)
        dc = 10
        assert lo == dc - 3
        assert hi == dc + 3
        assert k == 3

    def test_complex_window_is_symmetric(self):
        lo, hi, _k = _compute_freq_crop(is_complex=True, num_angular_mode=40, k_max=5)
        dc = 20
        assert lo == dc - 5
        assert hi == dc + 5

    def test_complex_num_freq_block(self):
        lo, hi, _k = _compute_freq_crop(is_complex=True, num_angular_mode=40, k_max=5)
        assert hi - lo == 2 * 5

    def test_complex_k_max_zero_raises(self):
        with pytest.raises(ValueError, match="k_max must be in"):
            _compute_freq_crop(is_complex=True, num_angular_mode=20, k_max=0)

    def test_complex_k_max_exceeds_half_raises(self):
        with pytest.raises(ValueError, match="k_max must be in"):
            _compute_freq_crop(is_complex=True, num_angular_mode=20, k_max=11)


# ===========================================================================
# precompute_volume_dft
# ===========================================================================


class TestPrecomputeVolumeDft:
    """precompute_volume_dft: shape, padding, DC zeroing."""

    # compute_cube_face_averages requires n=4 < d//2, so d >= 10; use 16.

    def test_output_dft_shape_no_padding(self):
        vol = torch.zeros(16, 16, 16)
        dft, _, pad_width = precompute_volume_dft(vol, pad_factor=1.0)
        assert dft.shape == (16, 16, 9)  # rfft last dim: 16//2+1 = 9
        assert pad_width == 0

    def test_output_dft_shape_with_padding(self):
        d = 16
        pad_factor = 2.0
        vol = torch.zeros(d, d, d)
        dft, _, pad_width = precompute_volume_dft(vol, pad_factor=pad_factor)
        d_padded = d + 2 * pad_width
        assert dft.shape == (d_padded, d_padded, d_padded // 2 + 1)
        assert pad_width > 0

    def test_dc_is_zeroed(self):
        vol = torch.ones(16, 16, 16)
        dft, _, _ = precompute_volume_dft(vol, pad_factor=1.0)
        dc_row = dft.shape[0] // 2
        dc_col = dft.shape[1] // 2
        assert dft[dc_row, dc_col, 0].abs().item() == pytest.approx(0.0, abs=1e-6)

    def test_volume_mean_scaled_zero_after_background_subtraction(self):
        d = 16
        vol = torch.ones(d, d, d) * 3.0
        _, volume_mean_scaled, _ = precompute_volume_dft(vol, pad_factor=1.0)
        # Constant volume: after edge subtraction the mean → 0
        assert abs(volume_mean_scaled) < 1e-4

    def test_zero_background_false_preserves_mean(self):
        d = 16
        vol = torch.ones(d, d, d) * 2.0
        _, volume_mean_scaled, _ = precompute_volume_dft(
            vol, pad_factor=1.0, zero_background=False
        )
        assert volume_mean_scaled == pytest.approx(2.0 * d, rel=1e-4)

    def test_output_is_complex(self):
        vol = torch.zeros(16, 16, 16)
        dft, _, _ = precompute_volume_dft(vol, pad_factor=1.0)
        assert torch.is_complex(dft)

    def test_pad_width_scales_with_pad_factor(self):
        d = 16
        vol = torch.zeros(d, d, d)
        _, _, pw1 = precompute_volume_dft(vol, pad_factor=1.5)
        _, _, pw2 = precompute_volume_dft(vol, pad_factor=2.0)
        assert pw2 > pw1 > 0


# ===========================================================================
# apply_fourier_filters
# ===========================================================================


class TestApplyFourierFilters:
    """apply_fourier_filters: output shape and approximate identity."""

    def _real_projections(self, b=3, h=16, w=16) -> torch.Tensor:
        return torch.randn(b, h, w)

    def _rfft_filters(self, f=2, h=16, w=16) -> torch.Tensor:
        return torch.ones(f, h, w // 2 + 1, dtype=torch.complex64)

    def _fft_filters(self, f=2, h=16, w=16) -> torch.Tensor:
        return torch.ones(f, h, w, dtype=torch.complex64)

    def test_output_shape_rfft_path(self):
        projs = self._real_projections(b=3, h=16, w=16)
        filters = self._rfft_filters(f=2, h=16, w=16)
        out = apply_fourier_filters(projs, filters)
        assert out.shape == (2, 3, 16, 16)

    def test_output_shape_fft_path(self):
        projs = self._real_projections(b=3, h=16, w=16)
        filters = self._fft_filters(f=2, h=16, w=16)
        out = apply_fourier_filters(projs, filters)
        assert out.shape == (2, 3, 16, 16)

    def test_identity_filter_rfft_path(self):
        projs = self._real_projections(b=4, h=16, w=16)
        filters = self._rfft_filters(f=1, h=16, w=16)
        out = apply_fourier_filters(projs, filters)
        # With an all-ones filter the output should equal the input (modulo dtype cast)
        assert out.shape[0] == 1
        assert torch.allclose(out[0].real, projs, atol=1e-4)

    def test_zero_filter_produces_zeros(self):
        projs = self._real_projections(b=4, h=16, w=16)
        filters = torch.zeros(1, 16, 9, dtype=torch.complex64)
        out = apply_fourier_filters(projs, filters)
        assert out.abs().max().item() == pytest.approx(0.0, abs=1e-5)

    def test_multiple_filters_independent(self):
        projs = self._real_projections(b=2, h=16, w=16)
        f1 = torch.ones(1, 16, 9, dtype=torch.complex64)
        f2 = torch.zeros(1, 16, 9, dtype=torch.complex64)
        filters = torch.cat([f1, f2], dim=0)
        out = apply_fourier_filters(projs, filters)
        assert out.shape == (2, 2, 16, 16)
        assert out[1].abs().max().item() == pytest.approx(0.0, abs=1e-5)


# ===========================================================================
# DecompositionResult — construction and shape validation
# ===========================================================================


class TestDecompositionResultConstruction:
    """DecompositionResult.__post_init__: shape validation."""

    def test_valid_real_construction(self):
        result = _make_result(is_complex=False)
        assert result.k_max == K_MAX
        assert result.is_complex_projection is False

    def test_valid_complex_construction(self):
        result = _make_result(is_complex=True)
        assert result.k_max == K_MAX
        assert result.is_complex_projection is True

    def test_wrong_S_shape_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="S shape"):
            DecompositionResult(
                U=rng.standard_normal((NUM_FF, NUM_OR, K_MAX, EIG_MAX)).astype(
                    np.float32
                ),
                S=rng.random((K_MAX + 1, EIG_MAX)).astype(np.float32),  # wrong
                Vh=rng.standard_normal((K_MAX, EIG_MAX, NUM_R)).astype(np.float32),
                k_max=K_MAX,
                eig_max=EIG_MAX,
                is_complex_projection=False,
                num_fourier_filters=NUM_FF,
                num_orientations=NUM_OR,
                num_angular_components=NUM_ANG,
                num_radial_components=NUM_R,
                coordinate_transform=None,
            )

    def test_wrong_U_shape_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="U shape"):
            DecompositionResult(
                U=rng.standard_normal((NUM_FF, NUM_OR, K_MAX + 1, EIG_MAX)).astype(
                    np.float32
                ),  # wrong k dim
                S=rng.random((K_MAX, EIG_MAX)).astype(np.float32),
                Vh=rng.standard_normal((K_MAX, EIG_MAX, NUM_R)).astype(np.float32),
                k_max=K_MAX,
                eig_max=EIG_MAX,
                is_complex_projection=False,
                num_fourier_filters=NUM_FF,
                num_orientations=NUM_OR,
                num_angular_components=NUM_ANG,
                num_radial_components=NUM_R,
                coordinate_transform=None,
            )

    def test_wrong_Vh_shape_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="Vh shape"):
            DecompositionResult(
                U=rng.standard_normal((NUM_FF, NUM_OR, K_MAX, EIG_MAX)).astype(
                    np.float32
                ),
                S=rng.random((K_MAX, EIG_MAX)).astype(np.float32),
                Vh=rng.standard_normal((K_MAX, EIG_MAX, NUM_R + 1)).astype(
                    np.float32
                ),  # wrong last dim
                k_max=K_MAX,
                eig_max=EIG_MAX,
                is_complex_projection=False,
                num_fourier_filters=NUM_FF,
                num_orientations=NUM_OR,
                num_angular_components=NUM_ANG,
                num_radial_components=NUM_R,
                coordinate_transform=None,
            )

    def test_wrong_phi_shape_raises(self):
        result = _make_result(is_complex=False)
        with pytest.raises(ValueError, match="phi_values shape"):
            DecompositionResult(
                U=result.U,
                S=result.S,
                Vh=result.Vh,
                k_max=K_MAX,
                eig_max=EIG_MAX,
                is_complex_projection=False,
                num_fourier_filters=NUM_FF,
                num_orientations=NUM_OR,
                num_angular_components=NUM_ANG,
                num_radial_components=NUM_R,
                phi_values=np.zeros(NUM_OR + 1),  # wrong length
                coordinate_transform=None,
            )


# ===========================================================================
# DecompositionResult — index helpers
# ===========================================================================


class TestKToStored:
    """k_to_stored: angular-frequency → storage row translation."""

    def test_real_positive_k(self):
        result = _make_result(is_complex=False)
        assert result.k_to_stored(3) == 3

    def test_real_negative_k_maps_to_abs(self):
        result = _make_result(is_complex=False)
        assert result.k_to_stored(-3) == 3

    def test_real_zero(self):
        result = _make_result(is_complex=False)
        assert result.k_to_stored(0) == 0

    def test_real_array_input(self):
        result = _make_result(is_complex=False)
        k = np.array([-2, 0, 2])
        stored = result.k_to_stored(k)
        np.testing.assert_array_equal(stored, [2, 0, 2])

    def test_complex_positive_k(self):
        result = _make_result(is_complex=True)
        assert result.k_to_stored(2) == 2 + K_MAX

    def test_complex_negative_k(self):
        result = _make_result(is_complex=True)
        assert result.k_to_stored(-2) == -2 + K_MAX

    def test_complex_zero(self):
        result = _make_result(is_complex=True)
        assert result.k_to_stored(0) == K_MAX


class TestIsConjugateMode:
    """is_conjugate_mode: when conjugate is needed."""

    def test_complex_always_false_scalar(self):
        result = _make_result(is_complex=True)
        assert result.is_conjugate_mode(3) is False
        assert result.is_conjugate_mode(-3) is False
        assert result.is_conjugate_mode(0) is False

    def test_real_positive_k_false(self):
        result = _make_result(is_complex=False)
        assert result.is_conjugate_mode(2) is False

    def test_real_negative_k_true(self):
        result = _make_result(is_complex=False)
        assert result.is_conjugate_mode(-2) is True

    def test_real_zero_false(self):
        result = _make_result(is_complex=False)
        assert result.is_conjugate_mode(0) is False

    def test_real_array_input(self):
        result = _make_result(is_complex=False)
        k = np.array([-3, 0, 3])
        conj = result.is_conjugate_mode(k)
        np.testing.assert_array_equal(conj, [True, False, False])


# ===========================================================================
# DecompositionResult — get_component
# ===========================================================================


class TestGetComponent:
    """get_component: retrieves correct shapes and applies conjugation."""

    def test_scalar_indices_real(self):
        result = _make_result(is_complex=False)
        u, s, vh = result.get_component(k_idx=1, eig_idx=0)
        assert u.shape == (NUM_FF, NUM_OR, 1)
        assert s.shape == (1,)
        assert vh.shape == (1, NUM_R)

    def test_scalar_indices_complex(self):
        result = _make_result(is_complex=True)
        u, s, vh = result.get_component(k_idx=2, eig_idx=1)
        assert u.shape == (NUM_FF, NUM_OR, 1)
        assert s.shape == (1,)
        assert vh.shape == (1, NUM_R)

    def test_return_flags_subset(self):
        result = _make_result(is_complex=False)
        u, s, vh = result.get_component(
            k_idx=1, eig_idx=0, return_u=False, return_s=True, return_vh=False
        )
        assert u is None
        assert s is not None
        assert vh is None

    def test_all_false_raises(self):
        result = _make_result(is_complex=False)
        with pytest.raises(ValueError):
            result.get_component(
                k_idx=1, eig_idx=0, return_u=False, return_s=False, return_vh=False
            )

    def test_negative_k_conjugates_real(self):
        result = _make_result(is_complex=False)
        _, _, vh_pos = result.get_component(k_idx=2, eig_idx=0, return_u=False)
        _, _, vh_neg = result.get_component(k_idx=-2, eig_idx=0, return_u=False)
        np.testing.assert_array_almost_equal(vh_neg, vh_pos.conj())

    def test_negative_k_no_conjugate_complex(self):
        result = _make_result(is_complex=True)
        _, _, vh_neg = result.get_component(k_idx=-2, eig_idx=0, return_u=False)
        k_stored = result.k_to_stored(-2)
        np.testing.assert_array_equal(vh_neg[0], result.Vh[k_stored, 0])


# ===========================================================================
# DecompositionResult — get_top_n
# ===========================================================================


class TestGetTopN:
    """get_top_n: sorting and shape."""

    def test_output_shape_real(self):
        result = _make_result(is_complex=False)
        top = result.get_top_n(top_k=4, include_negative=False)
        assert top.shape[1] == 2  # (k_idx, eig_idx) pairs
        assert top.shape[0] <= 4

    def test_descending_order_real(self):
        result = _make_result(is_complex=False)
        top = result.get_top_n(top_k=K_MAX * EIG_MAX, include_negative=False)
        k_stored = top[:, 0].astype(int)
        eig = top[:, 1].astype(int)
        svs = result.S[k_stored, eig]
        assert np.all(svs[:-1] >= svs[1:])

    def test_include_negative_expands_real(self):
        result = _make_result(is_complex=False)
        top_without = result.get_top_n(top_k=4, include_negative=False)
        top_with = result.get_top_n(top_k=4, include_negative=True)
        # include_negative pairs positive k with its negative, so pairs may expand
        assert top_with.shape[0] >= top_without.shape[0]

    def test_output_shape_complex(self):
        result = _make_result(is_complex=True)
        top = result.get_top_n(top_k=6)
        assert top.shape[1] == 2
        assert top.shape[0] <= 6

    def test_caching_returns_same_array(self):
        result = _make_result(is_complex=False)
        top1 = result.get_top_n(top_k=3, include_negative=False)
        top2 = result.get_top_n(top_k=3, include_negative=False)
        assert top1 is top2  # same object from cache


# ===========================================================================
# DecompositionResult — save / load round-trip
# ===========================================================================


class TestDecompositionResultIO:
    """save / load: HDF5 round-trip for real and complex results."""

    def _assert_arrays_equal(self, a: np.ndarray, b: np.ndarray) -> None:
        np.testing.assert_array_equal(a, b)

    def test_round_trip_real(self, tmp_path: Path):
        result = _make_result(is_complex=False)
        path = tmp_path / "result_real.h5"
        result.save(path)
        loaded = DecompositionResult.load(path)

        assert loaded.k_max == result.k_max
        assert loaded.eig_max == result.eig_max
        assert loaded.is_complex_projection == result.is_complex_projection
        self._assert_arrays_equal(loaded.S, result.S)
        self._assert_arrays_equal(loaded.U, result.U)
        self._assert_arrays_equal(loaded.Vh, result.Vh)

    def test_round_trip_complex(self, tmp_path: Path):
        result = _make_result(is_complex=True)
        path = tmp_path / "result_complex.h5"
        result.save(path)
        loaded = DecompositionResult.load(path)

        assert loaded.is_complex_projection is True
        self._assert_arrays_equal(loaded.S, result.S)

    def test_extension_normalized_to_h5(self, tmp_path: Path):
        result = _make_result(is_complex=False)
        path = tmp_path / "result.txt"
        result.save(path)
        assert (tmp_path / "result.h5").exists()

    def test_load_wrong_extension_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unknown file extension"):
            DecompositionResult.load(tmp_path / "result.csv")

    def test_metadata_preserved(self, tmp_path: Path):
        result = _make_result(is_complex=False)
        result.phi_values = np.linspace(0, 360, NUM_OR, endpoint=False)
        result.theta_values = np.zeros(NUM_OR)
        path = tmp_path / "result_with_angles.h5"
        result.save(path)
        loaded = DecompositionResult.load(path)

        np.testing.assert_array_equal(loaded.phi_values, result.phi_values)
        np.testing.assert_array_equal(loaded.theta_values, result.theta_values)

    def test_no_coordinate_transform_raises_on_save(self, tmp_path: Path):
        result = _make_result(is_complex=False)
        result.coordinate_transform = None
        with pytest.raises(ValueError, match="coordinate_transform is None"):
            result.save(tmp_path / "no_transform.h5")
