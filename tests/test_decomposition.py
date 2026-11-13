"""Unit tests for the decomposition pipeline and result classes.

Covers:
- _compute_freq_crop: index math for k_max cropping (real and complex modes)
- precompute_volume_dft: shape, padding, edge-value handling
- apply_fourier_filters: output shape and identity behavior
- the pipeline's block order (complex: centered on DC) and recorded block energies
- DecompositionResult: shape validation, index helpers, get_u/get_s/get_vh,
  get_top_n
- DecompositionResult save/load round-trip
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from panther_em.coordinates.spiral_polar import SpiralPolarTransform
from panther_em.coordinates.warp_config import WarpConfig
from panther_em.decomposition import PolarProjectionDecomposer
from panther_em.decomposition.projections import (
    _compute_freq_crop,
    apply_fourier_filters,
    do_pipelined_projection_and_transforms,
    precompute_volume_dft,
    process_batch,
)
from panther_em.decomposition.result import (
    DecompositionBase,
    DecompositionResult,
    SparseDecompositionResult,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NUM_VOL = 3  # volumes
NUM_FF = 2  # fourier filters
NUM_OR = 8  # orientations
K_MAX = 5
EIG_MAX = 4
NUM_R = 10  # radial components
NUM_ANG = 32  # angular components in polar space


def _make_result(is_complex: bool, num_volumes: int = NUM_VOL) -> DecompositionResult:
    """Minimal DecompositionResult with deterministic random arrays."""
    rng = np.random.default_rng(0)
    num_freq_blocks = K_MAX * 2 if is_complex else K_MAX

    # Use complex arrays even for real-valued decompositions so conjugation paths are
    # tested.
    U = (
        rng.standard_normal(
            (num_volumes, NUM_FF, NUM_OR, num_freq_blocks, EIG_MAX)
        ).astype(np.float32)
        + 1j
        * rng.standard_normal(
            (num_volumes, NUM_FF, NUM_OR, num_freq_blocks, EIG_MAX)
        ).astype(np.float32)
    ).astype(np.complex64)
    S = rng.random((num_freq_blocks, EIG_MAX)).astype(np.float32)
    Vh = (
        rng.standard_normal((num_freq_blocks, EIG_MAX, NUM_R)).astype(np.float32)
        + 1j * rng.standard_normal((num_freq_blocks, EIG_MAX, NUM_R)).astype(np.float32)
    ).astype(np.complex64)
    transform = SpiralPolarTransform.from_image(
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
        num_volumes=num_volumes,
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
    """precompute_volume_dft: shape, padding, edge-value handling."""

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

    def test_dc_is_not_zeroed(self):
        vol = torch.zeros(16, 16, 16)
        vol[6:10, 6:10, 6:10] = 5.0
        dft, _, _ = precompute_volume_dft(vol, pad_factor=1.0)
        dc_row = dft.shape[0] // 2
        dc_col = dft.shape[1] // 2
        assert dft[dc_row, dc_col, 0].abs().item() > 1e-3

    def test_edge_value_scaled_matches_edge_average_times_d(self):
        d = 16
        vol = torch.ones(d, d, d) * 3.0
        _, edge_value_scaled, _ = precompute_volume_dft(vol, pad_factor=1.0)
        # Constant volume: cube-face average equals the constant value.
        assert edge_value_scaled == pytest.approx(3.0 * d, rel=1e-4)

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


class TestPipelineBlockOrder:
    """The pipeline crops the angular blocks around DC and records their energies."""

    def _case(self, filters):
        rng = np.random.default_rng(0)
        volume = torch.from_numpy(
            gaussian_filter(rng.standard_normal((16,) * 3), 1.5).astype(np.float32)
        )
        transform = SpiralPolarTransform.from_image(
            image_shape=(16, 16), num_angle=16, num_radius=6
        )
        phi = torch.tensor([0.0, 40.0, 90.0, 150.0])
        theta = torch.tensor([10.0, 30.0, 60.0, 80.0])
        out = do_pipelined_projection_and_transforms(
            volume[None],
            phi,
            theta,
            torch.zeros(4),
            filters,
            transform,
            WarpConfig(),
            projection_batch_size=4,
            k_max=3,
            show_progress=False,
        )
        dft, edge, pad = precompute_volume_dft(volume)
        singles = [
            process_batch(
                dft,
                edge,
                pad,
                phi[i : i + 1],
                theta[i : i + 1],
                torch.zeros(1),
                filters,
                transform,
                WarpConfig(),
            )[0][0, 0]
            for i in range(4)
        ]
        return out, singles

    def test_complex_blocks_centered_on_dc(self):
        """Complex projections: the blocks [-3, 3) of each orientation, in order."""
        ramp = torch.exp(1j * torch.linspace(0.0, 1.0, 16))
        filters = (torch.ones(1, 16, 16) * ramp).to(torch.complex64)
        (out, is_complex, k_max, energy), singles = self._case(filters)
        assert is_complex and k_max == 3 and out.shape == (1, 1, 4, 6, 6)
        for i, single in enumerate(singles):
            shifted = torch.fft.fftshift(single, dim=0)
            torch.testing.assert_close(out[0, 0, i], shifted[8 - 3 : 8 + 3])
            torch.testing.assert_close(
                energy[0, 0, i],
                shifted.abs().square().sum(-1).double(),
                rtol=1e-5,
                atol=0,
            )

    def test_real_energies_before_the_crop(self):
        filters = torch.ones(1, 16, 9, dtype=torch.complex64)
        (out, is_complex, _, energy), singles = self._case(filters)
        assert not is_complex and energy.shape == (1, 1, 4, 9)
        for i, single in enumerate(singles):
            torch.testing.assert_close(out[0, 0, i], single[:3])
            torch.testing.assert_close(
                energy[0, 0, i],
                single.abs().square().sum(-1).double(),
                rtol=1e-5,
                atol=0,
            )


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
                U=rng.standard_normal((NUM_VOL, NUM_FF, NUM_OR, K_MAX, EIG_MAX)).astype(
                    np.float32
                ),
                S=rng.random((K_MAX + 1, EIG_MAX)).astype(np.float32),  # wrong
                Vh=rng.standard_normal((K_MAX, EIG_MAX, NUM_R)).astype(np.float32),
                k_max=K_MAX,
                eig_max=EIG_MAX,
                is_complex_projection=False,
                num_volumes=NUM_VOL,
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
                U=rng.standard_normal(
                    (NUM_VOL, NUM_FF, NUM_OR, K_MAX + 1, EIG_MAX)
                ).astype(np.float32),  # wrong k dim
                S=rng.random((K_MAX, EIG_MAX)).astype(np.float32),
                Vh=rng.standard_normal((K_MAX, EIG_MAX, NUM_R)).astype(np.float32),
                k_max=K_MAX,
                eig_max=EIG_MAX,
                is_complex_projection=False,
                num_volumes=NUM_VOL,
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
                U=rng.standard_normal((NUM_VOL, NUM_FF, NUM_OR, K_MAX, EIG_MAX)).astype(
                    np.float32
                ),
                S=rng.random((K_MAX, EIG_MAX)).astype(np.float32),
                Vh=rng.standard_normal((K_MAX, EIG_MAX, NUM_R + 1)).astype(
                    np.float32
                ),  # wrong last dim
                k_max=K_MAX,
                eig_max=EIG_MAX,
                is_complex_projection=False,
                num_volumes=NUM_VOL,
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
                num_volumes=NUM_VOL,
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
# DecompositionBase — get_u / get_s / get_vh
# ===========================================================================


class TestComponentGetters:
    """get_u / get_s / get_vh: shapes, conjugation, dense/sparse parity."""

    @pytest.mark.parametrize(("is_complex", "k"), [(False, 1), (True, 2)])
    def test_scalar_indices(self, is_complex: bool, k: int):
        result = _make_result(is_complex=is_complex)
        assert result.get_u(k, 0).shape == (NUM_VOL, NUM_FF, NUM_OR, 1)
        assert result.get_s(k, 0).shape == (1,)
        assert result.get_vh(k, 0).shape == (1, NUM_R)

    def test_array_indices_match_storage(self):
        result = _make_result(is_complex=False)
        ks, ms = np.array([0, 2, 1]), np.array([1, 0, 2])
        np.testing.assert_array_equal(result.get_u(ks, ms), result.U[..., ks, ms])
        np.testing.assert_array_equal(result.get_s(ks, ms), result.S[ks, ms])
        np.testing.assert_array_equal(result.get_vh(ks, ms), result.Vh[ks, ms])

    def test_scalar_k_broadcasts_over_eig(self):
        result = _make_result(is_complex=False)
        sparse = result.to_sparse(n=K_MAX * EIG_MAX)
        ms = np.arange(EIG_MAX)
        for r in (result, sparse):
            assert r.get_s(1, ms).shape == (EIG_MAX,)
            np.testing.assert_array_equal(r.get_s(1, ms), result.S[1, ms])

    def test_negative_k_conjugates_real(self):
        result = _make_result(is_complex=False)
        np.testing.assert_array_equal(result.get_vh(-2, 0), result.get_vh(2, 0).conj())
        np.testing.assert_array_equal(result.get_u(-2, 0), result.get_u(2, 0).conj())
        np.testing.assert_array_equal(result.get_s(-2, 0), result.get_s(2, 0))

    def test_negative_k_no_conjugate_complex(self):
        result = _make_result(is_complex=True)
        k_stored = result.k_to_stored(-2)
        np.testing.assert_array_equal(result.get_vh(-2, 0)[0], result.Vh[k_stored, 0])

    def test_conjugation_does_not_mutate_storage(self):
        result = _make_result(is_complex=False)
        u_before, vh_before = result.U.copy(), result.Vh.copy()
        result.get_u(-2, 0)
        result.get_vh(-2, 0)
        np.testing.assert_array_equal(result.U, u_before)
        np.testing.assert_array_equal(result.Vh, vh_before)

    def test_sparse_matches_dense_including_missing_pairs(self):
        result = _make_result(is_complex=False)
        sparse = result.to_sparse(n=5)
        truncated = sparse.to_dense()
        # every pair, including negative k and pairs outside the stored top 5
        ks, ms = np.meshgrid(
            np.arange(-(K_MAX - 1), K_MAX), np.arange(EIG_MAX), indexing="ij"
        )
        ks, ms = ks.reshape(-1), ms.reshape(-1)
        s = sparse.get_s(ks, ms)
        assert (s == 0).sum() > 0  # some pairs are outside the stored set
        np.testing.assert_array_equal(s, truncated.get_s(ks, ms))
        np.testing.assert_array_equal(sparse.get_u(ks, ms), truncated.get_u(ks, ms))
        np.testing.assert_array_equal(sparse.get_vh(ks, ms), truncated.get_vh(ks, ms))

    def test_get_svd_tensors_matches_getters(self):
        result = _make_result(is_complex=False)
        indices = np.array([[1, 0], [-2, 1]])
        u, s, vh = result.get_svd_tensors(indices, "cpu")
        ks, ms = indices[:, 0], indices[:, 1]
        np.testing.assert_array_equal(u.numpy(), result.get_u(ks, ms))
        np.testing.assert_array_equal(s.numpy(), result.get_s(ks, ms))
        np.testing.assert_array_equal(vh.numpy(), result.get_vh(ks, ms))


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

    def test_legacy_file_without_num_volumes_loads_as_single_volume(
        self, tmp_path: Path
    ):
        """Files saved before the multi-volume axis existed lack the
        ``num_volumes`` attr and store a 4D ``U`` array; loading must default
        to ``num_volumes=1`` and reinsert the leading axis."""
        result = _make_result(is_complex=False, num_volumes=1)
        path = tmp_path / "legacy_result.h5"
        result.save(path)

        # Rewrite the file to mimic the pre-multi-volume on-disk layout: drop
        # the num_volumes attr and squeeze U back down to 4D.
        with h5py.File(path, "r+") as f:
            del f.attrs["num_volumes"]
            legacy_U = f["U"][()][0]
            del f["U"]
            f.create_dataset("U", data=legacy_U)

        for mmap in (False, True):
            with DecompositionResult.load(path, mmap=mmap) as loaded:
                assert loaded.num_volumes == 1
                assert loaded.U.shape == (1, NUM_FF, NUM_OR, K_MAX, EIG_MAX)
                np.testing.assert_array_equal(loaded.U[0], result.U[0])

        # the sparse layout's legacy U_data is 3D
        sparse_path = tmp_path / "legacy_sparse.h5"
        result.to_sparse(5).save(sparse_path)
        with h5py.File(sparse_path, "r+") as f:
            legacy_U = f["U"][()][0]
            del f["U"]
            f.create_dataset("U", data=legacy_U)
        for mmap in (False, True):
            with SparseDecompositionResult.load(sparse_path, mmap=mmap) as loaded:
                assert loaded.num_volumes == 1
                np.testing.assert_array_equal(
                    loaded.to_dense().U, result.to_sparse(5).to_dense().U
                )

    def test_warp_config_round_trips_dense_and_sparse(self, tmp_path: Path):
        warp = WarpConfig(order=3, mode="nearest", preserve_energy=False)
        result = _make_result(is_complex=False)
        assert result.warp_config == WarpConfig()
        result.warp_config = warp
        result.save(tmp_path / "dense.h5")
        assert DecompositionResult.load(tmp_path / "dense.h5").warp_config == warp

        sparse = result.to_sparse(5)
        assert sparse.warp_config == warp
        assert sparse.to_dense().warp_config == warp
        sparse.save(tmp_path / "sparse.h5")
        assert DecompositionBase.load(tmp_path / "sparse.h5").warp_config == warp

    def test_legacy_file_without_warp_config_loads_the_default(self, tmp_path: Path):
        result = _make_result(is_complex=False)
        result.warp_config = WarpConfig(order=1)
        path = tmp_path / "no_warp.h5"
        result.save(path)
        with h5py.File(path, "r+") as f:
            del f.attrs["warp_config"]
        assert DecompositionResult.load(path).warp_config == WarpConfig()


# ===========================================================================
# DecompositionResult — multi-volume shapes
# ===========================================================================


class TestMultiVolumeShapes:
    """Shape validation and index helpers with num_volumes > 1."""

    def test_construction_valid(self):
        result = _make_result(is_complex=False, num_volumes=NUM_VOL)
        assert result.num_volumes == NUM_VOL
        assert result.U.shape[0] == NUM_VOL

    def test_wrong_volume_dim_raises(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="U shape"):
            DecompositionResult(
                U=rng.standard_normal(
                    (NUM_VOL + 1, NUM_FF, NUM_OR, K_MAX, EIG_MAX)
                ).astype(np.float32),
                S=rng.random((K_MAX, EIG_MAX)).astype(np.float32),
                Vh=rng.standard_normal((K_MAX, EIG_MAX, NUM_R)).astype(np.float32),
                k_max=K_MAX,
                eig_max=EIG_MAX,
                is_complex_projection=False,
                num_volumes=NUM_VOL,
                num_fourier_filters=NUM_FF,
                num_orientations=NUM_OR,
                num_angular_components=NUM_ANG,
                num_radial_components=NUM_R,
                coordinate_transform=None,
            )

    def test_get_u_includes_volume_axis(self):
        result = _make_result(is_complex=False, num_volumes=NUM_VOL)
        u = result.get_u(k_idx=1, eig_idx=0)
        assert u.shape == (NUM_VOL, NUM_FF, NUM_OR, 1)

    def test_volumes_have_independent_left_singular_vectors(self):
        result = _make_result(is_complex=False, num_volumes=NUM_VOL)
        u = result.get_u(k_idx=1, eig_idx=0)
        # Different volumes were filled with independent random data, so their
        # left-singular-vector slices should differ.
        assert not np.allclose(u[0], u[1])

    def test_save_load_round_trip_multi_volume(self, tmp_path: Path):
        result = _make_result(is_complex=False, num_volumes=NUM_VOL)
        path = tmp_path / "multi_volume.h5"
        result.save(path)
        loaded = DecompositionResult.load(path)

        assert loaded.num_volumes == NUM_VOL
        np.testing.assert_array_equal(loaded.U, result.U)

    def test_sparse_round_trip_multi_volume(self, tmp_path: Path):
        result = _make_result(is_complex=False, num_volumes=NUM_VOL)
        sparse = result.to_sparse(n=5)
        assert sparse.num_volumes == NUM_VOL

        path = tmp_path / "sparse_multi_volume.h5"
        sparse.save(path)

        loaded = SparseDecompositionResult.load(path)
        assert loaded.num_volumes == NUM_VOL

        dense = loaded.to_dense()
        assert dense.U.shape[0] == NUM_VOL


class TestDenseAndSparseAreSiblings:
    """Dense and sparse results share DecompositionBase, not each other."""

    def test_sparse_is_not_a_dense_result(self):
        sparse = _make_result(is_complex=False).to_sparse(n=5)
        assert isinstance(sparse, DecompositionBase)
        assert not isinstance(sparse, DecompositionResult)
        assert not hasattr(sparse, "U")  # absent, rather than a property that raises

    def test_load_dispatches_on_the_is_sparse_flag(self, tmp_path: Path):
        result = _make_result(is_complex=False)
        result.save(tmp_path / "dense.h5")
        result.to_sparse(n=5).save(tmp_path / "sparse.h5")

        assert (
            type(DecompositionBase.load(tmp_path / "dense.h5")) is DecompositionResult
        )
        loaded = DecompositionBase.load(tmp_path / "sparse.h5")
        assert type(loaded) is SparseDecompositionResult

    def test_concrete_load_rejects_the_other_kind(self, tmp_path: Path):
        result = _make_result(is_complex=False)
        result.save(tmp_path / "dense.h5")
        result.to_sparse(n=5).save(tmp_path / "sparse.h5")

        with pytest.raises(ValueError, match="contains a sparse result"):
            DecompositionResult.load(tmp_path / "sparse.h5")
        with pytest.raises(ValueError, match="contains a dense result"):
            SparseDecompositionResult.load(tmp_path / "dense.h5")

    def test_sparse_validates_metadata_like_dense(self):
        result = _make_result(is_complex=False)
        result.phi_values = np.zeros(NUM_OR + 1)
        with pytest.raises(ValueError, match="phi_values shape"):
            result.to_sparse(n=5)

    def test_sparse_rejects_inconsistent_arrays(self):
        sparse = _make_result(is_complex=False).to_sparse(n=5)
        with pytest.raises(ValueError, match="inconsistent with selected_indices"):
            SparseDecompositionResult.from_sparse_arrays(
                U_data=sparse._U_data[..., :-1],
                S_data=sparse._S_data,
                Vh_data=sparse._Vh_data,
                selected_indices=sparse._selected_indices,
                k_max=sparse.k_max,
                eig_max=sparse.eig_max,
                is_complex_projection=sparse.is_complex_projection,
                num_fourier_filters=sparse.num_fourier_filters,
                num_orientations=sparse.num_orientations,
                num_angular_components=sparse.num_angular_components,
                num_radial_components=sparse.num_radial_components,
                num_volumes=sparse.num_volumes,
            )


# ===========================================================================
# PolarProjectionDecomposer — the recorded warp
# ===========================================================================


def test_decomposer_warps_with_and_records_its_warp_config():
    rng = np.random.default_rng(0)
    volume = rng.standard_normal((16, 16, 16)).astype(np.float32)
    phi = np.array([0.0, 40.0, 80.0], dtype=np.float32)
    theta = np.array([10.0, 30.0, 50.0], dtype=np.float32)
    transform = SpiralPolarTransform.from_image(
        image_shape=(16, 16), num_angle=16, num_radius=6
    )

    def run(warp):
        return PolarProjectionDecomposer(
            volume, phi, theta, coordinate_transform=transform, warp=warp
        ).do_decomposition(eig_max=3)

    default = run(None)
    cubic = run(WarpConfig(order=3))
    assert default.warp_config == WarpConfig()
    assert cubic.warp_config == WarpConfig(order=3)
    # the warp is applied, not only recorded
    assert not np.allclose(default.S, cubic.S)


@pytest.mark.parametrize("svd_method", ["dense", "low_rank"])
def test_decomposer_eig_max_is_bounded_by_the_block_rank(svd_method):
    """Fewer rows than radii: eig_max defaults to, and may not exceed, the rows."""
    rng = np.random.default_rng(0)
    volume = rng.standard_normal((16, 16, 16)).astype(np.float32)
    phi = np.array([0.0, 40.0, 80.0], dtype=np.float32)
    theta = np.array([10.0, 30.0, 50.0], dtype=np.float32)
    transform = SpiralPolarTransform.from_image(
        image_shape=(16, 16), num_angle=16, num_radius=6
    )
    decomposer = PolarProjectionDecomposer(
        volume, phi, theta, coordinate_transform=transform
    )
    result = decomposer.do_decomposition(svd_method=svd_method)
    assert result.eig_max == 3 and result.U.shape[-1] == 3
    with pytest.raises(ValueError, match=r"eig_max must be in the range \[1, 3\]"):
        decomposer.do_decomposition(eig_max=4, svd_method=svd_method)
