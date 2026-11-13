"""approximation.energies and .selection: the cheap tier and the rank suggestions."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from panther_em.approximation import (
    Reconstructor,
    compute_errors,
    energy_spectrum,
    suggest_rectangles,
    truncation_error,
)
from panther_em.approximation.energies import parseval_weights
from panther_em.coordinates import SpiralPolarTransform
from panther_em.decomposition import (
    DecompositionBase,
    DecompositionResult,
    PolarProjectionDecomposer,
)

BOX = 32


@pytest.fixture(scope="module")
def fine():
    """``(volume, result)``: 64 orientations on a 128 x 32 polar grid of a 32 box.

    Fine enough that the warp loses little (round-trip floor ~0.08 pooled); every
    angular block is stored.
    """
    rng = np.random.default_rng(0)
    volume = gaussian_filter(rng.standard_normal((BOX,) * 3), 2.0).astype(np.float32)
    phi = np.linspace(0.0, 300.0, 64, dtype=np.float32)
    theta = np.linspace(10.0, 80.0, 64, dtype=np.float32)
    transform = SpiralPolarTransform.from_image(
        image_shape=(BOX, BOX), num_angle=128, num_radius=32
    )
    result = PolarProjectionDecomposer(
        volume, phi, theta, coordinate_transform=transform
    ).do_decomposition(normalize_projections=True, svd_method="dense")
    return volume, result


def test_parseval_weights():
    np.testing.assert_array_equal(parseval_weights(8, 5), [1, 2, 2, 2, 1])
    np.testing.assert_array_equal(parseval_weights(7, 4), [1, 2, 2, 2])


def test_decomposer_records_the_energies(fine):
    _, result = fine
    assert result.normalized_projections is True
    assert result.num_angular_blocks == 65
    assert result.block_energy.shape == (1, 1, 64, 65)
    # Parseval: every stored cell together captures each row's whole energy
    full = truncation_error(result, [(0, 0, result.k_max, result.eig_max)])
    assert full.rho.max() < 2e-3


def test_split_by_hand(synthetic_result):
    """``rho_k`` the unreached blocks, ``rho_m`` the reached blocks' left-out cells."""
    result = synthetic_result(k_max=4, eig_max=5, n_angular=24)  # 13 blocks
    w = parseval_weights(24, 13)
    us2 = np.abs(result.U[0, 0]) ** 2 * result.S**2  # (6, 4, 5)
    extra = np.arange(1, 14, dtype=np.float64)
    blocks = np.zeros((6, 13))
    blocks[:, :4] = us2.sum(-1)
    blocks += extra
    result.block_energy = blocks[None, None]

    err = truncation_error(result, [(0, 0, 2, 3)])
    energy = (blocks * w).sum(1)
    angular = (blocks[:, 2:] * w[2:]).sum(1)
    eigen = (w[:2] * (us2[:, :2, 3:].sum(-1) + extra[:2])).sum(1)
    np.testing.assert_allclose(err.energy, energy)
    np.testing.assert_allclose(err.angular_loss, angular)
    np.testing.assert_allclose(err.eigen_loss, eigen, rtol=1e-5)
    np.testing.assert_allclose(err.rho**2, err.rho_k**2 + err.rho_m**2)

    # two rectangles reaching the same blocks: the union of their cells
    both = truncation_error(result, [(0, 0, 2, 3), (2, 0, 1, 1)])
    assert both.rectangles == ((0, 0, 2, 3), (2, 0, 1, 1))
    assert both.angular_loss.sum() < err.angular_loss.sum()


def test_guards(synthetic_result):
    result = synthetic_result()
    with pytest.raises(ValueError, match="no block energies"):
        truncation_error(result, [(0, 0, 2, 2)])
    result.block_energy = np.ones((1, 1, 6, 13))
    with pytest.raises(ValueError, match="beyond the stored"):
        truncation_error(result, [(0, 0, 5, 2)])
    fields = {"U": result.U, "S": result.S, "Vh": result.Vh, **result._metadata()}
    with pytest.raises(ValueError, match="block_energy shape"):
        DecompositionResult(**{**fields, "block_energy": np.ones((1, 1, 6, 12))})


def test_cheap_tier_matches_the_exact_errors(fine):
    """Exact Cartesian error^2 = polar truncation error^2 + the warp's floor^2."""
    volume, result = fine
    reconstructor = Reconstructor(result)
    floor = compute_errors(
        volume, reconstructor, [(0, 0, result.k_max, result.eig_max)]
    ).by_orientation()
    assert 0 < float(floor.max()) < 0.15
    for rect in [
        (0, 0, 8, 8),
        (0, 0, 4, 2),
        (0, 0, 12, 3),
        (0, 0, 65, 4),
        (2, 0, 6, 5),
    ]:
        cheap = truncation_error(result, [rect])
        exact = compute_errors(volume, reconstructor, [rect]).by_orientation().numpy()
        estimate = np.sqrt(np.maximum(exact**2 - floor.numpy() ** 2, 0.0))
        np.testing.assert_allclose(estimate, cheap.rho, atol=0.015)


def test_energy_spectrum_grid(fine):
    _, result = fine
    spectrum = energy_spectrum(result)
    rho, rho_k, rho_m = spectrum.rho_grid()
    assert rho.shape == rho_k.shape == rho_m.shape == (65, 32)
    for k, m in [(1, 1), (8, 8), (65, 4), (17, 32)]:
        summary = truncation_error(result, [(0, 0, k, m)]).summary()
        # differences of float32 energies: rho resolves to ~1e-3
        assert rho[k - 1, m - 1] == pytest.approx(summary["rho"], abs=1e-3)
        assert rho_k[k - 1, m - 1] == pytest.approx(summary["rho_k"], abs=1e-3)
        assert rho_m[k - 1, m - 1] == pytest.approx(summary["rho_m"], abs=1e-3)
    assert np.all(np.diff(rho, axis=0) <= 1e-12)
    assert np.all(np.diff(rho, axis=1) <= 1e-12)


def test_suggest_for_an_error_target(fine):
    _, result = fine
    spectrum = energy_spectrum(result)
    rho = spectrum.rho_grid()[0]
    target = 0.2
    frontier = suggest_rectangles(spectrum, max_error=target)
    assert frontier and all(c.rho <= target for c in frontier)
    sizes = [c.components for c in frontier]
    assert sizes == sorted(sizes)
    assert all(a.rho > b.rho for a, b in itertools.pairwise(frontier))
    # the cheapest candidate is the fewest cells of any leading corner meeting it
    k, m = np.nonzero(rho <= target)
    assert frontier[0].components == int(((k + 1) * (m + 1)).min())
    for c in frontier:
        _, _, ke, me = c.rectangle
        assert c.components == ke * me
        assert c.rho**2 == pytest.approx(c.rho_k**2 + c.rho_m**2, rel=1e-6)

    assert suggest_rectangles(result, max_error=target) == frontier
    capped = suggest_rectangles(spectrum, max_error=target, max_k_extent=10)
    assert all(c.rectangle[2] <= 10 for c in capped)
    assert suggest_rectangles(spectrum, max_error=0.01, max_k_extent=2) == []


def test_suggest_for_a_budget(fine):
    _, result = fine
    frontier = suggest_rectangles(result, max_components=200)
    assert frontier and all(c.components <= 200 for c in frontier)
    rho = energy_spectrum(result).rho_grid()[0]
    best = min(rho[k - 1, min(32, 200 // k) - 1] for k in range(1, 66) if 200 // k)
    assert frontier[-1].rho == pytest.approx(best)

    both = suggest_rectangles(result, max_components=200, max_error=0.3)
    assert all(c.components <= 200 and c.rho <= 0.3 for c in both)
    with pytest.raises(ValueError, match="give max_error"):
        suggest_rectangles(result)


def test_energies_round_trip(fine, tmp_path):
    _, result = fine
    result.save(tmp_path / "d.h5")
    loaded = DecompositionBase.load(tmp_path / "d.h5")
    np.testing.assert_array_equal(loaded.block_energy, result.block_energy)
    assert loaded.normalized_projections is True

    sparse = result.to_sparse(50)
    sparse.save(tmp_path / "s.h5")
    for r in (sparse, DecompositionBase.load(tmp_path / "s.h5"), sparse.to_dense()):
        np.testing.assert_array_equal(r.block_energy, result.block_energy)
        assert r.normalized_projections is True
    # a sparse result counts only the components it stores
    rect = [(0, 0, result.k_max, result.eig_max)]
    assert (
        truncation_error(sparse, rect).summary()["rho"]
        > truncation_error(result, rect).summary()["rho"]
    )
