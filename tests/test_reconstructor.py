"""approximation.Reconstructor: features, projections, kernels, adjoint, one warp."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import count_kernel_builds, requires_cuda

from panther_em.approximation import Reconstructor
from panther_em.coordinates import WarpConfig

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array_wrap__ must accept.*:DeprecationWarning",
    "ignore:.*__array__ implementation doesn't accept.*:DeprecationWarning",
)

NON_DEFAULT = WarpConfig(order=3, preserve_energy=False)


@pytest.fixture
def recon(synthetic_result):
    return Reconstructor(synthetic_result(), device="cpu")


def _cells(*pairs):
    return torch.tensor(pairs, dtype=torch.long)


def _per_cell(recon, cells):
    """Reference: one :meth:`cartesian_feature` per cell."""
    return torch.stack(
        [recon.cartesian_feature(int(k), int(m)) for k, m in cells.tolist()]
    ).to(torch.complex64)


def test_kernels_match_per_cell_features_and_keep_nothing(recon):
    cells = _cells((0, 0), (2, 3), (1, 1), (3, 4), (2, 0))
    built = count_kernel_builds(recon)
    got = recon.kernels(cells)
    expected = _per_cell(recon, cells)
    assert got.shape == (5, 20, 20) and got.dtype == torch.complex64
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
    assert built[0] == 5
    # request order and duplicates; built again on every call (no cache)
    again = recon.kernels(_cells((3, 4), (0, 0), (0, 0)))
    torch.testing.assert_close(again[0], expected[3])
    torch.testing.assert_close(again[1], expected[0])
    torch.testing.assert_close(again[2], expected[0])
    assert built[0] == 7  # (3, 4) and (0, 0) once each
    assert recon.kernels([]).shape == (0, 20, 20)


def test_the_warp_defaults_to_the_decompositions(synthetic_result):
    result = synthetic_result()
    assert Reconstructor(result).warp == WarpConfig()
    result.warp_config = NON_DEFAULT
    assert Reconstructor(result).warp == NON_DEFAULT
    assert Reconstructor(result, WarpConfig()).warp == WarpConfig()  # explicit wins


def test_a_non_default_warp_reaches_every_path(synthetic_result):
    """Features, projections, kernels, the generic warp and the support all warp
    with the reconstructor's own settings."""
    result = synthetic_result()
    default = Reconstructor(result)
    custom = Reconstructor(result, NON_DEFAULT)
    transform = custom.transform

    def warped(polar, **override):
        kwargs = {**NON_DEFAULT.cartesian_kwargs(), **override}
        return transform.to_cartesian(polar, **kwargs)

    polar = custom.polar_feature(1, 2)
    torch.testing.assert_close(custom.cartesian_feature(1, 2), warped(polar))
    assert not torch.allclose(
        default.cartesian_feature(1, 2), custom.cartesian_feature(1, 2)
    )

    projection = custom.reconstruct(3, polar=True)
    torch.testing.assert_close(custom.reconstruct(3), warped(projection))

    cells = _cells((1, 2), (3, 0))
    torch.testing.assert_close(
        custom.kernels(cells), _per_cell(custom, cells), rtol=1e-5, atol=1e-6
    )
    assert not torch.allclose(default.kernels(cells), custom.kernels(cells))

    real = projection.real[None].to(torch.float32)
    torch.testing.assert_close(custom.to_cartesian(real), warped(real).float())

    ones = torch.ones(transform.polar_shape)
    expected = warped(ones, preserve_energy=False) > 0.5
    assert torch.equal(custom.support, expected)


def test_reconstruct_sums_every_component_and_its_mirror(synthetic_result):
    result = synthetic_result()
    recon = Reconstructor(result)
    n_angle = result.num_angular_components
    n = np.arange(n_angle)
    expected = np.zeros((n_angle, result.num_radial_components), dtype=np.complex128)
    for k in range(result.k_max):
        a = np.exp(2j * np.pi * k * n / n_angle) / np.sqrt(n_angle)
        radial = (result.U[0, 0, 2, k] * result.S[k]) @ result.Vh[k]
        term = np.outer(a, radial)
        expected += term if k == 0 else term + term.conj()
    got = recon.reconstruct(2, polar=True)
    np.testing.assert_allclose(got.numpy(), expected, rtol=1e-4, atol=1e-5)

    # one component: the leading singular value's cell
    k, m = result.get_top_n(1, include_negative=False)[0]
    single = recon.reconstruct(2, num_components=1, polar=True)
    a = np.exp(2j * np.pi * k * n / n_angle) / np.sqrt(n_angle)
    term = np.outer(a, result.U[0, 0, 2, k, m] * result.S[k, m] * result.Vh[k, m])
    np.testing.assert_allclose(
        single.numpy(), term if k == 0 else term + term.conj(), rtol=1e-4, atol=1e-5
    )


def test_in_plane_rotation_rolls_the_angular_axis(recon):
    step = 360.0 / recon.result.num_angular_components
    base = recon.reconstruct(1, polar=True)
    rotated = recon.reconstruct(1, in_plane_rotation=3 * step, polar=True)
    torch.testing.assert_close(rotated, torch.roll(base, -3, dims=0), atol=1e-5, rtol=0)


def test_reconstruct_batch_matches_single_queries(recon):
    queries = [(0, 0, 0.0), (4, 0, 30.0), (2, 0, -45.0)]
    batch = recon.reconstruct_batch(queries)
    assert batch.shape == (3, 20, 20)
    for i, (o, f, rot) in enumerate(queries):
        torch.testing.assert_close(
            batch[i], recon.reconstruct(o, f, in_plane_rotation=rot)
        )
    with pytest.raises(ValueError, match="orientation"):
        recon.reconstruct(6)
    with pytest.raises(ValueError, match="fourier_filter"):
        recon.reconstruct(0, 1)
    with pytest.raises(ValueError, match="volume"):
        recon.reconstruct(0, volume=1)


def test_generic_warp_and_support(recon):
    # the generic warp of a kernel's polar image is the kernel (real part)
    polar = recon.polar_feature(1, 2)
    expected = recon.kernels(_cells((1, 2)))[0].real
    torch.testing.assert_close(
        recon.to_cartesian(polar.real[None])[0], expected, rtol=1e-5, atol=1e-6
    )
    support = recon.support
    assert support.shape == (20, 20) and support.dtype == torch.bool
    assert support[10, 10] and not support[0, 0]
    assert recon.support is support  # built once


def test_adjoint_is_built_once_and_released(recon):
    adjoint = recon.adjoint
    assert adjoint is not None and recon.adjoint is adjoint
    bases = recon.adjoint_bases((0, 0, 2, 3))
    assert (bases.n_k, bases.n_m) == (2, 3)
    recon.release()
    assert recon.adjoint is not adjoint


def test_kernels_need_real_projections(synthetic_result):
    result = synthetic_result()
    result.is_complex_projection = True
    recon = Reconstructor(result)
    with pytest.raises(NotImplementedError):
        recon.kernels(_cells((0, 0)))
    with pytest.raises(NotImplementedError):
        _ = recon.adjoint


@requires_cuda
def test_separable_build_matches_generic_on_a_side_stream(synthetic_result):
    """The CuPy fast path must agree with the generic warp, also when torch runs on a
    non-default stream (its DLPack views and CuPy work are stream-ordered)."""
    result = synthetic_result()
    fast = Reconstructor(result, device="cuda")
    slow = Reconstructor(result, device="cuda")
    slow.__dict__["_separable"] = None  # force the generic path
    cells = _cells((0, 0), (1, 3), (3, 4), (2, 1))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        a = fast.kernels(cells)
        _ = torch.randn(64 << 20, device="cuda")  # churn the allocator on the stream
    torch.cuda.synchronize()
    b = slow.kernels(cells)
    torch.cuda.synchronize()
    rel = ((a - b).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1)).max().item()
    # both paths mirror the same seam padding; the residual is spline arithmetic
    assert fast._separable is not None and rel < 1e-4, rel
