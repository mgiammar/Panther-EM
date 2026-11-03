"""The adjoint-warp featurization against the kernel bank's Cartesian kernels."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import make_synthetic_result, requires_cuda

from panther_em.coordinates.spiral_polar import SpiralPolarTransform
from panther_em.decomposition.result import DecompositionResult
from panther_em.inference.projection_reconstruction import ProjectionReconstructor
from panther_em.inference.search.adjoint import bspline, build_bases
from panther_em.inference.search.template import KernelBank
from panther_em.inference.search.types import KernelConfig

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array_wrap__ must accept.*:DeprecationWarning",
    "ignore:.*__array__ implementation doesn't accept.*:DeprecationWarning",
)


def _cells(rect):
    k0, m0, ke, me = rect
    return [(k, m) for k in range(k0, k0 + ke) for m in range(m0, m0 + me)]


def _spiral_result(box=24, n_angular=32, n_radial=10, **kw):
    """The synthetic decomposition on a spiral polar grid (origin continuation,
    mirrored outer rings, support mask)."""
    base = make_synthetic_result(
        box=box, n_angular=n_angular, n_radial=n_radial, k_max=6, eig_max=4, **kw
    )
    transform = SpiralPolarTransform.from_image(
        image_shape=(box, box), num_angle=n_angular, num_radius=n_radial
    )
    return DecompositionResult(
        U=base.U,
        S=base.S,
        Vh=base.Vh,
        k_max=base.k_max,
        eig_max=base.eig_max,
        is_complex_projection=False,
        num_fourier_filters=base.num_fourier_filters,
        num_orientations=base.num_orientations,
        num_angular_components=n_angular,
        num_radial_components=n_radial,
        coordinate_transform=transform,
        phi_values=base.phi_values,
        theta_values=base.theta_values,
    )


def _reference_features(bank, cells, boxes):
    """``<box, conj K>`` for every cell from the bank's spatial kernels, in fp64."""
    kernels = bank.spatial(cells).reshape(len(cells), -1).to(torch.complex128)
    return torch.conj(kernels) @ boxes.to(torch.complex128).T


def _adjoint_features(bank, rect, boxes, precision="fp32"):
    adjoint = bank.adjoint()
    assert adjoint is not None
    bases = bank.adjoint_bases(rect)
    return bases.features(adjoint.apply(boxes, precision), precision)


def test_bspline_partition_of_unity_and_scipy_weights():
    x = torch.linspace(-0.49, 0.49, 7, dtype=torch.float64)
    for order in (1, 2, 3, 4, 5):
        taps = torch.arange(-(order + 1), order + 2, dtype=torch.float64)
        w = bspline(order, x[:, None] - taps[None, :])
        assert torch.allclose(w.sum(1), torch.ones_like(x))
    # order 3 at the half sample: the classic 1/48, 23/48, 23/48, 1/48
    w3 = bspline(3, torch.tensor([1.5, 0.5, -0.5, -1.5], dtype=torch.float64))
    assert torch.allclose(w3, torch.tensor([1, 23, 23, 1], dtype=torch.float64) / 48)


@pytest.mark.parametrize("order", [5, 3])
@pytest.mark.parametrize("spiral", [False, True])
def test_adjoint_matches_the_banks_kernels(order, spiral):
    result = _spiral_result() if spiral else make_synthetic_result(box=24, n_angular=32)
    recon = ProjectionReconstructor(result, device="cpu")
    bank = KernelBank(recon, KernelConfig(order=order))
    rect = (0, 0, 4, 3)
    cells = _cells(rect)
    gen = torch.Generator().manual_seed(1)
    boxes = torch.randn(5, 24 * 24, generator=gen)
    ref = _reference_features(bank, cells, boxes)
    got = _adjoint_features(bank, rect, boxes)
    assert got.shape == (len(cells), 5)
    err = (got.to(torch.complex128) - ref).abs().max() / ref.abs().max()
    assert err < 2e-5, err


def test_adjoint_forward_model_reproduces_a_kernel():
    """``S (filt(a) (x) filt(c))`` masked equals ``bank.spatial`` for a spiral cell."""
    recon = ProjectionReconstructor(_spiral_result(), device="cpu")
    bank = KernelBank(recon)
    adjoint = bank.adjoint()
    assert adjoint is not None
    rect = (2, 1, 1, 1)
    bases = build_bases(adjoint, recon, rect)
    k_count = bases.n_k
    a = torch.complex(bases.angular[:k_count], bases.angular[k_count:]).to(
        torch.complex128
    )
    c = torch.complex(bases.radial_re, bases.radial_im).to(torch.complex128)
    polar = (a[0][:, None] * c[0, 0][None, :]).reshape(-1)
    forward = adjoint.scatter.to_sparse_coo().t().to(torch.complex128) @ polar
    kernel = bank.spatial([(2, 1)])[0].reshape(-1).to(torch.complex128)
    err = (forward - kernel).abs().max() / kernel.abs().max()
    assert err < 2e-5, err


def test_unsupported_configs_fall_back_to_none():
    recon = ProjectionReconstructor(make_synthetic_result(), device="cpu")
    assert KernelBank(recon, KernelConfig(mode="edge")).adjoint() is None
    assert KernelBank(recon, KernelConfig(cval=1.0)).adjoint() is None
    assert KernelBank(recon, KernelConfig(order=0)).adjoint() is None
    bank = KernelBank(recon)
    with pytest.raises(ValueError, match="unsupported"):
        KernelBank(recon, KernelConfig(order=0)).adjoint_bases((0, 0, 1, 1))
    assert bank.adjoint() is not None


def test_spiral_radial_continuation_matches_pad_radial_axis():
    transform = SpiralPolarTransform.from_image(
        image_shape=(40, 40), num_angle=48, num_radius=16
    )
    cont = transform.radial_continuation([0, 3, 7], "cpu")
    n_angle, n_radius = transform.polar_shape
    n = torch.arange(n_angle, dtype=torch.float64)
    gen = torch.Generator().manual_seed(0)
    c = torch.complex(
        torch.randn(n_radius, generator=gen, dtype=torch.float64),
        torch.randn(n_radius, generator=gen, dtype=torch.float64),
    )
    for i, k in enumerate((0, 3, 7)):
        a = torch.exp(1j * 2 * np.pi * k * n / n_angle)
        for part in (lambda z: z.real, lambda z: z.imag):
            padded, n_pre = transform.pad_radial_axis(
                part(a[:, None] * c[None, :]).float()
            )
            inner = (cont[i].to(torch.complex128) @ c[: cont.shape[-1]]).flip(0)
            expected = part(a[:, None] * inner[None, :])
            assert n_pre == cont.shape[1]
            assert torch.allclose(padded[:, :n_pre].double(), expected, atol=1e-5)


@requires_cuda
def test_cuda_adjoint_matches_cpu_kernels_in_every_precision():
    result = _spiral_result()
    recon = ProjectionReconstructor(result, device="cuda")
    bank = KernelBank(recon, device="cuda")
    rect = (1, 0, 3, 4)
    cells = _cells(rect)
    gen = torch.Generator().manual_seed(2)
    boxes = torch.randn(8, 24 * 24, generator=gen).cuda()
    ref = _reference_features(bank, cells, boxes)
    for precision, tol in (("fp32", 2e-5), ("tf32", 3e-3), ("fp16", 3e-3)):
        got = _adjoint_features(bank, rect, boxes, precision)
        err = ((got.to(torch.complex128) - ref).abs().max() / ref.abs().max()).item()
        assert err < tol, (precision, err)
