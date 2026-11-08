"""TemplateWarp / WeightSource / Template (panther_em.inference.template)."""

from __future__ import annotations

import pytest
import torch
from conftest import count_kernel_builds, requires_cuda

from panther_em.inference.search.correlation import FeatureLayout
from panther_em.inference.template import (
    KernelConfig,
    Template,
    TemplateWarp,
    WeightSource,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array_wrap__ must accept.*:DeprecationWarning",
    "ignore:.*__array__ implementation doesn't accept.*:DeprecationWarning",
)


def build_block_kernels(recon, cells, **kwargs):
    """Reference: one ``construct_cartesian_feature`` call per cell."""
    return torch.stack(
        [
            torch.as_tensor(
                recon.construct_cartesian_feature(
                    int(k), int(m), return_torch=True, **kwargs
                )
            ).to(torch.complex64)
            for k, m in cells.tolist()
        ]
    )


def reference_layout_weights(result, layout):
    """Reference ``U * S`` in a layout's slot order, straight from the result arrays."""
    us = torch.as_tensor(result.U[0]) * torch.as_tensor(result.S)[None, None]
    ff, o = us.shape[:2]
    blocks = [
        us[:, :, k0 : k0 + ke, m0 : m0 + me].reshape(ff * o, ke * me)
        for k0, m0, ke, me in layout.rectangles
    ]
    return torch.cat(blocks, dim=1).to(torch.complex64)


@pytest.fixture
def warp(synthetic_result):
    return TemplateWarp(synthetic_result(), device="cpu")


def _cells(*pairs):
    return torch.tensor(pairs, dtype=torch.long)


def test_spatial_matches_per_cell_construction_and_keeps_nothing(warp):
    cells = _cells((0, 0), (2, 3), (1, 1), (3, 4), (2, 0))
    built = count_kernel_builds(warp)
    got = warp.spatial(cells)
    expected = build_block_kernels(warp.reconstructor, cells)
    assert got.shape == (5, 20, 20) and got.dtype == torch.complex64
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
    assert built[0] == 5
    # request order and duplicates; built again on every call (no cache)
    again = warp.spatial(_cells((3, 4), (0, 0), (0, 0)))
    torch.testing.assert_close(again[0], expected[3])
    torch.testing.assert_close(again[1], expected[0])
    torch.testing.assert_close(again[2], expected[0])
    assert built[0] == 7  # (3, 4) and (0, 0) once each
    assert warp.spatial([]).shape == (0, 20, 20)


def test_spatial_respects_the_kernel_config(synthetic_result):
    result = synthetic_result()
    cells = _cells((1, 2))
    default = TemplateWarp(result, device="cpu").spatial(cells)
    linear_warp = TemplateWarp(
        result, KernelConfig(order=1, preserve_energy=False), device="cpu"
    )
    linear = linear_warp.spatial(cells)
    assert not torch.allclose(default, linear)
    expected = build_block_kernels(
        linear_warp.reconstructor, cells, order=1, preserve_energy=False
    )
    torch.testing.assert_close(linear, expected, rtol=1e-5, atol=1e-6)


def test_generic_warp_and_support(warp):
    # the generic warp of a kernel's polar image is the kernel (real part)
    recon = warp.reconstructor
    polar = (
        recon.angular_phase_component(1, 0.0)[:, None]
        * torch.as_tensor(recon.result.Vh[1, 2]).to(torch.complex64)[None, :]
    )
    expected = warp.spatial(_cells((1, 2)))[0].real
    torch.testing.assert_close(
        warp.to_cartesian(polar.real[None])[0], expected, rtol=1e-5, atol=1e-6
    )
    support = warp.support
    assert support.shape == (20, 20) and support.dtype == torch.bool
    assert support[10, 10] and not support[0, 0]
    assert warp.support is support  # built once


def test_adjoint_is_built_once_and_released(warp):
    adjoint = warp.adjoint
    assert adjoint is not None and warp.adjoint is adjoint
    bases = warp.adjoint_bases((0, 0, 2, 3))
    assert (bases.n_k, bases.n_m) == (2, 3)
    warp.release()
    assert warp.adjoint is not adjoint


def test_weight_source_matches_reference_layout_weights(synthetic_result):
    result = synthetic_result()
    rects = [(0, 0, 2, 3), (2, 0, 2, 2), (0, 3, 2, 2)]
    layout = FeatureLayout.create(rects)
    src = WeightSource(result, "cpu")
    got = src.layout_weights(layout)
    legacy = reference_layout_weights(result, layout)
    torch.testing.assert_close(got, legacy)
    assert src.n_templates == 6 and src.nbytes == 6 * 4 * 5 * 8

    rows = torch.tensor([5, 1])
    torch.testing.assert_close(src.layout_weights(layout, rows), legacy[rows])
    blk = src.block((2, 0, 2, 2), rows)
    assert blk.shape == (2, 2, 2)


def test_weight_source_bounding_box_and_sparse(synthetic_result):
    result = synthetic_result()
    boxed = WeightSource(result, "cpu", cells_union=[(1, 1, 2, 2), (2, 0, 1, 1)])
    assert (boxed.k_start, boxed.m_start, boxed.k_stop, boxed.m_stop) == (1, 0, 3, 3)
    full = WeightSource(result, "cpu")
    torch.testing.assert_close(boxed.block((1, 1, 2, 2)), full.block((1, 1, 2, 2)))
    with pytest.raises(ValueError):
        boxed.block((0, 0, 1, 1))
    with pytest.raises(ValueError):
        WeightSource(result, "cpu", cells_union=(0, 0, 10, 10))

    sparse = result.to_sparse(7)
    sparse_src = WeightSource(sparse, "cpu")
    dense = full.us.clone()
    top = result.get_top_n(7, include_negative=False)
    keep = torch.zeros(4, 5, dtype=torch.bool)
    for k, m in top:
        keep[int(k), int(m)] = True
    dense[:, ~keep] = 0
    torch.testing.assert_close(sparse_src.us, dense)


def test_template_bundle_and_hdf5_round_trip(synthetic_result, tmp_path):
    result = synthetic_result()
    config = KernelConfig(order=3)
    template = Template(result, device="cpu", kernel=config, weights_cells=(0, 0, 4, 5))
    assert template.template_shape == (20, 20) and template.kernel_config is config
    assert (template.k_max, template.eig_max, template.n_templates) == (4, 5, 6)
    assert template.space.n_templates == 6 and template.space.phi_deg is not None
    assert template._weights is None
    assert template.weights.us.shape == (6, 4, 5)
    torch.testing.assert_close(
        template.warp.spatial(_cells((0, 0))),
        build_block_kernels(template.warp.reconstructor, _cells((0, 0)), order=3),
    )

    path = tmp_path / "decomp.h5"
    result.save(path)
    loaded = Template.from_hdf5(path, device="cpu", kernel=config)
    assert loaded.kernel_config == config
    torch.testing.assert_close(loaded.weights.us, template.weights.us)
    template.release()
    assert template._weights is None


@requires_cuda
def test_separable_build_matches_generic_on_a_side_stream(synthetic_result):
    """The CuPy fast path must agree with the generic warp, also when torch runs on a
    non-default stream (its DLPack views and CuPy work are stream-ordered)."""
    result = synthetic_result()
    fast = TemplateWarp(result, device="cuda")
    slow = TemplateWarp(result, device="cuda")
    slow.__dict__["_separable"] = None  # force the generic path
    cells = _cells((0, 0), (1, 3), (3, 4), (2, 1))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        a = fast.spatial(cells)
        _ = torch.randn(64 << 20, device="cuda")  # churn the allocator on the stream
    torch.cuda.synchronize()
    b = slow.spatial(cells)
    torch.cuda.synchronize()
    rel = ((a - b).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1)).max().item()
    # both paths mirror the same seam padding; the residual is spline arithmetic
    assert fast._separable is not None and rel < 1e-4, rel
