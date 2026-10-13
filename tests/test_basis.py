"""KernelBank / WeightSource / Template."""

from __future__ import annotations

import pytest
import torch
from conftest import count_kernel_builds

from panther_em.inference.projection_reconstruction import ProjectionReconstructor
from panther_em.inference.search.basis import (
    KernelBank,
    Template,
    WeightSource,
)
from panther_em.inference.search.tiling import FeatureLayout
from panther_em.inference.search.types import KernelConfig

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
    us = torch.as_tensor(result.U) * torch.as_tensor(result.S)[None, None]
    ff, o = us.shape[:2]
    blocks = [
        us[:, :, k0 : k0 + ke, m0 : m0 + me].reshape(ff * o, ke * me)
        for k0, m0, ke, me in layout.rectangles
    ]
    return torch.cat(blocks, dim=1).to(torch.complex64)


@pytest.fixture
def recon(synthetic_result):
    return ProjectionReconstructor(synthetic_result(), device="cpu")


def _cells(*pairs):
    return torch.tensor(pairs, dtype=torch.long)


def test_kernel_bank_matches_per_cell_construction(recon):
    cells = _cells((0, 0), (2, 3), (1, 1), (3, 4), (2, 0))
    bank = KernelBank(recon, KernelConfig())
    built = count_kernel_builds(bank)
    got = bank.spatial(cells)
    expected = build_block_kernels(recon, cells)
    assert got.shape == (5, 20, 20) and got.dtype == torch.complex64
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
    # order preserved and duplicates handled; nothing rebuilt on a repeat request
    assert built[0] == 5
    again = bank.spatial(_cells((3, 4), (0, 0), (0, 0)))
    torch.testing.assert_close(again[0], expected[3])
    torch.testing.assert_close(again[1], expected[0])
    torch.testing.assert_close(again[2], expected[0])
    assert built[0] == 5


def test_kernel_bank_respects_kernel_config(recon):
    cells = _cells((1, 2))
    default = KernelBank(recon).spatial(cells)
    linear = KernelBank(recon, KernelConfig(order=1, preserve_energy=False)).spatial(
        cells
    )
    assert not torch.allclose(default, linear)
    expected = build_block_kernels(recon, cells, order=1, preserve_energy=False)
    torch.testing.assert_close(linear, expected, rtol=1e-5, atol=1e-6)


def test_kernel_bank_budget_eviction_and_host_tier(recon):
    one = 20 * 20 * 8  # bytes per complex64 kernel
    cells = _cells((0, 0), (0, 1), (0, 2), (0, 3))
    ref = build_block_kernels(recon, cells)

    bank = KernelBank(recon, device_budget=2 * one, host_budget=10 * one)
    built = count_kernel_builds(bank)
    torch.testing.assert_close(bank.spatial(cells), ref)
    dev, host = bank.bytes()
    assert dev <= 2 * one and host == 2 * one and built[0] == 4
    torch.testing.assert_close(bank.spatial(cells), ref)  # served from the two tiers
    assert built[0] == 4
    # the host tier is inclusive: promoted kernels keep their host copy, so after the
    # round trip every kernel is on the host and device evictions copy nothing
    assert bank.bytes()[1] == 4 * one
    torch.testing.assert_close(bank.spatial(cells), ref)
    assert built[0] == 4 and bank.bytes() == (2 * one, 4 * one)

    no_host = KernelBank(recon, device_budget=2 * one)
    rebuilt = count_kernel_builds(no_host)
    no_host.spatial(cells)
    assert no_host.bytes() == (2 * one, 0)
    no_host.spatial(cells)
    assert rebuilt[0] > 4  # evicted kernels had to be rebuilt


def test_kernel_bank_spectral_and_planar(recon):
    bank = KernelBank(recon)
    cells = _cells((1, 0), (2, 2))
    spatial = bank.spatial(cells)
    spec = bank.spectral(cells, (32, 30))
    torch.testing.assert_close(spec, torch.fft.fft2(spatial, s=(32, 30)).conj())
    out = torch.empty_like(spec)
    assert bank.spectral(cells, (32, 30), out=out) is out
    torch.testing.assert_close(out, spec)
    with pytest.raises(ValueError):
        bank.spectral(
            cells, (32, 30), out=torch.empty(1, 32, 30, dtype=torch.complex64)
        )

    planar = bank.planar(cells)  # (2, n, kH*kW)
    assert planar.shape == (2, 2, 400)
    image = torch.randn(20, 20)
    feats = planar @ image.reshape(-1)  # (2, n)
    expected = (image[None] * spatial.conj()).sum(dim=(-2, -1))
    torch.testing.assert_close(
        torch.complex(feats[0], feats[1]), expected, rtol=1e-4, atol=1e-4
    )


def test_weight_source_matches_reference_layout_weights(recon):
    rects = [(0, 0, 2, 3), (2, 0, 2, 2), (0, 3, 2, 2)]
    layout = FeatureLayout(rects)
    src = WeightSource(recon.result, "cpu")
    got = src.layout_weights(layout)
    legacy = reference_layout_weights(recon.result, layout)
    torch.testing.assert_close(got, legacy)
    assert src.n_templates == 6 and src.nbytes == 6 * 4 * 5 * 8

    rows = torch.tensor([5, 1])
    torch.testing.assert_close(src.layout_weights(layout, rows), legacy[rows])
    blk = src.block((2, 0, 2, 2), rows)
    assert blk.shape == (2, 2, 2)


def test_weight_source_bounding_box_and_sparse(recon):
    boxed = WeightSource(recon.result, "cpu", cells_union=[(1, 1, 2, 2), (2, 0, 1, 1)])
    assert (boxed.k_start, boxed.m_start, boxed.k_stop, boxed.m_stop) == (1, 0, 3, 3)
    full = WeightSource(recon.result, "cpu")
    torch.testing.assert_close(boxed.block((1, 1, 2, 2)), full.block((1, 1, 2, 2)))
    with pytest.raises(ValueError):
        boxed.block((0, 0, 1, 1))
    with pytest.raises(ValueError):
        WeightSource(recon.result, "cpu", cells_union=(0, 0, 10, 10))

    sparse = recon.result.to_sparse(7)
    sparse_src = WeightSource(sparse, "cpu")
    dense = full.us.clone()
    top = recon.result.get_top_n(7, include_negative=False)
    keep = torch.zeros(4, 5, dtype=torch.bool)
    for k, m in top:
        keep[int(k), int(m)] = True
    dense[:, ~keep] = 0
    torch.testing.assert_close(sparse_src.us, dense)


def test_template_bundle_and_hdf5_round_trip(synthetic_result, tmp_path):
    result = synthetic_result()
    template = Template(result, device="cpu", weights_cells=(0, 0, 4, 5))
    assert template.template_shape == (20, 20)
    assert template.space.n_templates == 6 and template.space.phi_deg is not None
    assert template._weights is None
    assert template.weights.us.shape == (6, 4, 5)
    torch.testing.assert_close(
        template.kernels.spatial(_cells((0, 0))),
        build_block_kernels(template.reconstructor, _cells((0, 0))),
    )

    path = tmp_path / "decomp.h5"
    result.save(path)
    loaded = Template.from_hdf5(path, device="cpu")
    torch.testing.assert_close(loaded.weights.us, template.weights.us)
    template.release()
    assert template._weights is None and template.kernels.bytes() == (0, 0)


@pytest.fixture
def recon_cuda(synthetic_result):
    from panther_em.inference.projection_reconstruction import ProjectionReconstructor

    return ProjectionReconstructor(synthetic_result(), device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_kernel_bank_separable_build_matches_generic_on_a_side_stream(recon_cuda):
    """The CuPy fast path must agree with the generic warp, also when torch runs on a
    non-default stream (its DLPack views and CuPy work are stream-ordered)."""
    from panther_em.inference.search.basis import KernelBank

    fast = KernelBank(recon_cuda, device="cuda")
    slow = KernelBank(recon_cuda, device="cuda")
    slow._sep = None  # force the generic path
    cells = _cells((0, 0), (1, 3), (3, 4), (2, 1))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        a = fast.spatial(cells)
        _ = torch.randn(
            64 << 20, device="cuda"
        )  # churn the allocator on the same stream
    torch.cuda.synchronize()
    b = slow.spatial(cells)
    torch.cuda.synchronize()
    rel = ((a - b).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1)).max().item()
    # both paths mirror the same seam padding; the residual is spline arithmetic
    assert fast._sep is not None and rel < 1e-4, rel
