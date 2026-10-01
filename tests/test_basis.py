"""KernelBank / WeightSource / PreparedWeightCache / Template."""

from __future__ import annotations

import pytest
import torch

from panther_em.inference.projection_reconstruction import ProjectionReconstructor
from panther_em.inference.search.basis import (
    KernelBank,
    PreparedWeightCache,
    Template,
    WeightSource,
)
from panther_em.inference.search.contraction import ComplexGemmEngine
from panther_em.inference.search.hypotheses import HypBatch
from panther_em.inference.search.tiling import FeatureLayout, FeatureTiling
from panther_em.inference.search.types import KernelConfig
from panther_em.inference.search.utils import build_block_kernels, build_layout_weights

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array_wrap__ must accept.*:DeprecationWarning",
    "ignore:.*__array__ implementation doesn't accept.*:DeprecationWarning",
)


@pytest.fixture
def recon(synthetic_result):
    return ProjectionReconstructor(synthetic_result(), device="cpu")


def _cells(*pairs):
    return torch.tensor(pairs, dtype=torch.long)


def test_kernel_bank_matches_legacy_build_block_kernels(recon):
    cells = _cells((0, 0), (2, 3), (1, 1), (3, 4), (2, 0))
    bank = KernelBank(recon, KernelConfig())
    got = bank.spatial(cells)
    expected = build_block_kernels(recon, cells)
    assert got.shape == (5, 20, 20) and got.dtype == torch.complex64
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
    # order preserved and duplicates handled; nothing rebuilt on a repeat request
    built = bank.built
    again = bank.spatial(_cells((3, 4), (0, 0), (0, 0)))
    torch.testing.assert_close(again[0], expected[3])
    torch.testing.assert_close(again[1], expected[0])
    torch.testing.assert_close(again[2], expected[0])
    assert bank.built == built == 5


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
    torch.testing.assert_close(bank.spatial(cells), ref)
    dev, host = bank.bytes()
    assert dev <= 2 * one and host == 2 * one and bank.built == 4
    torch.testing.assert_close(bank.spatial(cells), ref)  # served from the two tiers
    assert bank.built == 4

    no_host = KernelBank(recon, device_budget=2 * one)
    no_host.spatial(cells)
    assert no_host.bytes() == (2 * one, 0)
    no_host.spatial(cells)
    assert no_host.built > 4  # evicted kernels had to be rebuilt


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


def test_weight_source_matches_legacy_layout_weights(recon):
    rects = [(0, 0, 2, 3), (2, 0, 2, 2), (0, 3, 2, 2)]
    layout = FeatureLayout(rects)
    src = WeightSource(recon.result, "cpu")
    got = src.layout_weights(layout)
    legacy = build_layout_weights(
        recon, FeatureTiling.from_extents(rects, device="cpu")
    )
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


def test_prepared_weight_cache_hits_evicts_and_pins(recon):
    layout = FeatureLayout([(0, 0, 3, 2), (0, 2, 3, 3), (3, 0, 1, 5)])
    src = WeightSource(recon.result, "cpu")
    engine = ComplexGemmEngine()
    run = layout.k_runs[0]  # k in [0, 3): both first rectangles
    batch = HypBatch(0, 1, 5, 1, 1, 0)

    cache = PreparedWeightCache(src)
    got = cache.get(layout, run, batch, engine)
    manual = engine.prepare_weights(
        torch.cat([src.block((0, 0, 3, 2))[1:5], src.block((0, 2, 3, 3))[1:5]], dim=-1)
    )
    torch.testing.assert_close(got, manual)
    assert got.shape == (3, 5, 4)
    assert cache.get(layout, run, batch, engine) is got and cache.stats == (1, 1)

    rows = torch.tensor([0, 2, 4, 5])
    sub = cache.get(layout, run, HypBatch(0, 0, 2, 0, 0, 0), engine, rows, rows_key="r")
    torch.testing.assert_close(
        sub,
        engine.prepare_weights(
            torch.cat(
                [src.block((0, 0, 3, 2))[[0, 2]], src.block((0, 2, 3, 3))[[0, 2]]],
                dim=-1,
            )
        ),
    )

    tiny = PreparedWeightCache(src, byte_budget=got.numel() * got.element_size())
    a = tiny.get(layout, run, batch, engine)
    tiny.pin("graph", [PreparedWeightCache.key(layout, run, batch, engine, None)])
    b = tiny.get(layout, layout.k_runs[1], batch, engine)  # over budget, newest: kept
    assert tiny.get(layout, layout.k_runs[1], batch, engine) is b
    tiny.get(layout, run, HypBatch(0, 0, 1, 0, 0, 0), engine)  # third entry evicts b
    assert tiny.get(layout, run, batch, engine) is a  # pinned survived
    assert tiny.get(layout, layout.k_runs[1], batch, engine) is not b
    tiny.unpin("graph")


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
