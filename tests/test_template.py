"""WeightSource / Template (panther_em.inference.search)."""

from __future__ import annotations

import pytest
import torch

from panther_em.coordinates import WarpConfig
from panther_em.inference.search.correlation import FeatureLayout
from panther_em.inference.search.template import Template
from panther_em.inference.search.weights import WeightSource

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array_wrap__ must accept.*:DeprecationWarning",
    "ignore:.*__array__ implementation doesn't accept.*:DeprecationWarning",
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
    config = WarpConfig(order=3)
    template = Template(result, device="cpu", warp=config, weights_cells=(0, 0, 4, 5))
    assert template.template_shape == (20, 20) and template.warp is config
    assert (template.k_max, template.eig_max, template.n_templates) == (4, 5, 6)
    assert template.space.n_templates == 6 and template.space.phi_deg is not None
    assert template._weights is None
    assert template.weights.us.shape == (6, 4, 5)
    assert template.reconstructor.warp is config

    path = tmp_path / "decomp.h5"
    result.save(path)
    loaded = Template.from_hdf5(path, device="cpu", warp=config)
    assert loaded.warp == config
    assert Template.from_hdf5(path, device="cpu").warp == result.warp_config
    torch.testing.assert_close(loaded.weights.us, template.weights.us)
    template.release()
    assert template._weights is None

    result.to_sparse(7).save(tmp_path / "sparse.h5")
    sparse = Template.from_hdf5(tmp_path / "sparse.h5", device="cpu")
    assert sparse.n_templates == template.n_templates


def test_template_rejects_complex_projections(synthetic_result):
    result = synthetic_result()
    result.is_complex_projection = True
    with pytest.raises(NotImplementedError, match="real-valued"):
        Template(result, device="cpu")
