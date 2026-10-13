"""HypothesisSpace / HypothesisSubset / HypothesisPlan index math."""

from __future__ import annotations

import pytest
import torch

from panther_em.inference.search.hypotheses import (
    HypothesisPlan,
    HypothesisSpace,
    HypothesisSubset,
    psi_degrees,
)


def _space(d=3, ff=2, o=7):
    return HypothesisSpace(
        n_defocus=d,
        n_filters=ff,
        n_orientations=o,
        phi_deg=torch.arange(o, dtype=torch.float32) * 10,
        theta_deg=torch.arange(o, dtype=torch.float32) * 5,
        defocus_values=torch.tensor([-1000.0, 0.0, 1000.0][:d]),
    )


def test_encode_decode_round_trip_with_passthrough():
    space = _space()
    d, f, o = torch.meshgrid(
        torch.arange(3), torch.arange(2), torch.arange(7), indexing="ij"
    )
    g = space.encode(d, f, o)
    assert g.max() == space.global_size - 1 and g.min() == 0
    assert torch.unique(g).numel() == space.global_size  # bijective
    back = space.decode(g)
    assert torch.equal(back.defocus, d) and torch.equal(back.filter, f)
    assert torch.equal(back.orientation, o)
    neg = space.decode(torch.tensor([-1, 5]))
    assert neg.defocus[0] == -1 and neg.filter[0] == -1 and neg.orientation[0] == -1
    assert neg.orientation[1] == 5


def test_angles_and_psi_degrees():
    space = _space()
    g = space.encode(torch.tensor([0, 2]), torch.tensor([1, 0]), torch.tensor([3, 6]))
    phi, theta, psi = space.angles(g, torch.tensor([0, 64]), n_psi=256)
    assert phi.tolist() == [30.0, 60.0] and theta.tolist() == [15.0, 30.0]
    assert psi.tolist() == [0.0, 90.0]
    assert torch.equal(psi_degrees(4), torch.tensor([0.0, 90.0, 180.0, 270.0]))
    phi_nan, _, psi_nan = space.angles(torch.tensor([-1]), torch.tensor([-1]), n_psi=8)
    assert torch.isnan(phi_nan).all() and torch.isnan(psi_nan).all()


def test_plan_full_space_is_contiguous_defocus_outer():
    space = _space(d=2, ff=1, o=10)
    plan = HypothesisPlan(space, None, hyp_batch=4, n_psi=16)
    assert plan.is_contiguous and plan.n_selected == 10 and plan.n_virtual == 20
    offsets = [(b.d_local, b.j0, b.j1, b.virtual_offset) for b in plan.batches]
    assert offsets == [
        (0, 0, 4, 0),
        (0, 4, 8, 4),
        (0, 8, 10, 8),
        (1, 0, 4, 10),
        (1, 4, 8, 14),
        (1, 8, 10, 18),
    ]
    assert len(plan.epochs) == 1 and all(
        b.kernel_offset == b.virtual_offset for b in plan.batches
    )
    # virtual ids decode to themselves here (FF = 1, all planes selected)
    v = torch.arange(20)
    assert torch.equal(plan.decode(v), v)
    assert plan.decode(torch.tensor([-1])).item() == -1
    assert plan.n_evaluated == 20 * 16


def test_plan_subset_decodes_to_global_ids():
    space = _space(d=3, ff=2, o=7)
    subset = HypothesisSubset(
        template_ids=torch.tensor([13, 1, 5, 5]),  # unsorted, duplicate
        defocus_ids=torch.tensor([2, 0]),
    )
    plan = HypothesisPlan(space, subset, hyp_batch=2, n_psi=8)
    assert plan.template_ids.tolist() == [1, 5, 13]
    assert plan.defocus_planes() == (0, 2)
    assert plan.n_virtual == 6 and not plan.is_contiguous
    # virtual (d_local, j) -> global: defocus_ids[d_local], template_ids[j]
    v = torch.arange(6)
    expected = torch.tensor(
        [0 * 14 + 1, 0 * 14 + 5, 0 * 14 + 13, 2 * 14 + 1, 2 * 14 + 5, 2 * 14 + 13]
    )
    assert torch.equal(plan.decode(v), expected)
    # a strided scout subset
    scout = HypothesisSubset.strided(space, 4)
    assert scout.template_ids.tolist() == [0, 4, 8, 12]
    ori = HypothesisSubset.orientations(space, torch.tensor([2, 3]))
    assert sorted(ori.template_ids.tolist()) == [2, 3, 9, 10]
    mask = torch.zeros(2, 7, dtype=torch.bool)
    mask[1, 6] = True
    assert HypothesisSubset.from_mask(space, mask).template_ids.tolist() == [13]


def test_plan_rejects_out_of_range_subset_and_bad_args():
    space = _space()
    with pytest.raises(ValueError):
        HypothesisPlan(
            space,
            HypothesisSubset(template_ids=torch.tensor([14])),
            hyp_batch=2,
            n_psi=8,
        )
    with pytest.raises(ValueError):
        HypothesisPlan(space, None, hyp_batch=0, n_psi=8)


def test_plan_epochs_respect_packed_index_width():
    space = _space(d=1, ff=1, o=40)
    # 8-bit budget: (offset + batch) * n_psi < 256 at n_psi = 4 -> 64 ids per epoch
    plan = HypothesisPlan(space, None, hyp_batch=10, n_psi=4, packed_index_bits=8)
    assert len(plan.epochs) == 1
    space2 = _space(d=1, ff=1, o=200)
    plan2 = HypothesisPlan(space2, None, hyp_batch=10, n_psi=4, packed_index_bits=8)
    assert len(plan2.epochs) == 4
    for ep in plan2.epochs:
        for b in plan2.batches[ep.batches]:
            assert b.kernel_offset == b.virtual_offset - ep.base
            assert (b.kernel_offset + b.size) * 4 < 256
    assert [b.epoch for b in plan2.batches] == sorted(b.epoch for b in plan2.batches)
    with pytest.raises(ValueError):
        HypothesisPlan(space2, None, hyp_batch=100, n_psi=4, packed_index_bits=8)


def test_plan_signature_distinguishes_subsets():
    space = _space()
    a = HypothesisPlan(space, None, hyp_batch=4, n_psi=8).signature
    b = HypothesisPlan(
        space, HypothesisSubset.strided(space, 2), hyp_batch=4, n_psi=8
    ).signature
    c = HypothesisPlan(
        space, HypothesisSubset.strided(space, 2), hyp_batch=4, n_psi=8
    ).signature
    assert a != b and b == c
