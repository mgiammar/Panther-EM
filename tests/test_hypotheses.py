"""HypothesisSpace / HypothesisSet index math and the HEALPix level sampling."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from panther_em.inference.search.hypotheses import (
    HypothesisSet,
    HypothesisSpace,
    healpix_level_ids,
    psi_degrees,
    template_rows_for,
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


def test_decode_is_the_inverse_of_the_global_id_with_passthrough():
    space = _space()
    d, f, o = torch.meshgrid(
        torch.arange(3), torch.arange(2), torch.arange(7), indexing="ij"
    )
    g = (d * 2 + f) * 7 + o
    assert torch.unique(g).numel() == 3 * 2 * 7  # bijective
    back_d, back_f, back_o = space.decode(g)
    assert torch.equal(back_d, d) and torch.equal(back_f, f) and torch.equal(back_o, o)
    neg_d, neg_f, neg_o = space.decode(torch.tensor([-1, 5]))
    assert neg_d[0] == -1 and neg_f[0] == -1 and neg_o[0] == -1 and neg_o[1] == 5
    assert torch.equal(psi_degrees(4), torch.tensor([0.0, 90.0, 180.0, 270.0]))


def test_set_full_space_is_contiguous_defocus_outer():
    space = _space(d=2, ff=1, o=10)
    hyps = HypothesisSet.create(space, None, None, hyp_batch=4, n_psi=16)
    assert hyps.is_contiguous and hyps.n_selected == 10 and hyps.n_virtual == 20
    offsets = [(b.d_local, b.j0, b.j1, b.virtual_offset) for b in hyps.batches]
    assert offsets == [
        (0, 0, 4, 0),
        (0, 4, 8, 4),
        (0, 8, 10, 8),
        (1, 0, 4, 10),
        (1, 4, 8, 14),
        (1, 8, 10, 18),
    ]
    # virtual ids decode to themselves here (FF = 1, all planes selected)
    v = torch.arange(20)
    assert torch.equal(hyps.decode(v), v)
    assert hyps.decode(torch.tensor([-1])).item() == -1
    assert hyps.n_evaluated == 20 * 16


def test_set_subset_decodes_to_global_ids():
    space = _space(d=3, ff=2, o=7)
    hyps = HypothesisSet.create(
        space,
        torch.tensor([13, 1, 5, 5]),  # unsorted, duplicate
        torch.tensor([2, 0]),
        hyp_batch=2,
        n_psi=8,
    )
    assert hyps.template_ids.tolist() == [1, 5, 13]
    assert hyps.defocus_planes() == (0, 2)
    assert hyps.n_virtual == 6 and not hyps.is_contiguous
    # virtual (d_local, j) -> global: defocus_ids[d_local], template_ids[j]
    v = torch.arange(6)
    expected = torch.tensor(
        [0 * 14 + 1, 0 * 14 + 5, 0 * 14 + 13, 2 * 14 + 1, 2 * 14 + 5, 2 * 14 + 13]
    )
    assert torch.equal(hyps.decode(v), expected)
    assert torch.equal(hyps.template_rows(), torch.tensor([1, 5, 13]))
    rows = template_rows_for(space, torch.tensor([2, 3]))
    assert sorted(rows.tolist()) == [2, 3, 9, 10]  # every filter x the orientations
    only_f1 = template_rows_for(space, torch.tensor([2]), filter_ids=torch.tensor([1]))
    assert only_f1.tolist() == [9]


def test_set_rejects_out_of_range_ids_bad_args_and_packed_index_overflow():
    space = _space()
    with pytest.raises(ValueError):
        HypothesisSet.create(space, torch.tensor([14]), None, hyp_batch=2, n_psi=8)
    with pytest.raises(ValueError):
        HypothesisSet.create(space, None, torch.tensor([3]), hyp_batch=2, n_psi=8)
    with pytest.raises(ValueError):
        HypothesisSet.create(space, None, None, hyp_batch=0, n_psi=8)
    # 42 hypotheses x 8 psi = 336 (virtual id, psi) pairs do not fit 8 bits
    with pytest.raises(ValueError, match="packed index"):
        HypothesisSet.create(space, None, None, hyp_batch=4, n_psi=8, packed_index_bits=8)
    # restricting the planes brings a stage back under the budget
    hyps = HypothesisSet.create(
        space, None, torch.tensor([1]), hyp_batch=4, n_psi=8, packed_index_bits=8
    )
    assert hyps.n_virtual * 8 < 256


def test_set_signature_distinguishes_selections():
    space = _space()
    a = HypothesisSet.create(space, None, None, hyp_batch=4, n_psi=8).signature
    b = HypothesisSet.create(space, torch.arange(0, 14, 2), None, hyp_batch=4, n_psi=8).signature
    c = HypothesisSet.create(space, torch.arange(0, 14, 2), None, hyp_batch=4, n_psi=8).signature
    assert a != b and b == c


def test_healpix_level_ids_keep_one_child_per_coarser_pixel():
    ipix = np.arange(48)  # nside = 2, NESTED: parents at nside = 1 own 4 children
    assert healpix_level_ids(ipix, 1).tolist() == list(range(0, 48, 4))
    assert healpix_level_ids(ipix, 2).tolist() == list(range(0, 48, 16))
    assert healpix_level_ids(ipix, 0).tolist() == list(range(48))
    # orientation ids index the template's own orientation order, not the sphere
    sub = np.array([5, 4, 8, 0])
    assert healpix_level_ids(sub, 1).tolist() == [1, 2, 3]
    with pytest.raises(ValueError):
        healpix_level_ids(ipix, -1)
