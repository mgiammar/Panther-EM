"""FeatureStore / BlockWriter / direct block source / compaction."""

from __future__ import annotations

import pytest
import torch
from conftest import read_group

from panther_em.inference.search.store import FeatureStore, ReadyBlock
from panther_em.inference.search.types import ImageSet, PixelSelection, SearchGeometry

CUDA = torch.cuda.is_available()


def _store(n_px=10, block_pixels=4, n_defocus=2, device="cpu", mask_every=None):
    geo = SearchGeometry.from_images(
        ImageSet.from_stack(torch.zeros(1, 21, 21)), (20, 20)
    )
    assert geo.n_pixels == 4  # 2x2 grid -> make a bigger one below
    geo = SearchGeometry.from_images(
        ImageSet.from_stack(torch.zeros(1, 24, 24)), (20, 20)
    )
    assert geo.n_pixels == 25
    index = torch.arange(geo.n_pixels)
    if mask_every:
        index = index[::mask_every]
    index = index[:n_px]
    pixels = PixelSelection(index, geo)
    column_index = index.flip(0)  # arbitrary permutation: compact order != sorted order
    return FeatureStore(
        geo,
        pixels,
        column_index,
        block_pixels=block_pixels,
        n_defocus=n_defocus,
        device=device,
    )


def _random_group_data(group, n_cols, seed=0):
    gen = torch.Generator().manual_seed(seed)
    re = torch.randn(group.n_defocus, group.num_cells, n_cols, generator=gen)
    im = torch.randn(group.n_defocus, group.num_cells, n_cols, generator=gen)
    return torch.complex(re, im)


def test_add_group_blocks_and_tail():
    store = _store(n_px=10, block_pixels=4)
    assert store.n_blocks == 3 and store.block_range(2) == (8, 10)
    g = store.add_group((1, 0, 3, 2), "fp32")
    assert len(g.blocks) == 3 and g.blocks[0].shape == (2, 3, 4, 2)
    assert g.blocks[0].dtype == torch.complex64 and g.id == 0
    with pytest.raises(ValueError):
        store.add_group((1, 0, 3, 2), "fp32")
    assert store.group_for((1, 0, 3, 2)) is g and store.group_for((0, 0, 1, 1)) is None


def test_writer_splits_across_blocks_and_requires_completeness():
    store = _store(n_px=10, block_pixels=4)
    g = store.add_group((0, 0, 2, 3), "fp32")
    data = _random_group_data(g, 10)
    w = store.writer(g)
    with pytest.raises(ValueError):
        store.blocks([g])  # incomplete
    # write in awkward pieces: columns [2, 9) straddle blocks 0/1/2; cells in two chunks
    w.write(slice(0, 4), (2, 9), data[:, 0:4, 2:9])
    w.write(slice(4, 6), (2, 9), data[:, 4:6, 2:9])
    with pytest.raises(ValueError):
        w.close()
    w.write(slice(0, 6), (0, 2), data[:, :, 0:2])
    w.write(slice(0, 6), (9, 10), data[0, :, 9:10], plane=0)
    w.write(slice(0, 6), (9, 10), data[1, :, 9:10], plane=1)
    w.close()
    assert g.complete
    for d in range(2):
        got = read_group(store, g, d)  # (P, num_k, num_m)
        expected = data[d].T.reshape(10, 2, 3)
        torch.testing.assert_close(got, expected)
    with pytest.raises(ValueError):
        w.write(slice(0, 6), (0, 2), data[:, :, 0:3])


def test_direct_block_source_yields_views_in_order():
    store = _store(n_px=10, block_pixels=4)
    g = store.add_group((0, 0, 2, 3), "fp32")
    data = _random_group_data(g, 10)
    w = store.writer(g)
    w.write(slice(0, 6), (0, 10), data)
    w.close()
    src = store.blocks([g])
    blocks = list(src)
    assert len(src) == 3 and [b.index for b in blocks] == [0, 1, 2]
    assert [b.n_valid for b in blocks] == [4, 4, 2]
    assert all(isinstance(b, ReadyBlock) and b.ready is None for b in blocks)
    assert blocks[1].operands[g.id] is g.blocks[1]
    blocks[0].release(None)  # no-op for device residency


def test_subset_compacts_columns_block_wise():
    store = _store(n_px=10, block_pixels=4)
    g1 = store.add_group((0, 0, 2, 3), "fp32")
    g2 = store.add_group((2, 1, 1, 2), "fp32")
    d1, d2 = _random_group_data(g1, 10, seed=1), _random_group_data(g2, 10, seed=2)
    for g, d in ((g1, d1), (g2, d2)):
        w = store.writer(g)
        w.write(slice(0, g.num_cells), (0, 10), d)
        w.close()
    keep = torch.tensor([1, 0, 0, 1, 1, 0, 1, 0, 0, 1], dtype=torch.bool)
    sub = store.subset(keep, block_pixels=3)
    assert sub.n_pixels == 5 and sub.n_blocks == 2 and sub.block_pixels == 3
    assert torch.equal(sub.column_index, store.column_index[keep])
    assert torch.equal(sub.pixels.index, torch.sort(sub.column_index).values)
    assert len(sub.groups) == 2 and all(g.complete for g in sub.groups)
    for g_old, g_new in ((g1, sub.groups[0]), (g2, sub.groups[1])):
        assert g_new.rect == g_old.rect
        for plane in range(2):
            torch.testing.assert_close(
                read_group(sub, g_new, plane), read_group(store, g_old, plane)[keep]
            )
    with pytest.raises(ValueError):
        store.subset(keep[:3])


@pytest.mark.skipif(not CUDA, reason="fp16 blocks need CUDA")
def test_fp16_group_round_trip_cuda():
    store = _store(n_px=10, block_pixels=4, device="cuda")
    g = store.add_group((0, 0, 2, 3), "fp16")
    assert g.blocks[0].dtype == torch.float16 and g.blocks[0].shape == (2, 2, 4, 6)
    data = _random_group_data(g, 10)
    w = store.writer(g)
    w.write(slice(0, 6), (0, 10), data)
    w.close()
    torch.testing.assert_close(
        read_group(store, g, 1).cpu(), data[1].T.reshape(10, 2, 3), rtol=2e-3, atol=2e-3
    )
    assert store.nbytes() == 3 * 2 * 2 * 4 * 6 * 2


def test_store_validation():
    geo = SearchGeometry.from_images(
        ImageSet.from_stack(torch.zeros(1, 24, 24)), (20, 20)
    )
    pixels = PixelSelection.all(geo)
    with pytest.raises(ValueError):
        FeatureStore(
            geo, pixels, torch.arange(3), block_pixels=4, n_defocus=1, device="cpu"
        )
    with pytest.raises(NotImplementedError):
        FeatureStore(
            geo,
            pixels,
            pixels.index,
            block_pixels=4,
            n_defocus=1,
            device="cpu",
            residency="transient",
        )


def test_plane_restricted_groups_cover_and_reject_overlap():
    store = _store(n_px=10, block_pixels=4, n_defocus=3)
    rect = (0, 0, 2, 3)
    g_a = store.add_group(rect, "fp32", planes=(2,))
    assert g_a.planes == (2,) and g_a.blocks[0].shape == (1, 2, 4, 3)
    assert store.missing_planes(rect, "fp32", (0, 1, 2)) == (0, 1)
    with pytest.raises(ValueError, match="already holds planes"):
        store.add_group(rect, "fp32", planes=(1, 2))
    with pytest.raises(ValueError):
        store.add_group(rect, "fp32", planes=(3,))
    g_b = store.add_group(rect, "fp32", planes=(0, 1))
    assert store.missing_planes(rect, "fp32", (0, 1, 2)) == ()
    covering = store.groups_covering(rect, "fp32", (0, 2, 1))
    assert covering == [(g_b, 0), (g_a, 0), (g_b, 1)]
    with pytest.raises(KeyError):
        store.groups_covering(rect, "fp16", (0,))
    assert store.group_for(rect, "fp32", planes=(2,)) is g_a
    assert store.group_for(rect, "fp32", planes=(0, 1)) is g_b
    assert store.group_for(rect, "fp32") is g_a  # first match when planes unspecified
    assert g_a.local_plane(2) == 0 and g_b.local_plane(1) == 1 and not g_a.holds(0)
    with pytest.raises(ValueError):
        g_a.local_plane(0)

    # writes address planes globally and completeness is tracked per plane
    data = _random_group_data(g_b, 10)  # (2, cells, 10) in group plane order
    w = store.writer(g_b)
    w.write(slice(0, 6), (0, 10), data[1], plane=1)
    with pytest.raises(ValueError, match="incomplete"):
        w.close()
    w.write(slice(0, 6), (0, 10), data[0], plane=0)
    w.close()
    torch.testing.assert_close(read_group(store, g_b, 1), data[1].T.reshape(10, 2, 3))
    with pytest.raises(ValueError):
        store.writer(g_a).write(
            slice(0, 6), (0, 10), data[0], plane=0
        )  # g_a lacks plane 0

    # compaction preserves plane sets and ids; gather_columns matches read_group
    wa = store.writer(g_a)
    wa.write(slice(0, 6), (0, 10), data[1], plane=2)
    wa.close()
    keep = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0, 1, 1], dtype=torch.bool)
    sub = store.subset(keep, block_pixels=3)
    assert [(g.id, g.planes) for g in sub.groups] == [(0, (2,)), (1, (0, 1))]
    positions = torch.tensor([9, 2, 6, 0])
    gathered = store.gather_columns(g_b, positions)
    assert gathered.shape == (2, 2, 4, 3)
    expected = torch.stack(
        [read_group(store, g_b, d)[positions].permute(1, 0, 2) for d in range(2)]
    )  # (D, num_k, n, num_m)
    torch.testing.assert_close(gathered, expected)
    assert store.gather_columns(g_b, torch.empty(0, dtype=torch.long)).shape == (
        2,
        2,
        0,
        3,
    )
