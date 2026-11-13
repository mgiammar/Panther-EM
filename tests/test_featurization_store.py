"""featurization.FeatureStore: chunks in, engine-layout blocks out."""

from __future__ import annotations

import pytest
import torch
from conftest import read_group, requires_cuda

from panther_em.search.featurization import FeatureChunk, FeatureStore

RECT = (0, 0, 2, 3)  # 6 cells


def features(n_planes, n_cells, n_cols, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.complex(
        torch.randn(n_planes, n_cells, n_cols, generator=gen),
        torch.randn(n_planes, n_cells, n_cols, generator=gen),
    )


def chunk(data, plane, cells, columns, offset=0):
    values = data[plane][cells, columns.start + offset : columns.stop + offset]
    return FeatureChunk(plane, cells, columns, values.contiguous())


def test_layout_and_lookup():
    store = FeatureStore([(1, 0, 3, 2), RECT], "fp32", (4, 1), 10, 4, "cpu")
    assert store.n_blocks == 3 and store.block_range(2) == (8, 10)
    g = store.group((1, 0, 3, 2))
    assert g.id == 0 and g.planes == (4, 1) and g.n_defocus == 2
    assert g.data.shape == (3, 2, 3, 4, 2) and g.data.dtype == torch.complex64
    assert len(g.blocks) == 3 and g.blocks[0].shape == (2, 3, 4, 2)
    # 3 blocks x (2 planes x 3 k x 4 px x 2 m) + 3 blocks x (2 x 2 x 4 x 3), complex64
    assert store.nbytes == (3 * 48 + 3 * 48) * 8
    # global plane ids map onto the blocks' leading axis
    assert store.groups_covering(RECT, "fp32", (1, 4)) == [
        (store.groups[1], 1),
        (store.groups[1], 0),
    ]
    for args in (
        (RECT, "fp16", (1,)),
        (RECT, "fp32", (0,)),
        ((0, 0, 1, 1), "fp32", (1,)),
    ):
        with pytest.raises(KeyError):
            store.groups_covering(*args)


def test_writes_split_across_blocks_and_completeness():
    store = FeatureStore([RECT], "fp32", (0, 1), 10, 4, "cpu")
    data = features(2, 6, 10)
    # awkward pieces: columns [2, 9) straddle all three blocks, cells in two chunks
    store.write(RECT, chunk(data, 0, slice(0, 4), slice(0, 7), offset=2), 2)
    store.write(RECT, chunk(data, 0, slice(4, 6), slice(0, 7), offset=2), 2)
    with pytest.raises(ValueError, match="fewer than"):
        store.check_complete()
    store.write(RECT, chunk(data, 0, slice(0, 6), slice(0, 2)))
    store.write(RECT, chunk(data, 0, slice(0, 6), slice(9, 10)))
    store.write(RECT, chunk(data, 1, slice(0, 6), slice(0, 10)))
    store.check_complete()
    for d in range(2):
        torch.testing.assert_close(
            read_group(store, store.groups[0], d), data[d].T.reshape(10, 2, 3)
        )
    assert not store.groups[0].data[2, :, :, 2:].any()  # the tail stays zero

    values = torch.zeros((6, 2), dtype=torch.complex64)
    for bad, offset in (
        (FeatureChunk(2, slice(0, 6), slice(0, 2), values), 0),  # no third plane
        (FeatureChunk(0, slice(1, 7), slice(0, 2), values), 0),  # beyond the cells
        (FeatureChunk(0, slice(0, 6), slice(0, 2), values), 9),  # beyond the columns
    ):
        with pytest.raises(ValueError):
            store.write(RECT, bad, offset)
    with pytest.raises(KeyError):
        store.write(
            (0, 0, 1, 1), FeatureChunk(0, slice(0, 1), slice(0, 0), values[:1, :0])
        )


def test_blocks_are_views_in_column_order():
    store = FeatureStore([RECT, (2, 0, 1, 1)], "fp32", (0,), 10, 4, "cpu")
    blocks = list(store.blocks(store.groups[:1]))
    assert [b.index for b in blocks] == [0, 1, 2]
    assert [b.col_range for b in blocks] == [(0, 4), (4, 8), (8, 10)]
    assert [b.n_valid for b in blocks] == [4, 4, 2]
    assert set(blocks[1].operands) == {0}
    assert blocks[1].operands[0].data_ptr() == store.groups[0].data[1].data_ptr()
    other = FeatureStore([RECT], "fp32", (0,), 10, 4, "cpu")
    with pytest.raises(ValueError):
        list(store.blocks(other.groups))


def test_validation():
    for args in (
        ([], "fp32", (0,), 4, 2),
        ([RECT, RECT], "fp32", (0,), 4, 2),
        ([RECT], "fp32", (), 4, 2),
        ([RECT], "fp32", (0, 0), 4, 2),
        ([RECT], "fp32", (-1,), 4, 2),
        ([RECT], "fp32", (0,), 4, 0),
    ):
        with pytest.raises(ValueError):
            FeatureStore(*args, device="cpu")


@requires_cuda
def test_fp16_layout_round_trip():
    store = FeatureStore([RECT], "fp16", (0, 1), 10, 4, "cuda")
    g = store.groups[0]
    assert g.blocks[0].dtype == torch.float16 and g.blocks[0].shape == (2, 2, 4, 6)
    data = features(2, 6, 10)
    for d in range(2):
        store.write(RECT, chunk(data, d, slice(0, 6), slice(0, 10)))
    store.check_complete()
    torch.testing.assert_close(
        read_group(store, g, 1).cpu(), data[1].T.reshape(10, 2, 3), rtol=2e-3, atol=2e-3
    )
