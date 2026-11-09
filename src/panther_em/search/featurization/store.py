"""The feature store: one pixel set's features, in the search loop's block layout.

A :class:`FeatureStore` holds the features of one pixel set (a band of a stage's
pixels, for one pass of its defocus planes) on the compute device: one
:class:`ColumnGroup` per rectangle, every group for the same planes and precision.
Featurizations fill it chunk by chunk (:meth:`FeatureStore.write`); the hypothesis
loop reads it block by block (:meth:`FeatureStore.blocks`).

Column ``p`` of the store is position ``p`` of the pixel set, in the order the caller
featurized it. A group's features live in one ``(n_blocks, *block)`` tensor whose
blocks are the contraction engine's operand layout (``(D, num_k, block_pixels,
num_m)`` complex64, or ``(D, num_k, block_pixels, 2 num_m)`` float16 ``[Re | Im]``);
the tail of the last block stays zero.

What the correlation reads of a store is the :class:`StoreLike` protocol; a
:class:`ColumnSubset` serves some of a store's columns as blocks of its own size
without copying the whole store (a seed pass over a lattice of a band's pixels).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from panther_em.decomposition.rectangles import as_rectangle
from panther_em.search.correlation.contraction import Precision, make_engine

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from panther_em.decomposition.rectangles import Rectangle
    from panther_em.search.correlation.contraction import ContractionEngine
    from panther_em.search.featurization.base import FeatureChunk


@dataclass(eq=False)
class ColumnGroup:
    """One rectangle's features for every column of the store.

    Attributes
    ----------
    id : int
        Position in :attr:`FeatureStore.groups` (the loop addresses blocks by it).
    rect : Rectangle
        ``(k_start, m_start, k_extent, m_extent)``.
    planes : tuple[int, ...]
        Global defocus plane ids, in the order of the blocks' leading axis.
    precision : Precision
        The contraction precision, which fixes the block layout through ``engine``.
    engine : ContractionEngine
    data : torch.Tensor
        ``(n_blocks, *engine.block_shape(D, num_k, num_m, block_pixels))``.
    """

    id: int
    rect: Rectangle
    planes: tuple[int, ...]
    precision: Precision
    engine: ContractionEngine
    data: torch.Tensor

    @property
    def n_defocus(self) -> int:
        """Planes held, ``D``."""
        return len(self.planes)

    @property
    def num_cells(self) -> int:
        """``num_k * num_m``."""
        return self.rect[2] * self.rect[3]

    @property
    def blocks(self) -> tuple[torch.Tensor, ...]:
        """The per-block views of :attr:`data` (each contiguous)."""
        return tuple(self.data.unbind(0))

    @property
    def nbytes(self) -> int:
        """Bytes of :attr:`data`."""
        return self.data.numel() * self.data.element_size()


@dataclass(frozen=True, eq=False)
class ReadyBlock:
    """One block of columns of every requested group.

    Attributes
    ----------
    index : int
        Block index.
    col_range : tuple[int, int]
        Columns ``[start, stop)`` the block covers.
    n_valid : int
        ``stop - start`` (less than ``block_pixels`` for the tail block).
    operands : Mapping[int, torch.Tensor]
        ``group.id -> block`` in the engine layout.
    """

    index: int
    col_range: tuple[int, int]
    n_valid: int
    operands: Mapping[int, torch.Tensor]


class StoreLike(Protocol):
    """What the correlation reads of a store: columns served in engine-layout blocks."""

    n_pixels: int
    block_pixels: int
    device: torch.device
    precision: Precision
    planes: tuple[int, ...]

    def groups_covering(
        self, rect: Sequence[int], precision: Precision | str, planes: Sequence[int]
    ) -> list[tuple[ColumnGroup, int]]:
        """``(group, local plane index)`` for each of ``planes``, in request order."""
        ...

    def blocks(self, groups: Sequence[ColumnGroup]) -> Iterator[ReadyBlock]:
        """The blocks of ``groups`` in column order."""
        ...


class FeatureStore:
    """Device-resident features of one pixel set, one group per rectangle.

    Parameters
    ----------
    rectangles : sequence of Rectangle
        One group each.
    precision : Precision or str
        Contraction precision of every group.
    planes : sequence of int
        Global defocus plane ids every group holds; a chunk's ``plane`` indexes them.
    n_pixels : int
        Columns ``P``.
    block_pixels : int
        Columns per block (a multiple of the loop's GEMM pixel batch).
    device : torch.device or str
    """

    def __init__(
        self,
        rectangles: Sequence[Sequence[int]],
        precision: Precision | str,
        planes: Sequence[int],
        n_pixels: int,
        block_pixels: int,
        device: torch.device | str,
    ) -> None:
        rects = tuple(as_rectangle(r) for r in rectangles)
        if not rects or len(set(rects)) != len(rects):
            raise ValueError(
                "rectangles must be a non-empty set of distinct rectangles"
            )
        ids = tuple(int(p) for p in planes)
        if not ids or len(set(ids)) != len(ids) or min(ids) < 0:
            raise ValueError(f"planes must be distinct non-negative ids, got {ids}")
        if n_pixels < 0 or block_pixels < 1:
            raise ValueError("n_pixels must be >= 0 and block_pixels >= 1")
        self.planes = ids
        self.precision = Precision.coerce(precision)
        self.n_pixels = int(n_pixels)
        self.block_pixels = int(block_pixels)
        self.device = torch.device(device)
        n_blocks = -(-self.n_pixels // self.block_pixels)
        self.groups: tuple[ColumnGroup, ...] = tuple(
            self._group(i, rect, n_blocks) for i, rect in enumerate(rects)
        )
        # cell-columns written per (group, plane), checked by check_complete
        self._written = torch.zeros((len(rects), len(ids)), dtype=torch.int64)

    def _group(self, index: int, rect: Rectangle, n_blocks: int) -> ColumnGroup:
        engine = make_engine(self.precision)
        shape = engine.block_shape(
            len(self.planes), rect[2], rect[3], self.block_pixels
        )
        data = torch.zeros(
            (n_blocks, *shape), dtype=engine.block_dtype, device=self.device
        )
        return ColumnGroup(index, rect, self.planes, engine.precision, engine, data)

    # -- the pixel axis -----------------------------------------------------------

    @property
    def n_blocks(self) -> int:
        """Blocks along the pixel axis (the last may be partial)."""
        return -(-self.n_pixels // self.block_pixels)

    def block_range(self, index: int) -> tuple[int, int]:
        """Columns ``[start, stop)`` of block ``index``."""
        start = index * self.block_pixels
        return start, min(start + self.block_pixels, self.n_pixels)

    @property
    def nbytes(self) -> int:
        """Bytes across all groups."""
        return sum(g.nbytes for g in self.groups)

    # -- groups ---------------------------------------------------------------------

    def group(self, rect: Sequence[int]) -> ColumnGroup:
        """The group of ``rect``."""
        key = as_rectangle(rect)
        for g in self.groups:
            if g.rect == key:
                return g
        raise KeyError(f"no group of rectangle {key}")

    def groups_covering(
        self, rect: Sequence[int], precision: Precision | str, planes: Sequence[int]
    ) -> list[tuple[ColumnGroup, int]]:
        """``(group, local plane index)`` for each of ``planes``, in request order."""
        if Precision.coerce(precision) is not self.precision:
            raise KeyError(f"the store holds {self.precision.value} features")
        group = self.group(rect)
        missing = [int(p) for p in planes if int(p) not in self.planes]
        if missing:
            raise KeyError(f"the store holds planes {self.planes}, not {missing}")
        return [(group, self.planes.index(int(p))) for p in planes]

    # -- writing --------------------------------------------------------------------

    def write(
        self, rect: Sequence[int], chunk: FeatureChunk, column_offset: int = 0
    ) -> None:
        """Write a featurization's chunk of ``rect`` at columns ``offset + columns``.

        ``chunk.plane`` indexes :attr:`planes`; ``chunk.cells`` the rectangle's
        row-major cells.
        """
        group = self.group(rect)
        if chunk.plane >= len(self.planes):
            raise ValueError(f"chunk plane {chunk.plane} beyond the {len(self.planes)}")
        if chunk.cells.stop > group.num_cells:
            raise ValueError(f"chunk cells {chunk.cells} beyond the {group.num_cells}")
        start = int(column_offset) + chunk.columns.start
        stop = int(column_offset) + chunk.columns.stop
        if start < 0 or stop > self.n_pixels:
            raise ValueError(f"columns [{start}, {stop}) beyond the {self.n_pixels}")
        bp = self.block_pixels
        values = chunk.values.to(self.device)
        for b in range(start // bp, -(-stop // bp)):
            b0, b1 = max(start, b * bp), min(stop, (b + 1) * bp)
            block = group.data[b, chunk.plane : chunk.plane + 1]
            group.engine.write_block(
                block,
                chunk.cells,
                slice(b0 - b * bp, b1 - b * bp),
                values[None, :, b0 - start : b1 - start],
            )
        n_cells = chunk.cells.stop - chunk.cells.start
        self._written[group.id, chunk.plane] += n_cells * (stop - start)

    def check_complete(self) -> None:
        """Raise unless every cell of every plane was written at every column."""
        for g in self.groups:
            expected = g.num_cells * self.n_pixels
            short = [
                p
                for p, written in zip(
                    self.planes, self._written[g.id].tolist(), strict=True
                )
                if written < expected
            ]
            if short:
                raise ValueError(
                    f"rectangle {g.rect}: planes {short} hold fewer than {expected} "
                    "cell-columns"
                )

    # -- reading --------------------------------------------------------------------

    def blocks(self, groups: Sequence[ColumnGroup]) -> Iterator[ReadyBlock]:
        """The blocks of ``groups`` in column order."""
        for g in groups:
            if g not in self.groups:
                raise ValueError("group does not belong to this store")
        for i in range(self.n_blocks):
            start, stop = self.block_range(i)
            yield ReadyBlock(
                i, (start, stop), stop - start, {g.id: g.data[i] for g in groups}
            )


class ColumnSubset:
    """Some columns of a store, served as blocks of their own size.

    Column ``s`` of the subset is column ``columns[s]`` of the store; each block is
    gathered from the store when read (the tail block zero-padded), so nothing is
    copied until the correlation asks for it. The groups are the store's own
    (``groups_covering`` delegates), which keeps the hypothesis loop's cache keys and
    plane lookup identical to the store's.

    Parameters
    ----------
    store : FeatureStore
        The store the columns come from.
    columns : torch.Tensor
        ``(S,)`` int64 store columns (any order; a column may repeat).
    block_pixels : int
        Columns per block of the subset (the seed pass's memory plan).
    """

    def __init__(
        self, store: FeatureStore, columns: torch.Tensor, block_pixels: int
    ) -> None:
        columns = columns.to(store.device, torch.long).reshape(-1)
        if columns.numel() and (
            int(columns.min()) < 0 or int(columns.max()) >= store.n_pixels
        ):
            raise ValueError("subset columns must lie within the store")
        if block_pixels < 1:
            raise ValueError("block_pixels must be >= 1")
        self.store = store
        self.columns = columns
        self.n_pixels = int(columns.numel())
        self.block_pixels = int(block_pixels)
        self.device = store.device
        self.precision = store.precision
        self.planes = store.planes
        self.groups = store.groups

    @property
    def n_blocks(self) -> int:
        """Blocks along the subset's pixel axis."""
        return -(-self.n_pixels // self.block_pixels)

    def block_range(self, index: int) -> tuple[int, int]:
        """Subset columns ``[start, stop)`` of block ``index``."""
        start = index * self.block_pixels
        return start, min(start + self.block_pixels, self.n_pixels)

    def groups_covering(
        self, rect: Sequence[int], precision: Precision | str, planes: Sequence[int]
    ) -> list[tuple[ColumnGroup, int]]:
        """The store's groups (see :meth:`FeatureStore.groups_covering`)."""
        return self.store.groups_covering(rect, precision, planes)

    def blocks(self, groups: Sequence[ColumnGroup]) -> Iterator[ReadyBlock]:
        """The subset's blocks, gathered from the store one at a time."""
        for g in groups:
            if g not in self.groups:
                raise ValueError("group does not belong to this store")
        bp = self.store.block_pixels
        for i in range(self.n_blocks):
            start, stop = self.block_range(i)
            cols = self.columns[start:stop]
            b_idx, p_idx = cols // bp, cols % bp
            operands = {}
            for g in groups:
                gathered = g.data[b_idx, :, :, p_idx]  # (n, D, K, inner)
                block = torch.zeros(
                    (*g.data.shape[1:3], self.block_pixels, g.data.shape[4]),
                    dtype=g.data.dtype,
                    device=g.data.device,
                )
                block[:, :, : stop - start] = gathered.permute(1, 2, 0, 3)
                operands[g.id] = block
            yield ReadyBlock(i, (start, stop), stop - start, operands)
