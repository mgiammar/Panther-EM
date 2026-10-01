"""Feature store: pixel-batch-aligned blocks of featurized pixels, by column group."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import torch

from panther_em.inference.search.contraction import ContractionEngine, make_engine
from panther_em.inference.search.types import Precision

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from panther_em.inference.search.types import (
        PixelSelection,
        Rectangle,
        SearchGeometry,
    )

Residency = Literal["device", "pinned_host", "transient"]


@dataclass(eq=False)
class ColumnGroup:
    """One rectangle of feature space, featurized for the store's pixels.

    Attributes
    ----------
    id : int
        Position in :attr:`FeatureStore.groups`.
    rect : Rectangle
        ``(k_start, m_start, k_extent, m_extent)``.
    n_defocus : int
        Defocus planes ``D``.
    precision : Precision
        Decides the block layout/dtype through :attr:`engine`.
    engine : ContractionEngine
        The engine whose A-operand layout the blocks use.
    blocks : list[torch.Tensor]
        One ``engine.block_shape(D, num_k, num_m, block_pixels)`` tensor per block.
    complete : bool
        Set by :meth:`BlockWriter.close` once every cell of every column is written.
    """

    id: int
    rect: Rectangle
    n_defocus: int
    precision: Precision
    engine: ContractionEngine
    blocks: list[torch.Tensor] = field(default_factory=list)
    complete: bool = False

    @property
    def num_k(self) -> int:
        """Angular frequencies in the rectangle."""
        return self.rect[2]

    @property
    def num_m(self) -> int:
        """Eigenvectors in the rectangle."""
        return self.rect[3]

    @property
    def num_cells(self) -> int:
        """``num_k * num_m``."""
        return self.rect[2] * self.rect[3]

    @property
    def nbytes(self) -> int:
        """Bytes across all blocks."""
        return sum(b.numel() * b.element_size() for b in self.blocks)


@dataclass(eq=False)
class ReadyBlock:
    """One pixel block of every requested group, resident on the compute device.

    Attributes
    ----------
    index : int
        Block index.
    col_range : tuple[int, int]
        Compact columns ``[start, stop)`` this block covers (``stop - start`` may be
        less than ``block_pixels`` for the tail block).
    n_valid : int
        ``stop - start``.
    operands : Mapping[int, torch.Tensor]
        ``group.id -> (D, num_k, block_pixels, inner)`` block tensor (engine layout).
    ready : torch.cuda.Event or None
        Fires when the block's data is on the device; ``None`` when no wait is needed.
    """

    index: int
    col_range: tuple[int, int]
    n_valid: int
    operands: Mapping[int, torch.Tensor]
    ready: torch.cuda.Event | None = None
    _release: object = None

    def release(self, done: torch.cuda.Event | None = None) -> None:
        """Hand the block's slot back once the consumer recorded ``done``."""
        if callable(self._release):
            self._release(done)


@runtime_checkable
class PixelBlockSource(Protocol):
    """What the inner loop iterates: blocks in order, each ready on the device."""

    def __iter__(self) -> Iterator[ReadyBlock]:
        """Blocks in pixel order."""
        ...

    def __len__(self) -> int:
        """Number of blocks."""
        ...


class _DirectBlockSource:
    """Device residency: the store's blocks *are* the operands (no copy)."""

    def __init__(self, store: FeatureStore, groups: Sequence[ColumnGroup]) -> None:
        self._store = store
        self._groups = list(groups)

    def __len__(self) -> int:
        return self._store.n_blocks

    def __iter__(self) -> Iterator[ReadyBlock]:
        for i in range(self._store.n_blocks):
            start, stop = self._store.block_range(i)
            yield ReadyBlock(
                index=i,
                col_range=(start, stop),
                n_valid=stop - start,
                operands={g.id: g.blocks[i] for g in self._groups},
                ready=None,
            )


class BlockWriter:
    """Writes featurized columns of one group, splitting across block boundaries."""

    def __init__(self, store: FeatureStore, group: ColumnGroup) -> None:
        self.store = store
        self.group = group
        self._written_cells = 0

    def write(
        self,
        cells: slice,
        col_range: tuple[int, int],
        data: torch.Tensor,
        *,
        plane: int | None = None,
    ) -> None:
        """Write ``data`` for cells ``cells`` (row-major within the rectangle).

        Parameters
        ----------
        cells : slice
            ``[c0, c1)`` into the rectangle's row-major ``(k, m)`` cells.
        col_range : tuple[int, int]
            Compact columns ``[start, stop)`` the data covers.
        data : torch.Tensor
            Complex features ``(D, n_cells, n_cols)``, or ``(n_cells, n_cols)`` when
            ``plane`` is given.
        plane : int, optional
            Write only defocus plane ``plane``.
        """
        start, stop = col_range
        if plane is None:
            if data.dim() != 3 or data.shape[0] != self.group.n_defocus:
                raise ValueError(
                    f"data must be (D={self.group.n_defocus}, n_cells, n_cols), "
                    f"got {tuple(data.shape)}"
                )
        elif data.dim() != 2:
            raise ValueError(
                f"plane data must be (n_cells, n_cols), got {tuple(data.shape)}"
            )
        n_cells = (cells.stop or 0) - (cells.start or 0)
        if data.shape[-2] != n_cells or data.shape[-1] != stop - start:
            raise ValueError(
                f"data {tuple(data.shape)} does not match cells {cells} x columns "
                f"{col_range}"
            )
        bp = self.store.block_pixels
        engine = self.group.engine
        first, last = start // bp, (stop - 1) // bp
        for b in range(first, last + 1):
            b0 = max(start, b * bp)
            b1 = min(stop, (b + 1) * bp)
            local = slice(b0 - b * bp, b1 - b * bp)
            chunk = data[..., b0 - start : b1 - start]
            block = self.group.blocks[b]
            if plane is None:
                engine.write_block(block, cells, local, chunk.to(block.device))
            else:
                engine.write_block(
                    block[plane : plane + 1], cells, local, chunk.to(block.device)[None]
                )
        if plane is None or plane == self.group.n_defocus - 1:
            self._written_cells += n_cells * (stop - start)

    def close(self) -> None:
        """Mark the group complete (every cell x column written at least once)."""
        expected = self.group.num_cells * self.store.n_pixels
        if self._written_cells < expected:
            raise ValueError(
                f"group {self.group.id} incomplete: {self._written_cells} of "
                f"{expected} "
                "cell-columns written"
            )
        self.group.complete = True


class FeatureStore:
    """Featurized pixels for one pixel set, as block-aligned column groups.

    Parameters
    ----------
    geometry : SearchGeometry
        The valid-correlation grid.
    pixels : PixelSelection
        The searched pixels (sorted flat indices).
    column_index : torch.Tensor
        ``(P,)`` int64 flat pixel index of every compact column (a permutation of
        ``pixels.index``, chosen by the featurizer).
    block_pixels : int
        Pixels per block; a multiple of the GEMM pixel batch.
    n_defocus : int
        Defocus planes ``D`` stored per column.
    device : torch.device or str
        Where blocks live (device residency).
    residency : {"device"}
        Only device residency exists yet.
    """

    def __init__(
        self,
        geometry: SearchGeometry,
        pixels: PixelSelection,
        column_index: torch.Tensor,
        *,
        block_pixels: int,
        n_defocus: int,
        device: torch.device | str,
        residency: Residency = "device",
    ) -> None:
        if residency != "device":
            raise NotImplementedError(f"residency {residency!r} not implemented yet")
        if column_index.dim() != 1 or int(column_index.numel()) != len(pixels):
            raise ValueError("column_index must be a (P,) permutation of pixels.index")
        if block_pixels < 1:
            raise ValueError("block_pixels must be >= 1")
        self.geometry = geometry
        self.pixels = pixels
        self.column_index = column_index.to(torch.long)
        self.block_pixels = int(block_pixels)
        self.n_defocus = int(n_defocus)
        self.device = torch.device(device)
        self.residency: Residency = residency
        self.groups: list[ColumnGroup] = []

    # -- geometry of the pixel axis -------------------------------------------

    @property
    def n_pixels(self) -> int:
        """Compact columns ``P``."""
        return int(self.column_index.numel())

    @property
    def n_blocks(self) -> int:
        """Blocks along the pixel axis (the last may be partial)."""
        return max(1, -(-self.n_pixels // self.block_pixels)) if self.n_pixels else 0

    def block_range(self, index: int) -> tuple[int, int]:
        """Compact columns ``[start, stop)`` of block ``index``."""
        start = index * self.block_pixels
        return start, min(start + self.block_pixels, self.n_pixels)

    # -- groups ---------------------------------------------------------------

    def group_for(
        self, rect: Rectangle, precision: Precision | str | None = None
    ) -> ColumnGroup | None:
        """The group holding exactly ``rect`` (at ``precision`` when given), if any."""
        rect = tuple(int(v) for v in rect)  # type: ignore[assignment]
        wanted = None if precision is None else Precision.coerce(precision)
        for g in self.groups:
            if g.rect == rect and (wanted is None or g.precision is wanted):
                return g
        return None

    def add_group(self, rect: Rectangle, precision: Precision | str) -> ColumnGroup:
        """Allocate zeroed blocks for a new rectangle (no featurization here).

        The same rectangle may be held once per precision (a later stage at a higher
        precision gets its own group).
        """
        rect = tuple(int(v) for v in rect)  # type: ignore[assignment]
        if self.group_for(rect, precision) is not None:
            raise ValueError(f"rectangle {rect} already has a {precision} group")
        engine = make_engine(precision)
        group = ColumnGroup(
            id=len(self.groups),
            rect=rect,
            n_defocus=self.n_defocus,
            precision=engine.precision,
            engine=engine,
        )
        shape = engine.block_shape(self.n_defocus, rect[2], rect[3], self.block_pixels)
        group.blocks = [
            torch.zeros(shape, dtype=engine.block_dtype, device=self.device)
            for _ in range(self.n_blocks)
        ]
        self.groups.append(group)
        return group

    def writer(self, group: ColumnGroup) -> BlockWriter:
        """A writer for ``group``."""
        return BlockWriter(self, group)

    def blocks(self, groups: Sequence[ColumnGroup]) -> PixelBlockSource:
        """Iterate the blocks of ``groups`` in order, ready on the device."""
        for g in groups:
            if g not in self.groups:
                raise ValueError("group does not belong to this store")
            if not g.complete:
                raise ValueError(f"group {g.id} is not complete")
        return _DirectBlockSource(self, groups)

    def read(self, group: ColumnGroup, plane: int = 0) -> torch.Tensor:
        """``(P, num_k, num_m)`` complex64 features of one plane (tests/compaction)."""
        parts = []
        for i, block in enumerate(group.blocks):
            start, stop = self.block_range(i)
            parts.append(group.engine.read_block(block, plane, slice(0, stop - start)))
        return torch.cat(parts, dim=0)

    def nbytes(self) -> int:
        """Bytes across all groups."""
        return sum(g.nbytes for g in self.groups)

    # -- compaction -----------------------------------------------------------

    def subset(
        self, keep: torch.Tensor, *, block_pixels: int | None = None
    ) -> FeatureStore:
        """Keep the compact columns where ``keep`` is True (order preserved).

        Works block by block, so the transient peak is the new store plus one old
        block's gathered columns.
        """
        if tuple(keep.shape) != (self.n_pixels,) or keep.dtype != torch.bool:
            raise ValueError(f"keep must be a ({self.n_pixels},) bool tensor")
        keep_cpu = keep.to("cpu")
        new_cols = self.column_index[keep_cpu]
        new_pixels = self.pixels.subset(
            torch.isin(self.pixels.index, new_cols.to(self.pixels.index.device))
        )
        new = FeatureStore(
            self.geometry,
            new_pixels,
            new_cols,
            block_pixels=block_pixels or self.block_pixels,
            n_defocus=self.n_defocus,
            device=self.device,
            residency=self.residency,
        )
        positions = torch.nonzero(keep_cpu, as_tuple=False).squeeze(-1)
        for g in self.groups:
            ng = new.add_group(g.rect, g.precision)
            cursor = 0
            for i, block in enumerate(g.blocks):
                start, stop = self.block_range(i)
                sel = positions[(positions >= start) & (positions < stop)] - start
                if sel.numel() == 0:
                    continue
                gathered = block.index_select(
                    2, sel.to(block.device)
                )  # (D, k, n, inner)
                n = int(sel.numel())
                _scatter_columns(new, ng, cursor, gathered)
                cursor += n
            ng.complete = g.complete
        return new


def _scatter_columns(
    store: FeatureStore, group: ColumnGroup, start: int, data: torch.Tensor
) -> None:
    """Place engine-layout columns ``data (D, num_k, n, inner)`` at ``start``."""
    n = int(data.shape[2])
    bp = store.block_pixels
    stop = start + n
    for b in range(start // bp, (stop - 1) // bp + 1):
        b0, b1 = max(start, b * bp), min(stop, (b + 1) * bp)
        group.blocks[b][:, :, b0 - b * bp : b1 - b * bp, :] = data[
            :, :, b0 - start : b1 - start, :
        ]
