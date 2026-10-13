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

Residency = Literal["device", "pinned_host"]


@dataclass(eq=False)
class ColumnGroup:
    """One rectangle of feature space, featurized for the store's pixels.

    Attributes
    ----------
    id : int
        Position in :attr:`FeatureStore.groups`.
    rect : Rectangle
        ``(k_start, m_start, k_extent, m_extent)``.
    planes : tuple[int, ...]
        Global defocus plane ids held, in the order the blocks' leading axis
        stores them (sorted ascending).
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
    planes: tuple[int, ...]
    precision: Precision
    engine: ContractionEngine
    blocks: list[torch.Tensor] = field(default_factory=list)
    complete: bool = False

    @property
    def n_defocus(self) -> int:
        """Planes held, ``D_g = len(planes)``."""
        return len(self.planes)

    def holds(self, plane: int) -> bool:
        """Whether global plane ``plane`` lives in this group."""
        return int(plane) in self.planes

    def local_plane(self, plane: int) -> int:
        """Position of global plane ``plane`` in the blocks' leading axis."""
        try:
            return self.planes.index(int(plane))
        except ValueError as exc:
            raise ValueError(
                f"group {self.id} holds planes {self.planes}, not {plane}"
            ) from exc

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


class _PinnedBlockSource:
    """Pinned-host residency: blocks stream through ``n_slots`` device slots.

    Block ``i`` is copied (on the store's copy stream) into slot ``i % n_slots`` as
    soon as that slot's previous block has been released by the consumer; copies run
    one block ahead of compute.
    """

    def __init__(self, store: FeatureStore, groups: Sequence[ColumnGroup]) -> None:
        self._store = store
        self._groups = list(groups)
        n = store.n_slots
        self._slots = {
            g.id: [torch.empty_like(g.blocks[0], device=store.device) for _ in range(n)]
            for g in self._groups
        }
        self._free: list[torch.cuda.Event | None] = [None] * n
        self._stream = store.copy_stream or torch.cuda.current_stream(store.device)

    def __len__(self) -> int:
        return self._store.n_blocks

    def _issue(self, i: int) -> torch.cuda.Event:
        slot = i % self._store.n_slots
        with torch.cuda.stream(self._stream):
            free = self._free[slot]
            if free is not None:
                self._stream.wait_event(free)
            for g in self._groups:
                self._slots[g.id][slot].copy_(g.blocks[i], non_blocking=True)
            ready = torch.cuda.Event()  # type: ignore[no-untyped-call]
            ready.record(self._stream)
        return ready

    def __iter__(self) -> Iterator[ReadyBlock]:
        n_blocks, n_slots = self._store.n_blocks, self._store.n_slots
        ready = {i: self._issue(i) for i in range(min(n_slots - 1, n_blocks))}
        for i in range(n_blocks):
            start, stop = self._store.block_range(i)
            slot = i % n_slots

            def release(done: torch.cuda.Event | None, slot: int = slot) -> None:
                if done is None:
                    done = torch.cuda.Event()  # type: ignore[no-untyped-call]
                    done.record(
                        torch.cuda.current_stream(self._store.device)
                    )
                self._free[slot] = done

            yield ReadyBlock(
                index=i,
                col_range=(start, stop),
                n_valid=stop - start,
                operands={g.id: self._slots[g.id][slot] for g in self._groups},
                ready=ready.pop(i),
                _release=release,
            )
            nxt = i + n_slots - 1
            if nxt < n_blocks:
                ready[nxt] = self._issue(nxt)


class BlockWriter:
    """Writes featurized columns of one group, splitting across block boundaries."""

    def __init__(self, store: FeatureStore, group: ColumnGroup) -> None:
        self.store = store
        self.group = group
        self._written = [0] * group.n_defocus  # cell-columns written per plane

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
            Complex features ``(D_g, n_cells, n_cols)`` in the group's plane order, or
            ``(n_cells, n_cols)`` when ``plane`` is given.
        plane : int, optional
            Write only the **global** defocus plane ``plane`` (which the group must
            hold).
        """
        start, stop = col_range
        local_plane = None if plane is None else self.group.local_plane(plane)
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
            block = self.store._write_target(self.group, b)
            if local_plane is None:
                engine.write_block(block, cells, local, chunk.to(block.device))
            else:
                engine.write_block(
                    block[local_plane : local_plane + 1],
                    cells,
                    local,
                    chunk.to(block.device)[None],
                )
            planes_written = self.group.n_defocus if local_plane is None else 1
            self.store._note_written(self.group, b, n_cells * (b1 - b0) * planes_written)
        written = n_cells * (stop - start)
        if local_plane is None:
            self._written = [w + written for w in self._written]
        else:
            self._written[local_plane] += written

    def close(self) -> None:
        """Mark the group complete (every cell x column written at least once)."""
        expected = self.group.num_cells * self.store.n_pixels
        short = [
            (plane, w)
            for plane, w in zip(self.group.planes, self._written, strict=True)
            if w < expected
        ]
        if short:
            raise ValueError(
                f"group {self.group.id} incomplete: planes {short} have fewer than "
                f"{expected} cell-columns written"
            )
        self.store.flush_staging()
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
        Defocus planes the image set carries (the valid range of plane ids; each group
        holds a subset, though may also hold all defocus planes).
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
        if residency not in ("device", "pinned_host"):
            raise ValueError(f"residency must be 'device' or 'pinned_host', got {residency!r}")
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
        # pinned_host: blocks live in pinned host memory; writes land in device
        # staging blocks that are copied out once every cell x plane of a block has
        # been written (pixel-major featurization completes blocks tile by tile), and
        # the executor streams blocks back through a few device slots.
        self.copy_stream: torch.cuda.Stream | None = None
        self.staging_bytes = 8 << 30  # device memory for partially written blocks
        self.n_slots = 3  # device slots the executor streams through
        self._staging: dict[tuple[int, int], torch.Tensor] = {}
        self._staged_written: dict[tuple[int, int], int] = {}

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

    def _planes(self, planes: Sequence[int] | None) -> tuple[int, ...]:
        if planes is None:
            return tuple(range(self.n_defocus))
        ids = tuple(sorted({int(p) for p in planes}))
        if not ids:
            raise ValueError("planes must not be empty")
        if ids[0] < 0 or ids[-1] >= self.n_defocus:
            raise ValueError(
                f"planes {ids} outside the image set's [0, {self.n_defocus}) planes"
            )
        return ids

    def group_for(
        self,
        rect: Rectangle,
        precision: Precision | str | None = None,
        planes: Sequence[int] | None = None,
    ) -> ColumnGroup | None:
        """The group holding ``rect`` (at ``precision`` / ``planes`` if given)."""
        rect = tuple(int(v) for v in rect)  # type: ignore[assignment]
        wanted = None if precision is None else Precision.coerce(precision)
        wanted_planes = None if planes is None else self._planes(planes)
        for g in self.groups:
            if (
                g.rect == rect
                and (wanted is None or g.precision is wanted)
                and (wanted_planes is None or g.planes == wanted_planes)
            ):
                return g
        return None

    def groups_covering(
        self, rect: Rectangle, precision: Precision | str, planes: Sequence[int]
    ) -> list[tuple[ColumnGroup, int]]:
        """``(group, local plane index)`` for each of ``planes``, in request order."""
        rect = tuple(int(v) for v in rect)  # type: ignore[assignment]
        wanted = Precision.coerce(precision)
        out: list[tuple[ColumnGroup, int]] = []
        missing = []
        for plane in planes:
            for g in self.groups:
                if g.rect == rect and g.precision is wanted and g.holds(plane):
                    out.append((g, g.local_plane(plane)))
                    break
            else:
                missing.append(int(plane))
        if missing:
            raise KeyError(
                f"no {wanted.value} group of rectangle {rect} holds planes {missing}"
            )
        return out

    def missing_planes(
        self, rect: Rectangle, precision: Precision | str, planes: Sequence[int]
    ) -> tuple[int, ...]:
        """The subset of ``planes`` no group of ``rect`` at ``precision`` holds."""
        rect = tuple(int(v) for v in rect)  # type: ignore[assignment]
        wanted = Precision.coerce(precision)
        held = {
            p
            for g in self.groups
            if g.rect == rect and g.precision is wanted
            for p in g.planes
        }
        return tuple(int(p) for p in planes if int(p) not in held)

    def add_group(
        self,
        rect: Rectangle,
        precision: Precision | str,
        planes: Sequence[int] | None = None,
    ) -> ColumnGroup:
        """Allocate zeroed blocks for a new rectangle (no featurization here)."""
        rect = tuple(int(v) for v in rect)  # type: ignore[assignment]
        ids = self._planes(planes)
        wanted = Precision.coerce(precision)
        overlap = sorted(
            p
            for g in self.groups
            if g.rect == rect and g.precision is wanted
            for p in g.planes
            if p in ids
        )
        if overlap:
            raise ValueError(
                f"rectangle {rect} already holds planes {overlap} at {wanted.value}"
            )
        engine = make_engine(precision)
        group = ColumnGroup(
            id=len(self.groups),
            rect=rect,
            planes=ids,
            precision=engine.precision,
            engine=engine,
        )
        shape = engine.block_shape(len(ids), rect[2], rect[3], self.block_pixels)
        if self.residency == "pinned_host":
            group.blocks = [
                torch.empty(shape, dtype=engine.block_dtype, pin_memory=True)
                for _ in range(self.n_blocks)
            ]
        else:
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
        if self.residency == "pinned_host":
            return _PinnedBlockSource(self, groups)
        return _DirectBlockSource(self, groups)

    # -- pinned-host staging (write side) ----------------------------------------

    def _write_target(self, group: ColumnGroup, b: int) -> torch.Tensor:
        """The device tensor a writer fills for block ``b`` of ``group``."""
        if self.residency != "pinned_host":
            return group.blocks[b]
        key = (group.id, b)
        staged = self._staging.get(key)
        if staged is None:
            block_bytes = group.blocks[b].numel() * group.blocks[b].element_size()
            if (len(self._staging) + 1) * block_bytes > self.staging_bytes:
                raise RuntimeError(
                    f"{len(self._staging)} partially written blocks staged on the "
                    f"device ({self.staging_bytes >> 30} GiB): pinned_host residency "
                    "needs a featurization order that completes blocks tile by tile "
                    "(pixel_major) and a tile that fits the staging budget"
                )
            staged = torch.zeros_like(group.blocks[b], device=self.device)
            self._staging[key] = staged
            self._staged_written[key] = 0
        return staged

    def _note_written(self, group: ColumnGroup, b: int, cell_columns: int) -> None:
        """Count writes into a staged block; flush it to the host when complete."""
        if self.residency != "pinned_host":
            return
        key = (group.id, b)
        self._staged_written[key] += cell_columns
        start, stop = self.block_range(b)
        expected = group.num_cells * group.n_defocus * (stop - start)
        if self._staged_written[key] >= expected:
            self._flush(key, group)

    def _flush(self, key: tuple[int, int], group: ColumnGroup) -> None:
        staged = self._staging.pop(key)
        self._staged_written.pop(key)
        # same stream as the writes, so the copy follows them and the staging memory
        # may be recycled stream-ordered
        group.blocks[key[1]].copy_(staged, non_blocking=True)

    def flush_staging(self) -> None:
        """Copy out whatever is still staged (e.g. the tail block) and wait."""
        for key in list(self._staging):
            group = self.groups[key[0]]
            self._flush(key, group)
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

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
            ng = new.add_group(g.rect, g.precision, g.planes)
            cursor = 0
            for i, block in enumerate(g.blocks):
                start, stop = self.block_range(i)
                sel = positions[(positions >= start) & (positions < stop)] - start
                if sel.numel() == 0:
                    continue
                gathered = block.index_select(2, sel.to(block.device))
                _scatter_columns(new, ng, cursor, gathered)
                cursor += int(sel.numel())
            ng.complete = g.complete
        return new

    def gather_columns(
        self, group: ColumnGroup, positions: torch.Tensor
    ) -> torch.Tensor:
        """Engine-layout columns ``(D_g, num_k, n, inner)`` at compact ``positions``.

        Gathers block by block (``positions`` need not be sorted; order is kept).
        """
        pos = positions.to(torch.long)
        if pos.numel() == 0:
            d, k, _, inner = group.blocks[0].shape
            return torch.empty(
                (d, k, 0, inner), dtype=group.blocks[0].dtype, device=self.device
            )
        dev_pos = pos.to(self.device)
        block_of = dev_pos // self.block_pixels
        within = dev_pos % self.block_pixels
        out = torch.empty(
            (*group.blocks[0].shape[:2], int(pos.numel()), group.blocks[0].shape[-1]),
            dtype=group.blocks[0].dtype,
            device=self.device,
        )
        for b in torch.unique(block_of).tolist():
            where = torch.nonzero(block_of == b, as_tuple=False).squeeze(-1)
            blk = group.blocks[b]
            idx = within[where].to(blk.device)
            out[:, :, where, :] = blk.index_select(2, idx).to(self.device)
        return out


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
