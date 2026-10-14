"""Feature space layouts, the contraction engines and the feature store.

* :class:`FeatureLayout` -- pairwise cell-disjoint rectangles of ``(angular frequency,
  eigenvector)`` space and their *k-runs*: maximal runs of ``k`` covered by a constant
  rectangle set, each contracted as one batched GEMM.
* :class:`ComplexGemmEngine` / :class:`Fp16GemmEngine` -- one precision's GEMM plus the
  block layout it consumes (``(D, num_k, P, num_m)`` complex64, or
  ``(D, num_k, P, 2 num_m)`` float16 as ``[Re Y | Im Y]`` with interleaved weights so
  the real GEMM output is bit-identical to a complex spectrum). The spectrum is always
  the kernels' native contiguous ``(F, P, N)`` (:class:`Spectrum`).
* :class:`FeatureStore` -- the featurized pixels of one pixel set as block-aligned
  :class:`ColumnGroup` s (one rectangle, one precision, some defocus planes), resident
  on the device or pinned on the host and streamed; :class:`BlockWriter` fills them,
  :meth:`FeatureStore.blocks` hands them to the hypothesis loop as :class:`ReadyBlock` s.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Literal, NamedTuple

import torch

from panther_em.inference.search.types import Precision

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from panther_em.inference.search.types import (
        PixelSelection,
        Rectangle,
        SearchGeometry,
    )

Residency = Literal["device", "pinned_host"]
# --------------------------------------------------------------------------- #
# Feature layouts: rectangles of (k, m) space and their k-runs
# --------------------------------------------------------------------------- #


def as_rectangle(rect: Sequence[int]) -> Rectangle:
    """``(k_start, m_start, k_extent, m_extent)`` as validated Python ints."""
    if len(rect) != 4:
        raise ValueError(
            f"a rectangle is (k_start, m_start, k_extent, m_extent), got {rect!r}"
        )
    k0, m0, k_ext, m_ext = (int(v) for v in rect)
    if k0 < 0 or m0 < 0:
        raise ValueError(f"rectangle starts must be >= 0, got {rect!r}")
    if k_ext <= 0 or m_ext <= 0:
        raise ValueError(f"rectangle extents must be > 0, got {rect!r}")
    return (k0, m0, k_ext, m_ext)


def rectangles_disjoint(rectangles: Sequence[Rectangle]) -> bool:
    """Whether ``(k_start, m_start, k_extent, m_extent)`` rectangles never overlap."""
    rects = list(rectangles)
    for i in range(len(rects)):
        k0, m0, ke, me = rects[i]
        for j in range(i + 1, len(rects)):
            l0, n0, le, ne = rects[j]
            if k0 < l0 + le and l0 < k0 + ke and m0 < n0 + ne and n0 < m0 + me:
                return False
    return True


class LayoutPart(NamedTuple):
    """One rectangle's contribution to a :class:`KRun`: its index into
    :attr:`FeatureLayout.rectangles` and its global eigenvector range."""

    rect: int
    m_start: int
    m_stop: int

    @property
    def num_m(self) -> int:
        """Eigenvectors contributed."""
        return self.m_stop - self.m_start


@dataclass(frozen=True)
class KRun:
    """A run ``[k_start, k_stop)`` of ``k`` covered by a fixed rectangle set."""

    k_start: int
    k_stop: int
    parts: tuple[LayoutPart, ...]

    @property
    def num_k(self) -> int:
        """Angular frequencies in the run."""
        return self.k_stop - self.k_start

    @property
    def num_m(self) -> int:
        """Total inner (eigenvector) dimension of the run's GEMM."""
        return sum(p.num_m for p in self.parts)


@dataclass(frozen=True)
class FeatureLayout:
    """Pure, hashable description of a rectangle set in ``(k, m)`` feature space.

    Parameters
    ----------
    rectangles : sequence of Rectangle
        Pairwise cell-disjoint ``(k_start, m_start, k_extent, m_extent)`` blocks.
    """

    rectangles: tuple[Rectangle, ...]

    @classmethod
    def create(cls, rectangles: Sequence[Sequence[int]]) -> FeatureLayout:
        """Build a layout from any sequence of 4-int sequences."""
        return cls(tuple(as_rectangle(r) for r in rectangles))

    def __post_init__(self) -> None:
        """Validate (never coerce): int 4-tuples, pairwise cell-disjoint."""
        rects = self.rectangles
        if not isinstance(rects, tuple) or not rects:
            raise ValueError(
                "FeatureLayout needs a non-empty tuple of rectangles; see create()"
            )
        for r in rects:
            if r != as_rectangle(r):
                raise ValueError(f"bad rectangle {r!r}; see FeatureLayout.create()")
        if not rectangles_disjoint(rects):
            raise ValueError("rectangles must be pairwise cell-disjoint")

    @cached_property
    def k_runs(self) -> tuple[KRun, ...]:
        """Maximal ``k`` runs covered by a constant rectangle set (one GEMM each)."""
        rects = self.rectangles
        bounds = sorted({r[0] for r in rects} | {r[0] + r[2] for r in rects})
        runs: list[KRun] = []
        for k_start, k_stop in itertools.pairwise(bounds):
            parts = tuple(
                LayoutPart(i, r[1], r[1] + r[3])
                for i, r in enumerate(rects)
                if r[0] <= k_start and r[0] + r[2] >= k_stop
            )
            if parts:
                runs.append(KRun(k_start, k_stop, parts))
        return tuple(runs)

    @property
    def num_features(self) -> int:
        """Total ``(k, m)`` cells."""
        return sum(r[2] * r[3] for r in self.rectangles)

    @property
    def k_stop(self) -> int:
        """Smallest angular-frequency axis length holding every rectangle."""
        return max(r[0] + r[2] for r in self.rectangles)

    def covered_k(self) -> tuple[bool, ...]:
        """For each ``k < k_stop``, whether some rectangle covers it."""
        covered = [False] * self.k_stop
        for run in self.k_runs:
            for k in range(run.k_start, run.k_stop):
                covered[k] = True
        return tuple(covered)


# --------------------------------------------------------------------------- #
# Contraction engines: one precision's GEMM and the block layout it consumes
# --------------------------------------------------------------------------- #


def _weights_fp16_4m(w_conj: torch.Tensor) -> torch.Tensor:
    r"""Interleave a pre-conjugated ``(N, num_k, num_m)`` block for the fp16 4M GEMM.

    The complex product ``C = Y @ w`` (``w`` already conjugated) is computed as ONE
    real GEMM per ``k``: ``[Yr | Yi] (P, 2 num_m) @ B (2 num_m, 2N)`` with the columns
    of ``B`` interleaved so that ``out[:, 2n] = Cr[:, n]`` and
    ``out[:, 2n+1] = Ci[:, n]``::

        Cr = Yr wr - Yi wi  ->  B[:num_m, 2n]   = wr[:, n], B[num_m:, 2n]   = -wi[:, n]
        Ci = Yr wi + Yi wr  ->  B[:num_m, 2n+1] = wi[:, n], B[num_m:, 2n+1] =  wr[:, n]

    The real ``(num_k, P, 2N)`` output is therefore bit-identical in memory to a
    ``(num_k, P, N)`` complex tensor -- exactly the layout the fused reduce kernels
    consume, with no transposes or copies.

    Parameters
    ----------
    w_conj : torch.Tensor
        Pre-conjugated complex weights ``(N, num_k, num_m)``.

    Returns
    -------
    torch.Tensor
        float16 ``(num_k, 2 num_m, 2N)``, contiguous.
    """
    n, num_k, num_m = w_conj.shape
    wk = w_conj.permute(1, 2, 0)  # (num_k, num_m, N)
    b = torch.empty(
        (num_k, 2 * num_m, 2 * n), dtype=torch.float16, device=w_conj.device
    )
    b[:, :num_m, 0::2] = wk.real
    b[:, num_m:, 0::2] = -wk.imag
    b[:, :num_m, 1::2] = wk.imag
    b[:, num_m:, 1::2] = wk.real
    return b


@dataclass(frozen=True, eq=False)
class Spectrum:
    """Angular-frequency spectrum ``C`` of one (pixel batch, hypothesis batch).

    Attributes
    ----------
    native : torch.Tensor
        Contiguous ``(F, P, N)`` complex64 or complex32 -- the layout the fused
        reduce kernels read directly and ``torch.bmm`` writes without a copy.
    """

    native: torch.Tensor

    def __post_init__(self) -> None:
        """Require the kernel-native contiguous layout."""
        if self.native.dim() != 3 or not self.native.is_contiguous():
            raise ValueError("Spectrum.native must be a contiguous (F, P, N) tensor")

    @property
    def n_freq(self) -> int:
        """``F``."""
        return int(self.native.shape[0])

    @property
    def n_pixels(self) -> int:
        """``P``."""
        return int(self.native.shape[1])

    @property
    def n_hyp(self) -> int:
        """``N``."""
        return int(self.native.shape[2])

    @property
    def dtype(self) -> torch.dtype:
        """complex64 or complex32."""
        return self.native.dtype

    @property
    def device(self) -> torch.device:
        """Device of the buffer."""
        return self.native.device

    def pnf(self) -> torch.Tensor:
        """``(P, N, F)`` permuted view (no copy)."""
        return self.native.permute(1, 2, 0)

    def zero_uncovered(self, covered: Sequence[bool]) -> None:
        """Zero the ``k`` slabs no rectangle writes (once per buffer, not per batch)."""
        for k, is_covered in enumerate(covered):
            if not is_covered:
                self.native[k].zero_()

    @classmethod
    def empty(
        cls,
        n_freq: int,
        n_pixels: int,
        n_hyp: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> Spectrum:
        """Allocate an uninitialised spectrum buffer."""
        return cls(torch.empty((n_freq, n_pixels, n_hyp), dtype=dtype, device=device))


def _cells_to_rows(cells: slice, num_m: int) -> list[tuple[int, int, int, int, int]]:
    """Split a row-major ``(k, m)`` cell slice into ``(k, m0, m1, i0, i1)`` rows.

    ``(m0, m1)`` is the eigenvector range of row ``k`` and ``(i0, i1)`` the matching
    range within the data chunk (i.e. relative to ``cells.start``).
    """
    c0, c1 = cells.start or 0, cells.stop
    if c1 is None:
        raise ValueError("cell slice needs an explicit stop")
    rows = []
    c = c0
    while c < c1:
        k, m0 = divmod(c, num_m)
        m1 = min(num_m, m0 + (c1 - c))
        rows.append((k, m0, m1, c - c0, c - c0 + (m1 - m0)))
        c += m1 - m0
    return rows


class ComplexGemmEngine:
    """fp32 (cuBLAS cgemm) or tf32 complex contraction."""

    block_dtype = torch.complex64
    spectrum_dtype = torch.complex64

    def __init__(self, precision: Precision | str = Precision.FP32) -> None:
        precision = Precision.coerce(precision)
        if precision not in (Precision.FP32, Precision.TF32):
            raise ValueError(
                f"ComplexGemmEngine handles fp32/tf32, not {precision.value}"
            )
        self.precision: Precision = precision

    def block_shape(
        self, n_defocus: int, num_k: int, num_m: int, n_pixels: int
    ) -> tuple[int, ...]:
        """Shape of a feature block for one rectangle of ``num_k x num_m`` cells."""
        return (n_defocus, num_k, n_pixels, num_m)

    def block_num_m(self, block: torch.Tensor) -> int:
        """Recover ``num_m`` from a block's shape."""
        return int(block.shape[-1])

    def write_block(
        self, block: torch.Tensor, cells: slice, cols: slice, data: torch.Tensor
    ) -> None:
        """Write ``(D, n_cells, n_cols)`` complex features into ``block``."""
        num_m = self.block_num_m(block)
        for k, m0, m1, i0, i1 in _cells_to_rows(cells, num_m):
            # data (D, n_cells, n_cols) -> block[:, k, cols, m0:m1] (D, n_cols, m1-m0)
            block[:, k, cols, m0:m1] = data[:, i0:i1, :].transpose(1, 2).to(block.dtype)

    def operand(
        self,
        block: torch.Tensor,
        d: int,
        k_start: int,
        k_stop: int,
        p_start: int,
        p_stop: int,
    ) -> torch.Tensor:
        """GEMM A operand view ``(num_k, P_g, inner)`` for a k- and pixel-range."""
        return block[d, k_start:k_stop, p_start:p_stop, :]

    def prepare_weights(self, w: torch.Tensor) -> torch.Tensor:
        """Un-conjugated ``(N, num_k, num_m)`` weights -> B operand."""
        # (N, num_k, num_m) -> conj -> (num_k, num_m, N) contiguous B operand
        return w.conj().permute(1, 2, 0).contiguous().to(torch.complex64)

    def contract(
        self,
        a_parts: Sequence[torch.Tensor],
        b_parts: Sequence[torch.Tensor],
        run: KRun,
        out: Spectrum,
    ) -> None:
        """``out.native[run.k_start:run.k_stop] = A @ B`` (parts concatenated)."""
        a = a_parts[0] if len(a_parts) == 1 else torch.cat(list(a_parts), dim=-1)
        b = b_parts[0] if len(b_parts) == 1 else torch.cat(list(b_parts), dim=1)
        slab = out.native[run.k_start : run.k_stop]
        if self.precision is Precision.TF32 and a.is_cuda:
            prev = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                torch.bmm(a, b, out=slab)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = prev
        else:
            torch.bmm(a, b, out=slab)


class Fp16GemmEngine:
    """fp16 tensor-core "4M" contraction; complex32 spectrum."""

    precision: Precision = Precision.FP16
    block_dtype = torch.float16
    spectrum_dtype = torch.complex32

    def block_shape(
        self, n_defocus: int, num_k: int, num_m: int, n_pixels: int
    ) -> tuple[int, ...]:
        """Shape of a feature block for one rectangle of ``num_k x num_m`` cells."""
        return (n_defocus, num_k, n_pixels, 2 * num_m)

    def block_num_m(self, block: torch.Tensor) -> int:
        """Recover ``num_m`` from a block's shape."""
        return int(block.shape[-1]) // 2

    def write_block(
        self, block: torch.Tensor, cells: slice, cols: slice, data: torch.Tensor
    ) -> None:
        """Write ``(D, n_cells, n_cols)`` complex features into ``block``."""
        num_m = self.block_num_m(block)
        for k, m0, m1, i0, i1 in _cells_to_rows(cells, num_m):
            chunk = data[:, i0:i1, :].transpose(1, 2)  # (D, n_cols, m1-m0)
            block[:, k, cols, m0:m1] = chunk.real.to(torch.float16)
            block[:, k, cols, num_m + m0 : num_m + m1] = chunk.imag.to(torch.float16)

    def operand(
        self,
        block: torch.Tensor,
        d: int,
        k_start: int,
        k_stop: int,
        p_start: int,
        p_stop: int,
    ) -> torch.Tensor:
        """GEMM A operand view ``(num_k, P_g, inner)`` for a k- and pixel-range."""
        return block[d, k_start:k_stop, p_start:p_stop, :]

    def prepare_weights(self, w: torch.Tensor) -> torch.Tensor:
        """Un-conjugated ``(N, num_k, num_m)`` weights -> B operand."""
        return _weights_fp16_4m(w.conj().to(torch.complex64))  # (num_k, 2 num_m, 2 N)

    def contract(
        self,
        a_parts: Sequence[torch.Tensor],
        b_parts: Sequence[torch.Tensor],
        run: KRun,
        out: Spectrum,
    ) -> None:
        """``out.native[run.k_start:run.k_stop] = A @ B`` (parts concatenated)."""
        a = a_parts[0] if len(a_parts) == 1 else torch.cat(list(a_parts), dim=-1)
        b = b_parts[0] if len(b_parts) == 1 else torch.cat(list(b_parts), dim=1)
        num_k = run.num_k
        p = int(a.shape[1])
        n = int(b.shape[-1]) // 2
        slab_real = torch.view_as_real(out.native[run.k_start : run.k_stop]).view(
            num_k, p, 2 * n
        )
        torch.bmm(a, b, out=slab_real)


ContractionEngine = ComplexGemmEngine | Fp16GemmEngine


def make_engine(precision: Precision | str) -> ContractionEngine:
    """The engine for a precision."""
    precision = Precision.coerce(precision)
    if precision is Precision.FP16:
        return Fp16GemmEngine()
    return ComplexGemmEngine(precision)


# --------------------------------------------------------------------------- #
# Feature store: block-aligned featurized pixels, by column group
# --------------------------------------------------------------------------- #


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


def _direct_blocks(store: FeatureStore, groups: Sequence[ColumnGroup]) -> Iterator[ReadyBlock]:
    """Device residency: the store's blocks *are* the operands (no copy)."""
    for i in range(store.n_blocks):
        start, stop = store.block_range(i)
        yield ReadyBlock(
            index=i,
            col_range=(start, stop),
            n_valid=stop - start,
            operands={g.id: g.blocks[i] for g in groups},
            ready=None,
        )


def _pinned_blocks(store: FeatureStore, groups: Sequence[ColumnGroup]) -> Iterator[ReadyBlock]:
    """Pinned-host residency: blocks stream through ``store.n_slots`` device slots.

    Block ``i`` is copied (on the store's copy stream) into slot ``i % n_slots`` as
    soon as that slot's previous block has been released by the consumer; copies run
    one block ahead of compute.
    """
    n_slots, n_blocks = store.n_slots, store.n_blocks
    slots = {
        g.id: [torch.empty_like(g.blocks[0], device=store.device) for _ in range(n_slots)]
        for g in groups
    }
    free: list[torch.cuda.Event | None] = [None] * n_slots
    stream = store.copy_stream or torch.cuda.current_stream(store.device)

    def issue(i: int) -> torch.cuda.Event:
        slot = i % n_slots
        with torch.cuda.stream(stream):
            waiting_on = free[slot]
            if waiting_on is not None:
                stream.wait_event(waiting_on)
            for g in groups:
                slots[g.id][slot].copy_(g.blocks[i], non_blocking=True)
            ready = torch.cuda.Event()  # type: ignore[no-untyped-call]
            ready.record(stream)
        return ready

    pending = {i: issue(i) for i in range(min(n_slots - 1, n_blocks))}
    for i in range(n_blocks):
        start, stop = store.block_range(i)
        slot = i % n_slots

        def release(done: torch.cuda.Event | None, slot: int = slot) -> None:
            if done is None:
                done = torch.cuda.Event()  # type: ignore[no-untyped-call]
                done.record(torch.cuda.current_stream(store.device))
            free[slot] = done

        yield ReadyBlock(
            index=i,
            col_range=(start, stop),
            n_valid=stop - start,
            operands={g.id: slots[g.id][slot] for g in groups},
            ready=pending.pop(i),
            _release=release,
        )
        nxt = i + n_slots - 1
        if nxt < n_blocks:
            pending[nxt] = issue(nxt)


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
            block = self.store.write_target(self.group, b)
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
            self.store.note_written(self.group, b, n_cells * (b1 - b0) * planes_written)
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
    residency : {"device", "pinned_host"}
        ``"device"`` keeps blocks on the compute device; ``"pinned_host"`` keeps
        them in pinned host memory and streams them through device slots.
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

    def blocks(self, groups: Sequence[ColumnGroup]) -> Iterator[ReadyBlock]:
        """Iterate the blocks of ``groups`` in order, each ready on the device."""
        for g in groups:
            if g not in self.groups:
                raise ValueError("group does not belong to this store")
            if not g.complete:
                raise ValueError(f"group {g.id} is not complete")
        if self.residency == "pinned_host":
            return _pinned_blocks(self, list(groups))
        return _direct_blocks(self, list(groups))

    # -- pinned-host staging (write side) ----------------------------------------

    def write_target(self, group: ColumnGroup, b: int) -> torch.Tensor:
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

    def note_written(self, group: ColumnGroup, b: int, cell_columns: int) -> None:
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
        self,
        keep: torch.Tensor,
        *,
        block_pixels: int | None = None,
        residency: Residency | None = None,
    ) -> FeatureStore:
        """Keep the compact columns where ``keep`` is True (order preserved).

        Works block by block, so the transient peak is the new store plus one old
        block's gathered columns. ``residency`` defaults to this store's; a subset small
        enough for the device is usually moved there.
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
            residency=residency or self.residency,
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
