"""Feature space layouts and the contraction engines.

* :class:`FeatureLayout` -- pairwise cell-disjoint rectangles of ``(angular frequency,
  eigenvector)`` space and their *k-runs*: maximal runs of ``k`` covered by a constant
  rectangle set, each contracted as one batched GEMM.
* :class:`ComplexGemmEngine` / :class:`Fp16GemmEngine` -- one precision's GEMM plus the
  block layout it consumes (``(D, num_k, P, num_m)`` complex64, or
  ``(D, num_k, P, 2 num_m)`` float16 as ``[Re Y | Im Y]`` with interleaved weights so
  the real GEMM output is bit-identical to a complex spectrum). The spectrum is always
  the kernels' native contiguous ``(F, P, N)`` (:class:`Spectrum`).

The store the engines' blocks live in is
:class:`~panther_em.inference.search.featurization.store.FeatureStore`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

import torch

from panther_em.inference.search.types import Precision

if TYPE_CHECKING:
    from collections.abc import Sequence

    from panther_em.inference.search.types import Rectangle

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
    :attr:`FeatureLayout.rectangles` and its global eigenvector range.
    """

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
