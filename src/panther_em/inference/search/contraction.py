"""Contraction engines dealing with floating point precision and feature-block layout.

Layouts (``D`` defocus planes, ``P`` pixels in a block):

* :class:`ComplexGemmEngine` -- block ``(D, num_k, P, num_m)`` complex64; weights
  ``(num_k, num_m, N)`` complex64 (pre-conjugated); spectrum complex64.
* :class:`Fp16GemmEngine` -- block ``(D, num_k, P, 2 num_m)`` float16 as
  ``[Re Y | Im Y]``; weights ``(num_k, 2 num_m, 2 N)`` float16 interleaved so the real
  GEMM output is bit-identical to a complex ``(num_k, P, N)`` tensor; spectrum
  complex32.

The spectrum is always handed out in the kernels' native ``(F, P, N)`` contiguous layout
(:class:`Spectrum`); ``Spectrum.pnf()`` is the ``(P, N, F)`` view the torch fallback
consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from panther_em.inference.search.types import Precision

if TYPE_CHECKING:
    from collections.abc import Sequence

    from panther_em.inference.search.tiling import KRun


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


@runtime_checkable
class ContractionEngine(Protocol):
    """One precision's GEMM, plus the feature-block layout it consumes."""

    precision: Precision
    spectrum_dtype: torch.dtype
    block_dtype: torch.dtype

    def block_shape(
        self, n_defocus: int, num_k: int, num_m: int, n_pixels: int
    ) -> tuple[int, ...]:
        """Shape of a feature block for one rectangle of ``num_k x num_m`` cells."""

    def block_num_m(self, block: torch.Tensor) -> int:
        """Recover ``num_m`` from a block's shape."""

    def write_block(
        self, block: torch.Tensor, cells: slice, cols: slice, data: torch.Tensor
    ) -> None:
        """Write ``data`` ``(D, n_cells, n_cols)`` complex features into ``block``."""

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

    def prepare_weights(self, w: torch.Tensor) -> torch.Tensor:
        """Un-conjugated ``(N, num_k, num_m)`` complex64 weights -> B operand."""

    def contract(
        self,
        a_parts: Sequence[torch.Tensor],
        b_parts: Sequence[torch.Tensor],
        run: KRun,
        out: Spectrum,
    ) -> None:
        """``out.native[run.k_start:run.k_stop] = A @ B`` (parts concatenated)."""


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
        """Shape of a feature block for one rectangle (see the protocol)."""
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
        """Shape of a feature block for one rectangle (see the protocol)."""
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


def make_engine(precision: Precision | str) -> ContractionEngine:
    """The engine for a precision."""
    precision = Precision.coerce(precision)
    if precision is Precision.FP16:
        return Fp16GemmEngine()
    return ComplexGemmEngine(precision)


__all__ = [
    "ComplexGemmEngine",
    "ContractionEngine",
    "Fp16GemmEngine",
    "Spectrum",
    "make_engine",
]
