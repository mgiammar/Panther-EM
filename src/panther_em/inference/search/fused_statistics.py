"""Fused-CUDA-kernel-backed :class:`PixelStats` specialization."""

from __future__ import annotations

from typing import Literal, overload

import torch

from panther_em.inference.search import fused_kernel_loader
from panther_em.inference.search.statistics import PixelStats, decode_argmax_packed


class FusedPixelStats(PixelStats):
    """:class:`PixelStats` that reduces via the fused CUDA kernels when possible.

    Two fast paths, tried in order:

    1. **In-kernel accumulation** (:meth:`update_graphed`): for the register-resident
       kernel's domain (``n_psi`` in ``{128, 256}``, ``NumFreq <= 64``, spectrum in
       the GEMM's native ``(NumFreq, P, N)`` layout, complex64 or complex32) the
       kernel adds the batch's moments straight into :attr:`corr_sum` /
       :attr:`corr_sum2` and merges a packed ``(value, global (hyp, psi) index)``
       maximum into :attr:`best_packed` -- the whole hypothesis loop is then two or
       three kernel launches per batch and no per-batch accumulate step. The packed
       best is decoded into ``best_corr`` / ``best_hypothesis`` / ``best_psi_angle``
       once, in :meth:`finalize`.
    2. **Fused reduce** (:meth:`_reduce` / :meth:`_reduce_raw`): the lean kernel or,
       failing that, the cuFFTDx block-FFT kernel, feeding the parent's accumulate.
    """

    best_packed: torch.Tensor

    def __init__(
        self,
        num_pixels: int,
        device: torch.device,
        accumulate_dtype: torch.dtype = torch.float32,
    ):
        super().__init__(num_pixels, device, accumulate_dtype)
        self._lean_sentinel = fused_kernel_loader.lean_sentinel_packed()
        self.best_packed = torch.full(
            (num_pixels,),
            self._lean_sentinel if self._lean_sentinel is not None else 0,
            device=device,
            dtype=torch.int64,
        )
        # num_psi the packed best was accumulated with, or None if untouched since
        # the last clear()/finalize().
        self._packed_num_psi: int | None = None

    @torch.no_grad()
    def clear(self) -> None:
        super().clear()
        if self._lean_sentinel is not None:
            self.best_packed.fill_(self._lean_sentinel)
        self._packed_num_psi = None

    @staticmethod
    def inkernel_supported(
        spectrum: torch.Tensor, num_psi: int, hyp_offset: int
    ) -> bool:
        """Whether :meth:`update_graphed` can accumulate ``spectrum`` in-kernel."""
        if not (
            spectrum.is_cuda
            and spectrum.dtype in (torch.complex64, torch.complex32)
            and spectrum.dim() == 3
        ):
            return False
        transposed = spectrum.permute(2, 0, 1)
        return bool(
            transposed.is_contiguous()
            and fused_kernel_loader.lean_supported(num_psi, transposed.shape[0])
            and (hyp_offset + transposed.shape[2]) * num_psi < 2**32
        )

    @torch.no_grad()
    def update_graphed(
        self,
        spectrum: torch.Tensor,
        hyp_offset: int,
        num_psi: int,
        reverse_psi_axis: bool = True,
    ) -> None:
        """Accumulate one contiguous-arange hypothesis batch, in-kernel when possible.

        Falls back to the parent's fused-reduce + graphed-accumulate path whenever
        the in-kernel path does not apply (see :meth:`inkernel_supported`, or
        ``reverse_psi_axis=False``, or a non-float32 accumulate dtype).
        """
        if (
            reverse_psi_axis
            and self._lean_sentinel is not None
            and self.corr_sum.dtype == torch.float32
            and self.inkernel_supported(spectrum, num_psi, hyp_offset)
        ):
            result = fused_kernel_loader.lean_irfft_stats_transposed(
                spectrum.permute(2, 0, 1),
                num_psi,
                decode=False,
                hyp_offset=hyp_offset,
                outs=[self.corr_sum, self.corr_sum2, self.best_packed],
            )
            if result is not None:
                self.hypothesis_count += spectrum.shape[1] * num_psi
                self._packed_num_psi = num_psi
                return None
        return super().update_graphed(spectrum, hyp_offset, num_psi, reverse_psi_axis)

    @torch.no_grad()
    def merge_packed(self) -> None:
        """Fold the in-kernel packed best into ``best_corr`` / ``_hypothesis`` / ``_psi``.

        Idempotent; a no-op when the in-kernel path was not used since the last
        :meth:`clear`. Uses ``reverse_psi_axis=True`` decoding, the only mode routed
        through the in-kernel path.
        """
        if self._packed_num_psi is None:
            return
        num_psi = self._packed_num_psi
        vmax, flat = decode_argmax_packed(self.best_packed)
        global_hyp = torch.div(flat, num_psi, rounding_mode="floor")
        psi = (num_psi - flat % num_psi) % num_psi
        vmax_cast = vmax.to(self.best_corr.dtype)
        improved = vmax_cast > self.best_corr
        # In place (`out=`), never rebinding the attributes: a CUDA graph captured by
        # compressed._HypLoopGraph holds these exact buffers, and its captured clear()
        # must keep resetting the tensors finalize() reads.
        torch.where(improved, vmax_cast, self.best_corr, out=self.best_corr)
        torch.where(improved, global_hyp, self.best_hypothesis, out=self.best_hypothesis)
        torch.where(improved, psi, self.best_psi_angle, out=self.best_psi_angle)
        self.best_packed.fill_(self._lean_sentinel)  # type: ignore[arg-type]
        self._packed_num_psi = None

    @torch.no_grad()
    def finalize(self) -> dict[str, torch.Tensor]:
        self.merge_packed()
        return super().finalize()

    @overload
    def _try_fused_reduce(
        self, spectrum: torch.Tensor, num_psi: int, decode: Literal[True]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None: ...
    @overload
    def _try_fused_reduce(
        self, spectrum: torch.Tensor, num_psi: int, decode: Literal[False]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None: ...
    def _try_fused_reduce(
        self, spectrum: torch.Tensor, num_psi: int, decode: bool
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None
    ):
        """Shared fast-path selection for :meth:`_reduce` and :meth:`_reduce_raw`.

        Returns ``None`` (rather than raising) whenever the fused kernel can't handle
        this call -- spectrum isn't CUDA complex64, the kernel failed to compile, or
        ``(num_psi, NumFreq)`` isn't a supported config -- signaling the caller to fall
        back to the parent's pure-torch reduction.
        """
        if not (
            spectrum.is_cuda
            and spectrum.dtype in (torch.complex64, torch.complex32)
        ):
            return None

        # Zero-copy fast path for internal transpose.
        # Triggers when `spectrum` is (P, N, NumFreq) view of a (NumFreq, P, N)
        # contiguous tensor (layout torch.bmm produces natively, in either precision).
        # `decode` is branched on explicitly (rather than forwarded as a plain bool)
        # so each call below keeps its `Literal[True]`/`Literal[False]` overload.
        transposed = spectrum.permute(2, 0, 1)
        if transposed.is_contiguous():
            # 1. Register-resident kernel: n_psi in {128, 256}, NumFreq <= 64, both
            #    complex dtypes. ~6x faster than the cuFFTDx kernel at n_psi=256.
            result = (
                fused_kernel_loader.lean_irfft_stats_transposed(
                    transposed, num_psi, decode=True
                )
                if decode
                else fused_kernel_loader.lean_irfft_stats_transposed(
                    transposed, num_psi, decode=False
                )
            )
            if result is not None:
                return result
            # 2. cuFFTDx block-FFT kernel (complex64 only).
            if spectrum.dtype == torch.complex64:
                result = (
                    fused_kernel_loader.fused_irfft_stats_transposed(
                        transposed, num_psi, decode=True
                    )
                    if decode
                    else fused_kernel_loader.fused_irfft_stats_transposed(
                        transposed, num_psi, decode=False
                    )
                )
                if result is not None:
                    return result

        # 3. Non-native layout: cuFFTDx kernel with an explicit .contiguous() copy.
        if spectrum.dtype != torch.complex64:
            spectrum = spectrum.to(torch.complex64)
        if decode:
            return fused_kernel_loader.fused_irfft_stats(spectrum, num_psi, decode=True)
        return fused_kernel_loader.fused_irfft_stats(spectrum, num_psi, decode=False)

    def _reduce(
        self, spectrum: torch.Tensor, num_psi: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        result = self._try_fused_reduce(spectrum, num_psi, decode=True)
        if result is not None:
            return result
        return super()._reduce(spectrum, num_psi)  # Parent torch fallback

    def _reduce_raw(
        self, spectrum: torch.Tensor, num_psi: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same fast-path selection as :meth:`_reduce`, but skips the argmax decode."""
        result = self._try_fused_reduce(spectrum, num_psi, decode=False)
        if result is not None:
            return result
        return super()._reduce_raw(spectrum, num_psi)  # Parent torch fallback
