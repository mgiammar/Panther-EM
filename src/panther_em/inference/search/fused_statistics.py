"""Fused-CUDA-kernel-backed :class:`PixelStats` specialization."""

from __future__ import annotations

from typing import Literal, overload

import torch

from panther_em.inference.search import fused_kernel_loader
from panther_em.inference.search.statistics import PixelStats


class FusedPixelStats(PixelStats):
    """:class:`PixelStats` that reduces via the fused CUDA kernel when possible."""

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
        if not (spectrum.is_cuda and spectrum.dtype == torch.complex64):
            return None

        # Zero-copy fast path for internal transpose.
        # Triggers when `spectrum` is (P, N, NumFreq) view of a (NumFreq, P, N)
        # contiguous tensor (layout torch.bmm produces natively).
        # `decode` is branched on explicitly (rather than forwarded as a plain bool)
        # so each call below keeps its `Literal[True]`/`Literal[False]` overload.
        transposed = spectrum.permute(2, 0, 1)
        if transposed.is_contiguous():
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
