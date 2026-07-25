"""Fused-CUDA-kernel-backed :class:`PixelStats` specialization."""

from __future__ import annotations

import torch

from panther_em.inference.search import fused_kernel_loader
from panther_em.inference.search.statistics import PixelStats


class FusedPixelStats(PixelStats):
    """:class:`PixelStats` that reduces via the fused CUDA kernel when possible."""

    def _reduce(
        self, spectrum: torch.Tensor, num_psi: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if spectrum.is_cuda and spectrum.dtype == torch.complex64:
            # Zero-copy fast path for internal transpose.
            # Triggers when `spectrum` is (P, N, NumFreq) view of a (NumFreq, P, N)
            # contiguous tensor (layout torch.bmm produces natively).
            transposed = spectrum.permute(2, 0, 1)
            if transposed.is_contiguous():
                result = fused_kernel_loader.fused_irfft_stats_transposed(
                    transposed, num_psi
                )
                if result is not None:
                    return result

            result = fused_kernel_loader.fused_irfft_stats(spectrum, num_psi)
            if result is not None:
                return result

        return super()._reduce(spectrum, num_psi)  # Parent torch fallback

    def _reduce_raw(
        self, spectrum: torch.Tensor, num_psi: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Same fast-path selection as :meth:`_reduce`, but skips the argmax decode."""
        if spectrum.is_cuda and spectrum.dtype == torch.complex64:
            transposed = spectrum.permute(2, 0, 1)
            if transposed.is_contiguous():
                result = fused_kernel_loader.fused_irfft_stats_transposed(
                    transposed, num_psi, decode=False
                )
                if result is not None:
                    return result

            result = fused_kernel_loader.fused_irfft_stats(
                spectrum, num_psi, decode=False
            )
            if result is not None:
                return result

        return super()._reduce_raw(spectrum, num_psi)  # Parent torch fallback
