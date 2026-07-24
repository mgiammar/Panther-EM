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
            result = fused_kernel_loader.fused_irfft_stats(spectrum, num_psi)

            if result is not None:
                return result

        return super()._reduce(spectrum, num_psi)  # Parent torch fallback
