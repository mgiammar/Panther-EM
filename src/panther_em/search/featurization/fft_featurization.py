"""Classes and associated methods for FFT-based featurization across large images.

The rectangle's Cartesian kernels are built once, on first use, and held by the
featurization (a searcher keeps one per rectangle across images); their spectra at a
region's FFT shape are computed per chunk of cells on every call.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import torch
from scipy.fft import next_fast_len

from panther_em.search.featurization.base import (
    BaseImageFeaturization,
    FeatureChunk,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from panther_em.approximation.reconstructor import Reconstructor

DEFAULT_CELL_CHUNK = 32
"""Kernel spectra computed (and held) per chunk."""


class FFTFeaturization(BaseImageFeaturization):
    """Overlap-save FFT correlation over the bounding box of the positions.

    Parameters
    ----------
    reconstructor : Reconstructor
        The template's forward model (kernels and their adjoint).
    rect : Rectangle
        ``(k_start, m_start, k_extent, m_extent)``.
    cell_chunk : int, optional
        Kernel spectra per chunk (each is ``8 Fh Fw`` bytes, 134 MB at 4096^2).
    """

    def __init__(
        self,
        reconstructor: Reconstructor,
        rect: Sequence[int],
        *,
        cell_chunk: int = DEFAULT_CELL_CHUNK,
    ) -> None:
        super().__init__(reconstructor, rect)

        if int(cell_chunk) < 1:
            raise ValueError("cell_chunk must be >= 1")

        self.cell_chunk = int(cell_chunk)

    @cached_property
    def kernels(self) -> torch.Tensor:
        """``(n_cells, kH, kW)`` complex64 kernels of the rectangle (built once)."""
        return self.reconstructor.kernels(self.cells)

    def region(
        self, rows: torch.Tensor, cols: torch.Tensor
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        """``(origin, output shape, FFT shape)`` of the correlated region.

        Parameters
        ----------
        rows : torch.Tensor
            ``(N,)`` int64 row positions.
        cols : torch.Tensor
            ``(N,)`` int64 column positions.
        """
        r0, c0 = int(rows.min()), int(cols.min())
        out = (int(rows.max()) - r0 + 1, int(cols.max()) - c0 + 1)
        k_h, k_w = self.box_shape
        fft_shape = (
            int(next_fast_len(out[0] + k_h - 1)),
            int(next_fast_len(out[1] + k_w - 1)),
        )
        return (r0, c0), out, fft_shape

    def chunks(
        self, planes: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor
    ) -> Iterator[FeatureChunk]:
        """See :meth:`~.base.BaseImageFeaturization.chunks`."""
        self.validate(planes, rows, cols)

        n_positions = int(rows.numel())

        if n_positions == 0:
            return

        device = self.device
        (r0, c0), (out_h, out_w), fft_shape = self.region(rows, cols)
        k_h, k_w = self.box_shape
        region = (slice(r0, r0 + out_h + k_h - 1), slice(c0, c0 + out_w + k_w - 1))
        image_ffts = [
            torch.fft.fft2(
                plane[region].to(device=device, dtype=torch.float32), s=fft_shape
            )
            for plane in planes
        ]

        local = ((rows - r0) * out_w + (cols - c0)).to(device)
        whole = n_positions == out_h * out_w and bool(
            torch.equal(local, torch.arange(n_positions, device=device))
        )
        for start in range(0, self.n_cells, self.cell_chunk):
            cells = slice(start, min(start + self.cell_chunk, self.n_cells))
            spectra = torch.fft.fft2(self.kernels[cells], s=fft_shape).conj()
            for d, image_fft in enumerate(image_ffts):
                values = torch.empty(
                    (len(spectra), n_positions), dtype=torch.complex64, device=device
                )
                for i, spectrum in enumerate(spectra):
                    correlation = torch.fft.ifft2(image_fft * spectrum)
                    valid = correlation[:out_h, :out_w].reshape(-1)
                    values[i] = valid if whole else valid[local]
                yield FeatureChunk(d, cells, slice(0, n_positions), values)
