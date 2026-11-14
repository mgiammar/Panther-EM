"""Classes and associated methods for FFT-based featurization across large images.

The rectangle's Cartesian kernels are built once, on first use, and held by the
featurization (a searcher keeps one per rectangle across images); their spectra at a
strip's FFT shape are computed per chunk of cells on every call.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import torch

from panther_em.search.featurization.base import (
    BaseImageFeaturization,
    FeatureChunk,
)
from panther_em.search.featurization.strips import Strip, StripSpec

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
        Kernel spectra per chunk (each is ``8 S Fw`` bytes, 134 MB at 4096^2).
    strip_heights, strip_widths : sequence of int, optional
        FFT heights / widths to run, or ``None`` for any size.
    """

    def __init__(
        self,
        reconstructor: Reconstructor,
        rect: Sequence[int],
        *,
        cell_chunk: int = DEFAULT_CELL_CHUNK,
        strip_heights: Sequence[int] | None = None,
        strip_widths: Sequence[int] | None = None,
    ) -> None:
        super().__init__(reconstructor, rect)

        if int(cell_chunk) < 1:
            raise ValueError("cell_chunk must be >= 1")

        self.cell_chunk = int(cell_chunk)
        self.strip_heights = (
            None if strip_heights is None else tuple(int(h) for h in strip_heights)
        )
        self.strip_widths = (
            None if strip_widths is None else tuple(int(w) for w in strip_widths)
        )

    @cached_property
    def kernels(self) -> torch.Tensor:
        """``(n_cells, kH, kW)`` complex64 kernels of the rectangle (built once)."""
        return self.reconstructor.kernels(self.cells)

    @cached_property
    def strip_spec(self) -> StripSpec:
        """The FFT sizes this featurization runs (see :mod:`.strips`)."""
        return StripSpec(self.box_shape, self.strip_heights, self.strip_widths)

    @staticmethod
    def region(
        rows: torch.Tensor, cols: torch.Tensor
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """``(origin, output shape)`` of the positions' bounding box.

        Parameters
        ----------
        rows : torch.Tensor
            ``(N,)`` int64 row positions.
        cols : torch.Tensor
            ``(N,)`` int64 column positions.
        """
        r0, c0 = int(rows.min()), int(cols.min())
        return (r0, c0), (int(rows.max()) - r0 + 1, int(cols.max()) - c0 + 1)

    def strips(
        self, rows: torch.Tensor, cols: torch.Tensor
    ) -> tuple[tuple[int, int], int, int, tuple[tuple[Strip, slice], ...]]:
        """``(origin, out_w, fft_width, ((strip, columns), ...))`` of the positions."""
        (r0, c0), (out_h, out_w) = self.region(rows, cols)
        spec = self.strip_spec
        fft_w = spec.fft_width(out_w)

        if fft_w is None:
            raise ValueError(
                f"no FFT width of {spec.widths} holds {out_w} box columns of a "
                f"{self.box_shape} box"
            )

        strips = spec.strips(out_h)
        n = int(rows.numel())

        if len(strips) == 1:
            return (r0, c0), out_w, fft_w, ((strips[0], slice(0, n)),)

        if not bool(torch.all(rows[1:] >= rows[:-1])):
            raise ValueError(
                "positions spanning several strips must be sorted by row "
                f"({out_h} rows over FFT heights {spec.heights})"
            )

        starts = torch.tensor(
            [r0 + s.row for s in strips] + [r0 + out_h], device=rows.device
        )
        bounds = torch.searchsorted(rows, starts).tolist()

        return (
            (r0, c0),
            out_w,
            fft_w,
            tuple((s, slice(bounds[i], bounds[i + 1])) for i, s in enumerate(strips)),
        )

    def chunks(
        self, planes: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor
    ) -> Iterator[FeatureChunk]:
        """See :meth:`~.base.BaseImageFeaturization.chunks`."""
        self.validate(planes, rows, cols)

        if int(rows.numel()) == 0:
            return

        (r0, c0), out_w, fft_w, strips = self.strips(rows, cols)
        for strip, columns in strips:
            if columns.stop > columns.start:
                yield from self._strip_chunks(
                    planes,
                    rows[columns] - (r0 + strip.row),
                    cols[columns] - c0,
                    strip,
                    (r0 + strip.row, c0),
                    (out_w, fft_w),
                    columns,
                )

    def _strip_chunks(
        self,
        planes: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        strip: Strip,
        origin: tuple[int, int],
        widths: tuple[int, int],
        columns: slice,
    ) -> Iterator[FeatureChunk]:
        """One strip's chunks; ``rows``/``cols`` relative to the strip's origin."""
        device = self.device
        n_positions = int(rows.numel())
        (r0, c0), (out_w, fft_w) = origin, widths
        k_h, k_w = self.box_shape
        fft_shape = (strip.height, fft_w)
        region = (
            slice(r0, r0 + strip.rows + k_h - 1),
            slice(c0, c0 + out_w + k_w - 1),
        )
        image_ffts = [
            torch.fft.fft2(
                plane[region].to(device=device, dtype=torch.float32), s=fft_shape
            )
            for plane in planes
        ]

        local = (rows * out_w + cols).to(device)
        whole = n_positions == strip.rows * out_w and bool(
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
                    valid = correlation[: strip.rows, :out_w].reshape(-1)
                    values[i] = valid if whole else valid[local]
                yield FeatureChunk(d, cells, columns, values)
