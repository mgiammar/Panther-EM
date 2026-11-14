"""Classes and associated methods for FFT-based featurization across large images.

The rectangle's Cartesian kernels are built once, on first use, and held by the
featurization (a searcher keeps one per rectangle across images). A strip runs on zipFFT
(:mod:`.zipfft_backend`) when its FFT shape is compiled for the box -- the kernel FFT,
product and inverse FFT fused, a chunk of cells split into calls of the compiled
batches -- and on cuFFT through torch otherwise, the kernels' spectra at the strip's
shape computed per chunk of cells.
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
from panther_em.search.featurization.zipfft_backend import (
    DEFAULT_ZIPFFT_BATCH,
    FFT_BACKENDS,
    correlate,
    split_batch,
    zipfft_shapes,
    zipfft_unavailable,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from panther_em.approximation.reconstructor import Reconstructor
    from panther_em.search.featurization.zipfft_backend import FFTBackend

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
        Cells per chunk (on cuFFT each holds a ``8 S FX`` byte spectrum, 134 MB at
        4096^2).
    strip_heights, strip_widths : sequence of int, optional
        FFT heights / widths to run, overriding the backend's (``None``: the zipFFT
        shapes compiled for the box, or any size on cuFFT).
    backend : {"auto", "torch", "zipfft"}, optional
        FFT backend (see :data:`.zipfft_backend.FFTBackend`).
    max_batch : int, optional
        Largest zipFFT call batch a chunk of cells is split into.
    """

    def __init__(
        self,
        reconstructor: Reconstructor,
        rect: Sequence[int],
        *,
        cell_chunk: int = DEFAULT_CELL_CHUNK,
        strip_heights: Sequence[int] | None = None,
        strip_widths: Sequence[int] | None = None,
        backend: FFTBackend = "auto",
        max_batch: int = DEFAULT_ZIPFFT_BATCH,
    ) -> None:
        super().__init__(reconstructor, rect)

        if int(cell_chunk) < 1:
            raise ValueError("cell_chunk must be >= 1")
        if backend not in FFT_BACKENDS:
            raise ValueError(f"unknown FFT backend {backend!r}")
        if int(max_batch) < 1:
            raise ValueError("max_batch must be >= 1")

        self.cell_chunk = int(cell_chunk)
        self.backend: FFTBackend = backend
        self.max_batch = int(max_batch)
        self._specs: dict[int, StripSpec] = {}
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
    def zipfft_shapes(self) -> dict[tuple[int, int], tuple[int, ...]]:
        """``{(S, FX): batches}`` zipFFT runs for the box here (``{}`` on torch)."""
        if self.backend == "torch":
            return {}
        return zipfft_shapes(self.box_shape, self.device)

    def strip_spec(self, width: int) -> StripSpec:
        """The FFT sizes that cover a region ``width`` box-origin columns wide."""
        spec = self._specs.get(width)
        if spec is not None:
            return spec

        box = self.box_shape
        if self.strip_heights is not None or self.strip_widths is not None:
            spec = StripSpec(box, self.strip_heights, self.strip_widths)
        else:
            shapes = self.zipfft_shapes
            need = width + box[1] - 1
            fft_w = min((w for _, w in shapes if w >= need), default=None)
            heights = sorted(h for h, w in shapes if w == fft_w)
            spec = (
                StripSpec(box, tuple(heights), (fft_w,))
                if fft_w is not None
                else StripSpec(box)
            )

        self._specs[width] = spec
        return spec

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
        spec = self.strip_spec(out_w)
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
        batches = self.zipfft_shapes.get(fft_shape)
        if batches is None and self.backend == "zipfft":
            reason = zipfft_unavailable(device) or (
                f"zipFFT has no correlation of a {self.box_shape} box at FFT shape "
                f"{fft_shape} (compiled: {sorted(self.zipfft_shapes)})"
            )
            raise ValueError(f"fft_backend 'zipfft': {reason}")

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
        index = None if whole else local
        if batches is None:
            run = self._torch_chunks(image_ffts, strip.rows, out_w, fft_shape, index)
        else:  # zipFFT reads the spectra transposed, (FX, S)
            image_ffts = [f.transpose(-2, -1).contiguous() for f in image_ffts]
            run = self._zipfft_chunks(
                image_ffts, strip.rows, out_w, fft_shape, index, batches
            )
        for d, cells, values in run:
            yield FeatureChunk(d, cells, columns, values)

    def _torch_chunks(
        self,
        image_ffts: list[torch.Tensor],
        rows: int,
        out_w: int,
        fft_shape: tuple[int, int],
        index: torch.Tensor | None,
    ) -> Iterator[tuple[int, slice, torch.Tensor]]:
        """``(plane, cells, values)`` of a strip on cuFFT (``index``: gather)."""
        n_positions = rows * out_w if index is None else int(index.numel())
        for start in range(0, self.n_cells, self.cell_chunk):
            cells = slice(start, min(start + self.cell_chunk, self.n_cells))
            spectra = torch.fft.fft2(self.kernels[cells], s=fft_shape).conj()
            for d, image_fft in enumerate(image_ffts):
                values = torch.empty(
                    (len(spectra), n_positions),
                    dtype=torch.complex64,
                    device=self.device,
                )
                for i, spectrum in enumerate(spectra):
                    correlation = torch.fft.ifft2(image_fft * spectrum)
                    valid = correlation[:rows, :out_w].reshape(-1)
                    values[i] = valid if index is None else valid[index]
                yield d, cells, values

    def _zipfft_chunks(
        self,
        spectra: list[torch.Tensor],
        rows: int,
        out_w: int,
        fft_shape: tuple[int, int],
        index: torch.Tensor | None,
        batches: tuple[int, ...],
    ) -> Iterator[tuple[int, slice, torch.Tensor]]:
        """``(plane, cells, values)`` of a strip on zipFFT (``index``: gather).

        ``spectra`` are the planes' ``(FX, S)`` transposed spectra. A strip that keeps
        every valid row and column of the FFT is written straight into the chunk's
        values; otherwise through an output buffer.
        """
        device = self.device
        k_h, k_w = self.box_shape
        valid = (fft_shape[0] - k_h + 1, fft_shape[1] - k_w + 1)
        n_positions = rows * out_w if index is None else int(index.numel())
        direct = index is None and (rows, out_w) == valid
        largest = split_batch(
            min(self.cell_chunk, self.n_cells), batches, self.max_batch
        )[0]
        workspace = torch.empty(
            (largest, *fft_shape), dtype=torch.complex64, device=device
        )
        output = (
            None
            if direct
            else torch.empty((largest, *valid), dtype=torch.complex64, device=device)
        )
        for start in range(0, self.n_cells, self.cell_chunk):
            cells = slice(start, min(start + self.cell_chunk, self.n_cells))
            kernels = self.kernels[cells].contiguous()
            calls = split_batch(len(kernels), batches, self.max_batch)
            for d, spectrum in enumerate(spectra):
                values = torch.empty(
                    (len(kernels), n_positions), dtype=torch.complex64, device=device
                )
                i = 0
                for b in calls:
                    out = (
                        values[i : i + b].view(b, *valid)
                        if output is None
                        else output[:b]
                    )
                    correlate(
                        kernels[i : i + b], workspace[:b], spectrum, out, fft_shape
                    )
                    if output is not None:
                        kept = out[:, :rows, :out_w].reshape(b, -1)
                        values[i : i + b] = kept if index is None else kept[:, index]
                    i += b
                yield d, cells, values
