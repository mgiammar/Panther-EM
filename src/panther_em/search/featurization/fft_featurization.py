"""Classes and associated methods for FFT-based featurization across large images.

The rectangle's Cartesian kernels are built once and held by the featurization (a
searcher keeps one per rectangle across images and builds them in :meth:`prepare`,
before it sizes a stage's store) when they take at most :data:`KERNEL_FRACTION` of the
free device memory; otherwise each chunk builds its own. Their spectra at a region's
FFT shape are computed per chunk of cells on every call.

Every call sizes its chunks to :data:`WORKSPACE_FRACTION` of the device memory left
free (by the store and the held kernels): :func:`working_set_bytes` models a chunk's
peak, and a chunk holds as many cells as fit, at most ``cell_chunk``. When not even one
cell fits the positions' bounding box, the positions are split into consecutive runs
(bands of rows, for the searcher's sorted selections), each correlated as its own
smaller region (:meth:`FFTFeaturization.tiles`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from scipy.fft import next_fast_len

from panther_em.search.featurization.base import (
    BaseImageFeaturization,
    FeatureChunk,
    free_device_bytes,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from panther_em.approximation.reconstructor import Reconstructor

DEFAULT_CELL_CHUNK = 32
"""Most kernel spectra computed (and held) per chunk; fewer when memory is short."""

KERNEL_FRACTION = 0.5
"""Share of the free device memory the rectangle's held kernels may take."""

WORKSPACE_FRACTION = 0.8
"""Share of the free device memory a call's chunks are sized to (the rest is headroom
for the caching allocator's fragmentation)."""

COMPLEX64_BYTES = 8

BUILD_BOXES_PER_CELL = 6
"""Kernel-box arrays per kernel a chunk builds (the kernel and its build's work)."""

BUILD_BOXES = 20
"""Kernel-box arrays of a chunk's build that do not grow with its cells."""


def working_set_bytes(
    n_cells: int,
    fft_size: int,
    n_positions: int,
    n_planes: int,
    *,
    kernel_size: int = 0,
) -> int:
    """Peak device bytes of correlating one region, ``n_cells`` kernel spectra a chunk.

    Per cell of a chunk: its zero-padded kernel, its spectrum and cuFFT's work area
    (four FFT-size arrays: the work area of a batched 2-D transform measured one to
    two times its data), and three copies of its features (the chunk being filled,
    the previous chunk the caller may still hold, and the caller's conversion on
    write). When the chunk builds its kernels, also the kernels and the build's work
    arrays (:data:`BUILD_BOXES_PER_CELL` boxes per cell, :data:`BUILD_BOXES` per
    build). Per region: the planes' spectra, two FFT-size arrays (with one cell a
    chunk, a correlation's product, inverse transform and work area outlast the
    spectrum's transform) and the positions' indices into the region.

    Parameters
    ----------
    n_cells : int
        Cells per chunk.
    fft_size : int
        Elements of the region's FFT shape.
    n_positions : int
        Positions in the region.
    n_planes : int
        Planes featurized together.
    kernel_size : int, optional
        Elements of a kernel box when each chunk builds its kernels; ``0`` (default)
        when they are held.
    """
    per_cell = 4 * fft_size + 3 * n_positions + BUILD_BOXES_PER_CELL * kernel_size
    per_region = (n_planes + 2) * fft_size + n_positions + BUILD_BOXES * kernel_size
    return COMPLEX64_BYTES * (n_cells * per_cell + per_region)


def fitting_cells(
    budget: int,
    fft_size: int,
    n_positions: int,
    n_planes: int,
    *,
    kernel_size: int = 0,
    limit: int = DEFAULT_CELL_CHUNK,
) -> int:
    """The most cells per chunk, at most ``limit``, whose working set fits ``budget``.

    ``0`` when not even one cell fits (see :func:`working_set_bytes` for the rest).
    """
    region = working_set_bytes(
        0, fft_size, n_positions, n_planes, kernel_size=kernel_size
    )
    per_cell = (
        working_set_bytes(1, fft_size, n_positions, n_planes, kernel_size=kernel_size)
        - region
    )
    return int(max(0, min(int(limit), (int(budget) - region) // per_cell)))


@dataclass(frozen=True)
class Tile:
    """Consecutive positions correlated as one region, and its cells per chunk."""

    columns: slice
    cell_chunk: int


class FFTFeaturization(BaseImageFeaturization):
    """Overlap-save FFT correlation over the bounding box of the positions.

    Parameters
    ----------
    reconstructor : Reconstructor
        The template's forward model (kernels and their adjoint).
    rect : Rectangle
        ``(k_start, m_start, k_extent, m_extent)``.
    cell_chunk : int, optional
        Most kernel spectra per chunk (each is ``8 Fh Fw`` bytes, 134 MB at 4096^2);
        fewer when the free device memory is short.

    Attributes
    ----------
    kernels : torch.Tensor or None
        ``(n_cells, kH, kW)`` complex64 kernels of the rectangle once held
        (:meth:`prepare`); ``None`` while each chunk builds its own.
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
        self.kernels: torch.Tensor | None = None

    @property
    def kernel_size(self) -> int:
        """Elements of one kernel box, ``kH kW``."""
        k_h, k_w = self.box_shape
        return k_h * k_w

    def prepare(self) -> None:
        """Build and hold the rectangle's kernels when they fit.

        They are held when they take at most :data:`KERNEL_FRACTION` of the free
        device memory (always off CUDA); otherwise each chunk builds its own, and the
        next call tries again.
        """
        if self.kernels is not None:
            return

        free = free_device_bytes(self.device)
        needed = COMPLEX64_BYTES * self.n_cells * self.kernel_size

        if free is None or needed <= KERNEL_FRACTION * free:
            self.kernels = self.reconstructor.kernels(self.cells)

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

    def tiles(
        self,
        rows: torch.Tensor,
        cols: torch.Tensor,
        n_planes: int,
        budget: int | None,
    ) -> list[Tile]:
        """The regions of a call: consecutive positions and their cells per chunk.

        One region (the bounding box of every position) unless not even one cell's
        working set fits ``budget`` bytes; then the positions are halved until each
        run's does. ``None`` sizes nothing: one region of ``cell_chunk`` cells.

        Parameters
        ----------
        rows, cols : torch.Tensor
            ``(P,)`` int64 box origins (``P >= 1``).
        n_planes : int
            Planes featurized together.
        budget : int, optional
            Bytes a region's working set may take (:func:`working_set_bytes`).
        """
        kernel_size = 0 if self.kernels is not None else self.kernel_size
        tiles: list[Tile] = []
        pending = [(0, int(rows.numel()))]

        while pending:
            start, stop = pending.pop()

            if budget is None:
                tiles.append(Tile(slice(start, stop), self.cell_chunk))
                continue

            _, _, (f_h, f_w) = self.region(rows[start:stop], cols[start:stop])
            cells = fitting_cells(
                budget,
                f_h * f_w,
                stop - start,
                n_planes,
                kernel_size=kernel_size,
                limit=self.cell_chunk,
            )

            if cells >= 1 or stop - start == 1:
                tiles.append(Tile(slice(start, stop), max(1, cells)))
            else:  # the first half is popped (and appended) first
                middle = (start + stop) // 2
                pending += [(middle, stop), (start, middle)]

        return tiles

    def chunks(
        self, planes: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor
    ) -> Iterator[FeatureChunk]:
        """See :meth:`~.base.BaseImageFeaturization.chunks`."""
        self.validate(planes, rows, cols)

        if int(rows.numel()) == 0:
            return

        self.prepare()
        free = free_device_bytes(self.device)
        budget = None if free is None else int(WORKSPACE_FRACTION * free)

        for tile in self.tiles(rows, cols, int(planes.shape[0]), budget):
            yield from self._region_chunks(planes, rows, cols, tile)

    def _region_chunks(
        self,
        planes: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        tile: Tile,
    ) -> Iterator[FeatureChunk]:
        """The chunks of one region (its arrays are released when it is done)."""
        rows, cols = rows[tile.columns], cols[tile.columns]
        n_positions = int(rows.numel())
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
        for start in range(0, self.n_cells, tile.cell_chunk):
            cells = slice(start, min(start + tile.cell_chunk, self.n_cells))
            kernels = (
                self.kernels[cells]
                if self.kernels is not None
                else self.reconstructor.kernels(self.cells[cells])
            )
            spectra = torch.fft.fft2(kernels, s=fft_shape).conj()
            del kernels
            for d, image_fft in enumerate(image_ffts):
                values = torch.empty(
                    (len(spectra), n_positions), dtype=torch.complex64, device=device
                )
                for i, spectrum in enumerate(spectra):
                    valid = torch.fft.ifft2(image_fft * spectrum)[:out_h, :out_w]
                    valid = valid.reshape(-1)
                    values[i] = valid if whole else valid[local]
                    del valid
                yield FeatureChunk(d, cells, tile.columns, values)
                del values  # the caller may hold the chunk; this frame does not
            del spectra
