"""Featurizers: the featurized pixels of an image set, one column group at a time.

For every selected pixel ``p`` and every ``(k, m)`` cell of a column group's rectangle
the feature is the valid cross-correlation with that cell's Cartesian kernel,

    ``Y[p, cell] = sum_ij  image[p + ij] * conj(kernel_cell[ij])``

Two featurizers implement the :class:`Featurizer` protocol:

* :class:`FFTCorrelationFeaturizer` (grid mode) -- overlap-save correlation by FFT over
  uniform output tiles. Tiles, rather than one full-image FFT per kernel chunk, bound
  the transient memory independently of image size.
* :class:`InnerProductFeaturizer` (center mode) -- a box equal to the template has one
  valid pixel, so the correlation collapses to an inner product: one real GEMM of the
  planar kernel rows against the flattened boxes, no FFT.

Defocus planes (``ImageSet.filters``) are produced by filtering the image on its full
grid, one plane at a time -- exact, and never more than one filtered copy alive.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import torch

from panther_em.inference.search.types import Precision

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from panther_em.inference.search.basis import KernelBank
    from panther_em.inference.search.store import BlockWriter, ColumnGroup
    from panther_em.inference.search.types import (
        ImageSet,
        PixelSelection,
        SearchGeometry,
    )

LoopOrder = Literal["pixel_major", "cell_major"]


# --------------------------------------------------------------------------- #
# Tile geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TileGrid:
    """Uniform tiling of a valid-correlation grid for overlap-save correlation.

    Parameters
    ----------
    out_h, out_w : int
        Valid-correlation grid size.
    template_shape : tuple[int, int]
        Kernel box ``(kH, kW)``.
    tile_side : int
        Requested output tile side; clamped to ``min(out_h, out_w)``.
    """

    out_h: int
    out_w: int
    template_shape: tuple[int, int]
    tile_side: int

    def __post_init__(self) -> None:
        """Clamp the tile side to the grid."""
        side = min(int(self.tile_side), int(self.out_h), int(self.out_w))
        if side < 1:
            raise ValueError("tile side must be >= 1")
        object.__setattr__(self, "tile_side", side)

    @property
    def n_tile_rows(self) -> int:
        """Tiles along the grid's rows."""
        return -(-self.out_h // self.tile_side)

    @property
    def n_tile_cols(self) -> int:
        """Tiles along the grid's columns."""
        return -(-self.out_w // self.tile_side)

    @property
    def n_tiles(self) -> int:
        """Tiles per image."""
        return self.n_tile_rows * self.n_tile_cols

    @property
    def region_shape(self) -> tuple[int, int]:
        """Input region a tile reads: ``tile + box - 1`` per side."""
        k_h, k_w = self.template_shape
        return self.tile_side + k_h - 1, self.tile_side + k_w - 1

    @property
    def fft_shape(self) -> tuple[int, int]:
        """FFT shape shared by every tile: the region rounded up to a fast length.

        Overlap-save only needs ``fft >= tile + box - 1``; the extra rows and columns
        are zero padding whose outputs are discarded. ``1535 = 5 x 307`` costs cuFFT
        a Bluestein transform, ``1536 = 2^9 x 3`` a radix one.
        """
        from scipy.fft import next_fast_len  # type: ignore[import-untyped]

        return tuple(int(next_fast_len(n)) for n in self.region_shape)  # type: ignore[return-value]

    def tile_of(self, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        """Per-pixel tile id ``tile_row * n_tile_cols + tile_col``."""
        return (row // self.tile_side) * self.n_tile_cols + (col // self.tile_side)

    def computed_origin(self, tile_id: int) -> tuple[int, int]:
        """``(row0, col0)`` of the (clamped) computed output region of a tile."""
        tile_row, tile_col = divmod(tile_id, self.n_tile_cols)
        row0 = min(tile_row * self.tile_side, self.out_h - self.tile_side)
        col0 = min(tile_col * self.tile_side, self.out_w - self.tile_side)
        return row0, col0


@dataclass(frozen=True, eq=False)
class Tile:
    """One scheduled output tile of one image (grid mode).

    Attributes
    ----------
    index : int
        Position in :attr:`TileSchedule.items`.
    image : int
        Image index ``b``.
    row0, col0 : int
        Origin of the computed output region (see :class:`TileGrid`).
    columns : slice
        Compact columns ``[c0, c1)`` of the pixels this tile owns.
    local : torch.Tensor
        ``(c1 - c0,)`` int64 offsets ``(row - row0) * tile_side + (col - col0)`` into
        the computed region, in compact-column order.
    """

    index: int
    image: int
    row0: int
    col0: int
    columns: slice
    local: torch.Tensor


@dataclass(frozen=True, eq=False)
class ImageBatch:
    """One batch of center-mode images, each of which is a single pixel."""

    index: int
    images: torch.Tensor  # (Bt,) int64 image ids
    columns: slice


@dataclass(frozen=True, eq=False)
class TileSchedule:
    """How a pixel selection maps onto tiles / image batches and compact columns.

    Attributes
    ----------
    column_index : torch.Tensor
        ``(P,)`` int64 flat pixel index of every compact column.
    items : tuple
        :class:`Tile` (grid mode) or :class:`ImageBatch` (center mode) entries, in
        compact-column order.
    grid : TileGrid or None
        The tiling (``None`` in center mode).
    """

    column_index: torch.Tensor
    items: tuple[Tile | ImageBatch, ...]
    grid: TileGrid | None

    @property
    def tile(self) -> int:
        """Output tile side (``0`` in center mode)."""
        return 0 if self.grid is None else self.grid.tile_side

    @property
    def fft_shape(self) -> tuple[int, int] | None:
        """FFT shape shared by every tile (``None`` in center mode)."""
        return None if self.grid is None else self.grid.fft_shape

    def tiles(self) -> list[Tile]:
        """The grid-mode items."""
        return [item for item in self.items if isinstance(item, Tile)]

    def image_batches(self) -> list[ImageBatch]:
        """The center-mode items."""
        return [item for item in self.items if isinstance(item, ImageBatch)]


# --------------------------------------------------------------------------- #
# Protocol and shared helpers
# --------------------------------------------------------------------------- #


@runtime_checkable
class Featurizer(Protocol):
    """Produces featurized pixels for an image set, one column group at a time."""

    def schedule(
        self, geometry: SearchGeometry, pixels: PixelSelection
    ) -> TileSchedule:
        """Tile the selection and fix the compact column order."""

    def produce(
        self,
        images: ImageSet,
        group: ColumnGroup,
        writer: BlockWriter,
        schedule: TileSchedule,
        *,
        chunk: int = 64,
        order: LoopOrder = "pixel_major",
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Featurize every cell of ``group`` for every scheduled column."""


def group_cells(group: ColumnGroup) -> torch.Tensor:
    """Row-major ``(num_cells, 2)`` ``(k, m)`` cells of a group's rectangle."""
    k_start, m_start, k_extent, m_extent = group.rect
    k = torch.arange(k_start, k_start + k_extent)
    m = torch.arange(m_start, m_start + m_extent)
    return torch.stack(torch.meshgrid(k, m, indexing="ij"), dim=-1).reshape(-1, 2)


def cell_chunks(n_cells: int, chunk: int) -> Iterator[slice]:
    """Successive ``[c0, c1)`` slices of at most ``chunk`` cells."""
    for c0 in range(0, n_cells, chunk):
        yield slice(c0, min(c0 + chunk, n_cells))


def apply_fourier_filter(
    image: torch.Tensor, filt: torch.Tensor | None
) -> torch.Tensor:
    """``irfft2(rfft2(image) * filt)`` on the image's grid (``image`` if no filter)."""
    if filt is None:
        return image
    height, width = image.shape[-2:]
    filtered: torch.Tensor = torch.fft.irfft2(
        torch.fft.rfft2(image) * filt.to(image.device), s=(height, width)
    )
    return filtered


def _stream_scope(
    stream: torch.cuda.Stream | None,
) -> contextlib.AbstractContextManager:
    return torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()


# --------------------------------------------------------------------------- #
# Grid mode: tiled FFT correlation
# --------------------------------------------------------------------------- #


class _FilteredImageCache:
    """One defocus plane's filtered images, built on first use on the device."""

    def __init__(self, images: ImageSet, plane: int, device: torch.device) -> None:
        self._images = images
        self._plane = plane
        self._device = device
        self._cache: dict[int, torch.Tensor] = {}

    def __getitem__(self, image: int) -> torch.Tensor:
        filtered = self._cache.get(image)
        if filtered is None:
            filt = self._images.filter_plane(self._plane, image)
            raw = self._images.images[image].to(self._device)
            filtered = apply_fourier_filter(raw, filt)
            self._cache[image] = filtered
        return filtered


class FFTCorrelationFeaturizer:
    """Tiled overlap-save valid cross-correlation against the kernel bank.

    Parameters
    ----------
    bank : KernelBank
        Supplies kernel spectra at the tile FFT shape.
    tile : int, optional
        Requested output tile side; clamped to the valid grid. Defaults to 1024.
    """

    def __init__(self, bank: KernelBank, *, tile: int = 1024) -> None:
        if tile < 1:
            raise ValueError("tile must be >= 1")
        self.bank = bank
        self.tile = int(tile)

    @property
    def device(self) -> torch.device:
        """Compute device (the bank's)."""
        return self.bank.device

    # -- scheduling ------------------------------------------------------------

    def schedule(
        self, geometry: SearchGeometry, pixels: PixelSelection
    ) -> TileSchedule:
        """Tile-major compact order; the only host sync is the per-tile count."""
        if geometry.mode != "grid":
            raise ValueError("FFTCorrelationFeaturizer needs grid mode")

        out_h, out_w = geometry.grid
        grid = TileGrid(out_h, out_w, geometry.template_shape, self.tile)

        flat = pixels.index.to("cpu")
        image, within_image = flat // (out_h * out_w), flat % (out_h * out_w)
        row, col = within_image // out_w, within_image % out_w
        global_tile = image * grid.n_tiles + grid.tile_of(row, col)

        order = torch.argsort(global_tile, stable=True)  # tile-major compact order
        populations = torch.bincount(
            global_tile[order], minlength=geometry.n_images * grid.n_tiles
        )

        items: list[Tile | ImageBatch] = []
        first_column = 0
        for global_id, population in enumerate(populations.tolist()):
            if population == 0:
                continue
            image_id, tile_id = divmod(global_id, grid.n_tiles)
            row0, col0 = grid.computed_origin(tile_id)
            columns = slice(first_column, first_column + population)
            owned = order[columns]
            local = (row[owned] - row0) * grid.tile_side + (col[owned] - col0)
            items.append(Tile(len(items), image_id, row0, col0, columns, local))
            first_column += population

        return TileSchedule(flat[order], tuple(items), grid)

    # -- production ------------------------------------------------------------

    def produce(
        self,
        images: ImageSet,
        group: ColumnGroup,
        writer: BlockWriter,
        schedule: TileSchedule,
        *,
        chunk: int = 64,
        order: LoopOrder = "pixel_major",
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Featurize ``group`` for every scheduled tile and defocus plane."""
        grid = schedule.grid
        if grid is None:
            raise ValueError("schedule is not a grid-mode schedule")
        cells = group_cells(group)
        n_cells = int(cells.shape[0])
        with _stream_scope(stream):
            for plane in group.planes:
                filtered = _FilteredImageCache(images, plane, self.device)
                for tile, image_fft, cell_range in self._iterate(
                    schedule.tiles(), filtered, grid, n_cells, chunk, order
                ):
                    spectra = self.bank.spectral_views(
                        cells[cell_range], grid.fft_shape
                    )
                    features = self._correlate_tile(image_fft, spectra, tile, grid)
                    writer.write(
                        cell_range,
                        (tile.columns.start, tile.columns.stop),
                        features,
                        plane=plane,
                    )
        writer.close()

    def _iterate(
        self,
        tiles: Sequence[Tile],
        filtered: _FilteredImageCache,
        grid: TileGrid,
        n_cells: int,
        chunk: int,
        order: LoopOrder,
    ) -> Iterator[tuple[Tile, torch.Tensor, slice]]:
        """Yield ``(tile, image_fft, cell_range)`` in the requested loop order.

        Notes
        -----
        ``pixel_major`` finishes one tile (all cells) before the next, so columns
        complete early; ``cell_major`` keeps every tile's image FFT resident and sweeps
        cells in the outer loop, so each kernel spectrum is fetched once.
        """
        if order == "pixel_major":
            for tile in tiles:
                image_fft = self._tile_fft(filtered[tile.image], tile, grid)
                for cell_range in cell_chunks(n_cells, chunk):
                    yield tile, image_fft, cell_range
        elif order == "cell_major":
            ffts = [self._tile_fft(filtered[tile.image], tile, grid) for tile in tiles]
            for cell_range in cell_chunks(n_cells, chunk):
                for tile, image_fft in zip(tiles, ffts, strict=True):
                    yield tile, image_fft, cell_range
        else:
            raise ValueError(f"unknown loop order {order!r}")

    @staticmethod
    def _tile_fft(
        filtered_image: torch.Tensor, tile: Tile, grid: TileGrid
    ) -> torch.Tensor:
        """FFT of the tile's input region (computed once per tile and plane)."""
        region_h, region_w = grid.region_shape
        region = filtered_image[
            tile.row0 : tile.row0 + region_h, tile.col0 : tile.col0 + region_w
        ]
        image_fft: torch.Tensor = torch.fft.fft2(region, s=grid.fft_shape)
        return image_fft

    @staticmethod
    def _correlate_tile(
        image_fft: torch.Tensor,
        kernel_spectra: Sequence[torch.Tensor],
        tile: Tile,
        grid: TileGrid,
    ) -> torch.Tensor:
        """Valid correlation of one tile with a chunk of kernels, at owned pixels.

        One cell at a time: the product and inverse FFT of a large tile (134 MB each
        at 4096^2) never exist for the whole chunk, the cached spectra are read in
        place, and a tile that owns its whole computed region skips the gather.
        """
        side = grid.tile_side
        n_owned = int(tile.local.numel())
        features = torch.empty(
            (len(kernel_spectra), n_owned),
            dtype=torch.complex64,
            device=image_fft.device,
        )
        local = tile.local.to(image_fft.device)
        full = n_owned == side * side and bool(
            torch.equal(local, torch.arange(n_owned, device=local.device))
        )
        for i, spectrum in enumerate(kernel_spectra):
            correlation = torch.fft.ifft2(image_fft * spectrum)[:side, :side]
            flat = correlation.reshape(-1)
            features[i] = flat if full else flat[local]
        return features


# --------------------------------------------------------------------------- #
# Center mode: inner product
# --------------------------------------------------------------------------- #


class InnerProductFeaturizer:
    """Center-only mode: ``Y[b, cell] = sum_ij box_b[ij] conj(K_cell[ij])`` by one GEMM.

    Parameters
    ----------
    bank : KernelBank
        Supplies planar kernel rows.
    image_batch : int, optional
        Images per GEMM. Defaults to 1024.
    """

    def __init__(self, bank: KernelBank, *, image_batch: int = 1024) -> None:
        if image_batch < 1:
            raise ValueError("image_batch must be >= 1")
        self.bank = bank
        self.image_batch = int(image_batch)

    @property
    def device(self) -> torch.device:
        """Compute device (the bank's)."""
        return self.bank.device

    def schedule(
        self, geometry: SearchGeometry, pixels: PixelSelection
    ) -> TileSchedule:
        """Compact order equals selection order; items are image batches."""
        if geometry.mode != "center":
            raise ValueError("InnerProductFeaturizer needs center mode")
        flat = pixels.index.to("cpu")
        items: list[Tile | ImageBatch] = []
        for columns in cell_chunks(int(flat.numel()), self.image_batch):
            items.append(ImageBatch(len(items), flat[columns], columns))
        return TileSchedule(flat, tuple(items), None)

    def produce(
        self,
        images: ImageSet,
        group: ColumnGroup,
        writer: BlockWriter,
        schedule: TileSchedule,
        *,
        chunk: int = 64,
        order: LoopOrder = "pixel_major",
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Featurize ``group`` for every scheduled image batch and defocus plane."""
        if schedule.grid is not None:
            raise ValueError("schedule is not a center-mode schedule")
        cells = group_cells(group)
        n_cells = int(cells.shape[0])
        with _stream_scope(stream):
            for plane in group.planes:
                dtype = (
                    torch.float16 if group.precision is Precision.FP16 else torch.float32
                )
                for batch in schedule.image_batches():
                    # one cast per image batch (8192 boxes x 512^2 is 8.6 GB in fp32)
                    boxes = self._centered_boxes(images, batch.images, plane).to(dtype)
                    for cell_range in cell_chunks(n_cells, chunk):
                        features = self._inner_products(
                            cells[cell_range], boxes, group.precision
                        )
                        writer.write(
                            cell_range,
                            (batch.columns.start, batch.columns.stop),
                            features,
                            plane=plane,
                        )
        writer.close()

    def _centered_boxes(
        self, images: ImageSet, image_ids: torch.Tensor, plane: int
    ) -> torch.Tensor:
        """Filtered, center-cropped boxes flattened to ``(Bt, kH * kW)``."""
        boxes = images.images[image_ids].to(self.device)
        if images.filters is not None:
            filt = (
                images.filters[image_ids, plane]
                if images.per_image_filters
                else images.filters[plane]
            )
            boxes = apply_fourier_filter(boxes, filt.to(self.device))
        k_h, k_w = self.bank.template_shape
        height, width = boxes.shape[-2:]
        row0, col0 = (height - k_h) // 2, (width - k_w) // 2
        return boxes[:, row0 : row0 + k_h, col0 : col0 + k_w].reshape(
            boxes.shape[0], k_h * k_w
        )

    def _inner_products(
        self,
        cells: torch.Tensor,
        boxes: torch.Tensor,
        precision: Precision = Precision.FP32,
    ) -> torch.Tensor:
        """``(n_cells, Bt)`` complex inner products of boxes with the cells' kernels.

        The GEMM runs at the group's precision: fp32 (exact), tf32 (tensor cores,
        ~5x faster, ~5e-4 relative error per product) or fp16 operands with fp32
        accumulation.
        """
        dtype = torch.float16 if precision is Precision.FP16 else torch.float32
        planar = self.bank.planar(cells, dtype)  # (2, n_cells, kH*kW): [Re K ; -Im K]
        rhs = boxes.T.to(dtype)  # no copy when the caller already cast the boxes
        if precision is Precision.TF32 and boxes.is_cuda:
            prev = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                real_imag = planar @ rhs
            finally:
                torch.backends.cuda.matmul.allow_tf32 = prev
        else:
            real_imag = planar @ rhs
        real_imag = real_imag.to(torch.float32)  # (2, n_cells, Bt)
        return torch.complex(real_imag[0], real_imag[1])


def make_featurizer(
    geometry: SearchGeometry,
    bank: KernelBank,
    *,
    tile: int = 1024,
    image_batch: int = 1024,
) -> Featurizer:
    """Inner-product featurizer in center mode, FFT correlation otherwise."""
    if geometry.mode == "center":
        return InnerProductFeaturizer(bank, image_batch=image_batch)

    return FFTCorrelationFeaturizer(bank, tile=tile)
