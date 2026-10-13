"""Featurization: the featurized pixels of an image set, one column group at a time.

:func:`schedule_tiles` fixes how a pixel selection maps onto work items and compact
columns (:class:`TileSchedule`); :func:`featurize` fills a store group for every
scheduled item and defocus plane, by tiled overlap-save FFT correlation on image
grids (:func:`featurize_grid`) or by one inner-product GEMM per box in center mode
(:func:`featurize_center`), where every box is a single pixel.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from panther_em.inference.search.types import Precision

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from panther_em.inference.search.template import KernelBank
    from panther_em.inference.search.features import BlockWriter, ColumnGroup
    from panther_em.inference.search.types import (
        ImageSet,
        PixelSelection,
        SearchGeometry,
    )

LoopOrder = Literal["pixel_major", "cell_major"]
Mode = Literal["grid", "center"]


# --------------------------------------------------------------------------- #
# Schedules
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class Tile:
    """One scheduled work item: an output tile of one image, or a batch of boxes.

    Attributes
    ----------
    index : int
        Position in :attr:`TileSchedule.items`.
    columns : slice
        Compact columns ``[c0, c1)`` of the pixels this item owns.
    image : int
        Grid mode: image index ``b`` (``-1`` in center mode).
    row0, col0 : int
        Grid mode: origin of the computed output region (see
        :meth:`TileSchedule.computed_origin`).
    local : torch.Tensor or None
        Grid mode: ``(c1 - c0,)`` int64 offsets ``(row - row0) * tile_side + (col -
        col0)`` into the computed region, in compact-column order.
    images : torch.Tensor or None
        Center mode: ``(c1 - c0,)`` int64 image ids of the boxes, one per column.
    """

    index: int
    columns: slice
    image: int = -1
    row0: int = 0
    col0: int = 0
    local: torch.Tensor | None = None
    images: torch.Tensor | None = None


@dataclass(frozen=True, eq=False)
class TileSchedule:
    """How a pixel selection maps onto work items and compact columns.

    Attributes
    ----------
    column_index : torch.Tensor
        ``(P,)`` int64 flat pixel index of every compact column.
    items : tuple[Tile, ...]
        Work items in compact-column order.
    mode : {"grid", "center"}
    grid : tuple[int, ...]
        Valid-correlation grid ``(out_h, out_w)`` (``(1,)`` in center mode).
    template_shape : tuple[int, int]
        Kernel box ``(kH, kW)``.
    tile_side : int
        Output tile side (grid mode; ``0`` in center mode).
    """

    column_index: torch.Tensor
    items: tuple[Tile, ...]
    mode: Mode
    grid: tuple[int, ...]
    template_shape: tuple[int, int]
    tile_side: int

    @property
    def tile(self) -> int:
        """Output tile side (``0`` in center mode)."""
        return self.tile_side

    @property
    def n_tile_cols(self) -> int:
        """Tiles along the grid's columns."""
        return -(-self.grid[1] // self.tile_side)

    @property
    def n_tiles(self) -> int:
        """Tiles per image."""
        return -(-self.grid[0] // self.tile_side) * self.n_tile_cols

    @property
    def region_shape(self) -> tuple[int, int]:
        """Input region a tile reads: ``tile + box - 1`` per side."""
        k_h, k_w = self.template_shape
        return self.tile_side + k_h - 1, self.tile_side + k_w - 1

    @property
    def fft_shape(self) -> tuple[int, int] | None:
        """FFT shape shared by every tile: the region rounded up to a fast length.

        Overlap-save only needs ``fft >= tile + box - 1``; the extra rows and columns
        are zero padding whose outputs are discarded. ``1535 = 5 x 307`` costs cuFFT
        a Bluestein transform, ``1536 = 2^9 x 3`` a radix one. ``None`` in center mode.
        """
        if self.mode != "grid":
            return None
        from scipy.fft import next_fast_len  # type: ignore[import-untyped]

        h, w = self.region_shape
        return int(next_fast_len(h)), int(next_fast_len(w))

    def tile_of(self, row: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
        """Per-pixel tile id ``tile_row * n_tile_cols + tile_col``."""
        return (row // self.tile_side) * self.n_tile_cols + (col // self.tile_side)

    def computed_origin(self, tile_id: int) -> tuple[int, int]:
        """``(row0, col0)`` of the (clamped) computed output region of a tile."""
        tile_row, tile_col = divmod(tile_id, self.n_tile_cols)
        row0 = min(tile_row * self.tile_side, self.grid[0] - self.tile_side)
        col0 = min(tile_col * self.tile_side, self.grid[1] - self.tile_side)
        return row0, col0


def schedule_tiles(
    geometry: SearchGeometry,
    pixels: PixelSelection,
    *,
    tile: int = 1024,
    image_batch: int = 1024,
) -> TileSchedule:
    """Tile the selection and fix the compact column order.

    Grid mode: tile-major compact order over output tiles of side ``tile`` (clamped
    to the grid); the only host sync is the per-tile population count. Center mode:
    compact order equals selection order, in batches of ``image_batch`` boxes.
    """
    if tile < 1 or image_batch < 1:
        raise ValueError("tile and image_batch must be >= 1")
    flat = pixels.index.to("cpu")
    grid = tuple(int(g) for g in geometry.grid)  # (out_h, out_w), or (1,) in center mode
    if geometry.mode == "center":
        items = [
            Tile(i, columns, images=flat[columns])
            for i, columns in enumerate(cell_chunks(int(flat.numel()), image_batch))
        ]
        return TileSchedule(flat, tuple(items), "center", grid, geometry.template_shape, 0)

    out_h, out_w = grid
    side = min(int(tile), out_h, out_w)
    skeleton = TileSchedule(flat, (), "grid", grid, geometry.template_shape, side)
    image, within_image = flat // (out_h * out_w), flat % (out_h * out_w)
    row, col = within_image // out_w, within_image % out_w
    global_tile = image * skeleton.n_tiles + skeleton.tile_of(row, col)

    order = torch.argsort(global_tile, stable=True)  # tile-major compact order
    populations = torch.bincount(
        global_tile[order], minlength=geometry.n_images * skeleton.n_tiles
    )
    items = []
    first_column = 0
    for global_id, population in enumerate(populations.tolist()):
        if population == 0:
            continue
        image_id, tile_id = divmod(global_id, skeleton.n_tiles)
        row0, col0 = skeleton.computed_origin(tile_id)
        columns = slice(first_column, first_column + population)
        owned = order[columns]
        local = (row[owned] - row0) * side + (col[owned] - col0)
        items.append(Tile(len(items), columns, image_id, row0, col0, local))
        first_column += population
    return TileSchedule(flat[order], tuple(items), "grid", grid, geometry.template_shape, side)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


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


Featurize = Callable[..., None]
"""Signature of :func:`featurize`: ``(bank, images, group, writer, schedule, *, chunk,
order, stream)``; tests inject stubs with the same signature."""


def featurize(
    bank: KernelBank,
    images: ImageSet,
    group: ColumnGroup,
    writer: BlockWriter,
    schedule: TileSchedule,
    *,
    chunk: int = 64,
    order: LoopOrder = "pixel_major",
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Featurize every cell of ``group`` for every scheduled column and plane."""
    if schedule.mode == "center":
        featurize_center(bank, images, group, writer, schedule, chunk=chunk, stream=stream)
    else:
        featurize_grid(
            bank, images, group, writer, schedule, chunk=chunk, order=order, stream=stream
        )


# --------------------------------------------------------------------------- #
# Grid mode: tiled FFT correlation
# --------------------------------------------------------------------------- #


def featurize_grid(
    bank: KernelBank,
    images: ImageSet,
    group: ColumnGroup,
    writer: BlockWriter,
    schedule: TileSchedule,
    *,
    chunk: int = 64,
    order: LoopOrder = "pixel_major",
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Tiled overlap-save valid cross-correlation against the kernel bank.

    ``pixel_major`` finishes one tile (all cells) before the next, so columns
    complete early (what a pinned-host store needs); ``cell_major`` keeps every
    tile's image FFT resident and sweeps cells in the outer loop, so each kernel
    spectrum is fetched once per plane.
    """
    if schedule.mode != "grid":
        raise ValueError("featurize_grid needs a grid-mode schedule")
    if order not in ("pixel_major", "cell_major"):
        raise ValueError(f"unknown loop order {order!r}")
    fft_shape = schedule.fft_shape
    assert fft_shape is not None
    cells = group_cells(group)
    n_cells = int(cells.shape[0])
    device = bank.device
    with _stream_scope(stream):
        for plane in group.planes:
            filtered: dict[int, torch.Tensor] = {}  # image -> plane-filtered image

            def tile_fft(tile: Tile) -> torch.Tensor:
                image = filtered.get(tile.image)
                if image is None:
                    raw = images.images[tile.image].to(device)
                    image = apply_fourier_filter(raw, images.filter_plane(plane, tile.image))
                    filtered[tile.image] = image
                return _tile_fft(image, tile, schedule)

            if order == "pixel_major":
                work = (
                    (tile, image_fft, cell_range)
                    for tile in schedule.items
                    for image_fft in (tile_fft(tile),)
                    for cell_range in cell_chunks(n_cells, chunk)
                )
            else:
                ffts = [tile_fft(tile) for tile in schedule.items]
                work = (
                    (tile, image_fft, cell_range)
                    for cell_range in cell_chunks(n_cells, chunk)
                    for tile, image_fft in zip(schedule.items, ffts, strict=True)
                )
            for tile, image_fft, cell_range in work:
                spectra = bank.spectral_views(cells[cell_range], fft_shape)
                features = _correlate_tile(image_fft, spectra, tile, schedule.tile_side)
                writer.write(
                    cell_range, (tile.columns.start, tile.columns.stop), features, plane=plane
                )
    writer.close()


def _tile_fft(filtered_image: torch.Tensor, tile: Tile, schedule: TileSchedule) -> torch.Tensor:
    """FFT of the tile's input region (computed once per tile and plane)."""
    region_h, region_w = schedule.region_shape
    region = filtered_image[
        tile.row0 : tile.row0 + region_h, tile.col0 : tile.col0 + region_w
    ]
    image_fft: torch.Tensor = torch.fft.fft2(region, s=schedule.fft_shape)
    return image_fft


def _correlate_tile(
    image_fft: torch.Tensor, kernel_spectra: Sequence[torch.Tensor], tile: Tile, side: int
) -> torch.Tensor:
    """Valid correlation of one tile with a chunk of kernels, at owned pixels.

    One cell at a time: the product and inverse FFT of a large tile (134 MB each at
    4096^2) never exist for the whole chunk, the cached spectra are read in place,
    and a tile that owns its whole computed region skips the gather.
    """
    assert tile.local is not None
    n_owned = int(tile.local.numel())
    features = torch.empty(
        (len(kernel_spectra), n_owned), dtype=torch.complex64, device=image_fft.device
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


def featurize_center(
    bank: KernelBank,
    images: ImageSet,
    group: ColumnGroup,
    writer: BlockWriter,
    schedule: TileSchedule,
    *,
    chunk: int = 64,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """``Y[b, cell] = sum_ij box_b[ij] conj(K_cell[ij])`` by one GEMM per box batch.

    The GEMM runs at the group's precision: fp32 (exact), tf32 (tensor cores, ~5e-4
    relative error per product) or fp16 operands with fp32 accumulation.
    """
    if schedule.mode != "center":
        raise ValueError("featurize_center needs a center-mode schedule")
    cells = group_cells(group)
    n_cells = int(cells.shape[0])
    dtype = torch.float16 if group.precision is Precision.FP16 else torch.float32
    with _stream_scope(stream):
        for plane in group.planes:
            for batch in schedule.items:
                assert batch.images is not None
                # one cast per box batch (8192 boxes x 512^2 is 8.6 GB in fp32)
                boxes = _centered_boxes(bank, images, batch.images, plane).to(dtype)
                for cell_range in cell_chunks(n_cells, chunk):
                    features = _inner_products(bank, cells[cell_range], boxes, group.precision)
                    writer.write(
                        cell_range,
                        (batch.columns.start, batch.columns.stop),
                        features,
                        plane=plane,
                    )
    writer.close()


def _centered_boxes(
    bank: KernelBank, images: ImageSet, image_ids: torch.Tensor, plane: int
) -> torch.Tensor:
    """Filtered, center-cropped boxes flattened to ``(Bt, kH * kW)``."""
    boxes = images.images[image_ids].to(bank.device)
    if images.filters is not None:
        filt = (
            images.filters[image_ids, plane]
            if images.per_image_filters
            else images.filters[plane]
        )
        boxes = apply_fourier_filter(boxes, filt.to(bank.device))
    k_h, k_w = bank.template_shape
    height, width = boxes.shape[-2:]
    row0, col0 = (height - k_h) // 2, (width - k_w) // 2
    return boxes[:, row0 : row0 + k_h, col0 : col0 + k_w].reshape(boxes.shape[0], k_h * k_w)


def _inner_products(
    bank: KernelBank, cells: torch.Tensor, boxes: torch.Tensor, precision: Precision
) -> torch.Tensor:
    """``(n_cells, Bt)`` complex inner products of boxes with the cells' kernels."""
    dtype = torch.float16 if precision is Precision.FP16 else torch.float32
    planar = bank.planar(cells, dtype)  # (2, n_cells, kH*kW): [Re K ; -Im K]
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
