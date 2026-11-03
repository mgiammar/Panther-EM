"""Featurization: the featurized pixels of an image set, one column group at a time.

:func:`schedule_tiles` fixes how a pixel selection maps onto work items and compact
columns (:class:`TileSchedule`) and which engine featurizes it; :func:`featurize` fills
a store group for every scheduled item and defocus plane with that engine:

``fft``
    Tiled overlap-save FFT correlation on image grids (:func:`featurize_grid`): one
    inverse FFT per cell and tile, whatever the number of selected pixels.
``direct``
    Patch gather + planar GEMM (:func:`featurize_direct`): the template-sized box
    under every selected position, from any image set -- a sparse selection in a
    micrograph, a particle stack at its center pixel, or a slightly larger stack at
    every valid offset -- times the ``[Re K; -Im K]`` kernels, at tensor-core rate.
    Its cost is one GEMM row per position.

:func:`choose_mode` picks between them from the constants at the top of this module.
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

    from panther_em.inference.search.features import BlockWriter, ColumnGroup
    from panther_em.inference.search.template import KernelBank
    from panther_em.inference.search.types import (
        ImageSet,
        PixelSelection,
        Rectangle,
        SearchGeometry,
    )

LoopOrder = Literal["pixel_major", "cell_major"]
Mode = Literal["fft", "direct"]

# --------------------------------------------------------------------------- #
# Engine selection constants
# --------------------------------------------------------------------------- #
# Measured on an RTX 6000 Ada (4096^2 micrograph, 512 box, rank 4096, one plane): the
# FFT engine takes 14.3 s for any selection, the direct engine 56.6 us per selected
# position with its GEMM at the fp16 tensor-core peak, so the two cost the same at
# 2.0 % of the valid grid. Both scale with the cell count, so the ratio is rank
# independent; it does scale with the box area (direct) vs the image area (FFT).
DIRECT_MAX_DENSITY = 0.02
"""Selected / valid-grid positions at or below which the direct engine is chosen."""

DIRECT_MAX_BOX_EXCESS = 32
"""Images at most this many pixels larger than the template per side (particle stacks)
are featurized directly at every selected position: such a stack's valid grid is at
most 33 x 33 positions per box, far below the ~4000 at which a per-box FFT breaks even.
Larger images fall through to the density rule."""

DIRECT_PLANAR_FRACTION = 0.6
"""Fraction of the free device memory a rectangle's planar block (``2 r x kH kW`` at
the group's precision: 1 MiB per cell in fp16 at a 512 box) may take when it still has
to be built; a block the kernel bank already holds is used regardless. Resident, every
position batch is one GEMM; otherwise the engine rebuilds the kernels per cell chunk
and box batch, which re-reads the spatial cache and runs ~1.8x slower (rank 8192,
2048 crop: 1.97 s resident vs 3.54 s chunked)."""

DIRECT_PLANAR_CHUNK_BYTES = 2**30
"""When the planar kernels are not resident, bytes of them per cell chunk: every chunk
re-reads the box batch (512 MiB at 1024 positions of a 512 box), so chunks are sized
by bytes rather than by the kernel count ``chunk`` (32 cells would read the boxes
256 times per batch at rank 8192)."""

DIRECT_IMAGE_CACHE = 8
"""Filtered images the direct engine keeps per plane (positions are image-major)."""


# --------------------------------------------------------------------------- #
# Schedules
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class Tile:
    """One scheduled work item: an output tile of one image, or a batch of positions.

    Attributes
    ----------
    index : int
        Position in :attr:`TileSchedule.items`.
    columns : slice
        Compact columns ``[c0, c1)`` of the pixels this item owns.
    image : int
        FFT mode: image index ``b`` (``-1`` in direct mode).
    row0, col0 : int
        FFT mode: origin of the computed output region (see
        :meth:`TileSchedule.computed_origin`).
    local : torch.Tensor or None
        FFT mode: ``(c1 - c0,)`` int64 offsets ``(row - row0) * tile_side + (col -
        col0)`` into the computed region, in compact-column order.
    images, rows, cols : torch.Tensor or None
        Direct mode: ``(c1 - c0,)`` int64 image id and box origin (top-left corner on
        the image) of every position, one per column.
    """

    index: int
    columns: slice
    image: int = -1
    row0: int = 0
    col0: int = 0
    local: torch.Tensor | None = None
    images: torch.Tensor | None = None
    rows: torch.Tensor | None = None
    cols: torch.Tensor | None = None


@dataclass(frozen=True, eq=False)
class TileSchedule:
    """How a pixel selection maps onto work items and compact columns.

    Attributes
    ----------
    column_index : torch.Tensor
        ``(P,)`` int64 flat pixel index of every compact column.
    items : tuple[Tile, ...]
        Work items in compact-column order.
    mode : {"fft", "direct"}
        The engine that featurizes the items.
    grid : tuple[int, ...]
        Valid-correlation grid ``(out_h, out_w)`` (``(1,)`` for a center geometry).
    template_shape : tuple[int, int]
        Kernel box ``(kH, kW)``.
    tile_side : int
        Output tile side (FFT mode; ``0`` in direct mode).
    """

    column_index: torch.Tensor
    items: tuple[Tile, ...]
    mode: Mode
    grid: tuple[int, ...]
    template_shape: tuple[int, int]
    tile_side: int

    @property
    def tile(self) -> int:
        """Output tile side (``0`` in direct mode)."""
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
        a Bluestein transform, ``1536 = 2^9 x 3`` a radix one. ``None`` in direct mode.
        """
        if self.mode != "fft":
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


def choose_mode(geometry: SearchGeometry, pixels: PixelSelection) -> Mode:
    """The engine for a selection, from the module constants.

    ``direct`` for a center geometry, for images within :data:`DIRECT_MAX_BOX_EXCESS`
    pixels of the template box per side, and for selections of at most
    :data:`DIRECT_MAX_DENSITY` of the valid grid; ``fft`` otherwise.
    """
    if geometry.mode == "center":
        return "direct"
    (h, w), (k_h, k_w) = geometry.image_shape, geometry.template_shape
    if h - k_h <= DIRECT_MAX_BOX_EXCESS and w - k_w <= DIRECT_MAX_BOX_EXCESS:
        return "direct"
    if len(pixels) <= DIRECT_MAX_DENSITY * geometry.n_pixels:
        return "direct"
    return "fft"


def _positions(
    geometry: SearchGeometry, flat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(image, row, col)`` box origins of flat pixel indices (center: the middle)."""
    if geometry.mode == "center":
        (h, w), (k_h, k_w) = geometry.image_shape, geometry.template_shape
        rows = torch.full_like(flat, (h - k_h) // 2)
        cols = torch.full_like(flat, (w - k_w) // 2)
        return flat, rows, cols
    out_h, out_w = geometry.grid
    image, within = flat // (out_h * out_w), flat % (out_h * out_w)
    return image, within // out_w, within % out_w


def schedule_tiles(
    geometry: SearchGeometry,
    pixels: PixelSelection,
    *,
    tile: int = 1024,
    image_batch: int = 1024,
    mode: Mode | None = None,
    order: torch.Tensor | None = None,
) -> TileSchedule:
    """Tile the selection, fix the compact column order and choose the engine.

    FFT mode: tile-major compact order over output tiles of side ``tile`` (clamped to
    the grid); the only host sync is the per-tile population count. Direct mode:
    compact order equals selection order (or ``order``, a permutation of the
    selection, so a store's existing column order can be kept), in batches of
    ``image_batch`` positions. ``mode=None`` lets :func:`choose_mode` decide.
    """
    if tile < 1 or image_batch < 1:
        raise ValueError("tile and image_batch must be >= 1")
    mode = mode or choose_mode(geometry, pixels)
    flat = pixels.index.to("cpu")
    grid = tuple(
        int(g) for g in geometry.grid
    )  # (out_h, out_w), or (1,) in center mode
    if mode == "direct":
        if order is not None:
            order = order.to("cpu")
            if not torch.equal(torch.sort(order).values, flat):
                raise ValueError("order must be a permutation of the selected pixels")
            flat = order
        image, row, col = _positions(geometry, flat)
        items = [
            Tile(
                i, columns, images=image[columns], rows=row[columns], cols=col[columns]
            )
            for i, columns in enumerate(cell_chunks(int(flat.numel()), image_batch))
        ]
        return TileSchedule(
            flat, tuple(items), "direct", grid, geometry.template_shape, 0
        )
    if mode != "fft":
        raise ValueError(f"unknown featurization mode {mode!r}")
    if geometry.mode != "grid":
        raise ValueError("the FFT engine needs a grid geometry")

    out_h, out_w = grid
    side = min(int(tile), out_h, out_w)
    skeleton = TileSchedule(flat, (), "fft", grid, geometry.template_shape, side)
    image, row, col = _positions(geometry, flat)
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
    return TileSchedule(
        flat[order], tuple(items), "fft", grid, geometry.template_shape, side
    )


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
    """``irfft2(rfft2(image) * filt)`` on the image's grid (``image`` if no filter).

    Call it on one ``(H, W)`` image at a time on CUDA: cuFFT's batched ``irfft2`` of a
    ``(B, 4096, 4096)`` stack is off by 6e-3 relative (torch 2.11 / CUDA 13) where the
    2-D transform is accurate to 4e-7.
    """
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


def _matmul(a: torch.Tensor, b: torch.Tensor, precision: Precision) -> torch.Tensor:
    """``a @ b`` with tf32 enabled for that precision (CUDA only)."""
    if precision is Precision.TF32 and a.is_cuda:
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            return a @ b
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
    return a @ b


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
    if schedule.mode == "direct":
        featurize_direct(
            bank, images, group, writer, schedule, chunk=chunk, stream=stream
        )
    else:
        featurize_grid(
            bank,
            images,
            group,
            writer,
            schedule,
            chunk=chunk,
            order=order,
            stream=stream,
        )


# --------------------------------------------------------------------------- #
# FFT engine: tiled overlap-save correlation of image grids
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
    if schedule.mode != "fft":
        raise ValueError("featurize_grid needs an fft-mode schedule")
    if order not in ("pixel_major", "cell_major"):
        raise ValueError(f"unknown loop order {order!r}")
    fft_shape = schedule.fft_shape
    assert fft_shape is not None
    cells = group_cells(group)
    n_cells = int(cells.shape[0])
    device = bank.device
    # Caching a sweep larger than the cache never hits and evicts other stages' spectra
    # (8192 cells at 2560^2 are 400 GiB): fill the cache only when it fits.
    cache = bank.spectral_fits(n_cells, fft_shape)
    with _stream_scope(stream):
        for plane in group.planes:
            filtered: dict[int, torch.Tensor] = {}  # image -> plane-filtered image

            def tile_fft(tile: Tile) -> torch.Tensor:
                image = filtered.get(tile.image)
                if image is None:
                    raw = images.images[tile.image].to(device)
                    image = apply_fourier_filter(
                        raw, images.filter_plane(plane, tile.image)
                    )
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
                spectra = bank.spectral_views(cells[cell_range], fft_shape, cache=cache)
                features = _correlate_tile(image_fft, spectra, tile, schedule.tile_side)
                writer.write(
                    cell_range,
                    (tile.columns.start, tile.columns.stop),
                    features,
                    plane=plane,
                )
    writer.close()


def _tile_fft(
    filtered_image: torch.Tensor, tile: Tile, schedule: TileSchedule
) -> torch.Tensor:
    """FFT of the tile's input region (computed once per tile and plane)."""
    region_h, region_w = schedule.region_shape
    region = filtered_image[
        tile.row0 : tile.row0 + region_h, tile.col0 : tile.col0 + region_w
    ]
    image_fft: torch.Tensor = torch.fft.fft2(region, s=schedule.fft_shape)
    return image_fft


def _correlate_tile(
    image_fft: torch.Tensor,
    kernel_spectra: Sequence[torch.Tensor],
    tile: Tile,
    side: int,
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
# Direct engine: box gather + planar GEMM
# --------------------------------------------------------------------------- #


class _BoxSource:
    """Template-sized boxes under scheduled positions, filtered for one plane."""

    def __init__(
        self, bank: KernelBank, images: ImageSet, plane: int, dtype: torch.dtype
    ) -> None:
        self.images = images
        self.plane = plane
        self.dtype = dtype
        self.device = bank.device
        self.k_h, self.k_w = bank.template_shape
        self._filtered: dict[int, torch.Tensor] = {}
        self._ar_h = torch.arange(self.k_h, device=self.device)
        self._ar_w = torch.arange(self.k_w, device=self.device)

    def _filtered_stack(self, ids: torch.Tensor) -> torch.Tensor:
        """``(len(ids), H, W)`` filtered images, in ``ids`` order."""
        for i in ids.tolist():
            if int(i) in self._filtered:
                continue
            # One 2-D transform per image, never a batched one: cuFFT's batched
            # irfft2 of a 4096^2 image is off by 6e-3 (torch 2.11 / CUDA 13), the
            # 2-D transform by 4e-7, and the FFT engine filters images 2-D as well.
            raw = self.images.images[int(i)].to(self.device)
            filtered = apply_fourier_filter(
                raw, self.images.filter_plane(self.plane, int(i))
            )
            self._filtered[int(i)] = filtered.to(self.dtype)
            while len(self._filtered) > DIRECT_IMAGE_CACHE:
                self._filtered.pop(next(iter(self._filtered)))
        return torch.stack([self._filtered[int(i)] for i in ids.tolist()])

    def boxes(self, tile: Tile) -> torch.Tensor:
        """``(B, kH * kW)`` boxes of a direct-mode item, in column order."""
        assert (
            tile.images is not None and tile.rows is not None and tile.cols is not None
        )
        unique, inverse = torch.unique(tile.images, return_inverse=True)
        stack = self._filtered_stack(unique)  # (U, H, W)
        inverse = inverse.to(self.device)
        n = int(tile.images.numel())
        if tuple(stack.shape[-2:]) == (self.k_h, self.k_w):
            return stack[inverse].reshape(n, -1)
        rows = tile.rows.to(self.device)[:, None] + self._ar_h  # (B, kH)
        cols = tile.cols.to(self.device)[:, None] + self._ar_w  # (B, kW)
        boxes = stack[inverse[:, None, None], rows[:, :, None], cols[:, None, :]]
        return boxes.reshape(n, -1)


def _resident_planar(
    bank: KernelBank, rect: Rectangle, dtype: torch.dtype, fraction: float
) -> torch.Tensor | None:
    """``(2 n, kH kW)`` planar block of ``rect`` from the bank's tier, or ``None``."""
    if not bank.planar_fits(rect, dtype):
        return None
    if not bank.has_planar_block(rect, dtype):
        if bank.device.type == "cuda":
            # driver-level free memory plus what the caching allocator already holds
            # unused (after a stage's loop that can be tens of GiB the next allocation
            # reuses)
            free = int(torch.cuda.mem_get_info(bank.device)[0]) + int(
                torch.cuda.memory_reserved(bank.device)
                - torch.cuda.memory_allocated(bank.device)
            )
        else:
            free = 2**30  # CPU: a 1 GiB working set at most
        if fraction <= 0 or bank.planar_bytes_of(rect, dtype) > fraction * free:
            return None
    block = bank.planar_block(rect, dtype)
    return block.reshape(2 * block.shape[1], block.shape[2])


def featurize_direct(
    bank: KernelBank,
    images: ImageSet,
    group: ColumnGroup,
    writer: BlockWriter,
    schedule: TileSchedule,
    *,
    chunk: int = 64,
    stream: torch.cuda.Stream | None = None,
    planar_fraction: float = DIRECT_PLANAR_FRACTION,
) -> None:
    """``Y[p, cell] = sum_ij box_p[ij] conj(K_cell[ij])`` by one GEMM per position batch.

    The GEMM runs at the group's precision: fp32 (exact), tf32 (tensor cores, ~5e-4
    relative error per product) or fp16 operands with fp32 accumulation (~4e-4
    relative error on the features). With the rectangle's planar block resident in
    the kernel bank (built once; see :data:`DIRECT_PLANAR_FRACTION` and
    ``KernelBank.planar_block``) every batch is a single GEMM over all cells;
    otherwise the cells are swept per batch in chunks of at least ``chunk`` cells,
    widened to :data:`DIRECT_PLANAR_CHUNK_BYTES` of planar kernels.
    """
    if schedule.mode != "direct":
        raise ValueError("featurize_direct needs a direct-mode schedule")
    cells = group_cells(group)
    n_cells = int(cells.shape[0])
    precision = group.precision
    dtype = torch.float16 if precision is Precision.FP16 else torch.float32
    k_h, k_w = bank.template_shape
    planar_cell_bytes = 2 * k_h * k_w * torch.empty((), dtype=dtype).element_size()
    chunk = max(int(chunk), DIRECT_PLANAR_CHUNK_BYTES // planar_cell_bytes)
    with _stream_scope(stream):
        planar = _resident_planar(bank, group.rect, dtype, planar_fraction)
        for plane in group.planes:
            source = _BoxSource(bank, images, plane, dtype)
            for batch in schedule.items:
                boxes = source.boxes(batch)  # (B, kH kW)
                columns = (batch.columns.start, batch.columns.stop)
                if planar is not None:
                    real_imag = _matmul(planar, boxes.T, precision).to(torch.float32)
                    real_imag = real_imag.view(2, n_cells, -1)
                    features = torch.complex(real_imag[0], real_imag[1])
                    writer.write(slice(0, n_cells), columns, features, plane=plane)
                    continue
                for cell_range in cell_chunks(n_cells, chunk):
                    features = _inner_products(
                        bank, cells[cell_range], boxes, precision
                    )
                    writer.write(cell_range, columns, features, plane=plane)
    writer.close()


def _inner_products(
    bank: KernelBank, cells: torch.Tensor, boxes: torch.Tensor, precision: Precision
) -> torch.Tensor:
    """``(n_cells, B)`` complex inner products of boxes with the cells' kernels."""
    dtype = torch.float16 if precision is Precision.FP16 else torch.float32
    planar = bank.planar(cells, dtype)  # (2, n_cells, kH*kW): [Re K ; -Im K]
    rhs = boxes.T.to(dtype)  # no copy when the caller already cast the boxes
    real_imag = _matmul(planar, rhs, precision).to(torch.float32)  # (2, n_cells, B)
    return torch.complex(real_imag[0], real_imag[1])
