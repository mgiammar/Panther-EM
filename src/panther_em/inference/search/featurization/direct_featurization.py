"""Direct featurization: the template-sized box under each position, contracted.

:class:`DirectFeaturization` gathers the box under every position (a sliding-window
view of each plane indexed by the box origins) and hands batches of boxes to a
:class:`BoxContraction`, which turns ``(N, kH kW)`` boxes into ``(n_cells, N)``
features. The cost is per position, so it suits sparse selections.

Two contractions compute the same features:

``AdjointContraction`` (default)
    Scatters each box onto the padded polar grid through the transposed kernel warp
    and contracts it with the rectangle's prefiltered polar vectors
    (:mod:`~panther_em.approximation.adjoint`).
``PlanarContraction`` (fallback)
    Multiplies the boxes by the ``[Re K; -Im K]`` Cartesian kernels; used where the
    warp has no adjoint (``Reconstructor.adjoint`` is ``None``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Literal

import torch

from panther_em.inference.search.correlation.contraction import Precision
from panther_em.inference.search.featurization.base import (
    BaseImageFeaturization,
    FeatureChunk,
    rectangle_cells,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from panther_em.approximation.reconstructor import Reconstructor
    from panther_em.decomposition.rectangles import Rectangle

ContractionName = Literal["adjoint", "planar"]

DEFAULT_PRECISION = Precision.FP16
DEFAULT_BATCH = 1024  # Positions per chunk

ADJOINT_CHUNK = 64  # Boxes per chunk, adjoint contraction (scattered, 2 MiB at 512^2)
PLANAR_FRACTION = 0.6
PLANAR_CHUNK_BYTES = 2**30


def _operand_dtype(precision: Precision, device: torch.device) -> torch.dtype:
    """fp16 for ``FP16`` on CUDA, else fp32."""
    if precision is Precision.FP16 and device.type == "cuda":
        return torch.float16
    return torch.float32


def _matmul(a: torch.Tensor, b: torch.Tensor, precision: Precision) -> torch.Tensor:
    """``a @ b`` in float32 (fp16 operands accumulate in fp32; tf32 when asked)."""
    if a.dtype is torch.float16 and a.is_cuda:
        return torch.mm(a, b, out_dtype=torch.float32)

    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = precision is Precision.TF32 and a.is_cuda

    try:
        return (a @ b).to(torch.float32)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _free_device_bytes(device: torch.device) -> int:
    """Driver free memory plus what the caching allocator holds unused (CPU: 1 GiB)."""
    if device.type != "cuda":
        return 2**30

    free = int(torch.cuda.mem_get_info(device)[0])

    return free + int(
        torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    )


# --------------------------------------------------------------------------- #
# Box contractions
# --------------------------------------------------------------------------- #


class BoxContraction(ABC):
    """``(N, kH kW)`` real boxes -> ``(n_cells, N)`` complex64 features of a rectangle.

    ``features[cell, n] = sum_ij box_n[ij] * conj(K_cell[ij])`` with cells row-major
    ``(k, m)``.

    Parameters
    ----------
    reconstructor : Reconstructor
    rect : Rectangle
        ``(k_start, m_start, k_extent, m_extent)``.
    precision : Precision or str
        Operand precision (see :data:`DEFAULT_PRECISION`).
    """

    def __init__(
        self,
        reconstructor: Reconstructor,
        rect: Rectangle,
        precision: Precision | str = DEFAULT_PRECISION,
    ) -> None:
        self.reconstructor = reconstructor
        self.rect: Rectangle = rect
        self.precision = Precision.coerce(precision)

    @property
    def n_cells(self) -> int:
        """Cells of the rectangle."""
        return self.rect[2] * self.rect[3]

    @property
    def box_dtype(self) -> torch.dtype:
        """The dtype boxes are best gathered in (the planes are cast once to it)."""
        return _operand_dtype(self.precision, self.reconstructor.device)

    @abstractmethod
    def features(self, boxes: torch.Tensor) -> torch.Tensor:
        """``(n_cells, N)`` complex64 features of ``(N, kH kW)`` boxes."""


class AdjointContraction(BoxContraction):
    """Features through the transposed kernel warp; no Cartesian kernels.

    The rectangle's bases are built at construction and held by the contraction.
    Raises ``ValueError`` when the warp has no adjoint (see
    :func:`~panther_em.approximation.adjoint.unsupported_reason`).
    """

    def __init__(
        self,
        reconstructor: Reconstructor,
        rect: Rectangle,
        precision: Precision | str = DEFAULT_PRECISION,
    ) -> None:
        super().__init__(reconstructor, rect, precision)
        adjoint = reconstructor.adjoint
        if adjoint is None:
            raise ValueError(
                "the template's warp has no adjoint (use PlanarContraction)"
            )
        self.adjoint = adjoint
        self.bases = reconstructor.adjoint_bases(rect)

    def features(self, boxes: torch.Tensor) -> torch.Tensor:
        """See :meth:`BoxContraction.features` (:data:`ADJOINT_CHUNK` boxes a time)."""
        n = int(boxes.shape[0])
        out = torch.empty(
            (self.n_cells, n), dtype=torch.complex64, device=self.reconstructor.device
        )
        for start in range(0, n, ADJOINT_CHUNK):
            part = boxes[start : start + ADJOINT_CHUNK]
            polar = self.adjoint.apply(part, self.precision)
            out[:, start : start + int(part.shape[0])] = self.bases.features(
                polar, self.precision
            )
        return out


class PlanarContraction(BoxContraction):
    """Features as a GEMM with the ``[Re K; -Im K]`` Cartesian kernels.

    When the rectangle's planar block fits :data:`PLANAR_FRACTION` of the free
    memory it is built on first use and held by the contraction, and a batch is one
    GEMM over every cell; otherwise every batch rebuilds the kernels in chunks of
    :data:`PLANAR_CHUNK_BYTES`.
    """

    @cached_property
    def cell_bytes(self) -> int:
        """Bytes of one cell's ``[Re K; -Im K]`` rows."""
        k_h, k_w = self.reconstructor.box_shape
        return 2 * k_h * k_w * torch.empty((), dtype=self.box_dtype).element_size()

    @cached_property
    def block(self) -> torch.Tensor | None:
        """``(2 n_cells, kH kW)`` planar block (``[Re K; -Im K]``), or ``None``."""
        needed = self.n_cells * self.cell_bytes
        if needed > PLANAR_FRACTION * _free_device_bytes(self.reconstructor.device):
            return None
        n = self.n_cells
        k_h, k_w = self.reconstructor.box_shape
        block = torch.empty(
            (2 * n, k_h * k_w), dtype=self.box_dtype, device=self.reconstructor.device
        )
        for start in range(0, n, self.cell_chunk):
            stop = min(start + self.cell_chunk, n)
            kernels = self.reconstructor.kernels(self.cells[start:stop]).reshape(
                stop - start, -1
            )
            block[start:stop] = kernels.real.to(self.box_dtype)
            block[n + start : n + stop] = (-kernels.imag).to(self.box_dtype)
        return block

    @cached_property
    def cell_chunk(self) -> int:
        """Cells per chunk when the kernels are built per batch."""
        return max(1, PLANAR_CHUNK_BYTES // self.cell_bytes)

    @cached_property
    def cells(self) -> torch.Tensor:
        """``(n_cells, 2)`` row-major ``(k, m)`` cells."""
        return rectangle_cells(self.rect)

    def features(self, boxes: torch.Tensor) -> torch.Tensor:
        """See :meth:`BoxContraction.features`."""
        rhs = boxes.T.to(self.box_dtype)
        if self.block is not None:
            return self._complex(_matmul(self.block, rhs, self.precision))
        out = torch.empty(
            (self.n_cells, int(boxes.shape[0])),
            dtype=torch.complex64,
            device=self.reconstructor.device,
        )
        for start in range(0, self.n_cells, self.cell_chunk):
            cells = self.cells[start : start + self.cell_chunk]
            kernels = self.reconstructor.kernels(cells).reshape(int(cells.shape[0]), -1)
            lhs = torch.cat([kernels.real, -kernels.imag]).to(self.box_dtype)
            out[start : start + int(cells.shape[0])] = self._complex(
                _matmul(lhs, rhs, self.precision)
            )
        return out

    @staticmethod
    def _complex(real_imag: torch.Tensor) -> torch.Tensor:
        """``(2 c, N)`` stacked ``[Re; Im]`` rows -> ``(c, N)`` complex64."""
        re, im = real_imag.view(2, -1, real_imag.shape[-1])
        return torch.complex(re, im)


def box_contraction(
    reconstructor: Reconstructor,
    rect: Rectangle,
    precision: Precision | str = DEFAULT_PRECISION,
    contraction: ContractionName = "adjoint",
) -> BoxContraction:
    """The adjoint contraction, or the planar one if asked or the adjoint is missing."""
    if contraction not in ("adjoint", "planar"):
        raise ValueError(f"unknown contraction {contraction!r}")
    if contraction == "adjoint" and reconstructor.adjoint is not None:
        return AdjointContraction(reconstructor, rect, precision)
    return PlanarContraction(reconstructor, rect, precision)


# --------------------------------------------------------------------------- #
# The featurization
# --------------------------------------------------------------------------- #


class DirectFeaturization(BaseImageFeaturization):
    """Box gather + :class:`BoxContraction` at every position.

    Yields one :class:`~.base.FeatureChunk` per plane and batch of ``batch`` positions,
    covering every cell.

    Parameters
    ----------
    reconstructor : Reconstructor
        The template's forward model (kernels and their adjoint).
    rect : Rectangle
        ``(k_start, m_start, k_extent, m_extent)``.
    batch : int, optional
        Positions per chunk.
    precision : Precision or str, optional
        Operand precision of the contraction (see :data:`DEFAULT_PRECISION`).
    contraction : {"adjoint", "planar"}, optional
        ``"adjoint"`` falls back to ``"planar"`` where the warp has no adjoint.
    """

    def __init__(
        self,
        reconstructor: Reconstructor,
        rect: Sequence[int],
        *,
        batch: int = DEFAULT_BATCH,
        precision: Precision | str = DEFAULT_PRECISION,
        contraction: ContractionName = "adjoint",
    ) -> None:
        super().__init__(reconstructor, rect)
        if int(batch) < 1:
            raise ValueError("batch must be >= 1")
        self.batch = int(batch)
        self.contraction = box_contraction(
            reconstructor, self.rect, precision, contraction
        )

    def chunks(
        self, planes: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor
    ) -> Iterator[FeatureChunk]:
        """See :meth:`~.base.BaseImageFeaturization.chunks`."""
        self.validate(planes, rows, cols)
        n_positions = int(rows.numel())
        if n_positions == 0:
            return
        device, (k_h, k_w) = self.device, self.box_shape
        rows, cols = rows.to(device), cols.to(device)
        cells = slice(0, self.n_cells)
        for d, plane in enumerate(planes):
            # every box row is one contiguous read of the window view: 13x faster than
            # a three-index gather (0.6 vs 8 ms per 512 boxes of 512^2)
            image = plane.to(device=device, dtype=self.contraction.box_dtype)
            windows = image.unfold(0, k_h, 1).unfold(1, k_w, 1)
            for start in range(0, n_positions, self.batch):
                stop = min(start + self.batch, n_positions)
                boxes = windows[rows[start:stop], cols[start:stop]].reshape(
                    stop - start, k_h * k_w
                )
                yield FeatureChunk(
                    d, cells, slice(start, stop), self.contraction.features(boxes)
                )
