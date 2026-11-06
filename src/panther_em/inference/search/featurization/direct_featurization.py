"""Direct featurization: the template-sized box under each position, contracted.

:class:`DirectFeaturization` gathers the box under every position (a sliding-window
view of each plane indexed by the box origins) and hands batches of boxes to a
:class:`BoxContraction`, which turns ``(N, kH kW)`` boxes into ``(n_cells, N)``
features. The cost is per position, so it suits sparse selections.

Two contractions compute the same features:

``AdjointContraction`` (default)
    Scatters each box onto the padded polar grid through the transposed kernel warp
    and contracts it with the rectangle's prefiltered polar vectors
    (:mod:`~panther_em.inference.search.adjoint`).
``PlanarContraction`` (fallback)
    Multiplies the boxes by the ``[Re K; -Im K]`` Cartesian kernels; used where the
    bank cannot mirror its warp (``KernelBank.adjoint()`` is ``None``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Literal

import torch

from panther_em.inference.search.featurization.base import (
    BaseImageFeaturization,
    FeatureChunk,
    rectangle_cells,
)
from panther_em.inference.search.types import Precision

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from panther_em.inference.search.template import KernelBank
    from panther_em.inference.search.types import Rectangle

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
    kernels : KernelBank
    rect : Rectangle
        ``(k_start, m_start, k_extent, m_extent)``.
    precision : Precision or str
        Operand precision (see :data:`DEFAULT_PRECISION`).
    """

    def __init__(
        self,
        kernels: KernelBank,
        rect: Rectangle,
        precision: Precision | str = DEFAULT_PRECISION,
    ) -> None:
        self.kernels = kernels
        self.rect: Rectangle = rect
        self.precision = Precision.coerce(precision)

    @property
    def n_cells(self) -> int:
        """Cells of the rectangle."""
        return self.rect[2] * self.rect[3]

    @property
    def box_dtype(self) -> torch.dtype:
        """The dtype boxes are best gathered in (the planes are cast once to it)."""
        return _operand_dtype(self.precision, self.kernels.device)

    @abstractmethod
    def features(self, boxes: torch.Tensor) -> torch.Tensor:
        """``(n_cells, N)`` complex64 features of ``(N, kH kW)`` boxes."""


class AdjointContraction(BoxContraction):
    """Features through the transposed kernel warp; no Cartesian kernels.

    Raises ``ValueError`` when the bank cannot mirror its warp (see
    :func:`~panther_em.inference.search.adjoint.unsupported_reason`).
    """

    def __init__(
        self,
        kernels: KernelBank,
        rect: Rectangle,
        precision: Precision | str = DEFAULT_PRECISION,
    ) -> None:
        super().__init__(kernels, rect, precision)
        adjoint = kernels.adjoint()
        if adjoint is None:
            raise ValueError(
                "the kernel bank's warp has no adjoint (use PlanarContraction)"
            )
        self.adjoint = adjoint
        self.bases = kernels.adjoint_bases(rect)

    def features(self, boxes: torch.Tensor) -> torch.Tensor:
        """See :meth:`BoxContraction.features` (:data:`ADJOINT_CHUNK` boxes a time)."""
        n = int(boxes.shape[0])
        out = torch.empty(
            (self.n_cells, n), dtype=torch.complex64, device=self.kernels.device
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

    With the rectangle's planar block resident in the bank (built once, see
    :data:`PLANAR_FRACTION`) a batch is one GEMM over every cell; otherwise the cells
    are swept in chunks of :data:`PLANAR_CHUNK_BYTES` of kernels.
    """

    @cached_property
    def block(self) -> torch.Tensor | None:
        """``(2 n_cells, kH kW)`` resident planar block, or ``None`` (chunked)."""
        bank, dtype = self.kernels, self.box_dtype
        if not bank.planar_fits(self.rect, dtype):
            return None
        if not bank.has_planar_block(self.rect, dtype):
            needed = bank.planar_bytes_of(self.rect, dtype)
            if needed > PLANAR_FRACTION * _free_device_bytes(bank.device):
                return None
        block = bank.planar_block(self.rect, dtype)
        return block.reshape(2 * block.shape[1], block.shape[2])

    @cached_property
    def cell_chunk(self) -> int:
        """Cells per chunk when the block is not resident."""
        k_h, k_w = self.kernels.template_shape
        per_cell = 2 * k_h * k_w * torch.empty((), dtype=self.box_dtype).element_size()
        return max(1, PLANAR_CHUNK_BYTES // per_cell)

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
            device=self.kernels.device,
        )
        for start in range(0, self.n_cells, self.cell_chunk):
            cells = self.cells[start : start + self.cell_chunk]
            planar = self.kernels.planar(cells, self.box_dtype)  # (2, c, kH kW)
            lhs = planar.reshape(2 * int(cells.shape[0]), -1)
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
    kernels: KernelBank,
    rect: Rectangle,
    precision: Precision | str = DEFAULT_PRECISION,
    contraction: ContractionName = "adjoint",
) -> BoxContraction:
    """The adjoint contraction, or the planar one if asked or the adjoint is missing."""
    if contraction not in ("adjoint", "planar"):
        raise ValueError(f"unknown contraction {contraction!r}")
    if contraction == "adjoint" and kernels.adjoint() is not None:
        return AdjointContraction(kernels, rect, precision)
    return PlanarContraction(kernels, rect, precision)


# --------------------------------------------------------------------------- #
# The featurization
# --------------------------------------------------------------------------- #


class DirectFeaturization(BaseImageFeaturization):
    """Box gather + :class:`BoxContraction` at every position.

    Yields one :class:`~.base.FeatureChunk` per plane and batch of ``batch`` positions,
    covering every cell.

    Parameters
    ----------
    kernels : KernelBank
        The template's kernel bank.
    rect : Rectangle
        ``(k_start, m_start, k_extent, m_extent)``.
    batch : int, optional
        Positions per chunk.
    precision : Precision or str, optional
        Operand precision of the contraction (see :data:`DEFAULT_PRECISION`).
    contraction : {"adjoint", "planar"}, optional
        ``"adjoint"`` falls back to ``"planar"`` where the bank has no adjoint.
    """

    def __init__(
        self,
        kernels: KernelBank,
        rect: Sequence[int],
        *,
        batch: int = DEFAULT_BATCH,
        precision: Precision | str = DEFAULT_PRECISION,
        contraction: ContractionName = "adjoint",
    ) -> None:
        super().__init__(kernels, rect)
        if int(batch) < 1:
            raise ValueError("batch must be >= 1")
        self.batch = int(batch)
        self.contraction = box_contraction(kernels, self.rect, precision, contraction)

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
