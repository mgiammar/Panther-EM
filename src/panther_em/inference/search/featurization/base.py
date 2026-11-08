"""Base image featurization class for SVD-based template matching searches.

Given ``D`` real-space images, each a plane at a different defocus, and a rectangle of
``(k, m)`` cells of kernels, featurization may be expressed as:

    ``Y[d, cell, p] = sum_ij planes[d, rows[p] + i, cols[p] + j] * conj(K_cell[i, j])``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from panther_em.inference.search.correlation.layout import Rectangle
    from panther_em.inference.template import TemplateWarp


def rectangle_cells(rect: Rectangle) -> torch.Tensor:
    """``(k_extent * m_extent, 2)`` int64 ``(k, m)`` cells of ``rect``, row-major."""
    k0, m0, k_ext, m_ext = rect
    k = torch.arange(k0, k0 + k_ext)
    m = torch.arange(m0, m0 + m_ext)
    return torch.stack(torch.meshgrid(k, m, indexing="ij"), dim=-1).reshape(-1, 2)


@dataclass(frozen=True, eq=False)
class FeatureChunk:
    """Features of a block of cells at a block of positions, for one plane.

    Attributes
    ----------
    plane : int
        Index into the featurized planes' ``D`` axis.
    cells : slice
        ``[c0, c1)`` into the rectangle's row-major ``(k, m)`` cells.
    columns : slice
        ``[p0, p1)`` into the caller's positions.
    values : torch.Tensor
        Rectangular region ``(c1 - c0, p1 - p0)`` complex64 features
    """

    plane: int
    cells: slice
    columns: slice
    values: torch.Tensor

    def __post_init__(self) -> None:
        """Validate (never coerce)."""
        if self.plane < 0:
            raise ValueError("plane must be >= 0")

        for name, s in (("cells", self.cells), ("columns", self.columns)):
            if s.start is None or s.stop is None or s.step not in (None, 1):
                raise ValueError(f"{name} must be a [start, stop) slice")
            if not 0 <= s.start <= s.stop:
                raise ValueError(f"{name} must satisfy 0 <= start <= stop")

        shape = (
            self.cells.stop - self.cells.start,
            self.columns.stop - self.columns.start,
        )

        if (
            self.values.dtype is not torch.complex64
            or tuple(self.values.shape) != shape
        ):
            raise ValueError(
                f"values must be a complex64 {shape} tensor, got "
                f"{self.values.dtype} {tuple(self.values.shape)}"
            )


class BaseImageFeaturization(ABC):
    """Features of image planes at box positions against one rectangle's kernels.

    Subclasses implement :meth:`chunks`; :meth:`featurize` assembles the chunks into
    one tensor (tests and small inputs).

    Parameters
    ----------
    warp : TemplateWarp
        The template's kernel warp; features are computed on its device.
    rect : Rectangle
        ``(k_start, m_start, k_extent, m_extent)`` within the stored decomposition.
    """

    def __init__(self, warp: TemplateWarp, rect: Sequence[int]) -> None:
        if len(rect) != 4:
            raise ValueError(
                f"a rectangle is (k_start, m_start, k_extent, m_extent), got {rect!r}"
            )

        k0, m0, k_ext, m_ext = (int(v) for v in rect)

        if k0 < 0 or m0 < 0 or k_ext < 1 or m_ext < 1:
            raise ValueError(
                f"rectangle {tuple(rect)} needs starts >= 0 and extents >= 1"
            )

        result = warp.result

        if k0 + k_ext > int(result.k_max) or m0 + m_ext > int(result.eig_max):
            raise ValueError(
                f"rectangle {tuple(rect)} exceeds the stored (k_max={result.k_max}, "
                f"eig_max={result.eig_max}) feature space"
            )

        self.warp = warp
        self.rect: Rectangle = (k0, m0, k_ext, m_ext)

    # -- geometry ---------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Where features are computed (the warp's device)."""
        return self.warp.device

    @property
    def box_shape(self) -> tuple[int, int]:
        """Kernel box ``(kH, kW)``."""
        k_h, k_w = self.warp.box_shape
        return int(k_h), int(k_w)

    @property
    def n_cells(self) -> int:
        """Cells of the rectangle, ``k_extent * m_extent``."""
        return self.rect[2] * self.rect[3]

    @cached_property
    def cells(self) -> torch.Tensor:
        """``(n_cells, 2)`` int64 ``(k, m)`` of every cell, row-major."""
        return rectangle_cells(self.rect)

    # -- featurization ------------------------------------------------------------

    def validate(
        self, planes: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor
    ) -> None:
        """Raise unless ``planes`` and the box origins fit together.

        ``planes`` must be a real ``(D, H, W)`` floating tensor at least the box size;
        ``rows`` and ``cols`` 1-D int64 tensors of equal length with every box inside
        the planes (``0 <= rows <= H - kH``, ``0 <= cols <= W - kW``).
        """
        if planes.dim() != 3 or planes.is_complex() or not planes.is_floating_point():
            raise ValueError(
                f"planes must be a real floating (D, H, W) tensor, got "
                f"{planes.dtype} {tuple(planes.shape)}"
            )

        if planes.shape[0] < 1:
            raise ValueError("planes must hold at least one plane")

        (h, w), (k_h, k_w) = tuple(planes.shape[1:]), self.box_shape

        if h < k_h or w < k_w:
            raise ValueError(f"planes {(h, w)} are smaller than the box {(k_h, k_w)}")

        for name, t in (("rows", rows), ("cols", cols)):
            if t.dim() != 1 or t.dtype is not torch.int64:
                raise ValueError(f"{name} must be a 1-D int64 tensor")

        if rows.numel() != cols.numel():
            raise ValueError(
                f"rows and cols differ in length ({rows.numel()} vs {cols.numel()})"
            )
        if rows.numel() == 0:
            return

        if int(rows.min()) < 0 or int(rows.max()) > h - k_h:
            raise ValueError(f"rows must lie in [0, {h - k_h}] for {(h, w)} planes")

        if int(cols.min()) < 0 or int(cols.max()) > w - k_w:
            raise ValueError(f"cols must lie in [0, {w - k_w}] for {(h, w)} planes")

    @abstractmethod
    def chunks(
        self, planes: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor
    ) -> Iterator[FeatureChunk]:
        """Stream the features of every plane, cell and position.

        Parameters
        ----------
        planes : torch.Tensor
            ``(D, H, W)`` real image planes, already filtered (one per CTF plane).
        rows, cols : torch.Tensor
            ``(P,)`` int64 top-left box origins on the planes.

        Yields
        ------
        FeatureChunk
            Rectangles that together cover every ``(plane, cell, position)`` once.
        """

    def featurize(
        self, planes: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor
    ) -> torch.Tensor:
        """``(D, n_cells, P)`` complex64 features: :meth:`chunks` assembled."""
        out = torch.empty(
            (int(planes.shape[0]), self.n_cells, int(rows.numel())),
            dtype=torch.complex64,
            device=self.device,
        )
        for chunk in self.chunks(planes, rows, cols):
            out[chunk.plane, chunk.cells, chunk.columns] = chunk.values
        return out
