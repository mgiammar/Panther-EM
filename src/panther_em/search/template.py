"""The search's view of a decomposition: its forward model, weights and hypotheses."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import torch

from panther_em.approximation.reconstructor import Reconstructor
from panther_em.decomposition.result import DecompositionBase
from panther_em.search.hypotheses import HypothesisSpace
from panther_em.search.weights import WeightSource

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from panther_em.coordinates.warp_config import WarpConfig
    from panther_em.decomposition.rectangles import Rectangle


class Template:
    """A decomposition, its forward model and its device-resident weights.

    Parameters
    ----------
    result : DecompositionBase
        The (dense or sparse) decomposition of real-valued projections, with its
        coordinate transform.
    device : torch.device or str
        Compute device for kernels and weights.
    warp : WarpConfig, optional
        Polar-to-Cartesian warp settings (default: the decomposition's own).
    weights_cells : Rectangle or sequence of Rectangle, optional
        Bounding box of the feature space any stage will use; restricts the resident
        ``U * S`` (see :class:`~.weights.WeightSource`).
    """

    def __init__(
        self,
        result: DecompositionBase,
        *,
        device: torch.device | str,
        warp: WarpConfig | None = None,
        weights_cells: Rectangle | Sequence[Rectangle] | None = None,
    ) -> None:
        if result.is_complex_projection:
            raise NotImplementedError(
                "the search supports decompositions of real-valued projections only"
            )
        self.result = result
        self.device = torch.device(device)
        self.reconstructor = Reconstructor(result, warp, device=self.device)
        self._weights_cells = weights_cells
        self._weights: WeightSource | None = None
        self._row_scale: torch.Tensor | None = None

    @property
    def warp(self) -> WarpConfig:
        """The warp of every kernel, adjoint and error table of this template."""
        return self.reconstructor.warp

    @property
    def template_shape(self) -> tuple[int, int]:
        """Kernel box ``(kH, kW)``."""
        return self.reconstructor.box_shape

    @property
    def k_max(self) -> int:
        """Stored angular-frequency blocks."""
        return int(self.result.k_max)

    @property
    def eig_max(self) -> int:
        """Stored eigenvectors per frequency."""
        return int(self.result.eig_max)

    @property
    def n_templates(self) -> int:
        """Template rows, ``n_filters * n_orientations``."""
        return int(self.result.num_fourier_filters) * int(self.result.num_orientations)

    @property
    def weights(self) -> WeightSource:
        """Device-resident ``U * S`` (built on first access), rows scaled if set."""
        if self._weights is None:
            self._weights = WeightSource(self.result, self.device, self._weights_cells)
            if self._row_scale is not None:
                self._weights.us.mul_(self._row_scale_on(self.device))
        return self._weights

    @property
    def row_scale(self) -> torch.Tensor | None:
        """``(N,)`` float64 CPU factor every template row's weights carry, or ``None``.

        Set by :meth:`scale_rows` (after-CTF normalization,
        :mod:`panther_em.search.normalization`).
        """
        return self._row_scale

    def scale_rows(self, scale: torch.Tensor) -> None:
        """Multiply template row ``n``'s weights ``U * S`` by ``scale[n]``.

        The correlation of row ``n`` with any image scales by ``scale[n]``; every set
        derived from the weights (moment forms, seed templates, packed operands) is
        built from the scaled weights. Replaces a previous scale. Call before the
        first search (derived sets cached by a searcher are not rebuilt).
        """
        s = torch.as_tensor(scale, dtype=torch.float64).reshape(-1).cpu()
        if s.numel() != self.n_templates:
            raise ValueError(f"{s.numel()} scales for {self.n_templates} template rows")
        if not bool(torch.isfinite(s).all()) or bool((s <= 0).any()):
            raise ValueError("row scales must be finite and positive")
        if self._weights is not None:
            new = s.to(self.device, torch.float32)[:, None, None]
            if self._row_scale is not None:
                new = new / self._row_scale_on(self.device)
            self._weights.us.mul_(new)
        self._row_scale = s

    def _row_scale_on(self, device: torch.device) -> torch.Tensor:
        assert self._row_scale is not None
        return self._row_scale.to(device, torch.float32)[:, None, None]

    @cached_property
    def space(self) -> HypothesisSpace:
        """The decomposition's hypothesis space (one defocus plane)."""
        return HypothesisSpace.from_result(self.result)

    @classmethod
    def from_hdf5(
        cls,
        path: str | Path,
        device: torch.device | str,
        *,
        warp: WarpConfig | None = None,
        weights_cells: Rectangle | Sequence[Rectangle] | None = None,
    ) -> Template:
        """Load a dense or sparse decomposition and wrap it."""
        result = DecompositionBase.load(path)
        return cls(result, device=device, warp=warp, weights_cells=weights_cells)

    def release(self) -> None:
        """Drop the resident weights and the reconstructor's adjoint."""
        self._weights = None
        self.reconstructor.release()
