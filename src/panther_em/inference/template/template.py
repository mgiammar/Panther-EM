"""A decomposition plus everything derived from it that its consumers share."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import torch

from panther_em.decomposition.result import DecompositionBase
from panther_em.inference.template.warp import TemplateWarp
from panther_em.inference.template.weights import WeightSource

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from panther_em.coordinates.warp_config import WarpConfig
    from panther_em.decomposition.rectangles import Rectangle
    from panther_em.inference.search.hypotheses import HypothesisSpace


class Template:
    """A decomposition, its kernel warp and its weights.

    Parameters
    ----------
    result : DecompositionBase
        The (dense or sparse) decomposition, with its coordinate transform.
    device : torch.device or str
        Compute device for kernels and weights.
    kernel : WarpConfig, optional
        Polar-to-Cartesian warp settings (default ``WarpConfig()``).
    weights_cells : Rectangle or sequence of Rectangle, optional
        Bounding box of the feature space any stage will use; restricts the resident
        ``U * S`` (see :class:`~.weights.WeightSource`).
    """

    def __init__(
        self,
        result: DecompositionBase,
        *,
        device: torch.device | str,
        kernel: WarpConfig | None = None,
        weights_cells: Rectangle | Sequence[Rectangle] | None = None,
    ) -> None:
        self.result = result
        self.device = torch.device(device)
        self.warp = TemplateWarp(result, kernel, device=self.device)
        self._weights_cells = weights_cells
        self._weights: WeightSource | None = None

    @property
    def kernel_config(self) -> WarpConfig:
        """The warp settings of every kernel, adjoint and error table."""
        return self.warp.config

    @property
    def template_shape(self) -> tuple[int, int]:
        """Kernel box ``(kH, kW)``."""
        return self.warp.box_shape

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
        """Device-resident ``U * S`` (built on first access)."""
        if self._weights is None:
            self._weights = WeightSource(self.result, self.device, self._weights_cells)
        return self._weights

    @cached_property
    def space(self) -> HypothesisSpace:
        """The decomposition's hypothesis space (one defocus plane)."""
        from panther_em.inference.search.hypotheses import HypothesisSpace

        return HypothesisSpace.from_result(self.result)

    @classmethod
    def from_hdf5(
        cls,
        path: str | Path,
        device: torch.device | str,
        *,
        kernel: WarpConfig | None = None,
        weights_cells: Rectangle | Sequence[Rectangle] | None = None,
    ) -> Template:
        """Load a dense or sparse decomposition and wrap it."""
        result = DecompositionBase.load(path)
        return cls(result, device=device, kernel=kernel, weights_cells=weights_cells)

    def release(self) -> None:
        """Drop the resident weights and the warp's adjoint."""
        self._weights = None
        self.warp.release()
