"""Dense per-pixel state of a multi-stage search: labels and last-known statistics."""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from panther_em.inference.search.types import PixelSelection

if TYPE_CHECKING:
    from panther_em.inference.search.executor import StageMaps
    from panther_em.inference.search.hypotheses import HypothesisSet
    from panther_em.inference.search.types import SearchGeometry


class PixelLabel(enum.IntEnum):
    """Per-pixel status across search stages.

    - PENDING: not yet searched.
    - UNCERTAIN: searched, but undecided (still eligible for the next stage).
    - REJECTED: decided below the detection threshold.
    - ACCEPTED: decided above the detection threshold.
    - ABSORBED: was UNCERTAIN but lies near an ACCEPTED pixel and was absorbed by
      morphological dilation.
    - MASKED: excluded by the caller's initial mask; never searched.
    """

    PENDING = 0
    UNCERTAIN = 1
    REJECTED = 2
    ACCEPTED = 3
    ABSORBED = 4
    MASKED = 5


class PixelLedger:
    """Dense ``(n_pixels,)`` per-pixel state, written once per stage.

    Statistics are *per stage* (a different rank is a different estimator), so the
    last stage to search a pixel overwrites its values. Hypothesis ids are stored as
    **global** ids (see :mod:`~panther_em.inference.search.hypotheses`).

    Parameters
    ----------
    geometry : SearchGeometry
        The valid-correlation grid.
    device : torch.device or str
        Where the state lives.
    initial : PixelSelection, optional
        Pixels eligible for the search; everything else is ``MASKED``.
    """

    FIELDS = (
        "mip",
        "mean",
        "variance",
        "zscore",
        "best_index",
        "best_psi",
        "last_stage",
    )

    def __init__(
        self,
        geometry: SearchGeometry,
        device: torch.device | str,
        *,
        initial: PixelSelection | None = None,
    ) -> None:
        self.geometry = geometry
        self.device = torch.device(device)
        n = geometry.n_pixels
        nan = float("nan")
        self.label = torch.full(
            (n,), int(PixelLabel.PENDING), dtype=torch.int8, device=self.device
        )
        self.mip = torch.full((n,), nan, dtype=torch.float32, device=self.device)
        self.mean = torch.full((n,), nan, dtype=torch.float32, device=self.device)
        self.variance = torch.full((n,), nan, dtype=torch.float32, device=self.device)
        self.zscore = torch.full((n,), nan, dtype=torch.float32, device=self.device)
        self.best_index = torch.full((n,), -1, dtype=torch.int64, device=self.device)
        self.best_psi = torch.full((n,), -1, dtype=torch.int64, device=self.device)
        self.last_stage = torch.full((n,), -1, dtype=torch.int16, device=self.device)
        self.n_psi_of_stage: list[int] = []
        if initial is not None and not initial.is_full:
            self.label.fill_(int(PixelLabel.MASKED))
            self.label[initial.index.to(self.device)] = int(PixelLabel.PENDING)

    @property
    def n_pixels(self) -> int:
        """Pixels in the geometry."""
        return self.geometry.n_pixels

    # -- writing ----------------------------------------------------------------

    @torch.no_grad()
    def write(
        self,
        maps: StageMaps,
        pixel_index: torch.Tensor,
        stage_index: int,
        hyp_plan: HypothesisSet,
    ) -> None:
        """Record a stage's maps at ``pixel_index`` (same order as the maps)."""
        idx = pixel_index.to(self.device, torch.long)
        self.mip[idx] = maps.mip.to(self.device)
        self.mean[idx] = maps.mean.to(self.device)
        self.variance[idx] = maps.variance.to(self.device)
        self.zscore[idx] = maps.zscore.to(self.device)
        self.best_index[idx] = hyp_plan.decode(maps.best_virtual).to(self.device)
        self.best_psi[idx] = maps.best_psi.to(self.device)
        self.last_stage[idx] = int(stage_index)
        self.label[idx] = int(PixelLabel.UNCERTAIN)
        while len(self.n_psi_of_stage) <= stage_index:
            self.n_psi_of_stage.append(0)
        self.n_psi_of_stage[stage_index] = int(hyp_plan.n_psi)

    def set_labels(self, pixel_index: torch.Tensor, labels: torch.Tensor) -> None:
        """Overwrite labels at ``pixel_index``."""
        self.label[pixel_index.to(self.device, torch.long)] = labels.to(
            self.device, torch.int8
        )

    # -- reading ----------------------------------------------------------------

    def pixels_with(self, *labels: PixelLabel) -> torch.Tensor:
        """Sorted flat indices of pixels carrying any of ``labels``."""
        mask = torch.zeros_like(self.label, dtype=torch.bool)
        for lab in labels:
            mask |= self.label == int(lab)
        return torch.nonzero(mask, as_tuple=False).squeeze(-1)

    def uncertain(self) -> PixelSelection:
        """The next stage's pixel set."""
        return PixelSelection(self.pixels_with(PixelLabel.UNCERTAIN), self.geometry)

    def std(self) -> torch.Tensor:
        """``sqrt(max(variance, 0))``."""
        return self.variance.clamp_min(0.0).sqrt()

    def grid(self, name: str) -> torch.Tensor:
        """A field as a dense ``(B, *grid)`` map."""
        return self.geometry.unflatten(getattr(self, name))

    # -- neighbourhood ----------------------------------------------------------

    @torch.no_grad()
    def absorb_near_accepted(self, radius: int) -> None:
        """Relabel ``UNCERTAIN`` pixels near an ``ACCEPTED`` one as ``ABSORBED``."""
        if radius <= 0 or self.geometry.mode != "grid":
            return
        accepted = (self.label == int(PixelLabel.ACCEPTED)).to(torch.float32)
        grid = self.geometry.unflatten(accepted).unsqueeze(1)  # (B, 1, H, W)
        pooled = F.max_pool2d(
            grid, kernel_size=2 * radius + 1, stride=1, padding=radius
        )
        offsets = torch.arange(
            -radius, radius + 1, device=self.device, dtype=torch.float32
        )
        disk = (offsets[:, None] ** 2 + offsets[None, :] ** 2) <= radius**2
        if not bool(disk.all()):
            # Square pooling over-reaches at the corners; redo with a disk via conv.
            kernel = disk.to(torch.float32)[None, None]
            pooled = F.conv2d(grid, kernel, padding=radius).clamp_max_(1.0)
        near = pooled.squeeze(1).reshape(-1) > 0
        uncertain = self.label == int(PixelLabel.UNCERTAIN)
        self.label[near & uncertain] = int(PixelLabel.ABSORBED)
