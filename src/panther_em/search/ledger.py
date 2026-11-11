"""The search's working state: dense per-pixel labels, statistics and scores."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from panther_em.search.decision import PixelLabel
from panther_em.search.images import PixelSelection

if TYPE_CHECKING:
    from panther_em.search.correlation.maps import StageMaps
    from panther_em.search.hypotheses import HypothesisSet
    from panther_em.search.images import SearchGeometry


class PixelLedger:
    """Dense ``(n_pixels,)`` per-pixel state, written once per stage.

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
        "score",
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
        self.score = torch.full((n,), nan, dtype=torch.float32, device=self.device)
        self.n_psi_of_stage: list[int] = []
        if initial is not None and not initial.is_full:
            self.label.fill_(int(PixelLabel.MASKED))
            self.label[initial.index.to(self.device)] = int(PixelLabel.PENDING)

    @property
    def n_pixels(self) -> int:
        """Pixels in the geometry."""
        return self.geometry.n_pixels

    @torch.no_grad()
    def write(
        self,
        maps: StageMaps,
        pixel_index: torch.Tensor,
        stage_index: int,
        hyps: HypothesisSet,
    ) -> None:
        """Record a stage's maps at ``pixel_index`` (same order as the maps).

        Written pixels are labelled AMBIGUOUS and their score is cleared; the stage's
        decision relabels (and scores) them.
        """
        idx = pixel_index.to(self.device, torch.long)
        self.mip[idx] = maps.mip.to(self.device)
        self.mean[idx] = maps.mean.to(self.device)
        self.variance[idx] = maps.variance.to(self.device)
        self.zscore[idx] = maps.zscore.to(self.device)
        self.best_index[idx] = hyps.decode(maps.best_virtual).to(self.device)
        self.best_psi[idx] = maps.best_psi.to(self.device)
        self.last_stage[idx] = int(stage_index)
        self.label[idx] = int(PixelLabel.AMBIGUOUS)
        self.score[idx] = float("nan")
        while len(self.n_psi_of_stage) <= stage_index:
            self.n_psi_of_stage.append(0)
        self.n_psi_of_stage[stage_index] = int(hyps.n_psi)

    def set_labels(self, pixel_index: torch.Tensor, labels: torch.Tensor) -> None:
        """Overwrite labels at ``pixel_index``."""
        self.label[pixel_index.to(self.device, torch.long)] = labels.to(
            self.device, torch.int8
        )

    def set_scores(self, pixel_index: torch.Tensor, scores: torch.Tensor) -> None:
        """Overwrite the threshold test's distances at ``pixel_index``."""
        self.score[pixel_index.to(self.device, torch.long)] = scores.to(
            self.device, torch.float32
        )

    def pixels_with(self, *labels: PixelLabel) -> torch.Tensor:
        """Sorted flat indices of pixels carrying any of ``labels``."""
        mask = torch.zeros_like(self.label, dtype=torch.bool)
        for lab in labels:
            mask |= self.label == int(lab)
        return torch.nonzero(mask, as_tuple=False).squeeze(-1)

    def next_stage_pixels(self) -> PixelSelection:
        """The pixels a following stage re-scores: AMBIGUOUS and ACCEPTED ones."""
        return PixelSelection(
            self.pixels_with(PixelLabel.AMBIGUOUS, PixelLabel.ACCEPTED), self.geometry
        )

    def std(self) -> torch.Tensor:
        """``sqrt(max(variance, 0))``."""
        return self.variance.clamp_min(0.0).sqrt()

    @torch.no_grad()
    def absorb_near_accepted(self, radius: int) -> None:
        """Relabel AMBIGUOUS pixels within ``radius`` of an ACCEPTED one as ABSORBED.

        Euclidean disk, per image (never across images); a no-op in center mode.
        """
        if radius <= 0 or self.geometry.mode != "grid":
            return
        accepted = (self.label == int(PixelLabel.ACCEPTED)).to(torch.float32)
        grid = self.geometry.unflatten(accepted).unsqueeze(1)  # (B, 1, H, W)
        offsets = torch.arange(
            -radius, radius + 1, device=self.device, dtype=torch.float32
        )
        disk = (offsets[:, None] ** 2 + offsets[None, :] ** 2) <= radius**2
        kernel = disk.to(torch.float32)[None, None]
        near = F.conv2d(grid, kernel, padding=radius).squeeze(1).reshape(-1) > 0
        ambiguous = self.label == int(PixelLabel.AMBIGUOUS)
        self.label[near & ambiguous] = int(PixelLabel.ABSORBED)
