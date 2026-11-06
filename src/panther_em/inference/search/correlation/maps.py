"""Per-pixel maps of one correlation pass and their merge across plane passes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(eq=False)
class StageMaps:
    """Per-pixel results of one stage in the store's compact column order.

    Notes
    -----
    ``best_virtual`` holds *virtual* hypothesis ids (see
    :mod:`~panther_em.inference.search.hypotheses`); decode with ``hyps.decode``.
    ``best_psi`` indexes ``psi_degrees(n_psi)``.
    """

    mip: torch.Tensor
    mean: torch.Tensor
    variance: torch.Tensor
    zscore: torch.Tensor
    best_virtual: torch.Tensor
    best_psi: torch.Tensor
    n_evaluated: int

    @classmethod
    def empty(
        cls, n_pixels: int, device: torch.device | str, n_evaluated: int
    ) -> StageMaps:
        """NaN / -1 filled maps."""
        f = torch.full((n_pixels,), float("nan"), dtype=torch.float32, device=device)
        i = torch.full((n_pixels,), -1, dtype=torch.int64, device=device)
        return cls(f.clone(), f.clone(), f.clone(), f, i.clone(), i, n_evaluated)

    def as_targets(self) -> dict[str, torch.Tensor]:
        """Keyed like ``PixelStats.finalize`` for :meth:`BlockStats.finalize_into`."""
        return {
            "mip": self.mip,
            "mean": self.mean,
            "variance": self.variance,
            "zscore": self.zscore,
            "best_index": self.best_virtual,
            "best_psi": self.best_psi,
        }

    @property
    def n_pixels(self) -> int:
        """``P``."""
        return int(self.mip.numel())


def merge_stage_maps(parts: Sequence[StageMaps], n_selected: int) -> StageMaps:
    """Maps of a stage searched plane by plane, from the single-plane maps.

    ``parts[d_local]`` searched the stage's ``d_local``-th plane over the same
    ``n_selected`` template rows, so its virtual ids ``j`` become ``d_local *
    n_selected + j`` in the stage's own enumeration (planes outer). The maximum and
    its hypothesis are taken across planes, the mean and variance pooled (every plane
    evaluated the same number of hypotheses), and the z-score recomputed, exactly as
    the fused kernel would have accumulated them in one pass.
    """
    if not parts:
        raise ValueError("merge_stage_maps needs at least one part")
    if len(parts) == 1:
        return parts[0]
    mips = torch.stack([p.mip for p in parts])  # (D, P)
    means = torch.stack([p.mean for p in parts])
    second = torch.stack([p.variance + p.mean * p.mean for p in parts])
    best_virtual = torch.stack(
        [p.best_virtual + d * n_selected for d, p in enumerate(parts)]
    )
    best_psi = torch.stack([p.best_psi for p in parts])
    # NaN-safe argmax: an unsearched pixel is NaN in every part and stays -1
    mip, which = torch.nan_to_num(mips, nan=-float("inf")).max(dim=0)
    searched = torch.isfinite(mips).any(dim=0)
    mip = torch.where(searched, mip, torch.full_like(mip, float("nan")))
    mean = means.mean(dim=0)
    variance = (second.mean(dim=0) - mean * mean).clamp_min_(0.0)
    zscore = (mip - mean) / variance.sqrt().clamp_min(1e-12)
    gather = which[None]
    none = torch.full_like(mip, -1, dtype=torch.long)
    return StageMaps(
        mip=mip,
        mean=mean,
        variance=variance,
        zscore=zscore,
        best_virtual=torch.where(searched, best_virtual.gather(0, gather)[0], none),
        best_psi=torch.where(searched, best_psi.gather(0, gather)[0], none),
        n_evaluated=sum(p.n_evaluated for p in parts),
    )
