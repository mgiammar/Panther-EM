"""Results: one stage's compact maps, and the dense final maps with decoding and I/O."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import h5py  # type: ignore[import-untyped]
import numpy as np
import torch

from panther_em.inference.search.hypotheses import HypothesisSpace
from panther_em.inference.search.schedule import SearchSchedule, StageSpec
from panther_em.inference.search.types import SearchGeometry

if TYPE_CHECKING:
    from pathlib import Path

    from panther_em.inference.search.executor import StageMaps
    from panther_em.inference.search.ledger import PixelLedger

MAP_NAMES = (
    "mip",
    "zscore",
    "mean",
    "variance",
    "best_index",
    "best_psi",
    "label",
    "last_stage",
)


@dataclass(eq=False)
class StageResult:
    """One stage's result, in the stage's compact pixel order.

    Attributes
    ----------
    stage_index : int
    spec : StageSpec
    pixel_index : torch.Tensor
        ``(P_stage,)`` flat pixel ids the stage searched (compact order).
    maps : StageMaps
        The executor's maps (``best_virtual`` is virtual; see ``best_index``).
    best_index : torch.Tensor
        ``(P_stage,)`` global hypothesis ids.
    labels_after : torch.Tensor
        ``(P_stage,)`` int8 labels after the stage's decision ran.
    n_uncertain_after : int
        Pixels still eligible for the next stage (whole ledger).
    elapsed_s : float
        Wall time of the stage; ``featurize_s`` (store preparation) and ``execute_s``
        (block loop) split it for tuning.
    """

    stage_index: int
    spec: StageSpec
    pixel_index: torch.Tensor
    maps: StageMaps
    best_index: torch.Tensor
    labels_after: torch.Tensor
    n_uncertain_after: int
    elapsed_s: float
    featurize_s: float = 0.0
    execute_s: float = 0.0

    @property
    def n_pixels(self) -> int:
        """Pixels searched in this stage."""
        return int(self.pixel_index.numel())

    def dense(self, geometry: SearchGeometry, name: str) -> torch.Tensor:
        """A map as ``(B, *grid)`` with NaN / -1 outside the stage's pixels."""
        values = self.best_index if name == "best_index" else getattr(self.maps, name)
        fill = (
            -1
            if values.dtype in (torch.int64, torch.int32, torch.int8)
            else float("nan")
        )
        out = torch.full(
            (geometry.n_pixels,), fill, dtype=values.dtype, device=values.device
        )
        out[self.pixel_index.to(values.device)] = values
        return geometry.unflatten(out)


@dataclass(eq=False)
class DecodedMaps:
    """Decoded best hypothesis per pixel, all ``(B, *grid)``."""

    defocus_index: torch.Tensor
    filter_index: torch.Tensor
    orientation_index: torch.Tensor
    phi: torch.Tensor | None
    theta: torch.Tensor | None
    psi: torch.Tensor
    defocus: torch.Tensor | None


@dataclass(eq=False)
class SearchResult:
    """Dense per-pixel maps of a finished (or snapshotted) search.

    Maps are ``(B, *grid)``: ``(B, out_h, out_w)`` in grid mode, ``(B, 1)`` in center
    mode. ``mip/zscore/mean/variance`` are NaN and ``best_index/best_psi/last_stage``
    are ``-1`` where a pixel was never searched; ``label`` holds
    :class:`~panther_em.inference.search.ledger.PixelLabel` values.
    """

    geometry: SearchGeometry
    space: HypothesisSpace
    schedule: SearchSchedule
    mip: torch.Tensor
    zscore: torch.Tensor
    mean: torch.Tensor
    variance: torch.Tensor
    best_index: torch.Tensor
    best_psi: torch.Tensor
    label: torch.Tensor
    last_stage: torch.Tensor
    n_psi_of_stage: tuple[int, ...]
    stages: list[StageResult] = field(default_factory=list)
    elapsed_s: float = 0.0

    @classmethod
    def from_ledger(
        cls,
        ledger: PixelLedger,
        space: HypothesisSpace,
        schedule: SearchSchedule,
        *,
        stages: list[StageResult] | None = None,
        elapsed_s: float = 0.0,
    ) -> SearchResult:
        """Snapshot a ledger (clones every field)."""
        g = ledger.geometry
        return cls(
            geometry=g,
            space=space,
            schedule=schedule,
            mip=g.unflatten(ledger.mip.clone()),
            zscore=g.unflatten(ledger.zscore.clone()),
            mean=g.unflatten(ledger.mean.clone()),
            variance=g.unflatten(ledger.variance.clone()),
            best_index=g.unflatten(ledger.best_index.clone()),
            best_psi=g.unflatten(ledger.best_psi.clone()),
            label=g.unflatten(ledger.label.clone()),
            last_stage=g.unflatten(ledger.last_stage.clone()),
            n_psi_of_stage=tuple(ledger.n_psi_of_stage),
            stages=list(stages or []),
            elapsed_s=elapsed_s,
        )

    # -- derived -----------------------------------------------------------------

    def std(self) -> torch.Tensor:
        """``sqrt(max(variance, 0))``."""
        return self.variance.clamp_min(0.0).sqrt()

    def searched(self) -> torch.Tensor:
        """Bool map of pixels some stage evaluated."""
        return self.last_stage >= 0

    def psi_degrees(self) -> torch.Tensor:
        """Best in-plane angle in degrees (NaN where unsearched)."""
        out = torch.full(
            self.best_psi.shape,
            float("nan"),
            dtype=torch.float32,
            device=self.best_psi.device,
        )
        for stage_index, n_psi in enumerate(self.n_psi_of_stage):
            if n_psi <= 0:
                continue
            here = (self.last_stage == stage_index) & (self.best_psi >= 0)
            out[here] = 360.0 * self.best_psi[here].to(torch.float32) / n_psi
        return out

    def decode(self) -> DecodedMaps:
        """Decode ``best_index`` into defocus / filter / orientation (and angles)."""
        d_idx, f_idx, o_idx = self.space.decode(self.best_index)
        valid = o_idx >= 0
        phi = theta = None
        if self.space.phi_deg is not None and self.space.theta_deg is not None:
            safe = torch.where(valid, o_idx, torch.zeros_like(o_idx))
            nan = torch.full(
                safe.shape, float("nan"), dtype=torch.float32, device=safe.device
            )
            phi = torch.where(valid, self.space.phi_deg.to(safe.device)[safe], nan)
            theta = torch.where(valid, self.space.theta_deg.to(safe.device)[safe], nan)
        defocus = None
        if self.space.defocus_values is not None:
            safe_d = torch.where(valid, d_idx, torch.zeros_like(d_idx))
            nan = torch.full(
                safe_d.shape, float("nan"), dtype=torch.float32, device=safe_d.device
            )
            defocus = torch.where(
                valid, self.space.defocus_values.to(safe_d.device)[safe_d], nan
            )
        return DecodedMaps(
            defocus_index=d_idx,
            filter_index=f_idx,
            orientation_index=o_idx,
            phi=phi,
            theta=theta,
            psi=self.psi_degrees(),
            defocus=defocus,
        )

    def per_particle(self) -> SearchResult:
        """Center mode only: squeeze the trailing ``(1,)`` grid axis to ``(B,)``."""
        if self.geometry.mode != "center":
            raise ValueError("per_particle() is only defined in center mode")
        squeezed = {
            name: getattr(self, name).reshape(self.geometry.n_images)
            for name in MAP_NAMES
        }
        return SearchResult(
            geometry=self.geometry,
            space=self.space,
            schedule=self.schedule,
            n_psi_of_stage=self.n_psi_of_stage,
            stages=self.stages,
            elapsed_s=self.elapsed_s,
            **squeezed,
        )

    def to(self, device: torch.device | str) -> SearchResult:
        """Copy with every map on ``device``."""
        moved = {name: getattr(self, name).to(device) for name in MAP_NAMES}
        return SearchResult(
            geometry=self.geometry,
            space=self.space,
            schedule=self.schedule,
            n_psi_of_stage=self.n_psi_of_stage,
            stages=self.stages,
            elapsed_s=self.elapsed_s,
            **moved,
        )

    def to_numpy(self) -> dict[str, np.ndarray]:
        """The eight maps as NumPy arrays (CPU)."""
        return {name: getattr(self, name).detach().cpu().numpy() for name in MAP_NAMES}

    # -- I/O -------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the maps plus enough metadata for :meth:`load` and :meth:`decode`."""
        with h5py.File(path, "w") as f:
            for name, array in self.to_numpy().items():
                f.create_dataset(name, data=array)
            g = self.geometry
            f.attrs["geometry"] = json.dumps(
                {
                    "n_images": g.n_images,
                    "image_shape": list(g.image_shape),
                    "template_shape": list(g.template_shape),
                    "grid": list(g.grid),
                    "mode": g.mode,
                }
            )
            f.attrs["space"] = json.dumps(
                {
                    "n_defocus": self.space.n_defocus,
                    "n_filters": self.space.n_filters,
                    "n_orientations": self.space.n_orientations,
                }
            )
            for name in ("phi_deg", "theta_deg", "defocus_values"):
                t = getattr(self.space, name)
                if t is not None:
                    f.create_dataset(name, data=t.detach().cpu().numpy())
            f.attrs["n_psi_of_stage"] = json.dumps(list(self.n_psi_of_stage))
            f.attrs["stages"] = json.dumps(
                [
                    {
                        "rectangles": [list(r) for r in s.rectangles],
                        "precision": s.precision.value,
                        "n_psi": s.n_psi,
                        "error": s.error,
                        "name": s.name,
                    }
                    for s in self.schedule.stages
                ]
            )
            f.attrs["elapsed_s"] = float(self.elapsed_s)

    @classmethod
    def load(cls, path: str | Path) -> SearchResult:
        """Read a result written by :meth:`save` (decisions restored as no-ops)."""
        with h5py.File(path, "r") as f:
            maps = {name: torch.from_numpy(np.asarray(f[name])) for name in MAP_NAMES}
            geo = json.loads(f.attrs["geometry"])
            geometry = SearchGeometry(
                n_images=int(geo["n_images"]),
                image_shape=(int(geo["image_shape"][0]), int(geo["image_shape"][1])),
                template_shape=(
                    int(geo["template_shape"][0]),
                    int(geo["template_shape"][1]),
                ),
                grid=tuple(geo["grid"]),
                mode=geo["mode"],
            )
            sp = json.loads(f.attrs["space"])

            def table(name: str) -> torch.Tensor | None:
                return torch.from_numpy(np.asarray(f[name])) if name in f else None

            space = HypothesisSpace(
                n_defocus=int(sp["n_defocus"]),
                n_filters=int(sp["n_filters"]),
                n_orientations=int(sp["n_orientations"]),
                phi_deg=table("phi_deg"),
                theta_deg=table("theta_deg"),
                defocus_values=table("defocus_values"),
            )
            stages = tuple(
                StageSpec.create(
                    [tuple(r) for r in s["rectangles"]],
                    precision=s["precision"],
                    n_psi=int(s["n_psi"]),
                    error=s["error"],
                    name=s.get("name", ""),
                )
                for s in json.loads(f.attrs["stages"])
            )
            n_psi_of_stage = tuple(
                int(v) for v in json.loads(f.attrs["n_psi_of_stage"])
            )
            elapsed = float(f.attrs.get("elapsed_s", 0.0))
        return cls(
            geometry=geometry,
            space=space,
            schedule=SearchSchedule(stages),
            mip=maps["mip"],
            zscore=maps["zscore"],
            mean=maps["mean"],
            variance=maps["variance"],
            best_index=maps["best_index"],
            best_psi=maps["best_psi"],
            label=maps["label"],
            last_stage=maps["last_stage"],
            n_psi_of_stage=n_psi_of_stage,
            elapsed_s=elapsed,
        )
