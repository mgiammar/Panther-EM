"""Per-pixel state and results: the dense ledger, one stage's maps, the final maps.

* :class:`PixelLedger` -- dense ``(n_pixels,)`` labels and last-known statistics,
  written once per stage; the search's working state.
* :class:`StageResult` -- one stage's maps in its compact pixel order, with the
  labels after its decision and timings.
* :class:`SearchResult` -- dense final maps with decoding (orientation angles,
  in-plane angle, defocus) and HDF5 I/O, including optional per-stage groups.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import h5py  # type: ignore[import-untyped]
import numpy as np
import torch
import torch.nn.functional as F

from panther_em.inference.search.decision import PixelLabel
from panther_em.inference.search.executor import StageMaps
from panther_em.inference.search.hypotheses import HypothesisSpace
from panther_em.inference.search.stage import Stage, stage_from_json, stage_to_json
from panther_em.inference.search.types import PixelSelection, SearchGeometry

if TYPE_CHECKING:
    from pathlib import Path

    from panther_em.inference.search.hypotheses import HypothesisSet

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
STAGE_MAP_NAMES = ("mip", "mean", "variance", "zscore", "best_index", "best_psi")


# --------------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------------- #


class PixelLedger:
    """Dense ``(n_pixels,)`` per-pixel state, written once per stage.

    Statistics are *per stage* (a different rank is a different estimator), so the
    last stage to search a pixel overwrites its values; ``last_stage`` says which.
    Hypothesis ids are stored as **global** ids (see :mod:`.hypotheses`).

    Parameters
    ----------
    geometry : SearchGeometry
        The valid-correlation grid.
    device : torch.device or str
        Where the state lives.
    initial : PixelSelection, optional
        Pixels eligible for the search; everything else is ``MASKED``.
    """

    FIELDS = ("mip", "mean", "variance", "zscore", "best_index", "best_psi", "last_stage")

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

    @torch.no_grad()
    def write(
        self,
        maps: StageMaps,
        pixel_index: torch.Tensor,
        stage_index: int,
        hyps: HypothesisSet,
    ) -> None:
        """Record a stage's maps at ``pixel_index`` (same order as the maps).

        Written pixels are labelled AMBIGUOUS; the stage's decision relabels them.
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
        while len(self.n_psi_of_stage) <= stage_index:
            self.n_psi_of_stage.append(0)
        self.n_psi_of_stage[stage_index] = int(hyps.n_psi)

    def set_labels(self, pixel_index: torch.Tensor, labels: torch.Tensor) -> None:
        """Overwrite labels at ``pixel_index``."""
        self.label[pixel_index.to(self.device, torch.long)] = labels.to(
            self.device, torch.int8
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
        offsets = torch.arange(-radius, radius + 1, device=self.device, dtype=torch.float32)
        disk = (offsets[:, None] ** 2 + offsets[None, :] ** 2) <= radius**2
        kernel = disk.to(torch.float32)[None, None]
        near = F.conv2d(grid, kernel, padding=radius).squeeze(1).reshape(-1) > 0
        ambiguous = self.label == int(PixelLabel.AMBIGUOUS)
        self.label[near & ambiguous] = int(PixelLabel.ABSORBED)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(eq=False)
class StageResult:
    """One stage's result, in the stage's compact pixel order.

    Attributes
    ----------
    stage_index : int
    stage : Stage
    pixel_index : torch.Tensor
        ``(P_stage,)`` flat pixel ids the stage searched (compact order).
    maps : StageMaps
        The executor's maps (``best_virtual`` is virtual; see ``best_index``).
    best_index : torch.Tensor
        ``(P_stage,)`` global hypothesis ids.
    labels_after : torch.Tensor
        ``(P_stage,)`` int8 labels after the stage's decision ran.
    n_ambiguous_after : int
        Pixels left AMBIGUOUS on the whole ledger after the decision.
    elapsed_s : float
        Wall time of the stage; ``featurize_s`` (store preparation) and
        ``execute_s`` (block loop) split it for tuning.
    """

    stage_index: int
    stage: Stage
    pixel_index: torch.Tensor
    maps: StageMaps
    best_index: torch.Tensor
    labels_after: torch.Tensor
    n_ambiguous_after: int
    elapsed_s: float
    featurize_s: float = 0.0
    execute_s: float = 0.0

    @property
    def n_pixels(self) -> int:
        """Pixels searched in this stage."""
        return int(self.pixel_index.numel())

    def to(self, device: torch.device | str) -> StageResult:
        """Copy with every tensor on ``device``."""
        maps = self.maps
        moved = StageMaps(
            mip=maps.mip.to(device),
            mean=maps.mean.to(device),
            variance=maps.variance.to(device),
            zscore=maps.zscore.to(device),
            best_virtual=maps.best_virtual.to(device),
            best_psi=maps.best_psi.to(device),
            n_evaluated=maps.n_evaluated,
        )
        return StageResult(
            stage_index=self.stage_index,
            stage=self.stage,
            pixel_index=self.pixel_index.to(device),
            maps=moved,
            best_index=self.best_index.to(device),
            labels_after=self.labels_after.to(device),
            n_ambiguous_after=self.n_ambiguous_after,
            elapsed_s=self.elapsed_s,
            featurize_s=self.featurize_s,
            execute_s=self.execute_s,
        )


@dataclass(eq=False)
class SearchResult:
    """Dense per-pixel maps of a finished (or snapshotted) search.

    Maps are ``(B, *grid)``: ``(B, out_h, out_w)`` in grid mode, ``(B, 1)`` in center
    mode. ``mip/zscore/mean/variance`` are NaN and ``best_index/best_psi/last_stage``
    are ``-1`` where a pixel was never searched; ``label`` holds
    :class:`~panther_em.inference.search.decision.PixelLabel` values.

    Attributes
    ----------
    stages : tuple[Stage, ...]
        The searched stages (their descriptions are saved with the maps).
    stage_results : list[StageResult]
        Per-stage maps, kept when the searcher was built with
        ``keep_stage_results=True``.
    """

    geometry: SearchGeometry
    space: HypothesisSpace
    stages: tuple[Stage, ...]
    mip: torch.Tensor
    zscore: torch.Tensor
    mean: torch.Tensor
    variance: torch.Tensor
    best_index: torch.Tensor
    best_psi: torch.Tensor
    label: torch.Tensor
    last_stage: torch.Tensor
    n_psi_of_stage: tuple[int, ...]
    stage_results: list[StageResult] = field(default_factory=list)
    elapsed_s: float = 0.0

    @classmethod
    def from_ledger(
        cls,
        ledger: PixelLedger,
        space: HypothesisSpace,
        stages: tuple[Stage, ...],
        *,
        stage_results: list[StageResult] | None = None,
        elapsed_s: float = 0.0,
    ) -> SearchResult:
        """Snapshot a ledger (clones every field)."""
        g = ledger.geometry
        return cls(
            geometry=g,
            space=space,
            stages=tuple(stages),
            mip=g.unflatten(ledger.mip.clone()),
            zscore=g.unflatten(ledger.zscore.clone()),
            mean=g.unflatten(ledger.mean.clone()),
            variance=g.unflatten(ledger.variance.clone()),
            best_index=g.unflatten(ledger.best_index.clone()),
            best_psi=g.unflatten(ledger.best_psi.clone()),
            label=g.unflatten(ledger.label.clone()),
            last_stage=g.unflatten(ledger.last_stage.clone()),
            n_psi_of_stage=tuple(ledger.n_psi_of_stage),
            stage_results=list(stage_results or []),
            elapsed_s=elapsed_s,
        )

    # -- derived -----------------------------------------------------------------

    def psi_degrees(self) -> torch.Tensor:
        """Best in-plane angle in degrees (NaN where unsearched)."""
        out = torch.full(
            self.best_psi.shape, float("nan"), dtype=torch.float32, device=self.best_psi.device
        )
        for stage_index, n_psi in enumerate(self.n_psi_of_stage):
            if n_psi <= 0:
                continue
            here = (self.last_stage == stage_index) & (self.best_psi >= 0)
            out[here] = 360.0 * self.best_psi[here].to(torch.float32) / n_psi
        return out

    def decode(self) -> dict[str, torch.Tensor | None]:
        """Decode ``best_index`` into defocus / filter / orientation indices and angles.

        Keys: ``defocus_index``, ``filter_index``, ``orientation_index`` (``-1``
        where unsearched), ``psi`` (degrees), and ``phi``/``theta`` (degrees) and
        ``defocus`` when the space carries those tables (``None`` otherwise).
        """
        d_idx, f_idx, o_idx = self.space.decode(self.best_index)
        valid = o_idx >= 0
        phi = theta = defocus = None
        if self.space.phi_deg is not None and self.space.theta_deg is not None:
            safe = torch.where(valid, o_idx, torch.zeros_like(o_idx))
            nan = torch.full(safe.shape, float("nan"), dtype=torch.float32, device=safe.device)
            phi = torch.where(valid, self.space.phi_deg.to(safe.device)[safe], nan)
            theta = torch.where(valid, self.space.theta_deg.to(safe.device)[safe], nan)
        if self.space.defocus_values is not None:
            safe_d = torch.where(valid, d_idx, torch.zeros_like(d_idx))
            nan = torch.full(
                safe_d.shape, float("nan"), dtype=torch.float32, device=safe_d.device
            )
            defocus = torch.where(
                valid, self.space.defocus_values.to(safe_d.device)[safe_d], nan
            )
        return {
            "defocus_index": d_idx,
            "filter_index": f_idx,
            "orientation_index": o_idx,
            "phi": phi,
            "theta": theta,
            "psi": self.psi_degrees(),
            "defocus": defocus,
        }

    def to(self, device: torch.device | str) -> SearchResult:
        """Copy with every map on ``device``."""
        moved = {name: getattr(self, name).to(device) for name in MAP_NAMES}
        return SearchResult(
            geometry=self.geometry,
            space=self.space,
            stages=self.stages,
            n_psi_of_stage=self.n_psi_of_stage,
            stage_results=[s.to(device) for s in self.stage_results],
            elapsed_s=self.elapsed_s,
            **moved,
        )

    def to_numpy(self) -> dict[str, np.ndarray]:
        """The eight maps as NumPy arrays (CPU)."""
        return {name: getattr(self, name).detach().cpu().numpy() for name in MAP_NAMES}

    # -- I/O -------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the maps plus enough metadata for :meth:`load` and :meth:`decode`.

        Layout: the eight dense maps and the space's angle/defocus tables as root
        datasets; ``geometry``, ``space``, ``stages``, ``n_psi_of_stage`` and
        ``elapsed_s`` as attributes; and, when stage results were kept, one group
        ``stages/stage{i}`` per stage holding the compact ``pixel_index``, the six
        stage maps (``best_index`` global), ``labels_after`` and timing attributes.
        """

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
            f.attrs["stages"] = json.dumps([stage_to_json(s) for s in self.stages])
            f.attrs["elapsed_s"] = float(self.elapsed_s)
            if self.stage_results:
                root = f.create_group("stages")
                for r in self.stage_results:
                    grp = root.create_group(f"stage{r.stage_index}")
                    grp.create_dataset("pixel_index", data=r.pixel_index.cpu().numpy())
                    for name in STAGE_MAP_NAMES:
                        source = r.best_index if name == "best_index" else getattr(r.maps, name)
                        grp.create_dataset(name, data=source.detach().cpu().numpy())
                    grp.create_dataset("labels_after", data=r.labels_after.cpu().numpy())
                    grp.attrs["n_pixels"] = r.n_pixels
                    grp.attrs["n_evaluated"] = int(r.maps.n_evaluated)
                    grp.attrs["n_ambiguous_after"] = int(r.n_ambiguous_after)
                    grp.attrs["elapsed_s"] = float(r.elapsed_s)
                    grp.attrs["featurize_s"] = float(r.featurize_s)
                    grp.attrs["execute_s"] = float(r.execute_s)

    @classmethod
    def load(cls, path: str | Path) -> SearchResult:
        """Read a result written by :meth:`save` (stage decisions are not restored)."""

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
            stages = tuple(stage_from_json(s) for s in json.loads(f.attrs["stages"]))
            n_psi_of_stage = tuple(int(v) for v in json.loads(f.attrs["n_psi_of_stage"]))
            elapsed = float(f.attrs.get("elapsed_s", 0.0))
            stage_results = []
            if "stages" in f:
                for name in sorted(f["stages"], key=lambda n: int(n.removeprefix("stage"))):
                    grp = f["stages"][name]
                    i = int(name.removeprefix("stage"))
                    tensors = {k: torch.from_numpy(np.asarray(grp[k])) for k in STAGE_MAP_NAMES}
                    best_index = tensors.pop("best_index")
                    stage_results.append(
                        StageResult(
                            stage_index=i,
                            stage=stages[i],
                            pixel_index=torch.from_numpy(np.asarray(grp["pixel_index"])),
                            maps=StageMaps(
                                mip=tensors["mip"],
                                mean=tensors["mean"],
                                variance=tensors["variance"],
                                zscore=tensors["zscore"],
                                best_virtual=best_index,
                                best_psi=tensors["best_psi"],
                                n_evaluated=int(grp.attrs["n_evaluated"]),
                            ),
                            best_index=best_index,
                            labels_after=torch.from_numpy(np.asarray(grp["labels_after"])),
                            n_ambiguous_after=int(grp.attrs["n_ambiguous_after"]),
                            elapsed_s=float(grp.attrs["elapsed_s"]),
                            featurize_s=float(grp.attrs["featurize_s"]),
                            execute_s=float(grp.attrs["execute_s"]),
                        )
                    )
        return cls(
            geometry=geometry,
            space=space,
            stages=stages,
            mip=maps["mip"],
            zscore=maps["zscore"],
            mean=maps["mean"],
            variance=maps["variance"],
            best_index=maps["best_index"],
            best_psi=maps["best_psi"],
            label=maps["label"],
            last_stage=maps["last_stage"],
            n_psi_of_stage=n_psi_of_stage,
            stage_results=stage_results,
            elapsed_s=elapsed,
        )
