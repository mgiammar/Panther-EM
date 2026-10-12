"""Sparse ``(pixel, template)`` pairs emitted by a dense stage, with dilation.

A pair-emitting stage (``StageSpec.pairs``) keeps each ``(pixel, hypothesis)``'s
maximum correlation over psi (the executor's pair slab), thresholds it with
:func:`~panther_em.inference.search.decisions.surpass_threshold`, and spreads the
survivors to neighbouring pixels and orientations. A template id is ``ff * O + o``;
the emitting stage's defocus plane is dropped (the sparse stage evaluates all planes).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import h5py  # type: ignore[import-untyped]
import numpy as np
import torch

from panther_em.inference.search.decisions import surpass_threshold
from panther_em.inference.search.fused_kernel_loader import PAIR_DTYPES
from panther_em.inference.search.ledger import PixelLabel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from panther_em.inference.search.hypotheses import HypothesisPlan
    from panther_em.inference.search.ledger import PixelLedger
    from panther_em.inference.search.orientations import S2Neighborhood
    from panther_em.inference.search.types import SearchGeometry



@dataclass(frozen=True)
class PairEmissionSpec:
    """Pair emission of a stage.

    Attributes
    ----------
    threshold : float
        Cutoff ``T`` of :func:`surpass_threshold`.
    pixel_radius : int
        Chebyshev radius of the pixel dilation (clipped at image borders, never across
        images; ignored in center mode).
    orientation_rings : int
        HEALPix ring hops of the orientation dilation (``0`` = none; otherwise the
        template's Euler angles must be HEALPix pixel centers).
    pair_dtype : torch.dtype
        Dtype of the executor's pair slab (float16 or float32).
    rank : int, optional
        Rank passed to the rule; defaults to the stage's feature count.
    """

    threshold: float
    pixel_radius: int = 1
    orientation_rings: int = 1
    pair_dtype: torch.dtype = torch.float16
    rank: int | None = None

    def __post_init__(self) -> None:
        """Validate."""
        if self.pair_dtype not in PAIR_DTYPES:
            raise ValueError(f"pair_dtype must be one of {PAIR_DTYPES}")
        if min(self.pixel_radius, self.orientation_rings) < 0:
            raise ValueError("pixel_radius and orientation_rings must be >= 0")


@dataclass(eq=False)
class PairList:
    """``(pixel, template)`` pairs with the correlation that emitted them.

    Attributes
    ----------
    pixel, template : torch.Tensor
        ``(M,)`` int64 flat pixel ids and template ids ``ff * O + o``.
    n_templates : int
        ``FF * O`` (the key stride).
    score : torch.Tensor
        ``(M,)`` float32; dilated-in pairs carry their best seed's score.
    """

    pixel: torch.Tensor
    template: torch.Tensor
    n_templates: int
    score: torch.Tensor

    def __post_init__(self) -> None:
        """Coerce dtypes and check lengths."""
        self.pixel = self.pixel.to(torch.long).reshape(-1)
        self.template = self.template.to(self.pixel.device, torch.long).reshape(-1)
        self.score = self.score.to(self.pixel.device, torch.float32).reshape(-1)
        if not (self.pixel.shape == self.template.shape == self.score.shape):
            raise ValueError("pixel, template and score must have the same length")

    def __len__(self) -> int:
        """Number of pairs."""
        return int(self.pixel.numel())

    def key(self) -> torch.Tensor:
        """``pixel * n_templates + template`` (unique per pair)."""
        return self.pixel * self.n_templates + self.template

    @classmethod
    def from_key(
        cls, key: torch.Tensor, n_templates: int, score: torch.Tensor
    ) -> PairList:
        """Inverse of :meth:`key`."""
        return cls(key // n_templates, key % n_templates, n_templates, score)

    @classmethod
    def concat(cls, lists: Sequence[PairList]) -> PairList:
        """Concatenate lists sharing ``n_templates``."""
        n_templates = {p.n_templates for p in lists}
        if len(n_templates) != 1:
            raise ValueError("concat needs >= 1 list, all with the same n_templates")
        return cls(
            torch.cat([p.pixel for p in lists]),
            torch.cat([p.template for p in lists]),
            n_templates.pop(),
            torch.cat([p.score for p in lists]),
        )

    def unique(self) -> PairList:
        """Deduplicate (sorted by key), keeping each pair's maximum score."""
        keys, inverse = torch.unique(self.key(), return_inverse=True)
        score = torch.full_like(keys, float("-inf"), dtype=torch.float32)
        score.scatter_reduce_(0, inverse, self.score, "amax", include_self=False)
        return PairList.from_key(keys, self.n_templates, score)

    def select(self, keep: torch.Tensor) -> PairList:
        """The pairs where ``keep`` (bool ``(M,)``) is true."""
        return PairList(
            self.pixel[keep], self.template[keep], self.n_templates, self.score[keep]
        )

    def pixels(self) -> torch.Tensor:
        """Sorted unique pixel ids."""
        return torch.unique(self.pixel)  # type: ignore[no-any-return]

    def to(self, device: torch.device | str) -> PairList:
        """Copy to ``device``."""
        return PairList(
            self.pixel.to(device),
            self.template.to(device),
            self.n_templates,
            self.score,
        )

    def save(self, target: str | Path | h5py.Group) -> None:
        """Write to an HDF5 path or open group."""
        if isinstance(target, (str, Path)):
            with h5py.File(target, "w") as f:
                self.save(f)
            return
        target.attrs["n_templates"] = self.n_templates
        for name in ("pixel", "template", "score"):
            target.create_dataset(name, data=getattr(self, name).cpu().numpy())

    @classmethod
    def load(cls, source: str | Path | h5py.Group) -> PairList:
        """Read a list written by :meth:`save`."""
        if isinstance(source, (str, Path)):
            with h5py.File(source, "r") as f:
                return cls.load(f)
        pixel, template, score = (
            torch.from_numpy(np.asarray(source[n]))
            for n in ("pixel", "template", "score")
        )
        return cls(pixel, template, int(source.attrs["n_templates"]), score)


def dilate_pairs(
    pairs: PairList,
    geometry: SearchGeometry,
    pixel_radius: int,
    orientation_table: torch.Tensor | None,
) -> PairList:
    """Spread every pair over its pixel and orientation neighbourhoods; deduplicate.

    ``orientation_table`` is ``S2Neighborhood.neighbor_table(rings)`` (``None`` for
    no orientation dilation). Dilated-in pairs inherit their seed's score.
    """
    n = pairs.n_templates
    pix, tmpl = pairs.pixel[:, None], pairs.template[:, None]
    valid_p = torch.ones_like(pix, dtype=torch.bool)
    if pixel_radius > 0 and geometry.mode == "grid":
        oh, ow = geometry.grid
        per_image = geometry.n_pixels_per_image
        offsets = torch.arange(-pixel_radius, pixel_radius + 1, device=pix.device)
        dr, dc = (
            o.reshape(1, -1) for o in torch.meshgrid(offsets, offsets, indexing="ij")
        )
        r, c = (
            (pairs.pixel % per_image // ow)[:, None] + dr,
            (pairs.pixel % ow)[:, None] + dc,
        )
        valid_p = (r >= 0) & (r < oh) & (c >= 0) & (c < ow)
        pix = (pairs.pixel // per_image)[:, None] * per_image + r * ow + c
    valid_t = torch.ones_like(tmpl, dtype=torch.bool)
    if orientation_table is not None:
        n_orient = orientation_table.shape[0]
        o = orientation_table.to(tmpl.device)[pairs.template % n_orient]  # (M, n_max)
        valid_t = o >= 0
        tmpl = (pairs.template // n_orient)[:, None] * n_orient + o.clamp_min(0)
    key = pix[:, :, None] * n + tmpl[:, None, :]  # (M, K, n_max)
    valid = valid_p[:, :, None] & valid_t[:, None, :]
    score = pairs.score[:, None, None].expand_as(key)
    return PairList.from_key(key[valid], n, score[valid]).unique()


class PairCollector:
    """The executor's ``BlockPairSink`` for one emitting stage.

    Thresholds each block's ``(n_valid, n_virtual)`` pair maxima, decodes slab
    columns to templates, dilates, and drops pairs on masked pixels.
    """

    def __init__(
        self,
        spec: PairEmissionSpec,
        ledger: PixelLedger,
        column_index: torch.Tensor,
        hyp_plan: HypothesisPlan,
        *,
        rank: int,
        s2: S2Neighborhood | None,
        device: torch.device | str,
    ) -> None:
        self.spec = spec
        self.geometry = ledger.geometry
        self.device = torch.device(device)
        self.column_index = column_index.to(device, torch.long)
        self.hyp_plan = hyp_plan
        self.n_templates = hyp_plan.space.n_templates
        self.rank = rank
        for name in ("image_norm", "noise_variance"):
            if name not in ledger.aux:
                raise ValueError(
                    f"pair emission needs the per-pixel map {name!r} "
                    f"(ImageSet.pixel_maps); available: {sorted(ledger.aux)}"
                )
        self.image_norm = ledger.aux["image_norm"].to(device)
        self.noise_variance = ledger.aux["noise_variance"].to(device)
        self.masked = (ledger.label == int(PixelLabel.MASKED)).to(device)
        self.table = (
            s2.neighbor_table(spec.orientation_rings, device)
            if s2 is not None and spec.orientation_rings > 0
            else None
        )
        self.n_raw = 0
        self._blocks: list[PairList] = []

    @torch.no_grad()
    def consume_block(
        self,
        pair_max: torch.Tensor,
        *,
        col_start: int,
        n_valid: int,
        mean: torch.Tensor,
        variance: torch.Tensor,
    ) -> None:
        """Threshold one block (one rule call, one host sync); dilation waits for finish.

        Per-block dilation and row chunking cost ~20 ms of GPU idle per block (the
        ``nonzero``/``unique`` syncs stall the graph replays); the seeds of a whole
        micrograph are small, so they are dilated once at the end.
        """
        spec = self.spec
        cols = self.column_index[col_start : col_start + n_valid]
        passed = surpass_threshold(
            pair_max,
            mean.to(self.device),
            variance.to(self.device),
            threshold=spec.threshold,
            image_norm=self.image_norm[cols],
            noise_variance=self.noise_variance[cols],
            rank=self.rank,
        )
        rows, virtual = torch.nonzero(passed, as_tuple=True)
        if rows.numel() == 0:
            return
        self.n_raw += int(rows.numel())
        template = self.hyp_plan.decode(virtual) % self.n_templates
        self._blocks.append(
            PairList(cols[rows], template, self.n_templates, pair_max[rows, virtual])
        )

    def finish(self) -> PairList:
        """The stage's dilated, deduplicated pair list (masked pixels dropped)."""
        empty = torch.zeros(0, dtype=torch.long, device=self.column_index.device)
        blocks = self._blocks or [PairList(empty, empty, self.n_templates, empty)]
        self._blocks = []
        pairs = PairList.concat(blocks).unique()
        if len(pairs):
            pairs = dilate_pairs(pairs, self.geometry, self.spec.pixel_radius, self.table)
            pairs = pairs.select(~self.masked[pairs.pixel])
        return pairs
