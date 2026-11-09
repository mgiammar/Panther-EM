"""Self-seeded stages: a seed pass picks each pixel's cell, a local pass scores it.

A stage with a :class:`Seeding` does, per featurized pass, two correlations over the
same feature store instead of one dense loop over every orientation:

1. the **seed pass**: the ordinary hypothesis loop over one *seed template* per
   HEALPix cell ``Seeding.level`` levels above the stage's grid -- the psi-aligned
   mean of the cell's templates (``"cell_mean"``, :func:`cell_mean_templates`,
   :class:`CellMeanWeights`) or one representative template of the cell
   (``"subset"``). Its best hypothesis names a *seed cell* per pixel. With
   ``lattice = L`` only the pixels on every ``L``-th row and column are seeded
   (:class:`LatticeSelection`; the store's other columns are never read by the seed
   pass) and a pixel's candidates are the seeds of the up to four lattice pixels
   bracketing it;
2. the **local pass**: the exact search of every pixel over the orientations within
   ``Seeding.rings`` of each of its candidate cells (:mod:`.correlation.local`), with
   the mean and variance of the stage's whole hypothesis set from the closed-form
   moments. ``top_k`` keeps only the candidates with the highest seed z-score;
   :func:`seed_work` turns candidates into the local pass's distinct (pixel, cell)
   entries.

The maps decode exactly as the dense stage's (global ids over the stage's own
hypotheses); the result equals the dense search's wherever the dense argmax lies in
one of the pixel's neighborhoods, and is lower elsewhere (more rejection, never
less). :class:`SeedResources` holds what a searcher builds once per seeded stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from panther_em.orientations import healpix_level_ids
from panther_em.search.correlation.local import LocalWork
from panther_em.search.correlation.reduction import hermitian_spectrum
from panther_em.search.hypotheses import (
    HypothesisSpace,
    LocalHypotheses,
    template_rows_for,
)
from panther_em.search.weights import TensorWeights

if TYPE_CHECKING:
    from panther_em.search.correlation.layout import FeatureLayout
    from panther_em.search.correlation.maps import StageMaps
    from panther_em.search.hypotheses import HypothesisSet
    from panther_em.search.images import PixelSelection, SearchGeometry

SeedTemplates = Literal["cell_mean", "subset"]


@dataclass(frozen=True)
class Seeding:
    """How a stage seeds itself (see the module notes).

    Parameters
    ----------
    level : int
        HEALPix levels above the stage's orientation grid of the seed cells (one
        seed template per cell; ``1`` = cells of four orientations).
    rings : int
        Neighbor rings on the stage's grid around a seed cell that the local pass
        searches.
    templates : {"cell_mean", "subset"}
        The seed templates: each cell's psi-aligned mean template, or one of the
        template's own orientations per cell.
    n_psi : int, optional
        In-plane samples of the seed pass (``None``: the stage's ``n_psi``); fewer
        samples make the seed pass cheaper, the local pass keeps the stage's.
    lattice : int
        Seed only every ``lattice``-th row and column of pixels; the others search
        the union of the neighborhoods of the (up to four) lattice pixels around
        them. ``1`` seeds every pixel.
    top_k : int, optional
        Keep only the ``top_k`` candidate seeds with the highest seed z-score per
        pixel (``None``: all of them).
    """

    level: int = 1
    rings: int = 2
    templates: SeedTemplates = "cell_mean"
    n_psi: int | None = None
    lattice: int = 1
    top_k: int | None = None

    def __post_init__(self) -> None:
        """Validate the choices."""
        if not isinstance(self.level, int) or self.level < 1:
            raise ValueError("seeding level must be an int >= 1")
        if not isinstance(self.rings, int) or self.rings < 0:
            raise ValueError("seeding rings must be an int >= 0")
        if self.templates not in ("cell_mean", "subset"):
            raise ValueError(f"unknown seed templates {self.templates!r}")
        for name, value, low in (
            ("n_psi", self.n_psi, 2),
            ("top_k", self.top_k, 1),
        ):
            if value is not None and (not isinstance(value, int) or value < low):
                raise ValueError(f"seeding {name} must be None or an int >= {low}")
        if not isinstance(self.lattice, int) or self.lattice < 1:
            raise ValueError("seeding lattice must be an int >= 1")

    @property
    def n_candidates(self) -> int:
        """Candidate seeds a pixel can have (before ``top_k``)."""
        return 1 if self.lattice == 1 else 4


# --------------------------------------------------------------------------- #
# Seed templates
# --------------------------------------------------------------------------- #


def _energy(w: torch.Tensor) -> torch.Tensor:
    """Self-correlation at psi = 0 of ``(..., K, M)`` weights."""
    p = (w.abs() ** 2).sum(-1)  # (..., K)
    return p[..., 0] + 2 * p[..., 1:].sum(-1)


@torch.no_grad()
def cell_mean_templates(
    us: torch.Tensor,
    cell_of: torch.Tensor,
    n_cells: int,
    n_psi: int,
    *,
    n_filters: int = 1,
    cell_mask: torch.Tensor | None = None,
    chunk: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The psi-aligned mean template of every non-empty cell.

    Parameters
    ----------
    us : torch.Tensor
        ``(FF * O, K, M)`` complex weights, row ``f * O + o``.
    cell_of : torch.Tensor
        ``(O,)`` int64 cell of each orientation.
    n_cells : int
        Cells ``cell_of`` indexes.
    n_psi : int
        In-plane samples the alignment angle is searched on.
    n_filters : int, optional
        ``FF``; a mean template per (filter, cell).
    cell_mask : torch.Tensor, optional
        ``(K, M)`` bool: weights outside the mask are zeroed first (cells no stage
        rectangle covers).
    chunk : int, optional
        Cells processed at once.

    Returns
    -------
    templates : torch.Tensor
        ``(FF * C, K, M)`` complex64, row ``f * C + c`` for the ``C`` non-empty cells.
    cells : torch.Tensor
        ``(C,)`` int64 cell id of each template row's cell, ascending.
    """
    cell_of = cell_of.to("cpu", torch.long)
    o_count = int(cell_of.numel())
    if o_count * int(n_filters) != int(us.shape[0]):
        raise ValueError("cell_of must name every orientation of every filter")
    counts = torch.bincount(cell_of, minlength=int(n_cells))
    cells = torch.nonzero(counts > 0).squeeze(-1)
    c_max = int(counts.max()) if cells.numel() else 0
    # members of each non-empty cell, -1 padded
    order = torch.sort(cell_of, stable=True).indices
    sorted_cells = cell_of[order]
    starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(o_count) - starts[sorted_cells]
    members = torch.full((int(n_cells), c_max), -1, dtype=torch.long)
    members[sorted_cells, rank] = order
    members = members[cells].to(us.device)  # (C, c_max)
    valid = members >= 0
    safe = torch.where(valid, members, torch.zeros_like(members))
    weight = valid.to(torch.float32)[..., None, None]
    n_members = valid.sum(1).to(torch.float32)  # (C,)

    k_count = int(us.shape[1])
    k = torch.arange(k_count, device=us.device, dtype=torch.float64)
    mask = None if cell_mask is None else cell_mask.to(us.device)
    out = torch.empty(
        (int(n_filters) * int(cells.numel()), *us.shape[1:]),
        dtype=torch.complex64,
        device=us.device,
    )
    n_c = int(cells.numel())
    for f in range(int(n_filters)):
        for c0 in range(0, n_c, int(chunk)):
            c1 = min(c0 + int(chunk), n_c)
            w = us[f * o_count + safe[c0:c1]].to(torch.complex64)  # (c, c_max, K, M)
            if mask is not None:
                w = torch.where(mask, w, torch.zeros_like(w))
            ref = w[:, 0]  # the first member is always real (non-empty cell)
            g = torch.einsum("ckm,cnkm->cnk", ref, w.conj())
            corr = torch.fft.irfft(
                hermitian_spectrum(g, n_psi)[0], n=n_psi, dim=-1, norm="forward"
            )
            psi = 2 * np.pi * corr.argmax(-1).to(torch.float64) / n_psi  # (c, c_max)
            phase = torch.exp(-1j * psi[..., None, None] * k[None, None, :, None])
            aligned = w * phase.to(torch.complex64)
            wt = weight[c0:c1]
            n = n_members[c0:c1]
            mean = (aligned * wt).sum(1) / n[:, None, None]
            energy = (_energy(w) * wt[..., 0, 0]).sum(1) / n
            scale = (energy / _energy(mean).clamp_min(1e-30)).sqrt()
            out[f * n_c + c0 : f * n_c + c1] = mean * scale[:, None, None]
    return out, cells


class CellMeanWeights(TensorWeights):
    """The cell-mean seed templates as a weight set.

    Row ``f * C + c`` is filter ``f`` of the ``c``-th non-empty cell; ``cells`` maps
    ``c`` to the cell id (:attr:`LocalHypotheses.rows` index).

    Parameters
    ----------
    templates : torch.Tensor
        ``(FF * C, K, M)`` from :func:`cell_mean_templates`.
    cells : torch.Tensor
        ``(C,)`` int64 cell ids.
    n_filters : int
        ``FF``.
    k_start, m_start : int
        Feature-space origin of the templates' box.
    key : Hashable
        Cache identity of the set.
    """

    def __init__(
        self,
        templates: torch.Tensor,
        *,
        cells: torch.Tensor,
        n_filters: int,
        k_start: int,
        m_start: int,
        key: tuple,
    ) -> None:
        super().__init__(
            templates,
            n_filters=n_filters,
            n_orientations=int(cells.numel()),
            k_start=k_start,
            m_start=m_start,
            key=key,
        )
        self.cells = cells.to("cpu", torch.long)


# --------------------------------------------------------------------------- #
# What a searcher builds once per seeded stage
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class SeedResources:
    """A seeded stage's seed templates and neighborhoods.

    Attributes
    ----------
    spec : Seeding
        The stage's seeding.
    local : LocalHypotheses
        The neighborhoods, one per seed cell (``base_level = spec.level``).
    space : HypothesisSpace
        The seed pass's hypothesis space (the template's own for ``"subset"``
        seeds, ``FF x C`` cell templates for ``"cell_mean"``).
    weights : TensorWeights or None
        The seed templates (``None``: the template's own weights).
    template_ids : torch.Tensor or None
        Template rows of ``space`` the seed pass searches (``None`` = all).
    cells : torch.Tensor
        ``(space.n_orientations,)`` int64 seed cell of each orientation of ``space``.
    """

    spec: Seeding
    local: LocalHypotheses
    space: HypothesisSpace
    weights: TensorWeights | None
    template_ids: torch.Tensor | None
    cells: torch.Tensor

    @property
    def n_cells(self) -> int:
        """Cells (groups) of the local pass."""
        return int(self.local.rows.shape[0])

    @property
    def rows_key(self) -> tuple:
        """Identity of the neighborhoods for the correlator's caches."""
        return (
            "seeding",
            self.local.level,
            self.local.base_level,
            self.local.rings,
            self.n_cells,
            self.local.width,
        )

    @classmethod
    @torch.no_grad()
    def build(
        cls,
        weights: TensorWeights,
        layout: FeatureLayout,
        spec: Seeding,
        space: HypothesisSpace,
        nside: int,
        ipix: np.ndarray,
        *,
        level: int,
        template_ids: torch.Tensor | None,
        n_psi: int,
    ) -> SeedResources:
        """Build for a stage.

        Parameters
        ----------
        weights : TensorWeights
            The template's weights.
        layout : FeatureLayout
            The stage's rectangles (the seed templates cover their bounding box).
        spec : Seeding
            The stage's seeding.
        space : HypothesisSpace
            The searcher's hypothesis space.
        nside, ipix : int, numpy.ndarray
            The template's HEALPix grid and its orientations' NESTED pixel ids.
        level : int
            The stage's orientation sampling level.
        template_ids : torch.Tensor, optional
            The stage's selected template rows (``None`` = all).
        n_psi : int
            The stage's in-plane samples (the cell-mean alignment grid).
        """
        if spec.level <= level:
            raise ValueError(
                f"seeding level {spec.level} must exceed the stage's orientation "
                f"level {level}"
            )
        local = LocalHypotheses.build(
            space,
            nside,
            ipix,
            level=level,
            seed_level=spec.level,
            rings=spec.rings,
            template_ids=template_ids,
        )
        if spec.templates == "subset":
            seed_ids = healpix_level_ids(ipix, spec.level)
            return cls(
                spec,
                local,
                space,
                None,
                template_rows_for(space, seed_ids),
                local.cell_of.clone(),
            )
        rects = layout.rectangles
        k0 = min(r[0] for r in rects)
        m0 = min(r[1] for r in rects)
        k1 = max(r[0] + r[2] for r in rects)
        m1 = max(r[1] + r[3] for r in rects)
        mask = torch.zeros((k1 - k0, m1 - m0), dtype=torch.bool)
        for rk, rm, ke, me in rects:
            mask[rk - k0 : rk - k0 + ke, rm - m0 : rm - m0 + me] = True
        us = weights.block((k0, m0, k1 - k0, m1 - m0))
        n_cells = int(local.rows.shape[0])
        templates, cells = cell_mean_templates(
            us,
            local.cell_of,
            n_cells,
            n_psi,
            n_filters=space.n_filters,
            cell_mask=None if bool(mask.all()) else mask,
        )
        seed_weights = CellMeanWeights(
            templates,
            cells=cells,
            n_filters=space.n_filters,
            k_start=k0,
            m_start=m0,
            key=("cell_mean", tuple(rects), spec.level, int(cells.numel()), n_psi),
        )
        seed_space = HypothesisSpace(
            n_defocus=space.n_defocus,
            n_filters=space.n_filters,
            n_orientations=int(cells.numel()),
            defocus_values=space.defocus_values,
        )
        return cls(spec, local, seed_space, seed_weights, None, cells)

    def seed_cells(self, best_global: torch.Tensor) -> torch.Tensor:
        """Seed cell of each pixel from the seed pass's best global hypothesis ids."""
        orientation = self.space.decode(best_global)[2]
        return self.cells.to(orientation.device)[orientation]


# --------------------------------------------------------------------------- #
# Lattices of seeded pixels and the local pass's entries
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class LatticeSelection:
    """Which columns of a band the seed pass scores, and each pixel's candidates.

    Attributes
    ----------
    columns : torch.Tensor
        ``(S,)`` int64 band columns seeded, ascending.
    candidates : torch.Tensor
        ``(C, P)`` int64 positions into ``columns`` of each pixel's candidate seeds
        (``-1`` = none); every pixel has at least one.
    """

    columns: torch.Tensor
    candidates: torch.Tensor

    @property
    def n_seeded(self) -> int:
        """``S``."""
        return int(self.columns.numel())

    @classmethod
    def build(
        cls, geometry: SearchGeometry, pixels: PixelSelection, lattice: int
    ) -> LatticeSelection:
        """The lattice of ``pixels`` (a sorted selection, one band).

        Every ``lattice``-th row and column of the grid is seeded; a pixel's
        candidates are the lattice pixels at the corners of the lattice cell it lies
        in, when they belong to the selection. A pixel with none (the selection's
        edge, a sparse mask, center-mode images) is seeded itself.
        """
        index = pixels.index.to("cpu", torch.long)
        n = int(index.numel())
        own = torch.arange(n)
        if lattice <= 1 or geometry.mode != "grid" or n == 0:
            return cls(own, own[None])
        image, rows, cols = geometry.positions(index)
        gh, gw = geometry.grid
        per_image = gh * gw
        on = (rows % lattice == 0) & (cols % lattice == 0)
        base_r, base_c = rows - rows % lattice, cols - cols % lattice
        found = []
        for dr in (0, lattice):
            for dc in (0, lattice):
                r2 = (base_r + dr).clamp(max=gh - 1)
                c2 = (base_c + dc).clamp(max=gw - 1)
                flat = image * per_image + r2 * gw + c2
                pos = torch.searchsorted(index, flat).clamp(max=n - 1)
                hit = (index[pos] == flat) & on[pos]
                found.append(torch.where(hit, pos, torch.full_like(pos, -1)))
        cand = torch.stack(found)  # (4, P) positions in the selection
        none = (cand < 0).all(0)
        on = on | none
        cand[0] = torch.where(none, own, cand[0])
        columns = torch.nonzero(on).squeeze(-1)
        lat_of = torch.full((n,), -1, dtype=torch.long)
        lat_of[columns] = torch.arange(int(columns.numel()))
        candidates = torch.where(
            cand >= 0, lat_of[cand.clamp(min=0)], torch.full_like(cand, -1)
        )
        return cls(columns, candidates)


def seed_work(
    resources: SeedResources,
    lattice: LatticeSelection,
    seed_hyps: HypothesisSet,
    seed_maps: StageMaps,
) -> LocalWork:
    """The local pass's entries from a seed pass over ``lattice.columns``.

    Parameters
    ----------
    resources : SeedResources
        The stage's seeding.
    lattice : LatticeSelection
        The band's seeded columns and each pixel's candidates.
    seed_hyps : HypothesisSet
        The seed pass's hypotheses (decode its virtual ids).
    seed_maps : StageMaps
        The seed pass's maps over ``lattice.columns`` (same order).
    """
    device = seed_maps.mip.device
    if int(seed_maps.mip.numel()) != lattice.n_seeded:
        raise ValueError("seed maps must cover exactly the lattice's columns")
    cell_sub = resources.seed_cells(seed_hyps.decode(seed_maps.best_virtual))  # (S,)
    cand = lattice.candidates.to(device)
    valid = cand >= 0
    safe = cand.clamp(min=0)
    cells = cell_sub[safe]  # (C, P)
    k = resources.spec.top_k
    if k is not None and k < int(cand.shape[0]):
        neg = torch.full_like(seed_maps.zscore, float("-inf"))[:1]
        score = torch.where(valid, seed_maps.zscore[safe], neg)
        pick = score.topk(k, dim=0).indices
        cells = cells.gather(0, pick)
        valid = valid.gather(0, pick)
    n_cells = resources.n_cells
    n_pixels = int(cand.shape[1])
    pixel = torch.arange(n_pixels, device=device)[None].expand_as(cells)
    key = torch.where(valid, pixel * n_cells + cells, torch.full_like(cells, -1))
    key = key.reshape(-1)
    key = torch.unique(key[key >= 0])
    columns, entry_cells = key // n_cells, key % n_cells
    counts = resources.local.counts.to(device)
    if bool((counts[entry_cells] == 0).any()):
        raise ValueError("a seed cell's neighborhood holds no orientation of the stage")
    return LocalWork(entry_cells, resources.local.rows, columns)
