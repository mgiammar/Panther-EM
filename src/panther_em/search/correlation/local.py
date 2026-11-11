"""Local correlation: each pixel against its own neighborhood of hypotheses.

A local stage scores every pixel over a small neighborhood of hypotheses instead of
the stage's whole set (:class:`~..hypotheses.LocalHypotheses`), and pixels whose
neighborhoods coincide form a *group*. Per pass:

1. **moments** -- the mean and variance over the stage's *whole* hypothesis set come
   from the closed-form quadratic forms of the features (:mod:`.moments`), one pass
   over the store's blocks;
2. **tiles** -- each group's store columns are cut into tiles of ``T`` columns (the
   last one padded by repeating a column), and :func:`choose_tile` picks ``T`` from
   the groups' sizes;
3. **contraction** -- per chunk of tiles, the tiles' features are gathered from the
   store and each tile's group weights straight from the template's weights (once
   per tile, shared by its ``T`` pixels and every plane), and one batched complex
   GEMM gives every tile's ``(T x planes) x W`` spectra;
4. **maximum** -- the inverse FFT over the in-plane angle and the per-pixel maximum
   over the neighborhood (:func:`~.reduction.max_spectrum`, the reference
   reduction's convention), decoded into the stage's virtual hypothesis ids.

The maps carry the neighborhood's maximum and the whole set's moments, so the
z-score and the decision keep their meaning; the maximum equals the full search's
wherever the full search's best hypothesis lies in the neighborhood.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from panther_em.search.correlation.maps import StageMaps
from panther_em.search.correlation.moments import (
    block_moments,
    moments_to_stats,
)
from panther_em.search.correlation.reduction import max_spectrum

if TYPE_CHECKING:
    from collections.abc import Callable

    from panther_em.search.correlation.loop import LoopPlan
    from panther_em.search.correlation.moments import MomentForms
    from panther_em.search.featurization.store import FeatureStore
    from panther_em.search.weights import WeightSource

TILE_CHOICES = (8, 16, 32, 64, 128)


@dataclass(frozen=True, eq=False)
class LocalWork:
    """A local pass's groups.

    Attributes
    ----------
    cells : torch.Tensor
        ``(P,)`` int64 group of each store column.
    rows : torch.Tensor
        ``(n_groups, W)`` int64 positions of each group's hypotheses in the stage's
        selected template rows (``LocalHypotheses.rows``).
    """

    cells: torch.Tensor
    rows: torch.Tensor

    @property
    def width(self) -> int:
        """``W``: hypotheses per pixel and plane."""
        return int(self.rows.shape[1])


def choose_tile(
    group_sizes: np.ndarray,
    *,
    n_planes: int,
    width: int,
    cells: int,
    n_freq: int,
    n_psi: int,
    choices: tuple[int, ...] = TILE_CHOICES,
) -> int:
    """The tile size with the least memory traffic over the groups."""
    sizes = np.asarray(group_sizes, dtype=np.int64)
    per_column = n_planes * (16 * cells + width * (24 * n_freq + 8 * n_psi))
    per_tile = 16 * width * cells
    best, best_cost = choices[0], None
    for t in choices:
        n_tiles = int((-(-sizes // t)).sum())
        cost = n_tiles * (t * per_column + per_tile)
        if best_cost is None or cost < best_cost:
            best, best_cost = t, cost
    return best


def make_tiles(
    cells: torch.Tensor, tile: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(columns, valid, tile_cell)`` of ``cells`` groups cut into ``tile`` columns."""
    cells = cells.to("cpu", torch.long)
    n = int(cells.numel())

    if n == 0:
        empty = torch.empty(0, tile, dtype=torch.long)
        return empty, empty.bool(), torch.empty(0, dtype=torch.long)

    order = torch.sort(cells, stable=True).indices
    sorted_cells = cells[order]
    ids, sizes = torch.unique_consecutive(sorted_cells, return_counts=True)
    starts = torch.cumsum(sizes, 0) - sizes
    group = torch.repeat_interleave(torch.arange(ids.numel()), sizes)
    rank = torch.arange(n) - starts[group]
    tiles_per_group = -(-sizes // tile)
    tile_offset = torch.cumsum(tiles_per_group, 0) - tiles_per_group
    tile_id = tile_offset[group] + rank // tile
    n_tiles = int(tiles_per_group.sum())
    columns = torch.full((n_tiles, tile), -1, dtype=torch.long)
    columns[tile_id, rank % tile] = order
    valid = columns >= 0
    columns = torch.where(valid, columns, columns[:, :1])
    tile_cell = torch.repeat_interleave(ids, tiles_per_group)

    return columns, valid, tile_cell


@torch.no_grad()
def correlate_local(
    store: FeatureStore,
    plan: LoopPlan,
    work: LocalWork,
    weights: WeightSource,
    forms: MomentForms,
    *,
    chunk_bytes: int,
    progress: Callable[[int], None] | None = None,
    out: StageMaps | None = None,
) -> StageMaps:
    """Score every store column over its group's hypotheses (see the module notes).

    Parameters
    ----------
    store : FeatureStore
        The pass's features.
    plan : LoopPlan
        The pass (layout, groups, plane lookup, hypotheses: the whole set).
    work : LocalWork
        The columns' groups and the groups' hypotheses.
    weights : WeightSource
        The template's weights.
    forms : MomentForms
        The closed-form moments of the pass's whole template set.
    chunk_bytes : int
        Working-memory budget of one chunk of tiles.
    progress : callable, optional
        ``progress(n_columns_done)``.
    out : StageMaps, optional
        Maps to fill in place (allocated by the caller, on its stream).
    """
    hyps = plan.hyps
    device = store.device
    n_psi = plan.stage.n_psi
    n_freq = plan.n_freq
    n_planes = hyps.n_defocus_selected
    n_sel = hyps.n_selected
    maps = (
        StageMaps.empty(store.n_pixels, device, hyps.n_evaluated)
        if out is None
        else out
    )

    # 1. the whole set's moments, block by block
    for block in store.blocks(plan.groups):
        start, n_valid = block.col_range[0], block.n_valid
        s1, s2 = block_moments(plan, forms, block.operands, n_valid)
        mean, variance = moments_to_stats(s1, s2, hyps.n_evaluated)
        maps.mean[start : start + n_valid] = mean
        maps.variance[start : start + n_valid] = variance

    # 2. tiles of each group's columns
    width = work.width
    cells_total = sum(run.num_k * run.num_m for run in plan.layout.k_runs)
    sizes = torch.unique(work.cells, return_counts=True)[1].numpy()
    tile = choose_tile(
        sizes,
        n_planes=n_planes,
        width=width,
        cells=cells_total,
        n_freq=n_freq,
        n_psi=n_psi,
    )
    columns, valid, tile_cell = make_tiles(work.cells, tile)
    template_rows = hyps.template_rows(device)[work.rows.to(device)]  # (G, W)
    group_rows = work.rows.to(device)
    per_tile = (
        8 * tile * n_planes * width * (3 * n_freq + n_psi)
        + 16 * (tile * n_planes + width) * cells_total
    )
    tiles_per_chunk = max(1, int(chunk_bytes) // max(1, per_tile))

    # 3-4. contraction and maximum per chunk of tiles
    bp = store.block_pixels
    for t0 in range(0, int(columns.shape[0]), tiles_per_chunk):
        cols = columns[t0 : t0 + tiles_per_chunk].to(device)
        ok = valid[t0 : t0 + tiles_per_chunk].to(device)
        cell = tile_cell[t0 : t0 + tiles_per_chunk].to(device)
        nt = int(cols.shape[0])
        b_idx, p_idx = cols // bp, cols % bp
        trows = template_rows[cell]  # (nt, W)
        spectrum = torch.zeros(
            nt, tile, n_planes, width, n_freq, dtype=torch.complex64, device=device
        )
        for run in plan.layout.k_runs:
            a_parts, w_parts = [], []
            for part in run.parts:
                rect = plan.layout.rectangles[part.rect]
                k_idx = torch.arange(
                    run.k_start - rect[0], run.k_stop - rect[0], device=device
                )
                lookup = plan.plane_lookup[part.rect]
                group = next(g for g in plan.groups if g.id == lookup[0][0])
                if any(gid != group.id for gid, _ in lookup):
                    raise ValueError("a rectangle's planes span several store groups")
                planes = torch.tensor([lp for _, lp in lookup], device=device)

                # (nt, K, D, T, inner) from the store's (n_blocks, D, K, bp, inner)
                x = group.data[
                    b_idx[:, None, None, :],
                    planes[None, None, :, None],
                    k_idx[None, :, None, None],
                    p_idx[:, None, None, :],
                ]
                if group.precision == "fp16":
                    m = int(x.shape[-1]) // 2
                    x = torch.complex(x[..., :m].float(), x[..., m:].float())
                a_parts.append(x)

                # (nt, K, W, M) straight from the weights (a view of U * S)
                us = weights.block(
                    (run.k_start, part.m_start, run.num_k, part.m_stop - part.m_start)
                )
                k_run = torch.arange(run.num_k, device=us.device)
                w_parts.append(us[trows[:, None, :], k_run[None, :, None]])

            a = a_parts[0] if len(a_parts) == 1 else torch.cat(a_parts, dim=-1)
            w = w_parts[0] if len(w_parts) == 1 else torch.cat(w_parts, dim=-1)
            a = a.reshape(nt, run.num_k, n_planes * tile, -1)
            # C[t, k, (d, i), n] = sum_m Y conj(W)

            c = torch.matmul(a, w.conj().transpose(-1, -2))
            spectrum[..., run.k_start : run.k_stop] = c.reshape(
                nt, run.num_k, n_planes, tile, width
            ).permute(0, 3, 2, 4, 1)

        vmax, amax = max_spectrum(
            spectrum.reshape(nt * tile, n_planes * width, n_freq), n_psi
        )
        slot = amax // n_psi
        w_slot = slot % width
        d_local = slot // width
        j = group_rows[cell].repeat_interleave(tile, dim=0).gather(1, w_slot[:, None])

        virtual = d_local * n_sel + j[:, 0]
        psi = (n_psi - amax % n_psi) % n_psi
        keep = ok.reshape(-1)
        target = cols.reshape(-1)[keep]

        maps.mip[target] = vmax[keep]
        maps.best_virtual[target] = virtual[keep]
        maps.best_psi[target] = psi[keep]

        if progress is not None:
            progress(int(keep.sum()))

    std = maps.variance.sqrt().clamp_min(1e-12)
    torch.div(maps.mip - maps.mean, std, out=maps.zscore)
    return maps
