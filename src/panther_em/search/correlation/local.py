"""Local correlation: each pixel against its own neighborhood of hypotheses.

A local pass scores pixels over small neighborhoods of hypotheses instead of the
stage's whole set (:class:`~..hypotheses.LocalHypotheses`). Its unit of work is an
*entry*: a store column paired with a neighborhood (a *group*); a column may appear
in several entries (several seed cells, see :mod:`~panther_em.search.seeding`) and
then takes the maximum over them. Per pass:

1. **moments** -- the mean and variance over the stage's *whole* hypothesis set come
   from the closed-form quadratic forms of the features (:mod:`.moments`), one pass
   over the store's blocks;
2. **tiles** -- the entries of each group are cut into tiles of ``T`` (the last one
   padded by repeating an entry), and :func:`choose_tile` picks ``T`` from the
   groups' sizes;
3. **contraction** -- per chunk of tiles, the tiles' features are gathered from the
   store and contracted against each tile's group weights, shared by its ``T``
   entries and every plane. Two paths: the fp16 *lean* path (an fp16 store on a
   device with the fused kernels, with the groups' weights prepared once per stage
   as :class:`LocalWeights`) runs one batched fp16 GEMM and one fused iRFFT +
   maximum kernel per chunk; the *reference* path gathers the weights per tile
   straight from the template and reduces with :func:`~.reduction.max_spectrum`;
4. **maximum** -- the per-entry maximum over the neighborhood and the planes,
   decoded into the stage's virtual hypothesis ids and written as the maximum over
   each column's entries.

The maps carry the neighborhood's maximum and the whole set's moments, so the
z-score and the decision keep their meaning; the maximum equals the full search's
wherever the full search's best hypothesis lies in one of the column's neighborhoods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from panther_em.search.correlation.contraction import Fp16GemmEngine, Precision
from panther_em.search.correlation.kernels import fused_kernel_loader
from panther_em.search.correlation.maps import StageMaps
from panther_em.search.correlation.moments import (
    block_moments,
    moments_to_stats,
)
from panther_em.search.correlation.reduction import max_spectrum

if TYPE_CHECKING:
    from collections.abc import Callable

    from panther_em.search.correlation.layout import FeatureLayout
    from panther_em.search.correlation.loop import LoopPlan
    from panther_em.search.correlation.moments import MomentForms
    from panther_em.search.featurization.store import ColumnGroup, StoreLike
    from panther_em.search.weights import TensorWeights

TILE_CHOICES = (8, 16, 32, 64, 128)


@dataclass(frozen=True, eq=False)
class LocalWork:
    """A local pass's entries: which column is scored over which group.

    Attributes
    ----------
    cells : torch.Tensor
        ``(E,)`` int64 group of each entry.
    rows : torch.Tensor
        ``(n_groups, W)`` int64 positions of each group's hypotheses in the stage's
        selected template rows (``LocalHypotheses.rows``).
    columns : torch.Tensor, optional
        ``(E,)`` int64 store column of each entry; ``None`` when entry ``e`` is
        column ``e`` (one entry per column). Every store column needs at least one
        entry.
    """

    cells: torch.Tensor
    rows: torch.Tensor
    columns: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate shapes (never coerce)."""
        if self.cells.dim() != 1 or self.rows.dim() != 2:
            raise ValueError("cells must be (E,) and rows (n_groups, W)")
        if self.columns is not None and (
            self.columns.dim() != 1 or self.columns.numel() != self.cells.numel()
        ):
            raise ValueError("columns must be (E,), one store column per entry")

    @property
    def width(self) -> int:
        """``W``: hypotheses per entry and plane."""
        return int(self.rows.shape[1])

    @property
    def n_entries(self) -> int:
        """``E``."""
        return int(self.cells.numel())

    def column_of(self, device: torch.device | str) -> torch.Tensor:
        """``(E,)`` store column of each entry, on ``device``."""
        if self.columns is None:
            return torch.arange(self.n_entries, device=device)
        return self.columns.to(device, torch.long)


@dataclass(frozen=True, eq=False)
class LocalWeights:
    """Every group's weights prepared for the fp16 lean path, one tensor per k-run.

    Attributes
    ----------
    runs : tuple[torch.Tensor, ...]
        Run ``i``: ``(num_k_i, n_groups, inner_i, 2W)`` float16, the fp16 engine's
        4M B operand of each group's ``W`` hypotheses (``inner_i = 2 * sum of the
        run's part widths``).
    width : int
        ``W``.
    """

    runs: tuple[torch.Tensor, ...]
    width: int

    @property
    def n_groups(self) -> int:
        """Groups prepared."""
        return int(self.runs[0].shape[1])

    @property
    def nbytes(self) -> int:
        """Resident bytes."""
        return sum(t.numel() * t.element_size() for t in self.runs)

    @staticmethod
    def estimate_bytes(layout: FeatureLayout, n_groups: int, width: int) -> int:
        """Bytes :meth:`build` would allocate for ``layout``."""
        cells = sum(run.num_k * run.num_m for run in layout.k_runs)
        return 2 * cells * 2 * int(n_groups) * 2 * int(width)

    @classmethod
    @torch.no_grad()
    def build(
        cls,
        weights: TensorWeights,
        plan: LoopPlan,
        rows: torch.Tensor,
        *,
        chunk: int = 256,
    ) -> LocalWeights:
        """Prepare the ``(n_groups, W)`` ``rows`` (positions in ``plan``'s rows).

        Parameters
        ----------
        weights : TensorWeights
            The template's weights.
        plan : LoopPlan
            The pass (its layout, fp16 engine and selected template rows).
        rows : torch.Tensor
            ``(n_groups, W)`` positions in the pass's template rows
            (``LocalHypotheses.rows``).
        chunk : int, optional
            Groups prepared per step.
        """
        engine = plan.engine
        if not isinstance(engine, Fp16GemmEngine):
            raise ValueError("LocalWeights are the fp16 engine's operands")
        device = weights.device
        trows = plan.hyps.template_rows(device)[rows.to(device, torch.long)]
        n_groups, width = (int(s) for s in trows.shape)
        out = []
        for run in plan.layout.k_runs:
            inner = 2 * run.num_m
            t = torch.empty(
                (run.num_k, n_groups, inner, 2 * width),
                dtype=torch.float16,
                device=device,
            )
            for g0 in range(0, n_groups, int(chunk)):
                sub = trows[g0 : g0 + int(chunk)]
                g = int(sub.shape[0])
                parts = [
                    engine.prepare_weights(
                        weights.block(
                            (
                                run.k_start,
                                part.m_start,
                                run.num_k,
                                part.m_stop - part.m_start,
                            ),
                            sub.reshape(-1),
                        )
                    )
                    for part in run.parts
                ]  # each (num_k, 2 m_part, 2 g W); columns (group, w, re/im)
                b = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
                t[:, g0 : g0 + g] = b.reshape(run.num_k, inner, g, 2 * width).permute(
                    0, 2, 1, 3
                )
            out.append(t)
        return cls(tuple(out), width)


def _best_tile(
    group_sizes: np.ndarray, per_column: int, per_tile: int, choices: tuple[int, ...]
) -> int:
    sizes = np.asarray(group_sizes, dtype=np.int64)
    best, best_cost = choices[0], None
    for t in choices:
        n_tiles = int((-(-sizes // t)).sum())
        cost = n_tiles * (t * per_column + per_tile)
        if best_cost is None or cost < best_cost:
            best, best_cost = t, cost
    return best


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
    """The tile size with the least memory traffic over the groups (reference path)."""
    per_column = n_planes * (16 * cells + width * (24 * n_freq + 8 * n_psi))
    per_tile = 16 * width * cells
    return _best_tile(group_sizes, per_column, per_tile, choices)


def choose_tile_lean(
    group_sizes: np.ndarray,
    *,
    n_planes: int,
    width: int,
    cells: int,
    n_freq: int,
    choices: tuple[int, ...] = TILE_CHOICES,
) -> int:
    """The tile size with the least memory traffic over the groups (lean path)."""
    per_column = n_planes * (8 * cells + 8 * width * n_freq)
    per_tile = 8 * cells * width
    return _best_tile(group_sizes, per_column, per_tile, choices)


def make_tiles(
    cells: torch.Tensor, tile: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(entries, valid, tile_cell)`` of ``cells`` groups cut into ``tile`` entries."""
    cells = cells.to(torch.long)
    device = cells.device
    n = int(cells.numel())

    if n == 0:
        empty = torch.empty(0, tile, dtype=torch.long, device=device)
        return empty, empty.bool(), torch.empty(0, dtype=torch.long, device=device)

    order = torch.sort(cells, stable=True).indices
    sorted_cells = cells[order]
    ids, sizes = torch.unique_consecutive(sorted_cells, return_counts=True)
    starts = torch.cumsum(sizes, 0) - sizes
    group = torch.repeat_interleave(torch.arange(ids.numel(), device=device), sizes)
    rank = torch.arange(n, device=device) - starts[group]
    tiles_per_group = -(-sizes // tile)
    tile_offset = torch.cumsum(tiles_per_group, 0) - tiles_per_group
    tile_id = tile_offset[group] + rank // tile
    n_tiles = int(tiles_per_group.sum())
    entries = torch.full((n_tiles, tile), -1, dtype=torch.long, device=device)
    entries[tile_id, rank % tile] = order
    valid = entries >= 0
    entries = torch.where(valid, entries, entries[:, :1])
    tile_cell = torch.repeat_interleave(ids, tiles_per_group)

    return entries, valid, tile_cell


def _store_group(plan: LoopPlan, rect_index: int) -> tuple[ColumnGroup, torch.Tensor]:
    """The store group a rectangle's planes live in and their local plane indices."""
    lookup = plan.plane_lookup[rect_index]
    group = next(g for g in plan.groups if g.id == lookup[0][0])
    if any(gid != group.id for gid, _ in lookup):
        raise ValueError("a rectangle's planes span several store groups")
    planes = torch.tensor([lp for _, lp in lookup], device=group.data.device)
    return group, planes


def _reference_chunk(
    plan: LoopPlan,
    weights: TensorWeights,
    template_rows: torch.Tensor,
    group_rows: torch.Tensor,
    b_idx: torch.Tensor,
    p_idx: torch.Tensor,
    cell: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(vmax, virtual, psi)`` of one chunk of tiles, flattened ``(tile, entry)``."""
    hyps = plan.hyps
    device = b_idx.device
    n_psi, n_freq = plan.stage.n_psi, plan.n_freq
    n_planes, n_sel = hyps.n_defocus_selected, hyps.n_selected
    nt, tile = (int(s) for s in b_idx.shape)
    width = int(group_rows.shape[1])
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
            group, planes = _store_group(plan, part.rect)

            # (nt, K, D, T, inner) from the store's (n_blocks, D, K, bp, inner)
            x = group.data[
                b_idx[:, None, None, :],
                planes[None, None, :, None],
                k_idx[None, :, None, None],
                p_idx[:, None, None, :],
            ]
            if group.precision is Precision.FP16:
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
    return vmax, virtual, psi


def _lean_chunk(
    plan: LoopPlan,
    prepared: LocalWeights,
    group_rows: torch.Tensor,
    b_idx: torch.Tensor,
    p_idx: torch.Tensor,
    cell: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(vmax, virtual, psi)`` of one chunk of tiles on the fp16 lean path."""
    hyps = plan.hyps
    device = b_idx.device
    n_psi, n_freq = plan.stage.n_psi, plan.n_freq
    n_planes, n_sel = hyps.n_defocus_selected, hyps.n_selected
    nt, tile = (int(s) for s in b_idx.shape)
    width = prepared.width
    n_rows = nt * n_planes * tile
    runs = plan.layout.k_runs
    whole = len(runs) == 1 and runs[0].k_start == 0 and runs[0].k_stop == n_freq
    c_full = (
        None
        if whole
        else torch.zeros(n_freq, n_rows, 2 * width, dtype=torch.float16, device=device)
    )
    for run_index, run in enumerate(runs):
        a_parts = []
        for part in run.parts:
            rect = plan.layout.rectangles[part.rect]
            k_idx = torch.arange(
                run.k_start - rect[0], run.k_stop - rect[0], device=device
            )
            group, planes = _store_group(plan, part.rect)
            # (K, nt, D, T, inner) fp16, straight from the store
            a_parts.append(
                group.data[
                    b_idx[None, :, None, :],
                    planes[None, None, :, None],
                    k_idx[:, None, None, None],
                    p_idx[None, :, None, :],
                ]
            )
        a = a_parts[0] if len(a_parts) == 1 else torch.cat(a_parts, dim=-1)
        a = a.reshape(run.num_k * nt, n_planes * tile, -1)
        b = prepared.runs[run_index][:, cell]  # (K, nt, inner, 2W)
        b = b.reshape(run.num_k * nt, int(b.shape[2]), 2 * width)
        c = torch.bmm(a, b).view(run.num_k, n_rows, 2 * width)
        if c_full is None:
            c_full = c
        else:
            c_full[run.k_start : run.k_stop] = c
    assert c_full is not None
    res = fused_kernel_loader.lean_irfft_stats_transposed(
        c_full, n_psi, decode=True, hyp_offset=0
    )
    if res is None:
        raise RuntimeError(
            f"the fused kernel rejected a {tuple(c_full.shape)} float16 spectrum at "
            f"n_psi={n_psi}"
        )
    _, _, vmax, amax = res
    vmax = vmax.view(nt, n_planes, tile)
    amax = amax.view(nt, n_planes, tile)
    best, d_best = vmax.max(dim=1)  # (nt, T) over the planes
    amax_best = amax.gather(1, d_best[:, None, :])[:, 0]
    slot = amax_best // n_psi
    psi = (n_psi - amax_best % n_psi) % n_psi
    j = group_rows[cell].gather(1, slot)  # (nt, T)
    virtual = d_best * n_sel + j
    return best.reshape(-1), virtual.reshape(-1), psi.reshape(-1)


def _write_max(
    maps: StageMaps,
    target: torch.Tensor,
    value: torch.Tensor,
    virtual: torch.Tensor,
    psi: torch.Tensor,
) -> None:
    """Keep the maximum at each ``target`` column (columns may repeat)."""
    maps.mip.scatter_reduce_(0, target, value, "amax")
    win = value >= maps.mip[target]
    maps.best_virtual[target[win]] = virtual[win]
    maps.best_psi[target[win]] = psi[win]


def lean_path_applies(plan: LoopPlan, prepared: LocalWeights | None) -> bool:
    """Whether a pass runs the fp16 lean contraction with ``prepared`` weights."""
    return (
        prepared is not None
        and isinstance(plan.engine, Fp16GemmEngine)
        and plan.fused
        and plan.memory.device.type == "cuda"
    )


@torch.no_grad()
def correlate_local(
    store: StoreLike,
    plan: LoopPlan,
    work: LocalWork,
    weights: TensorWeights,
    forms: MomentForms,
    *,
    chunk_bytes: int,
    progress: Callable[[int], None] | None = None,
    out: StageMaps | None = None,
    prepared: LocalWeights | None = None,
) -> StageMaps:
    """Score every entry over its group's hypotheses (see the module notes).

    Parameters
    ----------
    store : StoreLike
        The pass's features.
    plan : LoopPlan
        The pass (layout, groups, plane lookup, hypotheses: the whole set).
    work : LocalWork
        The entries: their columns and groups, and the groups' hypotheses.
    weights : TensorWeights
        The template's weights (the reference path reads them per tile).
    forms : MomentForms
        The closed-form moments of the pass's whole template set.
    chunk_bytes : int
        Working-memory budget of one chunk of tiles.
    progress : callable, optional
        ``progress(n_entries_done)``.
    out : StageMaps, optional
        Maps to fill in place (allocated by the caller, on its stream).
    prepared : LocalWeights, optional
        The groups' prepared fp16 weights; with an fp16 store on a device with the
        fused kernels the pass takes the lean path.
    """
    hyps = plan.hyps
    device = store.device
    n_psi = plan.stage.n_psi
    n_freq = plan.n_freq
    n_planes = hyps.n_defocus_selected
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

    # 2. tiles of each group's entries
    width = work.width
    lean = lean_path_applies(plan, prepared)
    if lean:
        assert prepared is not None
        if prepared.width != width:
            raise ValueError("prepared weights and work disagree on the width W")
    cells_total = sum(run.num_k * run.num_m for run in plan.layout.k_runs)
    sizes = torch.unique(work.cells, return_counts=True)[1].cpu().numpy()
    if lean:
        tile = choose_tile_lean(
            sizes, n_planes=n_planes, width=width, cells=cells_total, n_freq=n_freq
        )
        per_tile = 8 * tile * n_planes * (cells_total + width * n_freq) + (
            8 * cells_total * width
        )
    else:
        tile = choose_tile(
            sizes,
            n_planes=n_planes,
            width=width,
            cells=cells_total,
            n_freq=n_freq,
            n_psi=n_psi,
        )
        per_tile = (
            8 * tile * n_planes * width * (3 * n_freq + n_psi)
            + 16 * (tile * n_planes + width) * cells_total
        )
    entries, valid, tile_cell = make_tiles(work.cells, tile)
    column_of = work.column_of(device)
    template_rows = hyps.template_rows(device)[work.rows.to(device)]  # (G, W)
    group_rows = work.rows.to(device)
    tiles_per_chunk = max(1, int(chunk_bytes) // max(1, per_tile))

    # 3-4. contraction and maximum per chunk of tiles; the maximum over a column's
    # entries
    maps.mip.fill_(float("-inf"))
    bp = store.block_pixels
    for t0 in range(0, int(entries.shape[0]), tiles_per_chunk):
        ent = entries[t0 : t0 + tiles_per_chunk].to(device)
        ok = valid[t0 : t0 + tiles_per_chunk].to(device)
        cell = tile_cell[t0 : t0 + tiles_per_chunk].to(device)
        cols = column_of[ent]
        b_idx, p_idx = cols // bp, cols % bp
        if lean:
            assert prepared is not None
            vmax, virtual, psi = _lean_chunk(
                plan, prepared, group_rows, b_idx, p_idx, cell
            )
        else:
            vmax, virtual, psi = _reference_chunk(
                plan, weights, template_rows, group_rows, b_idx, p_idx, cell
            )
        keep = ok.reshape(-1)
        target = cols.reshape(-1)[keep]
        _write_max(maps, target, vmax[keep], virtual[keep], psi[keep])
        if progress is not None:
            progress(int(keep.sum()))

    if not bool(torch.isfinite(maps.mip).all()):
        raise ValueError("every store column needs at least one entry")
    std = maps.variance.sqrt().clamp_min(1e-12)
    torch.div(maps.mip - maps.mean, std, out=maps.zscore)
    return maps
