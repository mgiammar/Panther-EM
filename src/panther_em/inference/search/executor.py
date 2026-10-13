"""Stage planning and execution: run one stage's hypothesis loop over a store.

:func:`plan_memory` fixes the batch/block shape before the store's groups are
allocated; :func:`plan_stage` then freezes everything the executor needs into a
:class:`StagePlan`; :class:`StageExecutor` runs the block loop and returns
:class:`StageMaps` in the store's compact column order.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from panther_em.inference.search.context import (
    LoopInputs,
    MemoryPlan,
    Residency,
    choose_fused,
)
from panther_em.inference.search.contraction import make_engine
from panther_em.inference.search.hypotheses import HypothesisSet
from panther_em.inference.search.statistics import SpectrumSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from panther_em.inference.search.context import SearchContext
    from panther_em.inference.search.contraction import ContractionEngine
    from panther_em.inference.search.hypotheses import HypothesisSpace
    from panther_em.inference.search.schedule import StageSpec
    from panther_em.inference.search.store import ColumnGroup, FeatureStore
    from panther_em.inference.search.tiling import FeatureLayout
    from panther_em.inference.search.types import BatchShape, Precision


def feature_bytes_per_pixel(
    groups: Iterable[tuple[tuple[int, int, int, int], Precision | str, int]],
) -> int:
    """Bytes one pixel's features occupy across ``(rect, precision, n_planes)``."""
    total = 0
    for rect, precision, n_planes in groups:
        engine = make_engine(precision)
        elem = torch.empty((), dtype=engine.block_dtype).element_size()
        inner = engine.block_shape(1, rect[2], rect[3], 1)[-1]
        total += int(n_planes) * rect[2] * inner * elem
    return total


def resident_groups(
    spec: StageSpec, planes: Sequence[int], store: FeatureStore | None
) -> list[tuple[tuple[int, int, int, int], Precision, int]]:
    """``(rect, precision, n_planes)`` per group the stage will have resident."""
    out: list[tuple[tuple[int, int, int, int], Precision, int]] = []
    for rect in spec.rectangles:
        if store is None:
            out.append((rect, spec.precision, len(planes)))
            continue
        seen: set[int] = set()
        for g in store.groups:
            if g.rect == rect and g.precision is spec.precision and g.id not in seen:
                if any(g.holds(p) for p in planes):
                    seen.add(g.id)
                    out.append((rect, spec.precision, g.n_defocus))
        missing = store.missing_planes(rect, spec.precision, planes)
        if missing:
            out.append((rect, spec.precision, len(missing)))
    return out


def plan_memory(
    ctx: SearchContext,
    spec: StageSpec,
    *,
    n_pixels: int,
    space: HypothesisSpace,
    template_ids: torch.Tensor | None = None,
    defocus_ids: torch.Tensor | None = None,
    batch: BatchShape | None = None,
    block_pixels: int | None = None,
    use_cuda_graph: bool = True,
    use_fused_kernel: bool = True,
    store: FeatureStore | None = None,
    residency: Residency | Literal["auto"] = "auto",
) -> tuple[MemoryPlan, bool, ContractionEngine, HypothesisSet]:
    """Resolve batch shape, block size, engine, fused flag and hypothesis set.

    Notes
    -----
    ``store`` (when given) lets the per-pixel byte estimate count the groups that
    will actually be resident, including planes held beyond the stage's own.
    """
    engine = make_engine(spec.precision)
    provisional = HypothesisSet.create(
        space, template_ids, defocus_ids, hyp_batch=1, n_psi=spec.n_psi
    )
    n_virtual = provisional.n_virtual
    planes = provisional.defocus_planes()
    per_pixel = feature_bytes_per_pixel(resident_groups(spec, planes, store))

    # The backend decides whether graphs are possible; probe it with the eventual dtype.
    probe = SpectrumSpec(
        dtype=engine.spectrum_dtype,
        n_freq=spec.k_stop,
        n_psi=spec.n_psi,
        gemm_pixels=1,
        n_sub=1,
        hyp_batch=1,
        n_virtual=n_virtual,
        device=ctx.device,
    )
    fused = choose_fused(probe, ctx.caps, use_fused_kernel=use_fused_kernel)
    memory = ctx.planner.plan(
        ctx.caps,
        n_pixels=n_pixels,
        n_freq=spec.k_stop,
        n_psi=spec.n_psi,
        n_virtual=n_virtual,
        n_defocus=len(planes),
        feature_bytes_per_pixel=per_pixel,
        spectrum_dtype=engine.spectrum_dtype,
        precision=spec.precision,
        fused=fused,
        batch=batch,
        block_pixels=block_pixels,
        use_cuda_graph=use_cuda_graph,
        residency=residency,
    )
    hyp_plan = HypothesisSet.create(
        space, template_ids, defocus_ids, hyp_batch=memory.hyp_batch, n_psi=spec.n_psi
    )
    return memory, fused, engine, hyp_plan


@dataclass(frozen=True, eq=False)
class StagePlan:
    """Everything the executor needs for one stage, resolved up front.

    Attributes
    ----------
    index : int
    spec : StageSpec
    layout : FeatureLayout
        The stage's rectangles.
    groups : tuple[ColumnGroup, ...]
        Every group the stage reads (addressed by ``group.id``).
    plane_lookup : tuple[tuple[tuple[int, int], ...], ...]
        ``[rect_index][d_local] -> (group id, local plane index)``.
    engine : ContractionEngine
    fused : bool
        Whether the statistics run on the fused CUDA kernels.
    spectrum : SpectrumSpec
    memory : MemoryPlan
    hyp_plan : HypothesisSet
    signature : tuple
        Cache identity of the hypothesis loop (layout, precision, shapes, hypotheses).
    """

    index: int
    spec: StageSpec
    layout: FeatureLayout
    groups: tuple[ColumnGroup, ...]
    plane_lookup: tuple[tuple[tuple[int, int], ...], ...]
    engine: ContractionEngine
    fused: bool
    spectrum: SpectrumSpec
    memory: MemoryPlan
    hyp_plan: HypothesisSet
    signature: tuple

    @property
    def n_freq(self) -> int:
        """``k_stop``."""
        return self.spec.k_stop


def plan_stage(
    ctx: SearchContext,
    index: int,
    spec: StageSpec,
    store: FeatureStore,
    memory: MemoryPlan,
    fused: bool,
    engine: ContractionEngine,
    hyp_plan: HypothesisSet,
) -> StagePlan:
    """Freeze a stage plan against a store whose groups already exist."""
    layout = spec.layout()
    planes = hyp_plan.defocus_planes()
    groups: list[ColumnGroup] = []
    lookup: list[tuple[tuple[int, int], ...]] = []
    for rect in layout.rectangles:
        try:
            covering = store.groups_covering(rect, spec.precision, planes)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        lookup.append(tuple((g.id, local) for g, local in covering))
        for g, _ in covering:
            if g not in groups:
                groups.append(g)
    if memory.block_pixels != store.block_pixels:
        raise ValueError(
            f"memory plan block_pixels {memory.block_pixels} != store block_pixels "
            f"{store.block_pixels}"
        )
    spectrum = SpectrumSpec(
        dtype=engine.spectrum_dtype,
        n_freq=spec.k_stop,
        n_psi=spec.n_psi,
        gemm_pixels=memory.gemm_pixels,
        n_sub=memory.n_sub,
        hyp_batch=memory.hyp_batch,
        n_virtual=hyp_plan.n_virtual,
        device=ctx.device,
    )
    signature = (
        layout.rectangles,
        spec.precision,
        tuple((g.id, g.rect, g.precision, g.planes) for g in groups),
        tuple(lookup),
        memory.gemm_pixels,
        memory.block_pixels,
        memory.hyp_batch,
        spec.n_psi,
        hyp_plan.signature,
        fused,
        memory.loop,
    )
    return StagePlan(
        index=index,
        spec=spec,
        layout=layout,
        groups=tuple(groups),
        plane_lookup=tuple(lookup),
        engine=engine,
        fused=fused,
        spectrum=spectrum,
        memory=memory,
        hyp_plan=hyp_plan,
        signature=signature,
    )


@dataclass(eq=False)
class StageMaps:
    """Per-pixel results of one stage in the store's compact column order.

    Notes
    -----
    ``best_virtual`` holds *virtual* hypothesis ids (see
    :mod:`~panther_em.inference.search.hypotheses`); decode with ``hyp_plan.decode``.
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


class StageExecutor:
    """Runs one stage's block loop over a store."""

    def __init__(self, ctx: SearchContext) -> None:
        self.ctx = ctx

    @torch.no_grad()
    def run(
        self,
        store: FeatureStore,
        plan: StagePlan,
        *,
        progress: Callable[[int], None] | None = None,
    ) -> StageMaps:
        """Reduce every block of ``store`` for ``plan``; ``progress(n_pixels_done)``."""
        ctx = self.ctx
        rows = (
            plan.hyp_plan.template_rows(ctx.device)
            if not plan.hyp_plan.is_contiguous
            else None
        )
        rows_key = plan.hyp_plan.signature if rows is not None else None
        prepared = {
            (run_index, batch_index): ctx.prepared.get(
                plan.layout, run, batch, plan.engine, rows, rows_key
            )
            for run_index, run in enumerate(plan.layout.k_runs)
            for batch_index, batch in enumerate(plan.hyp_plan.batches)
        }
        inputs = LoopInputs(
            layout=plan.layout,
            plane_lookup=plan.plane_lookup,
            engine=plan.engine,
            hyp_plan=plan.hyp_plan,
            prepared=prepared,
            memory=plan.memory,
            n_freq=plan.n_freq,
        )
        maps = StageMaps.empty(store.n_pixels, ctx.device, plan.hyp_plan.n_evaluated)
        targets = maps.as_targets()
        loop = None
        stream: contextlib.AbstractContextManager[object] = (
            torch.cuda.stream(ctx.compute)
            if ctx.compute is not None
            else contextlib.nullcontext()
        )
        with stream:
            if ctx.compute is not None:
                # Inputs (store blocks, prepared weights) came from other streams.
                ctx.compute.wait_stream(torch.cuda.current_stream(ctx.device))
            for block in store.blocks(plan.groups):
                if loop is None:
                    loop = ctx.loop_for(
                        plan.signature,
                        inputs,
                        plan.fused,
                        plan.spectrum,
                        block.operands,
                    )
                if block.ready is not None:
                    torch.cuda.current_stream(ctx.device).wait_event(block.ready)
                loop.run(block.operands, block.n_valid)
                start, n_valid = block.col_range[0], block.n_valid
                loop.stats.finalize_into(targets, start, n_valid)
                block.release(None)
                if progress is not None:
                    progress(block.n_valid)
        if ctx.compute is not None:
            # Hand the maps back to the caller's stream.
            torch.cuda.current_stream(ctx.device).wait_stream(ctx.compute)
        return maps
