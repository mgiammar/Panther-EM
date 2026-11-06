"""The block loop: per GEMM sub-batch and hypothesis batch, CGEMM then reduction.

:class:`HypLoop` reduces one pixel block of features against every hypothesis: one
contraction into a static spectrum buffer and one statistics update per (sub-batch,
hypothesis batch, k-run); on CUDA with the fused kernels the whole block loop is
captured once as a CUDA graph and replayed per block. :class:`LoopPlan` is what the
loop reads: a pass's shapes (its memory plan) resolved against the store whose blocks
it will see.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any

import torch

from panther_em.inference.search.correlation.contraction import Spectrum
from panther_em.inference.search.correlation.reduction import SpectrumSpec

if TYPE_CHECKING:
    from panther_em.inference.search.correlation.contraction import ContractionEngine
    from panther_em.inference.search.correlation.layout import FeatureLayout
    from panther_em.inference.search.correlation.reduction import BlockStats
    from panther_em.inference.search.featurization.store import (
        ColumnGroup,
        FeatureStore,
    )
    from panther_em.inference.search.hypotheses import HypothesisSet
    from panther_em.inference.search.planning import MemoryPlan
    from panther_em.inference.search.stage import Stage


@functools.cache
def cuda_graph_capture_supported(device_index: int) -> bool:
    """Whether a CUDA graph captured on device ``device_index`` replays correctly.

    Probed once per device. The capture runs on a private stream bound to the device:
    a capture on the lazily created process-wide default capture stream records
    nothing (and replays as a no-op) when that stream belongs to another device.
    """
    try:
        device = torch.device("cuda", device_index)
        x = torch.zeros(1, device=device)
        stream = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            x.add_(1)  # warm-up
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            x.add_(1)
        graph.replay()
        torch.cuda.synchronize(device)
        return bool(x.item() == 2.0)
    except Exception:
        return False


@dataclass(frozen=True, eq=False)
class LoopPlan:
    """Everything the block loop needs for one pass, resolved against its store.

    Build with :meth:`create`.

    Attributes
    ----------
    stage : Stage
        The stage the pass belongs to.
    layout : FeatureLayout
        The stage's rectangles.
    groups : tuple[ColumnGroup, ...]
        Every store group the pass reads (addressed by ``group.id``).
    plane_lookup : tuple[tuple[tuple[int, int], ...], ...]
        ``[rect_index][d_local] -> (group id, local plane index)``.
    memory : MemoryPlan
        The pass's shapes.
    signature : tuple
        Cache identity of the hypothesis loop (layout, precision, shapes, hypotheses).
    """

    stage: Stage
    layout: FeatureLayout
    groups: tuple[ColumnGroup, ...]
    plane_lookup: tuple[tuple[tuple[int, int], ...], ...]
    memory: MemoryPlan
    signature: tuple

    @classmethod
    def create(cls, stage: Stage, store: FeatureStore, memory: MemoryPlan) -> LoopPlan:
        """Resolve a pass's memory plan against a store whose groups already exist."""
        layout = stage.layout()
        planes = memory.hyps.defocus_planes()
        groups: list[ColumnGroup] = []
        lookup: list[tuple[tuple[int, int], ...]] = []
        for rect in layout.rectangles:
            try:
                covering = store.groups_covering(rect, stage.precision, planes)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            lookup.append(tuple((g.id, local) for g, local in covering))
            for g, _ in covering:
                if g not in groups:
                    groups.append(g)
        if memory.block_pixels != store.block_pixels:
            raise ValueError(
                f"memory plan block_pixels {memory.block_pixels} != store "
                f"block_pixels {store.block_pixels}"
            )
        signature = (
            layout.rectangles,
            stage.precision,
            # a group's block shape is set by its plane count, not by which planes
            tuple((g.id, g.rect, g.precision, g.n_defocus) for g in groups),
            tuple(lookup),
            memory.gemm_pixels,
            memory.block_pixels,
            memory.hyp_batch,
            stage.n_psi,
            # which planes a pass searches is irrelevant to the loop: it reads the
            # groups/lookup above, and the batches depend on the plane *count* only
            memory.hyps.template_signature,
            memory.hyps.n_defocus_selected,
            memory.fused,
            memory.loop,
            str(memory.device),
        )
        return cls(
            stage=stage,
            layout=layout,
            groups=tuple(groups),
            plane_lookup=tuple(lookup),
            memory=memory,
            signature=signature,
        )

    @property
    def engine(self) -> ContractionEngine:
        """The stage's GEMM engine."""
        return self.memory.engine

    @property
    def hyps(self) -> HypothesisSet:
        """The pass's batched hypotheses."""
        return self.memory.hyps

    @property
    def fused(self) -> bool:
        """Whether the statistics run on the fused CUDA kernels."""
        return self.memory.fused

    @property
    def n_freq(self) -> int:
        """``k_stop``."""
        return self.stage.k_stop

    @cached_property
    def spectrum(self) -> SpectrumSpec:
        """Shape of the pass's spectra (the statistics' construction parameters)."""
        return SpectrumSpec(
            dtype=self.engine.spectrum_dtype,
            n_freq=self.n_freq,
            n_psi=self.stage.n_psi,
            gemm_pixels=self.memory.gemm_pixels,
            n_sub=self.memory.n_sub,
            hyp_batch=self.memory.hyp_batch,
            n_virtual=self.hyps.n_virtual,
            device=self.memory.device,
        )


Prepared = Mapping[tuple[int, int], torch.Tensor]  # (k-run index, batch index) -> B


def _make_spectra(plan: LoopPlan, device: torch.device) -> dict[int, Spectrum]:
    """One static spectrum buffer per distinct hypothesis-batch size (full + tail)."""
    spectra: dict[int, Spectrum] = {}
    for batch in plan.hyps.batches:
        if batch.size not in spectra:
            spectrum = Spectrum.empty(
                plan.n_freq,
                plan.memory.gemm_pixels,
                batch.size,
                plan.engine.spectrum_dtype,
                device,
            )
            spectrum.zero_uncovered(plan.layout.covered_k())
            spectra[batch.size] = spectrum
    return spectra


def _loop_body(
    plan: LoopPlan,
    prepared: Prepared,
    operands: Mapping[int, torch.Tensor],
    stats: BlockStats,
    spectra: Mapping[int, Spectrum],
    n_valid: int | None,
) -> None:
    """One block's loop: sub-batches x defocus planes x hypothesis batches."""
    engine = plan.engine
    memory = plan.memory
    stats.clear()
    for sub in range(memory.n_sub):
        p0 = sub * memory.gemm_pixels
        if n_valid is not None and p0 >= n_valid:
            break
        p1 = p0 + memory.gemm_pixels
        for batch_index, batch in enumerate(plan.hyps.batches):
            spectrum = spectra[batch.size]
            for run_index, run in enumerate(plan.layout.k_runs):
                a_parts = []
                for part in run.parts:
                    gid, local_plane = plan.plane_lookup[part.rect][batch.d_local]
                    k_start = plan.layout.rectangles[part.rect][0]
                    a_parts.append(
                        engine.operand(
                            operands[gid],
                            local_plane,
                            run.k_start - k_start,
                            run.k_stop - k_start,
                            p0,
                            p1,
                        )
                    )
                engine.contract(
                    a_parts, [prepared[(run_index, batch_index)]], run, spectrum
                )
            stats.consume(spectrum, batch, sub)


class HypLoop:
    """Reduces one block's features against every hypothesis into ``stats``.

    Built by :meth:`build`: on CUDA with the fused statistics the whole block loop
    is captured once as a CUDA graph over static copies of the block operands and
    replayed per block; otherwise the loop runs eagerly (any device, any backend).

    Attributes
    ----------
    plan : LoopPlan
    stats : BlockStats
        The block accumulator the loop fills.
    graph : torch.cuda.CUDAGraph or None
        The captured graph, or ``None`` for the eager loop.
    """

    def __init__(
        self,
        plan: LoopPlan,
        prepared: Prepared,
        stats: BlockStats,
        spectra: Mapping[int, Spectrum],
        *,
        graph: torch.cuda.CUDAGraph | None = None,
        statics: Mapping[int, torch.Tensor] | None = None,
    ) -> None:
        # Loop outlives the session that built it (cached by signature on the executor)
        # Store plan without blocks since groups hold the session's store blocks
        self.plan = replace(plan, groups=())
        self.prepared = prepared
        self.stats = stats
        self.spectra = spectra  # captured buffers: keep them alive with the graph
        self.graph = graph
        self.statics: Mapping[int, torch.Tensor] = statics or {}

    @classmethod
    @torch.no_grad()
    def build(
        cls,
        plan: LoopPlan,
        prepared: Prepared,
        stats: BlockStats,
        example: Mapping[int, torch.Tensor],
        *,
        pool: Any = None,
    ) -> HypLoop:
        """A loop for ``plan``; graphed when the plan asks for it and capture works."""
        device = stats.device
        spectra = _make_spectra(plan, device)
        if plan.memory.loop != "graph" or not stats.fused or device.type != "cuda":
            return cls(plan, prepared, stats, spectra)
        statics = {gid: t.detach().clone().contiguous() for gid, t in example.items()}

        def body() -> None:
            _loop_body(plan, prepared, statics, stats, spectra, None)

        side = torch.cuda.Stream(device=device)  # type: ignore[no-untyped-call]
        side.wait_stream(torch.cuda.current_stream(device))
        try:
            with torch.cuda.stream(side):
                for _ in range(2):
                    body()
            torch.cuda.current_stream(device).wait_stream(side)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            # NOTE: Not ``torch.cuda.graph(...)``: that context also empties the caching
            # allocator, releasing a dropped store's memory only for the next band or
            # micrograph to allocate it again.
            with torch.cuda.stream(side):
                graph.capture_begin(pool=pool)
                try:
                    body()
                finally:
                    graph.capture_end()
            torch.cuda.synchronize(device)
        except Exception:
            torch.cuda.synchronize(device)
            return cls(plan, prepared, stats, spectra)
        return cls(plan, prepared, stats, spectra, graph=graph, statics=statics)

    @torch.no_grad()
    def run(self, operands: Mapping[int, torch.Tensor], n_valid: int) -> None:
        """Reduce a block given its group tensors (``group.id -> block``)."""
        if self.graph is None:
            _loop_body(
                self.plan, self.prepared, operands, self.stats, self.spectra, n_valid
            )
            return
        for gid, static in self.statics.items():
            static.copy_(operands[gid])
        self.graph.replay()
        self.stats.note_replayed(self.plan.hyps.n_evaluated)
