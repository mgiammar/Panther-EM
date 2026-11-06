"""The block loop: per GEMM sub-batch and hypothesis batch, CGEMM then reduction.

:class:`HypLoop` reduces one pixel block of features against every hypothesis: one
contraction into a static spectrum buffer and one statistics update per (sub-batch,
hypothesis batch, k-run); on CUDA with the fused kernels the whole block loop is
captured once as a CUDA graph and replayed per block.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import torch

from panther_em.inference.search.correlation.contraction import Spectrum

if TYPE_CHECKING:
    from panther_em.inference.search.correlation.reduction import BlockStats
    from panther_em.inference.search.planning import StagePlan


Prepared = Mapping[tuple[int, int], torch.Tensor]  # (k-run index, batch index) -> B


def _make_spectra(plan: StagePlan, device: torch.device) -> dict[int, Spectrum]:
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
    plan: StagePlan,
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
            stats.consume(spectrum.pnf(), batch, sub)


class HypLoop:
    """Reduces one block's features against every hypothesis into ``stats``.

    Built by :meth:`build`: on CUDA with the fused statistics the whole block loop
    is captured once as a CUDA graph over static copies of the block operands and
    replayed per block; otherwise the loop runs eagerly (any device, any backend).

    Attributes
    ----------
    plan : StagePlan
    stats : BlockStats
        The block accumulator the loop fills.
    graph : torch.cuda.CUDAGraph or None
        The captured graph, or ``None`` for the eager loop.
    """

    def __init__(
        self,
        plan: StagePlan,
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
        plan: StagePlan,
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
