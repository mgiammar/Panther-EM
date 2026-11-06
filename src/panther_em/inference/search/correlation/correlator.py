"""The correlator: what outlives a pass, and the run over a store's blocks.

:class:`Correlator` owns the prepared-weight cache, the cache of built (captured)
loops, the compute stream and the graph pool; :meth:`Correlator.run` correlates
every block of a store against a pass's hypotheses into :class:`StageMaps`.
"""

from __future__ import annotations

import contextlib
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch

from panther_em.inference.search.correlation.loop import HypLoop
from panther_em.inference.search.correlation.maps import StageMaps
from panther_em.inference.search.correlation.reduction import make_block_stats

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping

    from panther_em.inference.search.correlation.contraction import ContractionEngine
    from panther_em.inference.search.correlation.layout import FeatureLayout, KRun
    from panther_em.inference.search.correlation.loop import LoopPlan, Prepared
    from panther_em.inference.search.featurization.store import FeatureStore
    from panther_em.inference.search.hypotheses import HypBatch
    from panther_em.inference.template import WeightSource


class _ByteLRU:
    """Insertion-ordered tensor cache with a byte budget (``None`` = unbounded)."""

    def __init__(self, budget: int | None) -> None:
        self.budget = budget
        self._items: OrderedDict[Hashable, torch.Tensor] = OrderedDict()
        self.nbytes = 0

    def get(self, key: Hashable) -> torch.Tensor | None:
        t = self._items.get(key)
        if t is not None:
            self._items.move_to_end(key)
        return t

    def put(self, key: Hashable, value: torch.Tensor) -> None:
        """Insert ``value``, evicting the least recently used entries over budget."""
        old = self._items.pop(key, None)
        if old is not None:
            self.nbytes -= old.numel() * old.element_size()
        self._items[key] = value
        self.nbytes += value.numel() * value.element_size()
        if self.budget is None:
            return
        while self.nbytes > self.budget and len(self._items) > 1:
            _, evicted = self._items.popitem(last=False)  # never the newest entry
            self.nbytes -= evicted.numel() * evicted.element_size()

    def clear(self) -> None:
        self._items.clear()
        self.nbytes = 0


class Correlator:
    """Long-lived correlation resources shared by every pass and image set.

    Parameters
    ----------
    device : torch.device or str
        The compute device.
    weights : WeightSource
        Backs the prepared-weight cache.
    prepared_budget : int, optional
        Byte budget of the prepared-weight cache (``None`` = unbounded).

    Attributes
    ----------
    compute : torch.cuda.Stream or None
        The stream the block loop runs on (CUDA only).
    loops : dict
        Captured or eager hypothesis loops keyed by ``LoopPlan.signature``.
    """

    def __init__(
        self,
        device: torch.device | str,
        weights: WeightSource,
        *,
        prepared_budget: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.weights = weights
        self._prepared = _ByteLRU(prepared_budget)
        self.loops: dict[tuple, HypLoop] = {}
        is_cuda = self.device.type == "cuda"
        self.graph_pool: Any = torch.cuda.graph_pool_handle() if is_cuda else None
        self.compute: torch.cuda.Stream | None = None
        if is_cuda:
            self.compute = torch.cuda.Stream(device=self.device)  # type: ignore[no-untyped-call]

    # -- prepared weights --------------------------------------------------------

    @staticmethod
    def prepared_key(
        layout: FeatureLayout,
        run: KRun,
        batch: HypBatch,
        engine: ContractionEngine,
        rows_key: Hashable,
    ) -> tuple:
        """Cache key of one prepared B operand."""
        parts = tuple(
            (layout.rectangles[p.rect], p.m_start, p.m_stop) for p in run.parts
        )
        return (
            parts,
            run.k_start,
            run.k_stop,
            engine.precision,
            rows_key,
            batch.j0,
            batch.j1,
        )

    def prepared(
        self,
        layout: FeatureLayout,
        run: KRun,
        batch: HypBatch,
        engine: ContractionEngine,
        template_rows: torch.Tensor | None = None,
        rows_key: Hashable = None,
    ) -> torch.Tensor:
        """The engine-specific B operand of ``run`` for hypotheses ``batch.j0:batch.j1``.

        Entries are shared across defocus planes (weights do not depend on the
        image), across stages that reuse a run, and across image sets.
        """
        key = self.prepared_key(layout, run, batch, engine, rows_key)
        cached = self._prepared.get(key)
        if cached is not None:
            return cached
        source = self.weights
        rows = (
            torch.arange(batch.j0, batch.j1, device=source.device)
            if template_rows is None
            else template_rows.to(source.device)[batch.j0 : batch.j1]
        )
        parts = []
        for part in run.parts:
            sub = (
                run.k_start,
                part.m_start,
                run.k_stop - run.k_start,
                part.m_stop - part.m_start,
            )
            parts.append(engine.prepare_weights(source.block(sub, rows)))
        operand = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        self._prepared.put(key, operand)
        return operand

    @property
    def prepared_bytes(self) -> int:
        """Bytes held by the prepared-weight cache."""
        return self._prepared.nbytes

    # -- loops and execution -----------------------------------------------------

    def loop_for(
        self, plan: LoopPlan, prepared: Prepared, example: Mapping[int, torch.Tensor]
    ) -> HypLoop:
        """The cached hypothesis loop for ``plan``; built (captured) once per signature."""
        loop = self.loops.get(plan.signature)
        if loop is None:
            stats = make_block_stats(plan.spectrum, fused=plan.fused)
            loop = HypLoop.build(plan, prepared, stats, example, pool=self.graph_pool)
            self.loops[plan.signature] = loop
        return loop

    @torch.no_grad()
    def run(
        self,
        store: FeatureStore,
        plan: LoopPlan,
        *,
        progress: Callable[[int], None] | None = None,
    ) -> StageMaps:
        """Reduce every block of ``store`` for ``plan``; ``progress(n_pixels_done)``."""
        hyps = plan.hyps
        rows = hyps.template_rows(self.device) if not hyps.is_contiguous else None
        rows_key = hyps.template_signature if rows is not None else None
        prepared = {
            (run_index, batch_index): self.prepared(
                plan.layout, run, batch, plan.engine, rows, rows_key
            )
            for run_index, run in enumerate(plan.layout.k_runs)
            for batch_index, batch in enumerate(hyps.batches)
        }
        maps = StageMaps.empty(store.n_pixels, self.device, hyps.n_evaluated)
        targets = maps.as_targets()
        loop = None
        stream: contextlib.AbstractContextManager[object] = (
            torch.cuda.stream(self.compute)
            if self.compute is not None
            else contextlib.nullcontext()
        )
        with stream:
            if self.compute is not None:
                # Inputs (store blocks, prepared weights) came from another stream.
                self.compute.wait_stream(torch.cuda.current_stream(self.device))
            for block in store.blocks(plan.groups):
                if loop is None:
                    loop = self.loop_for(plan, prepared, block.operands)
                loop.run(block.operands, block.n_valid)
                start, n_valid = block.col_range[0], block.n_valid
                loop.stats.finalize_into(targets, start, n_valid)
                if progress is not None:
                    progress(block.n_valid)
        if self.compute is not None:
            # Hand the maps back to the caller's stream.
            torch.cuda.current_stream(self.device).wait_stream(self.compute)
        return maps

    def release(self) -> None:
        """Drop cached loops and prepared weights."""
        self.loops.clear()
        self._prepared.clear()
