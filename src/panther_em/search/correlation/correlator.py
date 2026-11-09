"""The correlator: what outlives a pass, and the run over a store's blocks.

:class:`Correlator` owns the prepared-weight cache, the cache of built (captured)
loops, the closed-form moment forms, the prepared local weights, the compute stream
and the graph pool; :meth:`Correlator.run` correlates every block of a store against
a pass's hypotheses into :class:`StageMaps`, :meth:`Correlator.run_local` every
entry against its own neighborhood (:mod:`.local`). The correlator is bound to one
template's weights; a pass over another weight set (the seed templates of a
self-seeded stage, :mod:`~panther_em.search.seeding`) passes it explicitly and is
cached apart by the set's key.
"""

from __future__ import annotations

import contextlib
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch

from panther_em.search.correlation.contraction import Precision
from panther_em.search.correlation.local import LocalWeights, correlate_local
from panther_em.search.correlation.loop import HypLoop
from panther_em.search.correlation.maps import StageMaps
from panther_em.search.correlation.moments import MomentForms
from panther_em.search.correlation.reduction import make_block_stats

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterator, Mapping

    from panther_em.search.correlation.contraction import ContractionEngine
    from panther_em.search.correlation.layout import FeatureLayout, KRun
    from panther_em.search.correlation.local import LocalWork
    from panther_em.search.correlation.loop import LoopPlan, Prepared
    from panther_em.search.featurization.store import StoreLike
    from panther_em.search.hypotheses import HypBatch
    from panther_em.search.weights import TensorWeights


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
    weights : TensorWeights
        The template's weights; back the prepared-weight cache and the moments.
    prepared_budget : int, optional
        Byte budget of the prepared-weight cache (``None`` = unbounded).

    Attributes
    ----------
    compute : torch.cuda.Stream or None
        The stream the block loop runs on (CUDA only).
    loops : dict
        Captured or eager hypothesis loops keyed by ``LoopPlan.signature``.
    forms : dict
        Closed-form moment forms keyed by layout, engine, template set and ``n_psi``.
    local_weights_cache : dict
        Prepared local weights (:class:`~.local.LocalWeights`) keyed by layout,
        engine, template set and neighborhoods; ``None`` where they exceeded the
        budget they were asked for.
    """

    def __init__(
        self,
        device: torch.device | str,
        weights: TensorWeights,
        *,
        prepared_budget: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.weights = weights
        self._prepared = _ByteLRU(prepared_budget)
        self.loops: dict[tuple, HypLoop] = {}
        self.forms: dict[tuple, MomentForms] = {}
        self.local_weights_cache: dict[tuple, LocalWeights | None] = {}
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
        source: TensorWeights | None = None,
    ) -> torch.Tensor:
        """The engine's B operand of ``run`` for hypotheses ``batch.j0:batch.j1``.

        Entries are shared across defocus planes (weights do not depend on the
        image), across stages that reuse a run, and across image sets. ``source``
        (default: the correlator's weights) is the weight set; another set's entries
        are keyed apart by ``rows_key``, which the caller derives from its key.
        """
        key = self.prepared_key(layout, run, batch, engine, rows_key)
        cached = self._prepared.get(key)
        if cached is not None:
            return cached
        weights = self.weights if source is None else source
        rows = (
            torch.arange(batch.j0, batch.j1, device=weights.device)
            if template_rows is None
            else template_rows.to(weights.device)[batch.j0 : batch.j1]
        )
        parts = []
        for part in run.parts:
            sub = (
                run.k_start,
                part.m_start,
                run.k_stop - run.k_start,
                part.m_stop - part.m_start,
            )
            parts.append(engine.prepare_weights(weights.block(sub, rows)))
        operand = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        self._prepared.put(key, operand)
        return operand

    @property
    def prepared_bytes(self) -> int:
        """Bytes held by the prepared-weight cache."""
        return self._prepared.nbytes

    def _source_key(self, weights: TensorWeights | None) -> Hashable:
        """``None`` for the correlator's own weights, the set's key otherwise."""
        if weights is None or weights is self.weights:
            return None
        if weights.key is None:
            raise ValueError("an alternate weight set needs a key for the caches")
        return weights.key

    # -- loops and execution -----------------------------------------------------

    def loop_for(
        self,
        plan: LoopPlan,
        prepared: Prepared,
        example: Mapping[int, torch.Tensor],
        source_key: Hashable = None,
    ) -> HypLoop:
        """The hypothesis loop for ``plan``, built (captured) once per signature."""
        key = plan.signature if source_key is None else (plan.signature, source_key)
        loop = self.loops.get(key)
        if loop is None:
            stats = make_block_stats(plan.spectrum, fused=plan.fused)
            loop = HypLoop.build(plan, prepared, stats, example, pool=self.graph_pool)
            self.loops[key] = loop
        return loop

    @torch.no_grad()
    def run(
        self,
        store: StoreLike,
        plan: LoopPlan,
        *,
        weights: TensorWeights | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> StageMaps:
        """Reduce every block of ``store`` for ``plan``; ``progress(n_pixels_done)``."""
        hyps = plan.hyps
        source = self.weights if weights is None else weights
        if source.n_templates != hyps.space.n_templates:
            raise ValueError(
                f"the weights hold {source.n_templates} template rows; the plan's "
                f"space has {hyps.space.n_templates}"
            )
        rows = hyps.template_rows(self.device) if not hyps.is_contiguous else None
        source_key = self._source_key(weights)
        rows_key = (
            None
            if rows is None and source_key is None
            else (source_key, hyps.template_signature if rows is not None else None)
        )
        prepared = {
            (run_index, batch_index): self.prepared(
                plan.layout, run, batch, plan.engine, rows, rows_key, source
            )
            for run_index, run in enumerate(plan.layout.k_runs)
            for batch_index, batch in enumerate(hyps.batches)
        }
        maps = StageMaps.empty(store.n_pixels, self.device, hyps.n_evaluated)
        targets = maps.as_targets()
        loop = None
        with self._on_compute():
            for block in store.blocks(plan.groups):
                if loop is None:
                    loop = self.loop_for(plan, prepared, block.operands, source_key)
                loop.run(block.operands, block.n_valid)
                start, n_valid = block.col_range[0], block.n_valid
                loop.stats.finalize_into(targets, start, n_valid)
                if progress is not None:
                    progress(block.n_valid)
        return maps

    @contextlib.contextmanager
    def _on_compute(self) -> Iterator[None]:
        """Run the body on the compute stream, between the caller's work around it."""
        if self.compute is None:
            yield
            return
        caller = torch.cuda.current_stream(self.device)
        self.compute.wait_stream(caller)
        with torch.cuda.stream(self.compute):
            yield
        caller.wait_stream(self.compute)

    # -- local passes ------------------------------------------------------------

    def moment_forms(self, plan: LoopPlan) -> MomentForms:
        """The closed-form moments of ``plan``'s template set (built once, cached)."""
        hyps = plan.hyps
        key = (
            plan.layout.rectangles,
            plan.engine.precision,
            hyps.template_signature,
            plan.stage.n_psi,
        )
        forms = self.forms.get(key)
        if forms is None:
            forms = MomentForms.build(
                self.weights,
                plan.layout,
                plan.engine,
                plan.stage.n_psi,
                template_rows=None if hyps.is_contiguous else hyps.template_rows(),
            )
            self.forms[key] = forms
        return forms

    def local_weights(
        self,
        plan: LoopPlan,
        rows: torch.Tensor,
        rows_key: Hashable,
        *,
        budget: int | None,
    ) -> LocalWeights | None:
        """The groups' prepared fp16 weights for ``plan``, or ``None`` over budget.

        Parameters
        ----------
        plan : LoopPlan
            The pass (an fp16 engine; other engines get ``None``).
        rows : torch.Tensor
            ``(n_groups, W)`` positions in the pass's template rows
            (``LocalHypotheses.rows``).
        rows_key : Hashable
            Identity of ``rows`` for the cache (the neighborhoods' parameters).
        budget : int, optional
            Bytes the prepared weights may take (``None`` = unbounded).
        """
        if plan.engine.precision is not Precision.FP16:
            return None
        key = (
            plan.layout.rectangles,
            plan.engine.precision,
            plan.hyps.template_signature,
            rows_key,
            tuple(rows.shape),
            budget,
        )
        if key in self.local_weights_cache:
            return self.local_weights_cache[key]
        estimate = LocalWeights.estimate_bytes(
            plan.layout, int(rows.shape[0]), int(rows.shape[1])
        )
        prepared = (
            None
            if budget is not None and estimate > budget
            else LocalWeights.build(self.weights, plan, rows)
        )
        self.local_weights_cache[key] = prepared
        return prepared

    @torch.no_grad()
    def run_local(
        self,
        store: StoreLike,
        plan: LoopPlan,
        work: LocalWork,
        *,
        chunk_bytes: int,
        progress: Callable[[int], None] | None = None,
        local_weights: LocalWeights | None = None,
    ) -> StageMaps:
        """Score every entry of ``work`` over its group's hypotheses only."""
        forms = self.moment_forms(plan)
        maps = StageMaps.empty(store.n_pixels, self.device, plan.hyps.n_evaluated)
        with self._on_compute():
            correlate_local(
                store,
                plan,
                work,
                self.weights,
                forms,
                chunk_bytes=chunk_bytes,
                progress=progress,
                out=maps,
                prepared=local_weights,
            )
        return maps

    def release(self) -> None:
        """Drop cached loops, prepared weights, moment forms and local weights."""
        self.loops.clear()
        self._prepared.clear()
        self.forms.clear()
        self.local_weights_cache.clear()
