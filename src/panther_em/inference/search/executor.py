"""Stage planning and execution: plan a stage's shapes, then run its block loop.

* :func:`plan_memory` fixes the GEMM batch, block size, residency and loop kind for a
  stage before the store's groups are allocated (:class:`MemoryPlan`).
* :func:`plan_stage` freezes everything the loop needs against an existing store
  (:class:`StagePlan`).
* :class:`HypLoop` is the inner loop for one pixel block: for each GEMM sub-batch,
  defocus plane and hypothesis batch, one contraction into a static spectrum buffer
  and one statistics update; on CUDA with the fused kernels it is captured once as a
  CUDA graph and replayed.
* :class:`Executor` owns what outlives a stage and an image set: the device
  capabilities, the streams, the prepared-weight cache and the cache of captured
  loops; :meth:`Executor.run` reduces every block of a store into :class:`StageMaps`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal

import torch

from panther_em.inference.search import fused_kernel_loader
from panther_em.inference.search.basis import _ByteLRU
from panther_em.inference.search.contraction import Spectrum, make_engine
from panther_em.inference.search.hypotheses import HypothesisSet
from panther_em.inference.search.statistics import (
    SpectrumSpec,
    _cuda_graph_capture_supported,
    fused_stats_supported,
    make_block_stats,
)
from panther_em.inference.search.store import Residency

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Sequence

    from panther_em.inference.search.basis import WeightSource
    from panther_em.inference.search.contraction import ContractionEngine
    from panther_em.inference.search.hypotheses import HypBatch, HypothesisSpace
    from panther_em.inference.search.schedule import StageSpec
    from panther_em.inference.search.statistics import BlockStats
    from panther_em.inference.search.store import ColumnGroup, FeatureStore
    from panther_em.inference.search.tiling import FeatureLayout, KRun
    from panther_em.inference.search.types import BatchShape, Precision

LoopKind = Literal["eager", "graph"]
MIB = 1024**2


# --------------------------------------------------------------------------- #
# Sizing helpers
# --------------------------------------------------------------------------- #


def hyp_batch_multiple(n_psi: int) -> int:
    """The fused kernel's hypotheses-per-block granularity (``PAIRS``)."""
    if n_psi in fused_kernel_loader.LEAN_NUM_PSI:
        threads_per_pair = max(1, (n_psi // 64) // 2)
        return 128 // threads_per_pair
    return 32


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


# --------------------------------------------------------------------------- #
# Plans
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class MemoryPlan:
    """Shapes and residency of one stage, fixed before its store groups exist.

    Attributes
    ----------
    gemm_pixels : int
        Pixels per contraction ``P_g``.
    hyp_batch : int
        Hypotheses per contraction ``N_b``.
    block_pixels : int
        Pixels per store block (a multiple of ``gemm_pixels``).
    residency : {"device", "pinned_host"}
        Where the stage's feature store lives.
    loop : {"eager", "graph"}
        Whether the block loop is replayed as a CUDA graph.
    fused : bool
        Whether the statistics run on the fused CUDA kernels.
    engine : ContractionEngine
        The stage's GEMM engine (its precision and block layout).
    hyps : HypothesisSet
        The stage's hypotheses, batched by ``hyp_batch``.
    device : torch.device
        The compute device.
    """

    gemm_pixels: int
    hyp_batch: int
    block_pixels: int
    residency: Residency
    loop: LoopKind
    fused: bool
    engine: ContractionEngine
    hyps: HypothesisSet
    device: torch.device

    @property
    def n_sub(self) -> int:
        """GEMM sub-batches per block."""
        return self.block_pixels // self.gemm_pixels


def plan_memory(
    executor: Executor,
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
    residency: Residency | Literal["auto"] = "auto",
    store: FeatureStore | None = None,
    l2_target: int | None = None,
    block_bytes_max: int = 256 * MIB,
    device_store_fraction: float = 0.5,
    gemm_pixel_choices: Sequence[int] = (1024, 512),
) -> MemoryPlan:
    """Resolve a stage's GEMM batch, block size, residency, loop kind and hypotheses.

    Parameters
    ----------
    executor : Executor
        Supplies the device capabilities.
    spec : StageSpec
    n_pixels : int
        Pixels the stage searches.
    space : HypothesisSpace
    template_ids, defocus_ids : torch.Tensor, optional
        The stage's hypothesis restriction (``None`` = all), already normalised by
        ``StageSpec.hypothesis_ids``.
    batch, block_pixels : optional
        Explicit overrides of the L2-residency rule and the block budget.
    use_cuda_graph, use_fused_kernel : bool, optional
        Allow the graphed loop / the fused statistics kernels.
    residency : optional
        ``"auto"`` keeps the store on the device when it fits ``device_store_fraction``
        of the free memory, else pins it on the host and streams blocks.
    store : FeatureStore, optional
        Lets the per-pixel byte estimate count the groups that will actually be
        resident, including planes held beyond the stage's own.
    l2_target : int, optional
        Bytes of spectrum per contraction to keep L2-resident; defaults to two thirds
        of the device's L2 (64 MiB when unknown).
    block_bytes_max : int, optional
        Byte budget of one store block.
    device_store_fraction : float, optional
        See ``residency``.
    gemm_pixel_choices : sequence of int, optional
        Candidate ``P_g`` values, preferred first.
    """
    engine = make_engine(spec.precision)
    provisional = HypothesisSet.create(
        space, template_ids, defocus_ids, hyp_batch=1, n_psi=spec.n_psi
    )
    planes = provisional.defocus_planes()
    per_pixel = feature_bytes_per_pixel(resident_groups(spec, planes, store))

    probe = SpectrumSpec(
        dtype=engine.spectrum_dtype,
        n_freq=spec.k_stop,
        n_psi=spec.n_psi,
        gemm_pixels=1,
        n_sub=1,
        hyp_batch=1,
        n_virtual=provisional.n_virtual,
        device=executor.device,
    )
    fused = bool(
        use_fused_kernel and executor.fused_kernel and fused_stats_supported(probe)
    )

    elem = torch.empty((), dtype=engine.spectrum_dtype).element_size()
    target = l2_target or (
        (2 * executor.l2_bytes) // 3 if executor.l2_bytes else 64 * MIB
    )
    multiple = hyp_batch_multiple(spec.n_psi)
    if batch is not None:
        gemm_pixels, hyp_batch = batch.pixel, batch.hyp
    else:
        gemm_pixels = int(gemm_pixel_choices[0]) if executor.is_cuda else 256
        budget = max(multiple, target // (spec.k_stop * gemm_pixels * elem))
        hyp_batch = max(multiple, (budget // multiple) * multiple)
        hyp_batch = min(hyp_batch, max(multiple, provisional.n_virtual))
    gemm_pixels = max(1, min(gemm_pixels, max(1, n_pixels)))

    if block_pixels is None:
        fit = max(
            gemm_pixels,
            (int(block_bytes_max) // max(1, per_pixel)) // gemm_pixels * gemm_pixels,
        )
        needed = -(-max(1, n_pixels) // gemm_pixels) * gemm_pixels
        block_pixels = min(fit, needed)
    block_pixels = max(gemm_pixels, (block_pixels // gemm_pixels) * gemm_pixels)

    loop: LoopKind = (
        "graph"
        if use_cuda_graph and executor.is_cuda and executor.cuda_graphs and fused
        else "eager"
    )
    store_residency: Residency
    if residency == "auto":
        store_bytes = per_pixel * max(1, n_pixels)
        limit = int((executor.free_bytes or 0) * device_store_fraction)
        store_residency = (
            "pinned_host" if executor.is_cuda and store_bytes > limit else "device"
        )
    else:
        store_residency = residency
    hyps = HypothesisSet.create(
        space, template_ids, defocus_ids, hyp_batch=hyp_batch, n_psi=spec.n_psi
    )
    return MemoryPlan(
        gemm_pixels=gemm_pixels,
        hyp_batch=hyp_batch,
        block_pixels=block_pixels,
        residency=store_residency,
        loop=loop,
        fused=fused,
        engine=engine,
        hyps=hyps,
        device=executor.device,
    )


@dataclass(frozen=True, eq=False)
class StagePlan:
    """Everything the block loop needs for one stage, resolved against its store.

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
    memory : MemoryPlan
    signature : tuple
        Cache identity of the hypothesis loop (layout, precision, shapes, hypotheses).
    """

    index: int
    spec: StageSpec
    layout: FeatureLayout
    groups: tuple[ColumnGroup, ...]
    plane_lookup: tuple[tuple[tuple[int, int], ...], ...]
    memory: MemoryPlan
    signature: tuple

    @property
    def engine(self) -> ContractionEngine:
        """The stage's GEMM engine."""
        return self.memory.engine

    @property
    def hyps(self) -> HypothesisSet:
        """The stage's batched hypotheses."""
        return self.memory.hyps

    @property
    def fused(self) -> bool:
        """Whether the statistics run on the fused CUDA kernels."""
        return self.memory.fused

    @property
    def n_freq(self) -> int:
        """``k_stop``."""
        return self.spec.k_stop

    @cached_property
    def spectrum(self) -> SpectrumSpec:
        """Shape of the stage's spectra (the statistics' construction parameters)."""
        return SpectrumSpec(
            dtype=self.engine.spectrum_dtype,
            n_freq=self.n_freq,
            n_psi=self.spec.n_psi,
            gemm_pixels=self.memory.gemm_pixels,
            n_sub=self.memory.n_sub,
            hyp_batch=self.memory.hyp_batch,
            n_virtual=self.hyps.n_virtual,
            device=self.memory.device,
        )


def plan_stage(
    executor: Executor,
    index: int,
    spec: StageSpec,
    store: FeatureStore,
    memory: MemoryPlan,
) -> StagePlan:
    """Freeze a stage plan against a store whose groups already exist."""
    layout = spec.layout()
    planes = memory.hyps.defocus_planes()
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
    signature = (
        layout.rectangles,
        spec.precision,
        tuple((g.id, g.rect, g.precision, g.planes) for g in groups),
        tuple(lookup),
        memory.gemm_pixels,
        memory.block_pixels,
        memory.hyp_batch,
        spec.n_psi,
        memory.hyps.signature,
        memory.fused,
        memory.loop,
        str(executor.device),
    )
    return StagePlan(
        index=index,
        spec=spec,
        layout=layout,
        groups=tuple(groups),
        plane_lookup=tuple(lookup),
        memory=memory,
        signature=signature,
    )


@dataclass(eq=False)
class StageMaps:
    """Per-pixel results of one stage in the store's compact column order.

    Notes
    -----
    ``best_virtual`` holds *virtual* hypothesis ids (see
    :mod:`~panther_em.inference.search.hypotheses`); decode with ``hyps.decode``.
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


# --------------------------------------------------------------------------- #
# The hypothesis loop
# --------------------------------------------------------------------------- #

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
                engine.contract(a_parts, [prepared[(run_index, batch_index)]], run, spectrum)
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
        self.plan = plan
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
            with torch.cuda.graph(graph, pool=pool, stream=side):
                body()
        except Exception:
            torch.cuda.synchronize(device)
            return cls(plan, prepared, stats, spectra)
        return cls(plan, prepared, stats, spectra, graph=graph, statics=statics)

    @torch.no_grad()
    def run(self, operands: Mapping[int, torch.Tensor], n_valid: int) -> None:
        """Reduce a block given its group tensors (``group.id -> block``)."""
        if self.graph is None:
            _loop_body(self.plan, self.prepared, operands, self.stats, self.spectra, n_valid)
            return
        for gid, static in self.statics.items():
            static.copy_(operands[gid])
        self.graph.replay()
        self.stats.note_replayed(self.plan.hyps.n_evaluated)


# --------------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------------- #


class Executor:
    """Long-lived execution resources shared by every stage and image set.

    Parameters
    ----------
    device : torch.device or str
    weights : WeightSource
        Backs the prepared-weight cache.
    prepared_budget : int, optional
        Byte budget of the prepared-weight cache (``None`` = unbounded).

    Attributes
    ----------
    is_cuda, cuda_graphs, fused_kernel : bool
        Probed once: CUDA device; CUDA graph capture works; the fused iRFFT+stats
        extension compiled (probing it triggers the JIT build).
    l2_bytes, free_bytes : int or None
        L2 cache size and free device memory at construction, when known.
    compute, featurize, copy : torch.cuda.Stream or None
        The three streams stages run on (compute), featurize on, and stream blocks
        through (copy).
    loops : dict
        Captured or eager hypothesis loops keyed by ``StagePlan.signature``.
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
        self.is_cuda = self.device.type == "cuda"
        self.cuda_graphs = False
        self.fused_kernel = False
        self.l2_bytes: int | None = None
        self.free_bytes: int | None = None
        if self.is_cuda:
            index = (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
            self.device = torch.device("cuda", index)
            props = torch.cuda.get_device_properties(index)
            free, _total = torch.cuda.mem_get_info(index)
            self.cuda_graphs = _cuda_graph_capture_supported(index)
            self.fused_kernel = fused_kernel_loader._try_compile() is not None
            self.l2_bytes = int(getattr(props, "L2_cache_size", 0)) or None
            self.free_bytes = int(free)
        self._prepared = _ByteLRU(prepared_budget)
        self.loops: dict[tuple, HypLoop] = {}
        self.graph_pool: Any = torch.cuda.graph_pool_handle() if self.is_cuda else None
        self.compute: torch.cuda.Stream | None = None
        self.featurize: torch.cuda.Stream | None = None
        self.copy: torch.cuda.Stream | None = None
        if self.is_cuda:
            self.compute = torch.cuda.Stream(device=self.device)  # type: ignore[no-untyped-call]
            self.featurize = torch.cuda.Stream(device=self.device)  # type: ignore[no-untyped-call]
            self.copy = torch.cuda.Stream(device=self.device)  # type: ignore[no-untyped-call]

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
        return (parts, run.k_start, run.k_stop, engine.precision, rows_key, batch.j0, batch.j1)

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
        self, plan: StagePlan, prepared: Prepared, example: Mapping[int, torch.Tensor]
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
        plan: StagePlan,
        *,
        progress: Callable[[int], None] | None = None,
    ) -> StageMaps:
        """Reduce every block of ``store`` for ``plan``; ``progress(n_pixels_done)``."""
        hyps = plan.hyps
        rows = hyps.template_rows(self.device) if not hyps.is_contiguous else None
        rows_key = hyps.signature if rows is not None else None
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
                # Inputs (store blocks, prepared weights) came from other streams.
                self.compute.wait_stream(torch.cuda.current_stream(self.device))
            for block in store.blocks(plan.groups):
                if loop is None:
                    loop = self.loop_for(plan, prepared, block.operands)
                if block.ready is not None:
                    torch.cuda.current_stream(self.device).wait_event(block.ready)
                loop.run(block.operands, block.n_valid)
                start, n_valid = block.col_range[0], block.n_valid
                loop.stats.finalize_into(targets, start, n_valid)
                block.release(None)
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
