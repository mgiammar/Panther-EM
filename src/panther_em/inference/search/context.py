"""Execution context: device capabilities, memory planning, and the hypothesis loop.

* :class:`DeviceCaps` -- what the device can do (CUDA graphs, fused kernel, L2 size).
* :class:`MemoryPlan` / :class:`Planner` -- picks the GEMM batch shape, block size and
  loop kind for a stage from the L2-residency rule and the caller's overrides.
* :class:`SearchContext` -- long-lived per
  :class:`~panther_em.inference.search.searcher.Searcher`: streams, the prepared-weight
  cache, and the cache of captured hypothesis loops.
* :class:`EagerHypLoop` / :class:`GraphedHypLoop` -- the inner loop for one pixel block:
  for each GEMM sub-batch, each defocus plane and each hypothesis batch, one contraction
  into a static spectrum buffer and one statistics update. The graphed variant replays
  the whole thing as a CUDA graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import torch

from panther_em.inference.search import fused_kernel_loader
from panther_em.inference.search.contraction import Spectrum
from panther_em.inference.search.fused_statistics import FusedStatsBackend
from panther_em.inference.search.statistics import (
    BlockStats,
    SpectrumSpec,
    StatsBackend,
    TorchStatsBackend,
    _cuda_graph_capture_supported,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from panther_em.inference.search.basis import PreparedWeightCache, WeightSource
    from panther_em.inference.search.contraction import ContractionEngine
    from panther_em.inference.search.hypotheses import HypothesisSet
    from panther_em.inference.search.tiling import FeatureLayout
    from panther_em.inference.search.types import BatchShape, Precision

LoopKind = Literal["eager", "graph"]
Residency = Literal["device", "pinned_host", "transient"]
MIB = 1024**2


# --------------------------------------------------------------------------- #
# Device capabilities and memory planning
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DeviceCaps:
    """Probed once per device.

    Attributes
    ----------
    device : torch.device
    is_cuda : bool
    cuda_graphs : bool
        Whether CUDA graph capture works on this device.
    fused_kernel : bool
        Whether the fused iRFFT+stats extension compiled.
    l2_bytes : int or None
        L2 cache size, when known.
    free_bytes : int or None
        Free device memory at probe time.
    """

    device: torch.device
    is_cuda: bool
    cuda_graphs: bool
    fused_kernel: bool
    l2_bytes: int | None
    free_bytes: int | None

    @classmethod
    def probe(cls, device: torch.device | str) -> DeviceCaps:
        """Probe ``device`` (the kernel probe may trigger the JIT build)."""
        device = torch.device(device)
        if device.type != "cuda":
            return cls(device, False, False, False, None, None)
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        props = torch.cuda.get_device_properties(index)
        free, _total = torch.cuda.mem_get_info(index)
        return cls(
            device=torch.device("cuda", index),
            is_cuda=True,
            cuda_graphs=_cuda_graph_capture_supported(index),
            fused_kernel=fused_kernel_loader._try_compile() is not None,
            l2_bytes=int(getattr(props, "L2_cache_size", 0)) or None,
            free_bytes=int(free),
        )


@dataclass(frozen=True)
class MemoryPlan:
    """Batch/block shape and residency decisions for one stage.

    Attributes
    ----------
    gemm_pixels : int
        Pixels per contraction ``P_g``.
    hyp_batch : int
        Hypotheses per contraction ``N_b``.
    block_pixels : int
        Pixels per store block (a multiple of ``gemm_pixels``).
    store_residency : {"device", "pinned_host", "transient"}
    store_precision : Precision
        Block layout/dtype of newly featurized groups.
    loop : {"eager", "graph"}
    n_slots : int
        Prefetch slots (meaningful for non-device residency).
    """

    gemm_pixels: int
    hyp_batch: int
    block_pixels: int
    store_residency: Residency
    store_precision: Precision
    loop: LoopKind
    n_slots: int = 3

    @property
    def n_sub(self) -> int:
        """GEMM sub-batches per block."""
        return self.block_pixels // self.gemm_pixels

    def spectrum_bytes(self, n_freq: int, spectrum_dtype: torch.dtype) -> int:
        """Bytes of one ``(F, P_g, N_b)`` spectrum."""
        return (
            n_freq
            * self.gemm_pixels
            * self.hyp_batch
            * torch.empty((), dtype=spectrum_dtype).element_size()
        )


class Planner:
    """Chooses :class:`MemoryPlan` values the caller did not fix explicitly.

    Parameters
    ----------
    l2_target : int, optional
        Target bytes for the per-batch spectrum (keep it L2-resident). Defaults to
        two thirds of the device L2, or 64 MiB when unknown.
    block_bytes_max : int, optional
        Upper bound on one block's bytes (all groups of a stage). Defaults to 256 MiB.
    gemm_pixel_choices : sequence of int, optional
        Candidate ``P_g`` values, preferred first. Defaults to ``(1024, 512)``.
    """

    def __init__(
        self,
        *,
        l2_target: int | None = None,
        block_bytes_max: int = 256 * MIB,
        device_store_fraction: float = 0.5,
        gemm_pixel_choices: Sequence[int] = (1024, 512),
    ) -> None:
        self.l2_target = l2_target
        self.block_bytes_max = int(block_bytes_max)
        # a store larger than this fraction of the free device memory is pinned-host
        # resident and streamed through device slots
        self.device_store_fraction = float(device_store_fraction)
        self.gemm_pixel_choices = tuple(int(v) for v in gemm_pixel_choices)

    @staticmethod
    def hyp_batch_multiple(n_psi: int) -> int:
        """The fused kernel's hypotheses-per-block granularity (``PAIRS``)."""
        if n_psi in fused_kernel_loader.LEAN_NUM_PSI:
            threads_per_pair = max(1, (n_psi // 64) // 2)
            return 128 // threads_per_pair
        return 32

    def plan(
        self,
        caps: DeviceCaps,
        *,
        n_pixels: int,
        n_freq: int,
        n_psi: int,
        n_virtual: int,
        n_defocus: int,
        feature_bytes_per_pixel: int,
        spectrum_dtype: torch.dtype,
        precision: Precision,
        stats_backend: StatsBackend,
        batch: BatchShape | None = None,
        block_pixels: int | None = None,
        use_cuda_graph: bool = True,
        residency: Residency | Literal["auto"] = "auto",
    ) -> MemoryPlan:
        """Resolve a stage's plan.

        Parameters
        ----------
        caps : DeviceCaps
            Probed device capabilities.
        n_pixels : int
            Pixels in the stage.
        n_freq, n_psi, n_virtual, n_defocus : int
            Stage sizes (``k_stop``, in-plane samples, enumerated hypotheses, planes).
        feature_bytes_per_pixel : int
            Bytes one pixel's features occupy across every group the stage reads.
        spectrum_dtype : torch.dtype
            The engine's spectrum dtype.
        precision : Precision
            Stage precision (also the layout of newly featurized groups).
        stats_backend : StatsBackend
            Chosen backend; graphs need the fused one.
        batch, block_pixels : optional
            Explicit overrides.
        use_cuda_graph : bool, optional
            Allow the graphed loop when the device and backend permit it.
        residency : optional
            ``"auto"`` keeps the store on the device when it fits half the free
            memory, else pins it on the host.
        """
        elem = torch.empty((), dtype=spectrum_dtype).element_size()
        l2_target = self.l2_target or (
            (2 * caps.l2_bytes) // 3 if caps.l2_bytes else 64 * MIB
        )
        multiple = self.hyp_batch_multiple(n_psi)
        if batch is not None:
            gemm_pixels, hyp_batch = batch.pixel, batch.hyp
        else:
            gemm_pixels = self.gemm_pixel_choices[0] if caps.is_cuda else 256
            budget = max(multiple, l2_target // (n_freq * gemm_pixels * elem))
            hyp_batch = max(multiple, (budget // multiple) * multiple)
            hyp_batch = min(hyp_batch, max(multiple, -(-n_virtual // 1)))
        gemm_pixels = max(1, min(gemm_pixels, max(1, n_pixels)))

        if block_pixels is None:
            per_pixel = max(1, feature_bytes_per_pixel)
            fit = max(
                gemm_pixels,
                (self.block_bytes_max // per_pixel) // gemm_pixels * gemm_pixels,
            )
            needed = -(-max(1, n_pixels) // gemm_pixels) * gemm_pixels
            block_pixels = min(fit, needed)
        block_pixels = max(gemm_pixels, (block_pixels // gemm_pixels) * gemm_pixels)

        loop: LoopKind = (
            "graph"
            if use_cuda_graph
            and caps.is_cuda
            and caps.cuda_graphs
            and stats_backend.name == "fused"
            else "eager"
        )
        store_residency: Residency
        if residency == "auto":
            store_bytes = feature_bytes_per_pixel * max(1, n_pixels)
            limit = int((caps.free_bytes or 0) * self.device_store_fraction)
            if caps.is_cuda and store_bytes > limit:
                store_residency = "pinned_host"
            else:
                store_residency = "device"
        else:
            store_residency = residency
        return MemoryPlan(
            gemm_pixels=gemm_pixels,
            hyp_batch=hyp_batch,
            block_pixels=block_pixels,
            store_residency=store_residency,
            store_precision=precision,
            loop=loop,
        )


def choose_stats_backend(
    spec: SpectrumSpec,
    caps: DeviceCaps,
    *,
    use_fused_kernel: bool = True,
    backends: Sequence[StatsBackend] = (FusedStatsBackend(), TorchStatsBackend()),
) -> StatsBackend:
    """First backend supporting ``spec`` (fused only when allowed and available)."""
    for backend in backends:
        if backend.name == "fused" and not (use_fused_kernel and caps.fused_kernel):
            continue
        if backend.supports(spec):
            return backend
    raise RuntimeError("no statistics backend supports this stage")


# --------------------------------------------------------------------------- #
# Hypothesis loops
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class LoopInputs:
    """Everything a hypothesis loop needs besides the block itself.

    Attributes
    ----------
    layout : FeatureLayout
        The stage's rectangles.
    plane_lookup : tuple[tuple[tuple[int, int], ...], ...]
        ``plane_lookup[rect_index][d_local] -> (group id, local plane index)``.
    engine : ContractionEngine
    hyp_plan : HypothesisSet
    prepared : Mapping[tuple[int, int], torch.Tensor]
        ``(k_run index, batch index) -> B operand``.
    memory : MemoryPlan
    n_freq : int
    """

    layout: FeatureLayout
    plane_lookup: tuple[tuple[tuple[int, int], ...], ...]
    engine: ContractionEngine
    hyp_plan: HypothesisSet
    prepared: Mapping[tuple[int, int], torch.Tensor]
    memory: MemoryPlan
    n_freq: int


@runtime_checkable
class HypLoop(Protocol):
    """Reduces one block's features against every hypothesis into ``stats``."""

    stats: BlockStats

    def run(self, operands: Mapping[int, torch.Tensor], n_valid: int) -> None:
        """Reduce a block given its group tensors (``group.id -> block``)."""


def _make_spectra(inputs: LoopInputs, device: torch.device) -> dict[int, Spectrum]:
    """One static spectrum buffer per distinct hypothesis-batch size (full + tail)."""
    memory = inputs.memory
    spectra: dict[int, Spectrum] = {}
    for batch in inputs.hyp_plan.batches:
        if batch.size not in spectra:
            spectrum = Spectrum.empty(
                inputs.n_freq,
                memory.gemm_pixels,
                batch.size,
                inputs.engine.spectrum_dtype,
                device,
            )
            spectrum.zero_uncovered(inputs.layout.covered_k())
            spectra[batch.size] = spectrum
    return spectra


def _loop_body(
    inputs: LoopInputs,
    operands: Mapping[int, torch.Tensor],
    stats: BlockStats,
    spectra: Mapping[int, Spectrum],
    n_valid: int | None,
) -> None:
    """One block's loop: sub-batches x defocus planes x hypothesis batches."""
    engine = inputs.engine
    memory = inputs.memory
    plan = inputs.hyp_plan
    stats.clear()
    for sub in range(memory.n_sub):
        p0 = sub * memory.gemm_pixels
        if n_valid is not None and p0 >= n_valid:
            break
        p1 = p0 + memory.gemm_pixels
        for batch_index, batch in enumerate(plan.batches):
            spectrum = spectra[batch.size]
            for run_index, run in enumerate(inputs.layout.k_runs):
                a_parts = []
                for part in run.parts:
                    gid, local_plane = inputs.plane_lookup[part.rect][batch.d_local]
                    k_start = inputs.layout.rectangles[part.rect][0]
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
                weights = inputs.prepared[(run_index, batch_index)]
                engine.contract(a_parts, [weights], run, spectrum)
            stats.consume(spectrum.pnf(), batch, sub)


class EagerHypLoop:
    """Plain Python loop; works on any device and any statistics backend."""

    def __init__(self, inputs: LoopInputs, stats: BlockStats) -> None:
        self.inputs = inputs
        self.stats = stats
        self.spectra = _make_spectra(inputs, stats.device)

    @torch.no_grad()
    def run(self, operands: Mapping[int, torch.Tensor], n_valid: int) -> None:
        """See :class:`HypLoop`."""
        _loop_body(self.inputs, operands, self.stats, self.spectra, n_valid)


class GraphedHypLoop:
    """The same loop captured once as a CUDA graph over static input buffers.

    ``run`` copies the block's group tensors into the static buffers and replays.
    Requires the fused in-kernel statistics path (the torch path replays a child
    graph, which cannot be captured).
    """

    def __init__(
        self,
        inputs: LoopInputs,
        stats: BlockStats,
        graph: torch.cuda.CUDAGraph,
        statics: dict[int, torch.Tensor],
        spectra: Mapping[int, Spectrum],
    ) -> None:
        self.inputs = inputs
        self.stats = stats
        self.graph = graph
        self.statics = statics
        self.spectra = spectra  # captured buffers: keep them alive with the graph

    @classmethod
    @torch.no_grad()
    def try_capture(
        cls,
        inputs: LoopInputs,
        stats: BlockStats,
        example: Mapping[int, torch.Tensor],
        *,
        pool: Any = None,
    ) -> GraphedHypLoop | None:
        """Capture, or return ``None`` when this device/backend cannot capture it."""
        if not stats.inkernel or stats.device.type != "cuda":
            return None
        device = stats.device
        statics = {gid: t.detach().clone().contiguous() for gid, t in example.items()}
        spectra = _make_spectra(inputs, device)

        def body() -> None:
            _loop_body(inputs, statics, stats, spectra, None)

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
            return None
        return cls(inputs, stats, graph, statics, spectra)

    @torch.no_grad()
    def run(self, operands: Mapping[int, torch.Tensor], n_valid: int) -> None:
        """Copy ``operands`` into the static buffers and replay the graph."""
        for gid, static in self.statics.items():
            static.copy_(operands[gid])
        self.graph.replay()
        self.stats.note_replayed(self.inputs.hyp_plan.n_evaluated)


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #


class SearchContext:
    """Long-lived execution resources shared by every stage and image of a searcher.

    Parameters
    ----------
    device : torch.device or str
    weights : WeightSource
        Backs the prepared-weight cache.
    prepared_budget : int, optional
        Byte budget of the prepared-weight cache (``None`` = unbounded).
    planner : Planner, optional
    """

    def __init__(
        self,
        device: torch.device | str,
        weights: WeightSource,
        *,
        prepared_budget: int | None = None,
        planner: Planner | None = None,
    ) -> None:
        from panther_em.inference.search.basis import PreparedWeightCache

        self.device = torch.device(device)
        self.caps = DeviceCaps.probe(self.device)
        self.planner = planner or Planner()
        self.prepared: PreparedWeightCache = PreparedWeightCache(
            weights, prepared_budget
        )
        self.loops: dict[tuple, HypLoop] = {}
        self.graph_pool: Any = (
            torch.cuda.graph_pool_handle() if self.caps.is_cuda else None
        )
        self.compute: torch.cuda.Stream | None = None
        self.featurize: torch.cuda.Stream | None = None
        self.copy: torch.cuda.Stream | None = None
        if self.caps.is_cuda:
            self.compute = torch.cuda.Stream(device=self.device)  # type: ignore[no-untyped-call]
            self.featurize = torch.cuda.Stream(device=self.device)  # type: ignore[no-untyped-call]
            self.copy = torch.cuda.Stream(device=self.device)  # type: ignore[no-untyped-call]

    def loop_for(
        self,
        key: tuple,
        inputs: LoopInputs,
        stats_backend: StatsBackend,
        spec: SpectrumSpec,
        example: Mapping[int, torch.Tensor],
    ) -> HypLoop:
        """The cached hypothesis loop for a stage layout; captured once per ``key``."""
        loop = self.loops.get(key)
        if loop is not None:
            return loop
        stats = stats_backend.make(spec)
        if inputs.memory.loop == "graph":
            graphed = GraphedHypLoop.try_capture(
                inputs, stats, example, pool=self.graph_pool
            )
            if graphed is not None:
                self.loops[key] = graphed
                return graphed
        eager = EagerHypLoop(inputs, stats)
        self.loops[key] = eager
        return eager

    def release(self) -> None:
        """Drop cached loops and prepared weights."""
        self.loops.clear()
        self.prepared.clear()
