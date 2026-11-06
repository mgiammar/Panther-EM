"""Stage planning: a stage's shapes before its store exists, then against the store.

* :func:`plan_memory` fixes the GEMM batch, block size and loop kind for a stage
  before its store is allocated (:class:`MemoryPlan`).
* :func:`plan_stage` freezes everything the correlation loop needs against an
  existing store (:class:`StagePlan`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Literal

import torch

from panther_em.inference.search.correlation.contraction import make_engine
from panther_em.inference.search.correlation.kernels import fused_kernel_loader
from panther_em.inference.search.correlation.reduction import (
    SpectrumSpec,
    fused_stats_supported,
)
from panther_em.inference.search.hypotheses import HypothesisSet

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from panther_em.inference.search.correlation.contraction import ContractionEngine
    from panther_em.inference.search.correlation.executor import Executor
    from panther_em.inference.search.correlation.layout import FeatureLayout
    from panther_em.inference.search.featurization.store import (
        ColumnGroup,
        FeatureStore,
    )
    from panther_em.inference.search.hypotheses import HypothesisSpace
    from panther_em.inference.search.stage import Stage
    from panther_em.inference.search.types import BatchShape, Precision

LoopKind = Literal["eager", "graph"]
MIB = 1024**2


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


# --------------------------------------------------------------------------- #
# Plans
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class MemoryPlan:
    """Shapes of one stage, fixed before its store exists.

    Attributes
    ----------
    gemm_pixels : int
        Pixels per contraction ``P_g``.
    hyp_batch : int
        Hypotheses per contraction ``N_b``.
    block_pixels : int
        Pixels per store block (a multiple of ``gemm_pixels``).
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
    spec: Stage,
    *,
    n_pixels: int,
    space: HypothesisSpace,
    template_ids: torch.Tensor | None = None,
    defocus_ids: torch.Tensor | None = None,
    batch: BatchShape | None = None,
    block_pixels: int | None = None,
    use_cuda_graph: bool = True,
    use_fused_kernel: bool = True,
    l2_target: int | None = None,
    block_bytes_max: int = 256 * MIB,
    gemm_pixel_choices: Sequence[int] = (1024, 512),
) -> MemoryPlan:
    """Resolve a stage's GEMM batch, block size, loop kind and hypotheses.

    Parameters
    ----------
    executor : Executor
        Supplies the device capabilities.
    spec : Stage
    n_pixels : int
        Pixels the stage searches.
    space : HypothesisSpace
    template_ids, defocus_ids : torch.Tensor, optional
        The stage's hypothesis restriction (``None`` = all), already normalised by
        ``Stage.hypothesis_ids``.
    batch, block_pixels : optional
        Explicit overrides of the L2-residency rule and the block budget.
    use_cuda_graph, use_fused_kernel : bool, optional
        Allow the graphed loop / the fused statistics kernels.
    l2_target : int, optional
        Bytes of spectrum per contraction to keep L2-resident; defaults to a quarter
        of the device's L2 (24 MiB when unknown). Measured on an RTX 6000 Ada (96 MiB
        L2), the loop is fastest with the spectrum at 10-20 MiB and loses 7-20 %
        once it passes a third of L2.
    block_bytes_max : int, optional
        Byte budget of one store block.
    gemm_pixel_choices : sequence of int, optional
        Candidate ``P_g`` values, preferred first.
    """
    engine = make_engine(spec.precision)
    provisional = HypothesisSet.create(
        space, template_ids, defocus_ids, hyp_batch=1, n_psi=spec.n_psi
    )
    planes = provisional.defocus_planes()
    per_pixel = feature_bytes_per_pixel(
        (rect, spec.precision, len(planes)) for rect in spec.rectangles
    )

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
    target = l2_target or (executor.l2_bytes // 4 if executor.l2_bytes else 24 * MIB)
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
    hyps = HypothesisSet.create(
        space, template_ids, defocus_ids, hyp_batch=hyp_batch, n_psi=spec.n_psi
    )
    return MemoryPlan(
        gemm_pixels=gemm_pixels,
        hyp_batch=hyp_batch,
        block_pixels=block_pixels,
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
    spec : Stage
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
    spec: Stage
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
    spec: Stage,
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
        # a group's block shape is set by its plane count, not by which planes
        tuple((g.id, g.rect, g.precision, g.n_defocus) for g in groups),
        tuple(lookup),
        memory.gemm_pixels,
        memory.block_pixels,
        memory.hyp_batch,
        spec.n_psi,
        # which planes a pass searches is irrelevant to the loop: it reads the
        # groups/lookup above, and the batches depend on the plane *count* only
        memory.hyps.template_signature,
        memory.hyps.n_defocus_selected,
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
