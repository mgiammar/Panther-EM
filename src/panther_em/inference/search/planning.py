"""Stage planning: every decision about a stage's work, made before the work starts.

* :class:`DeviceInfo` -- what the compute device offers (probed once per searcher):
  CUDA, graph capture, the fused kernels, the L2 size; free memory on demand.
* :func:`plan_memory` -- one pass's shapes (:class:`MemoryPlan`): the GEMM batch, the
  hypothesis batch, the store block size, eager or graphed loop, fused or not.
* :func:`plan_stage` -- the whole stage (:class:`StagePlan`): its hypotheses, the
  featurization engine, the *bands* (row-contiguous chunks of its pixels whose
  features fit the device) and, per band, the *passes* (one defocus plane at a
  time, or all at once), each with its memory plan.

A plan is a value: it can be built and inspected (and tested) without running
anything; :class:`~panther_em.inference.search.runner.StageRunner` executes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from panther_em.inference.search.config import ExecutionConfig, Tuning
from panther_em.inference.search.correlation.contraction import make_engine
from panther_em.inference.search.correlation.kernels import fused_kernel_loader
from panther_em.inference.search.correlation.reduction import (
    SpectrumSpec,
    _cuda_graph_capture_supported,
    fused_stats_supported,
)
from panther_em.inference.search.featurization import choose_mode
from panther_em.inference.search.hypotheses import HypothesisSet
from panther_em.inference.search.types import PixelSelection

if TYPE_CHECKING:
    from collections.abc import Iterable

    import numpy as np

    from panther_em.inference.search.correlation.contraction import ContractionEngine
    from panther_em.inference.search.featurization import Mode
    from panther_em.inference.search.hypotheses import HypothesisSpace
    from panther_em.inference.search.stage import Stage
    from panther_em.inference.search.types import Precision, SearchGeometry

LoopKind = Literal["eager", "graph"]
MIB = 1024**2


# --------------------------------------------------------------------------- #
# The device
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DeviceInfo:
    """What the compute device offers the planner.

    Build with :meth:`probe`; the constructor only stores the values (tests build
    one by hand).

    Attributes
    ----------
    device : torch.device
        The compute device (CUDA with an explicit index).
    cuda_graphs : bool
        CUDA graph capture works on the device.
    fused_kernel : bool
        The fused iRFFT + statistics extension compiled (probing builds it).
    l2_bytes : int, optional
        L2 cache size, when known.
    """

    device: torch.device
    cuda_graphs: bool = False
    fused_kernel: bool = False
    l2_bytes: int | None = None

    @classmethod
    def probe(cls, device: torch.device | str) -> DeviceInfo:
        """Probe ``device`` once (the fused kernel's JIT build happens here)."""
        device = torch.device(device)
        if device.type != "cuda":
            return cls(device)
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        props = torch.cuda.get_device_properties(index)
        return cls(
            torch.device("cuda", index),
            cuda_graphs=_cuda_graph_capture_supported(index),
            fused_kernel=fused_kernel_loader._try_compile() is not None,
            l2_bytes=int(getattr(props, "L2_cache_size", 0)) or None,
        )

    @property
    def is_cuda(self) -> bool:
        """Whether the device is a CUDA device."""
        return self.device.type == "cuda"

    def free_bytes(self) -> int | None:
        """Memory a new allocation can use now (``None`` off CUDA).

        Driver-free memory plus the caching allocator's idle reserve.
        """
        if not self.is_cuda:
            return None
        index = self.device.index
        free, _total = torch.cuda.mem_get_info(index)
        idle = torch.cuda.memory_reserved(index) - torch.cuda.memory_allocated(index)
        return int(free) + int(idle)


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


# --------------------------------------------------------------------------- #
# One pass: the memory plan
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class MemoryPlan:
    """Shapes of one pass, fixed before its store exists.

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
        The pass's hypotheses, batched by ``hyp_batch``.
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
    device: DeviceInfo,
    stage: Stage,
    *,
    n_pixels: int,
    space: HypothesisSpace,
    template_ids: torch.Tensor | None = None,
    defocus_ids: torch.Tensor | None = None,
    tuning: Tuning | None = None,
    gemm_pixel_choices: tuple[int, ...] = (1024, 512),
) -> MemoryPlan:
    """Resolve a pass's GEMM batch, block size, loop kind and hypotheses.

    Parameters
    ----------
    device : DeviceInfo
        The device's capabilities.
    stage : Stage
        The stage the pass belongs to.
    n_pixels : int
        Pixels the pass searches.
    space : HypothesisSpace
        The searcher's hypothesis space.
    template_ids, defocus_ids : torch.Tensor, optional
        The pass's hypothesis restriction (``None`` = all), already normalised by
        ``Stage.hypothesis_ids``.
    tuning : Tuning, optional
        Explicit batch / block sizes, the L2 target, the block budget and whether
        graphs and the fused kernels are allowed.
    gemm_pixel_choices : tuple of int, optional
        Candidate ``P_g`` values, preferred first.

    Notes
    -----
    The hypothesis batch keeps the spectrum of one contraction L2-resident:
    ``tuning.l2_target`` bytes, a quarter of the device's L2 by default (24 MiB when
    unknown). Measured on an RTX 6000 Ada (96 MiB L2), the loop is fastest with the
    spectrum at 10-20 MiB and loses 7-20 % once it passes a third of L2.
    """
    tuning = tuning or Tuning()
    engine = make_engine(stage.precision)
    provisional = HypothesisSet.create(
        space, template_ids, defocus_ids, hyp_batch=1, n_psi=stage.n_psi
    )
    planes = provisional.defocus_planes()
    per_pixel = feature_bytes_per_pixel(
        (rect, stage.precision, len(planes)) for rect in stage.rectangles
    )

    probe = SpectrumSpec(
        dtype=engine.spectrum_dtype,
        n_freq=stage.k_stop,
        n_psi=stage.n_psi,
        gemm_pixels=1,
        n_sub=1,
        hyp_batch=1,
        n_virtual=provisional.n_virtual,
        device=device.device,
    )
    fused = bool(
        tuning.use_fused_kernel and device.fused_kernel and fused_stats_supported(probe)
    )

    elem = torch.empty((), dtype=engine.spectrum_dtype).element_size()
    target = tuning.l2_target or (device.l2_bytes // 4 if device.l2_bytes else 24 * MIB)
    multiple = hyp_batch_multiple(stage.n_psi)
    if tuning.batch is not None:
        gemm_pixels, hyp_batch = tuning.batch.pixel, tuning.batch.hyp
    else:
        gemm_pixels = int(gemm_pixel_choices[0]) if device.is_cuda else 256
        budget = max(multiple, target // (stage.k_stop * gemm_pixels * elem))
        hyp_batch = max(multiple, (budget // multiple) * multiple)
        hyp_batch = min(hyp_batch, max(multiple, provisional.n_virtual))
    gemm_pixels = max(1, min(gemm_pixels, max(1, n_pixels)))

    block_pixels = tuning.block_pixels
    if block_pixels is None:
        fit = max(
            gemm_pixels,
            (int(tuning.block_bytes_max) // max(1, per_pixel))
            // gemm_pixels
            * gemm_pixels,
        )
        needed = -(-max(1, n_pixels) // gemm_pixels) * gemm_pixels
        block_pixels = min(fit, needed)
    block_pixels = max(gemm_pixels, (block_pixels // gemm_pixels) * gemm_pixels)

    loop: LoopKind = (
        "graph"
        if tuning.use_cuda_graph and device.is_cuda and device.cuda_graphs and fused
        else "eager"
    )
    hyps = HypothesisSet.create(
        space, template_ids, defocus_ids, hyp_batch=hyp_batch, n_psi=stage.n_psi
    )
    return MemoryPlan(
        gemm_pixels=gemm_pixels,
        hyp_batch=hyp_batch,
        block_pixels=block_pixels,
        loop=loop,
        fused=fused,
        engine=engine,
        hyps=hyps,
        device=device.device,
    )


# --------------------------------------------------------------------------- #
# The whole stage: bands x passes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class PassPlan:
    """One featurize-and-correlate pass: the planes it holds and its shapes."""

    planes: tuple[int, ...]
    memory: MemoryPlan


@dataclass(frozen=True, eq=False)
class BandPlan:
    """One band: a row-contiguous chunk of the stage's pixels and its passes."""

    pixels: PixelSelection
    passes: tuple[PassPlan, ...]


@dataclass(frozen=True, eq=False)
class StagePlan:
    """Everything decided about one stage before its work starts.

    Attributes
    ----------
    index : int
        The stage's position in the searcher's list.
    stage : Stage
        What the stage searches.
    hyps : HypothesisSet
        The stage's hypotheses (all its planes); decodes the maps' virtual ids.
    mode : {"fft", "direct"}
        The featurization engine.
    bands : tuple[BandPlan, ...]
        The stage's pixels in bands (one band unless the features do not fit), each
        searched in its passes; the maps are concatenated in band order.
    """

    index: int
    stage: Stage
    hyps: HypothesisSet
    mode: Mode
    bands: tuple[BandPlan, ...]

    @property
    def pixel_index(self) -> torch.Tensor:
        """``(P,)`` flat pixel ids in the maps' order (the bands, concatenated)."""
        parts = [band.pixels.index.to("cpu") for band in self.bands]
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    @property
    def n_pixels(self) -> int:
        """Pixels the stage searches."""
        return sum(len(band.pixels) for band in self.bands)

    @property
    def n_passes(self) -> int:
        """Featurize-and-correlate passes over all bands."""
        return sum(len(band.passes) for band in self.bands)


def plan_bands(
    pixels: PixelSelection, per_pixel: int, limit: int | None, gemm_pixels: int
) -> list[PixelSelection]:
    """Row-contiguous chunks of ``pixels`` whose features fit ``limit`` bytes.

    One band (the whole selection) when the features fit or no limit is known;
    otherwise the sorted selection is cut into equal chunks of a multiple of
    ``gemm_pixels`` pixels that fit, each a band of rows the FFT engine covers with
    one region.
    """
    if limit is None or len(pixels) == 0 or per_pixel * len(pixels) <= limit:
        return [pixels]
    fit = max(gemm_pixels, (limit // max(1, per_pixel)) // gemm_pixels * gemm_pixels)
    n_bands = -(-len(pixels) // fit)
    per_band = -(-len(pixels) // n_bands)
    per_band = -(-per_band // gemm_pixels) * gemm_pixels
    index = pixels.index
    if not bool(torch.all(index[1:] >= index[:-1])):
        index = torch.sort(index).values
    return [
        PixelSelection(index[i : i + per_band], pixels.geometry)
        for i in range(0, int(index.numel()), per_band)
    ]


def plan_stage(
    index: int,
    stage: Stage,
    pixels: PixelSelection,
    geometry: SearchGeometry,
    *,
    space: HypothesisSpace,
    device: DeviceInfo,
    execution: ExecutionConfig | None = None,
    ipix: np.ndarray | None = None,
    free_bytes: int | None = None,
) -> StagePlan:
    """Plan one stage over ``pixels``.

    Parameters
    ----------
    index : int
        The stage's position in the searcher's list.
    stage : Stage
        What the stage searches.
    pixels : PixelSelection
        The pixels it searches.
    geometry : SearchGeometry
        Their grid (the engine choice reads its shape).
    space : HypothesisSpace
        The searcher's hypothesis space.
    device : DeviceInfo
        The device's capabilities.
    execution : ExecutionConfig, optional
        Engine override, the store fraction and the tuning overrides.
    ipix : numpy.ndarray, optional
        NESTED HEALPix ids of the template's orientations (for a stage that samples
        orientations by level).
    free_bytes : int, optional
        Free device memory; with ``execution.device_store_fraction`` it bounds one
        band's features. ``None`` keeps the stage in one band.
    """
    execution = execution or ExecutionConfig()
    tuning = execution.tuning
    template_ids, defocus_ids = stage.hypothesis_ids(space, ipix)

    def memory_for(planes: torch.Tensor | None, n_pixels: int) -> MemoryPlan:
        return plan_memory(
            device,
            stage,
            n_pixels=n_pixels,
            space=space,
            template_ids=template_ids,
            defocus_ids=planes,
            tuning=tuning,
        )

    memory = memory_for(defocus_ids, len(pixels))
    planes = memory.hyps.defocus_planes()
    # one pass per plane (only that plane's features are resident), or one pass
    # over every plane
    passes = (
        [(p,) for p in planes]
        if tuning.plane_major and len(planes) > 1
        else [tuple(planes)]
    )
    mode: Mode = (
        choose_mode(geometry, len(pixels))
        if execution.featurize_mode == "auto"
        else execution.featurize_mode
    )
    per_pixel = feature_bytes_per_pixel(
        (rect, stage.precision, len(passes[0])) for rect in stage.rectangles
    )
    limit = (
        None
        if free_bytes is None
        else int(free_bytes * execution.device_store_fraction)
    )
    bands = plan_bands(pixels, per_pixel, limit, memory.gemm_pixels)

    def pass_memory(planes: tuple[int, ...], band: PixelSelection) -> MemoryPlan:
        if len(passes) == 1 and len(bands) == 1:
            return memory
        return memory_for(torch.tensor(planes, dtype=torch.long), len(band))

    return StagePlan(
        index=index,
        stage=stage,
        hyps=memory.hyps,
        mode=mode,
        bands=tuple(
            BandPlan(band, tuple(PassPlan(p, pass_memory(p, band)) for p in passes))
            for band in bands
        ),
    )
