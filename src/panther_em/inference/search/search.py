"""The search API: ``ExecutionConfig``, ``Searcher`` and ``SearchSession`` (``Stage``
lives in :mod:`.stage`).

::

    template = Template.from_hdf5("decomp.h5", device="cuda:0")
    bound = ErrorModel.from_result(template.result)
    stages = [
        Stage.create([(0, 0, 32, 32)], precision="fp16", orientations=1, planes=(5,),
                     decision=Decision(7.8, bound, absorb_radius=1)),
        Stage.create([(0, 0, 64, 128)], decision=Decision(7.8, bound)),
    ]
    searcher = Searcher(template, stages, n_defocus=11)
    result = searcher.run(ImageSet.single(micrograph, filters=ctf_planes))
    result.save("search.h5")

A :class:`Searcher` is bound to one template and one list of stages and owns the
execution resources that amortise across image sets (kernels and spectra on the
template, prepared weights and captured hypothesis loops on the executor). A
:class:`SearchSession` runs the stages over one :class:`ImageSet`: stage 0 searches
every unmasked pixel, each later stage re-scores the pixels the previous decision
left AMBIGUOUS or ACCEPTED, and the session stops early when that set is empty.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import torch

from panther_em.inference.search.decision import Decision, PixelLabel
from panther_em.inference.search.executor import Executor, plan_memory, plan_stage
from panther_em.inference.search.features import FeatureStore
from panther_em.inference.search.featurize import featurize as _featurize
from panther_em.inference.search.featurize import schedule_tiles
from panther_em.inference.search.hypotheses import HypothesisSpace, healpix_ipix
from panther_em.inference.search.results import PixelLedger, SearchResult, StageResult
from panther_em.inference.search.stage import Stage
from panther_em.inference.search.types import (
    ImageSet,
    PixelSelection,
    Precision,
    SearchGeometry,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    import numpy as np

    from panther_em.inference.search.features import Residency
    from panther_em.inference.search.featurize import Featurize, LoopOrder
    from panther_em.inference.search.template import Template
    from panther_em.inference.search.types import BatchShape, Rectangle


@dataclass(frozen=True)
class ExecutionConfig:
    """Knobs of the execution layer (everything the planner does not decide itself).

    Attributes
    ----------
    batch : BatchShape, optional
        Explicit ``(pixels, hypotheses)`` per contraction; ``None`` lets the planner
        pick from the L2-residency rule.
    block_pixels : int, optional
        Explicit store block size (a multiple of the GEMM pixel batch).
    use_cuda_graph, use_fused_kernel : bool
        Allow the graphed hypothesis loop / the fused statistics kernel.
    tile : int
        Featurizer output tile side (grid mode) for a pinned-host store, whose
        blocks must complete tile by tile to be staged out. Larger tiles waste less
        of each overlap-save FFT but their kernel spectra are bigger (8 MiB per cell
        at 1024, 134 MiB at 4096), and once they no longer fit the template's
        ``spectral_budget`` every plane re-FFTs every kernel, which costs as much as
        the correlation itself. 1024 keeps a rank-1024 stage's spectra in 8 GiB.
    device_tile : int, optional
        Tile side for a device-resident store, which has no staging constraint.
        Featurizing a pixel subset costs one correlation per (cell, tile touched),
        so the per-pixel cost is set by the tile: measured on the 60S template at
        rank 8192, 0.49 ns per pixel-cell at 1024, 0.22 at 2048 (2560^2 FFTs) and
        0.47 at 4096 (134 MB spectra, memory-bound passes). ``None`` covers each
        image with one tile.
    cell_chunk : int
        Kernels per featurization chunk.
    image_batch : int
        Boxes per inner-product GEMM (center mode).
    loop_order : {"pixel_major", "cell_major"}
        Featurizer loop order of a pinned-host store (``pixel_major`` completes
        blocks tile by tile, which staging needs). A device-resident store is always
        featurized ``cell_major``: each kernel spectrum is FFT'd once and reused
        across the image's tiles instead of once per tile.
    ledger_device : device, optional
        Where the dense per-pixel state lives; defaults to the compute device.
    prepared_budget : int, optional
        Byte budget of the prepared-weight cache.
    show_progress : bool
        Print per-stage progress bars.
    store_residency : {"auto", "device", "pinned_host"}
        Where a stage's feature store lives; ``"auto"`` keeps it on the device when it
        fits ``device_store_fraction`` of the free memory and otherwise pins it on
        the host and streams blocks.
    l2_target : int, optional
        Bytes of spectrum per contraction to keep L2-resident (the hypothesis batch
        follows from it); defaults to two thirds of the device's L2.
    block_bytes_max : int
        Byte budget of one feature-store block (256 MiB).
    device_store_fraction : float
        See ``store_residency``.
    trace : bool
        Synchronise the device at every session mark (``SearchSession.marks``) so
        the marks' wall-clock deltas are per-phase device times.
    """

    batch: BatchShape | None = None
    block_pixels: int | None = None
    use_cuda_graph: bool = True
    use_fused_kernel: bool = True
    tile: int = 1024
    device_tile: int | None = 2048
    cell_chunk: int = 64
    image_batch: int = 1024
    loop_order: LoopOrder = "pixel_major"
    ledger_device: torch.device | str | None = None
    prepared_budget: int | None = None
    show_progress: bool = False
    store_residency: Residency | Literal["auto"] = "auto"
    l2_target: int | None = None
    block_bytes_max: int = 256 << 20
    device_store_fraction: float = 0.5
    trace: bool = False


# --------------------------------------------------------------------------- #
# Searcher and session
# --------------------------------------------------------------------------- #


class Searcher:
    """A template plus its stages, ready to search image sets.

    Parameters
    ----------
    template : Template
    stages : sequence of Stage
        Stages in execution order (typically increasing rank).
    execution : ExecutionConfig, optional
    n_defocus : int, optional
        Defocus planes every image set must carry (``ImageSet.filters``). Defaults to 1.
    defocus_values : torch.Tensor, optional
        ``(n_defocus,)`` labels for the planes, reported by ``SearchResult.decode()``.
    keep_stage_results : bool, optional
        Retain every stage's compact maps on the final :class:`SearchResult` (and in
        its HDF5 file). Defaults to False.
    featurize : callable, optional
        Override :func:`~.featurize.featurize` (tests inject a stub with the same
        signature).
    """

    def __init__(
        self,
        template: Template,
        stages: Sequence[Stage],
        *,
        execution: ExecutionConfig | None = None,
        n_defocus: int = 1,
        defocus_values: torch.Tensor | None = None,
        keep_stage_results: bool = False,
        featurize: Featurize | None = None,
    ) -> None:
        self.template = template
        self.stages = tuple(stages)
        if not self.stages:
            raise ValueError("a searcher needs at least one stage")
        self.execution = execution or ExecutionConfig()
        self.keep_stage_results = bool(keep_stage_results)
        self.space = HypothesisSpace.from_result(
            template.result, n_defocus=n_defocus, defocus_values=defocus_values
        )
        self.featurize: Featurize = featurize or _featurize
        # NESTED HEALPix ids of the template's orientations, when a stage samples
        # orientations by HEALPix level (None otherwise, or when not HEALPix)
        self.ipix = None
        if any(isinstance(s.orientations, int) for s in self.stages):
            found = healpix_ipix(self.space)
            self.ipix = None if found is None else found[1]
        self.executor = Executor(
            template.device, template.weights, prepared_budget=self.execution.prepared_budget
        )
        self._validate()

    def _validate(self) -> None:
        k_max, eig_max = self.template.k_max, self.template.eig_max
        for i, stage in enumerate(self.stages):
            for k0, m0, ke, me in stage.rectangles:
                if k0 + ke > k_max or m0 + me > eig_max:
                    raise ValueError(
                        f"stage {i}: rectangle {(k0, m0, ke, me)} exceeds the stored "
                        f"(k_max={k_max}, eig_max={eig_max}) feature space"
                    )
            if stage.precision is Precision.FP16 and not self.executor.is_cuda:
                raise ValueError(f"stage {i}: fp16 needs a CUDA device")
            template_ids, defocus_ids = stage.hypothesis_ids(self.space, self.ipix)
            if defocus_ids is not None and int(defocus_ids.max()) >= self.space.n_defocus:
                raise ValueError(
                    f"stage {i}: planes {stage.planes} exceed the {self.space.n_defocus} "
                    "defocus plane(s) of the searcher"
                )
            if template_ids is not None and int(template_ids.max()) >= self.space.n_templates:
                raise ValueError(f"stage {i}: orientation ids exceed the template's")

    @property
    def device(self) -> torch.device:
        """Compute device."""
        return self.template.device

    def session(self, images: ImageSet) -> SearchSession:
        """A session over one image set (iterate it for per-stage results)."""
        return SearchSession(self, images)

    def run(self, images: ImageSet) -> SearchResult:
        """Run every stage over ``images``."""
        return self.session(images).run_all()

    def run_many(self, image_sets: Iterable[ImageSet]) -> Iterator[SearchResult]:
        """Run the stages over each image set in turn, reusing cached resources."""
        for images in image_sets:
            yield self.run(images)

    def release(self) -> None:
        """Drop cached loops and prepared weights (kernels stay with the template)."""
        self.executor.release()


class SearchSession:
    """One image set's walk through the stages.

    Attributes
    ----------
    ledger : PixelLedger
        The dense per-pixel state.
    pixels : PixelSelection
        The pixels the next stage searches.
    stage_results : list[StageResult]
        Filled when the searcher keeps stage results.
    marks : list[tuple[str, float, int]]
        ``(name, time, allocated_bytes)`` timeline marks of every phase.
    """

    def __init__(self, searcher: Searcher, images: ImageSet) -> None:
        if images.n_defocus != searcher.space.n_defocus:
            raise ValueError(
                f"image set carries {images.n_defocus} defocus plane(s); the searcher "
                f"expects {searcher.space.n_defocus}"
            )
        self.searcher = searcher
        self.images = images
        self.geometry = SearchGeometry.from_images(images, searcher.template.template_shape)
        exe = searcher.execution
        initial = PixelSelection.from_mask(self.geometry, images.mask)
        self.ledger = PixelLedger(
            self.geometry, exe.ledger_device or searcher.device, initial=initial
        )
        self.pixels: PixelSelection = initial
        self.store: FeatureStore | None = None
        self._store_tile = 0  # the tile the store's column order was built with
        self.stage_index = 0
        self.stage_results: list[StageResult] = []
        self.elapsed_s = 0.0
        self._finished = False
        self.marks: list[tuple[str, float, int]] = []
        self._mark("session")

    def _mark(self, name: str) -> None:
        """Record ``(name, time, allocated_bytes)`` (synchronised under ``trace``)."""
        allocated = 0
        if self.searcher.executor.is_cuda:
            if self.searcher.execution.trace:
                torch.cuda.synchronize(self.searcher.device)
            allocated = torch.cuda.memory_allocated(self.searcher.device)
        self.marks.append((name, time.perf_counter(), allocated))

    # -- stage machinery ---------------------------------------------------------

    def _prepare_store(
        self,
        block_pixels: int,
        precision: Precision,
        rectangles: Sequence[Rectangle],
        planes: Sequence[int],
        residency: Residency = "device",
    ) -> FeatureStore:
        """Re-block / compact the store to the current pixels; featurize what is new."""
        exe = self.searcher.execution
        executor = self.searcher.executor
        pixels = self.pixels
        store = self.store
        rects = tuple(tuple(int(v) for v in r) for r in rectangles)
        reusable = store is not None and any(
            g.rect in rects and g.precision is precision for g in store.groups
        )
        if store is not None and not reusable:
            # Nothing of the previous stage's store is read again (a different
            # rectangle or precision): drop it rather than compact it.
            store = None
        if reusable:
            tile = self._store_tile  # the subset keeps the store's column order
        elif residency == "pinned_host":
            tile = exe.tile
        elif exe.device_tile is not None:
            tile = exe.device_tile
        else:
            tile = max(int(g) for g in self.geometry.grid)  # one tile per image
        schedule = schedule_tiles(
            self.geometry, pixels, tile=tile, image_batch=exe.image_batch
        )
        if store is None:
            store = FeatureStore(
                self.geometry,
                pixels,
                schedule.column_index,
                block_pixels=block_pixels,
                n_defocus=self.images.n_defocus,
                device=self.searcher.device,
                residency=residency,
            )
            store.copy_stream = executor.copy
        else:
            keep = torch.isin(store.column_index, pixels.index.to(store.column_index.device))
            if (
                not bool(keep.all())
                or store.block_pixels != block_pixels
                or store.residency != residency
            ):
                store = store.subset(keep, block_pixels=block_pixels, residency=residency)
                store.copy_stream = executor.copy
            if not torch.equal(
                schedule.column_index, store.column_index.to(schedule.column_index.device)
            ):
                raise RuntimeError("featurization column order diverged from the store's")
        self._store_tile = tile
        self._mark("store:ready")
        missing = [
            (rect, mp)
            for rect in rectangles
            if (mp := store.missing_planes(rect, precision, planes))
        ]
        if missing and executor.featurize is not None:
            # Blocks were zero-allocated on the current stream: featurize after it.
            executor.featurize.wait_stream(torch.cuda.current_stream(executor.device))
        for rect, missing_planes in missing:
            group = store.add_group(rect, precision, missing_planes)
            self._mark("store:group_allocated")
            self.searcher.featurize(
                self.searcher.template.kernels,
                self.images,
                group,
                store.writer(group),
                schedule,
                chunk=exe.cell_chunk,
                order=exe.loop_order if store.residency == "pinned_host" else "cell_major",
                stream=executor.featurize,
            )
        if missing and executor.featurize is not None:
            # Everything downstream (compute stream, compaction) sees complete blocks.
            torch.cuda.current_stream(executor.device).wait_stream(executor.featurize)
        self.store = store
        return store

    def _run_stage(self, index: int) -> StageResult:
        searcher = self.searcher
        exe = searcher.execution
        stage = searcher.stages[index]
        t0 = time.perf_counter()
        self._mark(f"stage{index}:start")
        template_ids, defocus_ids = stage.hypothesis_ids(searcher.space, searcher.ipix)
        memory = plan_memory(
            searcher.executor,
            stage,
            n_pixels=len(self.pixels),
            space=searcher.space,
            template_ids=template_ids,
            defocus_ids=defocus_ids,
            batch=exe.batch,
            block_pixels=exe.block_pixels,
            use_cuda_graph=exe.use_cuda_graph,
            use_fused_kernel=exe.use_fused_kernel,
            store=self.store,
            residency=exe.store_residency,
            l2_target=exe.l2_target,
            block_bytes_max=exe.block_bytes_max,
            device_store_fraction=exe.device_store_fraction,
        )
        hyps = memory.hyps
        store = self._prepare_store(
            memory.block_pixels,
            stage.precision,
            stage.rectangles,
            hyps.defocus_planes(),
            memory.residency,
        )
        if searcher.executor.is_cuda:
            torch.cuda.synchronize(searcher.device)
        t_featurized = time.perf_counter()
        self._mark(f"stage{index}:featurized")
        plan = plan_stage(searcher.executor, index, stage, store, memory)
        self._mark(f"stage{index}:planned")
        progress = None
        if exe.show_progress:
            from tqdm import tqdm  # type: ignore[import-untyped]

            bar = tqdm(total=store.n_pixels, desc=f"stage {index}", unit="px", unit_scale=True)
            progress = bar.update
        maps = searcher.executor.run(store, plan, progress=progress)
        if exe.show_progress:
            bar.close()
        if searcher.executor.is_cuda:
            torch.cuda.synchronize(searcher.device)
        t_executed = time.perf_counter()
        self._mark(f"stage{index}:executed")
        column_index = store.column_index
        self.ledger.write(maps, column_index, index, hyps)
        self._mark(f"stage{index}:ledger")
        if stage.decision is not None:
            stage.decision.apply(self.ledger, column_index, stage)
        self.pixels = self.ledger.next_stage_pixels()
        self._mark(f"stage{index}:decided")
        labels_after = self.ledger.label[column_index.to(self.ledger.device)].clone()
        n_ambiguous = int(self.ledger.pixels_with(PixelLabel.AMBIGUOUS).numel())
        elapsed = time.perf_counter() - t0
        self.elapsed_s += elapsed
        return StageResult(
            stage_index=index,
            stage=stage,
            pixel_index=column_index.clone(),
            maps=maps,
            best_index=hyps.decode(maps.best_virtual),
            labels_after=labels_after,
            n_ambiguous_after=n_ambiguous,
            elapsed_s=elapsed,
            featurize_s=t_featurized - t0,
            execute_s=t_executed - t_featurized,
        )

    # -- public ------------------------------------------------------------------

    def __iter__(self) -> Iterator[StageResult]:
        """Run the remaining stages one at a time (stops when no pixel is left)."""
        stages = self.searcher.stages
        while self.stage_index < len(stages) and not self._finished:
            if len(self.pixels) == 0:
                break
            result = self._run_stage(self.stage_index)
            self.stage_index += 1
            if self.searcher.keep_stage_results:
                self.stage_results.append(result)
            yield result
        self._finished = True

    def run_all(self) -> SearchResult:
        """Run every remaining stage and return the final result."""
        for _ in self:
            pass
        return self.snapshot()

    def snapshot(self) -> SearchResult:
        """The current ledger as a :class:`SearchResult`."""
        return SearchResult.from_ledger(
            self.ledger,
            self.searcher.space,
            self.searcher.stages,
            stage_results=self.stage_results,
            elapsed_s=self.elapsed_s,
        )

