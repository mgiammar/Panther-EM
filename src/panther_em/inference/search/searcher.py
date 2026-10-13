"""The developer-facing search API: ``Searcher`` and ``SearchSession``.

::

    template = Template.from_hdf5("decomp.h5", device="cuda:0")
    schedule = nested_schedule(
        [(32, 16), (128, 48)],
        precision=["fp16", "fp32"],
        decision=[ZScoreIntervalClassifier(7.8), ThresholdClassifier(7.8)],
        error_model=ErrorModel.from_result(template.result),
    )
    searcher = Searcher(template, schedule)
    result = searcher.run(ImageSet.single(micrograph))

A :class:`Searcher` is bound to one template and one schedule and owns the execution
resources that amortize across images (prepared weights, captured hypothesis loops).
A :class:`SearchSession` runs the schedule over one :class:`ImageSet`, yielding a
:class:`~panther_em.inference.search.results.StageResult` per stage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from panther_em.inference.search.executor import Executor, plan_memory, plan_stage
from panther_em.inference.search.featurize import featurize as _featurize
from panther_em.inference.search.featurize import schedule_tiles
from panther_em.inference.search.hypotheses import HypothesisSpace, healpix_ipix
from panther_em.inference.search.ledger import PixelLedger
from panther_em.inference.search.results import SearchResult, StageResult
from panther_em.inference.search.schedule import ErrorModel
from panther_em.inference.search.features import FeatureStore
from panther_em.inference.search.types import (
    ImageSet,
    PixelSelection,
    Precision,
    SearchGeometry,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from panther_em.inference.search.template import Template
    from panther_em.inference.search.features import Residency
    from panther_em.inference.search.featurize import Featurize, LoopOrder
    from panther_em.inference.search.schedule import SearchSchedule
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
        Featurizer output tile side (grid mode). Larger tiles waste less of each
        overlap-save FFT but their kernel spectra are bigger (8 MiB per cell at
        1024, 134 MiB at 4096), and once they no longer fit the template's
        ``spectral_budget`` every plane re-FFTs every kernel, which costs as much as
        the correlation itself. 1024 keeps a rank-1024 stage's spectra in 8 GiB.
    cell_chunk : int
        Kernels per featurization chunk.
    image_batch : int
        Boxes per inner-product GEMM (center mode).
    loop_order : {"pixel_major", "cell_major"}
        Featurizer loop order.
    ledger_device : device, optional
        Where the dense per-pixel state lives; defaults to the compute device.
    prepared_budget : int, optional
        Byte budget of the prepared-weight cache.
    show_progress : bool
        Print per-stage progress bars.
    store_residency : {"auto", "device", "pinned_host"}
        Where a stage's feature store lives; ``"auto"`` keeps it on the device when it
        fits half the free memory and otherwise pins it on the host and streams blocks.
    l2_target : int, optional
        Bytes of spectrum per contraction to keep L2-resident (the hypothesis batch
        follows from it); defaults to two thirds of the device's L2.
    block_bytes_max : int
        Byte budget of one feature-store block (256 MiB).
    device_store_fraction : float
        With ``store_residency="auto"``, a store larger than this fraction of the free
        device memory is pinned on the host and streamed.
    trace : bool
        Synchronise the device at every session mark (``SearchSession.marks``) so
        the marks' wall-clock deltas are per-phase device times.
    """

    batch: BatchShape | None = None
    block_pixels: int | None = None
    use_cuda_graph: bool = True
    use_fused_kernel: bool = True
    tile: int = 1024
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


class Searcher:
    """A template plus a schedule, ready to search image sets.

    Parameters
    ----------
    template : Template
    schedule : SearchSchedule
        Stage errors missing where a decision needs them are filled from
        ``error_model`` (default: the spectral model of the template).
    execution : ExecutionConfig, optional
    n_defocus : int, optional
        Defocus planes every image set must carry (``ImageSet.filters``). Defaults to 1.
    defocus_values : torch.Tensor, optional
        ``(n_defocus,)`` labels for the planes, reported by ``SearchResult.decode()``.
    error_model : ErrorModel, optional
    featurize : callable, optional
        Override :func:`~.featurize.featurize` (tests inject a stub with the same
        signature).
    """

    def __init__(
        self,
        template: Template,
        schedule: SearchSchedule,
        *,
        execution: ExecutionConfig | None = None,
        n_defocus: int = 1,
        defocus_values: torch.Tensor | None = None,
        error_model: ErrorModel | None = None,
        featurize: Featurize | None = None,
    ) -> None:
        self.template = template
        self.execution = execution or ExecutionConfig()
        self.space = HypothesisSpace.from_result(
            template.result, n_defocus=n_defocus, defocus_values=defocus_values
        )
        if schedule.requires_errors():
            model = error_model or ErrorModel.from_result(template.result)
            schedule = schedule.with_errors(model)
        self.schedule = schedule
        self.featurize: Featurize = featurize or _featurize
        # NESTED HEALPix ids of the template's orientations, when a stage samples
        # orientations by HEALPix level (None otherwise, or when not HEALPix)
        self.ipix = None
        if any(isinstance(s.orientations, int) for s in schedule.stages):
            found = healpix_ipix(self.space)
            self.ipix = None if found is None else found[1]
        self.executor = Executor(
            template.device, template.weights, prepared_budget=self.execution.prepared_budget
        )
        self._validate()

    def _validate(self) -> None:
        k_max, eig_max = self.template.k_max, self.template.eig_max
        for i, spec in enumerate(self.schedule.stages):
            for k0, m0, ke, me in spec.rectangles:
                if k0 + ke > k_max or m0 + me > eig_max:
                    raise ValueError(
                        f"stage {i}: rectangle {(k0, m0, ke, me)} exceeds the stored "
                        f"(k_max={k_max}, eig_max={eig_max}) feature space"
                    )
            if spec.precision is Precision.FP16 and not self.executor.is_cuda:
                raise ValueError(f"stage {i}: fp16 needs a CUDA device")
            template_ids, defocus_ids = spec.hypothesis_ids(self.space, self.ipix)
            if defocus_ids is not None and int(defocus_ids.max()) >= self.space.n_defocus:
                raise ValueError(
                    f"stage {i}: planes {spec.planes} exceed the {self.space.n_defocus} "
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
        """Run the schedule over each image set in turn, reusing cached resources."""
        for images in image_sets:
            yield self.run(images)

    def release(self) -> None:
        """Drop cached loops and prepared weights (kernels stay with the template)."""
        self.executor.release()


class SearchSession:
    """One image set's walk through the schedule."""

    def __init__(self, searcher: Searcher, images: ImageSet) -> None:
        if images.n_defocus != searcher.space.n_defocus:
            raise ValueError(
                f"image set carries {images.n_defocus} defocus plane(s); the searcher "
                f"expects {searcher.space.n_defocus}"
            )
        self.searcher = searcher
        self.images = images
        self.geometry = SearchGeometry.from_images(
            images, searcher.template.template_shape
        )
        exe = searcher.execution
        initial = PixelSelection.from_mask(self.geometry, images.mask)
        self.ledger = PixelLedger(
            self.geometry,
            exe.ledger_device or searcher.device,
            initial=initial,
        )
        self.pixels: PixelSelection = initial
        self.store: FeatureStore | None = None
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
        spec_rects: Sequence[Rectangle],
        planes: Sequence[int],
        residency: Residency = "device",
    ) -> FeatureStore:
        """Re-block / compact the store to the current pixels; featurize what is new."""
        exe = self.searcher.execution
        pixels = self.pixels
        store = self.store
        if store is None:
            schedule = schedule_tiles(
                self.geometry, pixels, tile=exe.tile, image_batch=exe.image_batch
            )
            store = FeatureStore(
                self.geometry,
                pixels,
                schedule.column_index,
                block_pixels=block_pixels,
                n_defocus=self.images.n_defocus,
                device=self.searcher.device,
                residency=residency,
            )
            store.copy_stream = self.searcher.executor.copy
        else:
            keep = torch.isin(
                store.column_index, pixels.index.to(store.column_index.device)
            )
            if not bool(keep.all()) or store.block_pixels != block_pixels:
                store = store.subset(keep, block_pixels=block_pixels)
            schedule = schedule_tiles(
                self.geometry, pixels, tile=exe.tile, image_batch=exe.image_batch
            )
            if not torch.equal(
                schedule.column_index,
                store.column_index.to(schedule.column_index.device),
            ):
                raise RuntimeError("featurizer column order diverged from the store's")
        self._mark("store:ready")
        ctx = self.searcher.executor
        missing = [
            (rect, mp)
            for rect in spec_rects
            if (mp := store.missing_planes(rect, precision, planes))
        ]
        if missing and ctx.featurize is not None:
            # Blocks were zero-allocated on the current stream: featurize after it.
            ctx.featurize.wait_stream(torch.cuda.current_stream(ctx.device))
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
                order=exe.loop_order,
                stream=ctx.featurize,
            )
        if missing and ctx.featurize is not None:
            # Everything downstream (compute stream, compaction) sees complete blocks.
            torch.cuda.current_stream(ctx.device).wait_stream(ctx.featurize)
        self.store = store
        return store

    def _run_stage(self, index: int) -> StageResult:
        searcher = self.searcher
        exe = searcher.execution
        spec = searcher.schedule.stages[index]
        t0 = time.perf_counter()
        self._mark(f"stage{index}:start")
        template_ids, defocus_ids = spec.hypothesis_ids(searcher.space, searcher.ipix)
        memory = plan_memory(
            searcher.executor,
            spec,
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
            spec.precision,
            spec.rectangles,
            hyps.defocus_planes(),
            memory.residency,
        )
        if searcher.executor.is_cuda:
            torch.cuda.synchronize(searcher.device)
        t_featurized = time.perf_counter()
        self._mark(f"stage{index}:featurized")
        plan = plan_stage(searcher.executor, index, spec, store, memory)
        self._mark(f"stage{index}:planned")
        progress = None
        if exe.show_progress:
            from tqdm import tqdm  # type: ignore[import-untyped]

            bar = tqdm(
                total=store.n_pixels, desc=f"stage {index}", unit="px", unit_scale=True
            )
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
        searcher.schedule.decisions[index](self.ledger, column_index, spec, index)
        self.pixels = self.ledger.uncertain()
        self._mark(f"stage{index}:decided")
        labels_after = self.ledger.label[column_index.to(self.ledger.device)].clone()
        elapsed = time.perf_counter() - t0
        self.elapsed_s += elapsed
        return StageResult(
            stage_index=index,
            spec=spec,
            pixel_index=column_index.clone(),
            maps=maps,
            best_index=hyps.decode(maps.best_virtual),
            labels_after=labels_after,
            n_uncertain_after=len(self.pixels),
            elapsed_s=elapsed,
            featurize_s=t_featurized - t0,
            execute_s=t_executed - t_featurized,
        )

    # -- public ------------------------------------------------------------------

    def __iter__(self) -> Iterator[StageResult]:
        """Run the remaining stages one at a time."""
        schedule = self.searcher.schedule
        while self.stage_index < len(schedule.stages) and not self._finished:
            if len(self.pixels) == 0:
                self._finished = True
                break
            result = self._run_stage(self.stage_index)
            self.stage_index += 1
            if schedule.keep_stage_results:
                self.stage_results.append(result)
            yield result
            if schedule.stop_when_empty and result.n_uncertain_after == 0:
                self._finished = True
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
            self.searcher.schedule,
            stages=self.stage_results,
            elapsed_s=self.elapsed_s,
        )
