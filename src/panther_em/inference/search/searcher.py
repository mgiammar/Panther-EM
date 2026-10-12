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
from functools import cached_property
from typing import TYPE_CHECKING, Literal

import torch

from panther_em.inference.search.context import Planner, SearchContext
from panther_em.inference.search.executor import (
    StageExecutor,
    StageMaps,
    plan_memory,
    plan_stage,
)
from panther_em.inference.search.featurize import make_featurizer
from panther_em.inference.search.hypotheses import HypothesisPlan, HypothesisSpace
from panther_em.inference.search.ledger import PixelLabel, PixelLedger
from panther_em.inference.search.orientations import S2Neighborhood
from panther_em.inference.search.pairs import PairCollector
from panther_em.inference.search.results import SearchResult, StageResult
from panther_em.inference.search.schedule import ErrorModel
from panther_em.inference.search.sparse import SparsePairEvaluator
from panther_em.inference.search.store import FeatureStore
from panther_em.inference.search.types import (
    ImageSet,
    PixelSelection,
    Precision,
    SearchGeometry,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from panther_em.inference.search.basis import Template
    from panther_em.inference.search.featurize import Featurizer, LoopOrder
    from panther_em.inference.search.context import Residency
    from panther_em.inference.search.pairs import PairList
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
    sparse_pixel_chunk, sparse_pair_chunk, sparse_cell_chunk : int
        Chunk sizes of the sparse pair evaluator: pixels featurized per chunk, pairs
        contracted per call, and kernels per inner-product GEMM. Each GEMM re-reads
        the chunk's patches (``pixel_chunk x box x 4`` bytes), so ``sparse_cell_chunk``
        sets that GEMM's arithmetic intensity; 512 cells (1 GiB of kernel rows) keep
        it compute-bound.
    sparse_patch_budget : int
        Bytes of survivor patches held per sparse chunk across all defocus planes; it
        caps ``sparse_pixel_chunk`` at ``budget / (n_planes * patch_bytes)`` so every
        plane of a chunk is featurized by the same GEMMs.
    clear_spectra_before_sparse : bool
        Drop the kernel bank's cached spectra when the dense store is released ahead
        of a sparse stage. Off by default so the next micrograph's dense stage finds
        its spectra cached; turn on when the sparse stage would otherwise not fit.
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
    sparse_pixel_chunk: int = 2048
    sparse_pair_chunk: int = 8192
    sparse_cell_chunk: int = 512
    sparse_patch_budget: int = 4 << 30
    clear_spectra_before_sparse: bool = False
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
    featurizer : Featurizer, optional
        Override the featurizer (tests inject a stub); otherwise chosen per geometry.
    planner : Planner, optional
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
        featurizer: Featurizer | None = None,
        planner: Planner | None = None,
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
        self._featurizer = featurizer
        self.ctx = SearchContext(
            template.device,
            template.weights,
            prepared_budget=self.execution.prepared_budget,
            planner=planner,
            want_pairs=any(s.emit_pairs for s in schedule.stages),
        )
        self._validate()
        # the sparse stage's GEMM operand (planar kernel rows) is shared by every
        # session: reserve it before any session buffers exist
        for spec in schedule.stages:
            if spec.mode == "sparse_pairs":
                template.kernels.reserve_planar(
                    torch.float16 if spec.precision is Precision.FP16 else torch.float32
                )

    @cached_property
    def s2(self) -> S2Neighborhood | None:
        """Orientation rings inferred from the template's Euler angles, if HEALPix."""
        return S2Neighborhood.from_space(self.space)

    def _validate(self) -> None:
        k_max, eig_max = self.template.k_max, self.template.eig_max
        for i, spec in enumerate(self.schedule.stages):
            for k0, m0, ke, me in spec.rectangles:
                if k0 + ke > k_max or m0 + me > eig_max:
                    raise ValueError(
                        f"stage {i}: rectangle {(k0, m0, ke, me)} exceeds the stored "
                        f"(k_max={k_max}, eig_max={eig_max}) feature space"
                    )
            if spec.precision is Precision.FP16 and not self.ctx.caps.is_cuda:
                raise ValueError(f"stage {i}: fp16 needs a CUDA device")
            if spec.hypotheses is not None:
                spec.hypotheses.resolve(self.space)
            if spec.pairs is not None and spec.pairs.orientation_rings > 0:
                if self.s2 is None:
                    raise ValueError(
                        f"stage {i}: orientation_rings > 0 needs a template whose "
                        "Euler angles are HEALPix pixel centers (see "
                        "healpix_orientations); orientation_rings=0 dilates pixels only"
                    )

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
        self.ctx.release()


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
        self.featurizer: Featurizer = searcher._featurizer or make_featurizer(
            self.geometry,
            searcher.template.kernels,
            tile=exe.tile,
            image_batch=exe.image_batch,
        )
        initial = PixelSelection.from_mask(self.geometry, images.mask)
        self.ledger = PixelLedger(
            self.geometry,
            exe.ledger_device or searcher.device,
            initial=initial,
            aux=self.geometry.validate_pixel_maps(images.pixel_maps),
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
        if self.searcher.ctx.caps.is_cuda:
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
            schedule = self.featurizer.schedule(self.geometry, pixels)
            store = FeatureStore(
                self.geometry,
                pixels,
                schedule.column_index,
                block_pixels=block_pixels,
                n_defocus=self.images.n_defocus,
                device=self.searcher.device,
                residency=residency,
            )
            store.copy_stream = self.searcher.ctx.copy
        else:
            keep = torch.isin(
                store.column_index, pixels.index.to(store.column_index.device)
            )
            if not bool(keep.all()) or store.block_pixels != block_pixels:
                store = store.subset(keep, block_pixels=block_pixels)
            schedule = self.featurizer.schedule(self.geometry, pixels)
            if not torch.equal(
                schedule.column_index,
                store.column_index.to(schedule.column_index.device),
            ):
                raise RuntimeError("featurizer column order diverged from the store's")
        self._mark("store:ready")
        ctx = self.searcher.ctx
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
            self.featurizer.produce(
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
        if spec.mode == "sparse_pairs":
            return self._run_sparse_stage(index, t0)
        self._mark(f"stage{index}:start")
        memory, backend, engine, hyp_plan = plan_memory(
            searcher.ctx,
            spec,
            n_pixels=len(self.pixels),
            space=searcher.space,
            subset=spec.hypotheses,
            batch=exe.batch,
            block_pixels=exe.block_pixels,
            use_cuda_graph=exe.use_cuda_graph,
            use_fused_kernel=exe.use_fused_kernel,
            store=self.store,
            pair_dtype=spec.pairs.pair_dtype if spec.pairs is not None else None,
            residency=exe.store_residency,
        )
        store = self._prepare_store(
            memory.block_pixels,
            spec.precision,
            spec.rectangles,
            hyp_plan.defocus_planes(),
            memory.store_residency,
        )
        if searcher.ctx.caps.is_cuda:
            torch.cuda.synchronize(searcher.device)
        t_featurized = time.perf_counter()
        self._mark(f"stage{index}:featurized")
        plan = plan_stage(
            searcher.ctx, index, spec, store, memory, backend, engine, hyp_plan
        )
        self._mark(f"stage{index}:planned")
        progress = None
        if exe.show_progress:
            from tqdm import tqdm  # type: ignore[import-untyped]

            bar = tqdm(
                total=store.n_pixels, desc=f"stage {index}", unit="px", unit_scale=True
            )
            progress = bar.update
        collector = None
        if spec.pairs is not None:
            collector = PairCollector(
                spec.pairs,
                self.ledger,
                store.column_index,
                hyp_plan,
                rank=spec.pairs.rank or spec.num_features,
                s2=searcher.s2,
                device=searcher.device,
            )
        maps = StageExecutor(searcher.ctx).run(
            store, plan, progress=progress, collector=collector
        )
        if exe.show_progress:
            bar.close()
        if searcher.ctx.caps.is_cuda:
            torch.cuda.synchronize(searcher.device)
        t_executed = time.perf_counter()
        self._mark(f"stage{index}:executed")
        column_index = store.column_index
        self.ledger.write(maps, column_index, index, hyp_plan)
        self._mark(f"stage{index}:ledger")
        pairs = None
        if collector is not None:
            pairs = self.ledger.pairs[index] = collector.finish()
            self._mark(f"stage{index}:pairs")
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
            best_index=hyp_plan.decode(maps.best_virtual),
            labels_after=labels_after,
            n_uncertain_after=len(self.pixels),
            elapsed_s=elapsed,
            pairs=pairs,
            featurize_s=t_featurized - t0,
            execute_s=t_executed - t_featurized,
        )

    def _run_sparse_stage(self, index: int, t0: float) -> StageResult:
        """Score the latest emitted pairs at the stage's rank; keep stage-1 moments."""
        searcher = self.searcher
        exe = searcher.execution
        spec = searcher.schedule.stages[index]
        ledger = self.ledger
        # The dense store is not read again unless a later dense stage follows; drop
        # it so the evaluator's kernels and patches get the memory (a 4k rank-1024
        # fp16 store is 52 GB).
        self._mark(f"stage{index}:start")
        if not any(s.mode == "dense" for s in searcher.schedule.stages[index + 1 :]):
            self.store = None
            if exe.clear_spectra_before_sparse:
                searcher.template.kernels.clear_spectra()
            torch.cuda.empty_cache() if searcher.ctx.caps.is_cuda else None
        self._mark(f"stage{index}:store_dropped")
        emitted = max(i for i in ledger.pairs if i < index)
        pairs = ledger.pairs[emitted].to(ledger.device)
        pairs = pairs.select(ledger.label[pairs.pixel] == int(PixelLabel.UNCERTAIN))
        planes = HypothesisPlan(
            searcher.space, spec.hypotheses, hyp_batch=1, n_psi=spec.n_psi
        ).defocus_planes()
        scores = SparsePairEvaluator(
            searcher.template,
            pixel_chunk=exe.sparse_pixel_chunk,
            pair_chunk=exe.sparse_pair_chunk,
            cell_chunk=exe.sparse_cell_chunk,
            patch_budget=exe.sparse_patch_budget,
        ).evaluate(self.images, pairs, spec, planes=planes, trace=self._mark)
        self._mark(f"stage{index}:scored")
        pixel, best = scores.best_per_pixel()
        pixel_l = pixel.to(ledger.device)
        mean, variance = ledger.mean[pixel_l], ledger.variance[pixel_l]
        mip = scores.value[best].to(ledger.device)
        maps = StageMaps(
            mip=mip,
            mean=mean,
            variance=variance,
            zscore=(mip - mean) / variance.clamp_min(0.0).sqrt(),
            best_virtual=scores.global_ids()[best].to(ledger.device),
            best_psi=scores.psi[best].to(ledger.device),
            n_evaluated=len(pairs) * spec.n_psi,
        )
        # all templates, all planes: virtual ids are global ids under this plan
        full = HypothesisPlan(searcher.space, None, hyp_batch=1, n_psi=spec.n_psi)
        ledger.write(maps, pixel_l, index, full)
        searcher.schedule.decisions[index](ledger, pixel_l, spec, index)
        self.pixels = ledger.uncertain()
        self._mark(f"stage{index}:decided")
        elapsed = time.perf_counter() - t0
        self.elapsed_s += elapsed
        return StageResult(
            stage_index=index,
            spec=spec,
            pixel_index=pixel_l.clone(),
            maps=maps,
            best_index=maps.best_virtual,
            labels_after=ledger.label[pixel_l].clone(),
            n_uncertain_after=len(self.pixels),
            elapsed_s=elapsed,
            scores=scores,
            execute_s=elapsed,  # featurization is interleaved per chunk
        )

    def set_pairs(self, stage_index: int, pairs: PairList) -> None:
        """Inject a pair list as if ``stage_index`` had emitted it."""
        self.ledger.pairs[stage_index] = pairs

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

    def featurize(self, stage_index: int | None = None) -> None:
        """Eagerly featurize for a dense stage (profiling warm-up); no search runs.

        A ``sparse_pairs`` stage featurizes per chunk while it runs: no-op here.
        """
        index = self.stage_index if stage_index is None else stage_index
        spec = self.searcher.schedule.stages[index]
        if spec.mode == "sparse_pairs":
            return
        exe = self.searcher.execution
        memory, _, _, hyp_plan = plan_memory(
            self.searcher.ctx,
            spec,
            n_pixels=len(self.pixels),
            space=self.searcher.space,
            subset=spec.hypotheses,
            batch=exe.batch,
            block_pixels=exe.block_pixels,
            use_cuda_graph=exe.use_cuda_graph,
            use_fused_kernel=exe.use_fused_kernel,
            store=self.store,
        )
        self._prepare_store(
            memory.block_pixels,
            spec.precision,
            spec.rectangles,
            hyp_plan.defocus_planes(),
            memory.store_residency,
        )
