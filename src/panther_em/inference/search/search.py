"""The search API: ``ExecutionConfig``, ``Searcher`` and ``SearchSession`` (``Stage``
lives in :mod:`.stage`).

::

    template = Template.from_hdf5("decomp.h5", device="cuda:0")  # under W_avg
    coarse, full = [(0, 0, 32, 32)], [(0, 0, 64, 128)]
    tables = {r[0]: RadialTables.load(path) for r, path in ...}  # .build once
    calibration = Calibration(alpha=5e-3, zeta=0.95, eps_rho=0.1, b_factor=40.0)
    stages = [
        Stage.create(
            coarse,
            precision="fp16",
            orientations=1,
            planes=(5,),
            decision=Decision(
                7.8, ThresholdTest(tables[coarse[0]], calibration), accept=False
            ),
        ),
        Stage.create(
            full, decision=Decision(7.8, ThresholdTest(tables[full[0]], calibration))
        ),
    ]
    searcher = Searcher(template, stages, n_defocus=11)
    images = ImageSet.single(micrograph, filters=ctf_planes)
    images = images.with_spectra(
        estimate_spectra(
            images, tables[coarse[0]], correction=whitening.correction_1d(name)
        )
    )
    result = searcher.run(images)
    result.save("search.h5")

A :class:`Searcher` is bound to one template and one list of stages and owns the
execution resources that amortise across image sets (one featurization per
rectangle and engine, holding its kernels or adjoint bases; prepared weights and
captured hypothesis loops on the executor). A
:class:`SearchSession` runs the stages over one :class:`ImageSet`: stage 0 searches
every unmasked pixel, each later stage re-scores the pixels the previous decision
left AMBIGUOUS or ACCEPTED, and the session stops early when that set is empty. A
stage whose decision runs the threshold test needs the image set's spectra
(``ImageSet.spectra``); the session refuses to start without them.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from tqdm import tqdm

from panther_em.inference.search.decision import PixelLabel
from panther_em.inference.search.executor import (
    Executor,
    StageMaps,
    feature_bytes_per_pixel,
    merge_stage_maps,
    plan_memory,
    plan_stage,
)
from panther_em.inference.search.featurization import (
    FeatureStore,
    choose_mode,
    make_featurization,
)
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
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from panther_em.inference.search.executor import MemoryPlan
    from panther_em.inference.search.featurization import (
        BaseImageFeaturization,
        Mode,
    )
    from panther_em.inference.search.featurization.direct_featurization import (
        ContractionName,
    )
    from panther_em.inference.search.types import BatchShape, Rectangle
    from panther_em.inference.template import Template, TemplateWarp

    FeaturizationFactory = Callable[
        [TemplateWarp, Rectangle, Mode], BaseImageFeaturization
    ]


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
    featurize_mode : {"auto", "fft", "direct"}
        Featurization engine: overlap-save FFT correlation of the selection's bounding
        box, or box gather + contraction per position. ``"auto"`` applies
        :func:`~panther_em.inference.search.featurization.choose_mode` to each
        stage's selection.
    cell_chunk : int
        Kernel spectra per chunk of the FFT engine.
    image_batch : int
        Positions per chunk of the direct engine.
    direct_contraction : {"adjoint", "planar"}
        How the direct engine turns boxes into features: ``"adjoint"`` scatters each
        box onto the polar grid through the transposed warp (no Cartesian kernels,
        2-3.5x the planar GEMM's speed; falls back to ``"planar"`` where the
        transform is not mirrored), ``"planar"`` the ``[Re K; -Im K]`` kernel GEMM.
    direct_precision : {"fp16", "tf32", "fp32"}
        Operand precision of the direct engine, independent of the stage's (store)
        precision: fp16 is 5e-4 relative on the adjoint's features, fp32 exact to
        1e-6.
    ledger_device : device, optional
        Where the dense per-pixel state lives; defaults to the compute device.
    prepared_budget : int, optional
        Byte budget of the prepared-weight cache.
    show_progress : bool
        Print per-stage progress bars.
    l2_target : int, optional
        Bytes of spectrum per contraction to keep L2-resident (the hypothesis batch
        follows from it); defaults to a quarter of the device's L2.
    block_bytes_max : int
        Byte budget of one feature-store block (256 MiB).
    device_store_fraction : float
        A stage whose store would exceed this fraction of the free device memory runs
        in *bands*: row-contiguous chunks of its pixels whose store fits, each
        featurized (one band-shaped FFT per cell) and reduced in turn; the maps are
        concatenated before the decision.
    plane_major : bool
        Search a multi-plane stage one defocus plane at a time: each pass treats the
        plane as its own image, featurizing and holding only that plane's features
        (a third of a three-plane store), and the per-pixel maxima, means and
        variances are merged across passes outside the kernel. The captured loop and
        prepared weights are shared by the passes. Off, every plane is resident at
        once and the kernel sweeps them in one loop.
    trace : bool
        Synchronise the device at every session mark (``SearchSession.marks``) so
        the marks' wall-clock deltas are per-phase device times.
    """

    batch: BatchShape | None = None
    block_pixels: int | None = None
    use_cuda_graph: bool = True
    use_fused_kernel: bool = True
    featurize_mode: Literal["auto", "fft", "direct"] = "auto"
    cell_chunk: int = 32
    image_batch: int = 1024
    direct_contraction: ContractionName = "adjoint"
    direct_precision: Literal["fp16", "tf32", "fp32"] = "fp16"
    ledger_device: torch.device | str | None = None
    prepared_budget: int | None = None
    show_progress: bool = False
    l2_target: int | None = None
    block_bytes_max: int = 256 << 20
    device_store_fraction: float = 0.5
    plane_major: bool = True
    trace: bool = False


def _concat_stage_maps(parts: Sequence[StageMaps]) -> StageMaps:
    """Maps of disjoint pixel bands, concatenated in band order."""
    return StageMaps(
        mip=torch.cat([p.mip for p in parts]),
        mean=torch.cat([p.mean for p in parts]),
        variance=torch.cat([p.variance for p in parts]),
        zscore=torch.cat([p.zscore for p in parts]),
        best_virtual=torch.cat([p.best_virtual for p in parts]),
        best_psi=torch.cat([p.best_psi for p in parts]),
        n_evaluated=parts[0].n_evaluated,
    )


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
    featurization : callable, optional
        ``(warp, rect, mode) -> BaseImageFeaturization``; defaults to
        :func:`~.featurization.make_featurization` configured by ``execution`` (tests
        inject a stub).
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
        featurization: FeaturizationFactory | None = None,
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
        self.make_featurization: FeaturizationFactory = (
            featurization
            or functools.partial(
                make_featurization,
                cell_chunk=self.execution.cell_chunk,
                batch=self.execution.image_batch,
                precision=self.execution.direct_precision,
                contraction=self.execution.direct_contraction,
            )
        )
        self._featurizations: dict[tuple[Rectangle, Mode], BaseImageFeaturization] = {}
        # NESTED HEALPix ids of the template's orientations, when a stage samples
        # orientations by HEALPix level (None otherwise, or when not HEALPix)
        self.ipix = None
        if any(isinstance(s.orientations, int) for s in self.stages):
            found = healpix_ipix(self.space)
            self.ipix = None if found is None else found[1]
        self.executor = Executor(
            template.device,
            template.weights,
            prepared_budget=self.execution.prepared_budget,
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
            if (
                defocus_ids is not None
                and int(defocus_ids.max()) >= self.space.n_defocus
            ):
                raise ValueError(
                    f"stage {i}: planes {stage.planes} exceed the {self.space.n_defocus} "
                    "defocus plane(s) of the searcher"
                )
            if (
                template_ids is not None
                and int(template_ids.max()) >= self.space.n_templates
            ):
                raise ValueError(f"stage {i}: orientation ids exceed the template's")
            if stage.decision is not None and stage.decision.test is not None:
                tables = stage.decision.test.tables
                if tables.n_templates != self.space.n_templates:
                    raise ValueError(
                        f"stage {i}: the threshold test's tables hold "
                        f"{tables.n_templates} template rows; the template has "
                        f"{self.space.n_templates}"
                    )
                if tables.template_shape != self.template.template_shape:
                    raise ValueError(
                        f"stage {i}: the threshold test's tables were built on a "
                        f"{tables.template_shape} box; the template's is "
                        f"{self.template.template_shape}"
                    )
                if sorted(tables.rectangles) != sorted(stage.rectangles):
                    raise ValueError(
                        f"stage {i}: the threshold test's tables were built for "
                        f"rectangles {tables.rectangles}, not the stage's "
                        f"{stage.rectangles}"
                    )

    @property
    def device(self) -> torch.device:
        """Compute device."""
        return self.template.device

    def featurization(self, rect: Rectangle, mode: Mode) -> BaseImageFeaturization:
        """The ``mode`` featurization of ``rect`` (built once, reused across images)."""
        key = (rect, mode)
        found = self._featurizations.get(key)
        if found is None:
            found = self.make_featurization(self.template.warp, rect, mode)
            self._featurizations[key] = found
        return found

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
        if images.spectra is None:
            for i, stage in enumerate(searcher.stages):
                if stage.decision is not None and stage.decision.test is not None:
                    raise ValueError(
                        f"stage {i}: its decision runs the threshold test, which "
                        "needs the image set's spectra (estimate_spectra, then "
                        "ImageSet.with_spectra)"
                    )
        self.searcher = searcher
        self.images = images
        self.geometry = SearchGeometry.from_images(
            images, searcher.template.template_shape
        )
        exe = searcher.execution
        initial = PixelSelection.from_mask(self.geometry, images.mask)
        self.ledger = PixelLedger(
            self.geometry, exe.ledger_device or searcher.device, initial=initial
        )
        self.pixels: PixelSelection = initial
        self.store: FeatureStore | None = None  # the last pass's features
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

    def _featurize(
        self,
        stage: Stage,
        pixels: PixelSelection,
        planes: Sequence[int],
        block_pixels: int,
        mode: Mode,
    ) -> FeatureStore:
        """A store of ``pixels`` (compact columns in selection order) for ``planes``."""
        searcher = self.searcher
        self.store = None  # the previous pass's features are not read again
        store = FeatureStore(
            stage.rectangles,
            stage.precision,
            planes,
            len(pixels),
            block_pixels,
            searcher.device,
        )
        self._mark("store:allocated")
        image, rows, cols = self.geometry.positions(pixels.index.to("cpu"))
        # the selection is sorted, so each image's positions are one run of columns
        ids, counts = torch.unique_consecutive(image, return_counts=True)
        start = 0
        for b, count in zip(ids.tolist(), counts.tolist(), strict=True):
            columns = slice(start, start + count)
            filtered = self.images.filtered_planes(b, planes, searcher.device)
            for rect in stage.rectangles:
                featurization = searcher.featurization(rect, mode)
                for chunk in featurization.chunks(
                    filtered, rows[columns], cols[columns]
                ):
                    store.write(rect, chunk, column_offset=start)
            start += count
        store.check_complete()
        self.store = store
        return store

    def _bands(
        self, stage: Stage, pass_planes: Sequence[int], gemm_pixels: int
    ) -> list[PixelSelection]:
        """Row-contiguous chunks of the stage's pixels whose store fits the device.

        One band (the whole selection) unless a pass's store on a CUDA device would
        exceed ``device_store_fraction`` of the free memory; then the sorted selection
        is cut into equal chunks of a multiple of ``gemm_pixels`` pixels that fit, each
        a band of rows the FFT engine covers with one region.
        """
        exe = self.searcher.execution
        executor = self.searcher.executor
        pixels = self.pixels
        if not executor.is_cuda or len(pixels) == 0 or not executor.free_bytes:
            return [pixels]
        per_pixel = feature_bytes_per_pixel(
            (rect, stage.precision, len(pass_planes)) for rect in stage.rectangles
        )
        limit = int(executor.free_bytes * exe.device_store_fraction)
        if per_pixel * len(pixels) <= limit:
            return [pixels]
        fit = max(
            gemm_pixels, (limit // max(1, per_pixel)) // gemm_pixels * gemm_pixels
        )
        n_bands = -(-len(pixels) // fit)
        per_band = -(-len(pixels) // n_bands)
        per_band = -(-per_band // gemm_pixels) * gemm_pixels
        index = pixels.index
        if not bool(torch.all(index[1:] >= index[:-1])):
            index = torch.sort(index).values
        return [
            PixelSelection(index[i : i + per_band], self.geometry)
            for i in range(0, int(index.numel()), per_band)
        ]

    def _run_stage(self, index: int) -> StageResult:
        searcher = self.searcher
        exe = searcher.execution
        stage = searcher.stages[index]
        t0 = time.perf_counter()
        self._mark(f"stage{index}:start")
        template_ids, defocus_ids = stage.hypothesis_ids(searcher.space, searcher.ipix)

        searcher.executor.refresh_free_bytes()

        def memory_for(
            planes: torch.Tensor | None, n_pixels: int | None = None
        ) -> MemoryPlan:
            return plan_memory(
                searcher.executor,
                stage,
                n_pixels=len(self.pixels) if n_pixels is None else n_pixels,
                space=searcher.space,
                template_ids=template_ids,
                defocus_ids=planes,
                batch=exe.batch,
                block_pixels=exe.block_pixels,
                use_cuda_graph=exe.use_cuda_graph,
                use_fused_kernel=exe.use_fused_kernel,
                l2_target=exe.l2_target,
                block_bytes_max=exe.block_bytes_max,
            )

        memory = memory_for(defocus_ids)
        hyps = memory.hyps
        planes = hyps.defocus_planes()
        # One pass per plane (the plane's features alone are resident), or one pass
        # over every plane.
        passes: list[tuple[int, ...]] = (
            [(p,) for p in planes] if exe.plane_major and len(planes) > 1 else [planes]
        )
        mode: Mode = (
            choose_mode(self.geometry, len(self.pixels))
            if exe.featurize_mode == "auto"
            else exe.featurize_mode
        )
        bands = self._bands(stage, passes[0], memory.gemm_pixels)
        progress = None
        if exe.show_progress:
            bar = tqdm(
                total=len(self.pixels) * len(passes),
                desc=f"stage {index}",
                unit="px",
                unit_scale=True,
            )
            progress = bar.update
        featurize_s = 0.0
        execute_s = 0.0
        band_maps: list[StageMaps] = []
        band_columns: list[torch.Tensor] = []
        for band in bands:
            parts: list[StageMaps] = []
            for pass_planes in passes:
                t_pass = time.perf_counter()
                pass_memory = (
                    memory
                    if len(passes) == 1 and len(bands) == 1
                    else memory_for(
                        torch.tensor(pass_planes, dtype=torch.long), len(band)
                    )
                )
                store = self._featurize(
                    stage, band, pass_planes, pass_memory.block_pixels, mode
                )
                if searcher.executor.is_cuda:
                    torch.cuda.synchronize(searcher.device)
                t_featurized = time.perf_counter()
                self._mark(f"stage{index}:featurized")
                plan = plan_stage(searcher.executor, index, stage, store, pass_memory)
                self._mark(f"stage{index}:planned")
                parts.append(searcher.executor.run(store, plan, progress=progress))
                if searcher.executor.is_cuda:
                    torch.cuda.synchronize(searcher.device)
                t_executed = time.perf_counter()
                self._mark(f"stage{index}:executed")
                featurize_s += t_featurized - t_pass
                execute_s += t_executed - t_featurized
            band_maps.append(merge_stage_maps(parts, hyps.n_selected))
            band_columns.append(band.index.to("cpu"))
        maps = band_maps[0] if len(band_maps) == 1 else _concat_stage_maps(band_maps)
        column_index = (
            band_columns[0] if len(band_columns) == 1 else torch.cat(band_columns)
        )
        self.ledger.write(maps, column_index, index, hyps)
        self._mark(f"stage{index}:ledger")
        if stage.decision is not None:
            stage.decision.apply(self.ledger, column_index, hyps, self.images.spectra)
        self.pixels = self.ledger.next_stage_pixels()
        self._mark(f"stage{index}:decided")
        ledger_index = column_index.to(self.ledger.device)
        labels_after = self.ledger.label[ledger_index].clone()
        score = self.ledger.score[ledger_index].clone()
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
            score=score,
            n_ambiguous_after=n_ambiguous,
            elapsed_s=elapsed,
            featurize_s=featurize_s,
            execute_s=execute_s,
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
            spectra=self.images.spectra,
        )
