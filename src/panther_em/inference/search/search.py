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
execution resources that amortise across image sets (kernels and spectra on the
template, prepared weights and captured hypothesis loops on the executor). A
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

    from panther_em.inference.search.features import Residency
    from panther_em.inference.search.featurize import Featurize, LoopOrder, TileSpec
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
    device_tile : int, (int, int), "fit" or None
        Tile of a device-resident store, which has no staging constraint.
        Featurizing a pixel subset costs one correlation per (cell, tile touched),
        so the per-pixel cost is set by how much of each tile the selection uses.
        ``"fit"`` (default) uses one tile the size of the selection's bounding box:
        a band of 180 rows of a 4k micrograph is then one 700 x 4096 FFT per cell
        instead of two 2560^2 ones of which 9 % is kept. A side or ``(h, w)`` pair
        tiles from the grid origin (measured on the 60S template at rank 8192: 0.49
        ns per pixel-cell at 1024, 0.22 at 2048, 0.47 at 4096 with 134 MB spectra);
        ``None`` covers each image with one tile.
    cell_chunk : int
        Kernels per featurization chunk.
    image_batch : int
        Positions per GEMM of the direct featurization engine.
    featurize_mode : {"auto", "fft", "direct"}
        Featurization engine: tiled FFT correlation or box gather + planar GEMM.
        ``"auto"`` applies :func:`~panther_em.inference.search.featurize.choose_mode`
        (its constants ``DIRECT_MAX_DENSITY`` / ``DIRECT_MAX_BOX_EXCESS`` decide).
    direct_precision : {"fp16", "tf32", "fp32"}
        Operand precision of the direct engine's GEMM, independent of the stage's
        (store) precision: fp16 operands accumulated and written in fp32 run on the
        tensor cores at 3x the fp32 CUDA-core rate with 2.4e-5 relative feature
        error; see ``featurize.DIRECT_GEMM_PRECISION``.
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
    device_tile: TileSpec | None = "fit"
    cell_chunk: int = 64
    image_batch: int = 1024
    featurize_mode: Literal["auto", "fft", "direct"] = "auto"
    direct_precision: Literal["fp16", "tf32", "fp32"] = "fp16"
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
        self.featurize: Featurize = featurize or functools.partial(
            _featurize, gemm_precision=self.execution.direct_precision
        )
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
        self.store: FeatureStore | None = None
        self._store_tile: TileSpec = 0  # tile the store's column order was built with
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
        tile: TileSpec
        if reusable:
            tile = self._store_tile  # the subset keeps the store's column order
        elif residency == "pinned_host":
            tile = exe.tile
        elif exe.device_tile is not None:
            tile = exe.device_tile
        else:
            tile = max(int(g) for g in self.geometry.grid)  # one tile per image
        schedule = schedule_tiles(
            self.geometry,
            pixels,
            tile=tile,
            image_batch=exe.image_batch,
            mode=None if exe.featurize_mode == "auto" else exe.featurize_mode,
            # a direct schedule keeps a reused store's column order (FFT tiles
            # already match through ``tile``)
            order=store.column_index if reusable and store is not None else None,
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
            keep = torch.isin(
                store.column_index, pixels.index.to(store.column_index.device)
            )
            if (
                not bool(keep.all())
                or store.block_pixels != block_pixels
                or store.residency != residency
            ):
                store = store.subset(
                    keep, block_pixels=block_pixels, residency=residency
                )
                store.copy_stream = executor.copy
            if not torch.equal(
                schedule.column_index,
                store.column_index.to(schedule.column_index.device),
            ):
                raise RuntimeError(
                    "featurization column order diverged from the store's"
                )
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
                order=(
                    exe.loop_order if store.residency == "pinned_host" else "cell_major"
                ),
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
        if stage.decision is not None:
            stage.decision.apply(
                self.ledger, column_index, hyps, self.images.spectra
            )
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
            spectra=self.images.spectra,
        )
