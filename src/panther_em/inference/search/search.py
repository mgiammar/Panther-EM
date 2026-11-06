"""The search driver: ``Searcher`` and ``SearchSession``.

``Stage`` lives in :mod:`.stage`, ``ExecutionConfig`` in :mod:`.config`; a stage is
planned by :mod:`.planning`, run by :mod:`.runner` (featurization, then
:mod:`.correlation`) and decided by :mod:`.decision`.

::

    # once per dataset and template
    whitening = WhiteningFilter.from_images(micrographs)
    template = Template.from_hdf5("decomp.h5", device="cuda:0")  # under W_avg * band
    coarse, full = [(0, 0, 32, 32)], [(0, 0, 64, 128)]
    tables = {
        r[0]: PerSpatialFrequencyErrorTable.load(path) for r, path in ...
    }  # compute_error_tables once
    calibration = Calibration(alpha=5e-3, zeta=0.95, eps_rho=0.1, b_factor=40.0)
    stages = [
        Stage.create(
            coarse,
            precision="fp16",
            decision=Decision(
                7.8, ThresholdTest(tables[coarse[0]], calibration), accept=False
            ),
        ),
        Stage.create(
            full, decision=Decision(7.8, ThresholdTest(tables[full[0]], calibration))
        ),
    ]
    searcher = Searcher(template, stages, n_defocus=11)

    # per micrograph
    images = ImageSet.from_micrograph(
        micrograph, name, whitening, ctf_planes, tables[coarse[0]], pixel_size=1.06
    )
    result = searcher.run(images)
    result.save("search.h5")

A :class:`Searcher` is bound to one template and one list of stages and owns the
execution resources that amortise across image sets (one featurization per
rectangle and engine, holding its kernels or adjoint bases; prepared weights and
captured hypothesis loops on the correlator); it checks every stage's error table
against the template (warp, box, rows, rectangles, decomposition filter). A
:class:`SearchSession` runs the stages over one micrograph
(:class:`~.noise_model.ImageSet`) or one plain :class:`~.types.SearchImages` stack:
stage 0 searches every unmasked pixel, each later stage re-scores the pixels the
previous decision left AMBIGUOUS or ACCEPTED, and the session stops early when that
set is empty. A *local* stage (``Stage.local_rings``) scores each of those pixels only
over a HEALPix neighborhood of the orientation the previous stage found best there,
with the mean and variance of its whole set from the closed-form moments. A stage
whose decision runs the threshold test needs the micrograph's noise model (an
``ImageSet`` on the table's frequency axis); the session refuses to start without it.
"""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING

import torch

from panther_em.inference.search.config import ExecutionConfig
from panther_em.inference.search.correlation import Correlator
from panther_em.inference.search.decision import PixelLabel
from panther_em.inference.search.featurization import make_featurization
from panther_em.inference.search.hypotheses import (
    HypothesisSpace,
    LocalHypotheses,
    healpix_ipix,
)
from panther_em.inference.search.noise_model import ImageSet
from panther_em.inference.search.planning import DeviceInfo, plan_stage
from panther_em.inference.search.results import PixelLedger, SearchResult, StageResult
from panther_em.inference.search.runner import StageRunner, Timeline
from panther_em.inference.search.types import (
    PixelSelection,
    Precision,
    SearchGeometry,
    SearchImages,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from panther_em.inference.search.featurization import (
        BaseImageFeaturization,
        FeatureStore,
        Mode,
    )
    from panther_em.inference.search.planning import StagePlan
    from panther_em.inference.search.runner import StageRun
    from panther_em.inference.search.stage import Stage
    from panther_em.inference.search.types import Rectangle
    from panther_em.inference.template import Template, TemplateWarp

    FeaturizationFactory = Callable[
        [TemplateWarp, Rectangle, Mode], BaseImageFeaturization
    ]


class Searcher:
    """A template plus its stages, ready to search image sets.

    Parameters
    ----------
    template : Template
        The SVD template (kernel warp and weights).
    stages : sequence of Stage
        Stages in execution order (typically increasing rank).
    execution : ExecutionConfig, optional
        How the stages run.
    n_defocus : int, optional
        Defocus planes every image set must carry (CTF planes). Defaults to 1.
    defocus_values : torch.Tensor, optional
        ``(n_defocus,)`` labels for the planes, reported by ``SearchResult.decode()``.
    keep_stage_results : bool, optional
        Retain every stage's compact maps on the final :class:`SearchResult` (and in
        its HDF5 file). Defaults to False.
    featurization : callable, optional
        ``(warp, rect, mode) -> BaseImageFeaturization``; defaults to
        :func:`~.featurization.make_featurization` configured by ``execution`` (tests
        inject a stub).

    Attributes
    ----------
    space : HypothesisSpace
        Every hypothesis of the template and the defocus planes.
    nside, ipix : int and numpy.ndarray, or None
        The template's HEALPix grid and its orientations' NESTED pixel ids, when a
        stage samples orientations by level or searches neighborhoods.
    local : dict[int, LocalHypotheses]
        The neighborhoods of each local stage, by stage index.
    device_info : DeviceInfo
        The compute device's capabilities (probed once).
    correlator : Correlator
        Prepared weights and built loops, reused across stages and image sets.
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
        tuning = self.execution.tuning
        self.keep_stage_results = bool(keep_stage_results)
        self.space = HypothesisSpace.from_result(
            template.result, n_defocus=n_defocus, defocus_values=defocus_values
        )
        self.make_featurization: FeaturizationFactory = (
            featurization
            or functools.partial(
                make_featurization,
                cell_chunk=tuning.cell_chunk,
                batch=tuning.image_batch,
                precision=self.execution.direct_precision,
                contraction=self.execution.direct_contraction,
            )
        )
        self._featurizations: dict[tuple[Rectangle, Mode], BaseImageFeaturization] = {}
        # NESTED HEALPix ids of the template's orientations, when a stage samples
        # orientations by HEALPix level or is local (None otherwise, or when the
        # template is not on a HEALPix grid)
        self.nside: int | None = None
        self.ipix = None
        if any(isinstance(s.orientations, int) or s.is_local for s in self.stages):
            found = healpix_ipix(self.space)
            if found is not None:
                self.nside, self.ipix = found
        self.local: dict[int, LocalHypotheses] = {}
        self.device_info = DeviceInfo.probe(template.device)
        self.correlator = Correlator(
            self.device_info.device,
            template.weights,
            prepared_budget=tuning.prepared_budget,
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
            if stage.precision is Precision.FP16 and not self.device_info.is_cuda:
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
            if stage.is_local:
                self.local[i] = self._local_hypotheses(i, template_ids)
            if stage.decision is not None and stage.decision.test is not None:
                table = stage.decision.test.table
                try:
                    table.check_template(self.template)
                except ValueError as exc:
                    raise ValueError(f"stage {i}: {exc}") from exc
                if sorted(table.rectangles) != sorted(stage.rectangles):
                    raise ValueError(
                        f"stage {i}: the threshold test's table was built for "
                        f"rectangles {table.rectangles}, not the stage's "
                        f"{stage.rectangles}"
                    )

    def _local_hypotheses(
        self, index: int, template_ids: torch.Tensor | None
    ) -> LocalHypotheses:
        stage = self.stages[index]
        assert stage.local_rings is not None
        if index == 0:
            raise ValueError(
                "stage 0 cannot be local: a local stage searches around the "
                "orientation the previous stage found best"
            )
        if self.ipix is None or self.nside is None:
            raise ValueError(
                f"stage {index}: a local stage needs a template whose orientations "
                "are HEALPix pixel centers (see healpix_orientations)"
            )
        try:
            return LocalHypotheses.build(
                self.space,
                self.nside,
                self.ipix,
                level=stage.level,
                seed_level=self.stages[index - 1].level,
                rings=stage.local_rings,
                template_ids=template_ids,
            )
        except ValueError as exc:
            raise ValueError(f"stage {index}: {exc}") from exc

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

    def plan(
        self,
        index: int,
        pixels: PixelSelection,
        geometry: SearchGeometry,
        seeds: torch.Tensor | None = None,
    ) -> StagePlan:
        """Plan stage ``index`` over ``pixels`` against the device's free memory now.

        A local stage needs ``seeds``: each pixel's previous best orientation.
        """
        return plan_stage(
            index,
            self.stages[index],
            pixels,
            geometry,
            space=self.space,
            device=self.device_info,
            execution=self.execution,
            ipix=self.ipix,
            free_bytes=self.device_info.free_bytes(),
            local=self.local.get(index),
            seeds=seeds,
        )

    def session(self, images: ImageSet | SearchImages) -> SearchSession:
        """A session over one image set (iterate it for per-stage results)."""
        return SearchSession(self, images)

    def run(self, images: ImageSet | SearchImages) -> SearchResult:
        """Run every stage over one micrograph (or a plain image stack)."""
        return self.session(images).run_all()

    def run_many(
        self, image_sets: Iterable[ImageSet | SearchImages]
    ) -> Iterator[SearchResult]:
        """Run the stages over each image set in turn, reusing cached resources."""
        for images in image_sets:
            yield self.run(images)

    def release(self) -> None:
        """Drop built loops and prepared weights (kernels stay with the template)."""
        self.correlator.release()


class SearchSession:
    """One image set's walk through the stages.

    Each stage: plan it over the current pixels (:meth:`Searcher.plan`), run the plan
    (:class:`~.runner.StageRunner`: featurize, correlate, merge), write the maps to
    the ledger, let the stage's decision label the pixels, and take the pixels the
    next stage searches from the ledger.

    Parameters
    ----------
    searcher : Searcher
        The template, stages and shared resources.
    images : ImageSet or SearchImages
        One micrograph with its noise model (searched as its whitened image under its
        CTF planes), or a plain stack for stages without a threshold test.

    Attributes
    ----------
    images : SearchImages
        What the stages search.
    noise : ImageSet or None
        The micrograph's noise model, read by the threshold tests.
    ledger : PixelLedger
        The dense per-pixel state.
    pixels : PixelSelection
        The pixels the next stage searches.
    runner : StageRunner
        Executes the stage plans over the images.
    stage_results : list[StageResult]
        Filled when the searcher keeps stage results.
    timeline : Timeline
        ``(name, time, allocated_bytes)`` marks of every phase (:attr:`marks`).
    """

    def __init__(self, searcher: Searcher, images: ImageSet | SearchImages) -> None:
        noise: ImageSet | None = None
        if isinstance(images, ImageSet):
            noise = images
            images = SearchImages.single(noise.image, filters=noise.ctf)
        if images.n_defocus != searcher.space.n_defocus:
            raise ValueError(
                f"image set carries {images.n_defocus} defocus plane(s); the searcher "
                f"expects {searcher.space.n_defocus}"
            )
        for i, stage in enumerate(searcher.stages):
            if stage.decision is None or stage.decision.test is None:
                continue
            if noise is None:
                raise ValueError(
                    f"stage {i}: its decision runs the threshold test, which needs "
                    "the micrograph's noise model (search a noise_model.ImageSet)"
                )
            if not torch.equal(
                noise.frequencies, stage.decision.test.table.frequencies
            ):
                raise ValueError(
                    f"stage {i}: the micrograph's curves are on another frequency "
                    "axis than the threshold test's table"
                )
        self.searcher = searcher
        self.images = images
        self.noise = noise
        self.geometry = SearchGeometry.from_images(
            images, searcher.template.template_shape
        )
        exe = searcher.execution
        initial = PixelSelection.from_mask(self.geometry, images.mask)
        self.ledger = PixelLedger(
            self.geometry, exe.tuning.ledger_device or searcher.device, initial=initial
        )
        self.pixels: PixelSelection = initial
        self.stage_index = 0
        self.stage_results: list[StageResult] = []
        self.elapsed_s = 0.0
        self._finished = False
        # the runner shares the timeline, not the session: no session <-> runner
        # cycle, so a finished session (and its last store) is freed at once
        self.timeline = Timeline(searcher.device_info, trace=exe.trace)
        self.runner = StageRunner(
            images,
            self.geometry,
            featurization=searcher.featurization,
            correlator=searcher.correlator,
            device=searcher.device_info,
            timeline=self.timeline,
            show_progress=exe.show_progress,
            local_bytes=exe.tuning.local_bytes,
        )
        self._mark("session")

    @property
    def store(self) -> FeatureStore | None:
        """The last pass's features (kept until the next stage starts)."""
        return self.runner.store

    @property
    def marks(self) -> list[tuple[str, float, int]]:
        """``(name, time, allocated_bytes)`` marks of every phase so far."""
        return self.timeline.marks

    def _mark(self, name: str) -> None:
        self.timeline.mark(name)

    def _run_stage(self, index: int) -> StageResult:
        """Plan, run, record and decide stage ``index``."""
        t0 = time.perf_counter()
        self._mark(f"stage{index}:start")
        self.runner.release()  # the previous stage's features: not counted as in use
        seeds = (
            self._seed_orientations(self.pixels)
            if self.searcher.stages[index].is_local
            else None
        )
        plan = self.searcher.plan(index, self.pixels, self.geometry, seeds)
        run = self.runner.run(plan)
        pixel_index = plan.pixel_index
        self.ledger.write(run.maps, pixel_index, index, plan.hyps)
        self._mark(f"stage{index}:ledger")
        decision = plan.stage.decision
        if decision is not None:
            decision.apply(self.ledger, pixel_index, plan.hyps, self.noise)
        self.pixels = self.ledger.next_stage_pixels()
        self._mark(f"stage{index}:decided")
        return self._stage_result(plan, run, pixel_index, time.perf_counter() - t0)

    def _seed_orientations(self, pixels: PixelSelection) -> torch.Tensor:
        """``(P,)`` best orientation the previous stage found at each pixel."""
        best = self.ledger.best_index[pixels.index.to(self.ledger.device)]
        if bool((best < 0).any()):
            raise RuntimeError(
                "a local stage needs the previous stage's best hypothesis at every "
                "pixel it searches"
            )
        return self.searcher.space.decode(best.cpu())[2]

    def _stage_result(
        self, plan: StagePlan, run: StageRun, pixel_index: torch.Tensor, elapsed: float
    ) -> StageResult:
        """The stage's maps with the labels and scores its decision left."""
        ledger_index = pixel_index.to(self.ledger.device)
        self.elapsed_s += elapsed
        return StageResult(
            stage_index=plan.index,
            stage=plan.stage,
            pixel_index=pixel_index.clone(),
            maps=run.maps,
            best_index=plan.hyps.decode(run.maps.best_virtual),
            labels_after=self.ledger.label[ledger_index].clone(),
            score=self.ledger.score[ledger_index].clone(),
            n_ambiguous_after=int(
                self.ledger.pixels_with(PixelLabel.AMBIGUOUS).numel()
            ),
            elapsed_s=elapsed,
            featurize_s=run.featurize_s,
            execute_s=run.execute_s,
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
