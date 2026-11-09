"""Running a stage plan: featurize each pass into a store, correlate it, merge the maps.

:class:`StageRunner` executes a :class:`~.planning.StagePlan` over one image set:
for every band and every pass of the band it fills a fresh
:class:`~.featurization.FeatureStore` with the pass's features (the plan's engine),
correlates the store against the pass's hypotheses
(:class:`~.correlation.Correlator`; a local stage scores each pixel over its group's
neighborhood, :meth:`~.correlation.Correlator.run_local`; a seeded stage first runs
the seed pass over the store's seeded columns, :mod:`~.seeding`), then merges the
passes of a band (:func:`~.correlation.merge_stage_maps`) and concatenates the bands
(:func:`~.correlation.concat_stage_maps`). One store is alive at a time: a pass's
store is dropped before the next one is allocated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from panther_em.search.correlation import (
    LocalWork,
    LoopPlan,
    concat_stage_maps,
    merge_stage_maps,
)
from panther_em.search.featurization import ColumnSubset, FeatureStore
from panther_em.search.seeding import seed_work

if TYPE_CHECKING:
    from collections.abc import Callable

    from panther_em.decomposition.rectangles import Rectangle
    from panther_em.search.correlation import Correlator, StageMaps
    from panther_em.search.featurization import (
        BaseImageFeaturization,
        Mode,
    )
    from panther_em.search.images import (
        PixelSelection,
        SearchGeometry,
        SearchImages,
    )
    from panther_em.search.planning import (
        BandPlan,
        DeviceInfo,
        PassPlan,
        StagePlan,
    )


class Timeline:
    """``(name, time, allocated_bytes)`` marks at every phase boundary of a session.

    Parameters
    ----------
    device : DeviceInfo
        The compute device (its allocated bytes are recorded on CUDA).
    trace : bool, optional
        Synchronise the device at every mark, so the deltas are device times.
    """

    def __init__(self, device: DeviceInfo, *, trace: bool = False) -> None:
        self.device = device
        self.trace = bool(trace)
        self.marks: list[tuple[str, float, int]] = []

    def mark(self, name: str) -> None:
        """Record a mark now."""
        allocated = 0
        if self.device.is_cuda:
            if self.trace:
                torch.cuda.synchronize(self.device.device)
            allocated = torch.cuda.memory_allocated(self.device.device)
        self.marks.append((name, time.perf_counter(), allocated))


@dataclass(frozen=True, eq=False)
class StageRun:
    """A stage's maps (in the plan's pixel order) and where the time went."""

    maps: StageMaps
    featurize_s: float
    execute_s: float


class StageRunner:
    """Executes stage plans over one image set.

    Parameters
    ----------
    images : SearchImages
        The image set the stages search.
    geometry : SearchGeometry
        Its grid.
    featurization : callable
        ``(rect, mode) -> BaseImageFeaturization``: the searcher's (cached)
        featurizations.
    correlator : Correlator
        The searcher's correlation resources.
    device : DeviceInfo
        The compute device.
    timeline : Timeline, optional
        Records a mark at every phase boundary (the session's).
    show_progress : bool, optional
        Print a progress bar per stage.
    local_bytes : int, optional
        Working-memory budget of one chunk of a local stage's tiles.
    local_weights_bytes : int, optional
        Byte budget of a local pass's prepared fp16 weights (``None`` = unbounded).

    Attributes
    ----------
    store : FeatureStore or None
        The last pass's features (dropped before the next pass allocates, and by
        :meth:`release`).
    """

    def __init__(
        self,
        images: SearchImages,
        geometry: SearchGeometry,
        *,
        featurization: Callable[[Rectangle, Mode], BaseImageFeaturization],
        correlator: Correlator,
        device: DeviceInfo,
        timeline: Timeline | None = None,
        show_progress: bool = False,
        local_bytes: int = 512 * 1024**2,
        local_weights_bytes: int | None = 2048 * 1024**2,
    ) -> None:
        self.images = images
        self.geometry = geometry
        self.featurization = featurization
        self.correlator = correlator
        self.device = device
        self.timeline = timeline or Timeline(device)
        self.show_progress = show_progress
        self.local_bytes = int(local_bytes)
        self.local_weights_bytes = (
            None if local_weights_bytes is None else int(local_weights_bytes)
        )
        self.store: FeatureStore | None = None

    def release(self) -> None:
        """Drop the last pass's features."""
        self.store = None

    def run(self, plan: StagePlan) -> StageRun:
        """Featurize and correlate every band and pass of ``plan``."""
        progress = None
        if self.show_progress:
            total = sum(len(b.pixels) * len(b.passes) for b in plan.bands)
            bar = tqdm(
                total=total, desc=f"stage {plan.index}", unit="px", unit_scale=True
            )
            progress = bar.update
        timing = [0.0, 0.0]
        band_maps = []
        for band in plan.bands:
            parts = [
                self._run_pass(plan, band, step, timing, progress)
                for step in band.passes
            ]
            band_maps.append(merge_stage_maps(parts, plan.hyps.n_selected))
        return StageRun(concat_stage_maps(band_maps), timing[0], timing[1])

    def _run_pass(
        self,
        plan: StagePlan,
        band: BandPlan,
        step: PassPlan,
        timing: list[float],
        progress: Callable[[int], None] | None,
    ) -> StageMaps:
        """One pass: its store, then its correlation (locals die with the call)."""
        t0 = time.perf_counter()
        store = self._featurize(plan, band.pixels, step)
        self._synchronize()
        t1 = time.perf_counter()
        self.timeline.mark(f"stage{plan.index}:featurized")
        loop_plan = LoopPlan.create(plan.stage, store, step.memory)
        self.timeline.mark(f"stage{plan.index}:planned")
        if plan.seeding is not None:
            maps = self._run_seeded(plan, band, step, store, loop_plan)
            if progress is not None:
                progress(store.n_pixels)
        elif plan.local is None:
            maps = self.correlator.run(store, loop_plan, progress=progress)
        else:
            assert band.cells is not None
            maps = self.correlator.run_local(
                store,
                loop_plan,
                LocalWork(band.cells, plan.local.rows),
                chunk_bytes=self.local_bytes,
                progress=progress,
            )
        self._synchronize()
        t2 = time.perf_counter()
        self.timeline.mark(f"stage{plan.index}:executed")
        timing[0] += t1 - t0
        timing[1] += t2 - t1
        return maps

    def _run_seeded(
        self,
        plan: StagePlan,
        band: BandPlan,
        step: PassPlan,
        store: FeatureStore,
        loop_plan: LoopPlan,
    ) -> StageMaps:
        """A seeded pass: the seed pass over the band's lattice, then the local pass."""
        assert plan.seeding is not None
        assert band.lattice is not None and step.seed_memory is not None
        resources, lattice = plan.seeding.resources, band.lattice
        seed_store: FeatureStore | ColumnSubset = store
        if lattice.n_seeded < store.n_pixels:
            seed_store = ColumnSubset(
                store, lattice.columns, step.seed_memory.block_pixels
            )
        seed_plan = LoopPlan.create(plan.seeding.stage, seed_store, step.seed_memory)
        seed_maps = self.correlator.run(
            seed_store, seed_plan, weights=resources.weights
        )
        self.timeline.mark(f"stage{plan.index}:seeded")
        work = seed_work(resources, lattice, seed_plan.hyps, seed_maps)
        local_weights = self.correlator.local_weights(
            loop_plan,
            resources.local.rows,
            resources.rows_key,
            budget=self.local_weights_bytes,
        )
        return self.correlator.run_local(
            store,
            loop_plan,
            work,
            chunk_bytes=self.local_bytes,
            local_weights=local_weights,
        )

    def _featurize(
        self, plan: StagePlan, pixels: PixelSelection, step: PassPlan
    ) -> FeatureStore:
        """A store of ``pixels`` (compact columns in selection order) for the pass."""
        stage = plan.stage
        self.store = None  # the previous pass's features are not read again
        store = FeatureStore(
            stage.rectangles,
            stage.precision,
            step.planes,
            len(pixels),
            step.memory.block_pixels,
            self.device.device,
        )
        self.timeline.mark("store:allocated")
        image, rows, cols = self.geometry.positions(pixels.index.to("cpu"))
        # the selection is sorted, so each image's positions are one run of columns
        ids, counts = torch.unique_consecutive(image, return_counts=True)
        start = 0
        for b, count in zip(ids.tolist(), counts.tolist(), strict=True):
            columns = slice(start, start + count)
            filtered = self.images.filtered_planes(b, step.planes, self.device.device)
            for rect in stage.rectangles:
                featurization = self.featurization(rect, plan.mode)
                for chunk in featurization.chunks(
                    filtered, rows[columns], cols[columns]
                ):
                    store.write(rect, chunk, column_offset=start)
            start += count
        store.check_complete()
        self.store = store
        return store

    def _synchronize(self) -> None:
        if self.device.is_cuda:
            torch.cuda.synchronize(self.device.device)
