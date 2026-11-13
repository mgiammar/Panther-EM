"""Program ``search``: the multi-stage search over a list of micrographs.

The template, the tables and the stages are built once; per micrograph the MRC is
read, its CTF planes built, the :class:`~panther_em.search.Micrograph` prepared (or
loaded from the cache) and searched, and the
:class:`~panther_em.search.SearchResult` written to
``output_dir/<name>_result.h5``, which also holds Leopard-EM's match-template maps
(:func:`~.leopard.write_match_template_maps`: the file is a Leopard-EM result too). A
micrograph whose result exists is skipped, so an interrupted run resumes where it
stopped. A micrograph's ``mask`` restricts the search to the positions whose box
center it covers (one whose mask leaves none is not searched). Results are written
on a background thread while the next micrograph is searched (``pending_writes``).
``search_summary.json`` collects each micrograph's label counts.
"""

from __future__ import annotations

import json
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar, Literal

import torch
from pydantic import Field, model_validator

from panther_em._precision import Precision
from panther_em.approximation import PerSpatialFrequencyErrorTable
from panther_em.filters import WhiteningFilter, ctf_planes
from panther_em.managers import leopard
from panther_em.managers._common import (
    Model,
    ProgramConfig,
    Rectangle,
    read_mask,
    read_micrograph,
    save_output,
)
from panther_em.search import (
    Calibration,
    Decision,
    ExecutionConfig,
    Micrograph,
    PixelLabel,
    Searcher,
    SearchImages,
    SearchResult,
    Stage,
    Template,
    ThresholdTest,
    Tuning,
    zscore_threshold,
)
from panther_em.search.correlation.contraction import BatchShape
from panther_em.search.images import SearchGeometry


class Optics(Model):
    """The dataset's optics (Leopard-EM's optics-group names and units)."""

    pixel_size: float = Field(gt=0)  # Angstrom / px
    voltage: float = 300.0  # kV
    spherical_aberration: float = 2.7  # mm
    amplitude_contrast_ratio: float = 0.07
    phase_shift: float = 0.0  # degrees


class MicrographEntry(Model):
    """One micrograph: its MRC file, fitted CTF (defocus in Angstroms) and mask."""

    path: Path
    name: str | None = None  # default: the file stem (the whitening filter's name)
    defocus_u: float
    defocus_v: float
    astigmatism_angle: float = 0.0  # degrees
    mask: Path | None = None

    @property
    def key(self) -> str:
        """The micrograph's name."""
        return self.name or self.path.stem


class CalibrationSpec(Model):
    """:class:`~panther_em.search.Calibration` of a stage's threshold test."""

    alpha: float = 5e-3
    zeta: float = 1.0
    eps_rho: float = 0.1
    b_factor: float = 0.0


class DecisionSpec(Model):
    """A stage's :class:`~panther_em.search.Decision`.

    ``threshold="auto"`` is :func:`~panther_em.search.zscore_threshold` of the
    micrograph's searched positions (those its mask leaves) and the stage's
    hypotheses (one expected false detection per micrograph). ``calibration`` adds
    the threshold test on the stage's table.
    """

    threshold: float | Literal["auto"] = "auto"
    calibration: CalibrationSpec | None = None
    accept: bool = True
    absorb_radius: int = Field(default=0, ge=0)


class StageSpec(Model):
    """A :class:`~panther_em.search.Stage` (see :meth:`Stage.create`)."""

    rectangles: list[Rectangle] = Field(min_length=1)
    table: str | None = None  # the ``tables`` tag of the threshold test
    precision: Precision = "fp32"
    n_psi: int = Field(default=256, ge=1)
    orientations: int | list[int] | None = None
    planes: list[int] | None = None
    local_rings: int | None = Field(default=None, ge=0)
    decision: DecisionSpec = DecisionSpec()
    name: str = ""


class SearchConfig(ProgramConfig):
    """Configuration of :func:`run`.

    Parameters
    ----------
    decomposition, whitening : Path
        The template's decomposition and the dataset's whitening filter.
    tables : dict
        ``tag -> error table file`` (the ``calibrate`` program's ``tables_<tag>.h5``).
    micrographs : list of MicrographEntry
        The micrographs to search, in order.
    optics : Optics
        The dataset's optics.
    defocus_offsets : list of float
        Defocus planes searched around each micrograph's fitted defocus, Angstroms;
        results report them as ``relative_defocus``. Default ``[0.0]``.
    invert_contrast : bool
        Negate each micrograph (when its contrast is opposite to the template's).
    stages : list of StageSpec
        The stages in execution order.
    weights_cells : Rectangle, optional
        Template weights kept resident (see :class:`~panther_em.search.Template`).
    execution : dict
        :class:`~panther_em.search.ExecutionConfig` fields; ``tuning`` a dict of
        :class:`~panther_em.search.Tuning` fields (``batch`` as ``[pixels,
        hypotheses]``).
    noise_table : str, optional
        The table the micrographs' noise curves are estimated on; default the first
        stage's table (or the first table).
    output_dir : Path
        Where results (and cached micrographs) go.
    cache_micrographs : bool
        Save each prepared micrograph (``<name>_micrograph.h5``) and reuse it.
    overwrite : bool
        Search (and prepare) again micrographs that already have results.
    keep_stage_results : bool
        Keep each stage's maps in the result files. Default True.
    pending_writes : int
        When >=1, the next micrograph is searched while the previous result is written.
        When 0, results are written before starting the next search. Default 1.
    device : str
        Compute device. Default ``"cpu"``.
    """

    program: ClassVar[str] = "search"

    decomposition: Path
    whitening: Path
    tables: dict[str, Path] = Field(min_length=1)
    micrographs: list[MicrographEntry] = Field(min_length=1)
    optics: Optics
    defocus_offsets: list[float] = Field(default=[0.0], min_length=1)
    invert_contrast: bool = False
    stages: list[StageSpec] = Field(min_length=1)
    weights_cells: Rectangle | None = None
    execution: dict[str, Any] = Field(default_factory=dict)
    noise_table: str | None = None
    output_dir: Path
    cache_micrographs: bool = True
    overwrite: bool = False
    keep_stage_results: bool = True
    pending_writes: int = Field(default=1, ge=0)
    device: str = "cpu"

    @model_validator(mode="after")
    def _links(self) -> SearchConfig:
        names = [m.key for m in self.micrographs]
        if len(set(names)) != len(names):
            raise ValueError("micrograph names must be unique")
        for i, stage in enumerate(self.stages):
            if stage.table is not None and stage.table not in self.tables:
                raise ValueError(f"stage {i}: unknown table {stage.table!r}")
            if stage.decision.calibration is not None and stage.table is None:
                raise ValueError(f"stage {i}: a threshold test needs a table")
        if self.noise_table is not None and self.noise_table not in self.tables:
            raise ValueError(f"unknown noise_table {self.noise_table!r}")
        return self

    @property
    def noise_tag(self) -> str:
        """The tag of the table the micrographs are prepared against."""
        if self.noise_table is not None:
            return self.noise_table
        first = self.stages[0].table
        return first if first is not None else next(iter(self.tables))

    def result_path(self, name: str) -> Path:
        """Where the result of micrograph ``name`` is written."""
        return self.output_dir / f"{name}_result.h5"

    def micrograph_path(self, name: str) -> Path:
        """Where the prepared micrograph ``name`` is cached."""
        return self.output_dir / f"{name}_micrograph.h5"


def execution_config(spec: dict[str, Any]) -> ExecutionConfig:
    """The :class:`~panther_em.search.ExecutionConfig` of a config's ``execution``."""
    fields = dict(spec)
    tuning = dict(fields.pop("tuning", {}))
    if tuning.get("batch") is not None:
        tuning["batch"] = BatchShape(*tuning["batch"])
    return ExecutionConfig(**fields, tuning=Tuning(**tuning))


class _Run:
    """The objects one run builds once: template, tables, searchers by image shape."""

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.template = Template.from_hdf5(
            config.decomposition, self.device, weights_cells=config.weights_cells
        )
        self.whitening = WhiteningFilter.load(config.whitening)
        filters = self.template.result.fourier_filters
        if filters is None:
            raise ValueError(
                f"'{config.decomposition}' was decomposed without the whitening's "
                "template filter"
            )
        self.whitening.check_template_filter(filters)
        self.tables = {
            tag: PerSpatialFrequencyErrorTable.load(path)
            for tag, path in config.tables.items()
        }
        self.execution = execution_config(config.execution)
        self.offsets = torch.tensor(config.defocus_offsets, dtype=torch.float32)
        self.searchers: dict[tuple[int, int], Searcher] = {}

    def threshold(self, spec: StageSpec, n_pixels: int) -> float:
        """The stage's threshold for a micrograph of ``n_pixels`` searched positions."""
        if spec.decision.threshold != "auto":
            return float(spec.decision.threshold)
        space = self.template.space
        n_hypotheses = (
            len(self.config.defocus_offsets)
            * space.n_filters
            * space.n_orientations
            * spec.n_psi
        )
        return zscore_threshold(n_pixels, n_hypotheses)

    def n_positions(self, shape: tuple[int, int]) -> int:
        """Search positions of an unmasked micrograph of ``shape``."""
        h, w = self.template.template_shape
        return (shape[0] - h + 1) * (shape[1] - w + 1)

    def n_searchable(self, micrograph: Micrograph) -> int:
        """Search positions of ``micrograph`` under its mask."""
        if micrograph.mask is None:
            return self.n_positions(micrograph.image_shape)
        geometry = SearchGeometry.from_images(
            SearchImages.single(micrograph.image), self.template.template_shape
        )
        return int(geometry.grid_mask(micrograph.mask).sum())

    def thresholds(self, micrograph: Micrograph) -> list[float | None] | None:
        """Per-stage thresholds of a masked micrograph (``None``: the searcher's)."""
        if micrograph.mask is None:
            return None
        n_pixels = self.n_searchable(micrograph)
        return [
            self.threshold(spec, n_pixels)
            if spec.decision.threshold == "auto"
            else None
            for spec in self.config.stages
        ]

    def searcher(self, shape: tuple[int, int]) -> Searcher:
        """The searcher of micrographs of ``shape`` (thresholds depend on it)."""
        if shape not in self.searchers:
            stages = []
            for spec in self.config.stages:
                d = spec.decision
                test = None
                if d.calibration is not None and spec.table is not None:
                    test = ThresholdTest(
                        self.tables[spec.table],
                        Calibration(**d.calibration.model_dump()),
                    )
                stages.append(
                    Stage.create(
                        spec.rectangles,
                        precision=spec.precision,
                        n_psi=spec.n_psi,
                        orientations=spec.orientations,
                        planes=spec.planes,
                        local_rings=spec.local_rings,
                        decision=Decision(
                            self.threshold(spec, self.n_positions(shape)),
                            test,
                            accept=d.accept,
                            absorb_radius=d.absorb_radius,
                        ),
                        name=spec.name,
                    )
                )
            self.searchers[shape] = Searcher(
                self.template,
                stages,
                execution=self.execution,
                n_defocus=len(self.config.defocus_offsets),
                defocus_values=self.offsets,
                keep_stage_results=self.config.keep_stage_results,
            )
        return self.searchers[shape]

    def micrograph(self, entry: MicrographEntry) -> Micrograph:
        """The prepared micrograph of ``entry`` (from the cache when allowed)."""
        config, name = self.config, entry.key
        cached = config.micrograph_path(name)
        if config.cache_micrographs and cached.exists() and not config.overwrite:
            micrograph = Micrograph.load(cached, self.device)
            mask = None
            if entry.mask is not None:
                mask = read_mask(entry.mask, micrograph.image_shape)
            return micrograph.with_mask(mask)
        raw = read_micrograph(entry.path, self.device)
        mask = None
        if entry.mask is not None:
            mask = read_mask(entry.mask, (int(raw.shape[0]), int(raw.shape[1])))
        if config.invert_contrast:
            raw = -raw
        optics = config.optics
        mean = (entry.defocus_u + entry.defocus_v) / 2
        ctf = ctf_planes(
            (int(raw.shape[0]), int(raw.shape[1])),
            [mean + o for o in config.defocus_offsets],
            pixel_size=optics.pixel_size,
            astigmatism=(entry.defocus_u - entry.defocus_v) / 2,
            astigmatism_angle=entry.astigmatism_angle,
            voltage=optics.voltage,
            spherical_aberration=optics.spherical_aberration,
            amplitude_contrast=optics.amplitude_contrast_ratio,
            phase_shift=optics.phase_shift,
            device=self.device,
        )
        micrograph = Micrograph.prepare(
            raw,
            name,
            self.whitening,
            ctf,
            self.tables[config.noise_tag],
            pixel_size=optics.pixel_size,
            mask=mask,
        )
        if config.cache_micrographs:
            save_output(micrograph.save, cached, config, micrograph=name)
        return micrograph

    def release(self) -> None:
        for searcher in self.searchers.values():
            searcher.release()
        self.searchers.clear()


def _label_counts(labels: torch.Tensor) -> dict[str, int]:
    labels = labels.cpu()
    return {lab.name: int((labels == int(lab)).sum()) for lab in PixelLabel}


def _summary(
    result: SearchResult, timing: dict[str, float], mask: Path | None = None
) -> dict:
    """Labels and wall-clock seconds of one micrograph.

    ``mask`` is the micrograph's mask file and ``n_searchable`` the positions it
    leaves (every position without a mask); ``final`` counts the rest ``MASKED``.

    ``timing`` holds the manager's steps (``prepare_s``: load, CTF and whitening, or
    the micrograph cache; ``search_s``; ``handoff_s``: Leopard-EM maps and the copy
    to host; then, on the writer thread and overlapping the next search,
    ``save_result_s`` and ``leopard_maps_s``); each stage adds its own
    ``featurize_s``, ``execute_s`` (correlation) and ``other_s`` (planning, seeding,
    ledger write and decision).
    """
    return {
        "mask": None if mask is None else str(mask),
        "n_searchable": int((result.label != int(PixelLabel.MASKED)).sum()),
        "elapsed_s": sum(timing.values()),
        **timing,
        "thresholds": [
            None if s.decision is None else float(s.decision.threshold)
            for s in result.stages
        ],
        "stages": [
            {
                "name": s.stage.name,
                "n_pixels": int(s.n_pixels),
                "elapsed_s": s.elapsed_s,
                "featurize_s": s.featurize_s,
                "execute_s": s.execute_s,
                "other_s": max(s.elapsed_s - s.featurize_s - s.execute_s, 0.0),
                **_label_counts(s.labels_after),
            }
            for s in result.stage_results
        ],
        "final": _label_counts(result.label),
    }


def _write(
    config: SearchConfig,
    name: str,
    path: Path,
    result: SearchResult,
    maps: dict[str, torch.Tensor],
    timing: dict[str, float],
    mask: Path | None = None,
) -> tuple[str, Path, dict]:
    """Write ``result`` and its Leopard-EM ``maps`` to ``path`` (writer thread)."""

    def save(file: Path) -> None:
        t0 = time.perf_counter()
        result.save(file)
        t1 = time.perf_counter()
        leopard.write_match_template_maps(file, result, maps)
        timing["save_result_s"] = t1 - t0
        timing["leopard_maps_s"] = time.perf_counter() - t1

    save_output(save, path, config, micrograph=name)
    return name, path, _summary(result, timing, mask)


def run(config: SearchConfig) -> dict[str, Path]:
    """Search every micrograph without a result; returns each micrograph's result."""
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "search_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    done = {}
    state = None
    pending: deque[Future[tuple[str, Path, dict]]] = deque()

    def finish(write: Future[tuple[str, Path, dict]]) -> None:
        name, path, record = write.result()
        done[name] = path
        summary[name] = record
        summary_path.write_text(json.dumps(summary, indent=1))

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="writer") as writer:
        try:
            for entry in config.micrographs:
                name, path = entry.key, config.result_path(entry.key)
                if path.exists() and not config.overwrite:
                    done[name] = path
                    continue
                state = state or _Run(config)
                timing: dict[str, float] = {}
                t0 = time.perf_counter()
                micrograph = state.micrograph(entry)
                shape = (
                    int(micrograph.image.shape[-2]),
                    int(micrograph.image.shape[-1]),
                )
                if micrograph.mask is not None and state.n_searchable(micrograph) == 0:
                    summary[name] = {
                        "mask": str(entry.mask),
                        "n_searchable": 0,
                        "skipped": "the mask leaves no position to search",
                    }
                    summary_path.write_text(json.dumps(summary, indent=1))
                    continue
                t1 = time.perf_counter()
                result = state.searcher(shape).run(
                    micrograph, thresholds=state.thresholds(micrograph)
                )
                t2 = time.perf_counter()
                maps = leopard.match_template_maps(result)  # on the compute device
                result = result.to("cpu")
                timing["prepare_s"] = t1 - t0
                timing["search_s"] = t2 - t1
                timing["handoff_s"] = time.perf_counter() - t2
                pending.append(
                    writer.submit(
                        _write, config, name, path, result, maps, timing, entry.mask
                    )
                )
                del micrograph, result, maps
                while len(pending) > config.pending_writes:
                    finish(pending.popleft())
        finally:
            if state is not None:
                state.release()
            while pending:
                finish(pending.popleft())
    return done
