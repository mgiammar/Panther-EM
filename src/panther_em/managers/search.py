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
on a background thread while the next micrograph is searched (``pending_writes``),
and the next micrographs can be read on another (``prefetch``).
``search_summary.json`` collects each micrograph's label counts. With ``fdr`` set (the
default) each micrograph's final picks are cut at its own false-discovery-rate
threshold after the search (:mod:`~panther_em.managers.fdr`).
"""

from __future__ import annotations

import json
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
from pydantic import Field, model_validator

from panther_em._precision import Precision
from panther_em.approximation import PerSpatialFrequencyErrorTable
from panther_em.filters import WhiteningFilter, ctf_planes
from panther_em.managers import leopard
from panther_em.managers._analytic import AnalyticSpec
from panther_em.managers._common import (
    Model,
    ProgramConfig,
    Rectangle,
    read_mask,
    read_micrograph,
    read_volume,
    save_output,
)
from panther_em.managers.fdr import (
    FdrNull,
    apply_fdr,
    calibrated_null_rate,
    null_cut_1fp,
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
from panther_em.search.normalization import (
    AfterCtfNormalization,
    CtfSpec,
    reference_noise,
    scale_table,
)

if TYPE_CHECKING:
    import numpy as np


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
        Save each prepared micrograph (``<name>_micrograph.h5``) and reuse it, which
        with modeled CTF planes is large. Default False.
    overwrite : bool
        Search (and prepare) again micrographs that already have results.
    keep_stage_results : bool
        Keep each stage's maps in the result files. Default True.
    pending_writes : int
        When >=1, the next micrograph is searched while the previous result is written.
        When 0, results are written before starting the next search. Default 1.
    prefetch : int
        Micrographs read ahead on a background thread while the current one is searched.
        Each holds its image in host memory Default 0 meaning each micrograph is read
        when its search starts.
    device : str
        Compute device. Default ``"cpu"``.
    normalization : {"after_ctf", "before_ctf"}
        Where each hypothesis's noise level is set. ``"after_ctf"`` (default): every
        ``(plane, template)`` hypothesis is rescaled to (nearly) the same noise
        variance under its CTF plane, as Leopard-EM and cisTEM normalize their
        CTF-filtered projections -- a factor per template row (once per run) times a
        factor per CTF plane (per micrograph). Lowers the null tail (the noise-only
        maximum) without changing particle z on average. ``"before_ctf"``: Panther's
        original behaviour, every template projection at unit variance before the
        CTF, so a hypothesis's noise variance varies with how much of its power the
        plane's CTF passes; set it to reproduce results made before the default
        changed (z-scores, null calibrations and thresholds from those runs do not
        carry over). See :mod:`panther_em.search.normalization` and
        ``docs/after_ctf_normalization.md``.
    fdr : float or None
        False discovery rate of the final picks, default 0.05: per micrograph, after
        the search, only the peaks above the Benjamini-Hochberg threshold under the
        micrograph's null (``fdr_null``) stay ACCEPTED. Applies when the final stage's
        threshold is ``"auto"``. ``None``: the fixed ``"auto"`` threshold, one
        expected false detection per micrograph (the behaviour before this option).
        See :mod:`panther_em.managers.fdr` and ``docs/fdr_detection.md``.
    fdr_reach : float
        With ``fdr`` set, the final stage's ``"auto"`` threshold -- the lowest z the
        FDR cut can reach (the floor) -- sits this many z below the 1-FP threshold
        (``fdr_anchor``'s). Default 0.5.
    fdr_gate_reach : float
        With ``fdr`` set, the earlier stages' ``"auto"`` thresholds (the stage-1
        gate) sit this many z below their 1-FP thresholds, moved with the final
        stage's (``fdr_anchor``). The gate's own margin keeps pixels about 0.3 z
        further down, so 0.25 covers the 0.5 floor; it sets the extra cost (1.6 to
        1.9x the search time on a 4k micrograph at B 0: stage 2 sees ~3x the
        pixels). Default 0.25.
    fdr_anchor : {"null", "formula"}
        Which 1-FP threshold ``fdr_reach`` and ``fdr_gate_reach`` are measured
        from. ``"null"`` (default): the micrograph's FDR null's own 1-FP cut, so
        the floor sits ``fdr_reach`` below the cut the null itself puts at one
        false peak. Every ``"auto"`` threshold moves by that cut minus the
        formula's: down for a smooth field (a blurred template), where the null is
        far below the formula and the FDR cut falls with it. ``"formula"``: the
        independent-trial 1-FP threshold, which ignores the template (the behaviour
        before this option). With ``fdr_null: formula``, or a null without a 1-FP
        crossing, both are the formula's.
    fdr_peak_radius : float
        Minimum distance between the peaks the FDR rule counts, pixels. Default 10
        (the particle export's ``peak_radius``).
    fdr_null : {"analytic", "formula"} or Path
        The FDR rule's null. ``"analytic"`` (default): each micrograph's analytic null,
        computed from the search's own template, filters, grid and the micrograph's
        power spectrum (``panther_em.managers.null.predict_search_null``; about a
        minute per 13-plane 4k micrograph on a GPU). ``"formula"``: the
        independent-trial count (conservative: about half the requested FDR on
        xenon_131 and Track S). A path: the ``null_summary.json`` of ``panther-em
        null`` for this search config (``fdr_null_source`` picks its null). A
        micrograph the analytic model cannot cover (or the summary lacks) keeps the
        formula, and its record says why.
    fdr_null_source : {"analytic", "empirical"}
        With a ``null_summary.json``: its analytic prediction (default) or the
        phase-randomised copies' rate table.
    fdr_volume : Path, optional
        The template volume for ``fdr_null: analytic``; default the one the
        decomposition records.
    fdr_analytic : AnalyticSpec
        Sampling of ``fdr_null: analytic`` (as ``panther-em null``'s ``analytic``).
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
    cache_micrographs: bool = False
    overwrite: bool = False
    keep_stage_results: bool = True
    pending_writes: int = Field(default=1, ge=0)
    prefetch: int = Field(default=0, ge=0)
    device: str = "cpu"
    normalization: Literal["after_ctf", "before_ctf"] = "after_ctf"
    fdr: float | None = Field(default=0.05, gt=0.0, lt=1.0)
    fdr_reach: float = Field(default=0.5, ge=0.0)
    fdr_gate_reach: float = Field(default=0.25, ge=0.0)
    fdr_anchor: Literal["null", "formula"] = "null"
    fdr_peak_radius: float = Field(default=10.0, gt=0.0)
    fdr_null: Literal["analytic", "formula"] | Path = "analytic"
    fdr_null_source: Literal["analytic", "empirical"] = "analytic"
    fdr_volume: Path | None = None
    fdr_analytic: AnalyticSpec = AnalyticSpec()

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


@dataclass(frozen=True)
class _Loaded:
    """A micrograph's host data: its cached :class:`Micrograph` (CPU) or raw image."""

    entry: MicrographEntry
    data: Micrograph | torch.Tensor
    mask: torch.Tensor | None


def _read(config: SearchConfig, entry: MicrographEntry) -> _Loaded:
    """Read ``entry``'s cached micrograph, or its MRC, and its mask (host only)."""
    cached = config.micrograph_path(entry.key)
    data: Micrograph | torch.Tensor
    if config.cache_micrographs and cached.exists() and not config.overwrite:
        data = Micrograph.load(cached)
        shape = data.image_shape
    else:
        data = read_micrograph(entry.path)
        if config.invert_contrast:
            data = -data
        shape = (int(data.shape[0]), int(data.shape[1]))
    mask = None if entry.mask is None else read_mask(entry.mask, shape)
    return _Loaded(entry, data, mask)


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
        self.normalization: AfterCtfNormalization | None = None
        # (frequencies, (D, F) reference noise) the row factors were computed under
        self.reference_noise: tuple[torch.Tensor, torch.Tensor] | None = None
        self.normalization_reports: dict[str, dict] = {}
        self.fdr_reports: dict[str, dict] = {}
        self._fdr_volume: np.ndarray | None = None
        if config.normalization == "after_ctf":
            self._normalize_after_ctf()

    def _final_table_tag(self) -> str:
        """The table of the last stage that has one (the reported z-scores' stage)."""
        for spec in reversed(self.config.stages):
            if spec.table is not None:
                return spec.table
        return self.config.noise_tag

    def _normalize_after_ctf(self) -> None:
        """Apply the run-wide row factors ``a_n`` to the template and every table.

        ``a_n`` equalizes, over the template rows, the noise variance of the final
        stage's kept correlation under a reference noise model: white noise under
        each searched plane's ring-averaged ``CTF^2``, averaged over the run's
        micrographs (:func:`~panther_em.search.normalization.reference_noise`). The
        per-plane factors follow per micrograph (:meth:`prepare`).
        """
        config = self.config
        table = self.tables[self._final_table_tag()]
        noise_table = self.tables[config.noise_tag]
        if not torch.equal(table.frequencies, noise_table.frequencies):
            # the micrographs' curves live on the noise table's axis
            table = noise_table
        optics = config.optics
        reference = reference_noise(
            table,
            [
                CtfSpec(
                    (m.defocus_u + m.defocus_v) / 2,
                    (m.defocus_u - m.defocus_v) / 2,
                    m.astigmatism_angle,
                )
                for m in config.micrographs
            ],
            config.defocus_offsets,
            pixel_size=optics.pixel_size,
            voltage=optics.voltage,
            spherical_aberration=optics.spherical_aberration,
            amplitude_contrast=optics.amplitude_contrast_ratio,
            phase_shift=optics.phase_shift,
            device=self.device,
        )
        self.normalization = AfterCtfNormalization.create(table, reference)
        self.reference_noise = (table.frequencies, reference)
        a = self.normalization.row_scale
        self.template.scale_rows(a)
        self.tables = {tag: scale_table(t, a) for tag, t in self.tables.items()}

    def n_hypotheses(self, spec: StageSpec) -> int:
        """Hypotheses per position: planes x filters x orientations x ``spec.n_psi``."""
        space = self.template.space
        return (
            len(self.config.defocus_offsets)
            * space.n_filters
            * space.n_orientations
            * spec.n_psi
        )

    @property
    def fdr_active(self) -> bool:
        """Whether the FDR rule sets the final picks (``fdr`` set, final stage auto)."""
        final = self.config.stages[-1].decision.threshold
        return self.config.fdr is not None and final == "auto"

    def threshold(self, spec: StageSpec, n_pixels: int, shift: float = 0.0) -> float:
        """The stage's threshold for a micrograph of ``n_pixels`` searched positions.

        Its own, or for ``"auto"`` the 1-FP formula's. When the FDR rule is on, an
        ``"auto"`` threshold is moved by ``shift`` (the anchor, :meth:`prepare_fdr`)
        and lowered by ``fdr_reach`` (final stage) or ``fdr_gate_reach`` (earlier
        stages).
        """
        if spec.decision.threshold != "auto":
            return float(spec.decision.threshold)
        t = zscore_threshold(n_pixels, self.n_hypotheses(spec))
        if not self.fdr_active:
            return t
        final = spec is self.config.stages[-1]
        reach = self.config.fdr_reach if final else self.config.fdr_gate_reach
        return t + shift - reach

    def prepare_fdr(
        self, n_pixels: int, name: str = "", micrograph: Micrograph | None = None
    ) -> FdrNull:
        """The micrograph's FDR null, and where it anchors the ``"auto"`` thresholds.

        Computed before the search: with ``fdr_anchor: null`` every ``"auto"``
        threshold moves by the null's 1-FP cut minus the formula's (``shift``), so
        the stages keep the pixels an FDR cut under this null can reach.
        """
        rate, record = self._fdr_null(n_pixels, name, micrograph)
        final = self.config.stages[-1]
        formula = zscore_threshold(n_pixels, self.n_hypotheses(final))
        anchor: dict[str, Any] = {"anchor": "formula", "formula_cut_1fp": formula}
        shift = 0.0
        if self.config.fdr_anchor == "null" and rate is not None:
            cut = null_cut_1fp(rate)
            if cut is None:
                anchor["anchor_note"] = "the null has no 1-FP crossing"
            else:
                shift = cut - formula
                anchor.update(anchor="null", anchor_cut_1fp=cut, anchor_shift=shift)
        return FdrNull(rate, {**(record or {"null": "formula"}), **anchor}, shift)

    def apply_fdr(
        self,
        result: SearchResult,
        n_pixels: int,
        name: str = "",
        micrograph: Micrograph | None = None,
        prepared: FdrNull | None = None,
    ) -> tuple[SearchResult, dict]:
        """Keep ACCEPTED only the peaks above the micrograph's BH threshold.

        ``n_pixels`` is the micrograph's searched positions (under its mask).
        ``prepared`` is the null the search's thresholds were set with
        (:meth:`prepare_fdr`); without it the null is computed here and the floor
        is the formula's (a search run with the unmoved thresholds). The null is
        ``fdr_null``'s: the analytic null of ``micrograph`` (the prepared one),
        ``name``'s from a null summary, or the formula's.
        """
        final = self.config.stages[-1]
        assert self.config.fdr is not None
        if prepared is None:
            rate, record = self._fdr_null(n_pixels, name, micrograph)
            prepared = FdrNull(rate, record or {"null": "formula"}, 0.0)
        return apply_fdr(
            result,
            q=self.config.fdr,
            n_positions=n_pixels,
            n_hypotheses=self.n_hypotheses(final),
            floor=self.threshold(final, n_pixels, prepared.shift),
            peak_radius=self.config.fdr_peak_radius,
            absorb_radius=final.decision.absorb_radius,
            rate=prepared.rate,
            null_record=prepared.record,
        )

    def _fdr_null(
        self, n_pixels: int, name: str, micrograph: Micrograph | None
    ) -> tuple[Any, dict[str, Any] | None]:
        """The FDR null's rate and its record (``None`` rate: the formula)."""
        choice = self.config.fdr_null
        if choice == "formula":
            return None, None
        if choice != "analytic":
            found = calibrated_null_rate(
                Path(choice), name, n_pixels, self.config.fdr_null_source
            )
            if found is None:
                return None, {
                    "null": "formula",
                    "note": f"no calibrated null for {name}",
                }
            return found
        if micrograph is None:
            return None, {"null": "formula", "note": "no micrograph to model"}
        # lazy: the null program imports this module
        from leopard_em.analysis.fdr import null_rate

        from panther_em.managers.null import predict_search_null, search_volume

        start = time.perf_counter()
        try:
            if self._fdr_volume is None:
                path = search_volume(self.config, self.config.fdr_volume)
                self._fdr_volume = read_volume(path)
            prediction = predict_search_null(
                self,
                micrograph,
                n_pixels,
                self._fdr_volume,
                spec=self.config.fdr_analytic,
                mask_radius=self.config.fdr_peak_radius,
            )
        except (NotImplementedError, ValueError, FileNotFoundError) as err:
            return None, {
                "null": "formula",
                "note": f"analytic null unavailable: {err}",
            }
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return null_rate(prediction), {
            "null": "analytic",
            "null_cut_1fp": prediction.zscore_cutoff(1.0),
            "null_seconds": time.perf_counter() - start,
        }

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

    def thresholds(
        self, micrograph: Micrograph, shift: float = 0.0
    ) -> list[float | None] | None:
        """Per-stage thresholds of a masked or FDR-anchored micrograph.

        ``shift`` moves every ``"auto"`` threshold (:meth:`prepare_fdr`). ``None``:
        the searcher's own (no mask, no shift).
        """
        if micrograph.mask is None and shift == 0.0:
            return None
        n_pixels = self.n_searchable(micrograph)
        return [
            (
                self.threshold(spec, n_pixels, shift)
                if spec.decision.threshold == "auto"
                else None
            )
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

    def prepare(self, loaded: _Loaded) -> Micrograph:
        """The searched micrograph of a :func:`_read` (device work and the cache).

        With after-CTF normalization, the per-plane factors are applied here to the
        micrograph under its unscaled CTF planes (which is what the cache holds).
        """
        micrograph = self._prepare_unscaled(loaded)
        if self.normalization is not None:
            micrograph, report = self.normalization.apply(micrograph)
            self.normalization_reports[loaded.entry.key] = report
        return micrograph

    def _prepare_unscaled(self, loaded: _Loaded) -> Micrograph:
        """The micrograph under its unscaled CTF planes (cached when allowed)."""
        config, entry = self.config, loaded.entry

        if isinstance(loaded.data, Micrograph):
            return loaded.data.to(self.device).with_mask(loaded.mask)

        raw = loaded.data.to(self.device)
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
            entry.key,
            self.whitening,
            ctf,
            self.tables[config.noise_tag],
            pixel_size=optics.pixel_size,
            mask=loaded.mask,
        )

        if config.cache_micrographs:
            save_output(
                micrograph.save,
                config.micrograph_path(entry.key),
                config,
                micrograph=entry.key,
            )

        return micrograph

    def micrograph(self, entry: MicrographEntry) -> Micrograph:
        """The prepared micrograph of ``entry`` (from the cache when allowed)."""
        return self.prepare(_read(self.config, entry))

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
    """
    return {
        "mask": None if mask is None else str(mask),
        "n_searchable": int((result.label != int(PixelLabel.MASKED)).sum()),
        "elapsed_s": sum(v for k, v in timing.items() if k != "read_wait_s"),
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
    todo = []
    for entry in config.micrographs:
        path = config.result_path(entry.key)
        if path.exists() and not config.overwrite:
            done[entry.key] = path
        else:
            todo.append(entry)
    state = None
    reads: dict[int, Future[tuple[_Loaded, float]]] = {}
    pending: deque[Future[tuple[str, Path, dict]]] = deque()

    def timed_read(entry: MicrographEntry) -> tuple[_Loaded, float]:
        t = time.perf_counter()
        loaded = _read(config, entry)
        return loaded, time.perf_counter() - t

    def finish(write: Future[tuple[str, Path, dict]]) -> None:
        name, path, record = write.result()
        done[name] = path
        if state is not None and name in state.normalization_reports:
            record["normalization"] = state.normalization_reports[name]
        if state is not None and name in state.fdr_reports:
            record["fdr"] = state.fdr_reports[name]
        summary[name] = record
        summary_path.write_text(json.dumps(summary, indent=1))

    with (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="writer") as writer,
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="reader") as reader,
    ):

        def read_ahead(start: int) -> None:
            """Read ``todo[start : start + prefetch]`` on the reader thread."""
            for j in range(start, min(start + config.prefetch, len(todo))):
                if j not in reads:
                    reads[j] = reader.submit(timed_read, todo[j])

        try:
            read_ahead(0)  # the first reads overlap building the template and tables
            for i, entry in enumerate(todo):
                name, path = entry.key, config.result_path(entry.key)
                state = state or _Run(config)
                timing: dict[str, float] = {}
                t0 = time.perf_counter()
                read = reads.pop(i, None)
                loaded, timing["read_s"] = (
                    timed_read(entry) if read is None else read.result()
                )
                read_ahead(i + 1)  # read while this micrograph is searched
                t1 = time.perf_counter()
                micrograph = state.prepare(loaded)
                del loaded
                shape = micrograph.image_shape
                if micrograph.mask is not None and state.n_searchable(micrograph) == 0:
                    summary[name] = {
                        "mask": str(entry.mask),
                        "n_searchable": 0,
                        "skipped": "the mask leaves no position to search",
                    }
                    summary_path.write_text(json.dumps(summary, indent=1))
                    continue
                t2 = time.perf_counter()
                prepared = None
                if state.fdr_active:  # the null first: it anchors the thresholds
                    prepared = state.prepare_fdr(
                        state.n_searchable(micrograph), name, micrograph
                    )
                t2b = time.perf_counter()
                timing["fdr_null_s"] = t2b - t2
                result = state.searcher(shape).run(
                    micrograph,
                    thresholds=state.thresholds(
                        micrograph, 0.0 if prepared is None else prepared.shift
                    ),
                )
                if prepared is not None:
                    result, state.fdr_reports[name] = state.apply_fdr(
                        result,
                        state.n_searchable(micrograph),
                        name,
                        micrograph,
                        prepared,
                    )
                t3 = time.perf_counter()
                maps = leopard.match_template_maps(result)  # on the compute device
                result = result.to("cpu")
                timing["read_wait_s"] = t1 - t0
                timing["prepare_s"] = t2 - t1
                timing["search_s"] = t3 - t2b  # the search and the FDR cut
                timing["handoff_s"] = time.perf_counter() - t3
                pending.append(
                    writer.submit(
                        _write, config, name, path, result, maps, timing, entry.mask
                    )
                )
                del micrograph, result, maps
                while len(pending) > config.pending_writes:
                    finish(pending.popleft())
        finally:
            for read in reads.values():
                read.cancel()
            if state is not None:
                state.release()
            while pending:
                finish(pending.popleft())
    return done
