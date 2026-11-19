"""Program ``null``: a search's null, measured on phase-randomised copies and predicted.

A detection threshold should come from the search's null: what the identical pipeline
finds in a micrograph that holds no particles. The independent-trials formula behind
the ``auto`` threshold only approximates it, because neighbouring hypotheses, planes
and pixels are correlated and the z-score is normalized per pixel.

**Measured** (``mode: empirical`` or ``both``). Each listed micrograph of a search
config is phase-randomised once per seed (its Fourier amplitudes kept, every phase
random: :func:`leopard_em.utils.phase_randomization.phase_randomize_image`) and the
copy is searched exactly as the micrograph would be, under the micrograph's own
name, so its whitening row, CTF planes and after-CTF plane factors apply, with the
search config's template, tables, stages and thresholds. The distinct peaks of each
copy's final z-score map (:func:`leopard_em.analysis.empirical_null.distinct_peaks`,
counted only where the final stage scored the pixel) are the null peaks. The
threshold for a target of ``f`` false positives per micrograph is the
``floor(R f)``-th highest of the R copies' pooled peaks
(:func:`leopard_em.analysis.empirical_null.empirical_zscore_cutoff`).

**Predicted** (``mode: analytic`` or ``both``). The null peak rate is computed from
the template, the CTF and whitening filters and the image power spectrum, with no
search (:mod:`panther_em.search.null_model`).

Outputs in ``output_dir``:

* ``null_peaks.h5``: per micrograph and copy the null peaks (``y``, ``x``, ``z``) and
  the valid area; it is written after every copy, and a rerun searches only the
  copies it lacks;
* ``null_summary.json``: per micrograph the null peak rate ``lambda(t)`` (peaks at or
  above ``t`` per micrograph, with Poisson intervals), the threshold for each
  ``false_positives`` target (measured, predicted and the independent-trials
  formula's), the production gate's completeness floor and, with the analytic mode,
  the prediction;
* ``<name>_s<seed>_result.h5``: each copy's search result, with ``keep_results``.

The gate. A multi-stage search drops pixels a stage's threshold test predicts cannot
reach the threshold; below a floor the final z-score map, and so the null count, can
be incomplete. The floor is recorded (``gate``) and every threshold is flagged
``complete`` when it lies at or above it. Seeds: micrograph ``i`` of the search config
uses ``seed + i * seed_stride`` for each listed ``seed``, so different micrographs get
different copies.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import h5py
import numpy as np
import torch
from leopard_em.analysis.analytic_null import refine_window
from leopard_em.analysis.empirical_null import NullPeaks, distinct_peaks, valid_region
from leopard_em.utils.phase_randomization import phase_randomize_image
from pydantic import Field, model_validator
from scipy.special import ndtri

from panther_em.filters import ctf_planes
from panther_em.managers._analytic import AnalyticSpec
from panther_em.managers._common import (
    Model,
    ProgramConfig,
    centered,
    read_config,
    read_image,
    read_mask,
    read_volume,
    save_output,
)
from panther_em.managers.decompose import DecomposeConfig
from panther_em.managers.search import (
    MicrographEntry,
    SearchConfig,
    StageSpec,
    _Loaded,
    _read,
    _Run,
)
from panther_em.search import PixelLabel, SearchImages
from panther_em.search.images import SearchGeometry
from panther_em.search.null_model import (
    SearchTemplates,
    healpix_nside,
    predict_null,
    predict_refined_null,
    reference_grid,
)

if TYPE_CHECKING:
    from leopard_em.analysis.analytic_null import (
        AnalyticNullPrediction,
        RefinedNullPrediction,
    )

    from panther_em.search import Micrograph, SearchResult

PEAKS_FORMAT = "null_peaks"
DEFAULT_FLOOR_OFFSET = 1.5  # default peak floor: the final threshold minus this


class RefineSpec(Model):
    """The refinement of the search's candidates the analytic refined null models.

    Parameters
    ----------
    config : Path
        The Leopard-EM refine_template YAML the candidates are refined with; only its
        orientation, defocus and pixel size refinement settings are read
        (:func:`leopard_em.analysis.analytic_null.refine_window`).
    candidate_threshold : float
        Panther z-score at or above which candidates are refined (exported).
    n_candidates : int
        Rotations sampled as null candidates. Default 8.
    n_draws : int
        Gaussian draws per candidate. Default 200,000.
    shift_radius : int
        Positions refinement scores about each candidate, +-pixels. Default 1.
    condition_radius : int
        Positions at which the candidate beat its grid neighbours (it is a distinct
        peak), +-pixels. Default 2.
    neighbour_rings, neighbour_psi_steps : int
        The grid neighbourhood the candidate beat: HEALPix rings around its view and
        in-plane steps around its angle. Default 1 and 1.
    defocus_step : float, optional
        Defocus step of the modelled window, A, when coarser than the refine's own.
        Default 40; ``None``: the refine's own step.
    """

    config: Path
    candidate_threshold: float
    n_candidates: int = Field(default=8, ge=1)
    n_draws: int = Field(default=200_000, ge=1000)
    shift_radius: int = Field(default=1, ge=0)
    condition_radius: int = Field(default=2, ge=0)
    neighbour_rings: int = Field(default=1, ge=0)
    neighbour_psi_steps: int = Field(default=1, ge=0)
    defocus_step: float | None = Field(default=40.0, gt=0.0)


class NullConfig(ProgramConfig):
    """Configuration of :func:`run`.

    Parameters
    ----------
    search : Path
        The search config (YAML) whose pipeline the nulls repeat.
    output_dir : Path
        Where the null peaks, summary and (optionally) results go; never the
        search's own output directory.
    mode : {"empirical", "analytic", "both"}
        Search phase-randomised copies, predict analytically, or both. Default
        ``"empirical"``.
    micrographs : list of str, optional
        Names of the search's micrographs to calibrate; default all.
    seeds : list of int
        One phase-randomised copy per seed (default 1 to 5); at least 5 copies for a
        1-false-positive target, about +-0.05 z.
    seed_stride : int
        Micrograph ``i`` of the search config uses ``seed + i * seed_stride``.
    false_positives : list of float
        Targets, false positives per micrograph. Default ``[1.0]``.
    rule : {"mean", "confidence"}
        ``"mean"`` delivers the target on average; ``"confidence"`` at most the target
        with probability ``confidence``
        (:func:`leopard_em.analysis.empirical_null.counted_rank`).
    confidence : float
        Probability of the ``"confidence"`` rule. Default 0.9.
    peak_radius : float
        Suppression radius of the peak finder, pixels (as the export's). Default 10.
    border : int
        Search positions within ``border`` of the z-score map's edge are not counted;
        use the border the micrograph's own picks use. Default 0.
    peak_floor : float, optional
        Lowest null peak z kept; default the final stage's threshold minus 1.5.
    rate_step : float
        Spacing of the thresholds of the ``lambda(t)`` table. Default 0.05.
    keep_results : bool
        Also write each copy's search result. Default False.
    volume : Path, optional
        The template volume (analytic mode); default the one the decomposition
        records.
    analytic : AnalyticSpec
        Sampling of the analytic prediction.
    refine : RefineSpec, optional
        With the analytic mode, also predict the null after the candidates are
        refined (a threshold on refined z-scores).
    device : str, optional
        Compute device; default the search's.
    overwrite : bool
        Search every copy again (default: keep those already in ``null_peaks.h5``).
    """

    program: ClassVar[str] = "null"

    search: Path
    output_dir: Path
    mode: Literal["empirical", "analytic", "both"] = "empirical"
    micrographs: list[str] | None = None
    seeds: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5], min_length=1)
    seed_stride: int = Field(default=1000, ge=1)
    false_positives: list[float] = Field(default_factory=lambda: [1.0], min_length=1)
    rule: Literal["mean", "confidence"] = "mean"
    confidence: float = Field(default=0.9, gt=0.0, lt=1.0)
    peak_radius: float = Field(default=10.0, gt=0.0)
    border: int = Field(default=0, ge=0)
    peak_floor: float | None = None
    rate_step: float = Field(default=0.05, gt=0.0)
    keep_results: bool = False
    volume: Path | None = None
    analytic: AnalyticSpec = AnalyticSpec()
    refine: RefineSpec | None = None
    device: str | None = None
    overwrite: bool = False

    @model_validator(mode="after")
    def _check(self) -> NullConfig:
        if len(set(self.seeds)) != len(self.seeds) or min(self.seeds) < 0:
            raise ValueError("seeds must be distinct and non-negative")
        if max(self.seeds) >= self.seed_stride:
            raise ValueError("every seed must be below seed_stride")
        if any(not f > 0.0 for f in self.false_positives):
            raise ValueError("false_positives must be positive")
        if self.micrographs is not None and len(set(self.micrographs)) != len(
            self.micrographs
        ):
            raise ValueError("micrographs must be unique")
        if self.refine is not None and not self.predicted:
            raise ValueError("refine needs mode 'analytic' or 'both'")
        return self

    @property
    def empirical(self) -> bool:
        """Whether phase-randomised copies are searched."""
        return self.mode in ("empirical", "both")

    @property
    def predicted(self) -> bool:
        """Whether the analytic prediction is made."""
        return self.mode in ("analytic", "both")

    @property
    def peaks_path(self) -> Path:
        """``null_peaks.h5``."""
        return self.output_dir / "null_peaks.h5"

    @property
    def summary_path(self) -> Path:
        """``null_summary.json``."""
        return self.output_dir / "null_summary.json"

    def result_path(self, name: str, seed: int) -> Path:
        """Where copy ``seed`` of micrograph ``name`` is written (``keep_results``)."""
        return self.output_dir / f"{name}_s{seed}_result.h5"


# --------------------------------------------------------------------------- #
# Phase-randomised copies and their peaks
# --------------------------------------------------------------------------- #


def null_image(
    path: str | Path, seed: int, invert_contrast: bool = False
) -> torch.Tensor:
    """The phase-randomised copy of a micrograph, as the search reads micrographs.

    The MRC file as stored, phase-randomised with ``seed``, then centred (and negated
    with ``invert_contrast``) as :func:`~._common.read_micrograph` and the search do.
    """
    image = centered(phase_randomize_image(read_image(path), seed))
    return -image if invert_contrast else image


def final_zscore(result: SearchResult) -> tuple[np.ndarray, np.ndarray]:
    """The z-score map where the final stage scored a pixel (else -inf), and the mask.

    Returns the ``(out_h, out_w)`` float32 map of the result's single image and its
    boolean exclusion mask (positions outside the micrograph's mask).
    """
    final = len(result.stages) - 1
    z = result.zscore[0].detach().cpu().to(torch.float32).numpy()
    scored = (result.last_stage[0] == final).cpu().numpy() & np.isfinite(z)
    masked = (result.label[0] == int(PixelLabel.MASKED)).cpu().numpy()
    return np.where(scored, z, np.float32(-np.inf)), masked


class _Copy:
    """One null copy's peaks (``z`` descending) and bookkeeping."""

    def __init__(
        self,
        y: np.ndarray,
        x: np.ndarray,
        z: np.ndarray,
        valid_area: int,
        seconds: float = 0.0,
    ) -> None:
        self.y = np.asarray(y, dtype=np.int32)
        self.x = np.asarray(x, dtype=np.int32)
        self.z = np.asarray(z, dtype=np.float64)
        self.valid_area = int(valid_area)
        self.seconds = float(seconds)


Records = dict[str, dict[int, _Copy]]


def _peak_settings(config: NullConfig, floor: dict[str, float]) -> dict[str, Any]:
    return {
        "peak_radius": float(config.peak_radius),
        "border": int(config.border),
        "peak_floor": {k: float(v) for k, v in floor.items()},
    }


def _write_peaks(
    config: NullConfig,
    search_config: SearchConfig,
    records: Records,
    settings: dict[str, Any],
) -> None:
    def write(path: Path) -> None:
        with h5py.File(path, "w") as f:
            f.attrs["format"] = PEAKS_FORMAT
            f.attrs["settings"] = json.dumps(settings)
            for name, copies in records.items():
                group = f.create_group(name)
                group.attrs["peak_floor"] = settings["peak_floor"][name]
                for seed, copy in sorted(copies.items()):
                    g = group.create_group(f"s{seed}")
                    g.create_dataset("y", data=copy.y)
                    g.create_dataset("x", data=copy.x)
                    g.create_dataset("z", data=copy.z)
                    g.attrs["seed"] = int(seed)
                    g.attrs["valid_area"] = copy.valid_area
                    g.attrs["seconds"] = copy.seconds

    save_output(
        write,
        config.peaks_path,
        config,
        search_config=search_config.model_dump_json(),
    )


def read_null_peaks(path: str | Path) -> tuple[Records, dict[str, Any]]:
    """The copies' peaks of a ``null_peaks.h5`` and its peak settings."""
    records: Records = {}
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != PEAKS_FORMAT:
            raise ValueError(f"'{path}' is not a null_peaks file")
        settings = json.loads(f.attrs["settings"])
        for name, group in f.items():
            records[name] = {
                int(g.attrs["seed"]): _Copy(
                    g["y"][()],
                    g["x"][()],
                    g["z"][()],
                    int(g.attrs["valid_area"]),
                    float(g.attrs["seconds"]),
                )
                for g in group.values()
            }
    return records, settings


def null_peaks(records: dict[int, _Copy], floor: float, radius: float) -> NullPeaks:
    """The :class:`~leopard_em.analysis.empirical_null.NullPeaks` of one micrograph."""
    seeds = sorted(records)
    return NullPeaks(
        zscores=[records[s].z for s in seeds],
        valid_areas=[records[s].valid_area for s in seeds],
        cutoff=float(floor),
        mask_radius=float(radius),
        seeds=seeds,
    )


# --------------------------------------------------------------------------- #
# The gate, thresholds and the summary
# --------------------------------------------------------------------------- #


def gate_floor(config: SearchConfig, thresholds: list[float]) -> dict[str, Any]:
    """The z below which a stage before the final one may drop null peaks.

    A stage with a threshold test discards a pixel when its prediction lies below
    ``zeta * T`` (``T`` its threshold) with confidence ``1 - alpha``, so final z at
    or above ``zeta * T`` are complete to about that confidence; a stage without a
    test rejects below ``T`` on its own z. The floor is the highest such value over
    the stages before the final one (``None``: no gate).
    """
    stages: list[dict[str, Any]] = []
    for spec, threshold in zip(config.stages[:-1], thresholds[:-1], strict=True):
        calibration = spec.decision.calibration
        zeta = 1.0 if calibration is None else calibration.zeta
        floor = float(threshold) * zeta
        stages.append(
            {
                "name": spec.name,
                "threshold": float(threshold),
                "zeta": zeta,
                "alpha": None if calibration is None else calibration.alpha,
                "floor": floor if math.isfinite(floor) and floor > -1e8 else None,
            }
        )
    floors = [s["floor"] for s in stages if s["floor"] is not None]
    return {"stages": stages, "floor": max(floors) if floors else None}


def _thresholds_of(state: _Run, micrograph: Micrograph) -> list[float]:
    """Every stage's threshold for ``micrograph`` (as its search sets them)."""
    n_pixels = state.n_searchable(micrograph)
    return [state.threshold(spec, n_pixels) for spec in state.config.stages]


def _formula_cut(fp: float, area: float, hypotheses: int) -> float:
    """Independent-trials threshold ``Q^-1(fp / (area * hypotheses))``."""
    return float(-ndtri(fp / (float(area) * float(hypotheses))))


def _cuts(
    config: NullConfig,
    peaks: NullPeaks | None,
    prediction: AnalyticNullPrediction | None,
    area: float,
    hypotheses: int,
    gate: float | None,
    refined: RefinedNullPrediction | None = None,
) -> list[dict[str, Any]]:
    out = []
    for fp in config.false_positives:
        cut: dict[str, Any] = {
            "false_positives": fp,
            "formula": _formula_cut(fp, area, hypotheses),
        }
        if peaks is not None:
            cut["rule"] = config.rule
            try:
                value = peaks.zscore_cutoff(fp, config.rule, config.confidence)
            except ValueError as err:
                value, cut["note"] = float("nan"), str(err)
            measured = float(value)
            cut["empirical"] = measured if math.isfinite(measured) else None
            cut["complete"] = (
                None
                if cut["empirical"] is None
                else bool(gate is None or measured >= gate)
            )
        if prediction is not None:
            predicted = prediction.zscore_cutoff(fp)
            cut["analytic"] = predicted if math.isfinite(predicted) else None
        if refined is not None:
            value = refined.zscore_cutoff(fp)
            cut["analytic_refined"] = value if math.isfinite(value) else None
        out.append(cut)
    return out


def _rate_table(peaks: NullPeaks, step: float) -> dict[str, Any]:
    pooled = np.concatenate(peaks.zscores) if peaks.zscores else np.zeros(0)
    top = float(pooled.max()) if pooled.size else peaks.cutoff
    start = math.ceil(peaks.cutoff / step - 1e-9) * step
    grid = np.round(np.arange(start, top + step, step), 6)
    table: dict[str, Any] = peaks.rate(grid).as_dict()
    return table


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


def _volume_path(config: NullConfig, search_config: SearchConfig) -> Path:
    return search_volume(search_config, config.volume)


def _valid_area(state: _Run, micrograph: Micrograph, border: int) -> int:
    """Search positions peaks are counted in (border and mask removed)."""
    geometry = SearchGeometry.from_images(
        SearchImages.single(micrograph.image), state.template.template_shape
    )
    shape = (int(geometry.grid[0]), int(geometry.grid[1]))
    exclude = None
    if micrograph.mask is not None:
        exclude = ~geometry.grid_mask(micrograph.mask).cpu().numpy()
    return int(valid_region(shape, border, exclude).sum())


def _model_inputs(
    spec: AnalyticSpec, state: _Run, micrograph: Micrograph
) -> dict[str, Any]:
    """The analytic model's inputs from the search (for the two predictions)."""
    final: StageSpec = state.config.stages[-1]
    if final.orientations is not None:
        raise NotImplementedError(
            "the analytic null models a final stage over every orientation"
        )
    result = state.template.result
    if result.phi_values is None or result.theta_values is None:
        raise ValueError("the decomposition records no orientations")
    nside = healpix_nside(result.phi_values, result.theta_values)
    device = spec.device or state.config.device
    box = state.template.template_shape
    reference = (
        None
        if state.reference_noise is None
        else reference_grid(*state.reference_noise, box, device=device)
    )
    report = state.normalization_reports.get(micrograph.name)
    return {
        "template_filter": state.whitening.template_filter(box),
        "nside": nside,
        "n_psi": final.n_psi,
        "planes": final.planes,
        "n_filters": state.template.space.n_filters,
        "support": state.template.reconstructor.support,
        "normalize_projections": bool(result.normalized_projections),
        "reference": reference,
        "n_mean_samples": spec.n_mean_samples,
        "seed": spec.seed,
        "device": device,
        "details": {
            "micrograph": micrograph.name,
            "plane_scale": None if report is None else report["plane_scale"],
        },
    }


def predict_search_null(
    state: _Run,
    micrograph: Micrograph,
    area: int,
    volume: np.ndarray,
    *,
    spec: AnalyticSpec | None = None,
    mask_radius: float = 10.0,
) -> AnalyticNullPrediction:
    """The analytic null of a prepared micrograph under a search run's own settings.

    The search's template, filters, grid, final stage and after-CTF state
    (:class:`~panther_em.managers.search._Run`), over ``area`` positions; ``volume``
    is the template volume the decomposition was made from (:func:`search_volume`).
    The FDR rule (``SearchConfig.fdr_null: analytic``) and :func:`run` use it.
    """
    spec = AnalyticSpec() if spec is None else spec
    inputs = _model_inputs(spec, state, micrograph)
    template_filter = inputs.pop("template_filter")
    return predict_null(
        micrograph,
        volume,
        template_filter,
        valid_area=area,
        mask_radius=mask_radius,
        n_kappa_pairs=spec.n_kappa_pairs,
        n_lattice_samples=spec.n_lattice_samples,
        batch_size=spec.batch_size,
        **inputs,
    )


def search_volume(search_config: SearchConfig, volume: Path | None = None) -> Path:
    """The template volume: ``volume``, else the one the decomposition records."""
    if volume is not None:
        return volume
    recorded = read_config(search_config.decomposition, DecomposeConfig)
    if recorded is None:
        raise ValueError(
            f"'{search_config.decomposition}' records no decompose config; give the "
            "template volume"
        )
    return recorded.volume


def _predict(
    config: NullConfig,
    state: _Run,
    micrograph: Micrograph,
    area: int,
    volume: np.ndarray,
) -> AnalyticNullPrediction:
    return predict_search_null(
        state,
        micrograph,
        area,
        volume,
        spec=config.analytic,
        mask_radius=config.peak_radius,
    )


def _predict_refined(
    config: NullConfig,
    state: _Run,
    entry: MicrographEntry,
    micrograph: Micrograph,
    volume: np.ndarray,
    search: AnalyticNullPrediction,
) -> RefinedNullPrediction:
    refine = config.refine
    assert refine is not None
    inputs = _model_inputs(config.analytic, state, micrograph)
    template_filter = inputs.pop("template_filter")
    offsets, defocus = refine_window(refine.config, refine.defocus_step)
    search_config = state.config
    optics = search_config.optics
    mean = (entry.defocus_u + entry.defocus_v) / 2
    planes = inputs["planes"]
    offsets_all = [float(o) for o in search_config.defocus_offsets]
    plane_defocus = offsets_all if planes is None else [offsets_all[i] for i in planes]

    box = state.template.template_shape

    def refine_ctf(relative: np.ndarray) -> torch.Tensor:
        # on the template box, as refine_template's calculate_ctf_filter_stack
        return ctf_planes(
            box,
            [mean + float(o) for o in relative],
            pixel_size=optics.pixel_size,
            astigmatism=(entry.defocus_u - entry.defocus_v) / 2,
            astigmatism_angle=entry.astigmatism_angle,
            voltage=optics.voltage,
            spherical_aberration=optics.spherical_aberration,
            amplitude_contrast=optics.amplitude_contrast_ratio,
            phase_shift=optics.phase_shift,
            device=inputs["device"],
        )

    final: StageSpec = state.config.stages[-1]
    templates = SearchTemplates(
        state.template.reconstructor,
        [tuple(int(v) for v in r) for r in final.rectangles],
    )
    return predict_refined_null(
        micrograph,
        volume,
        template_filter,
        state.whitening.average_filter(box, inputs["device"]),
        search,
        templates,
        orientation_offsets=offsets,
        defocus_offsets=defocus,
        candidate_threshold=refine.candidate_threshold,
        refine_ctf=refine_ctf,
        plane_defocus=plane_defocus,
        shift_radius=refine.shift_radius,
        condition_radius=refine.condition_radius,
        neighbour_rings=refine.neighbour_rings,
        neighbour_psi_steps=refine.neighbour_psi_steps,
        n_candidates=refine.n_candidates,
        n_draws=refine.n_draws,
        **inputs,
    )


def _search_copy(
    config: NullConfig,
    state: _Run,
    entry: MicrographEntry,
    seed: int,
    floor: float,
) -> _Copy:
    """Search one phase-randomised copy of ``entry``; its distinct null peaks."""
    t0 = time.perf_counter()
    raw = null_image(entry.path, seed, state.config.invert_contrast)
    shape = (int(raw.shape[0]), int(raw.shape[1]))
    mask = None if entry.mask is None else read_mask(entry.mask, shape)
    micrograph = state.prepare(_Loaded(entry, raw, mask))
    result = state.searcher(micrograph.image_shape).run(
        micrograph, thresholds=state.thresholds(micrograph)
    )
    result = result.to("cpu")
    if config.keep_results:
        save_output(
            result.save,
            config.result_path(entry.key, seed),
            config,
            micrograph=entry.key,
            seed=str(seed),
        )
    z, masked = final_zscore(result)
    found = distinct_peaks(z, floor, config.peak_radius, config.border, masked)
    return _Copy(
        found.pos_y,
        found.pos_x,
        found.zscore,
        found.valid_area,
        time.perf_counter() - t0,
    )


def _hypotheses(state: _Run) -> int:
    """Hypotheses per position of the final stage."""
    final = state.config.stages[-1]
    space = state.template.space
    planes = len(final.planes or state.config.defocus_offsets)
    return planes * space.n_filters * space.n_orientations * final.n_psi


def run(config: NullConfig) -> dict[str, Any]:
    """Measure and/or predict the null of every listed micrograph; the summary."""
    search_config = SearchConfig.from_yaml(config.search)
    out = Path(config.output_dir)
    if out.resolve() == Path(search_config.output_dir).resolve():
        raise ValueError("output_dir must not be the search's output directory")
    search_config = search_config.model_copy(
        update={
            "output_dir": out,
            "cache_micrographs": False,  # never cache a copy under the source's name
            "overwrite": True,
            **({} if config.device is None else {"device": config.device}),
        }
    )
    names = [m.key for m in search_config.micrographs]
    selected = names if config.micrographs is None else list(config.micrographs)
    unknown = sorted(set(selected) - set(names))
    if unknown:
        raise ValueError(f"micrographs not in the search config: {unknown}")
    out.mkdir(parents=True, exist_ok=True)

    records: Records = {}
    floors: dict[str, float] = {}
    if config.empirical and config.peaks_path.exists() and not config.overwrite:
        records, stored = read_null_peaks(config.peaks_path)
        mine = {"peak_radius": config.peak_radius, "border": config.border}
        if any(stored[k] != v for k, v in mine.items()):
            raise ValueError(
                f"'{config.peaks_path}' was made with other peak settings "
                f"({stored}); set overwrite: true to search the copies again"
            )
        floors = {k: float(v) for k, v in stored["peak_floor"].items()}

    state = _Run(search_config)
    volume = (
        read_volume(_volume_path(config, search_config)) if config.predicted else None
    )
    summary: dict[str, Any] = {
        "search": str(config.search),
        "normalization": search_config.normalization,
        "mode": config.mode,
        "micrographs": {},
    }
    try:
        for index, name in enumerate(names):
            if name not in selected:
                continue
            entry = search_config.micrographs[index]
            source = state.prepare(_read(search_config, entry))
            report = state.normalization_reports.get(name)
            if source.mask is not None and state.n_searchable(source) == 0:
                summary["micrographs"][name] = {"skipped": "fully masked"}
                continue
            thresholds = _thresholds_of(state, source)
            gate = gate_floor(search_config, thresholds)
            floor = (
                thresholds[-1] - DEFAULT_FLOOR_OFFSET
                if config.peak_floor is None
                else float(config.peak_floor)
            )
            if records.get(name):
                if floors[name] > floor + 1e-9:
                    raise ValueError(
                        f"'{config.peaks_path}' holds {name}'s peaks from z "
                        f"{floors[name]:.3f}, above the floor {floor:.3f}; set "
                        "overwrite: true to search the copies again"
                    )
                floor = floors[name]
            area = _valid_area(state, source, config.border)
            hypotheses = _hypotheses(state)
            prediction = None
            refined = None
            if volume is not None:
                prediction = _predict(config, state, source, area, volume)
                if config.refine is not None:
                    refined = _predict_refined(
                        config, state, entry, source, volume, prediction
                    )
            del source

            peaks = None
            if config.empirical:
                floors[name] = floor
                copies = records.setdefault(name, {})
                wanted = [seed + index * config.seed_stride for seed in config.seeds]
                for s in wanted:
                    if s not in copies:
                        copies[s] = _search_copy(config, state, entry, s, floor)
                        _write_peaks(
                            config,
                            search_config,
                            records,
                            _peak_settings(config, floors),
                        )
                peaks = null_peaks(
                    {s: copies[s] for s in wanted}, floor, config.peak_radius
                )
            summary["micrographs"][name] = _record(
                config,
                thresholds=thresholds,
                gate=gate,
                area=area,
                hypotheses=hypotheses,
                floor=floor,
                peaks=peaks,
                copies=records.get(name, {}),
                prediction=prediction,
                refined=refined,
                report=report,
            )
    finally:
        state.release()
    summary["config"] = config.model_dump(mode="json")
    config.summary_path.write_text(json.dumps(summary, indent=1))
    return summary


def _record(
    config: NullConfig,
    *,
    thresholds: list[float],
    gate: dict[str, Any],
    area: int,
    hypotheses: int,
    floor: float,
    peaks: NullPeaks | None,
    copies: dict[int, _Copy],
    prediction: AnalyticNullPrediction | None,
    report: dict[str, Any] | None,
    refined: RefinedNullPrediction | None = None,
) -> dict[str, Any]:
    """One micrograph's entry of ``null_summary.json``."""
    record: dict[str, Any] = {
        "thresholds": thresholds,
        "gate": gate,
        "valid_area": area,
        "hypotheses": hypotheses,
        "peak_floor": floor,
        "cuts": _cuts(
            config, peaks, prediction, area, hypotheses, gate["floor"], refined
        ),
    }
    if report is not None:
        record["normalization"] = report
    if peaks is not None:
        record["copies"] = [
            {
                "seed": s,
                "peaks": int(copies[s].z.size),
                "max_z": float(copies[s].z.max(initial=-np.inf)),
                "valid_area": copies[s].valid_area,
                "seconds": copies[s].seconds,
            }
            for s in peaks.seeds
        ]
        record["rate"] = _rate_table(peaks, config.rate_step)
    if prediction is not None:
        record["analytic"] = prediction.as_dict()
    if refined is not None:
        record["analytic_refined"] = refined.as_dict()
    return record
