"""Program ``calibrate``: the noise model's error tables of each feature selection.

Writes ``tables_<tag>.h5`` per tag into ``output_dir`` (the search's ``tables``)
and ``calibration.json``: each tag's rectangles and cheap truncation errors, and the
rank suggestions asked for (both need a decomposition with recorded block energies).

A tag's feature selection is either explicit rectangles or an error target
(:class:`TableTarget`): the smallest leading-corner rectangle ``(0, 0, K, M)`` whose
pooled truncation error meets it, so the selection follows the template. The table
records the rectangle it was built for, which a search stage with
``rectangles: auto`` reads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from pydantic import Field, model_validator

from panther_em.approximation import (
    PerSpatialFrequencyErrorTable,
    Reconstructor,
    compute_error_tables,
    energy_spectrum,
    suggest_rectangles,
    truncation_error,
)
from panther_em.decomposition import DecompositionBase
from panther_em.filters import WhiteningFilter
from panther_em.managers._common import (
    Model,
    ProgramConfig,
    Rectangle,
    read_volume,
    save_output,
)

if TYPE_CHECKING:
    from panther_em.approximation import EnergySpectrum, RectangleCandidate


class SuggestTarget(Model):
    """One call of :func:`~panther_em.approximation.suggest_rectangles`."""

    max_error: float | None = Field(default=None, gt=0)
    max_components: int | None = Field(default=None, ge=1)
    max_k_extent: int | None = Field(default=None, ge=1)


class TableTarget(Model):
    """A table's feature selection chosen by an error target.

    The smallest leading-corner rectangle ``(0, 0, K, M)`` (fewest kept cells
    ``K * M``; the lower error on a tie) whose pooled truncation error ``rho``
    (:func:`~panther_em.approximation.energy_spectrum`) is at most ``max_error``,
    among those with at most ``max_components`` cells and ``K <= max_k_extent``: the
    first candidate of :func:`~panther_em.approximation.suggest_rectangles`.
    """

    max_error: float = Field(gt=0)
    max_components: int | None = Field(default=None, ge=1)
    max_k_extent: int | None = Field(default=None, ge=1)


Selection = list[Rectangle] | TableTarget
"""A table's feature selection: explicit rectangles or an error target."""


class CalibrateConfig(ProgramConfig):
    """Configuration of :func:`run`.

    Parameters
    ----------
    volume : Path
        The template volume (MRC) the decomposition was made from.
    decomposition, whitening : Path
        The decomposition (built under the whitening's ``W_avg * band``) and the
        whitening filter.
    output_dir : Path
        Where ``tables_<tag>.h5`` and ``calibration.json`` are written.
    tables : dict
        ``tag -> rectangles or target``: one error table per feature selection (a
        stage's). A list of rectangles is kept as given; a :class:`TableTarget`
        (``{max_error: 0.31}``, optionally with ``max_components`` and
        ``max_k_extent``) picks the smallest leading-corner rectangle that meets it
        (needs recorded block energies) and ``calibration.json`` reports the pick.
    suggest : list of SuggestTarget
        Rank suggestions to report in ``calibration.json``.
    pad, batch : int
        Spectra padding factor and template rows per batch of the tables.
    device : str
        Compute device. Default ``"cpu"``.

    Examples
    --------
    One stage-1 table sized by the template, one fixed stage-2 table::

        tables:
          stage1: {max_error: 0.31}
          stage2: [[0, 0, 96, 80]]
    """

    program: ClassVar[str] = "calibrate"

    volume: Path
    decomposition: Path
    whitening: Path
    output_dir: Path
    tables: dict[str, Selection] = Field(default_factory=dict)
    suggest: list[SuggestTarget] = Field(default_factory=list)
    pad: int = Field(default=2, ge=1)
    batch: int = Field(default=128, ge=1)
    device: str = "cpu"

    @model_validator(mode="after")
    def _require_tables_or_suggestions(self) -> CalibrateConfig:
        if not self.tables and not self.suggest:
            raise ValueError("give tables, suggest targets or both")
        if any(not s for s in self.tables.values() if isinstance(s, list)):
            raise ValueError("every table needs at least one rectangle")
        return self


def table_path(output_dir: Path, tag: str) -> Path:
    """Where :func:`run` writes the table of ``tag``."""
    return Path(output_dir) / f"tables_{tag}.h5"


def choose_rectangle(
    spectrum: EnergySpectrum, target: TableTarget
) -> RectangleCandidate:
    """The smallest leading-corner rectangle meeting ``target`` (see TableTarget)."""
    candidates = suggest_rectangles(spectrum, **target.model_dump())
    if not candidates:
        raise ValueError(
            f"no leading-corner rectangle meets the target {target.model_dump()} "
            "within the decomposition's stored k_max and eig_max"
        )
    return candidates[0]


def _candidate(c: RectangleCandidate) -> dict:
    return {
        "rectangle": list(c.rectangle),
        "components": c.components,
        "rho": c.rho,
        "rho_k": c.rho_k,
        "rho_m": c.rho_m,
    }


def run(config: CalibrateConfig) -> dict[str, PerSpatialFrequencyErrorTable]:
    """Compute and write the tables (and the report); returns the tables by tag."""
    result = DecompositionBase.load(config.decomposition)
    whitening = WhiteningFilter.load(config.whitening)
    has_energies = result.block_energy is not None and not result.is_complex_projection
    report: dict = {"tables": {}, "suggestions": []}

    spectrum: EnergySpectrum | None = None
    targets = {t: s for t, s in config.tables.items() if isinstance(s, TableTarget)}
    if targets or config.suggest:
        if not has_energies:
            raise ValueError(
                "error targets and rank suggestions need a decomposition with "
                "recorded block energies"
            )
        spectrum = energy_spectrum(result)

    # resolve every target before the (expensive) tables
    selections: dict[str, list[Rectangle]] = {}
    chosen: dict[str, RectangleCandidate] = {}
    for tag, selection in config.tables.items():
        if isinstance(selection, TableTarget):
            assert spectrum is not None
            try:
                chosen[tag] = choose_rectangle(spectrum, selection)
            except ValueError as error:
                raise ValueError(f"table {tag!r}: {error}") from None
            selections[tag] = [chosen[tag].rectangle]
        else:
            selections[tag] = list(selection)

    tables = {}
    if selections:
        volume = read_volume(config.volume)
        reconstructor = Reconstructor(result, device=config.device)
        for tag, rects in selections.items():
            table = compute_error_tables(
                volume,
                reconstructor,
                rects,
                whitening,
                pad=config.pad,
                batch=config.batch,
            )
            save_output(table.save, table_path(config.output_dir, tag), config)
            tables[tag] = table
            entry: dict = {"rectangles": [list(r) for r in rects]}
            if tag in chosen:
                entry["target"] = targets[tag].model_dump()
                entry["chosen"] = _candidate(chosen[tag])
            if has_energies:
                entry["truncation_error"] = truncation_error(result, rects).summary()
            report["tables"][tag] = entry

    for target in config.suggest:
        assert spectrum is not None
        candidates = suggest_rectangles(spectrum, **target.model_dump())
        report["suggestions"].append(
            {
                "target": target.model_dump(),
                "candidates": [_candidate(c) for c in candidates],
            }
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration.json").write_text(json.dumps(report, indent=1))
    return tables
