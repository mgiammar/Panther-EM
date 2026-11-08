"""Program ``calibrate``: the noise model's error tables of each feature selection.

Writes ``tables_<tag>.h5`` per tag into ``output_dir`` (the search's ``tables``)
and ``calibration.json``: each tag's cheap truncation errors and the rank
suggestions asked for (both need a decomposition with recorded block energies).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

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


class SuggestTarget(Model):
    """One call of :func:`~panther_em.approximation.suggest_rectangles`."""

    max_error: float | None = Field(default=None, gt=0)
    max_components: int | None = Field(default=None, ge=1)
    max_k_extent: int | None = Field(default=None, ge=1)


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
        ``tag -> rectangles``: one error table per feature selection (a stage's).
    suggest : list of SuggestTarget
        Rank suggestions to report in ``calibration.json``.
    pad, batch : int
        Spectra padding factor and template rows per batch of the tables.
    device : str
        Compute device. Default ``"cpu"``.
    """

    program: ClassVar[str] = "calibrate"

    volume: Path
    decomposition: Path
    whitening: Path
    output_dir: Path
    tables: dict[str, list[Rectangle]] = Field(default_factory=dict)
    suggest: list[SuggestTarget] = Field(default_factory=list)
    pad: int = Field(default=2, ge=1)
    batch: int = Field(default=128, ge=1)
    device: str = "cpu"

    @model_validator(mode="after")
    def _something_to_do(self) -> CalibrateConfig:
        if not self.tables and not self.suggest:
            raise ValueError("give tables, suggest targets or both")
        if any(not rects for rects in self.tables.values()):
            raise ValueError("every table needs at least one rectangle")
        return self


def table_path(output_dir: Path, tag: str) -> Path:
    """Where :func:`run` writes the table of ``tag``."""
    return Path(output_dir) / f"tables_{tag}.h5"


def run(config: CalibrateConfig) -> dict[str, PerSpatialFrequencyErrorTable]:
    """Compute and write the tables (and the report); returns the tables by tag."""
    result = DecompositionBase.load(config.decomposition)
    whitening = WhiteningFilter.load(config.whitening)
    has_energies = result.block_energy is not None and not result.is_complex_projection
    report: dict = {"tables": {}, "suggestions": []}

    tables = {}
    if config.tables:
        volume = read_volume(config.volume)
        reconstructor = Reconstructor(result, device=config.device)
        for tag, rects in config.tables.items():
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
            if has_energies:
                entry["truncation_error"] = truncation_error(result, rects).summary()
            report["tables"][tag] = entry

    if config.suggest:
        if not has_energies:
            raise ValueError(
                "rank suggestions need a decomposition with recorded block energies"
            )
        spectrum = energy_spectrum(result)
        for target in config.suggest:
            candidates = suggest_rectangles(spectrum, **target.model_dump())
            report["suggestions"].append(
                {
                    "target": target.model_dump(),
                    "candidates": [
                        {
                            "rectangle": list(c.rectangle),
                            "components": c.components,
                            "rho": c.rho,
                            "rho_k": c.rho_k,
                            "rho_m": c.rho_m,
                        }
                        for c in candidates
                    ],
                }
            )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration.json").write_text(json.dumps(report, indent=1))
    return tables
