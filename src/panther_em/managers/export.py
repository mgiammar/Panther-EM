"""Program ``export``: search results to a Leopard-EM particle stack (HDF5).

One particle per accepted detection (:func:`~.leopard.pick_particles`: the z-score
maximum of each accepted or ambiguous region, ``peak_radius`` apart) goes into one
Leopard-EM ``ParticleStackHDF5`` that its template refinement reads directly. The
table holds no images and no statistic crops: Leopard-EM extracts the particles from
each row's ``micrograph_path`` and the statistic maps from its ``*_path`` columns,
the search result file (which the ``search`` program made a Leopard-EM
match-template result too).

The optics, the micrograph and the template volume come from the configs the result
files record. ``scaled_mip`` is Panther-EM's z-score; ``mip`` and the correlation mean
and standard deviation are Panther-EM's divided by each micrograph's median correlation
standard deviation, which brings them near Leopard-EM's unit-noise scale (see
:mod:`~.leopard`), as its refinement's pose selection and ``refined_scaled_mip`` need.
The ``panther_*`` columns are kept in the stack but not in Leopard-EM's refinement
output.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd
from leopard_em.pydantic_models.data_structures.particle_stack import (
    ParticleStackHDF5,
)
from pydantic import Field

from panther_em.managers import leopard
from panther_em.managers._common import (
    ProgramConfig,
    read_attr,
    read_config,
    save_output,
)
from panther_em.managers.decompose import DecomposeConfig
from panther_em.managers.search import SearchConfig
from panther_em.search import SearchResult


class ExportConfig(ProgramConfig):
    """Configuration of :func:`run`.

    Parameters
    ----------
    results : list of Path
        Result files written by the ``search`` program.
    output : Path
        The particle stack (HDF5).
    peak_radius : float
        Minimum distance between two particles, pixels. Default 10 (Leopard-EM's).
    box_margin : int
        Extracted particle boxes are this many pixels larger than the template (the
        room refinement has to shift particles). Default 16.
    """

    program: ClassVar[str] = "export"

    results: list[Path] = Field(min_length=1)
    output: Path
    peak_radius: float = Field(default=10.0, gt=0)
    box_margin: int = Field(default=16, ge=0)


def table(path: Path, radius: float) -> tuple[pd.DataFrame, tuple[int, int]]:
    """The particle table of one result file and its template box."""
    search = read_config(path, SearchConfig)
    name = read_attr(path, "micrograph")
    if search is None or name is None:
        raise ValueError(f"'{path}' was not written by the search program")
    entry = next(m for m in search.micrographs if m.key == name)
    decomposition = read_config(search.decomposition, DecomposeConfig)
    result = SearchResult.load(path)
    optics = search.optics
    rows = leopard.particle_table(
        result,
        result_path=path,
        micrograph_path=entry.path,
        template_path=None if decomposition is None else decomposition.volume,
        optics={
            "label": name,
            "pixel_size": optics.pixel_size,
            "voltage": optics.voltage,
            "spherical_aberration": optics.spherical_aberration,
            "amplitude_contrast_ratio": optics.amplitude_contrast_ratio,
            "phase_shift": optics.phase_shift,
            "defocus_u": entry.defocus_u,
            "defocus_v": entry.defocus_v,
            "astigmatism_angle": entry.astigmatism_angle,
        },
        radius=radius,
    )
    h, w = result.geometry.template_shape
    return rows, (int(h), int(w))


def run(config: ExportConfig) -> ParticleStackHDF5:
    """Write the particle stack; returns it (as read back from the file)."""
    tables, boxes = zip(
        *(table(Path(p), config.peak_radius) for p in config.results), strict=True
    )
    if len(set(boxes)) != 1:
        raise ValueError(f"the results come from templates of different boxes {boxes}")
    rows = pd.concat(tables, ignore_index=True)
    if rows.empty:
        raise ValueError("the results hold no accepted particle")
    rows["particle_index"] = range(len(rows))

    def write(path: Path) -> None:
        leopard.write_particle_stack(
            rows, path, template_shape=boxes[0], margin=config.box_margin
        )

    save_output(write, config.output, config)
    return ParticleStackHDF5(hdf5_path=str(config.output))
