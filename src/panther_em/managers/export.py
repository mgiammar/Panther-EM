"""Program ``export``: search results to a Leopard-EM particle table (CSV).

One row per pixel whose final label is among ``labels`` (default the accepted
ones; the absorb radius leaves one accepted pixel per particle). Columns follow
Leopard-EM's match-template table: positions are the template box's top-left on
the correlation grid (``pos_x``, ``pos_y``) and its center in the image
(``*_img``), angles the best hypothesis' ZYZ Euler angles in degrees,
``relative_defocus`` the searched offset in Angstroms, ``scaled_mip`` the z-score
and ``correlation_variance`` the standard deviation (as Leopard-EM's). The optics,
the micrograph and the template volume come from the configs the result files
record; ``panther_label`` and ``panther_score`` are added at the end.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import ClassVar, Literal

import torch
from pydantic import Field

from panther_em.managers._common import ProgramConfig, read_attr, read_config
from panther_em.managers.decompose import DecomposeConfig
from panther_em.managers.search import SearchConfig
from panther_em.search import PixelLabel, SearchResult

COLUMNS = (
    "particle_index",
    "mip",
    "scaled_mip",
    "correlation_mean",
    "correlation_variance",
    "total_correlations",
    "pos_x",
    "pos_y",
    "pos_x_img",
    "pos_y_img",
    "pos_x_img_angstrom",
    "pos_y_img_angstrom",
    "phi",
    "theta",
    "psi",
    "relative_defocus",
    "defocus_u",
    "defocus_v",
    "astigmatism_angle",
    "pixel_size",
    "refined_pixel_size",
    "voltage",
    "spherical_aberration",
    "amplitude_contrast_ratio",
    "phase_shift",
    "ctf_B_factor",
    "micrograph_path",
    "template_path",
    "panther_label",
    "panther_score",
)

LabelName = Literal["ACCEPTED", "AMBIGUOUS", "ABSORBED", "REJECTED"]


class ExportConfig(ProgramConfig):
    """Configuration of :func:`run`.

    Parameters
    ----------
    results : list of Path
        Result files written by the ``search`` program.
    output : Path
        The CSV table.
    labels : list of str
        Final labels to export. Default ``ACCEPTED`` only.
    """

    program: ClassVar[str] = "export"

    results: list[Path] = Field(min_length=1)
    output: Path
    labels: tuple[LabelName, ...] = ("ACCEPTED",)


def _rows(path: Path, labels: tuple[LabelName, ...]) -> list[dict]:
    search = read_config(path, SearchConfig)
    name = read_attr(path, "micrograph")
    if search is None or name is None:
        raise ValueError(f"'{path}' was not written by the search program")
    entry = next(m for m in search.micrographs if m.key == name)
    decomposition = read_config(search.decomposition, DecomposeConfig)
    template_path = "" if decomposition is None else str(decomposition.volume)

    result = SearchResult.load(path)
    label = result.label[0]
    keep = torch.zeros_like(label, dtype=torch.bool)
    for lab in labels:
        keep |= label == int(PixelLabel[lab])
    rows, cols = (i.tolist() for i in torch.nonzero(keep, as_tuple=True))
    decoded = {k: v[0] for k, v in result.decode().items() if v is not None}
    box_h, box_w = result.geometry.template_shape
    space = result.space
    optics = search.optics

    out = []
    for r, c in zip(rows, cols, strict=True):
        stage = int(result.last_stage[0, r, c])
        n_psi = result.n_psi_of_stage[stage]
        y_img, x_img = r + box_h // 2, c + box_w // 2

        def at(name: str, r: int = r, c: int = c) -> float:
            return float(decoded[name][r, c]) if name in decoded else float("nan")

        out.append(
            {
                "mip": float(result.mip[0, r, c]),
                "scaled_mip": float(result.zscore[0, r, c]),
                "correlation_mean": float(result.mean[0, r, c]),
                "correlation_variance": float(result.variance[0, r, c]) ** 0.5,
                "total_correlations": space.n_defocus
                * space.n_filters
                * space.n_orientations
                * n_psi,
                "pos_x": c,
                "pos_y": r,
                "pos_x_img": x_img,
                "pos_y_img": y_img,
                "pos_x_img_angstrom": x_img * optics.pixel_size,
                "pos_y_img_angstrom": y_img * optics.pixel_size,
                "phi": at("phi"),
                "theta": at("theta"),
                "psi": at("psi"),
                "relative_defocus": at("defocus"),
                "defocus_u": entry.defocus_u,
                "defocus_v": entry.defocus_v,
                "astigmatism_angle": entry.astigmatism_angle,
                "pixel_size": optics.pixel_size,
                "refined_pixel_size": optics.pixel_size,
                "voltage": optics.voltage,
                "spherical_aberration": optics.spherical_aberration,
                "amplitude_contrast_ratio": optics.amplitude_contrast_ratio,
                "phase_shift": optics.phase_shift,
                "ctf_B_factor": 0.0,
                "micrograph_path": str(entry.path),
                "template_path": template_path,
                "panther_label": PixelLabel(int(label[r, c])).name,
                "panther_score": float(result.score[0, r, c]),
            }
        )
    return out


def run(config: ExportConfig) -> int:
    """Write the particle table; returns the number of particles."""
    rows = [row for path in config.results for row in _rows(path, config.labels)]
    for i, row in enumerate(rows):
        row["particle_index"] = i
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)
