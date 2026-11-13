"""Leopard-EM interface: SVD search results as Leopard-EM inputs (pinned to v1.3.1).

Every coupling to Leopard-EM lives in this module, including two private names its
v1.3.1 has no public equivalent for (``_TENSOR_NAMES`` and
``ParticleStackHDF5._set_dataframe``); a later Leopard-EM interface replaces them
here.

* :func:`write_match_template_maps` adds Leopard-EM's eight match-template maps to a
  search result file (group ``tensors``, the layout of ``MatchTemplateResultHDF5``),
  so the file is also a Leopard-EM match-template result: a particle stack's
  ``*_path`` columns can point at it.
* :func:`pick_particles` keeps one pixel per particle (Leopard-EM's peak finder).
* :func:`particle_table` and :func:`write_particle_stack` build the
  ``ParticleStackHDF5`` that Leopard-EM's refinement reads.

Scales: Leopard-EM's refinement scores each pose by ``z = (cc - mu) / sigma``, where
``cc`` is its own freshly computed correlation (unit-variance background) and
``mu``, ``sigma`` are read from the exported ``correlation_average`` and
``correlation_variance`` (a standard deviation) maps. Panther-EM's correlations carry
an extra factor ``c`` (the template norm, ~185 on xenon_216), so unscaled maps give
``z = cc / (c sigma) - mu / sigma``: the correlation term is suppressed by ``c`` and
the argmax over the shift window follows ``mu / sigma``, pulling refined positions to
the window's edge. ``mip``, ``correlation_average`` and ``correlation_variance`` are
therefore divided by :func:`correlation_scale`, the median ``sigma``, which restores
``mu ~ 0`` and ``sigma ~ 1``. ``scaled_mip`` is a z-score and needs no scale.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import h5py

# PEP 810: make the leopard_em imports `lazy import`s once requires-python >= 3.15
# (~0.4 s, paid by every program though only export and search write for Leopard-EM).
import leopard_em
import torch
from leopard_em.analysis.match_template_peaks import (
    MatchTemplatePeaks,
    match_template_peaks_to_dataframe,
)
from leopard_em.analysis.zscore_metric import find_peaks_from_zscore
from leopard_em.pydantic_models.data_structures import OpticsGroup
from leopard_em.pydantic_models.data_structures.particle_stack import (
    ParticleStackHDF5,
)
from leopard_em.pydantic_models.formats import (
    HDF5_TENSORS_GROUP,
    LARGE_STD_PADDING,
    MATCH_TEMPLATE_DF_COLUMN_ORDER,
    STATISTIC_MAP_PATH_TO_HDF5_DATASET,
)
from leopard_em.pydantic_models.results.match_template_result import (
    _TENSOR_NAMES,
)

from panther_em._hdf5 import create_dataset
from panther_em.search import PixelLabel

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from panther_em.search import SearchResult

EXTRA_COLUMNS = ("panther_label", "panther_score", "panther_stage")
"""Panther-EM columns appended to Leopard-EM's (refinement does not keep them)."""


# --------------------------------------------------------------------------- #
# Match-template maps
# --------------------------------------------------------------------------- #


def _hypotheses(result: SearchResult, n_psi: int) -> int:
    space = result.space
    return space.n_defocus * space.n_filters * space.n_orientations * n_psi


def correlation_scale(result: SearchResult) -> float:
    """Median standard deviation of ``result``'s correlations over searched pixels.

    Estimated factor from approximated correlation units in Panther-EM and the
    normalized cross-correlation values in Leopard-EM.
    """
    searched = (result.last_stage[0] >= 0).cpu()
    std = result.variance[0].detach().cpu().double().clamp_min(0.0).sqrt()[searched]
    std = std[torch.isfinite(std)]
    scale = float(std.median()) if std.numel() else float("nan")
    if not scale > 0.0:
        raise ValueError("the result has no searched pixel with a positive variance")
    return scale


def match_template_maps(result: SearchResult) -> dict[str, torch.Tensor]:
    """Leopard-EM's eight ``(H - h + 1, W - w + 1)`` float32 CPU maps of ``result``."""
    if result.geometry.n_images != 1:
        raise ValueError("a Leopard-EM result holds one micrograph")
    decoded = result.decode()
    searched = (result.last_stage[0] >= 0).cpu()
    scale = correlation_scale(result)

    def plane(values: torch.Tensor | None) -> torch.Tensor:
        if values is None:
            return torch.zeros(searched.shape, dtype=torch.float32)
        return values[0].detach().to("cpu", torch.float32)

    maps = {  # correlations in units of the scale, Leopard-EM's
        "mip": plane(result.mip) / scale,
        "scaled_mip": plane(result.zscore),
        "correlation_average": plane(result.mean) / scale,
        "correlation_variance": plane(result.variance).clamp_min(0.0).sqrt() / scale,
        "orientation_psi": plane(decoded["psi"]),
        "orientation_theta": plane(decoded["theta"]),
        "orientation_phi": plane(decoded["phi"]),
        "relative_defocus": plane(decoded["defocus"]),
    }
    out = {}
    for name, values in maps.items():
        fill = LARGE_STD_PADDING if name == "correlation_variance" else 0.0
        valid = searched & torch.isfinite(values)
        out[name] = torch.where(valid, values, torch.full_like(values, fill))
    return out


def write_match_template_maps(
    path: str | Path,
    result: SearchResult,
    maps: dict[str, torch.Tensor] | None = None,
) -> None:
    """Add Leopard-EM's match-template maps and attributes to the file at ``path``."""
    if maps is None:
        maps = match_template_maps(result)
    n_psi = max(result.n_psi_of_stage)
    space = result.space
    with h5py.File(path, "a") as f:
        f.attrs["leopard_em_version"] = leopard_em.__version__
        f.attrs["total_projections"] = _hypotheses(result, n_psi)
        f.attrs["total_orientations"] = space.n_orientations * space.n_filters * n_psi
        f.attrs["total_defocus"] = space.n_defocus
        f.attrs["panther_em_correlation_scale"] = correlation_scale(result)
        if HDF5_TENSORS_GROUP in f:
            del f[HDF5_TENSORS_GROUP]
        group = f.create_group(HDF5_TENSORS_GROUP)
        for name in _TENSOR_NAMES:
            create_dataset(group, name, maps[name].numpy())


# --------------------------------------------------------------------------- #
# Particles
# --------------------------------------------------------------------------- #


def pick_particles(
    result: SearchResult, radius: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(pos_y, pos_x)`` of one pixel per accepted particle."""
    label = result.label[0].cpu()
    zscore = result.zscore[0].detach().to("cpu", torch.float32)
    accepted = label == int(PixelLabel.ACCEPTED)
    candidate = accepted | (label == int(PixelLabel.AMBIGUOUS))
    masked = torch.where(candidate, zscore, torch.full_like(zscore, float("-inf")))
    pos_y, pos_x = find_peaks_from_zscore(masked, float("-inf"), radius)
    if pos_y.numel() == 0 or not bool(accepted.any()):
        empty = torch.zeros(0, dtype=torch.long)
        return empty, empty
    peaks = torch.stack([pos_y, pos_x], dim=1).to(torch.float32)
    owner = torch.cdist(torch.nonzero(accepted).to(torch.float32), peaks).argmin(1)
    keep = torch.zeros(len(peaks), dtype=torch.bool)
    keep[owner] = True
    return pos_y[keep].long(), pos_x[keep].long()


def particle_table(
    result: SearchResult,
    *,
    result_path: str | Path,
    micrograph_path: str | Path,
    template_path: str | Path | None,
    optics: dict[str, Any],
    radius: float,
) -> pd.DataFrame:
    """Leopard-EM's particle table of ``result``'s particles (:func:`pick_particles`).

    Parameters
    ----------
    result : SearchResult
        One micrograph's search result.
    result_path : str or Path
        Its file.
    micrograph_path : str or Path
        The searched micrograph (MRC).
    template_path : str or Path, optional
        The template volume (MRC); empty when unknown.
    optics : dict
        Leopard-EM ``OpticsGroup`` fields (``label``, ``pixel_size``, ``voltage``,
        ``defocus_u``, ``defocus_v``, ``astigmatism_angle``, ...).
    radius : float
        Minimum distance between particles, pixels.
    """
    maps = match_template_maps(result)
    pos_y, pos_x = pick_particles(result, radius)
    peaks = MatchTemplatePeaks(
        pos_y=pos_y,
        pos_x=pos_x,
        mip=maps["mip"][pos_y, pos_x],
        scaled_mip=maps["scaled_mip"][pos_y, pos_x],
        psi=maps["orientation_psi"][pos_y, pos_x],
        theta=maps["orientation_theta"][pos_y, pos_x],
        phi=maps["orientation_phi"][pos_y, pos_x],
        relative_defocus=maps["relative_defocus"][pos_y, pos_x],
        correlation_mean=maps["correlation_average"][pos_y, pos_x],
        correlation_variance=maps["correlation_variance"][pos_y, pos_x],
        total_correlations=0,
    )
    df = match_template_peaks_to_dataframe(peaks)

    stage = result.last_stage[0].cpu()[pos_y, pos_x]
    df["total_correlations"] = [
        _hypotheses(result, result.n_psi_of_stage[int(s)]) for s in stage
    ]
    box_h, box_w = result.geometry.template_shape
    group = OpticsGroup(**optics)
    df["pos_y_img"] = df["pos_y"] + box_h // 2
    df["pos_x_img"] = df["pos_x"] + box_w // 2
    df["pos_y_img_angstrom"] = df["pos_y_img"] * group.pixel_size
    df["pos_x_img_angstrom"] = df["pos_x_img"] * group.pixel_size
    for name in (
        "defocus_u",
        "defocus_v",
        "astigmatism_angle",
        "pixel_size",
        "voltage",
        "spherical_aberration",
        "amplitude_contrast_ratio",
        "phase_shift",
        "ctf_B_factor",
    ):
        df[name] = getattr(group, name)
    df["refined_pixel_size"] = group.pixel_size
    for name in ("even_zernikes", "odd_zernikes"):
        value = getattr(group, name)
        df[name] = [None if value is None else json.dumps(value)] * len(df)
    df["mag_matrix"] = [group.mag_matrix] * len(df)
    df["micrograph_path"] = str(micrograph_path)
    df["template_path"] = "" if template_path is None else str(template_path)
    for column in STATISTIC_MAP_PATH_TO_HDF5_DATASET:
        df[column] = str(result_path)

    label = result.label[0].cpu()[pos_y, pos_x]
    df["panther_label"] = [PixelLabel(int(v)).name for v in label]
    df["panther_score"] = result.score[0].detach().cpu()[pos_y, pos_x].numpy()
    df["panther_stage"] = stage.numpy()
    df["particle_index"] = range(len(df))
    return df.reindex(columns=[*MATCH_TEMPLATE_DF_COLUMN_ORDER, *EXTRA_COLUMNS])


def write_particle_stack(
    table: pd.DataFrame,
    path: str | Path,
    *,
    template_shape: tuple[int, int],
    margin: int,
) -> ParticleStackHDF5:
    """Write ``table`` as a Leopard-EM ``ParticleStackHDF5`` at ``path``."""
    h, w = template_shape
    stack = ParticleStackHDF5(
        hdf5_path=str(path),
        skip_df_load=True,
        extracted_box_size=(h + margin, w + margin),
        original_template_size=(h, w),
    )
    stack._set_dataframe(table.reset_index(drop=True))  # no public setter in v1.3.1
    stack.to_hdf5()
    return stack
