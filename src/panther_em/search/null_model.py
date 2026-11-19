r"""Analytic null of a Panther-EM search: its null peak rate without a null search.

A phase-randomised micrograph keeps the micrograph's Fourier amplitudes and draws its
phases at random, so the correlation of the search image with the search templates is
a Gaussian random field over position and hypothesis whose covariance is fixed by the
templates and the image power spectrum. From that covariance Leopard-EM's engine-
agnostic analytic null (:mod:`leopard_em.analysis.analytic_null`) predicts the
expected number of distinct null peaks at or above each z-score per micrograph,
:math:`\lambda(t)`, and so the threshold for a target number of false positives:

1. lattice local maxima over position, in-plane angle, view direction and defocus
   plane (Worsley's discrete local maxima), from the field's correlations one and two
   grid steps along every axis;
2. the peak finder's suppression radius (crowding);
3. the per-pixel z-score (Panther's z is ``(max - mean) / sd`` over every hypothesis
   at the pixel), through ``kappa``, the mean squared correlation of random
   hypotheses.

This module supplies Panther's side of the calculation, from Panther's own pieces:

* the image: the prepared micrograph's search image (``W_avg * C_i^2``, zero mean,
  unit variance; :meth:`~panther_em.search.micrograph.Micrograph.prepare`);
* the templates (:class:`TemplateProjector`): the volume's projection exactly as the
  decomposition makes it (:func:`~panther_em.decomposition.projections.
  precompute_volume_dft`, :func:`~panther_em.decomposition.projections.
  project_from_precomputed_dft`), under the whitening's template filter
  ``W_avg * band``, normalized to unit variance before the CTF as the decomposition
  does, and masked to the polar grid's support disk;
* the CTF: the micrograph's own CTF planes (which carry the after-CTF plane factors
  ``b_d``), sampled on the lag grid of the template box (:func:`lag_grid_planes`).
  Panther applies the CTF to the image; for the covariance it is the same to apply
  it to the template;
* after-CTF normalization: every sampled template gets the row factor
  :math:`a = (\overline{e^{\mathrm{ref}}} / c)^{-1/2}` that
  :class:`~panther_em.search.normalization.AfterCtfNormalization` gives a template
  row: its noise variance under the run's reference noise (white noise under the
  ring-averaged ``CTF^2`` of every searched plane,
  :func:`~panther_em.search.normalization.reference_noise`), averaged over the
  planes (the constant ``c`` cancels in the z-score). The plane factors ``b_d`` are
  the micrograph's own, carried by its CTF planes. Computing ``a`` for each sampled
  template, rather than interpolating the grid rows' factors, equalizes the model's
  templates as Panther equalizes its rows on any grid;
* the grid: HEALPix ``nside`` x ``n_psi`` (:func:`lattice_neighbours`): in-plane
  steps of ``2 pi / n_psi`` about the body z axis and view steps of one HEALPix
  pixel, ``sqrt(pi / 3) / nside``, about the body x and y axes (the pixel lattice
  treated as locally square), each composed on the right of the rotation. A HEALPix
  x psi grid is uniform on SO(3), so its points are sampled as Haar-random
  rotations.

SVD rank truncation and the polar warp are not modelled (the field is that of the
exact templates); nor is the stage-1 gate, so the prediction is for a complete
search.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import healpy as hp
import numpy as np
import roma
import torch
from leopard_em.analysis.analytic_null import (
    AnalyticNullPrediction,
    AxisNeighbours,
    PowerWeighting,
    RefinedNullPrediction,
    RefinementWindow,
    estimate_kappa,
    estimate_mean_template,
    predict_null_rate,
    predict_refined_null_rate,
    sample_lattice,
    sample_refinement_gain,
)

from panther_em.decomposition.projections import (
    apply_fourier_filters,
    precompute_volume_dft,
    project_from_precomputed_dft,
)
from panther_em.filters.curves import curve_to_grid
from panther_em.orientations import healpix_ipix

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from panther_em.search.micrograph import Micrograph

__all__ = [
    "DEFAULT_THRESHOLDS",
    "SearchTemplates",
    "TemplateProjector",
    "haar_rotations",
    "healpix_nside",
    "lag_grid_planes",
    "lattice_neighbours",
    "predict_null",
    "predict_refined_null",
    "reference_grid",
]

DEFAULT_THRESHOLDS = np.round(np.arange(2.0, 10.0, 0.01), 3)


def lag_grid_planes(
    planes: torch.Tensor,
    image_shape: tuple[int, int],
    lag_shape: tuple[int, int],
) -> torch.Tensor:
    """Real rfft-layout planes of an image grid, sampled on a coarser grid.

    Plane values at the frequencies of the ``lag_shape`` rfft grid, by bilinear
    interpolation between the image grid's frequencies (rows periodic). Exact when
    ``lag_shape`` divides ``image_shape``.

    Parameters
    ----------
    planes : torch.Tensor
        ``(D, H, W // 2 + 1)`` real planes (e.g. a micrograph's CTF planes).
    image_shape : tuple[int, int]
        ``(H, W)``.
    lag_shape : tuple[int, int]
        ``(Ly, Lx)`` of the grid to sample on.

    Returns
    -------
    torch.Tensor
        ``(D, Ly, Lx // 2 + 1)`` float32, on ``planes``' device.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    ly, lx = int(lag_shape[0]), int(lag_shape[1])
    if planes.dim() != 3 or tuple(planes.shape[1:]) != (height, width // 2 + 1):
        raise ValueError(
            f"planes must be (D, {height}, {width // 2 + 1}), got {tuple(planes.shape)}"
        )
    device = planes.device
    # lag row i is frequency (i mod ly) / ly, image row i * H / ly (mod H); columns
    # j / lx, image column j * W / lx
    fy = torch.arange(ly, dtype=torch.float64) * height / ly
    fx = torch.arange(lx // 2 + 1, dtype=torch.float64) * width / lx
    y0 = torch.floor(fy).long()
    x0 = torch.floor(fx).long().clamp(max=width // 2)
    ty = (fy - y0).to(device, torch.float32)[:, None]
    tx = (fx - x0).to(device, torch.float32)[None, :]
    y1 = ((y0 + 1) % height).to(device)
    x1 = (x0 + 1).clamp(max=width // 2).to(device)
    y0, x0 = y0.to(device), x0.to(device)
    p = planes.to(torch.float32)
    rows0, rows1 = p[:, y0], p[:, y1]
    out = (1 - ty) * ((1 - tx) * rows0[..., x0] + tx * rows0[..., x1]) + ty * (
        (1 - tx) * rows1[..., x0] + tx * rows1[..., x1]
    )
    return out


def healpix_nside(
    phi_deg: np.ndarray | torch.Tensor, theta_deg: np.ndarray | torch.Tensor
) -> int:
    """``nside`` of a decomposition's orientations, which must be a HEALPix grid.

    Raises unless the ``(O,)`` orientations (degrees) are the complete set of HEALPix
    pixel centers at some ``nside``, the grid the analytic null models.
    """
    grid = healpix_ipix(phi_deg, theta_deg)
    if grid is None or 12 * grid[0] ** 2 != int(np.asarray(phi_deg).size):
        raise ValueError(
            "the analytic null models a complete HEALPix grid of orientations; the "
            "decomposition's orientations are not one"
        )
    return int(grid[0])


def reference_grid(
    frequencies: torch.Tensor,
    reference: torch.Tensor,
    box: tuple[int, int],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """The after-CTF reference noise on a template box's rfft grid.

    Parameters
    ----------
    frequencies : torch.Tensor
        ``(F,)`` the curves' frequency axis (the error table's).
    reference : torch.Tensor
        ``(D, F)`` reference noise of every plane of the run
        (:func:`~panther_em.search.normalization.reference_noise`).
    box : tuple[int, int]
        Template box ``(h, w)``.
    device : torch.device or str, optional
        Where the grid is returned.

    Returns
    -------
    torch.Tensor
        ``(D, h, w // 2 + 1)`` float32.
    """
    return torch.stack(
        [
            curve_to_grid(curve.to(torch.float32), frequencies, box, device=device)
            for curve in reference
        ]
    )


def haar_rotations(n: int, generator: torch.Generator) -> torch.Tensor:
    """``(n, 3, 3)`` float32 Haar-random rotations (normalized Gaussian quaternions)."""
    quat = torch.randn((n, 4), generator=generator, dtype=torch.float64)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    rotation: torch.Tensor = roma.unitquat_to_rotmat(quat)
    return rotation.to(torch.float32)


def _axis_rotation(axis: int, angle: float) -> torch.Tensor:
    vec = torch.zeros(3, dtype=torch.float64)
    vec[axis] = angle
    rotation: torch.Tensor = roma.rotvec_to_rotmat(vec)
    return rotation


def lattice_neighbours(
    nside: int, n_psi: int
) -> Callable[[torch.Tensor], list[AxisNeighbours]]:
    """Neighbours on Panther's HEALPix ``nside`` x ``n_psi`` grid.

    For rotations ``R`` (``(n, 3, 3)``), the rotations one and two steps away along
    each axis, composed on the right (``R @ step``): in-plane angle ``psi`` (body z,
    ``2 pi / n_psi``) and the view direction along body x and y (``view_x``,
    ``view_y``: one HEALPix pixel, ``sqrt(pi / 3) / nside``). Every axis is periodic.
    """
    steps = (
        ("psi", 2, 2.0 * math.pi / n_psi),
        ("view_x", 0, math.sqrt(math.pi / 3.0) / nside),
        ("view_y", 1, math.sqrt(math.pi / 3.0) / nside),
    )

    def neighbours(rotations: torch.Tensor) -> list[AxisNeighbours]:
        out = []
        for name, axis, step in steps:
            shifted = [
                rotations @ _axis_rotation(axis, m * step).to(rotations)
                for m in (1, -1, 2, -2)
            ]
            out.append(
                AxisNeighbours(
                    name=name, steps=(shifted[0], shifted[1], shifted[2], shifted[3])
                )
            )
        return out

    return neighbours


class TemplateProjector:
    """Panther's search templates at arbitrary rotations, on a lag grid.

    For rotations ``R``, the template of every searched plane ``d``:
    ``a(R) * b_d * CTF_d * m_R`` on the lag grid of ``weighting``, DC removed, where
    ``m_R`` is the volume's projection as the decomposition makes it (padded volume
    DFT, Fourier slice, the template filter, unit variance when the decomposition
    normalized its projections) masked to ``support``; ``CTF_d`` (which carries
    ``b_d``) is given on the lag grid; ``a(R)`` is the after-CTF row factor of
    ``m_R`` under ``reference`` (:meth:`row_scale`; 1 without a reference).

    Parameters
    ----------
    volume : torch.Tensor or numpy.ndarray
        ``(d, d, d)`` template volume.
    template_filter : torch.Tensor
        ``(d, d // 2 + 1)`` the decomposition's Fourier filter (``W_avg * band``).
    weighting : PowerWeighting
        The image's lag-grid power weighting (sets the lag grid).
    ctf : torch.Tensor
        ``(D, Ly, Lx // 2 + 1)`` the searched CTF planes on the lag grid.
    support : torch.Tensor, optional
        ``(d, d)`` bool support disk of the polar grid.
    normalize : bool
        Unit-variance projections (the decomposition's ``normalized_projections``).
    reference : torch.Tensor, optional
        ``(D_ref, d, d // 2 + 1)`` the after-CTF reference noise of every plane of
        the run on the box's rfft grid (:func:`reference_grid`); ``None`` for
        ``before_ctf``.
    device : torch.device or str
        Where the projections are made.
    """

    def __init__(
        self,
        volume: torch.Tensor | np.ndarray,
        template_filter: torch.Tensor,
        weighting: PowerWeighting,
        ctf: torch.Tensor,
        *,
        support: torch.Tensor | None = None,
        normalize: bool = True,
        reference: torch.Tensor | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        vol = torch.as_tensor(np.asarray(volume, dtype=np.float32)).to(self.device)
        if vol.dim() != 3 or len(set(vol.shape)) != 1:
            raise ValueError(f"volume must be cubic, got {tuple(vol.shape)}")
        box = int(vol.shape[-1])
        self.box = (box, box)
        if tuple(template_filter.shape) != (box, box // 2 + 1):
            raise ValueError(
                f"template_filter must be ({box}, {box // 2 + 1}), got "
                f"{tuple(template_filter.shape)}"
            )
        self.dft, self.edge, self.pad = precompute_volume_dft(vol)
        del vol
        self.filter = template_filter.to(self.device, torch.float32)
        self.weighting = weighting
        lag = tuple(int(s) for s in weighting.lag_shape)
        if tuple(ctf.shape[1:]) != (lag[0], lag[1] // 2 + 1):
            raise ValueError(
                f"ctf must be (D, {lag[0]}, {lag[1] // 2 + 1}) on the lag grid, got "
                f"{tuple(ctf.shape)}"
            )
        self.ctf = ctf.to(self.device, torch.float32)
        self.support = (
            None if support is None else support.to(self.device, torch.float32)
        )
        self.normalize = bool(normalize)
        self.reference: torch.Tensor | None = None
        self.white = 1.0
        if reference is not None:
            if tuple(reference.shape[1:]) != (box, box // 2 + 1):
                raise ValueError(
                    f"reference must be (D, {box}, {box // 2 + 1}), got "
                    f"{tuple(reference.shape)}"
                )
            hermitian = torch.full((box // 2 + 1,), 2.0, device=self.device)
            hermitian[0] = 1.0
            if box % 2 == 0:
                hermitian[-1] = 1.0
            weights = reference.to(self.device, torch.float32).mean(dim=0) * hermitian
            self.reference = weights
            # a white, unit-variance template's |m^|^2 is h w: its factor is 1
            self.white = float(weights.sum()) * box * box

    @property
    def n_planes(self) -> int:
        """Searched CTF planes ``D``."""
        return int(self.ctf.shape[0])

    @torch.no_grad()
    def box_filtered(
        self, rotations: torch.Tensor, filters: torch.Tensor
    ) -> torch.Tensor:
        """``(F, n, Ly, Lx // 2 + 1)`` projections under filters applied in the box.

        Each of the ``(F, d, d // 2 + 1)`` filters (e.g. whitening x CTF) multiplies
        the projection's rfft on the template box, as Leopard-EM's refine_template
        filters its templates, so whatever the CTF delocalises past the box wraps
        around it. No normalisation, support mask, row factor or lag-grid CTF.
        """
        rot = rotations.to(self.device, torch.float32)
        projections = project_from_precomputed_dft(self.dft, rot, self.edge, self.pad)
        spectrum = torch.fft.rfft2(projections)
        box = tuple(int(v) for v in projections.shape[-2:])
        out = [
            self.weighting.transform(
                torch.fft.irfft2(spectrum * f.to(self.device, torch.float32), s=box)
            )
            for f in filters
        ]
        return torch.stack(out)

    def with_filter(
        self, template_filter: torch.Tensor, *, normalize: bool = False
    ) -> TemplateProjector:
        """A projector of the same volume under another filter, sharing its DFT.

        No support mask and no after-CTF row factor: e.g. refinement's templates under
        the whitening alone.
        """
        other = copy.copy(self)
        other.filter = template_filter.to(self.device, torch.float32)
        other.normalize = bool(normalize)
        other.support = None
        other.reference = None
        other.white = 1.0
        return other

    def row_scale(self, templates: torch.Tensor) -> torch.Tensor | None:
        """``(n,)`` after-CTF row factor of ``(n, d, d)`` templates (or ``None``).

        ``a = (e / c)^(-1/2)``, ``e`` the template's noise variance under the
        reference noise averaged over the run's planes, ``c`` that of a white
        unit-variance template.
        """
        if self.reference is None:
            return None
        power = torch.fft.rfft2(templates).abs().square()
        e = (power * self.reference).sum(dim=(-2, -1))
        scale: torch.Tensor = (e / self.white).rsqrt()
        return scale

    @torch.no_grad()
    def __call__(
        self, rotations: torch.Tensor, ctf: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``(D, n, Ly, Lx // 2 + 1)`` templates of ``(n, 3, 3)`` rotations.

        ``ctf`` (``(D', Ly, Lx // 2 + 1)`` on the lag grid) replaces the searched
        planes, e.g. a refinement's defocus offsets.
        """
        rot = rotations.to(self.device, torch.float32)
        projections = project_from_precomputed_dft(self.dft, rot, self.edge, self.pad)
        m = apply_fourier_filters(projections, self.filter[None])[0]
        if self.normalize:
            var, mean = torch.var_mean(m, dim=(-2, -1), keepdim=True)
            m = (m - mean) / torch.sqrt(var + 1e-8)
        if self.support is not None:
            m = m * self.support
        return self.project_templates(m, ctf)

    @torch.no_grad()
    def project_templates(
        self, templates: torch.Tensor, ctf: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``(D, n, Ly, Lx // 2 + 1)`` of ``(n, d, d)`` real-space search templates.

        The after-CTF row factor, the lag grid and the planes, as :meth:`__call__`
        applies them to the exact templates; e.g. for the decomposition's own
        reconstructions (:class:`SearchTemplates`).
        """
        m = templates.to(self.device, torch.float32)
        scale = self.row_scale(m)
        if scale is not None:
            m = m * scale[:, None, None]
        q: torch.Tensor = self.weighting.transform(m)
        planes = self.ctf if ctf is None else ctf.to(self.device, torch.float32)
        return q[None] * planes[:, None]


def _batches(rotations: torch.Tensor, size: int) -> Iterator[torch.Tensor]:
    for i in range(0, int(rotations.shape[0]), size):
        yield rotations[i : i + size]


@dataclass(frozen=True)
class _Sampler:
    generator: torch.Generator

    def rotations(self, n: int) -> torch.Tensor:
        return haar_rotations(n, self.generator)

    def pairs(self, n_pairs: int, size: int) -> Iterator[tuple[Any, Any]]:
        for _ in range(n_pairs):
            yield self.rotations(size), self.rotations(size)


def _projector(
    micrograph: Micrograph,
    volume: torch.Tensor | np.ndarray,
    template_filter: torch.Tensor,
    *,
    planes: Sequence[int] | None,
    n_filters: int,
    support: torch.Tensor | None,
    normalize_projections: bool,
    reference: torch.Tensor | None,
    device: torch.device,
) -> tuple[PowerWeighting, TemplateProjector]:
    """The image's lag-grid power weighting and the search's template projector."""
    if n_filters != 1:
        raise NotImplementedError(
            "the analytic null models decompositions with one Fourier filter"
        )
    image_shape = micrograph.image_shape
    box = (int(template_filter.shape[0]), int(template_filter.shape[0]))
    weighting = PowerWeighting(
        torch.fft.rfft2(micrograph.image.to(device, torch.float32)), image_shape, box
    ).to(device)
    ctf = micrograph.ctf if planes is None else micrograph.ctf[list(planes)]
    lag_ctf = lag_grid_planes(ctf.to(device), image_shape, weighting.lag_shape)
    project = TemplateProjector(
        volume,
        template_filter,
        weighting,
        lag_ctf,
        support=support,
        normalize=normalize_projections,
        reference=reference,
        device=device,
    )
    return weighting, project


# pylint: disable=too-many-locals
def predict_null(
    micrograph: Micrograph,
    volume: torch.Tensor | np.ndarray,
    template_filter: torch.Tensor,
    *,
    nside: int,
    n_psi: int,
    valid_area: float,
    planes: Sequence[int] | None = None,
    n_filters: int = 1,
    support: torch.Tensor | None = None,
    normalize_projections: bool = True,
    reference: torch.Tensor | None = None,
    mask_radius: float = 10.0,
    n_mean_samples: int = 3000,
    n_kappa_pairs: int = 256,
    n_lattice_samples: int = 128,
    batch_size: int = 4,
    thresholds: np.ndarray | None = None,
    seed: int = 0,
    device: torch.device | str = "cpu",
    details: dict[str, Any] | None = None,
) -> AnalyticNullPrediction:
    """Predict the null peak rate of a Panther search of ``micrograph``.

    Parameters
    ----------
    micrograph : Micrograph
        The prepared micrograph (for after-CTF normalization, with its plane factors
        applied, as the search sees it). Only its search image's power spectrum and
        its CTF planes are used, so a phase-randomised copy gives the same answer.
    volume : torch.Tensor or numpy.ndarray
        ``(d, d, d)`` the template volume the decomposition was made from.
    template_filter : torch.Tensor
        ``(d, d // 2 + 1)`` the decomposition's Fourier filter
        (:meth:`~panther_em.filters.WhiteningFilter.template_filter`).
    nside, n_psi : int
        The searched HEALPix grid and in-plane angles (the final stage's).
    valid_area : float
        Search positions peaks are counted in (border and mask removed).
    planes : sequence of int, optional
        The searched CTF planes; default all of the micrograph's.
    n_filters : int
        Fourier filters of the decomposition (hypotheses per orientation); only 1
        is modelled.
    support : torch.Tensor, optional
        ``(d, d)`` bool support disk of the polar grid.
    normalize_projections : bool
        Whether the decomposition normalized its projections. Default True.
    reference : torch.Tensor, optional
        ``(D_ref, d, d // 2 + 1)`` after-CTF reference noise of the run's planes on
        the template box (:func:`reference_grid`); ``None`` for ``before_ctf``.
    mask_radius : float
        The peak finder's suppression radius, pixels. Default 10.
    n_mean_samples, n_kappa_pairs, n_lattice_samples : int
        Rotations for the mean template, pairs of rotation batches for ``kappa``,
        rotations for the lattice correlations (each with every plane). Defaults 3000,
        256 and 128. The lattice sample also sets the hypotheses' variance spread, which
        Panther has (5-7 % before the CTF, ~1.5 % after): with 32 rotations the cut
        moves by up to 0.05 z between seeds before the CTF; 128 matches 512 after it.
    batch_size : int
        Rotations per batch for ``kappa`` and the lattice. Default 4.
    thresholds : numpy.ndarray, optional
        Thresholds to evaluate; default 2 to 10 in steps of 0.01.
    seed : int
        Seed of the sampled rotations.
    device : torch.device or str
        Where the projections and inner products are computed.
    details : dict, optional
        Extras recorded in the prediction's ``details``.

    Returns
    -------
    AnalyticNullPrediction
        ``.zscore_cutoff(fp)`` is the predicted threshold for ``fp`` null peaks per
        micrograph; ``.as_dict()`` the JSON record.
    """
    start = time.time()
    device = torch.device(device)
    image_shape = micrograph.image_shape
    weighting, project = _projector(
        micrograph,
        volume,
        template_filter,
        planes=planes,
        n_filters=n_filters,
        support=support,
        normalize_projections=normalize_projections,
        reference=reference,
        device=device,
    )
    box = project.box
    n_planes = project.n_planes
    sampler = _Sampler(torch.Generator().manual_seed(int(seed)))

    per_batch = max(1, 16 // n_planes)
    mean = estimate_mean_template(
        project, weighting, _batches(sampler.rotations(n_mean_samples), per_batch)
    )
    kappa = estimate_kappa(
        project, weighting, mean.mean, sampler.pairs(n_kappa_pairs, batch_size)
    )
    lattice_batch = batch_size if n_planes > 1 else 2 * batch_size
    samples = sample_lattice(
        project,
        weighting,
        mean.mean,
        _batches(sampler.rotations(n_lattice_samples), lattice_batch),
        lattice_neighbours(nside, n_psi),
    )
    n_orientations = 12 * nside**2
    n_points = float(valid_area) * n_orientations * n_psi * n_filters * n_planes
    record: dict[str, Any] = {
        "engine": "panther_em.search",
        "nside": int(nside),
        "n_psi": int(n_psi),
        "orientations": n_orientations,
        "planes": n_planes,
        "hypotheses": n_orientations * n_psi * n_filters * n_planes,
        "normalization": "before_ctf" if reference is None else "after_ctf",
        "image_shape": list(image_shape),
        "box_shape": list(box),
        "lag_shape": list(weighting.lag_shape),
        "support_masked": support is not None,
        "n_mean_samples": int(n_mean_samples),
        "n_kappa_pairs": int(n_kappa_pairs),
        "n_lattice_samples": int(n_lattice_samples),
        "seed": int(seed),
        "weighting_negative_fraction": weighting.negative_fraction,
        "device": str(device),
    }
    record.update(details or {})
    record["seconds"] = time.time() - start
    prediction: AnalyticNullPrediction = predict_null_rate(
        samples,
        kappa,
        mean.rho_bar,
        n_points,
        valid_area,
        mask_radius,
        DEFAULT_THRESHOLDS if thresholds is None else thresholds,
        details=record,
    )
    return prediction


class SearchTemplates:
    """The search's own templates: the decomposition's reconstructions at grid points.

    The SVD reconstruction ``m~`` of a grid point (orientation row, in-plane angle)
    from the feature rectangles a stage keeps, warped to the Cartesian box as the
    search's kernels are. This is what the search correlates, so it carries the
    rank truncation and the polar warp (and the warp's centre) that the exact
    projections of :class:`TemplateProjector` do not.

    Parameters
    ----------
    reconstructor : Reconstructor
        The decomposition's (``Template.reconstructor``).
    rectangles : sequence of Rectangle
        The stage's ``(k_start, m_start, k_extent, m_extent)`` feature rectangles.
    fourier_filter : int
        The decomposition's Fourier filter row. Default 0.
    """

    def __init__(
        self,
        reconstructor: Any,
        rectangles: Sequence[Sequence[int]],
        fourier_filter: int = 0,
    ) -> None:
        result = reconstructor.result.to_dense()
        if result.num_volumes != 1 or result.is_complex_projection:
            raise NotImplementedError(
                "search templates of one real-projection volume only"
            )
        self.reconstructor = reconstructor
        self.device = reconstructor.device
        k_stop = max(int(r[0]) + int(r[2]) for r in rectangles)
        e_stop = max(int(r[1]) + int(r[3]) for r in rectangles)
        mask = torch.zeros((k_stop, e_stop))
        for k0, m0, ke, me in rectangles:
            mask[k0 : k0 + ke, m0 : m0 + me] = 1.0
        s = torch.as_tensor(np.asarray(result.S)[:k_stop, :e_stop]) * mask
        vh = torch.as_tensor(np.asarray(result.Vh)[:k_stop, :e_stop])
        self.weighted_vh = (s[..., None] * vh).to(self.device, torch.complex64)
        u = torch.as_tensor(np.asarray(result.U))  # (1, FF, O, K, E)
        self.u = u[0, int(fourier_filter), :, :k_stop, :e_stop]
        self.k = torch.arange(k_stop, device=self.device, dtype=torch.float32)
        self.n_angle = int(result.num_angular_components)
        self.phi = torch.as_tensor(np.asarray(result.phi_values, np.float64))
        self.theta = torch.as_tensor(np.asarray(result.theta_values, np.float64))

    @property
    def n_orientations(self) -> int:
        """Orientation rows of the decomposition."""
        return int(self.u.shape[0])

    def rotations(
        self, orientation: torch.Tensor, psi_deg: torch.Tensor
    ) -> torch.Tensor:
        """``(n, 3, 3)`` float32 rotations of grid points (ZYZ phi, theta, psi)."""
        angles = torch.stack(
            [
                self.phi[orientation.cpu()],
                self.theta[orientation.cpu()],
                psi_deg.cpu().double(),
            ],
            dim=-1,
        )
        rotation: torch.Tensor = roma.euler_to_rotmat("ZYZ", angles, degrees=True)
        return rotation.to(torch.float32)

    @torch.no_grad()
    def __call__(
        self, orientation: torch.Tensor, psi_deg: torch.Tensor
    ) -> torch.Tensor:
        """``(n, d, d)`` float32 reconstructions of grid points."""
        u = self.u[orientation.cpu()].to(self.device, torch.complex64)  # (n, K, E)
        blocks = torch.einsum("nke,ker->nkr", u, self.weighted_vh)
        # the in-plane angle is a phase per angular frequency k
        angle = torch.deg2rad(psi_deg.to(self.device, torch.float32))
        blocks = blocks * torch.exp(1j * self.k[None, :] * angle[:, None])[..., None]
        polar = torch.fft.irfft(blocks, n=self.n_angle, dim=1, norm="ortho")
        cartesian = self.reconstructor.to_cartesian(polar.to(torch.float32))
        out: torch.Tensor = cartesian.real.to(torch.float32)
        return out


def _neighbours(
    nside: int,
    n_psi: int,
    orientation: int,
    psi_index: int,
    rings: int = 1,
    psi_steps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A grid point's neighbourhood: views x in-plane angles, without the point.

    The HEALPix (NESTED) pixels within ``rings`` rings of its view, each at the
    in-plane angles within ``psi_steps`` steps of its own.
    """
    views = {int(orientation)}
    frontier = [int(orientation)]
    for _ in range(int(rings)):
        ring = hp.get_all_neighbours(int(nside), frontier, nest=True)
        new = {int(v) for v in np.asarray(ring).reshape(-1) if int(v) >= 0} - views
        views |= new
        frontier = sorted(new)
    orientations, psis = [], []
    for view in sorted(views):
        for step in range(-int(psi_steps), int(psi_steps) + 1):
            if view == orientation and step == 0:
                continue
            orientations.append(view)
            psis.append((psi_index + step) % n_psi)
    return torch.as_tensor(orientations), torch.as_tensor(psis)


# pylint: disable=too-many-locals,too-many-arguments,too-many-statements
def predict_refined_null(
    micrograph: Micrograph,
    volume: torch.Tensor | np.ndarray,
    template_filter: torch.Tensor,
    signal_filter: torch.Tensor,
    search: AnalyticNullPrediction,
    search_templates: SearchTemplates,
    *,
    nside: int,
    n_psi: int,
    orientation_offsets: torch.Tensor | np.ndarray,
    defocus_offsets: Sequence[float] | np.ndarray,
    candidate_threshold: float,
    refine_ctf: Callable[[np.ndarray], torch.Tensor],
    plane_defocus: Sequence[float],
    planes: Sequence[int] | None = None,
    n_filters: int = 1,
    support: torch.Tensor | None = None,
    normalize_projections: bool = True,
    reference: torch.Tensor | None = None,
    shift_radius: int = 1,
    condition_radius: int = 2,
    neighbour_rings: int = 1,
    neighbour_psi_steps: int = 1,
    n_candidates: int = 8,
    n_draws: int = 200_000,
    n_mean_samples: int = 3000,
    heights: np.ndarray | None = None,
    batch_size: int = 11,
    seed: int = 0,
    device: torch.device | str = "cpu",
    details: dict[str, Any] | None = None,
) -> RefinedNullPrediction:
    r"""Predict the null peak rate after Leopard-EM refines a search's candidates.

    The candidates at or above ``candidate_threshold`` are exported and refined by
    Leopard-EM's refine_template, which re-scores them with its own exact correlation
    over a local window (:func:`leopard_em.analysis.analytic_null.refine_window` reads
    its offsets) and normalises it by the search's per-pixel mean and standard
    deviation (:mod:`panther_em.managers.leopard`). The refined null follows from the
    gain of a null candidate conditioned on what the search established about it
    (:mod:`leopard_em.analysis.analytic_null.refine`), with two kinds of score on the
    same Gaussian field:

    * the search's: Panther's own templates (the decomposition's reconstructions,
      :class:`SearchTemplates`) of the candidate (its score is u) and of its grid
      neighbours, which it beat: the eight HEALPix neighbours at its in-plane angle,
      the in-plane angles either side, and the adjacent planes. On the z scale
      :math:`(m^\sim - M) / \sigma`, with M the search's mean template and
      :math:`\sigma` its field SD;
    * refinement's: the exact projection at every window offset (``offset @ R``, as
      refine_template composes them; defocus offsets about the candidate's plane)
      under the whitening alone (``signal_filter``, refine_template's full band),
      at unit variance as refine_template normalises it, less :math:`M / \sigma`.

    The gap between the two (rank truncation, polar warp, band limit) is what moves
    a candidate's refined z below its search z at the same pose.

    Parameters
    ----------
    micrograph : Micrograph
        As for :func:`predict_null`.
    volume : torch.Tensor or numpy.ndarray
        As for :func:`predict_null`.
    template_filter : torch.Tensor
        As for :func:`predict_null` (``W_avg * band``).
    signal_filter : torch.Tensor
        ``(d, d // 2 + 1)`` the whitening alone, ``W_avg``
        (``WhiteningFilter.average_filter``), for refinement's templates.
    search : AnalyticNullPrediction
        The search's prediction (:func:`predict_null`).
    search_templates : SearchTemplates
        The final stage's reconstructions.
    nside, n_psi : int
        The searched HEALPix grid and in-plane angles.
    orientation_offsets : torch.Tensor or numpy.ndarray
        Refinement's ZYZ Euler offsets ``(n, 3)``, degrees, including zero.
    defocus_offsets : sequence of float
        Refinement's defocus offsets about the candidate's plane, A, including zero.
    candidate_threshold : float
        Panther z-score at or above which candidates are refined.
    refine_ctf : callable
        Maps defocus offsets (A, relative to the micrograph's defocus, as the search
        config's ``defocus_offsets``) to ``(n, d, d // 2 + 1)`` CTF planes on the
        template box, as refine_template builds them: it filters its templates in
        the box, while the search applies the CTF to the whole image.
    plane_defocus : sequence of float
        Relative defocus of each searched plane (``planes`` order), A.
    planes : sequence of int, optional
        As for :func:`predict_null`.
    n_filters : int
        As for :func:`predict_null`.
    support : torch.Tensor, optional
        As for :func:`predict_null`.
    normalize_projections : bool
        As for :func:`predict_null`.
    reference : torch.Tensor, optional
        As for :func:`predict_null`.
    shift_radius : int
        Positions refinement scores about each candidate, +-pixels. Default 1 (on
        xenon_131, +-2 px gives the same gains within 0.03 z at 2.5x the cost).
    condition_radius : int
        Positions at which the candidate beat its grid neighbours, +-pixels.
        Default 2.
    neighbour_rings, neighbour_psi_steps : int
        The grid neighbourhood it beat: the HEALPix views within ``neighbour_rings``
        rings, each at the in-plane angles within ``neighbour_psi_steps`` steps (and
        its own view and angle on the adjacent planes). Default 1 and 1.
    n_candidates : int
        Grid points sampled as candidates, spread evenly over the planes. Default 8.
    n_draws : int
        Gaussian draws per candidate. Default 200,000.
    n_mean_samples : int
        Rotations for the mean template (with the same ``seed``, the search
        prediction's). Default 3000.
    heights : numpy.ndarray, optional
        Candidate heights to sample the gain at; default 5 to 9 in steps of 0.5.
    batch_size : int
        Rotations projected per batch. Default 11.
    seed : int
        Seed of the sampled grid points and draws.
    device : torch.device or str
        Where the projections and draws are computed.
    details : dict, optional
        Extras recorded in the prediction's ``details``.

    Returns
    -------
    RefinedNullPrediction
        ``.zscore_cutoff(fp)`` is the predicted threshold on refined z-scores.
    """
    start = time.time()
    device = torch.device(device)
    weighting, project = _projector(
        micrograph,
        volume,
        template_filter,
        planes=planes,
        n_filters=n_filters,
        support=support,
        normalize_projections=normalize_projections,
        reference=reference,
        device=device,
    )
    n_planes = project.n_planes
    plane_defocus = [float(v) for v in plane_defocus]
    if len(plane_defocus) != n_planes:
        raise ValueError(
            f"plane_defocus has {len(plane_defocus)} values for {n_planes} planes"
        )
    if search_templates.n_orientations != 12 * nside**2:
        raise ValueError("search_templates must hold the searched HEALPix grid")
    sampler = _Sampler(torch.Generator().manual_seed(int(seed)))
    mean = estimate_mean_template(
        project,
        weighting,
        _batches(sampler.rotations(n_mean_samples), max(1, 16 // n_planes)),
    )
    offsets = torch.as_tensor(np.asarray(orientation_offsets, np.float64))
    zero_offset = int(torch.argmin(offsets.abs().sum(dim=1)))
    df = np.asarray(defocus_offsets, float).reshape(-1)
    zero_df = int(np.argmin(np.abs(df)))
    if float(offsets[zero_offset].abs().sum()) > 1e-6 or abs(df[zero_df]) > 1e-6:
        raise ValueError("the refinement's offsets must include zero")
    offset_rot = roma.euler_to_rotmat("ZYZ", offsets, degrees=True).float()
    n_offsets = int(offsets.shape[0])
    box_filters: dict[int, torch.Tensor] = {}

    def window_filters(plane: int) -> torch.Tensor:
        """Whitening x CTF on the template box, per defocus offset of ``plane``."""
        if plane not in box_filters:
            ctf = refine_ctf(plane_defocus[plane] + df).to(device, torch.float32)
            box_filters[plane] = ctf * signal_filter.to(device, torch.float32)[None]
        return box_filters[plane]

    generator = np.random.default_rng(int(seed))
    k = int(n_candidates)
    grid = [
        (
            int(generator.integers(search_templates.n_orientations)),
            int(generator.integers(n_psi)),
            (i * n_planes) // k,
        )
        for i in range(k)
    ]
    step = 360.0 / n_psi

    def search_scores(orientation: int, psi_index: int, plane: int) -> torch.Tensor:
        """The search's own templates: the candidate, then its grid neighbours."""
        o_n, j_n = _neighbours(
            nside, n_psi, orientation, psi_index, neighbour_rings, neighbour_psi_steps
        )
        o = torch.cat([torch.tensor([orientation]), o_n])
        j = torch.cat([torch.tensor([psi_index]), j_n])
        m = search_templates(o, j.double() * step)
        q = project.project_templates(m)  # (D, n, ...)
        adjacent = [p for p in (plane - 1, plane + 1) if 0 <= p < n_planes]
        return torch.cat([q[plane], q[adjacent, 0]])  # own, neighbours, planes

    # the search's field SD: the exact one, scaled by its templates' captured power
    captured, exact_power = 0.0, 0.0
    for orientation, psi_index, plane in grid:
        own = search_scores(orientation, psi_index, plane)
        rot = search_templates.rotations(
            torch.tensor([orientation]), torch.tensor([psi_index * step])
        )
        exact = project(rot)[plane, 0]
        captured += float(weighting.inner(own[0], own[0]))
        exact_power += float(weighting.inner(exact, exact))
    sigma = math.sqrt(mean.sigma2 * captured / exact_power)
    mean_scaled = mean.mean / sigma

    def windows() -> Iterator[RefinementWindow]:
        for orientation, psi_index, plane in grid:
            scores = (search_scores(orientation, psi_index, plane) - mean.mean) / sigma
            r0 = search_templates.rotations(
                torch.tensor([orientation]), torch.tensor([psi_index * step])
            )[0]
            rotations = offset_rot @ r0  # offset @ R, as refine_template composes
            window = torch.cat(
                [
                    project.box_filtered(
                        rotations[j : j + batch_size], window_filters(plane)
                    )
                    for j in range(0, n_offsets, batch_size)
                ],
                dim=1,
            )  # (n_df, n_offsets, ...): index df * n_offsets + offset
            window = window.reshape(-1, *window.shape[2:])
            norm = torch.sqrt(weighting.inner(window, window))
            window = window / norm[:, None, None] - mean_scaled
            yield RefinementWindow(
                window=window,
                centre=zero_df * n_offsets + zero_offset,
                search=scores[1:],
                own=scores[0],
            )

    gain = sample_refinement_gain(
        windows(),
        weighting,
        None,
        shift_radius=shift_radius,
        condition_radius=condition_radius,
        heights=heights,
        n_draws=n_draws,
        seed=seed,
    )
    record: dict[str, Any] = {
        "engine": "panther_em.search + leopard_em.refine_template",
        "nside": int(nside),
        "n_psi": int(n_psi),
        "planes": n_planes,
        "normalization": "before_ctf" if reference is None else "after_ctf",
        "orientation_offsets": n_offsets,
        "defocus_offsets": df.tolist(),
        "shift_radius": int(shift_radius),
        "condition_radius": int(condition_radius),
        "neighbour_rings": int(neighbour_rings),
        "neighbour_psi_steps": int(neighbour_psi_steps),
        "n_candidates": k,
        "captured_power": captured / exact_power,
        "n_mean_samples": int(n_mean_samples),
        "seed": int(seed),
        "device": str(device),
    }
    record.update(details or {})
    record["seconds"] = time.time() - start
    return predict_refined_null_rate(search, gain, candidate_threshold, details=record)
