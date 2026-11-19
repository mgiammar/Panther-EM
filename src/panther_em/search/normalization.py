r"""After-CTF noise normalization: every ``(plane, template)`` hypothesis at one noise.

Why
---
Panther normalizes each whitened template projection ``m_n`` to unit variance
*before* the CTF and applies the CTF to the image, one plane per searched defocus
``d``. Under the coloured-noise model the noise variance of hypothesis ``(d, n)`` is
then

.. math:: e_{dn} = \sum_F \mathrm{noise}_{dF} \, P_{nF},

with :math:`P_{nF}` the ring power of template row ``n`` (its error table's
``reconstruction_autocorr``) and :math:`\mathrm{noise}_{dF}` the micrograph's noise
power under ``CTF_d`` (:attr:`~panther_em.search.micrograph.Micrograph.noise`; the
same product the stage-1 threshold test calls ``q_kept``). ``e`` varies with ``n``
(how much of a projection's power falls where the CTF passes it) and with ``d`` (the
CTF's own power). On xenon_131 (13 planes, +-1200 A) its relative sd is 4.6 % for
the unfiltered 60S template and 6.0 / 6.9 % at template B-factors 60 / 120.

The z-score divides every hypothesis by one pooled per-pixel sigma, so a hypothesis
with a larger ``e`` reaches a high z more easily. A particle's z comes from its own
hypothesis and is unchanged on average, but the maximum over millions of noise
hypotheses is drawn preferentially from the high-``e`` ones: the null tail is fatter
than Leopard-EM's and cisTEM's, which normalize every CTF-filtered projection, i.e.
*after* the CTF. On xenon_131 that raises the null-calibrated 1-FP cut by 0.12 to
0.21 z (more for blurrier templates) at no gain in particle z; this normalization
recovers 4 to 6 points of recall at matched false positives on synthetic data
(docs/after_ctf_normalization.md).

What
----
``normalization: after_ctf`` (:class:`~panther_em.managers.search.SearchConfig`; the
default since 2026-11-15, ``before_ctf`` restores the original) multiplies
hypothesis ``(d, n)`` by :math:`s_{dn} = a_n b_d`, chosen so that ``e``
becomes (nearly) constant:

* :math:`a_n`, one per template row (orientation x Fourier filter), once per run,
  from a reference noise model -- white noise under the ring-averaged ``CTF^2`` of
  every searched plane of every micrograph of the run
  (:func:`reference_noise`):
  :math:`a_n = (\bar e^{\mathrm{ref}}_{\cdot n} / \bar e^{\mathrm{ref}})^{-1/2}`.
  It multiplies the template weights ``U * S`` (:meth:`Template.scale_rows`) and the
  rows of every error table (:func:`scale_table`), so the threshold test predicts
  the scaled correlations.
* :math:`b_d`, one per CTF plane, per micrograph, from the micrograph's own noise
  curves after :math:`a`:
  :math:`b_d = (\overline{a^2 e_{d\cdot}} / \overline{a^2 e})^{-1/2}`. It multiplies
  the micrograph's CTF planes (so the image features of plane ``d``) and its noise
  and signal curves (:meth:`AfterCtfNormalization.apply`).

The mean of ``e`` over all hypotheses is unchanged, so the threshold test's and the
z-score's scales stay where they were. Both factors fold into tensors the search
already builds once (template weights, CTF planes), so the correlation loop, the
moment sums and their cost are untouched.

The product is separable. An exact per-``(d, n)`` scale is not a row operation on
either operand (the planes share the template rows; the CTF is applied to the
image) and would need the reduction kernel to multiply every ``(plane, row)``
spectrum. On xenon_131 the separable form leaves a 1.0 % sd of ``e`` (0.5 % in
sigma), against 2.6 to 3.7 % with :math:`a_n` alone (one factor per orientation,
as an earlier prototype did with rescaled decomposition copies).

Not covered: the in-plane (psi) dependence of ``e`` that an astigmatic CTF adds
(``e`` is computed from ring means of ``CTF^2``), and the defocus dependence of
:math:`a_n` beyond what one run-wide reference averages over.

The variance equalized is that of the correlation of the template's *kept* part
``m~`` in the final stage's feature rectangles (that stage's table), the
correlation the reported z-scores come from.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from panther_em.filters import ctf_planes

if TYPE_CHECKING:
    from collections.abc import Sequence

    from panther_em.approximation.errors import PerSpatialFrequencyErrorTable
    from panther_em.search.micrograph import Micrograph

# Error-table fields that are quadratic in the template (scale a^2) and linear in it
# (cross-spectra with the unscaled signal p; scale a).
QUADRATIC_TABLES = (
    "reconstruction_autocorr",
    "residual_autocorr",
    "reconstruction_residual_corr",
)
LINEAR_TABLES = ("signal_reconstruction_corr", "signal_residual_corr")


def template_power(table: PerSpatialFrequencyErrorTable) -> torch.Tensor:
    """``(N, F)`` float64 ring power of every row's kept template ``m~``."""
    return table.reconstruction_autocorr.to(torch.float64)


def hypothesis_variance(noise: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
    """``(D, N)`` float64 noise variance ``e`` of every ``(plane, row)``.

    ``noise (D, F)`` the noise power per plane and ring, ``power (N, F)`` the
    templates' ring power; the threshold test's ``q_kept`` at unit local level.
    """
    return noise.to(torch.float64) @ power.to(torch.float64).T


def relative_sd(values: torch.Tensor) -> float:
    """``sd / mean`` over every element."""
    v = values.to(torch.float64)
    return float(v.std(unbiased=False) / v.mean())


@dataclass(frozen=True)
class CtfSpec:
    """One micrograph's CTF parameters (see :func:`~panther_em.filters.ctf_planes`)."""

    defocus: float  # mean (u + v) / 2, Angstroms
    astigmatism: float = 0.0  # (u - v) / 2, Angstroms
    astigmatism_angle: float = 0.0  # degrees


def reference_noise(
    table: PerSpatialFrequencyErrorTable,
    micrographs: Sequence[CtfSpec],
    offsets: Sequence[float],
    *,
    pixel_size: float,
    voltage: float,
    spherical_aberration: float,
    amplitude_contrast: float,
    phase_shift: float = 0.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """``(D, F)`` float64 reference noise: white noise under each plane's ``CTF^2``.

    The ring mean of ``CTF_d^2`` on ``table``'s frequency axis (a whitened, unit
    variance image has ring power 1 in :class:`Micrograph`'s convention), averaged
    over ``micrographs``. Plane ``d`` sits at each micrograph's defocus plus
    ``offsets[d]``.
    """
    if not micrographs:
        raise ValueError("at least one micrograph is needed")
    box = max(int(table.support.shape[0]), int(table.support.shape[1]))
    size = max(1024, 2 * int(table.pad) * box)
    shape = (size, size)
    bins = table.image_bins(shape, device)
    total = torch.zeros((len(offsets), table.n_bins), dtype=torch.float64)
    for spec in micrographs:
        planes = ctf_planes(
            shape,
            [spec.defocus + float(o) for o in offsets],
            pixel_size=pixel_size,
            astigmatism=spec.astigmatism,
            astigmatism_angle=spec.astigmatism_angle,
            voltage=voltage,
            spherical_aberration=spherical_aberration,
            amplitude_contrast=amplitude_contrast,
            phase_shift=phase_shift,
            device=device,
        )
        total += bins.ring_mean(planes.square().to(torch.float64)).cpu()
    return total / len(micrographs)


def row_scales(variance: torch.Tensor) -> torch.Tensor:
    """``(N,)`` float64 ``a_n = (mean_d e_dn / mean e)^(-1/2)`` of a ``(D, N)`` e."""
    per_row = variance.mean(dim=0)
    return (per_row / per_row.mean()).rsqrt()


def plane_scales(variance: torch.Tensor) -> torch.Tensor:
    """``(D,)`` float64 ``b_d = (mean_n e_dn / mean e)^(-1/2)`` of a ``(D, N)`` e."""
    per_plane = variance.mean(dim=1)
    return (per_plane / per_plane.mean()).rsqrt()


def scale_table(
    table: PerSpatialFrequencyErrorTable, scale: torch.Tensor
) -> PerSpatialFrequencyErrorTable:
    """``table`` of the templates with row ``n`` multiplied by ``scale[n]``.

    Profiles quadratic in the template scale by ``scale^2``, its cross-spectra with
    the (unscaled) signal by ``scale``; ``rho`` and ``lam`` of the threshold test are
    ratios and stay unchanged.
    """
    s = scale.to(torch.float32).reshape(-1, 1)
    if s.shape[0] != table.n_templates:
        raise ValueError(
            f"{s.shape[0]} scales for a table of {table.n_templates} template rows"
        )
    fields = {name: getattr(table, name) * s.square() for name in QUADRATIC_TABLES}
    fields.update({name: getattr(table, name) * s for name in LINEAR_TABLES})
    return dataclasses.replace(table, **fields)


@dataclass(frozen=True, eq=False)
class AfterCtfNormalization:
    """The run-wide part of after-CTF normalization and its per-micrograph step.

    Parameters
    ----------
    power : torch.Tensor
        ``(N, F)`` float64 ring power of the *unscaled* kept templates.
    row_scale : torch.Tensor
        ``(N,)`` float64 :math:`a_n`, already applied to the template weights and
        the error tables.
    """

    power: torch.Tensor
    row_scale: torch.Tensor

    @classmethod
    def create(
        cls,
        table: PerSpatialFrequencyErrorTable,
        reference: torch.Tensor,
    ) -> AfterCtfNormalization:
        """:math:`a_n` of ``table``'s templates under the ``(D, F)`` ``reference``."""
        power = template_power(table)
        return cls(power, row_scales(hypothesis_variance(reference, power)))

    def apply(self, micrograph: Micrograph) -> tuple[Micrograph, dict]:
        """``micrograph`` with :math:`b_d` applied, and a report of the spreads.

        ``micrograph`` must be prepared with the unscaled CTF planes. Its noise and
        signal curves, and the CTF planes, are multiplied by :math:`b_d^2`,
        :math:`b_d^2` and :math:`b_d`.

        The report gives the relative sd of ``e`` over every ``(plane, row)``:
        ``before`` (no scaling), ``rows`` (:math:`a_n` only) and ``after``
        (:math:`a_n b_d`), with the factors' ranges.
        """
        if micrograph.frequencies.numel() != self.power.shape[1]:
            raise ValueError("the micrograph's curves are on another frequency axis")
        e = hypothesis_variance(micrograph.noise, self.power)
        a2 = self.row_scale.square()[None, :]
        b = plane_scales(e * a2)
        b2 = b.square()[:, None]
        scaled = dataclasses.replace(
            micrograph,
            ctf=micrograph.ctf
            * b.to(micrograph.ctf.device, torch.float32)[:, None, None],
            noise=micrograph.noise * b2,
            signal=micrograph.signal * b2,
        )
        a = self.row_scale
        report = {
            "variance_rel_sd": {
                "before": relative_sd(e),
                "rows": relative_sd(e * a2),
                "after": relative_sd(e * a2 * b2),
            },
            "row_scale": {
                "min": float(a.min()),
                "max": float(a.max()),
                "rel_sd": relative_sd(a),
            },
            "plane_scale": [float(v) for v in b],
        }
        return scaled, report
