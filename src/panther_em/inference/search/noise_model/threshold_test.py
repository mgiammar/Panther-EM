"""The threshold test of the coloured-noise model.

A low-rank search sees only the kept part ``m~`` of every template; the exact
correlation adds the missed part ``delta``. From a pixel's maximum correlation, its
mean and variance over the hypotheses, the maximum's ``(plane, template row)`` and the
local noise level ``ell``, the test predicts the exact correlation of the particle
branch and its spread, and measures how far below the (lowered) detection threshold
the prediction lies:

- ``mu`` the mean, ``x = mip - mu``, ``a = rho / (1 - rho)``;
- ``sigma_hat = sqrt(variance + ell qbar_perp)``, ``T_lo = mu + T zeta sigma_hat``;
- predicted mean ``mu + x / (1 - rho)``;
- variance ``(a - lam)^2 ell q_kept + ell q_perp + (eps_rho a x)^2``;
- ``D = (T_lo - mean) / sd``: a pixel is discarded when ``D >= z_alpha``.

``q_kept``, ``q_perp``, ``lam``, ``rho`` (:class:`NoiseScales`) are the micrograph's
noise and signal curves weighted by the template's error tables
(:class:`~.spatial_frequency_error.PerSpatialFrequencyErrorTable`); the rule itself
(:meth:`ThresholdTest.distance`) is a function of per-pixel values only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from scipy.special import ndtri  # type: ignore[import-untyped]

from panther_em.inference.search.noise_model.spatial_frequency_error import (
    TABLE_NAMES,
    PerSpatialFrequencyErrorTable,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from panther_em.inference.search.noise_model.image_set import ImageSet

SD_FLOOR = 1e-30  # the variance of the particle branch at full rank is exactly 0
SCALE_NAMES = ("q_kept", "q_perp", "lam", "rho")


@dataclass(frozen=True)
class Calibration:
    """Calibrated constants of the threshold test.

    Parameters
    ----------
    alpha : float
        Tolerated probability of discarding a pixel whose exact correlation would
        have crossed the threshold. Defaults to ``5e-3``.
    zeta : float
        ``zeta_beta <= 1``: the lower quantile of ``sigma / sigma_hat`` measured on a
        reference search (:func:`zeta_quantile`); it lowers the threshold to
        ``mu + T zeta sigma_hat``. Defaults to 1 (no lowering).
    eps_rho : float
        Relative uncertainty of the signal loss ``rho``. Defaults to ``0.1``.
    b_factor : float
        B-factor envelope (``A^2``) on the signal weight. Defaults to 0.
    """

    alpha: float = 5e-3
    zeta: float = 1.0
    eps_rho: float = 0.1
    b_factor: float = 0.0

    def __post_init__(self) -> None:
        """Validate the ranges."""
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0.0 < self.zeta <= 1.0:
            raise ValueError("zeta must lie in (0, 1]")
        if self.eps_rho < 0.0:
            raise ValueError("eps_rho must be >= 0")
        if self.b_factor < 0.0:
            raise ValueError("b_factor must be >= 0")

    @property
    def z_alpha(self) -> float:
        """``Phi^-1(1 - alpha)``: the distance at which a pixel is discarded."""
        return float(-ndtri(self.alpha))


@dataclass(frozen=True, eq=False)
class NoiseScales:
    """One micrograph's scales of every ``(plane, template row)``.

    Parameters
    ----------
    q_kept : torch.Tensor
        ``(D, N)`` float64 variance of the kept noise at unit level.
    q_perp : torch.Tensor
        ``(D, N)`` float64 variance of the missed noise independent of the kept noise.
    lam : torch.Tensor
        ``(D, N)`` float64 regression slope of the missed on the kept noise.
    rho : torch.Tensor
        ``(D, N)`` float64 signal loss, in ``[0, 1)``.
    qbar_perp : float
        Mean of ``q_perp`` over the stage's hypotheses.
    """

    q_kept: torch.Tensor
    q_perp: torch.Tensor
    lam: torch.Tensor
    rho: torch.Tensor
    qbar_perp: float

    def __post_init__(self) -> None:
        """Validate shapes and dtypes."""
        shape = tuple(self.q_kept.shape)
        if len(shape) != 2:
            raise ValueError("scales must be (D, N) tensors")
        for name in SCALE_NAMES:
            t = getattr(self, name)
            if tuple(t.shape) != shape or t.dtype is not torch.float64:
                raise ValueError(f"{name} must be a float64 tensor of shape {shape}")
        if not isinstance(self.qbar_perp, float):
            raise ValueError("qbar_perp must be a float")

    @property
    def n_templates(self) -> int:
        """``N``."""
        return int(self.q_kept.shape[1])


@dataclass(frozen=True, eq=False)
class ThresholdTest:
    """The per-pixel test of one feature selection.

    Parameters
    ----------
    table : PerSpatialFrequencyErrorTable
        The template's error tables for the stage's rectangles.
    calibration : Calibration, optional
        The test's constants.
    """

    table: PerSpatialFrequencyErrorTable
    calibration: Calibration = field(default_factory=Calibration)

    def __post_init__(self) -> None:
        """Validate the types."""
        if not isinstance(self.table, PerSpatialFrequencyErrorTable):
            raise ValueError("table must be a PerSpatialFrequencyErrorTable")
        if not isinstance(self.calibration, Calibration):
            raise ValueError("calibration must be a Calibration")

    def scales(
        self,
        images: ImageSet,
        *,
        rows: torch.Tensor | Sequence[int] | None = None,
        planes: Sequence[int] | None = None,
    ) -> NoiseScales:
        """The micrograph's noise and signal scales of every ``(plane, row)``.

        Parameters
        ----------
        images : ImageSet
            The micrograph; its curves must be on the table's frequency axis.
        rows, planes : optional
            The stage's template rows and CTF planes; ``qbar_perp`` averages
            ``q_perp`` over them (all of them when ``None``).
        """
        table = self.table
        if not torch.equal(images.frequencies, table.frequencies):
            raise ValueError(
                "the image set's curves are on another frequency axis than the table "
                "(prepare the micrograph with this table or one of the same box/pad)"
            )

        freq = table.frequencies.to(torch.float64)
        b_factor = self.calibration.b_factor
        envelope = torch.exp(-b_factor * (freq / images.pixel_size) ** 2 / 4.0)
        noise = images.noise  # (D, F)
        weight = images.signal * envelope
        t = {name: getattr(table, name).to(torch.float64).T for name in TABLE_NAMES}

        q_kept = noise @ t["reconstruction_autocorr"]  # (D, N)
        q_residual = noise @ t["residual_autocorr"]
        q_cross = noise @ t["reconstruction_residual_corr"]
        lam = torch.where(q_kept > 0, q_cross / q_kept.clamp_min(1e-300), 0.0)
        q_perp = (q_residual - lam**2 * q_kept).clamp_min(0.0)
        g_kept = weight @ t["signal_reconstruction_corr"]
        g_residual = weight @ t["signal_residual_corr"]
        total = g_kept + g_residual
        rho = torch.where(total > 0, g_residual / total.clamp_min(1e-300), 0.0)
        rho = rho.clamp(0.0, 1.0 - 1e-6)

        rows = torch.arange(table.n_templates) if rows is None else rows
        planes = range(images.n_defocus) if planes is None else planes
        qbar_perp = float(q_perp[list(planes)][:, torch.as_tensor(rows)].mean())
        return NoiseScales(q_kept, q_perp, lam, rho, qbar_perp)

    @torch.no_grad()
    def distance(
        self,
        scales: NoiseScales,
        threshold: float,
        *,
        level: torch.Tensor,
        plane: torch.Tensor,
        row: torch.Tensor,
        mip: torch.Tensor,
        mean: torch.Tensor,
        variance: torch.Tensor,
    ) -> torch.Tensor:
        """``D``: lowered threshold minus predicted exact correlation, per pixel.

        In standard deviations of the particle branch.

        Parameters
        ----------
        scales : NoiseScales
            :meth:`scales` of the micrograph.
        threshold : float
            The z-score threshold ``T``.
        level : torch.Tensor
            ``(P,)`` local noise level ``ell`` at each pixel's box centre.
        plane, row : torch.Tensor
            ``(P,)`` int64 CTF plane and template row of each pixel's maximum.
        mip, mean, variance : torch.Tensor
            ``(P,)`` the pixel's maximum, mean and variance over the hypotheses.

        Returns
        -------
        torch.Tensor
            ``(P,)`` float32 on ``mip``'s device.
        """
        if scales.n_templates != self.table.n_templates:
            raise ValueError(
                f"scales hold {scales.n_templates} template rows; the table "
                f"{self.table.n_templates}"
            )
        device = mip.device
        plane, row = plane.to(device), row.to(device)
        q_kept, q_perp, lam, rho = (
            getattr(scales, name).to(device)[plane, row] for name in SCALE_NAMES
        )
        ell = level.to(device, torch.float64)

        cal = self.calibration
        mu = mean.to(torch.float64)
        x = mip.to(torch.float64) - mu
        a = rho / (1.0 - rho)
        sigma_hat = (
            (variance.to(torch.float64) + ell * scales.qbar_perp).clamp_min(0.0).sqrt()
        )
        t_lo = mu + threshold * cal.zeta * sigma_hat
        mean_p = mu + x / (1.0 - rho)
        sd2 = (a - lam) ** 2 * ell * q_kept + ell * q_perp + (cal.eps_rho * a * x) ** 2
        dist: torch.Tensor = (t_lo - mean_p) / sd2.clamp_min(SD_FLOOR).sqrt()
        return dist.to(torch.float32)


# --------------------------------------------------------------------------- #
# Threshold and calibration helpers
# --------------------------------------------------------------------------- #


def zscore_threshold(n_pixels: int, n_hypotheses: int) -> float:
    """``T = Phi^-1(1 - 1 / (N_pix N))``: one expected false detection per search."""
    tests = float(n_pixels) * float(n_hypotheses)
    if tests < 1.0:
        raise ValueError("n_pixels and n_hypotheses must be positive")
    return float(-ndtri(1.0 / tests))


def zeta_quantile(
    sigma_exact: torch.Tensor, sigma_hat: torch.Tensor, beta: float
) -> float:
    """``zeta_beta``: the lower ``beta``-quantile of ``sigma / sigma_hat``.

    Measured on a reference (exact) search; the fraction ``beta`` of pixels whose
    exact ``sigma`` falls below ``zeta_beta * sigma_hat`` is the tolerated failure of
    the sigma bound.
    """
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie in (0, 1)")
    ratio = (sigma_exact.to(torch.float64) / sigma_hat.to(torch.float64)).reshape(-1)
    ratio = ratio[torch.isfinite(ratio)]
    if ratio.numel() == 0:
        raise ValueError("no finite sigma ratios")
    return float(torch.quantile(ratio, beta))
