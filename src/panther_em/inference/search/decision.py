"""The per-pixel decision: the threshold test rejects, the z-score threshold accepts.

Every stage ends with a :class:`Decision`. With a :class:`ThresholdTest` it applies the
per-pixel rule of the coloured-noise model: from the stage's maximum correlation, its
orientation, the pixel's mean and variance, the orientation's radial tables
(:class:`~.noise.RadialTables`) and the micrograph's spectra
(:class:`~.types.ImageSpectra`), it predicts the exact correlation of the particle
branch and REJECTS the pixel when the probability that it crosses the (lowered)
threshold is at most ``alpha``. Survivors are ACCEPTED where the stage's own z-score
exceeds the threshold -- when ``accept`` is set, the final stage -- and AMBIGUOUS
otherwise. Without a test the rule is the plain threshold: REJECTED below it,
ACCEPTED above it. An optional post-step ABSORBS ambiguous pixels within a radius of
an accepted one (the same particle's neighbours).

The distance ``D`` of every tested pixel is kept as its ``score``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from scipy.special import ndtri  # type: ignore[import-untyped]

from panther_em.inference.search.noise import NoiseScales, RadialTables

if TYPE_CHECKING:
    from panther_em.inference.search.hypotheses import HypothesisSet
    from panther_em.inference.search.results import PixelLedger
    from panther_em.inference.search.types import ImageSpectra, SearchGeometry

SD_FLOOR = 1e-30  # the variance of the particle branch at full rank is exactly 0


class PixelLabel(enum.IntEnum):
    """Per-pixel status across search stages.

    - PENDING: not yet searched.
    - AMBIGUOUS: searched, but the decision could not call it (the next stage's
      input; a legitimate final label for the external follow-up).
    - REJECTED: decided below the detection threshold.
    - ACCEPTED: decided above the detection threshold (re-scored by a later stage).
    - ABSORBED: was AMBIGUOUS but lies within the absorb radius of an ACCEPTED pixel.
    - MASKED: excluded from the search by the image set's mask.
    """

    PENDING = 0
    AMBIGUOUS = 1
    REJECTED = 2
    ACCEPTED = 3
    ABSORBED = 4
    MASKED = 5


# --------------------------------------------------------------------------- #
# The threshold test
# --------------------------------------------------------------------------- #


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
        reference search (:func:`~.noise.zeta_quantile`); it lowers the threshold to
        ``mu + T zeta sigma_hat``. Defaults to 1 (no lowering).
    eps_rho : float
        Relative uncertainty of the signal loss ``rho``. Defaults to ``0.1``.
    b_factor : float
        B-factor envelope (``A^2``) on the signal weight, see
        :meth:`~.noise.RadialTables.scales`. Defaults to 0.
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
class ThresholdTest:
    """The per-pixel test of one feature selection.

    Parameters
    ----------
    tables : RadialTables
        The selection's radial tables, built for the stage's rectangles.
    calibration : Calibration, optional
        The test's constants.
    """

    tables: RadialTables
    calibration: Calibration = field(default_factory=Calibration)

    def __post_init__(self) -> None:
        """Validate the types."""
        if not isinstance(self.tables, RadialTables):
            raise ValueError("tables must be a RadialTables")
        if not isinstance(self.calibration, Calibration):
            raise ValueError("calibration must be a Calibration")

    def scales(
        self, spectra: ImageSpectra, hyps: HypothesisSet | None = None
    ) -> NoiseScales:
        """The micrograph's scales under this test's B-factor (see the tables)."""
        return self.tables.scales(
            spectra, hyps=hyps, b_factor=self.calibration.b_factor
        )

    @torch.no_grad()
    def distance(
        self,
        threshold: float,
        spectra: ImageSpectra,
        geometry: SearchGeometry,
        pixel_index: torch.Tensor,
        mip: torch.Tensor,
        mean: torch.Tensor,
        variance: torch.Tensor,
        best_index: torch.Tensor,
        *,
        hyps: HypothesisSet | None = None,
        scales: NoiseScales | None = None,
    ) -> torch.Tensor:
        """``D``: lowered threshold minus predicted exact correlation, per pixel.

        In standard deviations of the particle branch.

        Parameters
        ----------
        threshold : float
            The stage's z-score threshold ``T``.
        spectra : ImageSpectra
            The image set's curves and level maps.
        geometry : SearchGeometry
            The grid ``pixel_index`` addresses.
        pixel_index : torch.Tensor
            ``(P,)`` flat pixel ids.
        mip, mean, variance : torch.Tensor
            ``(P,)`` the stage's maximum, mean and variance at those pixels.
        best_index : torch.Tensor
            ``(P,)`` global hypothesis ids of the maxima (``-1`` gives NaN).
        hyps : HypothesisSet, optional
            The stage's hypotheses (``qbar_perp`` averages over them).
        scales : NoiseScales, optional
            Precomputed :meth:`scales`; computed here when omitted.

        Returns
        -------
        torch.Tensor
            ``(P,)`` float32 on ``mip``'s device.
        """
        device = mip.device
        n_templates = self.tables.n_templates
        if scales is None:
            scales = self.scales(spectra, hyps)
        if scales.n_templates != n_templates:
            raise ValueError(
                f"scales hold {scales.n_templates} template rows; the tables "
                f"{n_templates}"
            )
        scales = scales.to(device)
        idx = pixel_index.to(device, torch.long)
        zeros = torch.zeros_like(idx)

        # the pixel's image and the centre of its template window
        per_image = geometry.n_pixels_per_image
        image = idx // per_image
        within = idx % per_image
        if geometry.mode == "grid":
            width = int(geometry.grid[1])
            row, col = within // width, within % width
        else:
            row, col = zeros, zeros
        k_h, k_w = geometry.template_shape
        row = row + (k_h - 1) // 2
        col = col + (k_w - 1) // 2
        level = spectra.level.to(device)
        ell = level[
            image if level.shape[0] > 1 else zeros,
            row if level.shape[1] > 1 else zeros,
            col if level.shape[2] > 1 else zeros,
        ].to(torch.float64)

        # the maximum's plane and template row
        best = best_index.to(device, torch.long)
        valid = best >= 0
        safe = torch.where(valid, best, zeros)
        plane, template = safe // n_templates, safe % n_templates
        b_axis = image if scales.q_kept.shape[0] > 1 else zeros
        d_axis = plane if scales.q_kept.shape[1] > 1 else zeros
        q_kept = scales.q_kept[b_axis, d_axis, template]
        q_perp = scales.q_perp[b_axis, d_axis, template]
        lam = scales.lam[b_axis, d_axis, template]
        rho = scales.rho[b_axis, d_axis, template]
        qbar_perp = scales.qbar_perp[b_axis]

        cal = self.calibration
        mu = mean.to(torch.float64)
        x = mip.to(torch.float64) - mu
        a = rho / (1.0 - rho)
        sigma_hat = (variance.to(torch.float64) + ell * qbar_perp).clamp_min(0.0).sqrt()
        t_lo = mu + threshold * cal.zeta * sigma_hat
        mean_p = mu + x / (1.0 - rho)
        sd2 = (a - lam) ** 2 * ell * q_kept + ell * q_perp + (cal.eps_rho * a * x) ** 2
        dist = (t_lo - mean_p) / sd2.clamp_min(SD_FLOOR).sqrt()
        nan = torch.full_like(dist, float("nan"))
        return torch.where(valid, dist, nan).to(torch.float32)

    def reject(
        self,
        threshold: float,
        spectra: ImageSpectra,
        geometry: SearchGeometry,
        pixel_index: torch.Tensor,
        mip: torch.Tensor,
        mean: torch.Tensor,
        variance: torch.Tensor,
        best_index: torch.Tensor,
        *,
        hyps: HypothesisSet | None = None,
        scales: NoiseScales | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(rejected, D)``: the pixels with ``D >= z_alpha`` and the distances."""
        dist = self.distance(
            threshold,
            spectra,
            geometry,
            pixel_index,
            mip,
            mean,
            variance,
            best_index,
            hyps=hyps,
            scales=scales,
        )
        return dist >= self.calibration.z_alpha, dist


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Decision:
    """Reject by the threshold test (or the plain threshold), accept by the threshold.

    Parameters
    ----------
    threshold : float
        z-score detection threshold ``T``.
    test : ThresholdTest, optional
        Rejects pixels under the coloured-noise model; ``None`` rejects below the
        threshold itself.
    accept : bool, optional
        Label survivors with ``zscore > threshold`` ACCEPTED (the final stage);
        ``False`` leaves every survivor AMBIGUOUS for the next stage. Defaults to
        True.
    absorb_radius : int, optional
        After labelling, AMBIGUOUS pixels within this Euclidean radius of an ACCEPTED
        pixel become ABSORBED (grid mode only). ``0`` disables the step.
    """

    threshold: float
    test: ThresholdTest | None = None
    accept: bool = True
    absorb_radius: int = 0

    def __post_init__(self) -> None:
        """Validate the test and the radius."""
        if self.test is not None and not isinstance(self.test, ThresholdTest):
            raise ValueError("test must be a ThresholdTest or None")
        if self.absorb_radius < 0:
            raise ValueError("absorb_radius must be >= 0")

    @torch.no_grad()
    def apply(
        self,
        ledger: PixelLedger,
        pixel_index: torch.Tensor,
        hyps: HypothesisSet,
        spectra: ImageSpectra | None = None,
    ) -> None:
        """Label the stage's pixels on the ledger (and absorb, when configured).

        Parameters
        ----------
        ledger : PixelLedger
            Holds the stage's maps at ``pixel_index``.
        pixel_index : torch.Tensor
            ``(P,)`` the pixels the stage searched.
        hyps : HypothesisSet
            The stage's hypotheses.
        spectra : ImageSpectra, optional
            The image set's spectra; required by a threshold test.
        """
        idx = pixel_index.to(ledger.device, torch.long)
        zscore = ledger.zscore[idx]
        if self.test is None:
            rejected = zscore < self.threshold
        else:
            if spectra is None:
                raise ValueError(
                    "the decision's threshold test needs the image set's spectra "
                    "(estimate_spectra, then ImageSet.with_spectra)"
                )
            rejected, score = self.test.reject(
                self.threshold,
                spectra,
                ledger.geometry,
                idx,
                ledger.mip[idx],
                ledger.mean[idx],
                ledger.variance[idx],
                ledger.best_index[idx],
                hyps=hyps,
            )
            ledger.set_scores(idx, score)
        labels = torch.full_like(idx, int(PixelLabel.AMBIGUOUS), dtype=torch.int8)
        labels[rejected] = int(PixelLabel.REJECTED)
        if self.accept:
            labels[~rejected & (zscore > self.threshold)] = int(PixelLabel.ACCEPTED)
        ledger.set_labels(idx, labels)
        if self.absorb_radius > 0:
            ledger.absorb_near_accepted(self.absorb_radius)
