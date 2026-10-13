"""The per-pixel decision: accept, reject, or leave ambiguous, with an error bound.

Every stage ends with the same rule. A pixel's z-score ``z = (mip - mean) / std`` is
widened into a worst-case interval by the stage's reconstruction error (the rank-r
features reconstruct the true correlations only approximately), and the interval is
compared with the threshold: entirely above it is ACCEPTED, entirely below it is
REJECTED, straddling it is AMBIGUOUS. Without a bound the interval collapses to the
z-score itself and the rule is a plain threshold. An optional post-step ABSORBS
ambiguous pixels within a radius of an accepted one (the same particle's neighbours).

The bound is pluggable: :class:`ErrorModel` derives the relative error from the
decomposition's singular values; any object with ``rho(rectangles) -> float`` (and
optionally ``intervals(zscore, rho) -> (lower, upper)``) can replace it.
"""

from __future__ import annotations

import enum
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from panther_em.decomposition.result import DecompositionResult
    from panther_em.inference.search.results import PixelLedger
    from panther_em.inference.search.stage import Stage
    from panther_em.inference.search.types import Rectangle


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
# Worst-case z-score intervals
# --------------------------------------------------------------------------- #


def zscore_error_intervals(
    error: float,
    corr: torch.Tensor,  # (P,)
    mu: torch.Tensor,  # (P,)
    sigma: torch.Tensor,  # (P,)
) -> torch.Tensor:  # (P, 2)
    r"""Compute worst-case intervals for z-scores for a fixed reconstruction error.

    Parameters
    ----------
    error : float
        Estimated reconstruction error, in the same units as ``corr``/``sigma``.
    corr, mu, sigma : torch.Tensor
        Per-pixel correlation, mean and standard deviation, shape ``(P,)``, estimated
        from the low-rank features.

    Returns
    -------
    torch.Tensor
        ``(P, 2)`` ``[lower, upper]`` intervals; if ``error`` is correct the interval
        contains the true z-score of every pixel.
    """
    sigma_min = sigma - error
    sigma_max = torch.sqrt(sigma**2 + 2 * error**2) + error
    sigma_min = torch.clamp(sigma_min, min=1e-6)
    sigma_max = torch.clamp(sigma_max, min=1e-6)

    a_max = 2 * error / sigma_min
    b_max = error * (2 * sigma_max + error) / (sigma * (sigma_min + sigma))

    zscore = (corr - mu) / sigma
    delta_z = (a_max + b_max * torch.abs(zscore)) / (1 - b_max)

    return torch.stack([zscore - delta_z, zscore + delta_z], dim=-1)


def zscore_bound_is_vacuous(error: float) -> bool:
    """Whether the relative-error z-score bound carries no information."""
    if error <= 0.0:
        return False
    sigma = torch.tensor(1.0, dtype=torch.float64)
    sigma_min = torch.clamp(sigma - error, min=1e-6)
    sigma_max = torch.sqrt(sigma**2 + 2 * error**2) + error
    b_max = error * (2 * sigma_max + error) / (sigma * (sigma_min + sigma))
    return bool(b_max >= 1.0)


def classify_zscore_interval(
    interval: torch.Tensor,  # (..., 2), [lower, upper]
    threshold: float,
) -> torch.Tensor:  # (...,) int8
    """Labels from worst-case z-score intervals against a fixed threshold.

    ``REJECTED`` where ``upper < threshold``, ``ACCEPTED`` where ``lower >
    threshold``, ``AMBIGUOUS`` otherwise.
    """
    lower = interval[..., 0]
    upper = interval[..., 1]
    label = torch.full_like(lower, int(PixelLabel.AMBIGUOUS), dtype=torch.int8)
    label.masked_fill_(upper < threshold, int(PixelLabel.REJECTED))
    label.masked_fill_(lower > threshold, int(PixelLabel.ACCEPTED))
    return label


# --------------------------------------------------------------------------- #
# Error bound models
# --------------------------------------------------------------------------- #


class ErrorBound(Protocol):
    """What a decision needs from an error model."""

    def rho(self, rectangles: Sequence[Rectangle]) -> float:
        """Relative L2 reconstruction error of the rectangles' features."""


@dataclass(frozen=True)
class ErrorModel:
    """Spectral relative reconstruction error of a feature selection.

    ``rho^2 = 1 - (1 - tail) * energy(rectangles) / total_energy``, with the energy
    the Parseval-weighted sum of the squared singular values over the selected
    cells.

    Parameters
    ----------
    S2 : torch.Tensor
        ``(K, E)`` float64 squared singular values.
    parseval : torch.Tensor
        ``(K,)`` float64 weight of each angular frequency block (2 for the
        conjugate-symmetric blocks of a real projection, 1 at k=0 and Nyquist).
    tail : float, optional
        Fraction of the energy the decomposition itself discarded. Defaults to 0.
    """

    S2: torch.Tensor
    parseval: torch.Tensor
    tail: float = 0.0

    def __post_init__(self) -> None:
        """Validate shapes and the tail fraction."""
        if self.S2.dim() != 2 or tuple(self.parseval.shape) != (self.S2.shape[0],):
            raise ValueError("S2 must be (K, E) and parseval (K,)")
        if not (0.0 <= self.tail < 1.0):
            raise ValueError("tail must lie in [0, 1)")

    @classmethod
    def from_result(cls, result: DecompositionResult, *, tail: float = 0.0) -> ErrorModel:
        """Spectral model from a decomposition's singular values."""
        try:
            s = np.asarray(result.S)
        except AttributeError:  # SparseDecompositionResult hides S; densify
            s = np.asarray(result.to_dense().S)
        s2 = torch.as_tensor(s, dtype=torch.float64) ** 2
        k_blocks = int(s2.shape[0])
        if result.is_complex_projection:
            weights = torch.ones(k_blocks, dtype=torch.float64)
        else:
            weights = torch.full((k_blocks,), 2.0, dtype=torch.float64)
            weights[0] = 1.0
            n_ang = int(result.num_angular_components)
            if n_ang % 2 == 0 and n_ang // 2 < k_blocks:
                weights[n_ang // 2] = 1.0
        return cls(S2=s2, parseval=weights, tail=float(tail))

    def energy(self, rectangles: Sequence[Rectangle]) -> float:
        """Parseval-weighted sum of ``S^2`` over the rectangles' cells."""
        total = 0.0
        k_blocks, e_max = self.S2.shape
        for k0, m0, k_ext, m_ext in rectangles:
            k1, m1 = min(k0 + k_ext, k_blocks), min(m0 + m_ext, e_max)
            if k1 <= k0 or m1 <= m0:
                continue
            block = self.S2[k0:k1, m0:m1] * self.parseval[k0:k1, None]
            total += float(block.sum())
        return total

    def total_energy(self) -> float:
        """Energy of everything stored."""
        return float((self.S2 * self.parseval[:, None]).sum())

    def rho(self, rectangles: Sequence[Rectangle]) -> float:
        """Relative reconstruction error of a rectangle set."""
        frac = self.energy(rectangles) / self.total_energy()
        return math.sqrt(max(0.0, 1.0 - (1.0 - self.tail) * frac))


# --------------------------------------------------------------------------- #
# The decision
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Decision:
    """Accept / reject / leave ambiguous by a z-score threshold under an error bound.

    Parameters
    ----------
    threshold : float
        z-score detection threshold.
    bound : ErrorBound, optional
        Error model giving the stage's relative reconstruction error; ``None``
        applies the threshold to the z-score itself (no interval).
    absorb_radius : int, optional
        After labelling, AMBIGUOUS pixels within this Euclidean radius of an ACCEPTED
        pixel become ABSORBED (grid mode only). ``0`` disables the step.
    """

    threshold: float
    bound: ErrorBound | None = None
    absorb_radius: int = 0

    def __post_init__(self) -> None:
        """Validate the radius."""
        if self.absorb_radius < 0:
            raise ValueError("absorb_radius must be >= 0")

    def intervals(self, zscore: torch.Tensor, rho: float) -> torch.Tensor:
        """``(P, 2)`` worst-case z-score intervals for a relative error ``rho``."""
        if rho <= 0.0:
            return torch.stack([zscore, zscore], dim=-1)
        return zscore_error_intervals(
            rho, zscore, torch.zeros_like(zscore), torch.ones_like(zscore)
        )

    @torch.no_grad()
    def apply(self, ledger: PixelLedger, pixel_index: torch.Tensor, stage: Stage) -> None:
        """Label the stage's pixels on the ledger (and absorb, when configured)."""
        idx = pixel_index.to(ledger.device, torch.long)
        rho = 0.0 if self.bound is None else float(self.bound.rho(stage.rectangles))
        if zscore_bound_is_vacuous(rho):
            warnings.warn(
                f"stage {stage.name!r}: relative error {rho:.3f} makes the z-score "
                "bound vacuous (>= ~0.431); every pixel stays AMBIGUOUS",
                UserWarning,
                stacklevel=2,
            )
            ledger.set_labels(
                idx, torch.full_like(idx, int(PixelLabel.AMBIGUOUS), dtype=torch.int8)
            )
            return
        interval = self.intervals(ledger.zscore[idx], rho)
        ledger.set_labels(idx, classify_zscore_interval(interval, self.threshold))
        if self.absorb_radius > 0:
            ledger.absorb_near_accepted(self.absorb_radius)
