"""Between-stage decision functions: accept / reject / follow-up pixels."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

import torch

from panther_em.inference.search.ledger import PixelLabel

if TYPE_CHECKING:
    from panther_em.inference.search.schedule import StageSpec


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
    corr : torch.Tensor
        Per-pixel correlation values, shape ``(P,)``, estimated from low-rank SVD.
    mu : torch.Tensor
        Per-pixel mean values, shape ``(P,)``, estimated from low-rank SVD.
    sigma : torch.Tensor
        Per-pixel standard deviation values, shape ``(P,)``, estimated from low-rank
        SVD.

    Returns
    -------
    torch.Tensor
        Per-pixel error intervals, shape ``(P, 2)``, where the second axis is
        ``[lower_bound, upper_bound]``. If error is correct, then interval guaranteed to
        contain the true z-score for each pixel.
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
    r"""Classify worst-case z-score intervals against a fixed threshold.

    Parameters
    ----------
    interval : torch.Tensor
        Per-pixel ``[lower_bound, upper_bound]`` intervals, shape ``(..., 2)``, as
        returned by :func:`zscore_error_intervals`.
    threshold : float
        Fixed z-score decision threshold, constant across all stages.

    Returns
    -------
    torch.Tensor
        int8 tensor of shape ``(...,)`` holding following :class:`PixelLabel` values:
        - :attr:`PixelLabel.REJECTED` where ``upper < threshold``
        - :attr:`PixelLabel.ACCEPTED` where ``lower > threshold``
        - :attr:`PixelLabel.UNCERTAIN` otherwise
    """
    lower = interval[..., 0]
    upper = interval[..., 1]

    label = torch.full_like(lower, int(PixelLabel.UNCERTAIN), dtype=torch.int8)
    label.masked_fill_(upper < threshold, int(PixelLabel.REJECTED))
    label.masked_fill_(lower > threshold, int(PixelLabel.ACCEPTED))

    return label


@runtime_checkable
class Decision(Protocol):
    """Between-stage policy operating on the dense pixel ledger.

    Attributes
    ----------
    pointwise : bool
        ``True`` when the label of pixel ``p`` depends only on the ledger fields at
        ``p`` (so the decision may be applied per pixel block); ``False`` for
        neighbourhood or ranking decisions.
    requires_error : bool
        ``True`` when the stage's reconstruction error must be resolved first.
    """

    pointwise: ClassVar[bool]
    requires_error: ClassVar[bool]

    def __call__(
        self,
        ledger: object,
        pixel_index: torch.Tensor,
        stage: StageSpec,
        stage_index: int,
    ) -> None:
        """Mutate ``ledger.label`` for (at least) the pixels in ``pixel_index``."""


class NoDecision:
    """Leave every searched pixel ``UNCERTAIN`` (single-stage / sweep use)."""

    pointwise: ClassVar[bool] = True
    requires_error: ClassVar[bool] = False

    def __call__(
        self,
        ledger: object,
        pixel_index: torch.Tensor,
        stage: StageSpec,
        stage_index: int,
    ) -> None:
        """No-op."""
        return None

    def __repr__(self) -> str:
        """Stable repr."""
        return "NoDecision()"
