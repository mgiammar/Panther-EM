"""Between-stage decision functions: accept / reject / follow-up pixels."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, runtime_checkable

import torch

from panther_em.inference.search.ledger import PixelLabel

if TYPE_CHECKING:
    from panther_em.inference.search.ledger import PixelLedger
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


def surpass_threshold(
    corr: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    *,
    threshold: float,
    image_norm: torch.Tensor | float,
    noise_variance: torch.Tensor | float,
    rank: int,
    calibration: Any | None = None,
) -> torch.Tensor:
    """Decide, per pixel or per (pixel, hypothesis) pair, whether a correlation passes.

    NOTE
    ----
    **Placeholder with desired signature. Need to update with appropriate function.**

    Parameters
    ----------
    corr : torch.Tensor
        Correlation values: ``(P,)`` per pixel, or ``(P, Q)`` per pixel and
        hypothesis (the pair-emission case).
    mean, variance : torch.Tensor
        Per-pixel correlation mean and variance over all hypotheses at this rank,
        ``(P,)``; broadcast over ``Q``.
    threshold : float
        The cutoff ``T`` applied consistently across pixels.
    image_norm : torch.Tensor or float
        Precomputed per-pixel image norm ``x_norm``, ``(P,)`` or scalar.
    noise_variance : torch.Tensor or float
        Precomputed per-pixel noise variance ``tau_hat``, ``(P,)`` or scalar.
    rank : int
        Reconstruction rank ``r`` (number of SVD features) the correlations came from.
    calibration : object, optional
        Reserved for the calibration table (``s(r)``, ``q(r)``, ...).

    Returns
    -------
    torch.Tensor
        Bool tensor shaped like ``corr``: ``True`` where the threshold is surpassed.
    """
    if corr.dim() not in (1, 2):
        raise ValueError(f"corr must be (P,) or (P, Q), got {tuple(corr.shape)}")
    if rank < 1:
        raise ValueError("rank must be >= 1")
    n_pixels = int(corr.shape[0])

    def per_pixel(name: str, values: torch.Tensor | float) -> torch.Tensor:
        t = torch.as_tensor(values, dtype=torch.float32, device=corr.device)
        if t.dim() == 0:
            t = t.expand(n_pixels)
        if tuple(t.shape) != (n_pixels,):
            raise ValueError(
                f"{name} must be a scalar or ({n_pixels},), got {tuple(t.shape)}"
            )
        return t.unsqueeze(-1) if corr.dim() == 2 else t

    mean_p = per_pixel("mean", mean)
    var_p = per_pixel("variance", variance)
    per_pixel("image_norm", image_norm)  # validated; used by the calibrated rule
    per_pixel("noise_variance", noise_variance)
    z = (corr.to(torch.float32) - mean_p) / var_p.clamp_min(1e-12).sqrt()
    return z > threshold


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
        ledger: PixelLedger,
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
        ledger: PixelLedger,
        pixel_index: torch.Tensor,
        stage: StageSpec,
        stage_index: int,
    ) -> None:
        """No-op."""
        return None

    def __repr__(self) -> str:
        """Stable repr."""
        return "NoDecision()"


def _resolve_error(decision_error: float | None, stage: StageSpec) -> float:
    if decision_error is not None:
        return float(decision_error)
    if stage.error is None:
        raise ValueError(
            f"stage {stage.name or stage.rectangles} carries no reconstruction error; "
            "supply StageSpec.error, an ErrorModel, or the classifier's `error`"
        )
    return float(stage.error)


class ZScoreIntervalClassifier:
    """Accept / reject by the worst-case z-score interval; otherwise stay uncertain.

    Parameters
    ----------
    threshold : float
        z-score detection threshold.
    error : float, optional
        Reconstruction error to use for every stage (overrides ``StageSpec.error``).
    units : {"relative", "absolute"}, optional
        ``"relative"`` (default) evaluates the bound in normalized coordinates
        (``corr = z, mu = 0, sigma = 1``) with the stage's *relative* error ``rho``.
    """

    pointwise: ClassVar[bool] = True
    requires_error: ClassVar[bool] = True

    def __init__(
        self,
        threshold: float,
        *,
        error: float | None = None,
        units: Literal["relative", "absolute"] = "relative",
    ) -> None:
        if units not in ("relative", "absolute"):
            raise ValueError("units must be 'relative' or 'absolute'")
        self.threshold = float(threshold)
        self.error = error
        self.units = units

    @torch.no_grad()
    def __call__(
        self,
        ledger: PixelLedger,
        pixel_index: torch.Tensor,
        stage: StageSpec,
        stage_index: int,
    ) -> None:
        """Classify the stage's pixels."""
        error = _resolve_error(self.error, stage)
        idx = pixel_index.to(ledger.device, torch.long)
        if self.units == "relative":
            if zscore_bound_is_vacuous(error):
                warnings.warn(
                    f"stage {stage_index}: relative error {error:.3f} makes the "
                    "z-score "
                    "bound vacuous (>= ~0.431); every pixel stays UNCERTAIN",
                    UserWarning,
                    stacklevel=2,
                )
                ledger.set_labels(
                    idx,
                    torch.full_like(idx, int(PixelLabel.UNCERTAIN), dtype=torch.int8),
                )
                return
            z = ledger.zscore[idx]
            interval = zscore_error_intervals(
                error, z, torch.zeros_like(z), torch.ones_like(z)
            )
        else:
            interval = zscore_error_intervals(
                error, ledger.mip[idx], ledger.mean[idx], ledger.std()[idx]
            )
        ledger.set_labels(idx, classify_zscore_interval(interval, self.threshold))

    def __repr__(self) -> str:
        """Stable repr."""
        return (
            f"ZScoreIntervalClassifier({self.threshold}, error={self.error}, "
            f"units={self.units!r})"
        )


class ThresholdClassifier:
    """Final decision: ``ACCEPTED`` where ``zscore > threshold``, else ``REJECTED``."""

    pointwise: ClassVar[bool] = True
    requires_error: ClassVar[bool] = False

    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)

    @torch.no_grad()
    def __call__(
        self,
        ledger: PixelLedger,
        pixel_index: torch.Tensor,
        stage: StageSpec,
        stage_index: int,
    ) -> None:
        """Threshold the stage's pixels."""
        idx = pixel_index.to(ledger.device, torch.long)
        accepted = ledger.zscore[idx] > self.threshold
        labels = torch.where(
            accepted,
            torch.tensor(
                int(PixelLabel.ACCEPTED), dtype=torch.int8, device=ledger.device
            ),
            torch.tensor(
                int(PixelLabel.REJECTED), dtype=torch.int8, device=ledger.device
            ),
        )
        ledger.set_labels(idx, labels)

    def __repr__(self) -> str:
        """Stable repr."""
        return f"ThresholdClassifier({self.threshold})"


class TopFractionSelector:
    """Keep the best ``fraction`` (or ``top_k``) pixels ``UNCERTAIN``, reject the rest.

    Parameters
    ----------
    fraction : float, optional
        Fraction of the stage's pixels to keep (``0 < fraction <= 1``).
    top_k : int, optional
        Absolute number to keep (mutually exclusive with ``fraction``).
    key : {"zscore", "mip"}, optional
        Ranking statistic. Defaults to ``"zscore"``.
    per_image : bool, optional
        Rank within each image plane instead of globally.
    """

    pointwise: ClassVar[bool] = False
    requires_error: ClassVar[bool] = False

    def __init__(
        self,
        fraction: float | None = None,
        *,
        top_k: int | None = None,
        key: Literal["zscore", "mip"] = "zscore",
        per_image: bool = False,
    ) -> None:
        if (fraction is None) == (top_k is None):
            raise ValueError("give exactly one of fraction or top_k")
        if fraction is not None and not (0.0 < fraction <= 1.0):
            raise ValueError("fraction must lie in (0, 1]")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be >= 1")
        if key not in ("zscore", "mip"):
            raise ValueError("key must be 'zscore' or 'mip'")
        self.fraction = fraction
        self.top_k = top_k
        self.key = key
        self.per_image = per_image

    def _keep_count(self, n: int) -> int:
        if self.top_k is not None:
            return min(self.top_k, n)
        assert self.fraction is not None
        return max(1, round(self.fraction * n)) if n else 0

    @torch.no_grad()
    def __call__(
        self,
        ledger: PixelLedger,
        pixel_index: torch.Tensor,
        stage: StageSpec,
        stage_index: int,
    ) -> None:
        """Rank the stage's pixels and reject all but the top ones."""
        idx = pixel_index.to(ledger.device, torch.long)
        values = getattr(ledger, self.key)[idx]
        labels = torch.full_like(idx, int(PixelLabel.REJECTED), dtype=torch.int8)
        if self.per_image:
            image_of = idx // ledger.geometry.n_pixels_per_image
            for b in torch.unique(image_of).tolist():
                members = torch.nonzero(image_of == b, as_tuple=False).squeeze(-1)
                k = self._keep_count(int(members.numel()))
                if k:
                    top = torch.topk(values[members], k).indices
                    labels[members[top]] = int(PixelLabel.UNCERTAIN)
        else:
            k = self._keep_count(int(idx.numel()))
            if k:
                labels[torch.topk(values, k).indices] = int(PixelLabel.UNCERTAIN)
        ledger.set_labels(idx, labels)

    def __repr__(self) -> str:
        """Stable repr."""
        return (
            f"TopFractionSelector(fraction={self.fraction}, top_k={self.top_k}, "
            f"key={self.key!r}, per_image={self.per_image})"
        )


class DilationAbsorb:
    """Post-step: ``UNCERTAIN`` pixels near an ``ACCEPTED`` one become ``ABSORBED``."""

    pointwise: ClassVar[bool] = False
    requires_error: ClassVar[bool] = False

    def __init__(self, radius: int) -> None:
        if radius < 0:
            raise ValueError("radius must be >= 0")
        self.radius = int(radius)

    def __call__(
        self,
        ledger: PixelLedger,
        pixel_index: torch.Tensor,
        stage: StageSpec,
        stage_index: int,
    ) -> None:
        """Absorb on the dense label map."""
        ledger.absorb_near_accepted(self.radius)

    def __repr__(self) -> str:
        """Stable repr."""
        return f"DilationAbsorb({self.radius})"


class DecisionChain:
    """Apply decisions in order; later ones only see labels left by earlier ones."""

    def __init__(self, *decisions: Decision) -> None:
        if not decisions:
            raise ValueError("DecisionChain needs at least one decision")
        self.decisions = tuple(decisions)

    @property
    def pointwise(self) -> bool:
        """True when every member is pointwise."""
        return all(d.pointwise for d in self.decisions)

    @property
    def requires_error(self) -> bool:
        """True when any member needs the stage error."""
        return any(d.requires_error for d in self.decisions)

    def __call__(
        self,
        ledger: PixelLedger,
        pixel_index: torch.Tensor,
        stage: StageSpec,
        stage_index: int,
    ) -> None:
        """Run every member in order."""
        for decision in self.decisions:
            decision(ledger, pixel_index, stage, stage_index)

    def __repr__(self) -> str:
        """Stable repr."""
        return f"DecisionChain({', '.join(repr(d) for d in self.decisions)})"


class SurpassThresholdDecision:
    """Per-pixel decision through :func:`surpass_threshold`.

    Parameters
    ----------
    threshold : float
        The cutoff ``T``.
    map_names : tuple[str, str], optional
        Names of the image-norm and noise-variance maps in ``ledger.aux``.
    rank : int, optional
        Rank passed to the rule; defaults to the stage's feature count.
    calibration : object, optional
        Forwarded to :func:`surpass_threshold`.
    pass_label, fail_label : PixelLabel, optional
        Labels for passing / failing pixels. ``pass_label=PixelLabel.UNCERTAIN``
        turns the decision into a follow-up filter for a later stage.
    """

    pointwise: ClassVar[bool] = True
    requires_error: ClassVar[bool] = False

    def __init__(
        self,
        threshold: float,
        *,
        map_names: tuple[str, str] = ("image_norm", "noise_variance"),
        rank: int | None = None,
        calibration: Any | None = None,
        pass_label: PixelLabel = PixelLabel.ACCEPTED,
        fail_label: PixelLabel = PixelLabel.REJECTED,
    ) -> None:
        self.threshold = float(threshold)
        self.map_names = map_names
        self.rank = rank
        self.calibration = calibration
        self.pass_label = pass_label
        self.fail_label = fail_label

    @torch.no_grad()
    def __call__(
        self,
        ledger: PixelLedger,
        pixel_index: torch.Tensor,
        stage: StageSpec,
        stage_index: int,
    ) -> None:
        """Label the stage's pixels."""
        idx = pixel_index.to(ledger.device, torch.long)
        maps = []
        for name in self.map_names:
            if name not in ledger.aux:
                raise KeyError(
                    f"ledger has no aux map {name!r}; available: "
                    f"{sorted(ledger.aux)} (supply ImageSet.pixel_maps)"
                )
            maps.append(ledger.aux[name][idx])
        passed = surpass_threshold(
            ledger.mip[idx],
            ledger.mean[idx],
            ledger.variance[idx],
            threshold=self.threshold,
            image_norm=maps[0],
            noise_variance=maps[1],
            rank=self.rank or stage.num_features,
            calibration=self.calibration,
        )
        labels = torch.where(
            passed,
            torch.tensor(int(self.pass_label), dtype=torch.int8, device=ledger.device),
            torch.tensor(int(self.fail_label), dtype=torch.int8, device=ledger.device),
        )
        ledger.set_labels(idx, labels)

    def __repr__(self) -> str:
        """Stable repr."""
        return (
            f"SurpassThresholdDecision({self.threshold}, map_names={self.map_names}, "
            f"rank={self.rank}, pass_label={self.pass_label.name}, "
            f"fail_label={self.fail_label.name})"
        )


class PairEmissionDecision:
    """The decision of a pair-emitting stage: only pixels with a pair stay eligible."""

    pointwise: ClassVar[bool] = False
    requires_error: ClassVar[bool] = False

    @torch.no_grad()
    def __call__(
        self,
        ledger: PixelLedger,
        pixel_index: torch.Tensor,
        stage: StageSpec,
        stage_index: int,
    ) -> None:
        """Relabel from the stage's recorded pairs."""
        if stage_index not in ledger.pairs:
            raise RuntimeError(
                f"stage {stage_index} emitted no pairs (StageSpec.pairs)"
            )
        idx = pixel_index.to(ledger.device, torch.long)
        keep = ledger.pairs[stage_index].pixels().to(ledger.device)
        ledger.label[idx] = int(PixelLabel.REJECTED)
        ledger.label[keep] = int(PixelLabel.UNCERTAIN)

    def __repr__(self) -> str:
        """Stable repr."""
        return "PairEmissionDecision()"
