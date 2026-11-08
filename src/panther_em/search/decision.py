"""The per-pixel decision: the threshold test rejects, the z-score threshold accepts.

Every stage may end with a :class:`Decision`. With a
:class:`~.threshold.ThresholdTest` it REJECTS the pixels whose exact correlation,
predicted from the stage's maps under the coloured-noise model of the micrograph
(:class:`~.micrograph.Micrograph`), cannot reach the threshold; survivors are ACCEPTED
where the stage's own z-score exceeds the threshold -- when ``accept`` is set, the
final stage -- and AMBIGUOUS otherwise. Without a test the rule is the plain
threshold: REJECTED below it, ACCEPTED above it. An optional post-step ABSORBS
ambiguous pixels within a radius of an accepted one (the same particle's Neighbors).

The test's distance ``D`` of every tested pixel is kept as its ``score``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from panther_em.search.threshold import ThresholdTest

if TYPE_CHECKING:
    from panther_em.search.hypotheses import HypothesisSet
    from panther_em.search.ledger import PixelLedger
    from panther_em.search.micrograph import Micrograph


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
        images: Micrograph | None = None,
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
        images : Micrograph, optional
            The micrograph's noise model; required by a threshold test.
        """
        idx = pixel_index.to(ledger.device, torch.long)
        zscore = ledger.zscore[idx]
        if self.test is None:
            rejected = zscore < self.threshold
        else:
            if images is None:
                raise ValueError(
                    "the decision's threshold test needs the micrograph's noise model "
                    "(a Micrograph)"
                )
            score = self.score(ledger, idx, hyps, images)
            rejected = score >= self.test.calibration.z_alpha
            ledger.set_scores(idx, score)
        labels = torch.full_like(idx, int(PixelLabel.AMBIGUOUS), dtype=torch.int8)
        labels[rejected] = int(PixelLabel.REJECTED)
        if self.accept:
            labels[~rejected & (zscore > self.threshold)] = int(PixelLabel.ACCEPTED)
        ledger.set_labels(idx, labels)
        if self.absorb_radius > 0:
            ledger.absorb_near_accepted(self.absorb_radius)

    @torch.no_grad()
    def score(
        self,
        ledger: PixelLedger,
        pixel_index: torch.Tensor,
        hyps: HypothesisSet,
        images: Micrograph,
    ) -> torch.Tensor:
        """The test's distance ``D`` at ``pixel_index`` (NaN where nothing was found).

        Gathers the ledger's maps, the level at each pixel's box centre and the
        ``(plane, template row)`` of its maximum, and hands them to the test.
        """
        if self.test is None:
            raise ValueError("a decision without a threshold test has no score")
        geometry = ledger.geometry
        if geometry.n_images != 1 or geometry.image_shape != tuple(images.level.shape):
            raise ValueError(
                f"the ledger's {geometry.n_images} x {geometry.image_shape} images are "
                f"not the noise model's {tuple(images.level.shape)} micrograph"
            )
        idx = pixel_index.to(ledger.device, torch.long)
        scales = self.test.scales(
            images, rows=hyps.template_rows(), planes=hyps.defocus_planes()
        )
        _, rows, cols = geometry.positions(idx)
        k_h, k_w = geometry.template_shape
        level = images.level.to(ledger.device)[
            rows + (k_h - 1) // 2, cols + (k_w - 1) // 2
        ]
        best = ledger.best_index[idx].to(torch.long)
        valid = best >= 0
        best = torch.where(valid, best, torch.zeros_like(best))
        n_templates = self.test.table.n_templates
        dist = self.test.distance(
            scales,
            self.threshold,
            level=level,
            plane=best // n_templates,
            row=best % n_templates,
            mip=ledger.mip[idx],
            mean=ledger.mean[idx],
            variance=ledger.variance[idx],
        )
        return torch.where(valid, dist, torch.full_like(dist, float("nan")))
