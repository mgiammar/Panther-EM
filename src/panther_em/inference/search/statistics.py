"""Per-pixel search statistics tracked online over a batch of hypotheses.

:class:`PixelStats` accumulates the first two moments plus the running best correlation
(and the hypothesis / in-plane angle that produced it) for a batch of image pixels, so a
search never has to materialize the full ``(pixels, hypotheses, psi)`` correlogram.
"""

from __future__ import annotations

import torch


class PixelStats:
    """Helper for tracing statistics over hypothesis space in a small pixel batch.

    Attributes
    ----------
    hypothesis_count : int
        The number of hypotheses processed.
    num_pixels : int
        The number of pixels in the batch.
    corr_sum : torch.Tensor
        The sum of correlation values.
    corr_sum2 : torch.Tensor
        The sum of squared correlation values.
    best_corr : torch.Tensor
        The best correlation value found.
    best_hypothesis : torch.Tensor
        Global index corresponding to the hypothesis which produced the best correlation
        value.
    best_psi_angle : torch.Tensor
        The in-plane rotation angle (also a hypothesis, but separate axis) which
        produced the best correlation value.

    Methods
    -------
    clear : None
        Clear all tracked statistics.
    update(corr: torch.Tensor, hyp_global_idx: torch.Tensor) : None
        Update tracked statistics with new corr values and hypothesis indices.
    finalize : dict[str, torch.Tensor]
        Return a dictionary of the final statistics (maximum intensity projection /
        maximum inner product, z-score, mean, variance, best hypothesis index,
        best psi index).
    """

    hypothesis_count: int
    num_pixels: int
    corr_sum: torch.Tensor
    corr_sum2: torch.Tensor
    best_corr: torch.Tensor
    best_hypothesis: torch.Tensor
    best_psi_angle: torch.Tensor

    def __init__(
        self,
        num_pixels: int,
        device: torch.device,
        accumulate_dtype: torch.dtype = torch.float32,
    ):
        """Initialize a new pixel statistics tracker."""
        self.hypothesis_count = 0
        self.num_pixels = num_pixels
        self.corr_sum = torch.zeros(num_pixels, device=device, dtype=accumulate_dtype)
        self.corr_sum2 = torch.zeros(num_pixels, device=device, dtype=accumulate_dtype)
        self.best_corr = torch.full(
            (num_pixels,), float("-inf"), device=device, dtype=accumulate_dtype
        )
        self.best_hypothesis = torch.full(
            (num_pixels,), -1, device=device, dtype=torch.long
        )
        self.best_psi_angle = torch.full(
            (num_pixels,), -1, device=device, dtype=torch.long
        )

    @torch.no_grad()
    def clear(self) -> None:
        """Clear all tracked statistics."""
        self.hypothesis_count = 0
        self.corr_sum.zero_()
        self.corr_sum2.zero_()
        self.best_corr.fill_(float("-inf"))
        self.best_hypothesis.fill_(-1)
        self.best_psi_angle.fill_(-1)

    @torch.no_grad()
    def update(self, corr: torch.Tensor, hyp_global_idx: torch.Tensor) -> None:
        """Update tracked statistics with new corr values and hypothesis indices.

        Parameters
        ----------
        corr : torch.Tensor
            Tensor with correlation values with shape (num_pixels, hyp_batch, num_psi)
            where `num_pixels` must match held in `self.num_pixels`, `hyp_batch` is
            number of hypotheses in the batch, and `num_psi` are all in-plane angles.
        hyp_global_idx : torch.Tensor
            Integer tensor referencing which indices these hypotheses correspond to,
            shape (hyp_batch,).
        """
        corr_cast = corr.to(self.best_corr.dtype)
        num_psi = corr.shape[2]  # in-plane angle axis
        total_hypotheses = corr.shape[1] * corr.shape[2]  # num(other) * num(in-plane)

        ### 1. Update sum and squared sum statistics
        self.corr_sum += corr_cast.sum(dim=(1, 2))
        self.corr_sum2 += (corr_cast**2).sum(dim=(1, 2))
        self.hypothesis_count += total_hypotheses

        ### 2. Conditional update on best correlation and hypotheses
        corr_flat = corr_cast.reshape(-1, total_hypotheses)
        vmax, amax = corr_flat.max(dim=1)
        mask = vmax > self.best_corr

        if not mask.any():  # Short circuit if none of the indices improve
            return

        # Decode indices for all pixels to undo flattening of last dim
        n_local = torch.div(amax, num_psi, rounding_mode="floor")
        psi = amax % num_psi
        global_hyp = hyp_global_idx[n_local]

        self.best_corr[mask] = vmax[mask]
        self.best_hypothesis[mask] = global_hyp[mask]
        self.best_psi_angle[mask] = psi[mask]

    @torch.no_grad()
    def finalize(self) -> dict[str, torch.Tensor]:
        """Finalize tracked statistics into summary dictionary.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the final tracked statistics. Tensors exists on device
            used to initialize the PixelStats object.
        """
        mean = self.corr_sum / self.hypothesis_count
        variance = (self.corr_sum2 / self.hypothesis_count) - (mean**2)
        variance.clamp_min_(0.0)
        std = variance.sqrt()
        zscore = (self.best_corr - mean) / std.clamp_min(1e-12)

        return {
            "mip": self.best_corr,
            "zscore": zscore,
            "mean": mean,
            "variance": variance,
            "best_index": self.best_hypothesis,
            "best_psi": self.best_psi_angle,
        }


# ---------------------------------------------------------------------------
# Error-aware partitioning (multi-precision follow-up) -- NOT YET IMPLEMENTED
# ---------------------------------------------------------------------------
#
# Between incremental stages the reconstructed correlations are partitioned,
# using the reconstruction-accuracy bounds implied by the retained singular
# values, into three sets:
#
#   * accepted   -- score far above expectation; record and stop refining.
#   * rejected   -- score far below expectation; record and stop refining.
#   * follow-up  -- ambiguous; carry forward to the next (higher-rank) stage.
#
# The follow-up set becomes the pixel mask (and joint hypothesis mask) handed to the
# next stage. This is the home for that logic; it is intentionally unimplemented for
# now.
