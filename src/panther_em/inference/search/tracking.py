"""Error-aware per-pixel tracking for multi-precision incremental search."""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import numpy as np
import torch
from skimage.morphology import isotropic_dilation

from panther_em.inference.search.compressed import search_output_shape
from panther_em.inference.search.incremental import compute_zscore_error_intervals

if TYPE_CHECKING:
    from collections.abc import Sequence

    from panther_em.inference.projection_reconstruction import ProjectionReconstructor


class PixelLabel(enum.IntEnum):
    """Per-pixel status tracked by :class:`MultiPrecisionPixelTracker`.

    - PENDING: Pixel has not yet been searched.
    - UNCERTAIN: Pixel has been searched, but z-score interval straddles the threshold.
    - REJECTED: Pixel has been searched, and z-score interval is below the threshold.
    - ACCEPTED: Pixel has been searched, and z-score interval is above the threshold.
    - ABSORBED: Pixel has been searched, and was UNCERTAIN but is near an ACCEPTED pixel
                and was absorbed by morphological dilation.
    """

    PENDING = 0
    UNCERTAIN = 1
    REJECTED = 2
    ACCEPTED = 3
    ABSORBED = 4


def classify_zscore_interval(
    interval: torch.Tensor,  # (..., 2), [lower, upper]
    threshold: float,
) -> torch.Tensor:  # (...,) int8
    r"""Classify worst-case z-score intervals against a fixed threshold.

    Parameters
    ----------
    interval : torch.Tensor
        Per-pixel ``[lower_bound, upper_bound]`` intervals, shape ``(..., 2)``, as
        returned by
        :func:`~panther_em.inference.search.incremental.compute_zscore_error_intervals`.
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


class MultiPrecisionPixelTracker:
    """Stateful, error-aware pixel partitioning for multi-stage incremental search.

    Parameters
    ----------
    stage_errors : Sequence[float]
        One reconstruction-error scalar per stage, in the same order as the
        ``stage_rectangles`` passed to ``incremental_search``.
    zscore_threshold : float
        Fixed z-score decision threshold, constant across all stages.
    dilation_radius : int, optional
        Pixel radius for absorbing ``UNCERTAIN`` pixels near ``ACCEPTED`` ones via
        morphological dilation. ``0`` (the default) disables absorption.
    out_shape : tuple[int, int, int], optional
        ``(b, out_h, out_w)`` valid-correlation grid shape. Mutually exclusive with
        ``reconstructor`` + ``image`` (exactly one of the two must be given).
    reconstructor : ProjectionReconstructor, optional
        Used with ``image`` to derive ``out_shape`` via
        :func:`~panther_em.inference.search.compressed.search_output_shape`.
    image : torch.Tensor, optional
        Used with ``reconstructor`` to derive ``out_shape``.
    device : torch.device | str, optional
        Device for the dense per-pixel state. Defaults to ``reconstructor.device``
        if ``reconstructor`` was given, else CPU. Should match the `incremental_search``
        to avoid a device-transfer on every :meth:`step` call.
    stats_dtype : torch.dtype, optional
        Dtype for the stored ``mip``/``mean``/``std``/``zscore`` state. Defaults to
        ``torch.float32``.

    Attributes
    ----------
    status : torch.Tensor
        ``(P_total,)`` int8 :class:`PixelLabel` values, dense over every pixel.
    mip, mean, std, zscore : torch.Tensor
        ``(P_total,)`` last-known per-pixel statistics, ``nan`` for ``PENDING``
        pixels.
    best_index, best_psi : torch.Tensor
        ``(P_total,)`` int64 last-known best hypothesis / psi index, ``-1`` for
        ``PENDING`` pixels.
    last_stage : torch.Tensor
        ``(P_total,)`` int64, the 0-indexed stage that last wrote each pixel, ``-1``
        for ``PENDING`` pixels.
    out_shape : tuple[int, int, int]
        ``(b, out_h, out_w)`` valid-correlation grid shape.
    n_px_total : int
        ``b * out_h * out_w``.
    """

    def __init__(
        self,
        stage_errors: Sequence[float],
        *,
        zscore_threshold: float,
        dilation_radius: int = 0,
        out_shape: tuple[int, int, int] | None = None,
        reconstructor: ProjectionReconstructor | None = None,
        image: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        stats_dtype: torch.dtype = torch.float32,
    ) -> None:
        have_out_shape = out_shape is not None
        have_recon_image = reconstructor is not None and image is not None

        if have_out_shape == have_recon_image:
            raise ValueError(
                "pass exactly one of `out_shape` or (`reconstructor` and `image`)"
            )

        if dilation_radius < 0:
            raise ValueError(f"dilation_radius must be >= 0, got {dilation_radius}")

        if out_shape is None:
            out_shape = search_output_shape(image, reconstructor)  # type: ignore[arg-type]
        b, out_h, out_w = (int(v) for v in out_shape)

        self.out_shape: tuple[int, int, int] = (b, out_h, out_w)
        self.n_px_total = b * out_h * out_w

        self.device = torch.device(
            device
            if device is not None
            else (reconstructor.device if reconstructor is not None else "cpu")
        )
        self.stage_errors = list(stage_errors)
        self.zscore_threshold = float(zscore_threshold)
        self.dilation_radius = int(dilation_radius)
        self._stage = 0

        n = self.n_px_total
        self.status = torch.full(
            (n,), int(PixelLabel.PENDING), dtype=torch.int8, device=self.device
        )
        self.mip = torch.full((n,), float("nan"), dtype=stats_dtype, device=self.device)
        self.mean = torch.full(
            (n,), float("nan"), dtype=stats_dtype, device=self.device
        )
        self.std = torch.full((n,), float("nan"), dtype=stats_dtype, device=self.device)
        self.zscore = torch.full(
            (n,), float("nan"), dtype=stats_dtype, device=self.device
        )
        self.best_index = torch.full((n,), -1, dtype=torch.int64, device=self.device)
        self.best_psi = torch.full((n,), -1, dtype=torch.int64, device=self.device)
        self.last_stage = torch.full((n,), -1, dtype=torch.int64, device=self.device)

    @torch.no_grad()
    def step(
        self,
        stage_maps: dict[str, torch.Tensor],
        pixel_index: torch.Tensor,
    ) -> torch.Tensor:
        """One ``incremental_search`` stage's ``follow_up_fn`` callback.

        Parameters
        ----------
        stage_maps : dict[str, torch.Tensor]
            This stage's finalized ``PixelStats.finalize()``-shaped maps (``"mip"``,
            ``"mean"``, ``"variance"``, ``"best_index"``, ``"best_psi"``), each
            ``(P_stage,)``, indexed by ``pixel_index``.
        pixel_index : torch.Tensor
            The flat pixel indices (into ``[0, n_px_total)``) this stage ran on.

        Returns
        -------
        torch.Tensor
            Flat indices of pixels still labeled ``UNCERTAIN`` after this stage's
            classification and absorption step -- the next stage's ``pixel_index``.

        Raises
        ------
        IndexError
            If called more times than ``len(stage_errors)``.
        """
        if self._stage >= len(self.stage_errors):
            raise IndexError(
                f"tracker.step() called {self._stage + 1} times but only "
                f"{len(self.stage_errors)} stage_errors were supplied"
            )
        error = self.stage_errors[self._stage]
        self._stage += 1

        px = pixel_index.to(device=self.device, dtype=torch.long)
        mip = stage_maps["mip"].to(device=self.device, dtype=self.mip.dtype)
        mean = stage_maps["mean"].to(device=self.device, dtype=self.mean.dtype)
        variance = stage_maps["variance"].to(device=self.device, dtype=self.mean.dtype)
        std = variance.clamp_min(0.0).sqrt()

        interval = compute_zscore_error_intervals(error, mip, mean, std)  # (P_stage, 2)
        raw_label = classify_zscore_interval(interval, self.zscore_threshold)

        self.status[px] = raw_label
        self.mip[px] = mip
        self.mean[px] = mean
        self.std[px] = std
        self.zscore[px] = (mip - mean) / std.clamp_min(1e-12)
        self.best_index[px] = stage_maps["best_index"].to(self.device)
        self.best_psi[px] = stage_maps["best_psi"].to(self.device)
        self.last_stage[px] = self._stage - 1

        if self.dilation_radius > 0:
            self._absorb_uncertain_near_accepted()

        return torch.nonzero(
            self.status == int(PixelLabel.UNCERTAIN), as_tuple=False
        ).squeeze(-1)

    def _absorb_uncertain_near_accepted(self) -> None:
        b, out_h, out_w = self.out_shape
        status_grid = self.status.reshape(b, out_h, out_w)
        accepted = (status_grid == int(PixelLabel.ACCEPTED)).cpu().numpy()
        uncertain = (status_grid == int(PixelLabel.UNCERTAIN)).cpu().numpy()

        absorbed = np.zeros_like(uncertain)
        for i in range(b):
            if not accepted[i].any():
                continue
            dilated = isotropic_dilation(accepted[i], radius=self.dilation_radius)
            absorbed[i] = dilated & uncertain[i]

        if absorbed.any():
            idx = torch.from_numpy(absorbed.reshape(-1)).to(self.device)
            self.status[idx] = int(PixelLabel.ABSORBED)

    @torch.no_grad()
    def finalize(self, *, reshape: bool = False) -> dict[str, torch.Tensor]:
        """Aggregate every pixel's last-known status and statistics.

        Parameters
        ----------
        reshape : bool, optional
            Reshape every returned tensor to ``self.out_shape``
            (``(b, out_h, out_w)``) instead of leaving it flat ``(P_total,)``.
            Defaults to ``False``, matching ``incremental_search``'s flat
            pixel-index convention.

        Returns
        -------
        dict[str, torch.Tensor]
            ``"status"`` (int8, :class:`PixelLabel` values -- ``PENDING`` for
            pixels no stage ever touched), ``"mip"``, ``"mean"``, ``"std"``,
            ``"zscore"``, ``"best_index"``, ``"best_psi"`` (the last-known
            ``PixelStats.finalize`` schema, with ``std`` in place of
            ``variance``), and ``"last_stage"`` (int64, the 0-indexed stage that
            last wrote each pixel, ``-1`` for ``PENDING`` pixels). Pixels still
            ``UNCERTAIN`` at the end of the run stay labeled ``UNCERTAIN`` -- there
            is no forced final collapse to accepted/rejected.
        """
        out = {
            "status": self.status,
            "mip": self.mip,
            "mean": self.mean,
            "std": self.std,
            "zscore": self.zscore,
            "best_index": self.best_index,
            "best_psi": self.best_psi,
            "last_stage": self.last_stage,
        }
        if reshape:
            out = {key: value.reshape(self.out_shape) for key, value in out.items()}
        return {key: value.clone() for key, value in out.items()}
