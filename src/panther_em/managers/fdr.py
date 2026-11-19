r"""False-discovery-rate detection threshold, set per micrograph after the search.

The search's default (``SearchConfig.fdr = 0.05``): instead of one fixed z threshold
that gives one expected false detection per micrograph, the final picks are the peaks
above the Benjamini-Hochberg threshold at false discovery rate ``q``.

Null model. The expected number of false peaks above ``t`` on the micrograph,
:math:`\lambda(t)`. By default (``fdr_null: analytic``) it is the micrograph's analytic
null, computed by the search from its own template, filters and grid and the
micrograph's power spectrum (:func:`panther_em.managers.null.predict_search_null`).
``fdr_null: formula`` uses the independent-trial count, :math:`\lambda(t) =
N_\mathrm{pos} N_\mathrm{hyp} Q(t)` (the formula behind the 1-FP threshold,
:func:`~panther_em.search.zscore_threshold`). The search's trials are correlated, so
the formula over-counts false peaks (on xenon_131 after CTF the null is 0.37 of it at
B 0 and 0.14 at B 120), and the realised FDR falls below ``q``. A ``panther-em null``
summary (its analytic prediction or its phase-randomised copies) can be given instead
(:func:`calibrated_null_rate`).

Rule (step-up, :math:`\pi_0 = 1`). With the peak z-scores sorted in decreasing order,
:math:`z_{(1)} \ge z_{(2)} \ge \dots`, the threshold is :math:`z_{(k)}` for the largest
``k`` with :math:`\lambda(z_{(k)}) \le q k`; no such ``k``: no picks. The rule is
Leopard-EM's engine-agnostic :func:`leopard_em.analysis.fdr.bh_cutoff`.

Peaks are local maxima at least ``peak_radius`` apart among the pixels the final stage
ACCEPTED, found as the particle export finds them (``find_peaks_from_zscore``), so the
count ``k`` is the number of particles the export would write.

Reach. The threshold is only known after the search, but the stages decide per pixel
during it. With ``fdr`` set, the final stage's ``"auto"`` threshold is therefore
lowered ``fdr_reach`` (0.5) z below the 1-FP threshold, which makes it the floor of the
FDR cut. The earlier stages' thresholds (the stage-1 gate) are lowered by
``fdr_gate_reach`` (0.25), because the gate keeps pixels about 0.3 z below its own
threshold. The final stage thus still evaluates exactly the pixels an FDR cut down to
the floor could pick, at 1.6 to 1.9 times the search time, instead of the ~6x a gate
lowered by the full 0.5 would cost.

The 1-FP threshold the reach is measured from is the null's own (``fdr_anchor:
null``, the default). The null is computed before the search (:class:`FdrNull`), and
every ``"auto"`` threshold moves by its 1-FP cut minus the formula's. A blurred
template gives a smoother field, whose null, and with it the FDR cut, falls well below
the formula: on Track S after CTF the null's 1-FP cut is 7.78, 7.67 and 7.58 at B 0,
60 and 120, against the formula's 7.86. With the reach measured from the formula, the
floor stays at 7.36 at every B. At B 120 it sits 0.03 to 0.12 z below the cuts on the
dense micrographs, and on one micrograph it stopped the cut.

A BH threshold below the floor is clamped to it (``capped``). Since the search keeps
no peaks below the floor, a cut the floor stops usually shows instead as every peak
picked. ``floor_limited`` in the report flags either case.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from leopard_em.analysis.fdr import NullRate, bh_cutoff, nominal_null_rate, null_rate
from scipy.special import ndtr

from panther_em.search import PixelLabel

if TYPE_CHECKING:
    from pathlib import Path

    from panther_em.search import SearchResult


def expected_false_peaks(
    z: torch.Tensor, n_positions: int, n_hypotheses: int
) -> torch.Tensor:
    r""":math:`\lambda(z) = N_\mathrm{pos} N_\mathrm{hyp} Q(z)`, float64."""
    sf = torch.from_numpy(ndtr(-z.double().cpu().numpy()))  # Q(z), accurate far out
    return float(n_positions) * float(n_hypotheses) * sf


def bh_threshold(
    peak_z: torch.Tensor,
    n_positions: int,
    n_hypotheses: int,
    q: float,
    rate: NullRate | None = None,
) -> float | None:
    """Benjamini-Hochberg threshold of ``peak_z`` at FDR ``q`` (``None``: no picks).

    The null is ``rate`` (expected false peaks >= z), by default the formula's.
    """
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie in (0, 1)")
    if rate is None:
        rate = nominal_null_rate(float(n_positions) * float(n_hypotheses))
    z = peak_z.double().cpu().numpy().reshape(-1)
    threshold: float | None = bh_cutoff(z, rate, q).bh_threshold
    return threshold


@dataclasses.dataclass(frozen=True)
class FdrNull:
    """A micrograph's FDR null, prepared before its search.

    Attributes
    ----------
    rate : NullRate or None
        Expected false peaks >= z on the micrograph; ``None``: the formula's.
    record : dict
        Where the null came from and where it anchors the thresholds (kept in the
        report).
    shift : float
        How far every ``"auto"`` stage threshold moves: the null's 1-FP cut minus
        the formula's (``fdr_anchor: null``), else 0.
    """

    rate: NullRate | None
    record: dict[str, Any]
    shift: float = 0.0


def null_cut_1fp(
    rate: NullRate, lo: float = 2.0, hi: float = 12.0, step: float = 1e-3
) -> float | None:
    """The z at which ``rate`` falls to one false peak per micrograph.

    Log-linear between the two grid points around the crossing; ``None`` when the
    rate does not fall below one within ``[lo, hi]`` or is below one from ``lo``.
    """
    z = np.arange(lo, hi + step / 2, step)
    r = np.asarray(rate(z), float)
    below = np.flatnonzero(r < 1.0)
    if below.size == 0 or below[0] == 0:
        return None
    i = int(below[0])
    r0, r1 = r[i - 1], r[i]
    if not (np.isfinite(r0) and r1 > 0.0):
        return float(z[i])
    frac = np.log(r0) / (np.log(r0) - np.log(r1))
    return float(z[i - 1] + frac * (z[i] - z[i - 1]))


def calibrated_null_rate(
    path: str | Path,
    name: str,
    n_positions: int,
    source: Literal["analytic", "empirical"] = "analytic",
) -> tuple[NullRate, dict[str, Any]] | None:
    """A micrograph's calibrated null from a ``panther-em null`` summary.

    Parameters
    ----------
    path : str or Path
        ``null_summary.json`` of ``panther-em null`` for the same search config.
    name : str
        The micrograph.
    n_positions : int
        Its searched positions: the null's rate (over its own valid area) is scaled
        to them.
    source : {"analytic", "empirical"}
        The analytic prediction, or the phase-randomised copies' rate table.

    Returns
    -------
    tuple or None
        The rate and a record of where it came from; ``None`` when the summary has
        no such null for the micrograph.
    """
    with open(path, encoding="utf-8") as f:
        record = json.load(f).get("micrographs", {}).get(name)
    if not record or "valid_area" not in record:
        return None
    table = record.get("analytic") if source == "analytic" else record.get("rate")
    if not table or "thresholds" not in table or "rate" not in table:
        return None
    scale = float(n_positions) / float(record["valid_area"])
    rate = null_rate(
        (np.asarray(table["thresholds"], float), np.asarray(table["rate"], float)),
        area_scale=scale,
    )
    return rate, {"null": source, "null_summary": str(path), "area_scale": scale}


def _disk_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """``(H, W)`` bool: within ``radius`` (Euclidean) of a True pixel of ``mask``."""
    r = int(radius)
    yy, xx = torch.meshgrid(
        torch.arange(-r, r + 1), torch.arange(-r, r + 1), indexing="ij"
    )
    kernel = ((yy * yy + xx * xx) <= r * r).to(torch.float32)[None, None]
    hit = torch.nn.functional.conv2d(
        mask.to(torch.float32)[None, None], kernel.to(mask.device), padding=r
    )
    return hit[0, 0] > 0.5


def apply_fdr(
    result: SearchResult,
    *,
    q: float,
    n_positions: int,
    n_hypotheses: int,
    floor: float,
    peak_radius: float,
    absorb_radius: int = 0,
    rate: NullRate | None = None,
    null_record: dict[str, Any] | None = None,
) -> tuple[SearchResult, dict]:
    """``result`` with only the pixels at or above the BH threshold left ACCEPTED.

    Parameters
    ----------
    result : SearchResult
        A one-micrograph result whose final stage accepted every pixel above
        ``floor``.
    q : float
        False discovery rate.
    n_positions, n_hypotheses : int
        The search's positions per micrograph and hypotheses per position (the
        1-FP threshold's counts).
    floor : float
        The lowered stage threshold; a BH threshold below it is clamped to it.
    peak_radius : float
        Minimum distance between peaks, pixels (the export's ``peak_radius``).
    absorb_radius : int
        The final stage's absorb radius: ABSORBED pixels no longer within it of an
        ACCEPTED pixel go back to AMBIGUOUS.
    rate : NullRate, optional
        The null (expected false peaks >= z on this micrograph); default the
        formula's (``n_positions`` x ``n_hypotheses``).
    null_record : dict, optional
        Where ``rate`` came from, kept in the report.

    Returns
    -------
    tuple
        The relabelled result and a report: the BH threshold (``None`` if no
        peak qualifies), the applied threshold, whether it was capped at the floor,
        whether the floor limited it (``floor_limited``: capped, or every peak
        above the floor picked, so a lower floor might have picked more), the peaks
        considered and picked, the null used and the expected false picks under it
        (and under the formula).
    """
    from leopard_em.analysis.zscore_metric import find_peaks_from_zscore

    if int(result.label.shape[0]) != 1:
        raise ValueError("apply_fdr takes a one-micrograph result")
    label = result.label[0]
    zscore = result.zscore[0].detach().to(torch.float32)
    accepted = label == int(PixelLabel.ACCEPTED)
    masked = torch.where(accepted, zscore, torch.full_like(zscore, float("-inf"))).cpu()
    pos_y, pos_x = find_peaks_from_zscore(masked, float("-inf"), peak_radius)
    peak_z = masked[pos_y, pos_x] if pos_y.numel() else torch.zeros(0)
    peak_z = peak_z[torch.isfinite(peak_z)]
    t_bh = bh_threshold(peak_z, n_positions, n_hypotheses, q, rate)
    threshold = float("inf") if t_bh is None else max(t_bh, floor)
    capped = t_bh is not None and t_bh < floor

    new = label.clone()
    drop = accepted & (zscore.to(label.device) < threshold)
    new[drop] = int(PixelLabel.REJECTED)
    if absorb_radius > 0:
        absorbed = new == int(PixelLabel.ABSORBED)
        if bool(absorbed.any()):
            near = _disk_dilate(new == int(PixelLabel.ACCEPTED), absorb_radius)
            new[absorbed & ~near] = int(PixelLabel.AMBIGUOUS)
    labels = result.label.clone()
    labels[0] = new
    n_picks = int((peak_z >= threshold).sum())
    formula = None
    if t_bh is not None:
        formula = float(
            expected_false_peaks(torch.tensor([threshold]), n_positions, n_hypotheses)[
                0
            ]
        )
    expected = formula
    if t_bh is not None and rate is not None:
        expected = float(np.asarray(rate(np.array([threshold])), float)[0])
    # The peaks stop at the floor, so a BH cut can only fall below it by picking all
    # of them: that is where a lower floor might have picked more ("capped" alone
    # cannot see it).
    all_picked = t_bh is not None and n_picks == int(peak_z.numel())
    report = {
        "q": q,
        **(null_record or {"null": "formula" if rate is None else "given"}),
        "bh_threshold": t_bh,
        "threshold": None if t_bh is None else threshold,
        "floor": floor,
        "capped": capped,
        "floor_limited": bool(capped or all_picked),
        "n_peaks_considered": int(peak_z.numel()),
        "n_picks": n_picks,
        "expected_false_picks": expected,
        "expected_false_picks_formula": formula,
    }
    return dataclasses.replace(result, label=labels), report
