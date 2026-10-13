"""Stage schedules: which features, at which precision, decided how."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from panther_em.inference.search.decisions import (
    Decision,
    NoDecision,
)
from panther_em.inference.search.tiling import FeatureLayout, rectangles_disjoint
from panther_em.inference.search.types import Precision, Rectangle

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from panther_em.decomposition.result import DecompositionResult
    from panther_em.inference.search.hypotheses import HypothesisSubset


def _as_rectangles(rectangles: Sequence[Sequence[int]]) -> tuple[Rectangle, ...]:
    out: list[Rectangle] = []
    for rect in rectangles:
        if len(rect) != 4:
            raise ValueError(
                f"a rectangle is (k_start, m_start, k_extent, m_extent), got {rect!r}"
            )
        k0, m0, k_ext, m_ext = (int(v) for v in rect)
        if k0 < 0 or m0 < 0:
            raise ValueError(f"rectangle starts must be >= 0, got {rect!r}")
        if k_ext <= 0 or m_ext <= 0:
            raise ValueError(f"rectangle extents must be > 0, got {rect!r}")
        out.append((k0, m0, k_ext, m_ext))
    return tuple(out)


def _normalize_rectangles(
    rectangles: Rectangle | Sequence[Rectangle],
) -> tuple[Rectangle, ...]:
    """Accept one rectangle or a sequence of them."""
    if len(rectangles) > 0 and isinstance(rectangles[0], int):
        return _as_rectangles([cast("Sequence[int]", rectangles)])
    return _as_rectangles(cast("Sequence[Sequence[int]]", rectangles))


@dataclass(frozen=True, eq=False)
class StageSpec:
    """One stage of the search.

    Parameters
    ----------
    rectangles : sequence of Rectangle
        Pairwise cell-disjoint ``(k_start, m_start, k_extent, m_extent)`` blocks of
        feature space contracted in this stage.
    precision : Precision or str, optional
        Contraction precision. Defaults to fp32.
    n_psi : int, optional
        In-plane samples recovered by the inverse FFT. Must satisfy
        ``k_stop <= n_psi // 2 + 1``. Defaults to 256 (the fused kernel's fast path).
    error : float, optional
        **Relative** L2 reconstruction error ``rho`` of the templates from this
        stage's features. ``None`` means "derive from an :class:`ErrorModel`".
    hypotheses : HypothesisSubset, optional
        Restrict this stage to a subset of the hypothesis space (e.g. a scout pass).
    name : str, optional
        Free-form label.
    """

    rectangles: tuple[Rectangle, ...]
    precision: Precision = Precision.FP32
    n_psi: int = 256
    error: float | None = None
    hypotheses: HypothesisSubset | None = None
    name: str = ""

    def __post_init__(self) -> None:
        """Normalise rectangles/precision and validate the psi cap."""
        rects = _as_rectangles(self.rectangles)
        if not rects:
            raise ValueError("a stage needs at least one rectangle")
        if not rectangles_disjoint(rects):
            raise ValueError(
                "rectangles within a stage must be pairwise cell-disjoint or the "
                "contraction would double count"
            )
        object.__setattr__(self, "rectangles", rects)
        object.__setattr__(self, "precision", Precision.coerce(self.precision))
        n_psi = int(self.n_psi)
        object.__setattr__(self, "n_psi", n_psi)
        k_stop = self.k_stop
        if k_stop > n_psi // 2 + 1:
            raise ValueError(
                f"k_stop ({k_stop}) exceeds n_psi // 2 + 1 ({n_psi // 2 + 1}); "
                "increase "
                f"n_psi to at least {2 * (k_stop - 1)} or cap k_extent at "
                f"{n_psi // 2 + 1} so the angular-frequency content is not aliased away"
            )
        if self.error is not None and not (0.0 <= float(self.error) < 1.0):
            raise ValueError(
                f"error must be a relative error in [0, 1), got {self.error}"
            )

    @property
    def k_stop(self) -> int:
        """Smallest angular-frequency axis length holding every rectangle."""
        return max(k0 + k_ext for k0, _, k_ext, _ in self.rectangles)

    @property
    def num_features(self) -> int:
        """Total ``(k, m)`` cells."""
        return sum(k_ext * m_ext for _, _, k_ext, m_ext in self.rectangles)

    def layout(self) -> FeatureLayout:
        """The CPU-side feature layout for these rectangles."""
        return FeatureLayout(self.rectangles)

    def with_error(self, error: float) -> StageSpec:
        """Copy with ``error`` set."""
        return replace(self, error=float(error))


@dataclass(frozen=True, eq=False)
class SearchSchedule:
    """Ordered stages plus the decision applied after each.

    Parameters
    ----------
    stages : sequence of StageSpec
        Stages in execution order (typically increasing rank).
    decisions : sequence of Decision, optional
        ``decisions[i]`` runs after ``stages[i]``. Defaults to :class:`NoDecision`
        for every stage.
    stop_when_empty : bool, optional
        Skip the remaining stages once no pixel is ``UNCERTAIN``. Defaults to True.
    keep_stage_results : bool, optional
        Retain every stage's compact result on the final :class:`SearchResult`.
    """

    stages: tuple[StageSpec, ...]
    decisions: tuple[Decision, ...] = field(default=())
    stop_when_empty: bool = True
    keep_stage_results: bool = False

    def __post_init__(self) -> None:
        """Default the decisions and check lengths."""
        stages = tuple(self.stages)
        if not stages:
            raise ValueError("a schedule needs at least one stage")
        decisions = tuple(self.decisions)
        if not decisions:
            decisions = tuple(NoDecision() for _ in stages)
        if len(decisions) != len(stages):
            raise ValueError(
                f"got {len(decisions)} decisions for {len(stages)} stages; "
                "need one per stage"
            )
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "decisions", decisions)

    def __len__(self) -> int:
        """Number of stages."""
        return len(self.stages)

    @classmethod
    def single(
        cls, stage: StageSpec, decision: Decision | None = None
    ) -> SearchSchedule:
        """A one-stage schedule."""
        return cls((stage,), (decision or NoDecision(),))

    def requires_errors(self) -> bool:
        """Whether some decision needs an error the stage does not carry."""
        return any(
            d.requires_error and s.error is None
            for s, d in zip(self.stages, self.decisions, strict=True)
        )

    def with_errors(
        self, model: ErrorModel, *, only_missing: bool = True
    ) -> SearchSchedule:
        """Fill stage errors from ``model`` (explicit ones are kept by default)."""
        stages = tuple(
            s if (only_missing and s.error is not None) else s.with_error(model.rho(s))
            for s in self.stages
        )
        return replace(self, stages=stages)


@dataclass(frozen=True, eq=False)
class ErrorModel:
    """Relative reconstruction error of a feature selection.

    TODO: Continue improving this class and the implemented models based on other
    analyses and tight error bound modeling.
    """

    S2: torch.Tensor | None = None  # (K, E) float64
    parseval: torch.Tensor | None = None  # (K,) float64
    tail: float = 0.0
    table: Mapping[tuple[int, int], float] | None = None

    def __post_init__(self) -> None:
        """Check that exactly one mode is configured."""
        if self.table is None:
            if self.S2 is None or self.parseval is None:
                raise ValueError("ErrorModel needs either (S2, parseval) or a table")
            if self.S2.dim() != 2 or tuple(self.parseval.shape) != (self.S2.shape[0],):
                raise ValueError("S2 must be (K, E) and parseval (K,)")
        if not (0.0 <= self.tail < 1.0):
            raise ValueError("tail must lie in [0, 1)")

    @classmethod
    def from_result(
        cls, result: DecompositionResult, *, tail: float = 0.0
    ) -> ErrorModel:
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

    @classmethod
    def from_table(
        cls,
        table: Mapping[tuple[int, int], float] | str | Path,
        *,
        column: str = "recon_error_used",
    ) -> ErrorModel:
        """Measured errors keyed by ``(k_extent, m_extent)``, or a CSV of them."""
        if isinstance(table, (str, Path)):
            rows: dict[tuple[int, int], float] = {}
            with open(table, newline="") as fh:
                for row in csv.DictReader(fh):
                    rows[(int(row["k_extent"]), int(row["m_extent"]))] = float(
                        row[column]
                    )
            table = rows
        return cls(table=dict(table))

    # -- spectral pieces -----------------------------------------------------

    def energy(self, rectangles: Rectangle | Sequence[Rectangle]) -> float:
        """Parseval-weighted sum of ``S^2`` over the rectangles' cells."""
        if self.S2 is None or self.parseval is None:
            raise ValueError("energy() needs the spectral model")
        total = 0.0
        k_blocks, e_max = self.S2.shape
        for k0, m0, k_ext, m_ext in _normalize_rectangles(rectangles):
            k1, m1 = min(k0 + k_ext, k_blocks), min(m0 + m_ext, e_max)
            if k1 <= k0 or m1 <= m0:
                continue
            block = self.S2[k0:k1, m0:m1] * self.parseval[k0:k1, None]
            total += float(block.sum())
        return total

    def total_energy(self) -> float:
        """Energy of everything stored."""
        if self.S2 is None or self.parseval is None:
            raise ValueError("total_energy() needs the spectral model")
        return float((self.S2 * self.parseval[:, None]).sum())

    def rho(self, selection: StageSpec | Rectangle | Sequence[Rectangle]) -> float:
        """Relative reconstruction error of a stage or rectangle set."""
        rects = (
            selection.rectangles
            if isinstance(selection, StageSpec)
            else _normalize_rectangles(selection)
        )
        if self.table is not None:
            if len(rects) != 1 or rects[0][0] != 0 or rects[0][1] != 0:
                raise ValueError(
                    "a table ErrorModel only covers single rectangles anchored "
                    "at (0, 0)"
                )
            key = (rects[0][2], rects[0][3])
            try:
                return float(self.table[key])
            except KeyError as exc:
                raise KeyError(
                    f"no measured error for (k_extent, m_extent) = {key}"
                ) from exc
        frac = self.energy(rects) / self.total_energy()
        return math.sqrt(max(0.0, 1.0 - (1.0 - self.tail) * frac))

    def decompose(self, k_extent: int, m_extent: int) -> tuple[float, float, float]:
        """``(rho, rho_k, rho_m)`` for a ``(0, 0, k_extent, m_extent)`` rectangle.

        ``rho_k`` is the angular-frequency truncation alone (all eigenvectors kept up
        to ``k_extent``), ``rho_m`` the eigenvector truncation within those
        frequencies; they compose in quadrature, ``rho^2 = rho_k^2 + rho_m^2``.
        """
        if self.S2 is None:
            raise ValueError("decompose() needs the spectral model")
        e_all = self.total_energy()
        e_k = self.energy((0, 0, k_extent, int(self.S2.shape[1])))
        e_km = self.energy((0, 0, k_extent, m_extent))
        scale = 1.0 - self.tail
        rho_k2 = max(0.0, 1.0 - scale * e_k / e_all)
        rho_m2 = max(0.0, scale * (e_k - e_km) / e_all)
        return math.sqrt(rho_k2 + rho_m2), math.sqrt(rho_k2), math.sqrt(rho_m2)


def _broadcast(value: object, n: int, name: str) -> list:
    if isinstance(value, (list, tuple)) and not (
        name == "decision" and len(value) == 0
    ):
        if len(value) != n:
            raise ValueError(f"{name} has {len(value)} entries for {n} stages")
        return list(value)
    return [value] * n


def nested_schedule(
    extents: Sequence[tuple[int, int]],
    *,
    precision: Precision | str | Sequence[Precision | str] = Precision.FP32,
    n_psi: int | Sequence[int] = 256,
    errors: Sequence[float | None] | None = None,
    error_model: ErrorModel | None = None,
    decision: Decision | Sequence[Decision] | None = None,
    scout: HypothesisSubset | None = None,
    stop_when_empty: bool = True,
    keep_stage_results: bool = False,
) -> SearchSchedule:
    """Schedule of nested ``(0, 0, k_extent, m_extent)`` rectangles.

    Parameters
    ----------
    extents : sequence of (k_extent, m_extent)
        One rectangle per stage, anchored at the origin of feature space.
    precision, n_psi : scalar or per-stage sequence
        Broadcast to every stage when scalar.
    errors : sequence of float or None, optional
        Explicit relative errors per stage (``None`` entries are derived).
    error_model : ErrorModel, optional
        Fills any missing errors.
    decision : Decision or per-stage sequence, optional
        Defaults to :class:`NoDecision` everywhere.
    scout : HypothesisSubset, optional
        Applied to the first stage only.
    stop_when_empty, keep_stage_results : bool, optional
        Forwarded to :class:`SearchSchedule`.
    """
    n = len(extents)
    if n == 0:
        raise ValueError("extents must not be empty")
    precisions = _broadcast(precision, n, "precision")
    psis = _broadcast(n_psi, n, "n_psi")
    errs = list(errors) if errors is not None else [None] * n
    if len(errs) != n:
        raise ValueError(f"errors has {len(errs)} entries for {n} stages")
    decisions = _broadcast(
        decision if decision is not None else NoDecision(), n, "decision"
    )

    stages = tuple(
        StageSpec(
            ((0, 0, int(k_ext), int(m_ext)),),
            precision=Precision.coerce(p),
            n_psi=int(ps),
            error=e,
            hypotheses=scout if i == 0 else None,
            name=f"stage{i}",
        )
        for i, ((k_ext, m_ext), p, ps, e) in enumerate(
            zip(extents, precisions, psis, errs, strict=True)
        )
    )
    schedule = SearchSchedule(
        stages,
        tuple(decisions),
        stop_when_empty=stop_when_empty,
        keep_stage_results=keep_stage_results,
    )
    if error_model is not None:
        schedule = schedule.with_errors(error_model)
    return schedule
