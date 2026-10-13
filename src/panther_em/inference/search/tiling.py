"""Rectangular layouts of ``(k, m)`` SVD feature space.

A :class:`FeatureLayout` is a pure, hashable description of a set of pairwise
cell-disjoint rectangles in ``(angular frequency, eigenvector)`` space together with
its *k-runs*: maximal runs of ``k`` covered by a constant set of rectangles, each of
which the contraction engines evaluate as one batched GEMM.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from panther_em.inference.search.types import Rectangle


# --------------------------------------------------------------------------- #
# CPU-only feature layout
# --------------------------------------------------------------------------- #


def rectangles_disjoint(rectangles: Sequence[Rectangle]) -> bool:
    """Whether ``(k_start, m_start, k_extent, m_extent)`` rectangles never overlap."""
    rects = list(rectangles)
    for i in range(len(rects)):
        k0, m0, ke, me = rects[i]
        for j in range(i + 1, len(rects)):
            l0, n0, le, ne = rects[j]
            if k0 < l0 + le and l0 < k0 + ke and m0 < n0 + ne and n0 < m0 + me:
                return False
    return True


@dataclass(frozen=True)
class LayoutPart:
    """One rectangle's contribution to a :class:`KRun`.

    Attributes
    ----------
    rect : int
        Index into :attr:`FeatureLayout.rectangles`.
    m_start, m_stop : int
        The rectangle's eigenvector range (global ``m`` indices).
    """

    rect: int
    m_start: int
    m_stop: int

    @property
    def num_m(self) -> int:
        """Eigenvectors contributed."""
        return self.m_stop - self.m_start


@dataclass(frozen=True)
class KRun:
    """A run ``[k_start, k_stop)`` of ``k`` covered by a fixed rectangle set.

    Within one run every covering rectangle contributes a disjoint ``m``-range, so
    the contraction over the run is ONE GEMM whose inner dimension is the
    concatenation of those ranges (in :attr:`parts` order).
    """

    k_start: int
    k_stop: int
    parts: tuple[LayoutPart, ...]

    @property
    def num_k(self) -> int:
        """Angular frequencies in the run."""
        return self.k_stop - self.k_start

    @property
    def num_m(self) -> int:
        """Total inner (eigenvector) dimension of the run's GEMM."""
        return sum(p.num_m for p in self.parts)


@dataclass(frozen=True)
class FeatureLayout:
    """Pure, hashable description of a rectangle set in ``(k, m)`` feature space.

    Built entirely from Python integers: no device tensors, no host syncs. This is
    the layout the redesigned search plans around; the contraction engines consume
    :attr:`k_runs`, the feature store allocates one column group per rectangle.

    Parameters
    ----------
    rectangles : sequence of Rectangle
        Pairwise cell-disjoint ``(k_start, m_start, k_extent, m_extent)`` blocks.
    """

    rectangles: tuple[Rectangle, ...]
    k_runs: tuple[KRun, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Normalise, validate and build the k-runs."""
        rects = tuple(tuple(int(v) for v in r) for r in self.rectangles)
        if not rects:
            raise ValueError("FeatureLayout needs at least one rectangle")
        for r in rects:
            if len(r) != 4 or r[0] < 0 or r[1] < 0 or r[2] <= 0 or r[3] <= 0:
                raise ValueError(f"bad rectangle {r!r}")
        if not rectangles_disjoint(rects):  # type: ignore[arg-type]
            raise ValueError("rectangles must be pairwise cell-disjoint")
        object.__setattr__(self, "rectangles", rects)
        object.__setattr__(self, "k_runs", self._build_k_runs(rects))  # type: ignore[arg-type]

    @staticmethod
    def _build_k_runs(rects: Sequence[Rectangle]) -> tuple[KRun, ...]:
        bounds = sorted({r[0] for r in rects} | {r[0] + r[2] for r in rects})
        runs: list[KRun] = []
        for k_start, k_stop in itertools.pairwise(bounds):
            parts = tuple(
                LayoutPart(i, r[1], r[1] + r[3])
                for i, r in enumerate(rects)
                if r[0] <= k_start and r[0] + r[2] >= k_stop
            )
            if parts:
                runs.append(KRun(k_start, k_stop, parts))
        return tuple(runs)

    @property
    def num_features(self) -> int:
        """Total ``(k, m)`` cells."""
        return sum(r[2] * r[3] for r in self.rectangles)

    @property
    def k_stop(self) -> int:
        """Smallest angular-frequency axis length holding every rectangle."""
        return max(r[0] + r[2] for r in self.rectangles)

    def covered_k(self) -> tuple[bool, ...]:
        """For each ``k < k_stop``, whether some rectangle covers it."""
        covered = [False] * self.k_stop
        for run in self.k_runs:
            for k in range(run.k_start, run.k_stop):
                covered[k] = True
        return tuple(covered)
