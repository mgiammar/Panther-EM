"""Feature-space layouts: rectangles of ``(angular frequency, eigenvector)`` space.

:class:`FeatureLayout` holds pairwise cell-disjoint rectangles and their *k-runs*:
maximal runs of ``k`` covered by a constant rectangle set, each contracted as one
batched GEMM.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

    from panther_em.inference.search.types import Rectangle


def as_rectangle(rect: Sequence[int]) -> Rectangle:
    """``(k_start, m_start, k_extent, m_extent)`` as validated Python ints."""
    if len(rect) != 4:
        raise ValueError(
            f"a rectangle is (k_start, m_start, k_extent, m_extent), got {rect!r}"
        )
    k0, m0, k_ext, m_ext = (int(v) for v in rect)
    if k0 < 0 or m0 < 0:
        raise ValueError(f"rectangle starts must be >= 0, got {rect!r}")
    if k_ext <= 0 or m_ext <= 0:
        raise ValueError(f"rectangle extents must be > 0, got {rect!r}")
    return (k0, m0, k_ext, m_ext)


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


class LayoutPart(NamedTuple):
    """One rectangle's contribution to a :class:`KRun`: its index into
    :attr:`FeatureLayout.rectangles` and its global eigenvector range.
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
    """A run ``[k_start, k_stop)`` of ``k`` covered by a fixed rectangle set."""

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

    Parameters
    ----------
    rectangles : sequence of Rectangle
        Pairwise cell-disjoint ``(k_start, m_start, k_extent, m_extent)`` blocks.
    """

    rectangles: tuple[Rectangle, ...]

    @classmethod
    def create(cls, rectangles: Sequence[Sequence[int]]) -> FeatureLayout:
        """Build a layout from any sequence of 4-int sequences."""
        return cls(tuple(as_rectangle(r) for r in rectangles))

    def __post_init__(self) -> None:
        """Validate (never coerce): int 4-tuples, pairwise cell-disjoint."""
        rects = self.rectangles
        if not isinstance(rects, tuple) or not rects:
            raise ValueError(
                "FeatureLayout needs a non-empty tuple of rectangles; see create()"
            )
        for r in rects:
            if r != as_rectangle(r):
                raise ValueError(f"bad rectangle {r!r}; see FeatureLayout.create()")
        if not rectangles_disjoint(rects):
            raise ValueError("rectangles must be pairwise cell-disjoint")

    @cached_property
    def k_runs(self) -> tuple[KRun, ...]:
        """Maximal ``k`` runs covered by a constant rectangle set (one GEMM each)."""
        rects = self.rectangles
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
