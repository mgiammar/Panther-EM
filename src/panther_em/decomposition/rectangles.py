"""Rectangles of a decomposition's ``(angular frequency, eigenvector)`` space.

A rectangle ``(k_start, m_start, k_extent, m_extent)`` selects the components
``k_start <= k < k_start + k_extent`` and ``m_start <= m < m_start + m_extent``: the
unit in which a search keeps components and an error table measures what is lost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

Rectangle = tuple[int, int, int, int]
"""``(k_start, m_start, k_extent, m_extent)`` block of components."""


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


def as_rectangles(rectangles: Sequence[Sequence[int]]) -> tuple[Rectangle, ...]:
    """A sequence of rectangles as validated int tuples (see :func:`as_rectangle`)."""
    return tuple(as_rectangle(r) for r in rectangles)
