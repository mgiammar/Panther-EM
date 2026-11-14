"""Strips which FFT engine can run which cover an entire region's rows."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from scipy.fft import next_fast_len


@dataclass(frozen=True)
class Strip:
    """One FFT of a region: ``rows`` box-origin rows from ``row``, at FFT ``height``.

    Attributes
    ----------
    row : int
        First box-origin row kept, relative to the region's first row (the strip's
        image rows start there too).
    rows : int
        Box-origin rows kept (``<= height - kH + 1``).
    height : int
        FFT height ``S``.
    """

    row: int
    rows: int
    height: int


@dataclass(frozen=True)
class StripSpec:
    """The FFT sizes a backend runs for one kernel box.

    Attributes
    ----------
    box : tuple[int, int]
        Kernel box ``(kH, kW)``.
    heights : tuple[int, ...], optional
        FFT heights the backend runs, ascending; ``None``: any (one strip per region,
        of a fast size).
    widths : tuple[int, ...], optional
        FFT widths the backend runs, ascending; ``None``: any.
    """

    box: tuple[int, int]
    heights: tuple[int, ...] | None = None
    widths: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """Validate (never coerce)."""
        k_h, k_w = self.box

        if k_h < 1 or k_w < 1:
            raise ValueError(f"box must be positive, got {self.box}")

        for name, sizes, k in (
            ("heights", self.heights, k_h),
            ("widths", self.widths, k_w),
        ):
            if sizes is None:
                continue
            if not sizes or list(sizes) != sorted(set(sizes)):
                raise ValueError(f"{name} must be ascending and distinct, got {sizes}")
            if sizes[0] < k:
                raise ValueError(f"{name} {sizes} include a size below the box {k}")

    @property
    def fixed(self) -> bool:
        """Whether the backend runs only the listed heights."""
        return self.heights is not None

    def valid_rows(self, height: int) -> int:
        """Box-origin rows a strip of FFT ``height`` yields."""
        return height - self.box[0] + 1

    def fft_width(self, width: int) -> int | None:
        """FFT width for a region ``width`` box-origin columns (``None``: none fits)."""
        need = width + self.box[1] - 1
        if self.widths is None:
            return int(next_fast_len(need))
        return next((w for w in self.widths if w >= need), None)

    # -- covering rows --------------------------------------------------------------

    def cover(self, n_rows: int) -> tuple[int, ...]:
        """FFT heights (descending) of the cheapest strips covering ``n_rows`` rows."""
        if n_rows < 1:
            return ()
        if self.heights is None:
            return (int(next_fast_len(n_rows + self.box[0] - 1)),)

        heights = []

        while n_rows > 0:
            height = self._table(n_rows)[1]
            heights.append(height)
            n_rows -= self.valid_rows(height)

        return tuple(sorted(heights, reverse=True))

    def cost(self, n_rows: int) -> int:
        """Total FFT height of :meth:`cover` (0 for no rows)."""
        return sum(self.cover(n_rows))

    def strips(self, n_rows: int) -> tuple[Strip, ...]:
        """:meth:`cover` laid out from the region's first row (the remainder last)."""
        out, row = [], 0

        for height in self.cover(n_rows):
            rows = min(self.valid_rows(height), n_rows - row)
            out.append(Strip(row, rows, height))
            row += rows

        return tuple(out)

    def _table(self, n_rows: int) -> tuple[int, int]:
        """``(cost, first height)`` of the cheapest cover of ``n_rows`` (fixed heights).

        Cheapest = least total height, then fewest strips.
        """
        table = self._costs
        while len(table) <= n_rows:
            n = len(table)
            best = None
            for height in self.heights or ():
                rest = table[max(0, n - self.valid_rows(height))]
                key = (rest[0] + height, rest[2] + 1, height)
                if best is None or key < best:
                    best = key
            assert best is not None
            table.append(best)
        cost, _, height = table[n_rows]
        return cost, height

    @cached_property
    def _costs(self) -> list[tuple[int, int, int]]:
        """Memo of ``(cost, n_strips, first height)`` by rows (row 0: nothing)."""
        return [(0, 0, 0)]

    # -- bands ----------------------------------------------------------------------

    def band_rows(self, max_rows: int, n_rows: int) -> int:
        """Rows per band (at most ``max_rows``) for ``n_rows`` rows of positions."""
        max_rows = max(1, min(int(max_rows), int(n_rows)))
        n_bands = -(-n_rows // max_rows)
        equal = -(-n_rows // n_bands)
        if self.heights is None or n_rows < 1:
            return equal
        candidates = {equal}
        for height in self.heights:
            step = self.valid_rows(height)
            candidates.update(range(step, max_rows + 1, step))

        def total(rows: int) -> tuple[int, int]:
            full, rest = divmod(n_rows, rows)
            return (full * self.cost(rows) + self.cost(rest), full + (rest > 0))

        return min(sorted(candidates, reverse=True), key=total)
