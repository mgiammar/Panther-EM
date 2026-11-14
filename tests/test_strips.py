"""Strips: covering a region's rows with FFTs of the sizes a backend runs."""

from __future__ import annotations

import pytest
from scipy.fft import next_fast_len

from panther_em.search.featurization import Strip, StripSpec

BOX = (512, 512)
FIXED = StripSpec(BOX, heights=(1024, 2048, 4096), widths=(2048, 4096))


def test_any_size_is_one_fast_strip():
    spec = StripSpec(BOX)
    assert not spec.fixed
    assert spec.cover(718) == (next_fast_len(718 + 511),) == (1232,)
    assert spec.strips(718) == (Strip(0, 718, 1232),)
    assert spec.fft_width(3585) == next_fast_len(3585 + 511)
    assert spec.cover(0) == () and spec.cost(0) == 0


def test_fixed_heights_cover_with_the_least_fft_height():
    assert FIXED.valid_rows(1024) == 513
    assert FIXED.cover(513) == (1024,)
    assert FIXED.cover(514) == (2048,)  # one 2048 ties two 1024s: fewer strips
    assert FIXED.cover(1026) == (2048,)
    assert FIXED.cover(1600) == (2048, 1024)  # 1537 rows + the 63-row remainder
    assert FIXED.cover(3585) == (4096,)
    assert FIXED.cover(3586) == (4096, 1024)
    assert FIXED.strips(1600) == (Strip(0, 1537, 2048), Strip(1537, 63, 1024))
    assert FIXED.cost(1600) == 3072
    # every row is kept exactly once, in order
    for n in (1, 512, 513, 2000, 5000, 7171):
        strips = FIXED.strips(n)
        assert [s.row for s in strips] == [
            sum(t.rows for t in strips[:i]) for i in range(len(strips))
        ]
        assert sum(s.rows for s in strips) == n
        assert all(0 < s.rows <= FIXED.valid_rows(s.height) for s in strips)


def test_fixed_widths_hold_the_region_or_none():
    assert FIXED.fft_width(1537) == 2048
    assert FIXED.fft_width(1538) == 4096
    assert FIXED.fft_width(3585) == 4096
    assert FIXED.fft_width(3586) is None


def test_band_rows_hold_whole_strips():
    # xenon on a 48 GB card: 718 rows fit; 7 bands of one 1024 strip beat 5 of 2048
    assert FIXED.band_rows(718, 3585) == 513
    assert StripSpec(BOX).band_rows(718, 3585) == 717  # any size: 5 equal bands
    # 1200 rows fit: 3 equal bands of one 2048 strip each (6144 rows of FFT) beat
    # whole 1024 strips (7 x 1024 = 7168), wasted rows and all
    assert FIXED.band_rows(1200, 3585) == 1195
    # 1600 rows fit: 2 bands of a whole 2048 strip, the 511-row remainder in a 1024
    # strip (5120 rows of FFT)
    assert FIXED.band_rows(1600, 3585) == 1537
    assert FIXED.cover(3585 - 2 * 1537) == (1024,)
    # the whole image fits: one band of one 4096 strip
    assert FIXED.band_rows(4000, 3585) == 3585
    # no strip fits a band: the fewest equal bands
    assert FIXED.band_rows(300, 3585) == 299


@pytest.mark.parametrize(
    "kwargs",
    [
        {"heights": (2048, 1024)},
        {"heights": (1024, 1024)},
        {"heights": ()},
        {"heights": (256, 1024)},
        {"widths": (128,)},
    ],
)
def test_spec_validates(kwargs):
    with pytest.raises(ValueError):
        StripSpec(BOX, **kwargs)
