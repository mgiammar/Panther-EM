"""Which featurization a selection of pixels gets, and building it."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from panther_em.inference.search.featurization.direct_featurization import (
    DEFAULT_BATCH,
    DEFAULT_PRECISION,
    DirectFeaturization,
)
from panther_em.inference.search.featurization.fft_featurization import (
    DEFAULT_CELL_CHUNK,
    FFTFeaturization,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from panther_em.inference.search.correlation.contraction import Precision
    from panther_em.inference.search.featurization.base import BaseImageFeaturization
    from panther_em.inference.search.featurization.direct_featurization import (
        ContractionName,
    )
    from panther_em.inference.search.images import SearchGeometry
    from panther_em.inference.template import TemplateWarp

Mode = Literal["fft", "direct"]


# Percentage of image's pixels below which the direct engine is chosen
DIRECT_MAX_DENSITY = 0.02

# For near box-sized images (within 32 pixels) use direct engine
DIRECT_MAX_BOX_EXCESS = 32


def choose_mode(geometry: SearchGeometry, n_selected: int) -> Mode:
    """``direct`` for a center geometry, box-sized images and sparse selections."""
    if geometry.mode == "center":
        return "direct"

    (h, w), (k_h, k_w) = geometry.image_shape, geometry.template_shape

    if h - k_h <= DIRECT_MAX_BOX_EXCESS and w - k_w <= DIRECT_MAX_BOX_EXCESS:
        return "direct"

    if n_selected <= DIRECT_MAX_DENSITY * geometry.n_pixels:
        return "direct"

    return "fft"


def make_featurization(
    warp: TemplateWarp,
    rect: Sequence[int],
    mode: Mode,
    *,
    cell_chunk: int = DEFAULT_CELL_CHUNK,
    batch: int = DEFAULT_BATCH,
    precision: Precision | str = DEFAULT_PRECISION,
    contraction: ContractionName = "adjoint",
) -> BaseImageFeaturization:
    """The ``mode`` featurization of ``rect``."""
    if mode == "fft":
        return FFTFeaturization(warp, rect, cell_chunk=cell_chunk)

    if mode == "direct":
        return DirectFeaturization(
            warp, rect, batch=batch, precision=precision, contraction=contraction
        )

    raise ValueError(f"unknown featurization mode {mode!r}")
