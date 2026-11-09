"""Image featurization against a template's SVD kernels.

* :class:`BaseImageFeaturization` -- the interface: ``(D, H, W)`` planes and ``(P,)``
  box origins in, a stream of :class:`FeatureChunk` s out (caller's position order).
* :class:`FFTFeaturization` -- overlap-save FFT correlation over the positions'
  bounding box (dense selections: a band of a micrograph).
* :class:`DirectFeaturization` -- box gather + :class:`BoxContraction` per position
  (sparse selections); the contraction is :class:`AdjointContraction`, with
  :class:`PlanarContraction` as the fallback.
* :func:`choose_mode` / :func:`make_featurization` -- which engine a selection gets.
* :class:`FeatureStore` -- one pixel set's features in the search loop's block layout,
  filled from the chunks.
"""

from panther_em.search.featurization.base import (
    BaseImageFeaturization,
    FeatureChunk,
    rectangle_cells,
)
from panther_em.search.featurization.direct_featurization import (
    AdjointContraction,
    BoxContraction,
    DirectFeaturization,
    PlanarContraction,
    box_contraction,
)
from panther_em.search.featurization.fft_featurization import (
    FFTFeaturization,
)
from panther_em.search.featurization.selection import (
    DIRECT_MAX_BOX_EXCESS,
    DIRECT_MAX_DENSITY,
    Mode,
    choose_mode,
    make_featurization,
)
from panther_em.search.featurization.store import (
    ColumnGroup,
    ColumnSubset,
    FeatureStore,
    ReadyBlock,
    StoreLike,
)

__all__ = [
    "DIRECT_MAX_BOX_EXCESS",
    "DIRECT_MAX_DENSITY",
    "AdjointContraction",
    "BaseImageFeaturization",
    "BoxContraction",
    "ColumnGroup",
    "ColumnSubset",
    "DirectFeaturization",
    "FFTFeaturization",
    "FeatureChunk",
    "FeatureStore",
    "Mode",
    "PlanarContraction",
    "ReadyBlock",
    "StoreLike",
    "box_contraction",
    "choose_mode",
    "make_featurization",
    "rectangle_cells",
]
