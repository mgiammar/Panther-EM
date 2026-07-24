r"""Symmetry exploiting SVD-2DTM search including in-plane rotations.

This package splits the SVD-2DTM search into composable pieces:

* :mod:`~panther_em.inference.search.tiling` -- rectangular selections of
  ``(k, m)`` feature space (:class:`FeatureTiling`) and the persistent featurized
  image (:class:`FeaturizedImageStore`).
* :mod:`~panther_em.inference.search.utils` -- reconstructor-facing helpers that
  build Cartesian kernels, featurize an image, and build contraction weights.
* :mod:`~panther_em.inference.search.statistics` -- online per-pixel statistics
  (:class:`PixelStats`) and the error-aware partitioning process for incremental search.
* :mod:`~panther_em.inference.search.compressed` -- a single-stage search
  (:func:`compressed_search`) for producing approximate 2DTM results.
* :mod:`~panther_em.inference.search.incremental` -- the multi-stage,
  multi-precision driver (:func:`incremental_search`) to enable initial, low-precision,
  and low-cost searches followed by sparse, higher-precision searches.


Pre-processing Stage
--------------------
1.  Given a "SVD result" object, select the top-r features each with
    ``(k_idx, eig_idx)`` plus the associated singular values, ``s``. Then, reorder the
    indices of the top-r features such that they appear in increasing ``k_idx`` order
    (for efficiency in a downstream grouped sum reduction stage).
2.  Using the associated "reconstruction" object, produce the complex-valued Cartesian
    template feature kernels for the top-r selected features (from
    :func:`build_block_kernels`).
3.  If appropriate (decomposition- and image-dependent), apply Fourier filters to the
    the template feature kernels. Assume the image has already been pre-processed.


Image Featurization Stage
-------------------------
1.  Do a complex-valued cross-correlation of the image (shape ``[H, W]``) with the stack
    of kernels (shape ``[r, h, w]``) into an output featurized image for the valid
    cross-correlation region, denoted as variable ``Z``, with shape
    ``[H - h + 1, W - w + 1, r]``. NOTE: last (feature) dimension is contiguous for
    efficient per-pixel memory access.
2.  Based on an optional real-space correlation pixel mask, ``M``, with shape
    ``[H - h + 1, W - w + 1]``, apply the mask to the featurized image ``Z``.
3.  Then, flatten the featurized pixel image into a 2D tensor of shape ``[P, r]`` where
    ``P = (H - h + 1) * (W - w + 1)`` for no mask or ``P = sum(M)`` for a mask. Refer to
    this flattened featurized 2D tensor as ``Y``.

NOTE: Arbitrarily batched images are also supported for shapes ``(..., H, W)``. The
      featurization stage still collapses the spatial dimension into a single pixel axis
      but tracking input shape permits reconstruction.


Inner Product Search Stage
--------------------------
1.  Construct the per-hypotheses weights from the "SVD result" object (using
    :func:`build_block_weights`) which represent ``U\Sigma`` for the selected ``r``
    features. These weights have shape ``(N', r)`` where ``N'`` indexes out-of-plane
    orientation and possibly defocus hypotheses.
2.  Choose batch dimensions across flattened pixel space ``PX_BATCH_SIZE`` and the
    hypothesis space ``HY_BATCH_SIZE``.
3.  Do a partial sum reduction inner product for a batch combination (across only
    ``eig_idx`` for each ``k_idx``) in a tiled kernel via:
    a.  Extract a ``[PX_BLOCK_DIM, FT_BLOCK_DIM]`` tile from the the flattened
        featurized image as ``y``.
    b.  Extract a ``[HY_BLOCK_DIM, FT_BLOCK_DIM]`` tile from the reconstruction weights
        as ``w``.
    c.  Compute batched element-wise multiplication between ``y`` and ``w`` along the
        feature dimension reducing (sum-wise) to an output tile ``c`` of shape
        ``[PX_BLOCK_DIM, HY_BLOCK_DIM, NUM_K]`` based on the associated angular
        frequency ``k_idx`` of each feature.
    d.  Accumulate results from each tile into a final output tensor, ``C``, for the
        batch combination with shape ``[PX_BATCH_SIZE, HY_BATCH_SIZE, NUM_K]``.
    e.  Apply a inverse real-valued FFT across the angular frequency dimension (-1) to
        produce a real-valued correlogram for the batch combination with shape
        ``[PX_BATCH_SIZE, HY_BATCH_SIZE, N_psi]``.
    f.  Apply statistics tracking and reduction (online mean/variance update, max/argmax
        tracking across hypothesis and psi dimensions) to the correlogram.
4.  Repeat step 3 for all combinations across batch sizes to process full images.


A Multi-Precision Approach
--------------------------
Leveraging knowledge of SVD truncation rank (selected top-r features) and reconstruction
accuracy, can begin with a much lower-precision correlogram reconstruction and
progressively increase reconstruction rank. Based on current iteration result and the
reconstruction accuracy (from singular values, plus/minus some bounds), can iterate on
the algorithm as:
1.  Partition reconstructed cross-correlogram values into following three categories:
    a.  Accepted detections (based on reconstruction accuracy, far above expectation)
    b.  Rejected detections (based on reconstruction accuracy, far below expectation)
    c.  Image positions for further follow-up.
2.  Record accepted and rejected image locations plus associated statistics, update
    image mask to select for only follow-up positions.
3.  Repeat though pre-processing stage with the updated image mask and a selected
    ``r_{i+1} > r_{i}``. NOTE: Can leverage pre-calculated data, like previous kernels,
    if stored appropriately.
"""

from __future__ import annotations

from panther_em.inference.search.compressed import compressed_search
from panther_em.inference.search.fused_statistics import FusedPixelStats
from panther_em.inference.search.incremental import incremental_search
from panther_em.inference.search.statistics import PixelStats
from panther_em.inference.search.tiling import (
    FeatureTiling,
    FeaturizedImageStore,
    Rectangle,
    RectangularFeatureRegion,
)
from panther_em.inference.search.utils import (
    build_block_feature_stack,
    build_block_kernels,
    build_block_weights,
    build_layout_weights,
    build_multichannel_correlogram,
    featurize_cells,
    psi_degrees,
)

__all__ = [
    "FeatureTiling",
    "FeaturizedImageStore",
    "FusedPixelStats",
    "PixelStats",
    "Rectangle",
    "RectangularFeatureRegion",
    "build_block_feature_stack",
    "build_block_kernels",
    "build_block_weights",
    "build_layout_weights",
    "build_multichannel_correlogram",
    "compressed_search",
    "featurize_cells",
    "incremental_search",
    "psi_degrees",
]
