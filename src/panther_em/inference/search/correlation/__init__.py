"""Correlation: scoring a store of features against every hypothesis.

The search's inner loop, per block of pixels and batch of hypotheses:

1. **contraction** (:mod:`.contraction`): the CGEMM of the block's features with the
   hypotheses' weights into an angular-frequency spectrum, per k-run of the
   stage's :class:`~.layout.FeatureLayout`;
2. **reduction** (:mod:`.reduction`): the iRFFT over the in-plane angle and the
   per-pixel moments and running maximum (fused CUDA kernels in :mod:`.kernels`, or
   the torch reference);
3. **loop** (:mod:`.loop`): the two above for every sub-batch and hypothesis batch of
   a block, captured once as a CUDA graph where the kernels allow;
4. **executor** (:mod:`.executor`): the resources that outlive a stage (prepared
   weights, captured loops, streams) and the run over a store's blocks into
   :class:`~.maps.StageMaps`.

The package knows store blocks and weights, not images, pixel positions or decisions.
"""

from panther_em.inference.search.correlation.contraction import (
    ComplexGemmEngine,
    ContractionEngine,
    Fp16GemmEngine,
    Spectrum,
    make_engine,
)
from panther_em.inference.search.correlation.executor import Executor
from panther_em.inference.search.correlation.layout import (
    FeatureLayout,
    KRun,
    LayoutPart,
    as_rectangle,
    rectangles_disjoint,
)
from panther_em.inference.search.correlation.loop import HypLoop
from panther_em.inference.search.correlation.maps import StageMaps, merge_stage_maps

__all__ = [
    "ComplexGemmEngine",
    "ContractionEngine",
    "Executor",
    "FeatureLayout",
    "Fp16GemmEngine",
    "HypLoop",
    "KRun",
    "LayoutPart",
    "Spectrum",
    "StageMaps",
    "as_rectangle",
    "make_engine",
    "merge_stage_maps",
    "rectangles_disjoint",
]
