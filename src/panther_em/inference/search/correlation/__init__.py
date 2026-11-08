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
4. **correlator** (:mod:`.correlator`): the resources that outlive a pass (prepared
   weights, captured loops, moment forms, the stream) and the run over a store's
   blocks into :class:`~.maps.StageMaps`, given a pass's :class:`~.loop.LoopPlan`.

A *local* pass (:mod:`.local`) scores each pixel over its own small neighborhood of
hypotheses instead, with the mean and variance of the whole set from the
closed-form moments (:mod:`.moments`).

The package knows store blocks and weights, not images, pixel positions or decisions.
"""

from panther_em.inference.search.correlation.contraction import (
    ComplexGemmEngine,
    ContractionEngine,
    Fp16GemmEngine,
    Spectrum,
    make_engine,
)
from panther_em.inference.search.correlation.correlator import Correlator
from panther_em.inference.search.correlation.layout import (
    FeatureLayout,
    KRun,
    LayoutPart,
    rectangles_disjoint,
)
from panther_em.inference.search.correlation.local import LocalWork
from panther_em.inference.search.correlation.loop import HypLoop, LoopPlan
from panther_em.inference.search.correlation.maps import (
    StageMaps,
    concat_stage_maps,
    merge_stage_maps,
)
from panther_em.inference.search.correlation.moments import MomentForms

__all__ = [
    "ComplexGemmEngine",
    "ContractionEngine",
    "Correlator",
    "FeatureLayout",
    "Fp16GemmEngine",
    "HypLoop",
    "KRun",
    "LayoutPart",
    "LocalWork",
    "LoopPlan",
    "MomentForms",
    "Spectrum",
    "StageMaps",
    "concat_stage_maps",
    "make_engine",
    "merge_stage_maps",
    "rectangles_disjoint",
]
