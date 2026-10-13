r"""Symmetry-exploiting SVD-2DTM search including in-plane rotations.

The search reconstructs, for every image pixel, the cross-correlation with every
template hypothesis (defocus plane x Fourier filter x orientation x in-plane angle)
from a low-rank decomposition of the polar-coordinate projections, and reduces it to
per-pixel statistics (maximum, its hypothesis, mean, variance, z-score) without ever
materialising the full correlogram.

Objects, by lifetime
--------------------
* **Per template** (:mod:`.basis`): :class:`Template` bundles a
  :class:`~panther_em.decomposition.result.DecompositionResult` with a lazily built,
  byte-budgeted :class:`KernelBank` (Cartesian feature kernels) and a device-resident
  :class:`WeightSource` (``U * S``).
* **Per search** (:mod:`.schedule`, :mod:`.searcher`): a :class:`SearchSchedule` of
  :class:`StageSpec`\ s (feature rectangles, precision, ``n_psi``, relative
  reconstruction error) each followed by a :class:`~.decisions.Decision`;
  :func:`nested_schedule` builds the common nested case and :class:`ErrorModel`
  derives the errors from the singular values. :class:`Searcher` binds a template to a
  schedule and owns the execution resources that amortise across images (prepared
  weights, captured CUDA graphs).
* **Per image set** (:mod:`.types`, :mod:`.searcher`, :mod:`.results`): an
  :class:`ImageSet` (already preprocessed images, optional defocus filter planes,
  optional mask) runs through a :class:`SearchSession`, which yields a
  :class:`StageResult` per stage and a dense :class:`SearchResult` at the end.

Pipeline of one stage
---------------------
1. **Plan** (:mod:`.executor`): batch shape, block size, contraction
   engine (:mod:`.contraction`: fp32 / tf32 complex GEMM or the fp16 "4M" GEMM),
   statistics (:mod:`.statistics`: pure torch or
   the fused CUDA iRFFT+statistics kernel) and the :class:`~.hypotheses.HypothesisSet`
   (contiguous *virtual* ids over selected templates and defocus planes).
2. **Featurize** (:mod:`.featurize`, :mod:`.store`): the pixels that survived the
   previous stage are cross-correlated with the kernels of every new rectangle, by
   tiled FFT (grid mode) or one GEMM (center-only particle mode), straight into the
   block-aligned :class:`FeatureStore` in the engine's operand layout.
3. **Execute** (:mod:`.executor`): per block, per GEMM sub-batch, per defocus plane and
   per hypothesis batch: one contraction into a static spectrum, one statistics
   update (in-kernel accumulation on CUDA), optionally replayed as one CUDA graph.
4. **Decide** (:mod:`.ledger`, :mod:`.decisions`): the stage's maps are written to the
   dense :class:`PixelLedger`; the stage's decision labels pixels
   ``ACCEPTED`` / ``REJECTED`` / ``UNCERTAIN`` (``ABSORBED`` near accepted ones);
   ``UNCERTAIN`` pixels form the next stage's pixel set.

The two search types, as schedules
-----------------------------------------
1. **Single rank**: one dense stage::

       SearchSchedule.single(
           StageSpec.create([(0, 0, 128, 64)], n_psi=256), ThresholdClassifier(7.8)
       )

2. **Pixel rejection**: a low-rank dense stage on one defocus plane rejects pixels; the
   survivors are re-searched densely at high rank over every plane::

       SearchSchedule(
           (
               StageSpec.create(
                   [(0, 0, 32, 16)],
                   precision="fp16",
                   orientations=1, planes=(D // 2,)),
               ),
               StageSpec.create([(0, 0, 128, 64)]),
           ),
           (ZScoreIntervalClassifier(7.8), ThresholdClassifier(7.8)),
       )

   Each store group holds only the planes its stage uses, so the first stage pays
   the featurization of one plane.
Example
-------
::

    template = Template.from_hdf5("decomp_60S.h5", device="cuda:0")
    schedule = nested_schedule(
        [(32, 16), (64, 32), (128, 48)],
        precision=["fp16", "fp16", "fp32"],
        n_psi=256,
        decision=[DecisionChain(ZScoreIntervalClassifier(7.8), DilationAbsorb(1))] * 2
        + [ThresholdClassifier(7.8)],
        error_model=ErrorModel.from_result(template.result),
    )
    searcher = Searcher(template, schedule)
    result = searcher.run(ImageSet.single(micrograph))  # (1, out_h, out_w) maps
    result.decode().phi, result.decode().psi, result.label

Particle stacks: ``ImageSet.from_boxes(boxes)`` (box == template -> center-only
mode, ``result.per_particle()``). Defocus planes: ``Searcher(..., n_defocus=D)`` and
``ImageSet.single(image, filters=ctf_stack)`` with ``(D, H, W // 2 + 1)`` filters;
the statistics then run over defocus as a hypothesis axis and
``result.decode().defocus_index`` reports the best plane.

Inputs are expected to be preprocessed already (whitened, normalised, contrast
convention applied); Panther-EM applies no image preprocessing.
"""

from __future__ import annotations

from panther_em.inference.search.template import (
    KernelBank,
    Template,
    WeightSource,
)
from panther_em.inference.search.decisions import (
    Decision,
    DecisionChain,
    DilationAbsorb,
    NoDecision,
    ThresholdClassifier,
    TopFractionSelector,
    ZScoreIntervalClassifier,
    classify_zscore_interval,
    zscore_error_intervals,
)
from panther_em.inference.search.executor import Executor, MemoryPlan, StagePlan
from panther_em.inference.search.featurize import (
    TileSchedule,
    featurize,
    schedule_tiles,
)
from panther_em.inference.search.hypotheses import (
    HypBatch,
    HypothesisSet,
    HypothesisSpace,
    healpix_ipix,
    healpix_level_ids,
    healpix_orientations,
    psi_degrees,
)
from panther_em.inference.search.ledger import PixelLabel, PixelLedger
from panther_em.inference.search.results import DecodedMaps, SearchResult, StageResult
from panther_em.inference.search.schedule import (
    ErrorModel,
    SearchSchedule,
    StageSpec,
    nested_schedule,
)
from panther_em.inference.search.searcher import (
    ExecutionConfig,
    Searcher,
    SearchSession,
)
from panther_em.inference.search.features import FeatureStore
from panther_em.inference.search.features import FeatureLayout
from panther_em.inference.search.types import (
    BatchShape,
    ImageSet,
    KernelConfig,
    PixelSelection,
    Precision,
    Rectangle,
    SearchGeometry,
)

__all__ = [
    "BatchShape",
    "Decision",
    "DecisionChain",
    "DecodedMaps",
    "DilationAbsorb",
    "ErrorModel",
    "ExecutionConfig",
    "Executor",
    "FeatureLayout",
    "FeatureStore",
    "HypBatch",
    "HypothesisSet",
    "HypothesisSpace",
    "ImageSet",
    "KernelBank",
    "KernelConfig",
    "MemoryPlan",
    "NoDecision",
    "PixelLabel",
    "PixelLedger",
    "PixelSelection",
    "Precision",
    "Rectangle",
    "SearchGeometry",
    "SearchResult",
    "SearchSchedule",
    "SearchSession",
    "StagePlan",
    "TileSchedule",
    "Searcher",
    "StageResult",
    "StageSpec",
    "Template",
    "ThresholdClassifier",
    "TopFractionSelector",
    "WeightSource",
    "ZScoreIntervalClassifier",
    "classify_zscore_interval",
    "healpix_ipix",
    "healpix_level_ids",
    "healpix_orientations",
    "featurize",
    "nested_schedule",
    "psi_degrees",
    "schedule_tiles",
    "zscore_error_intervals",
]
