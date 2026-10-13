r"""Symmetry-exploiting SVD-2DTM search including in-plane rotations.

The search reconstructs, for every image pixel, the cross-correlation with every
template hypothesis (defocus plane x Fourier filter x orientation x in-plane angle)
from a low-rank decomposition of the polar-coordinate projections, and reduces it to
per-pixel statistics (maximum, its hypothesis, mean, variance, z-score) without ever
materialising the full correlogram.

Objects, by lifetime
--------------------
* **Per template** (:mod:`.template`): :class:`Template` bundles a
  :class:`~panther_em.decomposition.result.DecompositionResult` with a lazily built,
  byte-budgeted :class:`KernelBank` (Cartesian feature kernels, their spectra) and a
  device-resident :class:`WeightSource` (``U * S``).
* **Per search** (:mod:`.stage`, :mod:`.decision`, :mod:`.search`): a list of :class:`Stage` s
  (feature rectangles, precision, ``n_psi``, orientation sampling, CTF planes) each
  ending in a :class:`Decision` (threshold, error bound, absorb radius).
  :class:`Searcher` binds a template to its stages and owns the execution resources
  that amortise across image sets (:class:`~.executor.Executor`: prepared weights,
  captured hypothesis loops, streams).
* **Per image set** (:class:`SearchSession`): the dense :class:`PixelLedger`, the
  block-aligned :class:`FeatureStore`, and a :class:`StageResult` per stage;
  :meth:`SearchSession.snapshot` / :meth:`Searcher.run` give the :class:`SearchResult`.

How a stage runs
----------------
1. **Plan** (:mod:`.executor`): batch shape, block size, residency and the stage's
   :class:`HypothesisSet` from the device capabilities and the execution knobs.
2. **Featurize** (:mod:`.featurize`, :mod:`.features`): the store is compacted to the
   stage's pixels and every missing (rectangle, precision, plane) group is produced,
   by tiled FFT correlation on image grids or inner products on particle boxes.
3. **Execute** (:mod:`.executor`, :mod:`.statistics`): per block, per GEMM sub-batch,
   per defocus plane and per hypothesis batch, one contraction and one statistics
   update (in-kernel accumulation on CUDA), optionally replayed as one CUDA graph.
4. **Decide** (:mod:`.decision`): the maps are written to the ledger and the stage's
   decision labels its pixels ``ACCEPTED`` / ``REJECTED`` / ``AMBIGUOUS`` (plus
   ``ABSORBED`` near accepted ones); a following stage re-scores the ``AMBIGUOUS``
   and ``ACCEPTED`` pixels.

The two search types
--------------------
1. **Single stage**: one rectangle at rank ``r0``, then the decision::

       Searcher(template, [Stage.create([(0, 0, 128, 64)], decision=Decision(7.8))])

2. **Two stages**: a low-rank stage over a coarser HEALPix orientation level and one
   CTF plane rejects most pixels; the survivors are re-scored at high rank over every
   orientation and plane with the same rule and a tighter error bound::

       bound = ErrorModel.from_result(template.result)
       Searcher(template, [
           Stage.create([(0, 0, 32, 16)], precision="fp16", orientations=1,
                        planes=(D // 2,), decision=Decision(7.8, bound, absorb_radius=1)),
           Stage.create([(0, 0, 128, 64)], decision=Decision(7.8, bound)),
       ], n_defocus=D)

   Each store group holds only the planes its stage uses, so the first stage pays
   the featurization of one plane. A HEALPix level needs a template whose Euler
   angles are HEALPix pixel centers (``healpix_orientations(nside)`` generates them);
   explicit orientation ids work for any template.

Example
-------
::

    template = Template.from_hdf5("decomp_60S.h5", device="cuda:0")
    searcher = Searcher(template, stages, n_defocus=11, keep_stage_results=True)
    result = searcher.run(ImageSet.single(micrograph, filters=ctf_planes))
    result.decode()["phi"], result.decode()["psi"], result.label
    result.save("search.h5")
"""

from panther_em.inference.search.decision import (
    Decision,
    ErrorModel,
    PixelLabel,
    classify_zscore_interval,
    zscore_error_intervals,
)
from panther_em.inference.search.executor import Executor, MemoryPlan, StagePlan
from panther_em.inference.search.features import FeatureLayout, FeatureStore
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
from panther_em.inference.search.results import PixelLedger, SearchResult, StageResult
from panther_em.inference.search.search import ExecutionConfig, Searcher, SearchSession
from panther_em.inference.search.stage import FixedError, Stage
from panther_em.inference.search.template import KernelBank, Template, WeightSource
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
    "ErrorModel",
    "ExecutionConfig",
    "Executor",
    "FeatureLayout",
    "FeatureStore",
    "FixedError",
    "HypBatch",
    "HypothesisSet",
    "HypothesisSpace",
    "ImageSet",
    "KernelBank",
    "KernelConfig",
    "MemoryPlan",
    "PixelLabel",
    "PixelLedger",
    "PixelSelection",
    "Precision",
    "Rectangle",
    "SearchGeometry",
    "SearchResult",
    "SearchSession",
    "Searcher",
    "Stage",
    "StagePlan",
    "StageResult",
    "Template",
    "TileSchedule",
    "WeightSource",
    "classify_zscore_interval",
    "featurize",
    "healpix_ipix",
    "healpix_level_ids",
    "healpix_orientations",
    "psi_degrees",
    "schedule_tiles",
    "zscore_error_intervals",
]
