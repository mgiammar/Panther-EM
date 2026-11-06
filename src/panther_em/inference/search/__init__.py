r"""Symmetry-exploiting SVD-2DTM search including in-plane rotations.

The search reconstructs, for every image pixel, the cross-correlation with every
template hypothesis (defocus plane x Fourier filter x orientation x in-plane angle)
from a low-rank decomposition of the polar-coordinate projections, and reduces it to
per-pixel statistics (maximum, its hypothesis, mean, variance, z-score) without ever
materialising the full correlogram.

Objects, by lifetime
--------------------
* **Per template** (:mod:`panther_em.inference.template`):
  :class:`~panther_em.inference.template.Template` bundles a
  :class:`~panther_em.decomposition.result.DecompositionResult` with its kernel warp
  (:class:`~panther_em.inference.template.TemplateWarp`: Cartesian kernels on demand,
  the transposed warp the direct engine uses) and device-resident ``U * S``.
* **Per template and feature selection** (:mod:`.noise`): :class:`RadialTables`, the
  ring profiles of the kept, residual and signal templates of every orientation,
  which the coloured-noise :class:`ThresholdTest` reads.
* **Per search** (:mod:`.stage`, :mod:`.decision`, :mod:`.search`): a list of
  :class:`Stage` s (feature rectangles, precision, ``n_psi``, orientation sampling,
  CTF planes) each ending in a :class:`Decision` (threshold, optional threshold test,
  whether it accepts, absorb radius). :class:`Searcher` binds a template to its
  stages and owns the execution resources that amortise across image sets
  (:class:`~.executor.Executor`: prepared weights, captured hypothesis loops,
  streams).
* **Per image set** (:class:`SearchSession`): the images' :class:`ImageSpectra`
  (noise and signal curves, level map; :func:`estimate_spectra`), the dense
  :class:`PixelLedger`, the block-aligned :class:`FeatureStore`, and a
  :class:`StageResult` per stage; :meth:`SearchSession.snapshot` /
  :meth:`Searcher.run` give the :class:`SearchResult`.

How a stage runs
----------------
1. **Plan** (:mod:`.executor`): batch shape, block size and the stage's
   :class:`HypothesisSet` from the device capabilities and the execution knobs; a
   stage whose store would not fit the device is cut into bands of rows.
2. **Featurize** (:mod:`.featurization`): for each band and plane pass, a fresh
   :class:`~.featurization.FeatureStore` is filled with the features of every
   rectangle, by overlap-save FFT correlation (dense selections) or a box gather +
   contraction (sparse selections, particle boxes; see
   :func:`~.featurization.choose_mode` and its constants).
3. **Execute** (:mod:`.executor`, :mod:`.statistics`): per block, per GEMM sub-batch,
   per defocus plane and per hypothesis batch, one contraction and one statistics
   update (in-kernel accumulation on CUDA), optionally replayed as one CUDA graph.
4. **Decide** (:mod:`.decision`): the maps are written to the ledger and the stage's
   decision labels its pixels: the threshold test (or the plain threshold)
   ``REJECTS``, the plain threshold ``ACCEPTS`` when the stage accepts, the rest stay
   ``AMBIGUOUS`` (plus ``ABSORBED`` near accepted ones); a following stage re-scores
   the ``AMBIGUOUS`` and ``ACCEPTED`` pixels.

The two search types
--------------------
1. **Single stage**: one rectangle at rank ``r0``, then the decision::

       Searcher(template, [Stage.create([(0, 0, 128, 64)], decision=Decision(7.8))])

2. **Two stages**: a low-rank stage over a coarser HEALPix orientation level and one
   CTF plane rejects most pixels under the coloured-noise threshold test and leaves
   the rest ambiguous; the survivors are re-scored at high rank over every
   orientation and plane, where the test rejects and the threshold accepts::

       cal = Calibration(alpha=5e-3, zeta=0.95, eps_rho=0.1, b_factor=40.0)
       Searcher(
           template,
           [
               Stage.create(
                   [(0, 0, 32, 16)],
                   precision="fp16",
                   orientations=1,
                   planes=(D // 2,),
                   decision=Decision(
                       7.8, ThresholdTest(tables_32x16, cal), accept=False
                   ),
               ),
               Stage.create(
                   [(0, 0, 128, 64)],
                   decision=Decision(7.8, ThresholdTest(tables_128x64, cal)),
               ),
           ],
           n_defocus=D,
       )

   Each stage featurizes only the planes it uses, so the first stage pays the
   featurization of one plane. A HEALPix level needs a template whose Euler
   angles are HEALPix pixel centers (``healpix_orientations(nside)`` generates them);
   explicit orientation ids work for any template.

The pipeline around a search
----------------------------
Panther-EM supplies the steps; an external application drives the loops.

1. Select the micrographs (names and a generator of pre-processed real-space images).
2. :class:`~panther_em.spectra.WhiteningFilter` ``.from_images(generator)``: the
   averaged whitening filter ``W_avg`` and every micrograph's spectrum and
   correction ``C_i`` (saved once per dataset).
3. Decompose the structure under ``W_avg`` (``PolarProjectionDecomposer`` with
   ``fourier_filters=whitening.average_grid(box) * low_pass``).
4. Per feature selection: :meth:`RadialTables.build` ``(template, rectangles,
   volume=, whitening=)`` (saved once per template). Per micrograph: the search
   image carries ``W_avg * C_i^2`` (:meth:`WhiteningFilter.image_grid`);
   :func:`estimate_spectra` gives its noise/signal curves and level map
   (:class:`SpectraConfig` holds the parameters), :meth:`RadialTables.scales` the
   per-stage scales (both storable); ``ImageSet.with_spectra`` carries the spectra
   into the search; :meth:`Searcher.run`; :meth:`SearchResult.save` (maps, ``score``
   and the spectra).

``scripts/threshold_test_pipeline.py`` runs the chain on synthetic data.

Example
-------
::

    template = Template.from_hdf5("decomp_60S.h5", device="cuda:0")
    searcher = Searcher(template, stages, n_defocus=11, keep_stage_results=True)
    images = ImageSet.single(micrograph, filters=ctf_planes)
    images = images.with_spectra(estimate_spectra(images, tables_32x16, config))
    result = searcher.run(images)
    result.decode()["phi"], result.decode()["psi"], result.label, result.score
    result.save("search.h5")
"""

from panther_em.inference.search.decision import (
    Calibration,
    Decision,
    PixelLabel,
    ThresholdTest,
)
from panther_em.inference.search.executor import Executor, MemoryPlan, StagePlan
from panther_em.inference.search.features import FeatureLayout
from panther_em.inference.search.featurization import (
    BaseImageFeaturization,
    DirectFeaturization,
    FeatureStore,
    FFTFeaturization,
    choose_mode,
    make_featurization,
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
from panther_em.inference.search.noise import (
    NoiseScales,
    RadialTables,
    SpectraConfig,
    estimate_spectra,
    level_map,
    zeta_quantile,
    zscore_threshold,
)
from panther_em.inference.search.results import PixelLedger, SearchResult, StageResult
from panther_em.inference.search.search import ExecutionConfig, Searcher, SearchSession
from panther_em.inference.search.stage import Stage
from panther_em.inference.search.types import (
    BatchShape,
    ImageSet,
    ImageSpectra,
    PixelSelection,
    Precision,
    Rectangle,
    SearchGeometry,
)

__all__ = [
    "BaseImageFeaturization",
    "BatchShape",
    "Calibration",
    "Decision",
    "DirectFeaturization",
    "ExecutionConfig",
    "Executor",
    "FFTFeaturization",
    "FeatureLayout",
    "FeatureStore",
    "HypBatch",
    "HypothesisSet",
    "HypothesisSpace",
    "ImageSet",
    "ImageSpectra",
    "MemoryPlan",
    "NoiseScales",
    "PixelLabel",
    "PixelLedger",
    "PixelSelection",
    "Precision",
    "RadialTables",
    "Rectangle",
    "SearchGeometry",
    "SearchResult",
    "SearchSession",
    "Searcher",
    "SpectraConfig",
    "Stage",
    "StagePlan",
    "StageResult",
    "ThresholdTest",
    "choose_mode",
    "estimate_spectra",
    "healpix_ipix",
    "healpix_level_ids",
    "healpix_orientations",
    "level_map",
    "make_featurization",
    "psi_degrees",
    "zeta_quantile",
    "zscore_threshold",
]
