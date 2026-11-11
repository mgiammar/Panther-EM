"""Symmetry-exploiting SVD-2DTM search including in-plane rotations.

The search reconstructs, for every image pixel, the cross-correlation with every
template hypothesis (defocus plane x Fourier filter x orientation x in-plane angle)
from a low-rank decomposition of the polar-coordinate projections, and reduces it to
per-pixel statistics (maximum, its hypothesis, mean, variance, z-score) without ever
materializing the full correlogram.

Objects, by lifetime
--------------------
* **Per template** (:mod:`.template`): :class:`Template` bundles a
  :class:`~panther_em.decomposition.DecompositionResult` with its forward model
  (:class:`~panther_em.approximation.Reconstructor`: Cartesian kernels on demand,
  the transposed warp the direct engine uses) and device-resident ``U * S``.
* **Per dataset** (:mod:`panther_em.filters`): the
  :class:`~panther_em.filters.WhiteningFilter` (``W_avg``, every micrograph's
  ``C_i``, the template band) the decomposition and every search image are built
  under.
* **Per template and feature selection** (:mod:`panther_em.approximation`):
  :class:`~panther_em.approximation.PerSpatialFrequencyErrorTable`, the ring
  profiles of the reconstruction, residual and signal of every template row, which the
  colored-noise :class:`~.threshold.ThresholdTest` reads.
* **Per search** (:mod:`.stage`, :mod:`.decision`, :mod:`.searcher`): a list of
  :class:`Stage` s (feature rectangles, precision, ``n_psi``, orientation sampling or
  a local neighborhood, CTF planes) each ending in a :class:`Decision` (threshold,
  optional threshold test, whether it accepts, absorb radius). :class:`Searcher`
  binds a template to its stages and owns the resources that amortize across image
  sets (the featurizations; :class:`~.correlation.Correlator`: prepared weights,
  captured hypothesis loops, the stream). :class:`ExecutionConfig` (:mod:`.config`)
  sets how the stages run: a few everyday choices and a :class:`Tuning` of
  overrides.
* **Per micrograph** (:class:`SearchSession`): the :class:`Micrograph` (whitened
  search image, CTF planes, noise and signal curves, level map; saved and loaded
  with HDF5), the dense :class:`~.ledger.PixelLedger`, the block-aligned
  :class:`~.featurization.FeatureStore`, and a :class:`StageResult` per stage;
  :meth:`SearchSession.snapshot` / :meth:`Searcher.run` give the
  :class:`SearchResult`.

How a stage runs
----------------
1. **Plan** (:mod:`.planning`): one :class:`~.planning.StagePlan` before any work --
   the stage's :class:`~.hypotheses.HypothesisSet`, the featurization engine, the
   bands (a stage whose store would not fit the device is cut into bands of rows)
   and, per band, the plane passes with their batch shapes and block sizes.
2. **Featurize** (:mod:`.runner`, :mod:`.featurization`): for each band and plane
   pass, a fresh :class:`~.featurization.FeatureStore` is filled with the features
   of every rectangle, by overlap-save FFT correlation (dense selections) or a box
   gather + contraction (sparse selections, particle boxes; see
   :func:`~.featurization.choose_mode` and its constants).
3. **Correlate** (:mod:`.correlation`): per block, per GEMM sub-batch and per
   hypothesis batch, one contraction (CGEMM) and one reduction (iRFFT over the
   in-plane angle, moments, running maximum; in-kernel on CUDA), optionally replayed
   as one CUDA graph; the passes of a band are merged, the bands concatenated.
4. **Decide** (:mod:`.decision`): the maps are written to the ledger and the stage's
   decision labels its pixels: the threshold test (or the plain threshold)
   ``REJECTS``, the plain threshold ``ACCEPTS`` when the stage accepts, the rest stay
   ``AMBIGUOUS`` (plus ``ABSORBED`` near accepted ones); a following stage re-scores
   the ``AMBIGUOUS`` and ``ACCEPTED`` pixels.

The two search types
--------------------
1. **Single stage**: one rectangle at rank ``r0``, then the decision::

       Searcher(template, [Stage.create([(0, 0, 128, 64)], decision=Decision(7.8))])

2. **Two stages**: a cheap first stage at low rank, possibly over a coarser HEALPix
   level and fewer CTF planes, rejects the background pixels without wrongly
   rejecting true particles; the survivors are re-scored by a full second stage::

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

   How the stages are built (local second stages, orientation levels, planes) is
   documented on :class:`Stage`.

The pipeline around a search
----------------------------
The programs of :mod:`panther_em.managers` (and the ``panther-em`` command) run these
steps from YAML configs; an application can also drive them directly:

1. Select the micrographs (names and a generator of pre-processed real-space images).
2. :class:`~panther_em.filters.WhiteningFilter` ``.from_images(generator)``: the
   averaged whitening filter ``W_avg``, every micrograph's correction ``C_i`` and the
   template band (saved once per dataset).
3. Decompose the structure under ``W_avg * band``
   (:class:`~panther_em.decomposition.PolarProjectionDecomposer` with
   ``fourier_filters=whitening.template_filter(box)[None]``); the result records the
   warp every later step uses.
4. Per feature selection (:func:`~panther_em.approximation.suggest_rectangles`
   proposes ranks for an error target or a budget):
   :func:`~panther_em.approximation.compute_error_tables`
   ``(volume, template.reconstructor, rectangles, whitening)`` (saved once per
   template).
5. Per micrograph: :func:`~panther_em.filters.ctf_planes` at its fitted defocus plus
   the searched offsets; :meth:`Micrograph.prepare` whitens it by ``W_avg * C_i^2``
   and estimates its curves and level map; :meth:`Searcher.run`;
   :meth:`SearchResult.save` (maps and ``score``).
6. Export the accepted pixels as a particle table.

``scripts/threshold_test_pipeline.py`` runs the chain on synthetic data through the
programs.

Example
-------
::

    template = Template.from_hdf5("decomp_60S.h5", device="cuda:0")
    searcher = Searcher(template, stages, n_defocus=11, keep_stage_results=True)
    micrograph = Micrograph.prepare(
        raw, name, whitening, ctf_planes, tables_32x16, pixel_size=1.06
    )
    result = searcher.run(micrograph)
    result.decode()["phi"], result.decode()["psi"], result.label, result.score
    result.save("search.h5")
"""

from panther_em.search.config import ExecutionConfig, Tuning
from panther_em.search.correlation.contraction import Precision
from panther_em.search.decision import Decision, PixelLabel
from panther_em.search.hypotheses import HypothesisSpace
from panther_em.search.images import SearchImages
from panther_em.search.micrograph import Micrograph
from panther_em.search.results import SearchResult, StageResult
from panther_em.search.searcher import Searcher, SearchSession
from panther_em.search.stage import Stage
from panther_em.search.template import Template
from panther_em.search.threshold import (
    Calibration,
    NoiseScales,
    ThresholdTest,
    zscore_threshold,
)

__all__ = [
    "Calibration",
    "Decision",
    "ExecutionConfig",
    "HypothesisSpace",
    "Micrograph",
    "NoiseScales",
    "PixelLabel",
    "Precision",
    "SearchImages",
    "SearchResult",
    "SearchSession",
    "Searcher",
    "Stage",
    "StageResult",
    "Template",
    "ThresholdTest",
    "Tuning",
    "zscore_threshold",
]
