"""Panther-EM: eigendecomposition-accelerated 2D template matching in cryo-EM.

Pipelined AcceleratioN of Template matcHing via Eigendecomposition of Rotational
projections in cryo-EM (Panther-EM) decomposes a structure's projections in polar
coordinates, where an in-plane rotation is a phase, and searches micrographs with
the low-rank decomposition instead of the full template stack.

The subpackages form layers; each imports only from the ones above it in this list:

* :mod:`panther_em.coordinates` -- polar <-> Cartesian transforms and the
  :class:`~panther_em.coordinates.WarpConfig` of their interpolation.
* :mod:`panther_em.orientations` -- HEALPix orientation grids and in-plane angles.
* :mod:`panther_em.filters` -- the dataset's whitening filter and 1-D spectra.
* :mod:`panther_em.decomposition` -- the polar projection decomposer and its results.
* :mod:`panther_em.approximation` -- the decomposition's forward model
  (:class:`~panther_em.approximation.Reconstructor`) and its error tables.
* :mod:`panther_em.search` -- the multi-stage template search over micrographs.
* :mod:`panther_em.managers` -- the pipeline's programs (whitening, decompose,
  calibrate, search, export), each a YAML-loadable config and a ``run``.
* :mod:`panther_em.cli` -- the ``panther-em`` command running those programs.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("panther-em")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Matthew Giammar"
__email__ = "mdgiammar@gmail.com"
