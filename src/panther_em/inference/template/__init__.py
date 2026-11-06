"""An SVD template: a decomposition, the warp of its kernels and its weights.

Shared by every consumer of a decomposition: the search's featurizations and executor,
and the noise model's error tables.

* :class:`KernelConfig` -- the polar-to-Cartesian warp settings.
* :class:`TemplateWarp` -- the single implementation of that warp: Cartesian kernels
  on demand, the transposed warp (:class:`WarpAdjoint`) and its per-rectangle bases,
  the generic warp of polar images and the support disk.
* :class:`WeightSource` -- device-resident ``U * S``, served as rectangular blocks.
* :class:`Template` -- the bundle.

The package does not import :mod:`panther_em.inference.search` at module level, so
the search and the noise model can import it freely.
"""

from panther_em.inference.template.adjoint import (
    AdjointBases,
    WarpAdjoint,
    build_bases,
    unsupported_reason,
)
from panther_em.inference.template.template import Template
from panther_em.inference.template.warp import KernelConfig, TemplateWarp
from panther_em.inference.template.weights import WeightSource

__all__ = [
    "AdjointBases",
    "KernelConfig",
    "Template",
    "TemplateWarp",
    "WarpAdjoint",
    "WeightSource",
    "build_bases",
    "unsupported_reason",
]
