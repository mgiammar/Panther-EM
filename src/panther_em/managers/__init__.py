"""The programs of a Panther-EM pipeline, each a config (pure data, from YAML) + run.

In pipeline order:

1. :mod:`.whitening` -- micrographs -> the dataset's whitening filter;
2. :mod:`.decompose` -- template volume (+ whitening) -> decomposition;
3. :mod:`.calibrate` -- decomposition + volume -> error tables per feature selection
   (and the cheap errors and rank suggestions);
4. :mod:`.search` -- micrographs -> one search result each (resumable);
5. :mod:`.export` -- search results -> a Leopard-EM particle stack (HDF5) for its
   template refinement (the Leopard-EM interface: :mod:`.leopard`).

Each program's ``run(config)`` builds the domain objects and calls them; the domain
constructors validate the values. Every file a program writes records its config
(HDF5 attributes ``panther_em_program`` / ``panther_em_config``), which ``export``
follows back to the optics, the micrograph and the template volume. The
``panther-em`` command (:mod:`panther_em.cli`) runs a program on a YAML file.
"""

from panther_em.managers.calibrate import CalibrateConfig
from panther_em.managers.decompose import DecomposeConfig
from panther_em.managers.export import ExportConfig
from panther_em.managers.search import SearchConfig
from panther_em.managers.whitening import WhiteningConfig

__all__ = [
    "CalibrateConfig",
    "DecomposeConfig",
    "ExportConfig",
    "SearchConfig",
    "WhiteningConfig",
]
