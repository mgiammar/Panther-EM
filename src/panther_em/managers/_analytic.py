"""Sampling of the analytic null, shared by the ``null`` and ``search`` programs."""

from __future__ import annotations

from pydantic import Field

from panther_em.managers._common import Model


class AnalyticSpec(Model):
    """Sampling of the analytic null (``panther_em.search.null_model.predict_null``)."""

    n_mean_samples: int = Field(default=3000, ge=2)
    n_kappa_pairs: int = Field(default=256, ge=2)
    n_lattice_samples: int = Field(default=128, ge=1)
    batch_size: int = Field(default=4, ge=1)
    seed: int = 0
    device: str | None = None  # default: the search's device
