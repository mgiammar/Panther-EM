"""Shared fixtures: a small synthetic decomposition and GPU gating helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from panther_em.coordinates.offset_polar import OffsetPolarTransform
from panther_em.decomposition.result import DecompositionResult


def make_synthetic_result(
    *,
    n_filters: int = 1,
    n_orientations: int = 6,
    k_max: int = 4,
    eig_max: int = 5,
    n_radial: int = 8,
    n_angular: int = 24,
    box: int = 20,
    seed: int = 0,
    with_angles: bool = True,
) -> DecompositionResult:
    """Random real-valued decomposition on a ``box x box`` template grid."""
    rng = np.random.default_rng(seed)
    shape_u = (n_filters, n_orientations, k_max, eig_max)
    U = (rng.standard_normal(shape_u) + 1j * rng.standard_normal(shape_u)).astype(
        np.complex64
    )
    S = rng.random((k_max, eig_max)).astype(np.float32)
    Vh = (
        rng.standard_normal((k_max, eig_max, n_radial))
        + 1j * rng.standard_normal((k_max, eig_max, n_radial))
    ).astype(np.complex64)
    transform = OffsetPolarTransform.from_image(
        image_shape=(box, box), num_angle=n_angular, num_radius=n_radial
    )
    angles = {}
    if with_angles:
        angles = {
            "phi_values": np.linspace(0.0, 300.0, n_orientations, dtype=np.float32),
            "theta_values": np.linspace(0.0, 90.0, n_orientations, dtype=np.float32),
        }
    return DecompositionResult(
        U=U,
        S=S,
        Vh=Vh,
        k_max=k_max,
        eig_max=eig_max,
        is_complex_projection=False,
        num_fourier_filters=n_filters,
        num_orientations=n_orientations,
        num_angular_components=n_angular,
        num_radial_components=n_radial,
        coordinate_transform=transform,
        **angles,
    )


@pytest.fixture
def synthetic_result():
    """Factory fixture: ``synthetic_result(**overrides) -> DecompositionResult``."""
    return make_synthetic_result


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
