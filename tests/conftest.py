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
    s2_nside: int | None = None,
) -> DecompositionResult:
    """Random real-valued decomposition on a ``box x box`` template grid.

    ``s2_nside`` samples the orientations on that HEALPix grid (NESTED order, sets
    ``n_orientations = 12 * nside**2``); only the Euler angles are recorded.
    """
    rng = np.random.default_rng(seed)
    if s2_nside is not None:
        n_orientations = 12 * s2_nside * s2_nside
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
    angles: dict = {}
    if s2_nside is not None:
        from panther_em.inference.search.hypotheses import healpix_orientations

        phi, theta = healpix_orientations(s2_nside)
        angles = {"phi_values": phi, "theta_values": theta}
    elif with_angles:
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


# --------------------------------------------------------------------------- #
# Test-only helpers that peek inside engine layouts / compute dense references
# --------------------------------------------------------------------------- #


def reference_spectrum(y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Dense ``C[p, n, k] = sum_m Y[p, k, m] conj(W[n, k, m])`` (complex64)."""
    return torch.einsum(
        "pkm,nkm->pnk", y.to(torch.complex64), w.conj().to(torch.complex64)
    )


def read_engine_block(
    engine, block: torch.Tensor, plane: int, cols: slice | None = None
):
    """``(P, num_k, num_m)`` complex64 features of one plane of an engine block."""
    num_m = engine.block_num_m(block)
    sub = block[plane] if cols is None else block[plane][:, cols]  # (num_k, P, inner)
    if block.dtype == torch.float16:
        sub = torch.complex(sub[..., :num_m].float(), sub[..., num_m:].float())
    return sub.permute(1, 0, 2)


def read_group(store, group, plane: int = 0) -> torch.Tensor:
    """``(P, num_k, num_m)`` complex64 features of a column group's plane."""
    parts = []
    for i, block in enumerate(group.blocks):
        start, stop = store.block_range(i)
        parts.append(
            read_engine_block(group.engine, block, plane, slice(0, stop - start))
        )
    return torch.cat(parts, dim=0)


def count_kernel_builds(bank):
    """Wrap ``bank._build_group`` so ``counter[0]`` counts kernels built from now on."""
    counter = [0]
    original = bank._build_group

    def counting(k, ms):
        counter[0] += len(ms)
        return original(k, ms)

    bank._build_group = counting
    return counter
