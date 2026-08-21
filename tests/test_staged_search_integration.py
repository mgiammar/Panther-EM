"""Regression test for stage-boundary handling in `_run_stage`'s pixel loop.

`PixelStager` (see `panther_em.inference.search.staging`) can split the pixel axis
into chunks ("stages") whose size is driven by a byte budget (`stage_bytes`) that has
no relationship to the caller's `pixel_batch`. A stage can therefore end mid-batch.
This previously desynced `_run_stage`'s pixel-index slice (taken from the full,
un-clipped `pixel_index`) from the actual (stage-clipped) feature data, silently
handing a smaller batch of features to a `PixelStats` accumulator sized for a larger
one -- caught by running a real `compressed_search` with a tiny `stage_bytes` forcing
several misaligned stage boundaries and comparing against a single-stage baseline.

This runs entirely on CPU with a small synthetic decomposition, so it stays in the
regular (GPU-less) test suite even though the bug only reproduces with `pixel_batch`
values that don't evenly divide the forced stage size.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import panther_em.inference.search.compressed as compressed_mod
from panther_em.coordinates.offset_polar import OffsetPolarTransform
from panther_em.decomposition.result import DecompositionResult
from panther_em.inference.projection_reconstruction import ProjectionReconstructor
from panther_em.inference.search import compressed_search

NUM_FF = 1
NUM_OR = 6
K_MAX = 4
EIG_MAX = 5
NUM_R = 8
NUM_ANG = 24
CARTESIAN_SIZE = 20
IMAGE_SIZE = 30  # gives a (11, 11) = 121-pixel valid-correlation grid


def _make_reconstructor(device: torch.device) -> ProjectionReconstructor:
    rng = np.random.default_rng(0)
    num_freq_blocks = K_MAX

    U = (
        rng.standard_normal((NUM_FF, NUM_OR, num_freq_blocks, EIG_MAX)).astype(
            np.float32
        )
        + 1j
        * rng.standard_normal((NUM_FF, NUM_OR, num_freq_blocks, EIG_MAX)).astype(
            np.float32
        )
    ).astype(np.complex64)
    S = rng.random((num_freq_blocks, EIG_MAX)).astype(np.float32)
    Vh = (
        rng.standard_normal((num_freq_blocks, EIG_MAX, NUM_R)).astype(np.float32)
        + 1j * rng.standard_normal((num_freq_blocks, EIG_MAX, NUM_R)).astype(np.float32)
    ).astype(np.complex64)
    transform = OffsetPolarTransform.from_image(
        image_shape=(CARTESIAN_SIZE, CARTESIAN_SIZE),
        num_angle=NUM_ANG,
        num_radius=NUM_R,
    )
    result = DecompositionResult(
        U=U,
        S=S,
        Vh=Vh,
        k_max=K_MAX,
        eig_max=EIG_MAX,
        is_complex_projection=False,
        num_fourier_filters=NUM_FF,
        num_orientations=NUM_OR,
        num_angular_components=NUM_ANG,
        num_radial_components=NUM_R,
        coordinate_transform=transform,
    )
    return ProjectionReconstructor(result=result, device=device)


def _stub_featurize_cells(
    image, reconstructor, cells, *, feature_chunk=None, feature_store_device=None, **_
):
    """Bypass the real polar-to-cartesian kernel construction.

    The coordinate-transform pipeline is irrelevant to what this test targets
    (`_run_stage`'s pixel/stage-boundary batching) and this also sidesteps an
    unrelated numpy/torch deprecation warning raised deep in
    `GridTransform.to_cartesian` under this environment's numpy version.
    """
    image_bhw = image if image.dim() == 3 else image.unsqueeze(0)
    b, h, w = image_bhw.shape
    k_h, k_w = reconstructor.image_shape
    n_px = b * (h - k_h + 1) * (w - k_w + 1)
    gen = torch.Generator().manual_seed(1)
    real = torch.randn(int(cells.shape[0]), n_px, generator=gen)
    imag = torch.randn(int(cells.shape[0]), n_px, generator=gen)
    feats = torch.complex(real, imag).to(torch.complex64)
    return feats.to(feature_store_device or reconstructor.device)


@pytest.mark.parametrize("pixel_batch", [5, 7, 11, 40])
def test_tiny_stage_bytes_matches_single_stage_baseline(pixel_batch, monkeypatch):
    monkeypatch.setattr(compressed_mod, "featurize_cells", _stub_featurize_cells)

    device = torch.device("cpu")
    reconstructor = _make_reconstructor(device)
    gen = torch.Generator().manual_seed(0)
    image = torch.randn(IMAGE_SIZE, IMAGE_SIZE, generator=gen)
    rectangles = [(0, 0, K_MAX, EIG_MAX)]

    common_kwargs = {
        "pixel_batch": pixel_batch,
        "hyp_batch": 5,  # also doesn't evenly divide NUM_OR (6)
        "n_psi": 16,
        "feature_chunk": 8,
        "compute_device": device,
        "feature_store_device": device,
        "show_progress": False,
        "use_fused_kernel": False,
    }

    baseline = compressed_search(
        image, reconstructor, rectangles, stage_bytes=1024**3, **common_kwargs
    )
    # A handful of bytes forces `PixelStager` into many small stages whose
    # boundaries essentially never align with `pixel_batch`.
    staged = compressed_search(
        image, reconstructor, rectangles, stage_bytes=64, **common_kwargs
    )

    for key in baseline:
        assert torch.allclose(baseline[key].float(), staged[key].float()), key
