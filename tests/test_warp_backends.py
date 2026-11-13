"""The warp backends against each other."""

from __future__ import annotations

import numpy as np
import torch
from conftest import requires_cuda

from panther_em.coordinates.warp_backends import warp_numpy, warp_torch_cuda


@requires_cuda
def test_cuda_warp_matches_cpu_warp_at_large_coordinates():
    """cuCIM is run in float64 for float64 coordinates on a float32 image: its own
    float32 path shifts the spline stencil by a sample where a coordinate rounds up
    to an integer (values above ~600)."""
    gen = torch.Generator().manual_seed(0)
    image = torch.randn(1100, 900, generator=gen)
    coords = torch.stack(
        [
            torch.rand(256, 256, generator=gen, dtype=torch.float64) * 1099,
            torch.rand(256, 256, generator=gen, dtype=torch.float64) * 899,
        ]
    )
    cpu = warp_numpy(image.numpy(), coords.numpy(), output_shape=(256, 256), order=5)
    gpu = warp_torch_cuda(image.cuda(), coords.cuda(), output_shape=(256, 256), order=5)
    assert gpu.dtype == torch.float32
    err = np.abs(gpu.cpu().numpy() - cpu).max() / np.abs(cpu).max()
    assert err < 1e-5, err
