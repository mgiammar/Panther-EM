"""A small noise-model case shared by tests.

Three coloured micrographs, their whitening filter, a Gaussian-blob volume
decomposed under ``W_avg * band`` (the template every table and search of the case
uses), its error tables and a prepared micrograph.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch_grid_utils.fftfreq_grid import fftfreq_grid

from panther_em.coordinates import OffsetPolarTransform
from panther_em.decomposition import PolarProjectionDecomposer
from panther_em.filters import WhiteningFilter
from panther_em.inference.search.noise_model import ImageSet, compute_error_tables
from panther_em.inference.template import Template

SHAPE = (96, 96)
BOX = 24
BAND = (0.35, 0.05)
PIXEL_SIZE = 1.5


def coloured(shape, seed, scale=1.0):
    """Gaussian noise under a smooth low-pass envelope (a non-flat spectrum)."""
    x = torch.randn(shape, generator=torch.Generator().manual_seed(seed))
    r = fftfreq_grid(image_shape=shape, rfft=True, fftshift=False, norm=True)
    envelope = 0.2 + torch.exp(-((r / 0.15) ** 2))
    return scale * torch.fft.irfft2(torch.fft.rfft2(x) * envelope, s=shape)


def micrographs():
    return [(f"m{i}", coloured(SHAPE, 10 + i, s)) for i, s in enumerate((1, 0.7, 1.4))]


def ctf_planes(shape=SHAPE, n=3):
    """Real CTF-like planes ``cos(a r^2 + 0.1)``."""
    r = fftfreq_grid(image_shape=shape, rfft=True, fftshift=False, norm=True)
    return torch.stack([torch.cos((60 + 40 * d) * r**2 + 0.1) for d in range(n)])


def decompose(filters):
    """``(volume, Template)``: 6 orientations, 32 angular x 12 radial, eig_max 6."""
    rng = np.random.default_rng(0)
    volume = gaussian_filter(rng.standard_normal((BOX,) * 3), 2.0).astype(np.float32)
    phi = np.linspace(0.0, 300.0, 6, dtype=np.float32)
    theta = np.linspace(10.0, 80.0, 6, dtype=np.float32)
    transform = OffsetPolarTransform.from_image(
        image_shape=(BOX, BOX), num_angle=32, num_radius=12
    )
    result = PolarProjectionDecomposer(
        volume, phi, theta, coordinate_transform=transform, fourier_filters=filters
    ).do_decomposition(eig_max=6, normalize_projections=True, svd_method="dense")
    return volume, Template(result, device="cpu")


class NoiseCase(NamedTuple):
    whitening: WhiteningFilter
    volume: np.ndarray
    template: Template

    def table(self, rects, **kwargs):
        return compute_error_tables(
            self.volume, self.template, rects, self.whitening, **kwargs
        )

    def images(self, table, name="m2", device="cpu"):
        return prepare(self.whitening, table, name, device)


def prepare(whitening, table, name="m2", device="cpu"):
    """The micrograph ``name`` prepared against ``table``."""
    return ImageSet.from_micrograph(
        dict(micrographs())[name].to(device),
        name,
        whitening,
        ctf_planes(),
        table,
        pixel_size=PIXEL_SIZE,
    )


@functools.cache
def noise_case() -> NoiseCase:
    """The whitening filter and the template decomposed under it (built once)."""
    whitening = WhiteningFilter.from_images(
        micrographs(), band_cutoff=BAND[0], band_falloff=BAND[1]
    )
    volume, template = decompose(whitening.template_filter((BOX, BOX))[None])
    return NoiseCase(whitening, volume, template)
