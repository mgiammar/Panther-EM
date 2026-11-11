"""filters.ctf: the CTF planes in Angstroms, one per defocus."""

from __future__ import annotations

import torch
from torch_ctf import calculate_ctf_2d

from panther_em.filters import ctf_planes


def test_ctf_planes_match_torch_ctf_in_micrometers():
    planes = ctf_planes(
        (32, 48),
        [10000.0, 15000.0, 20000.0],
        pixel_size=1.2,
        astigmatism=300.0,
        astigmatism_angle=30.0,
        phase_shift=10.0,
    )
    assert planes.shape == (3, 32, 25) and planes.dtype is torch.float32
    reference = calculate_ctf_2d(
        defocus=torch.tensor([1.0, 1.5, 2.0]),
        astigmatism=0.03,
        astigmatism_angle=30.0,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.07,
        phase_shift=10.0,
        pixel_size=1.2,
        image_shape=(32, 48),
        rfft=True,
        fftshift=False,
    )
    torch.testing.assert_close(planes, reference.float())
    one = ctf_planes((32, 48), torch.tensor(15000.0), pixel_size=1.2)
    assert one.shape == (1, 32, 25)
