"""filters.ctf: the CTF planes in Angstroms, one per defocus."""

from __future__ import annotations

import pytest
import torch
from leopard_em.pydantic_models.data_structures import OpticsGroup
from leopard_em.utils.ctf_utils import calculate_ctf_filter_stack
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


@pytest.mark.parametrize(
    ("defocus_u", "defocus_v", "angle"),
    [(15000.0, 15000.0, 0.0), (15600.0, 14400.0, 30.0)],
)
def test_ctf_planes_match_leopard_em(defocus_u, defocus_v, angle):
    """Same planes as Leopard-EM's filter stack (for ``defocus_u >= defocus_v``)."""
    offsets = [-500.0, 0.0, 500.0]
    optics = OpticsGroup(
        label="g",
        pixel_size=1.06,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast_ratio=0.07,
        phase_shift=10.0,
        defocus_u=defocus_u,
        defocus_v=defocus_v,
        astigmatism_angle=angle,
    )
    reference = calculate_ctf_filter_stack(
        (32, 48), optics, torch.tensor(offsets), torch.zeros(1)
    )[0]
    mean = (defocus_u + defocus_v) / 2
    planes = ctf_planes(
        (32, 48),
        [mean + o for o in offsets],
        pixel_size=1.06,
        astigmatism=(defocus_u - defocus_v) / 2,
        astigmatism_angle=angle,
        phase_shift=10.0,
    )
    torch.testing.assert_close(planes, reference.float())


def test_negative_astigmatism_is_the_rotated_positive_one():
    """``defocus_u < defocus_v``: finite, and the swapped axes rotated by 90 degrees."""
    kwargs = {"pixel_size": 1.06, "phase_shift": 10.0}
    flipped = ctf_planes(
        (32, 48), [15000.0], astigmatism=-600.0, astigmatism_angle=30.0, **kwargs
    )
    rotated = ctf_planes(
        (32, 48), [15000.0], astigmatism=600.0, astigmatism_angle=120.0, **kwargs
    )
    assert torch.isfinite(flipped).all()
    torch.testing.assert_close(flipped, rotated)
