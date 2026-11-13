"""CTF planes of a micrograph: one rfft-grid filter per searched defocus."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch_ctf import calculate_ctf_2d

if TYPE_CHECKING:
    from collections.abc import Sequence


def ctf_planes(
    shape: tuple[int, int],
    defocus: Sequence[float] | torch.Tensor,
    *,
    pixel_size: float,
    astigmatism: float = 0.0,
    astigmatism_angle: float = 0.0,
    voltage: float = 300.0,
    spherical_aberration: float = 2.7,
    amplitude_contrast: float = 0.07,
    phase_shift: float = 0.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """``(D, H, W // 2 + 1)`` float32 CTFs of an ``(H, W)`` image, one per defocus.

    Parameters
    ----------
    shape : tuple of int
        ``(H, W)`` real-space image shape.
    defocus : sequence of float or torch.Tensor
        ``(D,)`` mean defocus ``(defocus_u + defocus_v) / 2`` of each plane, in
        Angstroms (positive is underfocus).
    pixel_size : float
        Angstroms per pixel.
    astigmatism : float
        ``(defocus_u - defocus_v) / 2`` in Angstroms, shared by every plane. A
        negative value is the same CTF as its magnitude at ``astigmatism_angle + 90``.
        Leopard-EM's ``calculate_ctf_filter_stack`` takes ``|defocus_u - defocus_v| /
        2`` at the unrotated angle, so the two agree only when
        ``defocus_u >= defocus_v``.
    astigmatism_angle : float
        Degrees; 0 places ``defocus_u`` along the y-axis.
    voltage : float
        Acceleration voltage in kV.
    spherical_aberration : float
        Millimeters.
    amplitude_contrast : float
        Fraction in ``[0, 1]``.
    phase_shift : float
        Degrees.
    device : torch.device or str, optional
        Where the planes are built. Default CPU.
    """
    values = torch.as_tensor(defocus, dtype=torch.float32, device=device).reshape(-1)
    if astigmatism < 0:  # torch-ctf needs a non-negative astigmatism: the same CTF
        astigmatism, astigmatism_angle = -astigmatism, astigmatism_angle + 90.0
    ctf = calculate_ctf_2d(
        defocus=values * 1e-4,  # torch-ctf takes micrometers
        astigmatism=astigmatism * 1e-4,
        astigmatism_angle=astigmatism_angle,
        voltage=voltage,
        spherical_aberration=spherical_aberration,
        amplitude_contrast=amplitude_contrast,
        phase_shift=phase_shift,
        pixel_size=pixel_size,
        image_shape=(int(shape[0]), int(shape[1])),
        rfft=True,
        fftshift=False,
    )
    h, w = shape
    return ctf.reshape(values.numel(), h, w // 2 + 1).to(torch.float32)
