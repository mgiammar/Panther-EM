"""1-D spectra on ``torch_fourier_filter``'s frequency axis and their DFT grids."""

from __future__ import annotations

import torch
from torch_fourier_filter.utils import curve_1dim_to_ndim
from torch_fourier_filter.whitening import (
    calculate_num_freq_bins,
    power_spectral_density,
)
from torch_grid_utils.fftfreq_grid import fftfreq_grid  # type: ignore[import-untyped]

CORNER_FREQ = 2**0.5 / 2  # the 2-D Fourier corner, the end of the frequency axis


def frequency_axis(
    num_freq_bins: int, device: torch.device | str | None = None
) -> torch.Tensor:
    """The ``(F,)`` frequency axis ``torch_fourier_filter`` bins spectra on."""
    return torch.linspace(0.0, CORNER_FREQ, num_freq_bins, device=device)


def amplitude_spectrum_1d(
    image: torch.Tensor, *, num_freq_bins: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMS amplitude spectrum of a real image.

    Parameters
    ----------
    image : torch.Tensor
        ``(H, W)`` real image.
    num_freq_bins : int, optional
        Bins of the axis; defaults to ``calculate_num_freq_bins`` of the shape.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``freq (F,)`` and ``spectrum (F,)``, float32.
    """
    if image.dim() != 2 or image.is_complex():
        raise ValueError(
            f"image must be a real (H, W) tensor, got {tuple(image.shape)}"
        )

    shape = (int(image.shape[0]), int(image.shape[1]))

    if num_freq_bins is None:
        num_freq_bins = calculate_num_freq_bins(shape)

    # a CUDA rfft2 result is not contiguous; power_spectral_density flattens with view
    dft = torch.fft.rfft2(image.to(torch.float32)).contiguous()
    dft[0, 0] = 0.0
    spectrum = power_spectral_density(dft, rfft=True, num_freq_bins=num_freq_bins)

    return frequency_axis(num_freq_bins, device=spectrum.device), spectrum


def whitening_curve(
    spectrum: torch.Tensor, freq: torch.Tensor, max_freq: float | None = 0.5
) -> torch.Tensor:
    """``1 / spectrum`` with a normalization to the maximum value below ``max_freq``.

    Parameters
    ----------
    spectrum : torch.Tensor
        ``(F,)`` amplitude spectrum.
    freq : torch.Tensor
        ``(F,)`` its frequency axis.
    max_freq : float, optional
        Frequency above which the filter is 1 (``None``: nowhere). Defaults to 0.5.

    Returns
    -------
    torch.Tensor
        ``(F,)`` filter.
    """
    if tuple(spectrum.shape) != tuple(freq.shape):
        raise ValueError("spectrum and freq must have the same shape")

    curve = torch.where(spectrum == 0, torch.zeros_like(spectrum), 1 / spectrum)
    above = (
        torch.zeros_like(curve, dtype=torch.bool)
        if max_freq is None
        else freq.to(curve.device) > max_freq
    )
    curve = curve / curve[~above].max()

    return torch.where(above, torch.ones_like(curve), curve)


def curve_to_grid(
    curve: torch.Tensor,
    freq: torch.Tensor,
    shape: tuple[int, int],
    *,
    rfft: bool = True,
    fftshift: bool = False,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """A filter curve on the DFT grid of ``shape`` (1 beyond the axis, as a filter).

    Parameters
    ----------
    curve : torch.Tensor
        ``(F,)`` filter values.
    freq : torch.Tensor
        ``(F,)`` their frequency axis.
    shape : tuple[int, int]
        Real-space ``(H, W)`` of the grid.
    rfft, fftshift : bool, optional
        Grid layout, as ``fftfreq_grid``.
    device : torch.device or str, optional
        Where the grid is returned.

    Returns
    -------
    torch.Tensor
        The grid, float32.
    """
    grid = fftfreq_grid(image_shape=shape, rfft=rfft, fftshift=fftshift, norm=True)
    out = curve_1dim_to_ndim(
        curve.cpu(), freq.cpu(), grid, fill_lower=1.0, fill_upper=1.0
    )
    return out.to(device) if device is not None else out
