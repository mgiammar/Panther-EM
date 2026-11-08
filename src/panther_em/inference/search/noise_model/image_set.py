"""Prepared micrograph for SVD-based search including power and CTF noise-model curves.

- the search image ``x``: the micrograph under ``W_avg * C_i^2``, normalized to have
  zero mean and unit variance.
- ``noise (D, F)``: the ring mean of ``|x^ * CTF_d|^2 / (H W)`` per CTF plane.
- ``signal (D, F)``: the ring mean of ``CTF_d^2 * C_i^2``.
- ``level (H, W)``: the relative local noise level, the mean over the template's
  support disk at every pixel of the image flattened by its own noise curve inside
  the band (mean 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from panther_em.inference.search.noise_model.curves import curve_to_grid

if TYPE_CHECKING:
    from panther_em.inference.search.noise_model.spatial_frequency_error import (
        PerSpatialFrequencyErrorTable,
    )
    from panther_em.inference.search.noise_model.whitening import WhiteningFilter

# Fraction of the in-band mean noise power the level map's flattening is floored at
# (keeps near-empty rings from being amplified).
FLATTEN_FLOOR = 0.05

# Band grid values above which a frequency counts as in band.
IN_BAND = 1e-3


def disk_mean(values: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Mean of ``values (H, W)`` over the support disk centred at every pixel."""
    h, w = int(values.shape[-2]), int(values.shape[-1])
    k_h, k_w = int(support.shape[0]), int(support.shape[1])
    full = (h + k_h - 1, w + k_w - 1)

    kernel = support.to(values.dtype) / support.sum()
    kernel_spectrum = torch.fft.rfft2(kernel, s=full)
    numerator = torch.fft.irfft2(
        torch.fft.rfft2(values, s=full) * kernel_spectrum, s=full
    )
    denominator = torch.fft.irfft2(
        torch.fft.rfft2(torch.ones_like(values), s=full) * kernel_spectrum, s=full
    )

    # Windowed re-centering to per-pixel
    r0, c0 = (k_h - 1) // 2, (k_w - 1) // 2
    window = (slice(r0, r0 + h), slice(c0, c0 + w))

    mean: torch.Tensor = numerator[window] / denominator[window].clamp_min(1e-12)
    return mean


@dataclass(frozen=True, eq=False)
class ImageSet:
    """One micrograph's search image, CTF planes and noise-model curves.

    Build with :meth:`from_micrograph`; the constructor only validates.

    Parameters
    ----------
    name : str
        The micrograph's name in the :class:`~.whitening.WhiteningFilter`.
    image : torch.Tensor
        ``(H, W)`` float32 search image (the micrograph under ``W_avg * C_i^2``, zero
        mean, unit variance).
    ctf : torch.Tensor
        ``(D, H, W // 2 + 1)`` float32 CTF planes in rfft layout, on ``image``'s device.
    frequencies : torch.Tensor
        ``(F,)`` float32 CPU axis of the curves (the error tables' axis).
    noise, signal : torch.Tensor
        ``(D, F)`` float64 CPU noise power and signal weight per plane and ring.
    level : torch.Tensor
        ``(H, W)`` float32 CPU relative noise level (mean 1).
    pixel_size : float
        Angstrom per pixel, for the B-factor envelope of the signal weight.
    """

    name: str
    image: torch.Tensor
    ctf: torch.Tensor
    frequencies: torch.Tensor
    noise: torch.Tensor
    signal: torch.Tensor
    level: torch.Tensor
    pixel_size: float

    def __post_init__(self) -> None:
        """Validate (never coerce); see :meth:`from_micrograph`."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")

        x = self.image
        if x.dim() != 2 or x.dtype is not torch.float32:
            raise ValueError("image must be a float32 (H, W) tensor")
        h, w = int(x.shape[0]), int(x.shape[1])

        c = self.ctf
        if c.dim() != 3 or c.dtype is not torch.float32 or c.shape[0] < 1:
            raise ValueError("ctf must be a float32 (D, H, W // 2 + 1) tensor, D >= 1")
        if tuple(c.shape[1:]) != (h, w // 2 + 1):
            raise ValueError(
                f"ctf planes must be ({h}, {w // 2 + 1}) for a {(h, w)} image, got "
                f"{tuple(c.shape[1:])}"
            )
        if c.device != x.device:
            raise ValueError("ctf must be on the image's device")

        f = self.frequencies
        if f.dim() != 1 or f.dtype is not torch.float32 or f.device.type != "cpu":
            raise ValueError("frequencies must be a 1-D float32 CPU tensor")
        if f.numel() < 2 or not bool(torch.all(f[1:] > f[:-1])):
            raise ValueError("frequencies must increase over at least two bins")

        n_planes, n_bins = int(c.shape[0]), int(f.numel())
        for label, t in (("noise", self.noise), ("signal", self.signal)):
            if (
                tuple(t.shape) != (n_planes, n_bins)
                or t.dtype is not torch.float64
                or t.device.type != "cpu"
            ):
                raise ValueError(
                    f"{label} must be a float64 CPU ({n_planes}, {n_bins}) tensor"
                )
            if not bool(torch.isfinite(t).all()) or bool((t < 0).any()):
                raise ValueError(f"{label} must be finite and non-negative")

        lv = self.level
        if (
            tuple(lv.shape) != (h, w)
            or lv.dtype is not torch.float32
            or lv.device.type != "cpu"
        ):
            raise ValueError(f"level must be a float32 CPU {(h, w)} tensor")
        if not bool(torch.isfinite(lv).all()) or bool((lv <= 0).any()):
            raise ValueError("level must be finite and positive")

        if not self.pixel_size > 0.0:
            raise ValueError("pixel_size must be positive")

    @classmethod
    def from_micrograph(
        cls,
        micrograph: torch.Tensor,
        name: str,
        whitening: WhiteningFilter,
        ctf: torch.Tensor,
        table: PerSpatialFrequencyErrorTable,
        *,
        pixel_size: float,
    ) -> ImageSet:
        """Whiten one micrograph and estimate its noise-model curves and level map.

        Parameters
        ----------
        micrograph : torch.Tensor
            ``(H, W)`` real micrograph, pre-processed exactly as the images the
            whitening filter was built from (checked).
        name : str
            Key reference into ``whitening`` for the micrograph's correction filters.
        whitening : WhiteningFilter
            ``W_avg`` across dataset, the micrograph's ``C_i`` and the band.
        ctf : torch.Tensor
            ``(D, H, W // 2 + 1)`` real CTF planes in rfft layout.
        table : PerSpatialFrequencyErrorTable
            An error table of the template along its spatial  frequency axis. Carries
            the curves, its support disk the level map.
        pixel_size : float
            Angstrom per pixel.

        Returns
        -------
        ImageSet
            Image and CTF planes on the micrograph's device; curves and map on the CPU.
        """
        if micrograph.dim() != 2 or micrograph.is_complex():
            raise ValueError(
                f"micrograph must be a real (H, W) tensor, got "
                f"{tuple(micrograph.shape)}"
            )

        if ctf.is_complex():
            raise ValueError("ctf planes must be real")

        h, w = int(micrograph.shape[0]), int(micrograph.shape[1])

        if ctf.dim() != 3 or tuple(ctf.shape[1:]) != (h, w // 2 + 1):
            raise ValueError(
                f"ctf planes must be (D, {h}, {w // 2 + 1}) for a {(h, w)} micrograph, "
                f"got {tuple(ctf.shape)}"
            )

        whitening.check_template_filter(table.template_filter)
        whitening.check_image(name, micrograph)

        device = micrograph.device
        shape = (h, w)
        ctf = ctf.to(device=device, dtype=torch.float32)
        freq = table.frequencies

        # Prepare image with W_avg * C_i^2, then set mean zero, variance one.
        filtered = torch.fft.irfft2(
            torch.fft.rfft2(micrograph.to(torch.float32))
            * whitening.image_filter(name, shape, device),
            s=shape,
        )
        image = (filtered - filtered.mean()) / filtered.std()
        spectrum = torch.fft.rfft2(image)

        # full-grid ring means on the table's axis (bins shared by every micrograph)
        bins = table.image_bins(shape, device)

        def ring_mean(values: torch.Tensor) -> torch.Tensor:
            return bins.ring_mean(values[None].to(torch.float64))[0].cpu()

        power = spectrum.abs().square().to(torch.float64)
        correction = whitening.correction_grid(name, shape, device).square()
        n_planes = int(ctf.shape[0])
        noise = torch.empty((n_planes, freq.numel()), dtype=torch.float64)
        signal = torch.empty_like(noise)
        for d in range(n_planes):
            ctf2 = ctf[d].square()
            noise[d] = ring_mean(power * ctf2) / float(h * w)
            signal[d] = ring_mean(ctf2 * correction)

        image_noise = ring_mean(power) / float(h * w)  # the unfiltered image's curve
        level = _level_map(spectrum, image_noise, freq, table.support, whitening, shape)

        return cls(str(name), image, ctf, freq, noise, signal, level, float(pixel_size))

    # -- sizes -----------------------------------------------------------------

    @property
    def image_shape(self) -> tuple[int, int]:
        """``(H, W)``."""
        return int(self.image.shape[0]), int(self.image.shape[1])

    @property
    def n_defocus(self) -> int:
        """CTF planes ``D``."""
        return int(self.ctf.shape[0])

    @property
    def n_bins(self) -> int:
        """``F``."""
        return int(self.frequencies.numel())


def _level_map(
    spectrum: torch.Tensor,
    noise: torch.Tensor,
    frequencies: torch.Tensor,
    support: torch.Tensor,
    whitening: WhiteningFilter,
    shape: tuple[int, int],
) -> torch.Tensor:
    """``(H, W)`` relative local noise level of the search image (mean 1, CPU).

    Notes
    -----
    The image (``spectrum``, its rfft) is flattened inside the band by
    ``band / sqrt(max(S, floor * mean_band S))`` with ``S`` its own noise curve
    ``noise (F,)``, squared, and averaged over the support disk at every pixel.
    """
    device = spectrum.device
    s_grid = curve_to_grid(noise.to(torch.float32), frequencies, shape, device=device)
    band = whitening.band_grid(shape, device)
    s_band = s_grid[band > IN_BAND].mean()
    flatten = band / torch.sqrt(s_grid.clamp_min(FLATTEN_FLOOR * s_band))
    flat = torch.fft.irfft2(spectrum * flatten, s=shape)
    local = disk_mean(flat.square(), support.to(device))
    local = local.clamp_min(torch.finfo(torch.float32).tiny)
    return (local / local.mean()).to(torch.float32).cpu()
