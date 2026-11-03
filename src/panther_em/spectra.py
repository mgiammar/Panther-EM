"""Average whitening filter for a set of micrographs and per-micrograph corrections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import h5py  # type: ignore[import-untyped]
import numpy as np
import torch
from torch_fourier_filter.utils import curve_1dim_to_ndim
from torch_fourier_filter.whitening import (
    calculate_num_freq_bins,
    power_spectral_density,
)
from torch_grid_utils.fftfreq_grid import fftfreq_grid  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

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


@dataclass(frozen=True, eq=False)
class WhiteningFilter:
    """Per-micrograph amplitude spectra of a dataset and their whitening filters.

    Parameters
    ----------
    freq : torch.Tensor
        ``(F,)`` float32 frequency axis (:func:`frequency_axis`).
    spectra : torch.Tensor
        ``(n, F)`` float32 amplitude spectrum of each micrograph.
    names : tuple[str, ...]
        One distinct identifier per micrograph.
    max_freq : float or None
        Frequency above which the filters are 1.
    """

    freq: torch.Tensor
    spectra: torch.Tensor
    names: tuple[str, ...]
    max_freq: float | None = 0.5

    def __post_init__(self) -> None:
        """Validate (never coerce)."""
        if self.freq.dim() != 1 or self.freq.dtype is not torch.float32:
            raise ValueError("freq must be a 1-D float32 tensor")

        if self.spectra.dim() != 2 or self.spectra.dtype is not torch.float32:
            raise ValueError("spectra must be a 2-D float32 tensor (n, F)")

        if self.spectra.shape[1] != self.freq.numel():
            raise ValueError("spectra must have one column per frequency")

        if not bool(torch.isfinite(self.spectra).all()) or bool(
            (self.spectra < 0).any()
        ):
            raise ValueError("spectra must be finite and non-negative")

        if (
            not isinstance(self.names, tuple)
            or len(self.names) != self.spectra.shape[0]
        ):
            raise ValueError("names must be a tuple with one entry per spectrum")

        if len(set(self.names)) != len(self.names):
            raise ValueError("names must be distinct")

        if self.max_freq is not None and not self.max_freq > 0.0:
            raise ValueError("max_freq must be positive or None")

    @classmethod
    def from_images(
        cls,
        images: Iterable[torch.Tensor | tuple[str, torch.Tensor]],
        *,
        max_freq: float | None = 0.5,
        num_freq_bins: int | None = None,
    ) -> WhiteningFilter:
        """Spectra of every image of an iterable, consumed one image at a time.

        Notes
        -----
        Each iterated item is a real-valued ``(H, W)`` image already pre-processed in
        real space (the caller's masking etc.), or a ``(name, image)`` pair; unnamed
        images are numbered.

        Parameters
        ----------
        images : iterable
            Images or ``(name, image)`` pairs.
        max_freq : float, optional
            Frequency above which the filters are 1. Defaults to 0.5.
        num_freq_bins : int, optional
            Bins of the shared frequency axis; defaults to
            ``calculate_num_freq_bins`` of the first image's shape.

        Returns
        -------
        WhiteningFilter
        """
        names: list[str] = []
        rows: list[torch.Tensor] = []
        freq: torch.Tensor | None = None
        for item in images:
            if isinstance(item, torch.Tensor):
                name, image = str(len(names)), item
            else:
                name, image = str(item[0]), item[1]
            if num_freq_bins is None:
                num_freq_bins = calculate_num_freq_bins(tuple(image.shape[-2:]))
            f, spectrum = amplitude_spectrum_1d(image, num_freq_bins=num_freq_bins)
            del image, item  # hold one micrograph at a time
            if freq is None:
                freq = f.cpu()
            names.append(name)
            rows.append(spectrum.cpu())
        if freq is None:
            raise ValueError("no images given")
        return cls(freq, torch.stack(rows), tuple(names), max_freq)

    # -- lookup --------------------------------------------------------------

    @property
    def n_images(self) -> int:
        """Micrographs the filter was built from."""
        return int(self.spectra.shape[0])

    def index(self, which: str | int) -> int:
        """Row of a micrograph given its name or index."""
        if isinstance(which, str):
            try:
                return self.names.index(which)
            except ValueError:
                raise KeyError(f"no micrograph named {which!r}") from None
        i = int(which)
        if not -self.n_images <= i < self.n_images:
            raise IndexError(f"micrograph index {i} out of range for {self.n_images}")
        return i % self.n_images

    # -- curves --------------------------------------------------------------

    @property
    def average_spectrum(self) -> torch.Tensor:
        """``(F,)`` RMS over the micrographs of their amplitude spectra."""
        return self.spectra.square().mean(dim=0).sqrt()

    @property
    def average_1d(self) -> torch.Tensor:
        """``W_avg``: the whitening curve of the average spectrum."""
        return whitening_curve(self.average_spectrum, self.freq, self.max_freq)

    def filter_1d(self, which: str | int) -> torch.Tensor:
        """``W_i``: the whitening curve of one micrograph's spectrum."""
        return whitening_curve(
            self.spectra[self.index(which)], self.freq, self.max_freq
        )

    def correction_1d(self, which: str | int) -> torch.Tensor:
        """``C_i = W_i / W_avg`` (1 where ``W_avg`` is 0)."""
        avg = self.average_1d
        return torch.where(avg == 0, torch.ones_like(avg), self.filter_1d(which) / avg)

    # -- grids ---------------------------------------------------------------

    def average_grid(
        self,
        shape: tuple[int, int],
        *,
        rfft: bool = True,
        fftshift: bool = False,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """``W_avg`` on the DFT grid of ``shape``."""
        return curve_to_grid(
            self.average_1d,
            self.freq,
            shape,
            rfft=rfft,
            fftshift=fftshift,
            device=device,
        )

    def correction_grid(
        self,
        which: str | int,
        shape: tuple[int, int],
        *,
        rfft: bool = True,
        fftshift: bool = False,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """``C_i`` on the DFT grid of ``shape``."""
        return curve_to_grid(
            self.correction_1d(which),
            self.freq,
            shape,
            rfft=rfft,
            fftshift=fftshift,
            device=device,
        )

    def image_grid(
        self,
        which: str | int,
        shape: tuple[int, int],
        *,
        rfft: bool = True,
        fftshift: bool = False,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """``W_avg * C_i^2`` on the DFT grid of ``shape``."""
        return (
            self.average_grid(shape, rfft=rfft, fftshift=fftshift, device=device)
            * self.correction_grid(
                which, shape, rfft=rfft, fftshift=fftshift, device=device
            ).square()
        )

    # -- I/O -----------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write ``frequency_1d``, ``spectra_1d``, ``names`` and ``filter_1d``."""
        with h5py.File(path, "w") as f:
            f.create_dataset("frequency_1d", data=self.freq.numpy())
            f.create_dataset("spectra_1d", data=self.spectra.numpy())
            f.create_dataset("filter_1d", data=self.average_1d.numpy())
            f.create_dataset(
                "names", data=np.array(self.names, dtype=h5py.string_dtype())
            )
            f.attrs["max_freq"] = -1.0 if self.max_freq is None else self.max_freq

    @classmethod
    def load(cls, path: str | Path) -> WhiteningFilter:
        """Read a file written by :meth:`save`."""
        with h5py.File(path, "r") as f:
            freq = torch.from_numpy(np.asarray(f["frequency_1d"], dtype=np.float32))
            spectra = torch.from_numpy(np.asarray(f["spectra_1d"], dtype=np.float32))
            names = tuple(
                str(n) for n in np.asarray(f["names"].asstr()[()]).reshape(-1)
            )
            max_freq = float(f.attrs["max_freq"])
        return cls(freq, spectra, names, None if max_freq < 0 else max_freq)
