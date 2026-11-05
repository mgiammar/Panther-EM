"""The dataset's whitening filter: the one source of every whitening grid in a search.

A search applies the whitening in three places, and all three must agree:

- the decomposition's Fourier filter, ``W_avg * band`` on the template box
  (:meth:`WhiteningFilter.template_filter`), and the noise model's signal projection
  under ``W_avg`` alone (:meth:`WhiteningFilter.average_filter`; the micrograph is
  not band-limited);
- the search image, ``W_avg * C_i^2`` on the micrograph
  (:meth:`WhiteningFilter.image_filter`), so that template and image together carry
  the micrograph's own ``W_i^2``;
- the signal weight of the noise model, ``C_i^2``
  (:meth:`WhiteningFilter.correction_grid`).

``W_avg`` is the whitening curve of the RMS amplitude spectrum over the micrographs,
``W_i`` the curve of micrograph ``i`` alone and ``C_i = W_i / W_avg`` its correction.
The band (a low-pass) belongs to the filter as well so the decomposition and the level
map use the same one. Two guards check the links that cannot be removed: the
decomposition must have been built under :meth:`template_filter`
(:meth:`check_template_filter`), and a micrograph must be the one its name refers to
(:meth:`check_image`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import h5py  # type: ignore[import-untyped]
import numpy as np
import torch
from torch_fourier_filter.bandpass import low_pass_filter
from torch_fourier_filter.whitening import calculate_num_freq_bins

from panther_em.spectra import (
    CORNER_FREQ,
    amplitude_spectrum_1d,
    curve_to_grid,
    whitening_curve,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

# Relative L2 tolerance to check amplitude spectra of micrograph against stored filter.
SPECTRUM_TOLERANCE = 1e-3

# Agreement of a decomposition's Fourier filter with template_filter (float32 grids).
FILTER_RTOL = 1e-4
FILTER_ATOL = 1e-6


@dataclass(frozen=True, eq=False)
class WhiteningFilter:
    """Per-micrograph amplitude spectra of a dataset, their whitening and the band.

    Parameters
    ----------
    frequencies : torch.Tensor
        ``(F,)`` float32 frequency axis (cycles per pixel).
    spectra : torch.Tensor
        ``(n, F)`` float32 RMS amplitude spectrum of each micrograph.
    names : tuple[str, ...]
        One distinct name per micrograph.
    band_cutoff, band_falloff : float
        The template band, ``torch_fourier_filter.bandpass.low_pass_filter``'s
        ``cutoff`` and ``falloff`` in cycles per pixel.
    max_freq : float
        Frequency above which the whitening curves are 1. Defaults to 0.5.
    """

    frequencies: torch.Tensor
    spectra: torch.Tensor
    names: tuple[str, ...]
    band_cutoff: float
    band_falloff: float
    max_freq: float = 0.5

    def __post_init__(self) -> None:
        """Validate (never coerce)."""
        f = self.frequencies
        if f.dim() != 1 or f.dtype is not torch.float32 or f.device.type != "cpu":
            raise ValueError("frequencies must be a 1-D float32 CPU tensor")

        s = self.spectra
        if s.dim() != 2 or s.dtype is not torch.float32 or s.device.type != "cpu":
            raise ValueError("spectra must be a 2-D float32 CPU tensor (n, F)")

        if s.shape[0] < 1 or s.shape[1] != f.numel():
            raise ValueError("spectra must hold at least one row of one value per bin")

        if not bool(torch.isfinite(s).all()) or bool((s < 0).any()):
            raise ValueError("spectra must be finite and non-negative")

        if not isinstance(self.names, tuple) or len(self.names) != s.shape[0]:
            raise ValueError("names must be a tuple with one entry per spectrum")

        if len(set(self.names)) != len(self.names):
            raise ValueError("names must be distinct")

        if not 0.0 < self.band_cutoff <= CORNER_FREQ:
            raise ValueError(f"band_cutoff must lie in (0, {CORNER_FREQ:.4f}]")

        if not self.band_falloff >= 0.0:
            raise ValueError("band_falloff must be >= 0")

        if not self.max_freq > 0.0:
            raise ValueError("max_freq must be positive")

    @classmethod
    def from_images(
        cls,
        images: Iterable[tuple[str, torch.Tensor]],
        *,
        band_cutoff: float,
        band_falloff: float = 0.0,
        max_freq: float = 0.5,
    ) -> WhiteningFilter:
        """Spectra of named micrographs, consumed one at a time.

        Parameters
        ----------
        images : iterable of (str, torch.Tensor)
            ``(name, image)`` pairs. Each image is a real ``(H, W)`` micrograph already
            pre-processed in real space.
        band_cutoff, band_falloff : float
            The template band.
        max_freq : float, optional
            Frequency above which the whitening curves are 1. Defaults to 0.5.

        Returns
        -------
        WhiteningFilter
            On the frequency axis of the first image's shape.
        """
        names: list[str] = []
        rows: list[torch.Tensor] = []
        frequencies: torch.Tensor | None = None
        n_bins: int | None = None

        for item in images:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("images must yield (name, image) pairs")

            name, image = item

            if n_bins is None:
                n_bins = calculate_num_freq_bins(tuple(image.shape[-2:]))
            freq, spectrum = amplitude_spectrum_1d(image, num_freq_bins=n_bins)

            del image  # hold one micrograph at a time

            if frequencies is None:
                frequencies = freq.cpu()

            names.append(str(name))
            rows.append(spectrum.cpu())

        if frequencies is None:
            raise ValueError("no images given")

        return cls(
            frequencies,
            torch.stack(rows),
            tuple(names),
            float(band_cutoff),
            float(band_falloff),
            float(max_freq),
        )

    # -- lookup ----------------------------------------------------------------

    @property
    def n_images(self) -> int:
        """Micrographs the filter was built from."""
        return int(self.spectra.shape[0])

    @property
    def n_bins(self) -> int:
        """``F``."""
        return int(self.frequencies.numel())

    def row(self, name: str) -> int:
        """Row of the micrograph called ``name``."""
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"no micrograph named {name!r}") from None

    # -- curves ----------------------------------------------------------------

    @cached_property
    def average_spectrum(self) -> torch.Tensor:
        """``(F,)`` RMS over the micrographs of their amplitude spectra."""
        return self.spectra.square().mean(dim=0).sqrt()

    @cached_property
    def average_curve(self) -> torch.Tensor:
        """``(F,)`` ``W_avg``: the whitening curve of the average spectrum."""
        return whitening_curve(self.average_spectrum, self.frequencies, self.max_freq)

    def image_curve(self, name: str) -> torch.Tensor:
        """``(F,)`` ``W_i``: the whitening curve of one micrograph's spectrum."""
        return whitening_curve(
            self.spectra[self.row(name)], self.frequencies, self.max_freq
        )

    def correction_curve(self, name: str) -> torch.Tensor:
        """``(F,)`` ``C_i = W_i / W_avg`` (1 where ``W_avg`` is 0)."""
        avg = self.average_curve
        ratio = self.image_curve(name) / avg.where(avg != 0, torch.ones_like(avg))
        return ratio.where(avg != 0, torch.ones_like(avg))

    # -- grids (rfft layout, unshifted, float32) --------------------------------

    def _grid(
        self,
        curve: torch.Tensor,
        shape: tuple[int, int],
        device: torch.device | str | None,
    ) -> torch.Tensor:
        return curve_to_grid(curve, self.frequencies, shape, device=device)

    def band_grid(
        self, shape: tuple[int, int], device: torch.device | str | None = None
    ) -> torch.Tensor:
        """The band on the rfft grid of the real-space ``shape``."""
        return low_pass_filter(
            self.band_cutoff,
            self.band_falloff,
            (int(shape[0]), int(shape[1])),
            rfft=True,
            fftshift=False,
        ).to(device=device, dtype=torch.float32)

    def average_filter(
        self, shape: tuple[int, int], device: torch.device | str | None = None
    ) -> torch.Tensor:
        """``W_avg``: the filter of a particle's signal in the search image."""
        return self._grid(self.average_curve, shape, device)

    def template_filter(
        self, shape: tuple[int, int], device: torch.device | str | None = None
    ) -> torch.Tensor:
        """``W_avg * band``: the decomposition's Fourier filter on the template box."""
        return self.average_filter(shape, device) * self.band_grid(shape, device)

    def image_filter(
        self,
        name: str,
        shape: tuple[int, int],
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """``W_avg * C_i^2``: the search-image filter of micrograph ``name``."""
        return (
            self.average_filter(shape, device)
            * self.correction_grid(name, shape, device).square()
        )

    def correction_grid(
        self,
        name: str,
        shape: tuple[int, int],
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """``C_i`` of micrograph ``name``."""
        return self._grid(self.correction_curve(name), shape, device)

    # -- guards ----------------------------------------------------------------

    def check_image(self, name: str, image: torch.Tensor) -> None:
        """Raise unless ``image`` has the spectrum recorded for ``name``."""
        row = self.spectra[self.row(name)].to(torch.float64)
        _, spectrum = amplitude_spectrum_1d(image, num_freq_bins=self.n_bins)
        distance = float(
            torch.linalg.vector_norm(spectrum.cpu().to(torch.float64) - row)
            / torch.linalg.vector_norm(row).clamp_min(1e-30)
        )
        if distance > SPECTRUM_TOLERANCE:
            raise ValueError(
                f"the image's amplitude spectrum differs from micrograph {name!r}'s by "
                f"{distance:.2e} (relative L2, tolerance {SPECTRUM_TOLERANCE:.0e}): it "
                "is another micrograph or was pre-processed differently"
            )

    def check_template_filter(self, filters: torch.Tensor | np.ndarray) -> None:
        """Raise unless a decomposition's Fourier filter is :meth:`template_filter`.

        Parameters
        ----------
        filters : torch.Tensor or numpy.ndarray
            ``(kH, kW // 2 + 1)`` or ``(1, kH, kW // 2 + 1)`` rfft grid (the
            decomposition's ``fourier_filters``); the box is ``(kH, kH)``.
        """
        grid = torch.as_tensor(np.asarray(filters))

        if grid.dim() == 3 and grid.shape[0] == 1:
            grid = grid[0]

        if grid.dim() != 2:
            raise ValueError(
                "the decomposition must carry exactly one (kH, kW // 2 + 1) Fourier "
                f"filter, got shape {tuple(grid.shape)}"
            )

        box = (int(grid.shape[0]), int(grid.shape[0]))

        if int(grid.shape[1]) != box[1] // 2 + 1:
            raise ValueError(
                f"filter of shape {tuple(grid.shape)} is not the rfft grid of a square "
                "template box"
            )

        expected = self.template_filter(box).to(grid.dtype)

        if not torch.allclose(grid, expected, rtol=FILTER_RTOL, atol=FILTER_ATOL):
            worst = float((grid - expected).abs().max())
            raise ValueError(
                "the decomposition's Fourier filter is not this whitening filter's "
                f"W_avg * band (max difference {worst:.2e}); the decomposition was "
                "built under another W_avg or band"
            )

    # -- I/O -------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the filter to an HDF5 file."""
        with h5py.File(path, "w") as f:
            f.create_dataset("frequencies", data=self.frequencies.numpy())
            f.create_dataset("spectra", data=self.spectra.numpy())
            f.create_dataset(
                "names", data=np.array(self.names, dtype=h5py.string_dtype())
            )
            f.attrs["band_cutoff"] = self.band_cutoff
            f.attrs["band_falloff"] = self.band_falloff
            f.attrs["max_freq"] = self.max_freq

    @classmethod
    def load(cls, path: str | Path) -> WhiteningFilter:
        """Read a file written by :meth:`save`."""
        with h5py.File(path, "r") as f:
            frequencies = torch.from_numpy(
                np.asarray(f["frequencies"], dtype=np.float32)
            )
            spectra = torch.from_numpy(np.asarray(f["spectra"], dtype=np.float32))
            names = tuple(
                str(n) for n in np.asarray(f["names"].asstr()[()]).reshape(-1)
            )
            return cls(
                frequencies,
                spectra,
                names,
                float(f.attrs["band_cutoff"]),
                float(f.attrs["band_falloff"]),
                float(f.attrs["max_freq"]),
            )
