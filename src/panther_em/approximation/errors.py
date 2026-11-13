"""Exact errors of a decomposition's truncated templates against a reference volume.

One pass over the template rows (:func:`_template_rows`) gives every row's

- reconstruction ``m~``: the SVD template restricted to the feature rectangles,
  warped to Cartesian,
- exact template ``m``: the volume's projection under the decomposition's filter,
  normalized (as the decomposition's projections were) and masked to the support,
- residual ``delta = m - m~``,
- signal ``p``: the same projection under ``W_avg`` alone (noise model only).

Two consumers reduce it: :func:`compute_errors` to the
:class:`ApproximationErrors` (by frequency, radius, orientation, total) and
:func:`compute_error_tables` to the noise model's ring cross-spectra
(:class:`PerSpatialFrequencyErrorTable`, under the dataset's whitening).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

import h5py
import numpy as np
import torch
from torch_fourier_filter.whitening import calculate_num_freq_bins
from torch_grid_utils.fftfreq_grid import fftfreq_grid

from panther_em._hdf5 import check_format, write_format
from panther_em.coordinates.warp_config import WarpConfig
from panther_em.decomposition.projections import (
    generate_projection_batch,
    precompute_volume_dft,
)
from panther_em.decomposition.rectangles import as_rectangles
from panther_em.filters.curves import frequency_axis
from panther_em.filters.whitening import (
    FILTER_ATOL,
    FILTER_RTOL,
    canonical_device,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from panther_em.approximation.reconstructor import Reconstructor
    from panther_em.decomposition.rectangles import Rectangle
    from panther_em.decomposition.result import DecompositionResult
    from panther_em.filters.whitening import WhiteningFilter

TABLE_NAMES = (
    "reconstruction_autocorr",
    "residual_autocorr",
    "reconstruction_residual_corr",
    "signal_reconstruction_corr",
    "signal_residual_corr",
)


@dataclass(frozen=True, eq=False)
class PerSpatialFrequencyErrorTable:
    """Helper class for spatial frequency profiles of error and overlap.

    Parameters
    ----------
    frequencies : torch.Tensor
        ``(F,)`` float32 CPU frequency axis (cycles per pixel).
    reconstruction_autocorr : torch.Tensor
        ``(N, F)`` float32 power of the reconstruction ``m~`` per ring, ``N`` the
        template rows.
    residual_autocorr : torch.Tensor
        ``(N, F)`` float32 power of the residual ``delta`` per ring.
    reconstruction_residual_corr : torch.Tensor
        ``(N, F)`` float32 cross-spectrum of ``m~`` and ``delta`` per ring.
    signal_reconstruction_corr : torch.Tensor
        ``(N, F)`` float32 cross-spectrum of the signal ``p`` and ``m~`` per ring.
    signal_residual_corr : torch.Tensor
        ``(N, F)`` float32 cross-spectrum of ``p`` and ``delta`` per ring.
    support : torch.Tensor
        ``(kH, kW)`` bool template support disk.
    template_filter : torch.Tensor
        ``(kH, kW // 2 + 1)`` float32 ``W_avg * band`` the tables were built under.
    rectangles : tuple of Rectangle
        The kept ``(k_start, m_start, k_extent, m_extent)`` feature regions.
    warp : WarpConfig
        The warp the reconstructions and the support disk were built with; a search
        must warp its kernels the same way (:meth:`check`).
    pad : int
        Zero-padding factor of the spectra the tables were binned from. Default 2.
    """

    frequencies: torch.Tensor
    reconstruction_autocorr: torch.Tensor
    residual_autocorr: torch.Tensor
    reconstruction_residual_corr: torch.Tensor
    signal_reconstruction_corr: torch.Tensor
    signal_residual_corr: torch.Tensor
    support: torch.Tensor
    template_filter: torch.Tensor
    rectangles: tuple[Rectangle, ...]
    warp: WarpConfig
    pad: int = 2

    def __post_init__(self) -> None:
        """Validate (never coerce); see :func:`compute_error_tables`."""
        f = self.frequencies
        if f.dim() != 1 or f.dtype is not torch.float32 or f.device.type != "cpu":
            raise ValueError("frequencies must be a 1-D float32 CPU tensor")
        if f.numel() < 2 or not bool(torch.all(f[1:] > f[:-1])):
            raise ValueError("frequencies must increase over at least two bins")

        n_rows = int(self.reconstruction_autocorr.shape[0])
        for name in TABLE_NAMES:
            t = getattr(self, name)
            if (
                tuple(t.shape) != (n_rows, f.numel())
                or t.dtype is not torch.float32
                or t.device.type != "cpu"
            ):
                raise ValueError(
                    f"{name} must be a float32 CPU ({n_rows}, {f.numel()}) tensor"
                )

        s = self.support
        if s.dim() != 2 or s.dtype is not torch.bool or s.device.type != "cpu":
            raise ValueError("support must be a 2-D bool CPU tensor")
        k_h, k_w = int(s.shape[0]), int(s.shape[1])

        g = self.template_filter
        if (
            tuple(g.shape) != (k_h, k_w // 2 + 1)
            or g.dtype is not torch.float32
            or g.device.type != "cpu"
        ):
            raise ValueError(
                f"template_filter must be a float32 CPU ({k_h}, {k_w // 2 + 1}) tensor"
            )

        if not isinstance(self.rectangles, tuple) or not self.rectangles:
            raise ValueError("rectangles must be a non-empty tuple")
        for r in self.rectangles:
            if (
                not isinstance(r, tuple)
                or len(r) != 4
                or not all(isinstance(v, int) for v in r)
                or min(r) < 0
                or r[2] < 1
                or r[3] < 1
            ):
                raise ValueError(f"bad rectangle {r!r}; see compute_error_tables")

        if not isinstance(self.warp, WarpConfig):
            raise ValueError("warp must be a WarpConfig")

        if self.pad < 1:
            raise ValueError("pad must be >= 1")

    # -- sizes -----------------------------------------------------------------

    @property
    def n_templates(self) -> int:
        """Template rows ``N``."""
        return int(self.reconstruction_autocorr.shape[0])

    @property
    def n_bins(self) -> int:
        """``F``."""
        return int(self.frequencies.numel())

    @property
    def template_shape(self) -> tuple[int, int]:
        """``(kH, kW)``."""
        return int(self.support.shape[0]), int(self.support.shape[1])

    def summary(self) -> dict:
        """JSON-serializable description (for result files)."""
        return {
            "n_templates": self.n_templates,
            "n_bins": self.n_bins,
            "template_shape": list(self.template_shape),
            "rectangles": [list(r) for r in self.rectangles],
            "warp": self.warp.to_dict(),
            "pad": self.pad,
        }

    # ------ decomposition guard -----------------------------------------------

    def check(self, reconstructor: Reconstructor) -> None:
        """Raise unless the tables describe ``reconstructor``'s templates and warp."""
        result = reconstructor.result
        if reconstructor.warp != self.warp:
            raise ValueError(
                f"the tables were built with the warp {self.warp}, the reconstructor "
                f"warps with {reconstructor.warp}"
            )
        if tuple(reconstructor.box_shape) != self.template_shape:
            raise ValueError(
                f"the tables were built on a {self.template_shape} box, the "
                f"decomposition's is {tuple(reconstructor.box_shape)}"
            )
        n_templates = int(result.num_fourier_filters) * int(result.num_orientations)
        if n_templates != self.n_templates:
            raise ValueError(
                f"the tables hold {self.n_templates} template rows, the decomposition "
                f"{n_templates}"
            )
        for k0, m0, ke, me in self.rectangles:
            if k0 + ke > result.k_max or m0 + me > result.eig_max:
                raise ValueError(
                    f"rectangle {(k0, m0, ke, me)} lies beyond the decomposition's "
                    f"stored (k_max={result.k_max}, eig_max={result.eig_max})"
                )
        filters = result.fourier_filters
        if filters is None:
            raise ValueError("the decomposition carries no Fourier filter")
        grid = torch.as_tensor(np.asarray(filters))
        if grid.dim() == 3 and grid.shape[0] == 1:
            grid = grid[0]
        if grid.is_complex():
            grid = grid.real
        expected = self.template_filter
        if tuple(grid.shape) != tuple(expected.shape) or not torch.allclose(
            grid.to(expected.dtype), expected, rtol=FILTER_RTOL, atol=FILTER_ATOL
        ):
            raise ValueError(
                "the decomposition was built under another Fourier filter "
                "than the tables' W_avg * band"
            )

    # -- image binning ---------------------------------------------------------

    @cached_property
    def _image_bins(self) -> dict[tuple[tuple[int, int], torch.device], RadialBins]:
        return {}

    def image_bins(
        self, shape: tuple[int, int], device: torch.device | str | None = None
    ) -> RadialBins:
        """float64 bins of the rfft grid of an image ``shape`` onto this table's axis.

        Raises
        ------
        ValueError
            If a bin of the axis holds no frequency of the grid (an image smaller
            than the padded template box).
        """
        key = ((int(shape[0]), int(shape[1])), canonical_device(device))
        bins = self._image_bins.get(key)
        if bins is None:
            bins = RadialBins.create(
                key[0], self.frequencies, device=key[1], dtype=torch.float64
            )
            if bool((bins.counts <= 0).any()):
                raise ValueError(
                    f"the frequency axis has bins no frequency of a {key[0]} grid "
                    "falls in"
                )
            self._image_bins[key] = bins
        return bins

    # -- I/O -------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write to HDF5."""
        with h5py.File(path, "w") as f:
            write_format(f, "error_table")
            f["frequencies"] = self.frequencies.numpy()
            for name in TABLE_NAMES:
                f[name] = getattr(self, name).numpy()
            f["support"] = self.support.numpy()
            f["template_filter"] = self.template_filter.numpy()
            f.attrs["rectangles"] = json.dumps(self.rectangles)
            f.attrs["warp_config"] = json.dumps(self.warp.to_dict())
            f.attrs["pad"] = self.pad

    @classmethod
    def load(cls, path: str | Path) -> PerSpatialFrequencyErrorTable:
        """Read a file written by :meth:`save`."""
        with h5py.File(path, "r") as f:
            check_format(f, "error_table", path)
            return cls(
                frequencies=torch.from_numpy(f["frequencies"][()]),
                **{name: torch.from_numpy(f[name][()]) for name in TABLE_NAMES},
                support=torch.from_numpy(np.asarray(f["support"][()], dtype=bool)),
                template_filter=torch.from_numpy(f["template_filter"][()]),
                rectangles=as_rectangles(json.loads(f.attrs["rectangles"])),
                # tables saved before the rename recorded "kernel_config"
                warp=WarpConfig.from_dict(
                    json.loads(f.attrs.get("warp_config", f.attrs.get("kernel_config")))
                ),
                pad=int(f.attrs["pad"]),
            )


# --------------------------------------------------------------------------- #
# Radial binning
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RadialBins:
    """Linear-interpolation binning of an rfft grid onto a frequency axis."""

    shape: tuple[int, int]
    left: torch.Tensor
    right: torch.Tensor
    w_left: torch.Tensor
    w_right: torch.Tensor
    n_bins: int

    @classmethod
    def create(
        cls,
        shape: tuple[int, int],
        freq: torch.Tensor,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> RadialBins:
        """Bins of the rfft grid of the real-space ``shape`` onto ``freq``."""
        h, w = shape
        n_bins = freq.numel()
        axis = freq.to(device=device, dtype=torch.float64)
        radius = fftfreq_grid(
            image_shape=shape, rfft=True, fftshift=False, norm=True, device=device
        ).to(torch.float64)

        idx = torch.bucketize(radius, axis)
        left = (idx - 1).clamp(0, n_bins - 1)
        right = idx.clamp(0, n_bins - 1)
        span = (axis[right] - axis[left]).clamp_min(1e-6)
        t = torch.where(left == right, 0.0, (radius - axis[left]) / span)

        # rfft columns other than 0 (and w/2 for even w) stand for two cells
        hermitian = torch.full((w // 2 + 1,), 2.0, dtype=torch.float64, device=device)
        hermitian[0] = 1.0
        if w % 2 == 0:
            hermitian[-1] = 1.0
        inside = radius < axis[-1] + (axis[1] - axis[0])
        scale = inside * hermitian / (h * w)

        return cls(
            shape=(h, w),
            left=left.flatten(),
            right=right.flatten(),
            w_left=((1.0 - t) * scale).flatten().to(dtype),
            w_right=(t * scale).flatten().to(dtype),
            n_bins=n_bins,
        )

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        """``(B, h, w // 2 + 1)`` real grid values -> ``(B, F)`` ring profiles."""
        flat = values.flatten(1)
        out = flat.new_zeros((flat.shape[0], self.n_bins))
        out.index_add_(1, self.left, flat * self.w_left)
        out.index_add_(1, self.right, flat * self.w_right)
        return out

    @cached_property
    def counts(self) -> torch.Tensor:
        """``(F,)`` total weight of each bin (the binning of a grid of ones)."""
        out = self.w_left.new_zeros(self.n_bins)
        out.index_add_(0, self.left, self.w_left)
        out.index_add_(0, self.right, self.w_right)
        return out

    def ring_mean(self, values: torch.Tensor) -> torch.Tensor:
        """``(B, h, w // 2 + 1)`` -> ``(B, F)`` means over the full grid's rings.

        The Hermitian weights cancel, so each rfft column counts as often as it
        appears in the full spectrum.
        """
        return self(values) / self.counts


# --------------------------------------------------------------------------- #
# One pass over the template rows
# --------------------------------------------------------------------------- #


def reconstruct_polar(
    u: torch.Tensor, weighted_vh: torch.Tensor, n_angle: int
) -> torch.Tensor:
    """``(B, K, E)`` x ``(K, E, R)`` -> ``(B, n_angle, R)`` float32 polar images."""
    blocks = torch.einsum("bke,ker->bkr", u, weighted_vh)
    polar: torch.Tensor = torch.fft.irfft(blocks, n=n_angle, dim=1, norm="ortho")
    return polar.to(torch.float32)


def exact_and_signal(
    projections: torch.Tensor,
    exact_filter: torch.Tensor | None,
    signal_filter: torch.Tensor | None,
    support: torch.Tensor,
    *,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Exact templates ``m`` (filtered, normalized, masked) and signals ``p``.

    A ``None`` ``exact_filter`` leaves the projections unfiltered; a ``None``
    ``signal_filter`` computes no signal.
    """
    size = projections.shape[-2:]
    spectrum = torch.fft.rfft2(projections)

    exact = (
        projections
        if exact_filter is None
        else torch.fft.irfft2(spectrum * exact_filter, s=size)
    )
    if normalize:
        var, mean = torch.var_mean(exact, dim=(-2, -1), keepdim=True)
        exact = (exact - mean) / torch.sqrt(var + 1e-8)
    exact = exact * support

    signal = (
        None
        if signal_filter is None
        else torch.fft.irfft2(spectrum * signal_filter, s=size)
    )
    return exact, signal


def _row_filters(result: DecompositionResult) -> torch.Tensor | None:
    """``(FF, kH, kW // 2 + 1)`` recorded Fourier filters (``None``: unfiltered)."""
    if result.fourier_filters is None:
        return None
    filters = torch.as_tensor(np.asarray(result.fourier_filters))
    h, w = result.coordinate_transform.cartesian_shape  # type: ignore[union-attr]
    if filters.dim() != 3 or tuple(filters.shape[1:]) != (h, w // 2 + 1):
        raise NotImplementedError(
            f"the decomposition's Fourier filters are {tuple(filters.shape)}; only "
            f"rfft-grid filters (FF, {h}, {w // 2 + 1}) are supported"
        )
    return filters if filters.is_complex() else filters.to(torch.float32)


class _Rows(NamedTuple):
    """One batch of template rows: the rows, ``m~``, ``m`` and ``p`` (or ``None``)."""

    rows: slice
    reconstruction: torch.Tensor
    exact: torch.Tensor
    signal: torch.Tensor | None


@torch.no_grad()
def _template_rows(
    volume: torch.Tensor | np.ndarray,
    reconstructor: Reconstructor,
    rects: tuple[Rectangle, ...],
    *,
    exact_filter: torch.Tensor | None = None,
    signal_filter: torch.Tensor | None = None,
    batch: int,
) -> Iterator[_Rows]:
    """Every template row's reconstruction, exact template and signal, in batches.

    Rows run over ``(Fourier filter, orientation)`` of the decomposition's one volume.
    The exact template ``m`` is the volume's projection under ``exact_filter`` (the
    same for every row) or, when ``None``, under each row's recorded decomposition
    filter; normalized when the decomposition normalized its projections (or did not
    record it); masked to the reconstructor's support. ``p`` is the projection under
    ``signal_filter``, ``None`` without one.
    """
    result = reconstructor.result.to_dense()
    if result.num_volumes != 1:
        raise NotImplementedError(
            f"error tables support one volume per decomposition, got "
            f"{result.num_volumes}"
        )
    device = reconstructor.device
    h, w = reconstructor.box_shape
    n_or = result.num_orientations
    n_rows = result.num_fourier_filters * n_or

    vol = torch.as_tensor(np.asarray(volume), dtype=torch.float32)
    if vol.dim() != 3 or tuple(vol.shape) != (h, h, w):
        raise ValueError(
            f"volume must be (d, d, d) with (d, d) the template box {(h, w)}, got "
            f"{tuple(vol.shape)}"
        )

    k_stop = max(k0 + ke for k0, _, ke, _ in rects)
    e_stop = max(m0 + me for _, m0, _, me in rects)
    if k_stop > result.k_max or e_stop > result.eig_max:
        raise ValueError(
            f"rectangles reach ({k_stop}, {e_stop}) beyond the stored "
            f"({result.k_max}, {result.eig_max})"
        )

    # S * Vh over the kept cells, cropped to the rectangles' bounding box
    mask = torch.zeros((k_stop, e_stop))
    for k0, m0, ke, me in rects:
        mask[k0 : k0 + ke, m0 : m0 + me] = 1.0
    s = torch.as_tensor(np.asarray(result.S)[:k_stop, :e_stop]) * mask
    vh = torch.as_tensor(np.asarray(result.Vh)[:k_stop, :e_stop])
    weighted_vh = (s[..., None] * vh).to(device, torch.complex64)
    # U is (num_volumes = 1, FF, O, K, E): drop the volume axis, flatten (FF, O)
    u = torch.as_tensor(np.asarray(result.U)).reshape(n_rows, *result.U.shape[3:])
    u = u[:, :k_stop, :e_stop]

    support = reconstructor.support.to(device)
    row_filters = None
    if exact_filter is None:
        row_filters = _row_filters(result)
        if row_filters is not None:
            row_filters = row_filters.to(device)
    else:
        exact_filter = exact_filter.to(device)
    if signal_filter is not None:
        signal_filter = signal_filter.to(device)
    normalize = result.normalized_projections is not False

    dft, edge_value_scaled, pad_width = precompute_volume_dft(vol.to(device))
    phi = torch.as_tensor(result.phi_values, dtype=torch.float32, device=device)
    theta = torch.as_tensor(result.theta_values, dtype=torch.float32, device=device)

    for r0 in range(0, n_rows, batch):
        rows = slice(r0, min(r0 + batch, n_rows))
        index = torch.arange(rows.start, rows.stop, device=device)
        orientation = index % n_or

        polar = reconstruct_polar(
            u[rows].to(device, torch.complex64),
            weighted_vh,
            result.num_angular_components,
        )
        reconstruction = reconstructor.to_cartesian(polar)

        psi = torch.zeros_like(phi[orientation])
        projections = generate_projection_batch(
            dft,
            edge_value_scaled,
            pad_width,
            phi[orientation],
            theta[orientation],
            psi,
        )
        filt = exact_filter
        if row_filters is not None:
            filt = row_filters[index // n_or]
        exact, signal = exact_and_signal(
            projections, filt, signal_filter, support, normalize=normalize
        )
        yield _Rows(rows, reconstruction, exact, signal)


def _rfft_power(images: torch.Tensor, bins: RadialBins) -> torch.Tensor:
    """``(B, F)`` ring power of ``(B, h, w)`` images zero-padded to ``bins.shape``."""
    spectrum = torch.fft.rfft2(images.to(bins.w_left.dtype), s=bins.shape)
    return bins(spectrum.real.square() + spectrum.imag.square())


def error_profiles(
    reconstruction: torch.Tensor,
    exact: torch.Tensor,
    signal: torch.Tensor,
    bins: RadialBins,
) -> torch.Tensor:
    """``(5, B, F)`` ring profiles, in :data:`TABLE_NAMES` order."""
    dtype = bins.w_left.dtype
    fm = torch.fft.rfft2(reconstruction.to(dtype), s=bins.shape)
    fd = torch.fft.rfft2((exact - reconstruction).to(dtype), s=bins.shape)
    fp = torch.fft.rfft2(signal.to(dtype), s=bins.shape)

    def corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return bins(a.real * b.real + a.imag * b.imag)  # Re[conj(a) b]

    return torch.stack(
        [corr(fm, fm), corr(fd, fd), corr(fm, fd), corr(fp, fm), corr(fp, fd)]
    )


def _padded_bins(
    box: tuple[int, int], pad: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, RadialBins]:
    """The float32 CPU frequency axis of the ``pad``-times padded box, its bins."""
    padded = (pad * box[0], pad * box[1])
    frequencies = frequency_axis(calculate_num_freq_bins(padded))
    return frequencies, RadialBins.create(
        padded, frequencies, device=device, dtype=dtype
    )


# --------------------------------------------------------------------------- #
# Noise-model tables
# --------------------------------------------------------------------------- #


@torch.no_grad()
def compute_error_tables(
    volume: torch.Tensor | np.ndarray,
    reconstructor: Reconstructor,
    rectangles: Sequence[Rectangle],
    whitening: WhiteningFilter,
    *,
    pad: int = 2,
    batch: int = 128,
    dtype: torch.dtype = torch.float32,
) -> PerSpatialFrequencyErrorTable:
    """Error tables of every template row against the reference ``volume``.

    Parameters
    ----------
    volume : torch.Tensor or numpy.ndarray
        ``(d, d, d)`` reference volume; ``(d, d)`` is the template box.
    reconstructor : Reconstructor
        The decomposition (built under ``whitening.template_filter``; checked) and
        its warp, which the reconstructions and the support use and the table
        records.
    rectangles : sequence of Rectangle
        Kept ``(k_start, m_start, k_extent, m_extent)`` feature regions.
    whitening : WhiteningFilter
        The dataset's filter: ``W_avg * band`` for ``m`` and ``W_avg`` for ``p``.
    pad : int
        Zero-padding factor of the spectra. Default 2.
    batch : int
        Template rows per batch. Default 128.
    dtype : torch.dtype
        Dtype of the padded spectra and the binning. Default float32.
    """
    result = reconstructor.result
    if result.fourier_filters is None:
        raise ValueError(
            "the decomposition carries no Fourier filter; build it under "
            "whitening.template_filter(box)"
        )
    whitening.check_template_filter(result.fourier_filters)

    device = reconstructor.device
    box = reconstructor.box_shape
    rects = as_rectangles(rectangles)
    template_filter = whitening.template_filter(box)
    frequencies, bins = _padded_bins(box, pad, device, dtype)

    n_rows = result.num_fourier_filters * result.num_orientations
    tables = torch.empty((len(TABLE_NAMES), n_rows, bins.n_bins))
    for b in _template_rows(
        volume,
        reconstructor,
        rects,
        exact_filter=template_filter,
        signal_filter=whitening.average_filter(box, device),
        batch=batch,
    ):
        assert b.signal is not None
        tables[:, b.rows] = error_profiles(
            b.reconstruction, b.exact, b.signal, bins
        ).cpu()

    return PerSpatialFrequencyErrorTable(
        frequencies=frequencies,
        **dict(zip(TABLE_NAMES, tables.to(torch.float32), strict=True)),
        support=reconstructor.support.clone(),
        template_filter=template_filter.cpu(),
        rectangles=rects,
        warp=reconstructor.warp,
        pad=int(pad),
    )


# --------------------------------------------------------------------------- #
# Exact approximation errors
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class ApproximationErrors:
    """Exact errors of a decomposition's truncated templates, per row.

    ``m`` is the reference volume's projection under the decomposition's own recorded
    filter (normalized like the decomposition's projections, masked to the support),
    ``m~`` its reconstruction from the kept cells, warped to Cartesian, and
    ``delta = m - m~``. Every reduction pools energies over the rows it combines,
    ``sqrt(sum |delta|^2 / sum |m|^2)``.

    Parameters
    ----------
    frequencies : torch.Tensor
        ``(F,)`` float32 frequency axis (cycles per pixel).
    template_spectrum : torch.Tensor
        ``(N, F)`` float64 ring power of ``m``; ``N`` the template rows.
    residual_spectrum : torch.Tensor
        ``(N, F)`` float64 ring power of ``delta``.
    template_radial : torch.Tensor
        ``(N, R)`` float64 energy of ``m`` in each one-pixel annulus about the box
        center (annulus ``r`` holds the pixels at distance ``[r, r + 1)``).
    residual_radial : torch.Tensor
        ``(N, R)`` float64 energy of ``delta`` per annulus.
    rectangles : tuple of Rectangle
        The kept ``(k_start, m_start, k_extent, m_extent)`` regions.
    warp : WarpConfig
        The warp of the reconstructions.
    """

    frequencies: torch.Tensor
    template_spectrum: torch.Tensor
    residual_spectrum: torch.Tensor
    template_radial: torch.Tensor
    residual_radial: torch.Tensor
    rectangles: tuple[Rectangle, ...]
    warp: WarpConfig

    @property
    def radii(self) -> torch.Tensor:
        """``(R,)`` inner radius (pixels) of each annulus."""
        return torch.arange(self.template_radial.shape[1], dtype=torch.float32)

    def by_frequency(self) -> torch.Tensor:
        """``(F,)`` relative error of each frequency ring over every row."""
        return _ratio(self.residual_spectrum.sum(0), self.template_spectrum.sum(0))

    def by_radius(self) -> torch.Tensor:
        """``(R,)`` relative error of each annulus over every row.

        NaN for the annuli outside the support, where ``m`` is zero; what the
        reconstructions leak there still counts in :meth:`by_orientation` and
        :meth:`total`.
        """
        return _ratio(self.residual_radial.sum(0), self.template_radial.sum(0))

    def by_orientation(self) -> torch.Tensor:
        """``(N,)`` relative error of each template row (whole box)."""
        return _ratio(self.residual_radial.sum(1), self.template_radial.sum(1))

    def total(self) -> float:
        """Relative error of every row together (whole box)."""
        return float(_ratio(self.residual_radial.sum(), self.template_radial.sum()))


def _ratio(residual: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """``sqrt(residual / template)``, NaN where the template has no energy."""
    return torch.where(template > 0, torch.sqrt(residual / template), torch.nan).float()


def _annuli(box: tuple[int, int], device: torch.device) -> tuple[torch.Tensor, int]:
    """Flat annulus index of every pixel of ``box`` about its center, and the count."""
    h, w = box
    y = torch.arange(h, device=device, dtype=torch.float64) - h // 2
    x = torch.arange(w, device=device, dtype=torch.float64) - w // 2
    r = torch.sqrt(y[:, None] ** 2 + x[None, :] ** 2).floor().long().flatten()
    return r, int(r.max()) + 1


@torch.no_grad()
def compute_errors(
    volume: torch.Tensor | np.ndarray,
    reconstructor: Reconstructor,
    rectangles: Sequence[Rectangle],
    *,
    pad: int = 2,
    batch: int = 128,
) -> ApproximationErrors:
    """Exact errors of ``rectangles`` against the reference ``volume``'s projections.

    Unlike :func:`compute_error_tables`, which the noise model reads, this needs no
    whitening filter: ``m`` is filtered as the decomposition recorded (by its own
    Fourier filters, or not at all), so an unwhitened decomposition gives unwhitened
    errors.

    Parameters
    ----------
    volume : torch.Tensor or numpy.ndarray
        ``(d, d, d)`` reference volume; ``(d, d)`` is the template box.
    reconstructor : Reconstructor
        The decomposition and its warp.
    rectangles : sequence of Rectangle
        Kept ``(k_start, m_start, k_extent, m_extent)`` feature regions.
    pad : int
        Zero-padding factor of the spectra. Default 2.
    batch : int
        Template rows per batch. Default 128.
    """
    device = reconstructor.device
    box = reconstructor.box_shape
    rects = as_rectangles(rectangles)
    frequencies, bins = _padded_bins(box, pad, device, torch.float64)
    annulus, n_annuli = _annuli(box, device)

    def radial(images: torch.Tensor) -> torch.Tensor:
        flat = images.to(torch.float64).square().flatten(1)
        out = flat.new_zeros((flat.shape[0], n_annuli))
        return out.index_add_(1, annulus, flat)

    parts: dict[str, list[torch.Tensor]] = {
        "template_spectrum": [],
        "residual_spectrum": [],
        "template_radial": [],
        "residual_radial": [],
    }
    for b in _template_rows(volume, reconstructor, rects, batch=batch):
        residual = b.exact - b.reconstruction
        parts["template_spectrum"].append(_rfft_power(b.exact, bins).cpu())
        parts["residual_spectrum"].append(_rfft_power(residual, bins).cpu())
        parts["template_radial"].append(radial(b.exact).cpu())
        parts["residual_radial"].append(radial(residual).cpu())

    return ApproximationErrors(
        frequencies=frequencies,
        **{name: torch.cat(chunks) for name, chunks in parts.items()},
        rectangles=rects,
        warp=reconstructor.warp,
    )
