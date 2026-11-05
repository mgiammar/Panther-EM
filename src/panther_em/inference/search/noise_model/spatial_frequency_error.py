"""Per-spatial-frequency reconstruction error tables of an SVD template.

- reconstruction ``m~``: the SVD template restricted to the feature rectangles,
- exact template ``m``: the volume's projection, filtered, normalized and masked,
- residual ``delta = m - m~``,
- signal ``p``: the same projection under the signal filter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import h5py  # type: ignore[import-untyped]
import torch
from torch_fourier_filter.whitening import calculate_num_freq_bins
from torch_grid_utils.fftfreq_grid import fftfreq_grid  # type: ignore[import-untyped]

from panther_em.decomposition.pipeline_projections import (
    generate_projection_batch,
    precompute_volume_dft,
)
from panther_em.spectra import frequency_axis

if TYPE_CHECKING:
    from panther_em.coordinates.transform_base import CoordinateTransform
    from panther_em.inference.search.template import Template
    from panther_em.inference.search.types import KernelConfig, Rectangle

TABLE_NAMES = (
    "reconstruction_autocorr",
    "residual_autocorr",
    "reconstruction_residual_corr",
    "signal_reconstruction_corr",
    "signal_residual_corr",
)


@dataclass(frozen=True)
class PerSpatialFrequencyErrorTable:
    """Helper class for spatial frequency profiles of error and overlap.

    Parameters
    ----------
    frequencies : torch.Tensor
        Shape ``(F,)`` float32 tensor defining frequency axis (cycles/pixel).
    reconstruction_autocorr : torch.Tensor
        Shape ``(N, F)`` float32 tensor, where ``N`` is the number of orientations, for
        the autocorrelation (power) of the reconstruction at each frequency.
    residual_autocorr : torch.Tensor
        Shape ``(N, F)`` float32 tensor, where ``N`` is the number of orientations, for
        the autocorrelation (power) of the residual at each frequency.
    reconstruction_residual_corr : torch.Tensor
        Shape ``(N, F)`` float32 tensor for the cross-correlation of the reconstruction
        and residual at each frequency.
    signal_reconstruction_corr : torch.Tensor
        Shape ``(N, F)`` float32 tensor for the cross-correlation of the signal and
        reconstruction at each frequency.
    signal_residual_corr : torch.Tensor
        Shape ``(N, F)`` float32 tensor for the cross-correlation of the signal and
        residual at each frequency.
    rectangles : tuple of Rectangle
        The rectangular feature regions used for reconstructions and residuals.
    pad : int
        Zero-padding factor of the spectra the tables were binned from. By default 2.

    """

    frequencies: torch.Tensor
    reconstruction_autocorr: torch.Tensor
    residual_autocorr: torch.Tensor
    reconstruction_residual_corr: torch.Tensor
    signal_reconstruction_corr: torch.Tensor
    signal_residual_corr: torch.Tensor
    rectangles: tuple[Rectangle, ...]
    pad: int = 2

    def save(self, path: str) -> None:
        """Write to HDF5."""
        with h5py.File(path, "w") as f:
            f["frequencies"] = self.frequencies.numpy()
            for name in TABLE_NAMES:
                f[name] = getattr(self, name).numpy()
            f.attrs["rectangles"] = json.dumps(self.rectangles)
            f.attrs["pad"] = self.pad

    @classmethod
    def load(cls, path: str) -> PerSpatialFrequencyErrorTable:
        """Read a file written by :meth:`save`."""
        with h5py.File(path, "r") as f:
            return cls(
                frequencies=torch.from_numpy(f["frequencies"][()]),
                **{name: torch.from_numpy(f[name][()]) for name in TABLE_NAMES},
                rectangles=tuple(tuple(r) for r in json.loads(f.attrs["rectangles"])),
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


# --------------------------------------------------------------------------- #
# Batch kernels
# --------------------------------------------------------------------------- #


def reconstruct_polar(
    u: torch.Tensor, weighted_vh: torch.Tensor, n_angle: int
) -> torch.Tensor:
    """``(B, K, E)`` x ``(K, E, R)`` -> ``(B, n_angle, R)`` float32 polar images."""
    blocks = torch.einsum("bke,ker->bkr", u, weighted_vh)
    return torch.fft.irfft(blocks, n=n_angle, dim=1, norm="ortho").to(torch.float32)


def warp_to_cartesian(
    polar: torch.Tensor, transform: CoordinateTransform, config: KernelConfig
) -> torch.Tensor:
    """``(B, n_angle, R)`` polar -> ``(B, kH, kW)`` float32 Cartesian images."""
    cart = transform.to_cartesian(
        polar,
        preserve_energy=config.preserve_energy,
        wrap_angular_axis=config.wrap_angular_axis,
        order=config.order,
        mode=config.mode,
        cval=config.cval,
    )
    return torch.as_tensor(cart, device=polar.device).to(torch.float32)


def exact_and_signal(
    projections: torch.Tensor,
    exact_filters: torch.Tensor,
    signal_filter: torch.Tensor,
    support: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact templates ``m`` (filtered, normalized, masked) and signals ``p``."""
    size = projections.shape[-2:]
    spectrum = torch.fft.rfft2(projections)

    exact = torch.fft.irfft2(spectrum * exact_filters, s=size)
    var, mean = torch.var_mean(exact, dim=(-2, -1), keepdim=True)
    exact = (exact - mean) / torch.sqrt(var + 1e-8) * support

    signal = torch.fft.irfft2(spectrum * signal_filter, s=size)
    return exact, signal


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


# --------------------------------------------------------------------------- #
# Table construction
# --------------------------------------------------------------------------- #


@torch.no_grad()
def compute_error_tables(
    volume: torch.Tensor,
    template: Template,
    rectangles: tuple[Rectangle, ...],
    signal_filter: torch.Tensor | None = None,
    pad: int = 2,
    batch: int = 128,
    dtype: torch.dtype = torch.float32,
) -> PerSpatialFrequencyErrorTable:
    """Error tables of every template row against the reference ``volume``.

    Parameters
    ----------
    volume : torch.Tensor
        ``(d, d, d)`` reference volume; ``(d, d)`` is the template box.
    template : Template
        The SVD decomposition, its coordinate transform and warp settings.
    rectangles : tuple of Rectangle
        Kept ``(k_start, m_start, k_extent, m_extent)`` feature regions.
    signal_filter : torch.Tensor, optional
        ``(d, d // 2 + 1)`` rfft-grid filter giving the signal ``p``. Defaults to the
        template's first Fourier filter.
    pad : int
        Zero-padding factor of the spectra. Default 2.
    batch : int
        Template rows per batch. Default 128.
    dtype : torch.dtype
        Dtype of the padded spectra and the binning. Default float32.
    """
    result = template.result.to_dense()
    device = template.device
    transform = template.reconstructor.coordinate_transform
    config = template.kernel_config
    h, w = template.template_shape
    n_orient = result.num_orientations
    n_rows = result.num_fourier_filters * n_orient

    # S * Vh over the kept cells, cropped to the rectangles' bounding box
    k_stop = max(k0 + ke for k0, _, ke, _ in rectangles)
    e_stop = max(m0 + me for _, m0, _, me in rectangles)
    mask = torch.zeros((k_stop, e_stop))
    for k0, m0, ke, me in rectangles:
        mask[k0 : k0 + ke, m0 : m0 + me] = 1.0
    s = torch.as_tensor(result.S[:k_stop, :e_stop]) * mask
    vh = torch.as_tensor(result.Vh[:k_stop, :e_stop])
    weighted_vh = (s[..., None] * vh).to(device, torch.complex64)
    u = torch.from_numpy(result.U.reshape(n_rows, *result.U.shape[2:]))
    u = u[:, :k_stop, :e_stop]

    # the template's polar support: where a warped all-ones polar image exceeds 1/2
    support = transform.to_cartesian(
        torch.ones(transform.polar_shape),
        preserve_energy=False,
        wrap_angular_axis=config.wrap_angular_axis,
        order=config.order,
        mode=config.mode,
        cval=config.cval,
    )
    support = torch.as_tensor(support, device=device) > 0.5

    # volume projections and their filters
    dft, mean_scaled, pad_width = precompute_volume_dft(
        torch.as_tensor(volume, dtype=torch.float32, device=device)
    )
    phi = torch.as_tensor(result.phi_values, dtype=torch.float32, device=device)
    theta = torch.as_tensor(result.theta_values, dtype=torch.float32, device=device)
    if result.fourier_filters is None:
        exact_filters = torch.ones((1, h, w // 2 + 1), device=device)
    else:
        exact_filters = torch.as_tensor(result.fourier_filters, device=device)
    if signal_filter is None:
        signal_filter = exact_filters[0]
    signal_filter = signal_filter.to(device)

    padded = (pad * h, pad * w)
    frequencies = frequency_axis(calculate_num_freq_bins(padded))
    bins = RadialBins.create(padded, frequencies, device=device, dtype=dtype)

    tables = torch.empty((len(TABLE_NAMES), n_rows, bins.n_bins))
    for r0 in range(0, n_rows, batch):
        rows = torch.arange(r0, min(r0 + batch, n_rows), device=device)
        f_idx, o_idx = rows // n_orient, rows % n_orient

        polar = reconstruct_polar(
            u[r0 : r0 + batch].to(device, torch.complex64),
            weighted_vh,
            result.num_angular_components,
        )
        reconstruction = warp_to_cartesian(polar, transform, config)

        psi = torch.zeros_like(phi[o_idx])
        projections = generate_projection_batch(
            dft, mean_scaled, pad_width, phi[o_idx], theta[o_idx], psi
        )
        filters = exact_filters[f_idx % exact_filters.shape[0]]
        exact, signal = exact_and_signal(projections, filters, signal_filter, support)

        tables[:, r0 : r0 + batch] = error_profiles(
            reconstruction, exact, signal, bins
        ).cpu()

    return PerSpatialFrequencyErrorTable(
        frequencies,
        *tables,
        rectangles=tuple(tuple(r) for r in rectangles),
        pad=pad,
    )
