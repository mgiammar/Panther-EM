"""Per-spatial-frequency reconstruction error tables of an SVD template.

- reconstruction ``m~``: the SVD template restricted to the feature rectangles,
- exact template ``m``: the volume's projection under the decomposition's filter
  ``W_avg * band``, normalized and masked to the template support,
- residual ``delta = m - m~``,
- signal ``p``: the same projection under ``W_avg`` alone.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import h5py  # type: ignore[import-untyped]
import numpy as np
import torch
from torch_fourier_filter.whitening import calculate_num_freq_bins
from torch_grid_utils.fftfreq_grid import fftfreq_grid  # type: ignore[import-untyped]

from panther_em.decomposition.pipeline_projections import (
    generate_projection_batch,
    precompute_volume_dft,
)
from panther_em.inference.search.noise import NoiseScales
from panther_em.inference.search.noise_model.whitening import (
    FILTER_ATOL,
    FILTER_RTOL,
    canonical_device,
)
from panther_em.inference.template import KernelConfig
from panther_em.spectra import frequency_axis

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from panther_em.inference.search.hypotheses import HypothesisSet
    from panther_em.inference.search.noise_model.image_set import ImageSet
    from panther_em.inference.search.noise_model.whitening import WhiteningFilter
    from panther_em.inference.search.types import Rectangle
    from panther_em.inference.template import Template

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
    kernel_config : KernelConfig
        The warp the reconstructions and the support disk were built with; a search
        must use a template with the same (:meth:`check_template`).
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
    kernel_config: KernelConfig
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

        if not isinstance(self.kernel_config, KernelConfig):
            raise ValueError("kernel_config must be a KernelConfig")

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

    # ------ template guard ----------------------------------------------------

    def check_template(self, template: Template) -> None:
        """Raise unless the tables describe ``template``'s kernels."""
        if template.kernel_config != self.kernel_config:
            raise ValueError(
                f"the tables were built with the kernel warp {self.kernel_config}, the "
                f"template warps its kernels with {template.kernel_config}"
            )
        if tuple(template.template_shape) != self.template_shape:
            raise ValueError(
                f"the tables were built on a {self.template_shape} box, the template's "
                f"is {tuple(template.template_shape)}"
            )
        if template.n_templates != self.n_templates:
            raise ValueError(
                f"the tables hold {self.n_templates} template rows, the template "
                f"{template.n_templates}"
            )
        for k0, m0, ke, me in self.rectangles:
            if k0 + ke > template.k_max or m0 + me > template.eig_max:
                raise ValueError(
                    f"rectangle {(k0, m0, ke, me)} lies beyond the template's stored "
                    f"(k_max={template.k_max}, eig_max={template.eig_max})"
                )
        filters = template.result.fourier_filters
        if filters is None:
            raise ValueError("the template's decomposition carries no Fourier filter")
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
                "the template's decomposition was built under another Fourier filter "
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

    # -- scales ----------------------------------------------------------------

    def scales(
        self,
        images: ImageSet,
        *,
        hyps: HypothesisSet | None = None,
        b_factor: float = 0.0,
    ) -> NoiseScales:
        """The micrograph's noise and signal scales of every ``(plane, row)``.

        Parameters
        ----------
        images : ImageSet
            The micrograph; its curves must be on this table's frequency axis.
        hyps : HypothesisSet, optional
            The stage's hypotheses; ``qbar_perp`` averages ``q_perp`` over their
            template rows and planes (all of them when ``None``).
        b_factor : float, optional
            Envelope ``exp(-B s^2 / 4)`` (``s = f / pixel_size`` in 1/A) on the signal
            weight. Defaults to 0.

        Returns
        -------
        NoiseScales
            ``(1, D, N)`` scales (one micrograph).
        """
        if not torch.equal(images.frequencies, self.frequencies):
            raise ValueError(
                "the image set's curves are on another frequency axis than the table "
                "(prepare the micrograph with this table or one of the same box/pad)"
            )

        freq = self.frequencies.to(torch.float64)
        envelope = torch.exp(-b_factor * (freq / images.pixel_size) ** 2 / 4.0)
        noise = images.noise  # (D, F)
        weight = images.signal * envelope
        t = {name: getattr(self, name).to(torch.float64).T for name in TABLE_NAMES}

        q_kept = noise @ t["reconstruction_autocorr"]  # (D, N)
        q_residual = noise @ t["residual_autocorr"]
        q_cross = noise @ t["reconstruction_residual_corr"]
        lam = torch.where(q_kept > 0, q_cross / q_kept.clamp_min(1e-300), 0.0)
        q_perp = (q_residual - lam**2 * q_kept).clamp_min(0.0)
        g_kept = weight @ t["signal_reconstruction_corr"]
        g_residual = weight @ t["signal_residual_corr"]
        total = g_kept + g_residual
        rho = torch.where(total > 0, g_residual / total.clamp_min(1e-300), 0.0)
        rho = rho.clamp(0.0, 1.0 - 1e-6)

        if hyps is None:
            rows = torch.arange(self.n_templates)
            planes = list(range(images.n_defocus))
        else:
            rows = hyps.template_rows()
            planes = list(hyps.defocus_planes())
        qbar_perp = q_perp[planes][:, rows].mean().reshape(1)

        return NoiseScales(
            q_kept[None],
            q_residual[None],
            q_cross[None],
            lam[None],
            q_perp[None],
            rho[None],
            qbar_perp,
            self.rectangles,
            float(b_factor),
        )

    # -- I/O -------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write to HDF5."""
        with h5py.File(path, "w") as f:
            f["frequencies"] = self.frequencies.numpy()
            for name in TABLE_NAMES:
                f[name] = getattr(self, name).numpy()
            f["support"] = self.support.numpy()
            f["template_filter"] = self.template_filter.numpy()
            f.attrs["rectangles"] = json.dumps(self.rectangles)
            f.attrs["kernel_config"] = json.dumps(
                dataclasses.asdict(self.kernel_config)
            )
            f.attrs["pad"] = self.pad

    @classmethod
    def load(cls, path: str | Path) -> PerSpatialFrequencyErrorTable:
        """Read a file written by :meth:`save`."""
        with h5py.File(path, "r") as f:
            return cls(
                frequencies=torch.from_numpy(f["frequencies"][()]),
                **{name: torch.from_numpy(f[name][()]) for name in TABLE_NAMES},
                support=torch.from_numpy(np.asarray(f["support"][()], dtype=bool)),
                template_filter=torch.from_numpy(f["template_filter"][()]),
                rectangles=as_rectangles(json.loads(f.attrs["rectangles"])),
                kernel_config=KernelConfig(**json.loads(f.attrs["kernel_config"])),
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
# Batch kernels
# --------------------------------------------------------------------------- #


def as_rectangles(rectangles: Sequence[Sequence[int]]) -> tuple[Rectangle, ...]:
    """``(k_start, m_start, k_extent, m_extent)`` int tuples of a rectangle sequence."""
    rects: list[Rectangle] = []
    for r in rectangles:
        if len(r) != 4:
            raise ValueError(f"a rectangle has four entries, got {r!r}")
        rects.append((int(r[0]), int(r[1]), int(r[2]), int(r[3])))
    return tuple(rects)


def reconstruct_polar(
    u: torch.Tensor, weighted_vh: torch.Tensor, n_angle: int
) -> torch.Tensor:
    """``(B, K, E)`` x ``(K, E, R)`` -> ``(B, n_angle, R)`` float32 polar images."""
    blocks = torch.einsum("bke,ker->bkr", u, weighted_vh)
    polar: torch.Tensor = torch.fft.irfft(blocks, n=n_angle, dim=1, norm="ortho")
    return polar.to(torch.float32)


def exact_and_signal(
    projections: torch.Tensor,
    exact_filter: torch.Tensor,
    signal_filter: torch.Tensor,
    support: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact templates ``m`` (filtered, normalized, masked) and signals ``p``."""
    size = projections.shape[-2:]
    spectrum = torch.fft.rfft2(projections)

    exact = torch.fft.irfft2(spectrum * exact_filter, s=size)
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
    volume: torch.Tensor | np.ndarray,
    template: Template,
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
    template : Template
        The SVD decomposition (built under ``whitening.template_filter``; checked)
        and its kernel warp, which the reconstructions and the support use and the
        table records.
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
    result = template.result.to_dense()
    if result.fourier_filters is None:
        raise ValueError(
            "the decomposition carries no Fourier filter; build it under "
            "whitening.template_filter(box)"
        )
    whitening.check_template_filter(result.fourier_filters)

    device = template.device
    warp = template.warp
    h, w = template.template_shape
    n_rows = result.num_fourier_filters * result.num_orientations  # one filter

    vol = torch.as_tensor(np.asarray(volume), dtype=torch.float32)
    if vol.dim() != 3 or tuple(vol.shape) != (h, h, w):
        raise ValueError(
            f"volume must be (d, d, d) with (d, d) the template box {(h, w)}, got "
            f"{tuple(vol.shape)}"
        )

    rects = as_rectangles(rectangles)
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
    u = torch.as_tensor(np.asarray(result.U)).reshape(n_rows, *result.U.shape[2:])
    u = u[:, :k_stop, :e_stop]

    support = warp.support.clone()

    # volume projections, the exact template's filter and the signal's
    dft, mean_scaled, pad_width = precompute_volume_dft(vol.to(device))
    phi = torch.as_tensor(result.phi_values, dtype=torch.float32, device=device)
    theta = torch.as_tensor(result.theta_values, dtype=torch.float32, device=device)
    template_filter = whitening.template_filter((h, w))
    exact_filter = template_filter.to(device)
    signal_filter = whitening.average_filter((h, w), device)

    padded = (pad * h, pad * w)
    frequencies = frequency_axis(calculate_num_freq_bins(padded))
    bins = RadialBins.create(padded, frequencies, device=device, dtype=dtype)

    tables = torch.empty((len(TABLE_NAMES), n_rows, bins.n_bins))
    for r0 in range(0, n_rows, batch):
        rows = torch.arange(r0, min(r0 + batch, n_rows), device=device)

        polar = reconstruct_polar(
            u[r0 : r0 + batch].to(device, torch.complex64),
            weighted_vh,
            result.num_angular_components,
        )
        reconstruction = warp.to_cartesian(polar)

        psi = torch.zeros_like(phi[rows])
        projections = generate_projection_batch(
            dft, mean_scaled, pad_width, phi[rows], theta[rows], psi
        )
        exact, signal = exact_and_signal(
            projections, exact_filter, signal_filter, support.to(device)
        )

        tables[:, r0 : r0 + batch] = error_profiles(
            reconstruction, exact, signal, bins
        ).cpu()

    return PerSpatialFrequencyErrorTable(
        frequencies=frequencies,
        **dict(zip(TABLE_NAMES, tables.to(torch.float32), strict=True)),
        support=support,
        template_filter=template_filter.cpu(),
        rectangles=rects,
        kernel_config=template.kernel_config,
        pad=int(pad),
    )
