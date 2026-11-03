"""Colored-noise error model for threshold exceedence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import h5py  # type: ignore[import-untyped]
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import ndtri  # type: ignore[import-untyped]
from torch_fourier_filter.bandpass import low_pass_filter
from torch_fourier_filter.utils import bin_1dim_with_lerp, curve_1dim_to_ndim
from torch_fourier_filter.whitening import (
    calculate_num_freq_bins,
    power_spectral_density,
)
from torch_grid_utils.fftfreq_grid import fftfreq_grid  # type: ignore[import-untyped]
from tqdm import tqdm

from panther_em.decomposition.pipeline_projections import (
    apply_fourier_filters,
    generate_projection_batch,
    precompute_volume_dft,
)
from panther_em.inference.search.features import as_rectangle, rectangles_disjoint
from panther_em.inference.search.types import (
    ImageSet,
    ImageSpectra,
    KernelConfig,
    Rectangle,
)
from panther_em.spectra import WhiteningFilter, frequency_axis

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from panther_em.coordinates.transform_base import CoordinateTransform
    from panther_em.inference.search.hypotheses import HypothesisSet
    from panther_em.inference.search.template import Template

TABLE_NAMES = ("kept", "residual", "cross", "kept_signal", "residual_signal")
SCALE_NAMES = ("q_kept", "q_residual", "q_cross", "lam", "q_perp", "rho")
CHI2_1_MEDIAN = 0.4549364231195728  # median of chi^2 with one degree of freedom


# --------------------------------------------------------------------------- #
# Ring profiles and curves
# --------------------------------------------------------------------------- #


def _resample(
    freq_in: torch.Tensor, values: torch.Tensor, freq_out: torch.Tensor
) -> torch.Tensor:
    """``(..., F_in)`` curves linearly interpolated onto ``freq_out`` (float64)."""
    flat = values.reshape(-1, values.shape[-1]).to(torch.float64).cpu()
    out = curve_1dim_to_ndim(
        flat, freq_in.to(torch.float64).cpu(), freq_out.to(torch.float64).cpu()
    )
    return out.to(torch.float64).reshape(*values.shape[:-1], -1)


def _hermitian_weights(width: int, device: torch.device | str) -> torch.Tensor:
    """``(width // 2 + 1,)`` multiplicity of each rfft column in the full spectrum."""
    weight = torch.full((width // 2 + 1,), 2.0, dtype=torch.float64, device=device)
    weight[0] = 1.0
    if width % 2 == 0:
        weight[-1] = 1.0
    return weight


def ring_sums(
    spectra: torch.Tensor, freq: torch.Tensor, shape: tuple[int, int]
) -> torch.Tensor:
    """Hermitian-weighted ring sums of rfft-grid quantities, divided by ``h * w``.

    ``ring_sums(|a^|^2).sum(-1)`` is ``||a||^2`` and
    ``ring_sums(Re[conj(a^) b^]).sum(-1)`` is ``<a, b>`` (Parseval).

    Parameters
    ----------
    spectra : torch.Tensor
        ``(n, h, w // 2 + 1)`` real grids.
    freq : torch.Tensor
        ``(F,)`` bin centres (an evenly spaced axis reaching the grid corner).
    shape : tuple[int, int]
        Real-space ``(h, w)`` of the grids.

    Returns
    -------
    torch.Tensor
        ``(n, F)`` float64 profiles.
    """
    h, w = int(shape[0]), int(shape[1])
    if spectra.is_complex() or tuple(spectra.shape[-2:]) != (h, w // 2 + 1):
        raise ValueError(f"spectra must be real (n, {h}, {w // 2 + 1}) grids")

    radius = fftfreq_grid(
        image_shape=(h, w),
        rfft=True,
        fftshift=False,
        norm=True,
        device=spectra.device,
    )
    radius = radius.to(torch.float64)
    weight = _hermitian_weights(w, spectra.device).expand(h, -1)
    axis = freq.to(torch.float64).to(spectra.device)
    rows = [
        bin_1dim_with_lerp(radius, s.to(torch.float64) * weight, axis, False)
        for s in spectra.reshape(-1, h, w // 2 + 1)
    ]
    return torch.stack(rows).reshape(*spectra.shape[:-2], -1) / float(h * w)


def support_mask(transform: CoordinateTransform, config: KernelConfig) -> torch.Tensor:
    """``(kH, kW)`` bool for disc mask around template."""
    ones = torch.ones(transform.polar_shape, dtype=torch.float32)
    warped = transform.to_cartesian(
        ones,
        preserve_energy=False,
        wrap_angular_axis=config.wrap_angular_axis,
        order=config.order,
        mode=config.mode,
        cval=config.cval,
    )
    return (
        torch.as_tensor(
            np.asarray(warped) if isinstance(warped, np.ndarray) else warped
        ).cpu()
        > 0.5
    )


# --------------------------------------------------------------------------- #
# Template sources
# --------------------------------------------------------------------------- #


def _rectangles(rectangles: Rectangle | Sequence[Rectangle]) -> tuple[Rectangle, ...]:
    if len(rectangles) > 0 and isinstance(rectangles[0], int):
        rects: tuple[Rectangle, ...] = (as_rectangle(rectangles),)  # type: ignore[arg-type]
    else:
        rects = tuple(as_rectangle(r) for r in rectangles)  # type: ignore[arg-type]
    if not rects:
        raise ValueError("at least one rectangle is needed")
    if not rectangles_disjoint(rects):
        raise ValueError("rectangles must be pairwise cell-disjoint")
    return rects


def _selection_mask(
    rectangles: Sequence[Rectangle], k_max: int, eig_max: int
) -> torch.Tensor:
    """``(K, E)`` bool: the cells of the rectangles (which must fit the stored grid)."""
    mask = torch.zeros((k_max, eig_max), dtype=torch.bool)
    for k0, m0, ke, me in rectangles:
        if k0 + ke > k_max or m0 + me > eig_max:
            raise ValueError(
                f"rectangle {(k0, m0, ke, me)} exceeds the stored (k_max={k_max}, "
                f"eig_max={eig_max}) grid"
            )
        mask[k0 : k0 + ke, m0 : m0 + me] = True
    return mask


def _rows_to_filter_orientation(
    rows: torch.Tensor, n_orientations: int
) -> tuple[np.ndarray, np.ndarray]:
    r = rows.detach().cpu().to(torch.long).numpy()
    return r // n_orientations, r % n_orientations


class SvdTemplates:
    """Kept and full reconstructions of template rows at ``psi = 0``.

    Parameters
    ----------
    template : Template
        Supplies the decomposition, the transform and the warp settings.
    rectangles : Rectangle or sequence of Rectangle
        The feature selection (pairwise cell-disjoint).
    """

    def __init__(self, template: Template, rectangles: Rectangle | Sequence[Rectangle]):
        result = template.result.to_dense()
        if result.is_complex_projection:
            raise NotImplementedError("Complex-valued projections not yet supported")
        self.result = result
        self.device = template.device
        self.transform = template.reconstructor.coordinate_transform
        self.config = template.kernel_config
        self.rectangles = _rectangles(rectangles)
        self.n_orientations = int(result.num_orientations)
        self.n_templates = int(result.num_fourier_filters) * self.n_orientations
        self.template_shape: tuple[int, int] = template.template_shape
        self.n_angle = int(result.num_angular_components)
        mask = _selection_mask(self.rectangles, int(result.k_max), int(result.eig_max))
        s = torch.as_tensor(np.asarray(result.S), dtype=torch.float32)
        self.kept_weights = (s * mask).to(torch.complex64).to(self.device)
        self.full_weights = s.to(torch.complex64).to(self.device)
        self.vh = (
            torch.as_tensor(np.asarray(result.Vh)).to(torch.complex64).to(self.device)
        )
        self.support = support_mask(self.transform, self.config).to(self.device)

    def _reconstruct(self, u: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        blocks = torch.einsum("bke,ke,ker->bkr", u, weights, self.vh)
        polar = torch.fft.irfft(blocks, n=self.n_angle, dim=1, norm="ortho")
        cfg = self.config
        cart = self.transform.to_cartesian(
            polar.to(torch.float32),
            preserve_energy=cfg.preserve_energy,
            wrap_angular_axis=cfg.wrap_angular_axis,
            order=cfg.order,
            mode=cfg.mode,
            cval=cfg.cval,
        )
        return torch.as_tensor(cart, device=self.device).to(torch.float32)

    def __call__(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(kept, full)`` templates ``(len(rows), kH, kW)`` float32 on the device."""
        f_idx, o_idx = _rows_to_filter_orientation(rows, self.n_orientations)
        u = torch.as_tensor(np.asarray(self.result.U)[f_idx, o_idx]).to(torch.complex64)
        u = u.to(self.device)
        return self._reconstruct(u, self.kept_weights), self._reconstruct(
            u, self.full_weights
        )


class ProjectedTemplates:
    """Exact templates and whitened projections of template rows at ``psi = 0``.

    The exact template ``m`` is the projection of the volume at the row's orientation,
    filtered by the decomposition's Fourier filter of the row's filter index, normalized
    like the decomposition's input (``normalize``) and restricted to the polar support
    (``mask_exact``). The signal projection ``p`` is the raw projection under ``W_avg``
    (neither normalized nor masked), the particle the note expects in a micrograph;
    without a whitening filter ``p`` is ``m`` itself.

    Parameters
    ----------
    template : Template
        Supplies the orientations, the Fourier filters and the transform.
    volume : torch.Tensor or numpy.ndarray
        The ``(d, d, d)`` structure the template was decomposed from.
    whitening : WhiteningFilter or torch.Tensor, optional
        ``W_avg``, or an rfft-grid filter of the template box.
    normalize : bool, optional
        Whether the decomposition normalised its projections (default True).
    mask_exact : bool, optional
        Restrict ``m`` to the polar support (default True).
    """

    def __init__(
        self,
        template: Template,
        volume: torch.Tensor | np.ndarray,
        *,
        whitening: WhiteningFilter | torch.Tensor | None = None,
        normalize: bool = True,
        mask_exact: bool = True,
    ):
        result = template.result.to_dense()

        if result.phi_values is None or result.theta_values is None:
            raise ValueError("the decomposition records no orientations to project")

        self.device = template.device
        self.n_orientations = int(result.num_orientations)
        self.n_templates = int(result.num_fourier_filters) * self.n_orientations
        self.template_shape: tuple[int, int] = template.template_shape
        self.normalize = bool(normalize)
        self.mask_exact = bool(mask_exact)

        vol = torch.as_tensor(np.asarray(volume), dtype=torch.float32).to(self.device)

        if (
            vol.dim() != 3
            or len(set(vol.shape)) != 1
            or tuple(vol.shape[1:]) != (self.template_shape)
        ):
            raise ValueError(
                f"volume must be (d, d, d) with d x d = the template box "
                f"{self.template_shape}, got {tuple(vol.shape)}"
            )

        self.dft, self.mean_scaled, self.pad_width = precompute_volume_dft(vol)
        self.phi = torch.as_tensor(np.asarray(result.phi_values), dtype=torch.float32)
        self.theta = torch.as_tensor(
            np.asarray(result.theta_values), dtype=torch.float32
        )
        self.phi, self.theta = self.phi.to(self.device), self.theta.to(self.device)
        self.filters = (
            None
            if result.fourier_filters is None
            else torch.as_tensor(np.asarray(result.fourier_filters)).to(self.device)
        )

        if whitening is None:
            self.w_avg: torch.Tensor | None = None
        elif isinstance(whitening, WhiteningFilter):
            self.w_avg = whitening.average_grid(self.template_shape, device=self.device)
        else:
            self.w_avg = torch.as_tensor(whitening).to(self.device)

        self.support = support_mask(
            template.reconstructor.coordinate_transform, template.kernel_config
        ).to(self.device)

    @property
    def signal_source(self) -> Literal["projection", "template"]:
        """Whether ``p`` is the whitened projection or the exact template itself."""
        return "projection" if self.w_avg is not None else "template"

    def __call__(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(exact, signal)`` templates, ``(len(rows), kH, kW)`` float32 on device."""
        f_idx, o_idx = _rows_to_filter_orientation(rows, self.n_orientations)
        o = torch.as_tensor(o_idx, device=self.device)
        proj = generate_projection_batch(
            self.dft,
            self.mean_scaled,
            self.pad_width,
            self.phi[o],
            self.theta[o],
            torch.zeros_like(self.phi[o]),
        ).to(torch.float32)
        if self.filters is None:
            exact = proj
        else:
            exact = torch.empty_like(proj)
            for f in np.unique(f_idx):
                here = torch.as_tensor(f_idx == f, device=self.device)
                exact[here] = apply_fourier_filters(proj[here], self.filters[f][None])[
                    0
                ]

        if self.normalize:
            var, mean = torch.var_mean(exact, dim=(-2, -1), keepdim=True)
            exact = (exact - mean) / torch.sqrt(var + 1e-8)

        if self.mask_exact:
            exact = exact * self.support

        if self.w_avg is None:
            return exact, exact
        signal = apply_fourier_filters(proj, self.w_avg[None])[0].to(torch.float32)
        return exact, signal


# --------------------------------------------------------------------------- #
# Radial tables
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, eq=False)
class RadialTables:
    """Ring profiles of the kept, residual and signal templates, per template row.

    Parameters
    ----------
    freq : torch.Tensor
        ``(F,)`` float32 frequency axis (cycles per pixel).
    kept, residual, cross, kept_signal, residual_signal : torch.Tensor
        ``(N, F)`` float32 tables ``Gamma_m~``, ``Gamma_delta``, ``Gamma_m~delta``,
        ``Gamma_m~p``, ``Gamma_delta p`` of every template row ``f * O + o``.
    support : torch.Tensor
        ``(kH, kW)`` bool polar support of the template box.
    rectangles : tuple of Rectangle
        The feature selection.
    exact : {"projection", "reconstruction"}
        Whether ``m`` was the volume's projection or the full stored reconstruction.
    signal_source : {"projection", "template"}
        Whether ``p`` was the ``W_avg``-whitened projection or ``m`` itself.
    pad : int
        Zero-padding factor of the spectra the tables were binned from.
    """

    freq: torch.Tensor
    kept: torch.Tensor
    residual: torch.Tensor
    cross: torch.Tensor
    kept_signal: torch.Tensor
    residual_signal: torch.Tensor
    support: torch.Tensor
    rectangles: tuple[Rectangle, ...]
    exact: Literal["projection", "reconstruction"]
    signal_source: Literal["projection", "template"]
    pad: int = 2

    def __post_init__(self) -> None:
        """Validate shapes, dtypes and the enumerations."""
        if self.freq.dim() != 1 or self.freq.dtype is not torch.float32:
            raise ValueError("freq must be a 1-D float32 tensor")

        n_bins = self.freq.numel()
        n = None

        for name in TABLE_NAMES:
            t = getattr(self, name)
            if t.dim() != 2 or t.dtype is not torch.float32 or t.shape[1] != n_bins:
                raise ValueError(f"{name} must be a float32 (N, {n_bins}) table")
            if n is None:
                n = int(t.shape[0])
            elif t.shape[0] != n:
                raise ValueError("every table must have the same number of rows")

        if self.support.dim() != 2 or self.support.dtype is not torch.bool:
            raise ValueError("support must be a 2-D bool tensor")

        if self.exact not in ("projection", "reconstruction"):
            raise ValueError(f"unknown exact {self.exact!r}")

        if self.signal_source not in ("projection", "template"):
            raise ValueError(f"unknown signal_source {self.signal_source!r}")

        if self.pad < 1:
            raise ValueError("pad must be >= 1")

    @classmethod
    def build(
        cls,
        template: Template,
        rectangles: Rectangle | Sequence[Rectangle],
        *,
        volume: torch.Tensor | np.ndarray | None = None,
        whitening: WhiteningFilter | torch.Tensor | None = None,
        normalize: bool = True,
        mask_exact: bool = True,
        pad: int = 2,
        batch: int = 32,
        show_progress: bool = False,
    ) -> RadialTables:
        """Tabulate every template row of ``template`` for one feature selection.

        Parameters
        ----------
        template : Template
            The decomposition, its transform and the warp settings.
        rectangles : Rectangle or sequence of Rectangle
            The feature selection of the stage the tables serve.
        volume : torch.Tensor or numpy.ndarray, optional
            The structure; with it the exact template is the projection (the
            reference search's template) and the residual includes what the
            decomposition itself dropped. Without it the exact template is the full
            stored reconstruction.
        whitening : WhiteningFilter or torch.Tensor, optional
            ``W_avg`` for the signal projection ``p`` (needs ``volume``); without it
            the signal tables use ``m`` for ``p``.
        normalize, mask_exact : bool, optional
            See :class:`ProjectedTemplates`.
        pad : int, optional
            Zero-padding factor of the template spectra (default 2).
        batch : int, optional
            Template rows per reconstruction batch.
        show_progress : bool, optional
            Print a progress bar.

        Returns
        -------
        RadialTables
        """
        svd = SvdTemplates(template, rectangles)
        projected = None

        if volume is not None:
            projected = ProjectedTemplates(
                template,
                volume,
                whitening=whitening,
                normalize=normalize,
                mask_exact=mask_exact,
            )

        elif whitening is not None:
            raise ValueError("whitening needs a volume: the signal tables project it")

        k_h, k_w = svd.template_shape
        padded = (pad * k_h, pad * k_w)
        n_bins = calculate_num_freq_bins(padded)
        freq = frequency_axis(n_bins)
        n = svd.n_templates
        tables = {
            name: torch.empty((n, n_bins), dtype=torch.float64) for name in TABLE_NAMES
        }
        progress = None

        if show_progress:
            progress = tqdm(total=n, desc="radial tables", unit="row")

        for r0 in range(0, n, batch):
            rows = torch.arange(r0, min(r0 + batch, n))
            kept, full = svd(rows)

            if projected is None:
                exact, signal = full, full
            else:
                exact, signal = projected(rows)

            delta = exact - kept
            fk = torch.fft.rfft2(kept.to(torch.float64), s=padded)
            fd = torch.fft.rfft2(delta.to(torch.float64), s=padded)
            fp = torch.fft.rfft2(signal.to(torch.float64), s=padded)
            sl = slice(int(rows[0]), int(rows[-1]) + 1)
            tables["kept"][sl] = ring_sums(fk.abs().square(), freq, padded).cpu()
            tables["residual"][sl] = ring_sums(fd.abs().square(), freq, padded).cpu()
            tables["cross"][sl] = ring_sums((fk.conj() * fd).real, freq, padded).cpu()
            tables["kept_signal"][sl] = ring_sums(
                (fk.conj() * fp).real, freq, padded
            ).cpu()
            tables["residual_signal"][sl] = ring_sums(
                (fd.conj() * fp).real, freq, padded
            ).cpu()

            if progress is not None:
                progress.update(int(rows.numel()))
        if progress is not None:
            progress.close()
        return cls(
            freq=freq,
            **{name: t.to(torch.float32) for name, t in tables.items()},
            support=svd.support.cpu(),
            rectangles=svd.rectangles,
            exact="reconstruction" if projected is None else "projection",
            signal_source="template" if projected is None else projected.signal_source,
            pad=int(pad),
        )

    # -- sizes -----------------------------------------------------------------

    @property
    def n_templates(self) -> int:
        """Template rows ``N``."""
        return int(self.kept.shape[0])

    @property
    def n_bins(self) -> int:
        """``F``."""
        return int(self.freq.numel())

    @property
    def template_shape(self) -> tuple[int, int]:
        """``(kH, kW)``."""
        return int(self.support.shape[0]), int(self.support.shape[1])

    def summary(self) -> dict:
        """JSON-serialisable description (for result files)."""
        return {
            "n_templates": self.n_templates,
            "n_bins": self.n_bins,
            "template_shape": list(self.template_shape),
            "rectangles": [list(r) for r in self.rectangles],
            "exact": self.exact,
            "signal_source": self.signal_source,
            "pad": self.pad,
        }

    # -- scales ----------------------------------------------------------------

    def scales(
        self,
        spectra: ImageSpectra,
        *,
        hyps: HypothesisSet | None = None,
        b_factor: float = 0.0,
    ) -> NoiseScales:
        """The per-micrograph scales of the note for every ``(image, plane, row)``.

        Parameters
        ----------
        spectra : ImageSpectra
            The micrograph's curves (resampled onto this table's axis).
        hyps : HypothesisSet, optional
            The stage's hypotheses; ``qbar_perp`` averages over its template rows and
            planes (all rows and planes when ``None``).
        b_factor : float, optional
            Envelope ``exp(-B s^2 / 4)`` (``s = f / pixel_size`` in 1/A) on the signal
            weight, as ``torch_fourier_filter.envelopes.b_envelope``. Defaults to 0.

        Returns
        -------
        NoiseScales
        """
        freq = self.freq.to(torch.float64)
        noise = _resample(spectra.freq, spectra.noise, freq)  # (B', D', F)
        envelope = torch.exp(-b_factor * (freq / spectra.pixel_size) ** 2 / 4.0)
        weight = _resample(spectra.freq, spectra.signal, freq) * envelope
        tables = {name: getattr(self, name).to(torch.float64) for name in TABLE_NAMES}
        q_kept = noise @ tables["kept"].T
        q_residual = noise @ tables["residual"].T
        q_cross = noise @ tables["cross"].T
        lam = torch.where(q_kept > 0, q_cross / q_kept.clamp_min(1e-300), 0.0)
        q_perp = (q_residual - lam**2 * q_kept).clamp_min(0.0)
        g_kept = weight @ tables["kept_signal"].T
        g_residual = weight @ tables["residual_signal"].T
        total = g_kept + g_residual
        rho = torch.where(total > 0, g_residual / total.clamp_min(1e-300), 0.0)
        rho = rho.clamp(0.0, 1.0 - 1e-6)

        # the noise and signal curves may broadcast over different axes
        shape = tuple(max(a, b) for a, b in zip(q_kept.shape, rho.shape, strict=True))
        q_kept, q_residual, q_cross, lam, q_perp, rho = (
            t.expand(shape).contiguous()
            for t in (q_kept, q_residual, q_cross, lam, q_perp, rho)
        )

        if hyps is None:
            rows = torch.arange(self.n_templates)
            planes: tuple[int, ...] = tuple(range(q_perp.shape[1]))
        else:
            rows = hyps.template_rows()
            planes = hyps.defocus_planes()

        plane_idx = [0 if q_perp.shape[1] == 1 else int(p) for p in planes]
        qbar_perp = q_perp[:, plane_idx][:, :, rows].mean(dim=(1, 2))

        return NoiseScales(
            q_kept,
            q_residual,
            q_cross,
            lam,
            q_perp,
            rho,
            qbar_perp,
            self.rectangles,
            b_factor,
        )

    # -- I/O -------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the tables and their description to an HDF5 file."""
        with h5py.File(path, "w") as f:
            f.create_dataset("freq", data=self.freq.numpy())
            for name in TABLE_NAMES:
                f.create_dataset(name, data=getattr(self, name).numpy())
            f.create_dataset("support", data=self.support.numpy())
            f.attrs["rectangles"] = json.dumps([list(r) for r in self.rectangles])
            f.attrs["exact"] = self.exact
            f.attrs["signal_source"] = self.signal_source
            f.attrs["pad"] = int(self.pad)

    @classmethod
    def load(cls, path: str | Path) -> RadialTables:
        """Read a file written by :meth:`save`."""
        with h5py.File(path, "r") as f:
            tables = {
                name: torch.from_numpy(np.asarray(f[name], dtype=np.float32))
                for name in TABLE_NAMES
            }
            freq = torch.from_numpy(np.asarray(f["freq"], dtype=np.float32))
            support = torch.from_numpy(np.asarray(f["support"], dtype=bool))
            rects = tuple(as_rectangle(r) for r in json.loads(f.attrs["rectangles"]))
            exact = str(f.attrs["exact"])
            signal_source = str(f.attrs["signal_source"])
            pad = int(f.attrs["pad"])
        return cls(
            freq=freq,
            **tables,
            support=support,
            rectangles=rects,
            exact=exact,  # type: ignore[arg-type]
            signal_source=signal_source,  # type: ignore[arg-type]
            pad=pad,
        )


@dataclass(frozen=True, eq=False)
class NoiseScales:
    """The note's per-micrograph scales for one feature selection.

    Parameters
    ----------
    q_kept, q_residual, q_cross : torch.Tensor
        ``(B', D', N)`` float64 noise-weighted energies ``q_m~``, ``q_delta``,
        ``q_m~delta`` (variances of the kept / missed noise and their covariance at
        unit level).
    lam : torch.Tensor
        ``(B', D', N)`` regression slope ``lambda_o = q_m~delta / q_m~``.
    q_perp : torch.Tensor
        ``(B', D', N)`` variance of the missed noise independent of the kept noise.
    rho : torch.Tensor
        ``(B', D', N)`` signal loss, in ``[0, 1)``.
    qbar_perp : torch.Tensor
        ``(B',)`` mean of ``q_perp`` over the stage's hypotheses.
    rectangles : tuple of Rectangle
        The feature selection the scales belong to.
    b_factor : float
        The envelope applied to the signal weight.
    """

    q_kept: torch.Tensor
    q_residual: torch.Tensor
    q_cross: torch.Tensor
    lam: torch.Tensor
    q_perp: torch.Tensor
    rho: torch.Tensor
    qbar_perp: torch.Tensor
    rectangles: tuple[Rectangle, ...]
    b_factor: float = 0.0

    def __post_init__(self) -> None:
        """Validate shapes and dtypes."""
        shape = tuple(self.q_kept.shape)
        if len(shape) != 3:
            raise ValueError("scales must be (B', D', N) tensors")
        for name in SCALE_NAMES:
            t = getattr(self, name)
            if tuple(t.shape) != shape or t.dtype is not torch.float64:
                raise ValueError(f"{name} must be a float64 tensor of shape {shape}")
        if tuple(self.qbar_perp.shape) != (shape[0],):
            raise ValueError("qbar_perp must have one entry per image axis entry")

    @property
    def n_templates(self) -> int:
        """``N``."""
        return int(self.q_kept.shape[-1])

    def to(self, device: torch.device | str) -> NoiseScales:
        """Copy with every tensor on ``device``."""
        moved = {name: getattr(self, name).to(device) for name in SCALE_NAMES}
        return NoiseScales(
            **moved,
            qbar_perp=self.qbar_perp.to(device),
            rectangles=self.rectangles,
            b_factor=self.b_factor,
        )

    def save(self, path: str | Path) -> None:
        """Write the scales to an HDF5 file."""
        with h5py.File(path, "w") as f:
            for name in (*SCALE_NAMES, "qbar_perp"):
                f.create_dataset(name, data=getattr(self, name).cpu().numpy())
            f.attrs["rectangles"] = json.dumps([list(r) for r in self.rectangles])
            f.attrs["b_factor"] = float(self.b_factor)

    @classmethod
    def load(cls, path: str | Path) -> NoiseScales:
        """Read a file written by :meth:`save`."""
        with h5py.File(path, "r") as f:
            data = {
                name: torch.from_numpy(np.asarray(f[name], dtype=np.float64))
                for name in (*SCALE_NAMES, "qbar_perp")
            }
            rects = tuple(as_rectangle(r) for r in json.loads(f.attrs["rectangles"]))
            b_factor = float(f.attrs["b_factor"])
        return cls(**data, rectangles=rects, b_factor=b_factor)


# --------------------------------------------------------------------------- #
# Image spectra and the level map
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SpectraConfig:
    """Parameters of :func:`estimate_spectra` and :func:`level_map`.

    Attributes
    ----------
    pixel_size : float
        Angstrom per pixel (recorded on the spectra for the B-factor envelope).
    num_freq_bins : int, optional
        Bins of the curves' axis; ``None`` uses the tables' axis (no resampling).
    band_cutoff, band_falloff : float
        Template band of the level map, in cycles per pixel
        (``torch_fourier_filter.bandpass.low_pass_filter``); ``band_cutoff=None``
        keeps every frequency.
    level : {"disk", "tiles", "none"}
        How the local noise level is estimated: the mean of the flattened squared
        image over the template's support disk at every pixel; robust tile medians
        interpolated between tile centres; or not at all (level 1 everywhere).
    floor : float
        Floor of the noise power in the flattening, as a fraction of its in-band mean
        (keeps CTF zeros from being amplified).
    tile, margin : int
        Tile side and ignored image border of the ``"tiles"`` estimate.
    """

    pixel_size: float = 1.0
    num_freq_bins: int | None = None
    band_cutoff: float | None = None
    band_falloff: float = 0.0
    level: Literal["disk", "tiles", "none"] = "disk"
    floor: float = 0.05
    tile: int = 256
    margin: int = 32

    def __post_init__(self) -> None:
        """Validate."""
        if not self.pixel_size > 0.0:
            raise ValueError("pixel_size must be positive")
        if self.num_freq_bins is not None and self.num_freq_bins < 2:
            raise ValueError("num_freq_bins must be >= 2")
        if self.band_cutoff is not None and not 0.0 < self.band_cutoff:
            raise ValueError("band_cutoff must be positive or None")
        if self.band_falloff < 0.0:
            raise ValueError("band_falloff must be >= 0")
        if self.level not in ("disk", "tiles", "none"):
            raise ValueError(f"unknown level estimate {self.level!r}")
        if not 0.0 < self.floor <= 1.0:
            raise ValueError("floor must lie in (0, 1]")
        if self.tile < 2 or self.margin < 0:
            raise ValueError("tile must be >= 2 and margin >= 0")


def _ring_power(dft: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Ring mean of ``|dft|^2`` on the standard axis (float64, CPU)."""
    return (
        power_spectral_density(dft, rfft=True, num_freq_bins=n_bins)
        .double()
        .square()
        .cpu()
    )


def estimate_spectra(
    images: ImageSet,
    tables: RadialTables,
    config: SpectraConfig | None = None,
    *,
    correction: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> ImageSpectra:
    """The curves and level maps of an image set (step 4a-4b of the pipeline).

    Parameters
    ----------
    images : ImageSet
        The search images (whitened as the caller sees fit) and their CTF planes.
    tables : RadialTables
        Supplies the frequency axis and the template support of the level map.
    config : SpectraConfig, optional
        Estimation parameters.
    correction : tuple[torch.Tensor, torch.Tensor], optional
        ``(freq, curves)``: the whitening corrections ``C_i`` of the images on their
        own axis, ``(F_c,)`` shared or ``(B, F_c)`` per image
        (``WhiteningFilter.correction_1d``). Their square enters the signal weight.

    Returns
    -------
    ImageSpectra
        ``noise (B, D, F)``, ``signal (1 | B, D, F)``, ``level (B, H, W)`` (or
        ``(1, 1, 1)`` with ``level="none"``).
    """
    if config is None:
        config = SpectraConfig()
    n_images, n_planes = images.n_images, images.n_defocus
    h, w = images.image_shape
    n_bins = config.num_freq_bins or tables.n_bins
    freq = frequency_axis(n_bins)
    x = images.images.to(torch.float32)
    noise = torch.empty((n_images, n_planes, n_bins), dtype=torch.float64)
    shared = images.filters is None or not images.per_image_filters
    signal = torch.ones(
        (1 if shared else n_images, n_planes, n_bins), dtype=torch.float64
    )
    for b in range(n_images):
        dft = torch.fft.rfft2(x[b] - x[b].mean())
        for d in range(n_planes):
            filt = images.filter_plane(d, b)
            plane = dft if filt is None else dft * filt.to(dft.device)
            noise[b, d] = _ring_power(plane, n_bins) / float(h * w)
            if filt is not None and (b == 0 or not shared):
                signal[0 if shared else b, d] = _ring_power(filt, n_bins)
    if correction is not None:
        c_freq, curves = correction
        c = _resample(
            c_freq,
            torch.as_tensor(curves).reshape(-1, int(torch.as_tensor(curves).shape[-1])),
            freq,
        )
        if c.shape[0] not in (1, n_images):
            raise ValueError(
                f"correction needs 1 or {n_images} curves, got {c.shape[0]}"
            )
        signal = signal * c.square()[:, None, :]
    level = None
    if config.level != "none":
        level = level_map(x, noise, freq, tables.support, config)
    return ImageSpectra.create(freq, noise, signal, level, config.pixel_size)


def _disk_mean(y2: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Mean of ``y2`` over the support disk centred at every pixel (FFT convolution)."""
    h, w = int(y2.shape[-2]), int(y2.shape[-1])
    k_h, k_w = int(support.shape[0]), int(support.shape[1])
    kernel = support.to(y2.dtype) / support.sum()
    full = (h + k_h - 1, w + k_w - 1)
    k_spec = torch.fft.rfft2(kernel, s=full)
    num: torch.Tensor = torch.fft.irfft2(torch.fft.rfft2(y2, s=full) * k_spec, s=full)
    den: torch.Tensor = torch.fft.irfft2(
        torch.fft.rfft2(torch.ones_like(y2), s=full) * k_spec, s=full
    )
    r0, c0 = (k_h - 1) // 2, (k_w - 1) // 2
    local: torch.Tensor = num[r0 : r0 + h, c0 : c0 + w]
    return local / den[r0 : r0 + h, c0 : c0 + w].clamp_min(1e-12)


def _tile_median(y2: torch.Tensor, tile: int, margin: int) -> torch.Tensor:
    """Tile medians of ``y2`` over the chi^2_1 median, interpolated to every pixel."""
    h, w = int(y2.shape[-2]), int(y2.shape[-1])
    n_h, n_w = h // tile, w // tile
    if n_h < 1 or n_w < 1:
        raise ValueError(
            f"image {(h, w)} is smaller than a {tile} px tile; use level='disk'"
        )
    y = y2[: n_h * tile, : n_w * tile].clone()
    if margin > 0:
        y[:margin], y[-margin:], y[:, :margin], y[:, -margin:] = (float("nan"),) * 4
    tiles = y.view(n_h, tile, n_w, tile).permute(0, 2, 1, 3).reshape(n_h, n_w, -1)
    med = tiles.nanmedian(dim=-1).values / CHI2_1_MEDIAN
    if n_h == 1 and n_w == 1:
        return med.expand(h, w).clone()
    up = F.interpolate(
        med[None, None], size=(h, w), mode="bicubic", align_corners=False
    )[0, 0]
    return up.clamp_min(1e-12)


def level_map(
    images: torch.Tensor,
    noise: torch.Tensor,
    freq: torch.Tensor,
    support: torch.Tensor,
    config: SpectraConfig | None = None,
) -> torch.Tensor:
    """Relative local noise level ``kappa^2(s) / kappa_i^2`` at every image pixel.

    Each image is flattened to unit-level white noise inside the template band
    (``L(f) / sqrt(max(S(f), floor * mean_band S))`` with ``S`` its plane-averaged
    noise curve), squared, and averaged locally: over the support disk at every pixel
    (``"disk"``) or by tile medians (``"tiles"``). The map is normalised to mean 1.

    Parameters
    ----------
    images : torch.Tensor
        ``(B, H, W)`` real images.
    noise : torch.Tensor
        ``(B, D, F)`` noise curves of their planes.
    freq : torch.Tensor
        ``(F,)`` the curves' axis.
    support : torch.Tensor
        ``(kH, kW)`` bool template support (the disk).
    config : SpectraConfig, optional
        Band, flattening floor and the estimate (``"disk"`` or ``"tiles"``).

    Returns
    -------
    torch.Tensor
        ``(B, H, W)`` float32 maps (CPU).
    """
    if config is None:
        config = SpectraConfig()
    if config.level == "none":
        raise ValueError("level='none' estimates no map")
    n_images, h, w = int(images.shape[0]), int(images.shape[-2]), int(images.shape[-1])
    device = images.device
    radius = fftfreq_grid(image_shape=(h, w), rfft=True, fftshift=False, norm=True)
    band = (
        torch.ones((h, w // 2 + 1), device=device)
        if config.band_cutoff is None
        else low_pass_filter(
            config.band_cutoff,
            config.band_falloff,
            (h, w),
            rfft=True,
            fftshift=False,
            device=device,
        ).to(torch.float32)
    )
    in_band = band > 1e-3
    out = torch.empty((n_images, h, w), dtype=torch.float32)
    for b in range(n_images):
        curve = noise[b].mean(dim=0).to(torch.float32)
        s_grid = curve_1dim_to_ndim(curve.cpu(), freq.cpu(), radius).to(device)
        s_band = s_grid[in_band].mean()
        flatten = band / torch.sqrt(s_grid.clamp_min(config.floor * s_band))
        x = images[b].to(torch.float32)
        y = torch.fft.irfft2(torch.fft.rfft2(x - x.mean()) * flatten, s=(h, w))
        y2 = y.square()
        local = (
            _disk_mean(y2, support.to(device))
            if config.level == "disk"
            else _tile_median(y2, config.tile, config.margin)
        )
        out[b] = (local / local.mean()).cpu()
    return out


# --------------------------------------------------------------------------- #
# Threshold and calibration helpers
# --------------------------------------------------------------------------- #


def zscore_threshold(n_pixels: int, n_hypotheses: int) -> float:
    """``T = Phi^-1(1 - 1 / (N_pix N))``: one expected false detection per search."""
    tests = float(n_pixels) * float(n_hypotheses)
    if tests < 1.0:
        raise ValueError("n_pixels and n_hypotheses must be positive")
    return float(-ndtri(1.0 / tests))


def zeta_quantile(
    sigma_exact: torch.Tensor, sigma_hat: torch.Tensor, beta: float
) -> float:
    """``zeta_beta``: the lower ``beta``-quantile of ``sigma / sigma_hat``.

    Measured on a reference (exact) search; the fraction ``beta`` of pixels whose
    exact ``sigma`` falls below ``zeta_beta * sigma_hat`` is the tolerated failure of
    the sigma bound.
    """
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must lie in (0, 1)")
    ratio = (sigma_exact.to(torch.float64) / sigma_hat.to(torch.float64)).reshape(-1)
    ratio = ratio[torch.isfinite(ratio)]
    if ratio.numel() == 0:
        raise ValueError("no finite sigma ratios")
    return float(torch.quantile(ratio, beta))
