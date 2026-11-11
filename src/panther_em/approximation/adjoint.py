"""Adjoint-warp featurization: a box's features without its Cartesian kernels.

A feature kernel is the polar-to-Cartesian warp of a separable polar image. The bank
builds ``K_km = mask * Warp(Wrap(Pad_k(a_k (x) b_km / jac)))``: the angular phase
vector ``a_k`` times the radial SVD vector with the Jacobian removed, the radial axis
continued (``Pad_k``; a spiral transform's origin continuation keeps the image an
outer product ``a_k (x) c_pad`` through a small linear map on the innermost rings),
the angular axis circularly padded, B-spline interpolated at the Cartesian pixels'
polar coordinates (``Warp`` = ``grid-constant`` prefilter + interpolation) and zeroed
outside the transform's support. Every step is linear, so the inner products of a box
with all kernels of a rectangle are

    <box, conj K_km> = conj( filt(a~_k)^T  [S^T (mask * box)]  filt(c_pad,km) )

with ``S`` the interpolation matrix (pixels x padded polar grid) and ``filt`` the
prefilter. :class:`WarpAdjoint` applies ``S^T`` to a box batch as one sparse matmul;
:class:`AdjointBases` holds the prefiltered angular and radial vectors of a rectangle
and finishes with a small angular GEMM and a per-``k`` radial contraction. On a 512
box with 5760 cells this runs at 2x (fp32, 1e-6 relative) to 3.5x (fp16 scatter and
GEMM, 5e-4) the speed of the ``2 r x kH kW`` planar GEMM and needs no planar block.
"""

from __future__ import annotations

import contextlib
import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from panther_em._precision import check_precision

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from panther_em.approximation.reconstructor import Reconstructor
    from panther_em.coordinates.transform_base import CoordinateTransform
    from panther_em.coordinates.warp_config import WarpConfig
    from panther_em.decomposition.rectangles import Rectangle

NPAD = 12
"""ndimage's ``grid-constant`` pre-padding (samples of ``cval`` per edge) before the
spline prefilter, for interpolation orders above 1."""

SELF_CHECK_TOL = 1e-5
"""Tolerance of the build-time check that the transform's radial continuation of an
angularly monochromatic image is the outer product the adjoint assumes."""


def bspline(order: int, x: torch.Tensor) -> torch.Tensor:
    """Centered cardinal B-spline of degree ``order`` (support ``(order + 1) / 2``)."""
    out = torch.zeros_like(x)
    for k in range(order + 2):
        term = (x + (order + 1) / 2 - k).clamp(min=0.0) ** order
        out = out + ((-1) ** k) * math.comb(order + 1, k) * term
    return out / math.factorial(order)


def _axis_taps(
    c: torch.Tensor, order: int, size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coefficient indices and weights of ``order + 1`` taps per sample along one axis.

    ndimage's convention: the footprint starts at ``floor(c) - order // 2`` for odd
    orders and ``floor(c + 0.5) - order // 2`` for even ones; ``grid-constant`` reads
    ``cval = 0`` outside ``[0, size)``, so those taps are dropped (weights zeroed).
    """
    half = order // 2
    start = torch.floor(c) - half if order % 2 else torch.floor(c + 0.5) - half
    idx = start[:, None] + torch.arange(order + 1, dtype=c.dtype, device=c.device)
    weights = bspline(order, c[:, None] - idx)
    valid = (idx >= 0) & (idx <= size - 1)
    return idx.to(torch.int64), weights * valid, valid


@contextlib.contextmanager
def _quiet_sparse() -> Iterator[None]:
    """Silence torch's one-off "sparse CSR is beta" / invariant-check notices."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*[Ss]parse.*", category=UserWarning)
        yield


@contextlib.contextmanager
def _tf32(enabled: bool) -> Iterator[None]:
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def unsupported_reason(
    transform: CoordinateTransform, config: WarpConfig
) -> str | None:
    """Why the adjoint cannot mirror ``to_cartesian`` for this transform/config."""
    if config.mode != "constant" or float(config.cval) != 0.0:
        return "only mode='constant' with cval=0 (ndimage grid-constant) is mirrored"
    if not 1 <= int(config.order) <= 5:
        return "interpolation order must be in [1, 5]"
    if transform.has_periodic_axis and transform.periodic_axis not in (None, 0):
        return "periodic axis must be axis 0"
    if config.preserve_energy and transform.jacobian_grid is not None:
        jac = np.asarray(transform.jacobian_grid)
        if not np.array_equal(jac, np.broadcast_to(jac[:1], jac.shape)):
            return "the Jacobian varies with angle"
    probe = np.zeros(transform.polar_shape, dtype=np.float32)
    padded, n_pre = transform.pad_radial_axis(probe)
    if (n_pre or padded.shape[1] != probe.shape[1]) and (
        getattr(transform, "radial_continuation", None) is None
        or transform.radial_continuation([0], torch.device("cpu")) is None
    ):
        return "the radial padding has no continuation operator"
    return None


@dataclass(eq=False)
class WarpAdjoint:
    """The transpose of the bank's polar-to-Cartesian interpolation, as a sparse matrix.

    Built once per transform and box shape by :meth:`build`; :meth:`apply` scatters a
    box batch onto the padded polar grid (``S^T``), :meth:`pad_radial` and
    :meth:`prefilter` mirror the radial continuation and the spline prefilter for
    :func:`build_bases`.

    Attributes
    ----------
    transform : CoordinateTransform
    order, wrap, npad : int
        Spline order, circular padding of the angular axis, ``grid-constant`` padding.
    n_pre, n_post : int
        Rows the transform prepends / appends to the radial axis.
    angular_size, radial_size : int
        Padded polar grid ``(A2, R2)`` the scatter targets.
    jacobian : torch.Tensor or None
        ``(R,)`` float64 radial Jacobian removed from the radial vectors.
    scatter : torch.Tensor
        CSR ``(A2 * R2, H * W)`` float32 on ``device``: interpolation weights of the
        pixels inside the support, transposed.
    """

    transform: Any
    order: int
    wrap: int
    npad: int
    n_pre: int
    n_post: int
    angular_size: int
    radial_size: int
    jacobian: torch.Tensor | None
    scatter: torch.Tensor
    _half: torch.Tensor | None = field(default=None, repr=False)

    @property
    def device(self) -> torch.device:
        """Resident torch.device for the scatter."""
        return self.scatter.device

    @classmethod
    def build(
        cls,
        transform: CoordinateTransform,
        config: WarpConfig,
        device: torch.device | str,
    ) -> WarpAdjoint | None:
        """The adjoint for ``transform`` under ``config``; ``None`` if unsupported."""
        if unsupported_reason(transform, config) is not None:
            return None
        device = torch.device(device)
        order = int(config.order)
        wrap = (
            4 * order
            if (config.wrap_angular_axis and transform.has_periodic_axis)
            else 0
        )
        npad = NPAD if order > 1 else 0
        n_angle, n_radius = transform.polar_shape
        probe = np.zeros((n_angle, n_radius), dtype=np.float32)
        padded, n_pre = transform.pad_radial_axis(probe)
        n_post = int(padded.shape[1]) - n_radius - int(n_pre)
        a2 = n_angle + 2 * wrap + 2 * npad
        r2 = n_radius + int(n_pre) + n_post + 2 * npad
        jac = None
        if config.preserve_energy and transform.jacobian_grid is not None:
            jac = torch.as_tensor(
                np.asarray(transform.jacobian_grid)[0], dtype=torch.float64
            )
        cart = torch.as_tensor(
            np.asarray(transform.device_coords("numpy")["cartesian"]),
            dtype=torch.float64,
        )
        height, width = int(cart.shape[1]), int(cart.shape[2])
        cy = cart[0].reshape(-1) + wrap + npad
        cx = cart[1].reshape(-1) + int(n_pre) + npad
        iy, wy, vy = _axis_taps(cy, order, a2)
        ix, wx, vx = _axis_taps(cx, order, r2)
        keep = vy[:, :, None] & vx[:, None, :]
        outer = transform.outer_radius_index()
        if outer is not None:
            keep &= (cart[1].reshape(-1) <= float(outer))[:, None, None]
        pixel = torch.arange(height * width)[:, None, None].expand(
            -1, order + 1, order + 1
        )
        polar = iy[:, :, None] * r2 + ix[:, None, :]
        values = wy[:, :, None] * wx[:, None, :]
        with _quiet_sparse():
            coo = torch.sparse_coo_tensor(
                torch.stack([polar[keep], pixel[keep]]),
                values[keep].to(torch.float32),
                (a2 * r2, height * width),
            )
            scatter = coo.coalesce().to_sparse_csr().to(device)
        adjoint = cls(
            transform, order, wrap, npad, int(n_pre), n_post, a2, r2, jac, scatter
        )
        if not adjoint._radial_padding_is_separable():
            return None
        return adjoint

    # -- the polar side ----------------------------------------------------------

    def pad_radial(self, c: torch.Tensor, ks: Sequence[int]) -> torch.Tensor:
        """``(K, ..., R)`` complex radial vectors -> the padded ``(K, ..., Rp)``.

        Mirrors ``transform.pad_radial_axis`` for images ``a_k (x) c``: the prepended
        rows are ``flip(E_k c[:n_nodes])`` with ``E_k`` from
        ``transform.radial_continuation``, the appended rows the last ``n_post``
        samples mirrored.
        """
        if not self.n_pre and not self.n_post:
            return c
        parts = []
        if self.n_pre:
            cont = self.transform.radial_continuation(list(ks), torch.device("cpu"))
            cont = cont.to(c.dtype)  # (K, n_pre, n_nodes)
            n_nodes = int(cont.shape[-1])
            lead = c[..., :n_nodes].reshape(c.shape[0], -1, n_nodes)
            inner = torch.einsum("kmi,kpi->kmp", lead, cont).reshape(*c.shape[:-1], -1)
            parts.append(inner.flip(-1))
        parts.append(c)
        if self.n_post:
            parts.append(c[..., c.shape[-1] - self.n_post :].flip(-1))
        return torch.cat(parts, dim=-1)

    def prefilter(self, x: torch.Tensor) -> torch.Tensor:
        """``grid-constant`` spline prefilter of real ``(..., L)`` rows (float64).

        Pads ``npad`` zeros per edge first, as ndimage does before filtering; orders
        0 and 1 are not prefiltered (and not padded) by ``map_coordinates``.
        """
        x = x.to(torch.float64)
        if self.order <= 1:
            return x
        from scipy.ndimage import spline_filter1d

        padded = torch.nn.functional.pad(x, (self.npad, self.npad)).numpy()
        out = spline_filter1d(padded, order=self.order, axis=-1, mode="grid-constant")
        return torch.from_numpy(np.ascontiguousarray(out))

    def _radial_padding_is_separable(self) -> bool:
        """Check ``pad_radial_axis(Re(a_k (x) c)) == Re(a_k (x) pad_radial(c))``."""
        if not self.n_pre and not self.n_post:
            return True
        n_angle, n_radius = self.transform.polar_shape
        gen = torch.Generator().manual_seed(0)
        c = torch.complex(
            torch.randn(n_radius, generator=gen, dtype=torch.float64),
            torch.randn(n_radius, generator=gen, dtype=torch.float64),
        )
        n = torch.arange(n_angle, dtype=torch.float64)
        for k in (0, 1, 2):
            a = torch.exp(1j * 2 * math.pi * k * n / n_angle)
            image = (a[:, None] * c[None, :]).real.to(torch.float32)
            padded, _ = self.transform.pad_radial_axis(image)
            padded = torch.as_tensor(padded, dtype=torch.float64)
            mine = (a[:, None] * self.pad_radial(c[None], [k])[0][None, :]).real
            if (padded - mine).abs().max() > SELF_CHECK_TOL * padded.abs().max():
                return False
        return True

    # -- the Cartesian side ------------------------------------------------------

    def apply(self, boxes: torch.Tensor, precision: str) -> torch.Tensor:
        """``S^T boxes^T``: ``(B, H W)`` boxes -> ``(A2, R2 * B)`` padded polar images.

        fp16 on CUDA runs the sparse matmul in half precision (half the traffic, ~3e-4
        relative on the features); anything else runs in fp32.
        """
        n = int(boxes.shape[0])
        half = check_precision(precision) == "fp16" and self.device.type == "cuda"
        if half:
            if self._half is None:
                with _quiet_sparse():
                    self._half = self.scatter.to(torch.float16)
            matrix = self._half
        else:
            matrix = self.scatter
        dense = boxes.T.contiguous().to(torch.float16 if half else torch.float32)
        with _quiet_sparse():
            polar: torch.Tensor = torch.sparse.mm(matrix, dense)
        return polar.view(self.angular_size, self.radial_size * n)


@dataclass(eq=False)
class AdjointBases:
    """Prefiltered polar vectors of a rectangle, ready to contract scattered boxes.

    Attributes
    ----------
    angular : torch.Tensor
        ``(2 K, A2)`` float32 ``[Re filt(a~_k); Im filt(a~_k)]``.
    radial_re, radial_im : torch.Tensor
        ``(K, M, R2)`` float32 real and imaginary parts of ``filt(c_pad,km)``.
    """

    angular: torch.Tensor
    radial_re: torch.Tensor
    radial_im: torch.Tensor
    _angular_half: torch.Tensor | None = field(default=None, repr=False)

    @property
    def n_k(self) -> int:
        """Angular frequencies in the rectangle."""
        return int(self.radial_re.shape[0])

    @property
    def n_m(self) -> int:
        """Eigenvectors in the rectangle."""
        return int(self.radial_re.shape[1])

    def features(self, polar: torch.Tensor, precision: str) -> torch.Tensor:
        """``(K M, B)`` complex64 features ``<box, conj K_km>`` of scattered boxes.

        Parameters
        ----------
        polar : torch.Tensor
            ``(A2, R2 * B)`` from :meth:`WarpAdjoint.apply`.
        precision : str
            fp16: the angular GEMM on fp16 tensor cores (fp32 accumulation); tf32 or
            fp16: the radial contraction in tf32. fp32: CUDA cores throughout.
        """
        name = check_precision(precision)
        r2 = int(self.radial_re.shape[-1])
        n = int(polar.shape[1]) // r2
        cuda = polar.is_cuda
        if name == "fp16" and cuda:
            if self._angular_half is None:
                self._angular_half = self.angular.to(torch.float16)
            try:
                z = torch.mm(
                    self._angular_half, polar.to(torch.float16), out_dtype=torch.float32
                )
            except TypeError:  # torch without out_dtype
                z = (self._angular_half @ polar.to(torch.float16)).to(torch.float32)
        else:
            with _tf32(cuda and name == "tf32"):
                z = self.angular @ polar.to(torch.float32)
        z = z.view(2, self.n_k, r2, n)
        with _tf32(cuda and name != "fp32"):
            re = torch.bmm(self.radial_re, z[0]) - torch.bmm(self.radial_im, z[1])
            im = torch.bmm(self.radial_re, z[1]) + torch.bmm(self.radial_im, z[0])
        return torch.complex(re, -im).reshape(self.n_k * self.n_m, n)


def build_bases(
    adjoint: WarpAdjoint,
    reconstructor: Reconstructor,
    rect: Rectangle,
) -> AdjointBases:
    """The :class:`AdjointBases` of a ``(k_start, m_start, k_extent, m_extent)``."""
    k0, m0, kx, mx = (int(v) for v in rect)
    ks = list(range(k0, k0 + kx))
    ms = list(range(m0, m0 + mx))
    device = adjoint.device
    angular = torch.stack(
        [
            reconstructor.angular_phase_component(k).to(torch.complex128).cpu()
            for k in ks
        ]
    )
    if adjoint.wrap:
        w = adjoint.wrap
        angular = torch.cat([angular[:, -w:], angular, angular[:, :w]], dim=1)
    fa = torch.complex(adjoint.prefilter(angular.real), adjoint.prefilter(angular.imag))
    indices = np.array([(k, m) for k in ks for m in ms], dtype=np.int64)
    _, _, vh = reconstructor.result.get_svd_tensors(indices, "cpu")
    c = vh.to(torch.complex128).reshape(kx, mx, -1)
    if adjoint.jacobian is not None:
        c = c / adjoint.jacobian
    c_pad = adjoint.pad_radial(c, ks)
    fc = torch.complex(adjoint.prefilter(c_pad.real), adjoint.prefilter(c_pad.imag))
    return AdjointBases(
        angular=torch.cat([fa.real, fa.imag]).to(device=device, dtype=torch.float32),
        radial_re=fc.real.to(device=device, dtype=torch.float32).contiguous(),
        radial_im=fc.imag.to(device=device, dtype=torch.float32).contiguous(),
    )
