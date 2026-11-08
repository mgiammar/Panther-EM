"""The polar-to-Cartesian warp of a decomposition's feature kernels.

A feature kernel ``K_km`` is the polar outer product of the angular phase ``a_k`` and
the radial SVD vector ``Vh_km``, warped onto the template's Cartesian box. Every
consumer of the decomposition must warp it the same way: the featurizations (forward
kernels, or the warp's adjoint) and the noise model's error tables (the reconstruction
and the support disk). :class:`WarpConfig` names the warp and :class:`TemplateWarp`
is its single implementation:

* :meth:`TemplateWarp.spatial` -- the kernels ``(n, kH, kW)``, built on demand (no
  cache: a featurization holds what it needs),
* :attr:`TemplateWarp.adjoint` -- the transposed warp
  (:class:`~panther_em.inference.template.adjoint.WarpAdjoint`), built once,
* :meth:`TemplateWarp.adjoint_bases` -- a rectangle's prefiltered polar vectors,
* :meth:`TemplateWarp.to_cartesian` and :attr:`TemplateWarp.support` -- the generic
  warp of polar images and the support disk, for the error tables.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from panther_em.coordinates.warp_config import WarpConfig
from panther_em.inference.projection_reconstruction import ProjectionReconstructor
from panther_em.inference.template.adjoint import WarpAdjoint, build_bases

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from panther_em.coordinates.transform_base import CoordinateTransform
    from panther_em.decomposition.result import DecompositionBase
    from panther_em.inference.template.adjoint import AdjointBases

BUILD_CHUNK = 64
"""Kernels built per polar-to-Cartesian call."""


def _as_cells(cells: torch.Tensor | Iterable[Sequence[int]]) -> list[tuple[int, int]]:
    """``(n, 2)`` tensors or iterables of pairs as Python ``(k, m)`` tuples."""
    if isinstance(cells, torch.Tensor):
        if cells.dim() != 2 or cells.shape[1] != 2:
            raise ValueError(f"cells must have shape (n, 2), got {tuple(cells.shape)}")
        return [(int(k), int(m)) for k, m in cells.tolist()]
    return [(int(c[0]), int(c[1])) for c in cells]


class TemplateWarp:
    """The warp of one decomposition's kernels under one :class:`WarpConfig`.

    Parameters
    ----------
    result : DecompositionBase
        The decomposition and its coordinate transform.
    config : WarpConfig, optional
        Warp settings; defaults to ``WarpConfig()``.
    device : torch.device or str
        Where kernels, the adjoint and the bases are built.
    """

    def __init__(
        self,
        result: DecompositionBase,
        config: WarpConfig | None = None,
        *,
        device: torch.device | str,
    ) -> None:
        if result.is_complex_projection:
            raise NotImplementedError("Complex-valued projections not yet supported")
        self.result = result
        self.config = config or WarpConfig()
        self.device = torch.device(device)
        self.reconstructor = ProjectionReconstructor(result, device=self.device)
        self.box_shape: tuple[int, int] = tuple(  # type: ignore[assignment]
            int(v) for v in self.reconstructor.image_shape
        )

    @property
    def transform(self) -> CoordinateTransform:
        """The decomposition's polar coordinate transform."""
        return self.reconstructor.coordinate_transform

    def _warp(self, polar: torch.Tensor) -> torch.Tensor:
        """``transform.to_cartesian`` of ``polar`` under :attr:`config`."""
        cfg = self.config
        return self.transform.to_cartesian(
            polar,
            preserve_energy=cfg.preserve_energy,
            wrap_angular_axis=cfg.wrap_angular_axis,
            order=cfg.order,
            mode=cfg.mode,
            cval=cfg.cval,
        )

    # -- forward: Cartesian kernels ------------------------------------------------

    def spatial(self, cells: torch.Tensor | Iterable[Sequence[int]]) -> torch.Tensor:
        """``(n, kH, kW)`` complex64 kernels of ``(k, m)`` cells, in request order.

        Built on every call, grouped by ``k`` (the angular phase is shared and the
        radial vectors are gathered once per chunk).
        """
        keys = _as_cells(cells)
        out = torch.empty(
            (len(keys), *self.box_shape), dtype=torch.complex64, device=self.device
        )
        by_k: dict[int, dict[int, list[int]]] = {}
        for i, (k, m) in enumerate(keys):
            by_k.setdefault(k, {}).setdefault(m, []).append(i)
        for k in sorted(by_k):
            ms = sorted(by_k[k])
            for start in range(0, len(ms), BUILD_CHUNK):
                chunk = ms[start : start + BUILD_CHUNK]
                for m, kernel in zip(chunk, self._build_group(k, chunk), strict=True):
                    out[by_k[k][m]] = kernel
        return out

    def _build_group(self, k: int, ms: Sequence[int]) -> torch.Tensor:
        """Kernels ``(k, m)`` for every ``m`` in ``ms`` -> ``(len(ms), kH, kW)``."""
        if self._separable is not None:
            cupy = self._separable["cupy"]
            # CuPy work on torch's current stream: torch recycles freed memory
            # stream-ordered, so a DLPack view read on another stream would race.
            stream = cupy.cuda.Stream.from_external(
                torch.cuda.current_stream(self.device)
            )
            with stream:
                return self._build_group_separable(k, ms, self._separable)
        recon = self.reconstructor
        indices = np.column_stack([np.full(len(ms), k), np.asarray(ms)])
        _, _, vh = recon.result.get_svd_tensors(indices, recon.device)  # (M, R)
        angular = recon.angular_phase_component(k, 0.0)  # (A,)
        polar = angular[None, :, None] * vh[:, None, :]  # (M, A, R) complex64
        return self._warp(polar).to(torch.complex64)

    # A polar kernel is the outer product of an angular phase vector a_k (A,) and a
    # radial SVD vector b_km (R,), the Jacobian correction depends on the radius only,
    # and the B-spline prefilter of ``map_coordinates`` is linear and applied axis by
    # axis. Hence prefilter2D(a (x) b) == prefilter1D(a) (x) prefilter1D(b) (to
    # 1e-15), so the 2-D recursive filter (1.85 ms of the 2.43 ms per cell) is replaced
    # by two cached 1-D filters, an outer product and the interpolation itself.

    @cached_property
    def _separable(self) -> dict[str, Any] | None:
        """Device-side pieces of the separable fast path (``None`` if inapplicable).

        Mirrors ``skimage.transform.warp`` as used by ``to_cartesian``: mode
        ``constant`` becomes ndimage's ``grid-constant``, which pre-pads the input by
        12 samples of ``cval`` before prefiltering (zero padding keeps the outer
        product separable).
        """
        tf = self.transform
        if self.device.type != "cuda" or self.config.mode != "constant":
            return None
        try:
            import cupy
            import cupyx.scipy.ndimage as ndi
        except ImportError:
            return None
        dc = tf.device_coords(self.device)
        jac = dc["jacobian"]
        if jac is not None and not bool(torch.all(jac == jac[0:1])):
            return None  # Jacobian varies with angle: not separable
        if tf.has_periodic_axis and tf.periodic_axis not in (None, 0):
            return None
        order = self.config.order
        # the generic warp pads the periodic axis by 4 * order rows (the prefilter's
        # boundary error decays like |pole|**rows); mirror it exactly or the fast and
        # generic kernels differ at the 0/360 seam by ~1e-3
        wrap = (
            4 * order if (self.config.wrap_angular_axis and tf.has_periodic_axis) else 0
        )
        # the generic warp may continue the radial axis through the origin (spiral
        # transforms); that continuation mixes angle and radius, so it is not separable
        probe = torch.zeros(tf.polar_shape, device=self.device)
        try:
            _, num_prepended = tf.pad_radial_axis(probe)
        except Exception:
            return None
        if num_prepended:
            return None
        outer = tf.outer_radius_index()
        npad = 12  # ndimage's grid-constant pre-padding
        coords = dc["cartesian"].to(torch.float32).clone()
        support = (
            None if outer is None else (coords[1] <= float(outer)).to(torch.float32)
        )
        coords[0] += wrap + npad
        coords[1] += npad
        return {
            "ndi": ndi,
            "cupy": cupy,
            # own copy: a DLPack view of the torch temporary would dangle once torch
            # recycles that memory
            "coords": cupy.array(cupy.asarray(coords.contiguous()), copy=True),
            "jac": None if jac is None else jac[0].to(torch.float32),  # (R,)
            "wrap": wrap,
            "npad": npad,
            "order": order,
            "support": support,  # (kH, kW) 0/1 mask outside the transform's support
            "angular": {},  # k -> (F a_re, F a_im) cupy float32
        }

    def _build_group_separable(
        self, k: int, ms: Sequence[int], sep: dict[str, Any]
    ) -> torch.Tensor:
        cupy, ndi, order = sep["cupy"], sep["ndi"], sep["order"]
        wrap, npad = sep["wrap"], sep["npad"]
        recon = self.reconstructor
        cval = float(self.config.cval)
        keep: list[torch.Tensor] = []  # torch temporaries viewed by CuPy, kept alive

        def filt(vecs: torch.Tensor) -> Any:
            """1-D grid-constant prefilter of ``(..., L)`` rows, pre-padded by npad."""
            x = torch.nn.functional.pad(
                vecs.to(torch.float32), (npad, npad), value=cval
            ).contiguous()
            keep.append(x)
            return ndi.spline_filter1d(
                cupy.asarray(x),
                order=order,
                axis=-1,
                mode="grid-constant",
                output=cupy.float32,  # default float64 would make the planes/maps f64
            )

        ang = sep["angular"].get(k)
        if ang is None:
            a = recon.angular_phase_component(k, 0.0).to(torch.complex64)
            if wrap:
                a = torch.nn.functional.pad(
                    a[None, None], (wrap, wrap), mode="circular"
                )[0, 0]
            ang = (filt(a.real), filt(a.imag))  # (A',)
            sep["angular"][k] = ang
        indices = np.column_stack([np.full(len(ms), k), np.asarray(ms)])
        _, _, vh = recon.result.get_svd_tensors(indices, recon.device)  # (M, R)
        vh = vh.to(torch.complex64)
        if self.config.preserve_energy and sep["jac"] is not None:
            vh = vh / sep["jac"][None, :]
        fb_r, fb_i = filt(vh.real), filt(vh.imag)  # (M, R')
        fa_r, fa_i = ang
        # prefiltered polar planes for the whole group: (M, A', R')
        planes_r = (
            fa_r[None, :, None] * fb_r[:, None, :]
            - fa_i[None, :, None] * fb_i[:, None, :]
        )
        planes_i = (
            fa_r[None, :, None] * fb_i[:, None, :]
            + fa_i[None, :, None] * fb_r[:, None, :]
        )
        out = torch.empty(
            (len(ms), *self.box_shape), dtype=torch.complex64, device=self.device
        )
        for i in range(len(ms)):
            parts = [
                ndi.map_coordinates(
                    planes[i],
                    sep["coords"],
                    order=order,
                    prefilter=False,
                    mode="grid-constant",
                    cval=cval,
                )
                for planes in (planes_r, planes_i)
            ]
            out[i] = torch.complex(
                torch.as_tensor(parts[0], device=self.device),
                torch.as_tensor(parts[1], device=self.device),
            )
        if sep["support"] is not None:
            out *= sep["support"]
        return out

    # -- adjoint: boxes onto the polar grid -----------------------------------------

    @cached_property
    def adjoint(self) -> WarpAdjoint | None:
        """The transposed warp (built once), or ``None`` if it cannot be mirrored.

        See :func:`~panther_em.inference.template.adjoint.unsupported_reason`.
        """
        return WarpAdjoint.build(self.transform, self.config, self.device)

    def adjoint_bases(self, rect: Sequence[int]) -> AdjointBases:
        """Prefiltered polar vectors of ``rect`` for :attr:`adjoint` (built anew)."""
        adjoint = self.adjoint
        if adjoint is None:
            raise ValueError("this warp has no adjoint (see unsupported_reason)")
        key = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
        return build_bases(adjoint, self.reconstructor, key)

    # -- the generic warp (error tables) ------------------------------------------

    def to_cartesian(self, polar: torch.Tensor) -> torch.Tensor:
        """``(B, n_angle, R)`` polar images -> ``(B, kH, kW)`` float32 Cartesian."""
        return self._warp(polar).to(torch.float32)

    @cached_property
    def support(self) -> torch.Tensor:
        """``(kH, kW)`` bool CPU support: where a warped image of ones exceeds 1/2."""
        warped = self.transform.to_cartesian(
            torch.ones(self.transform.polar_shape),
            preserve_energy=False,
            wrap_angular_axis=self.config.wrap_angular_axis,
            order=self.config.order,
            mode=self.config.mode,
            cval=self.config.cval,
        )
        return warped > 0.5

    def release(self) -> None:
        """Drop the adjoint and the separable build's device state."""
        for name in ("adjoint", "_separable"):
            self.__dict__.pop(name, None)
