"""The forward model of a decomposition: polar features, projections and kernels.

A decomposition stores, per angular frequency ``k``, radial vectors ``Vh_km`` and the
weights ``U * S`` of every template row. A polar feature is the outer product of the
angular phase ``a_k`` and ``Vh_km``; a projection is the weighted sum of its features;
both reach the Cartesian box through the polar-to-Cartesian warp. Every consumer must
warp the same way, so :class:`Reconstructor` is the single implementation and its
:class:`~panther_em.coordinates.WarpConfig` defaults to the one the decomposition
recorded:

* :meth:`Reconstructor.polar_feature`, :meth:`Reconstructor.cartesian_feature` -- one
  basis feature;
* :meth:`Reconstructor.reconstruct`, :meth:`Reconstructor.reconstruct_batch` --
  template projections from their leading components;
* :meth:`Reconstructor.kernels` -- the search's Cartesian feature kernels
  ``(n, kH, kW)``, built on demand (no cache: a featurization holds what it needs);
* :attr:`Reconstructor.adjoint`, :meth:`Reconstructor.adjoint_bases` -- the transposed
  warp (:class:`~panther_em.approximation.adjoint.WarpAdjoint`) and a rectangle's
  prefiltered polar vectors;
* :meth:`Reconstructor.to_cartesian`, :attr:`Reconstructor.support` -- the warp of any
  polar image and the support disk.
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from panther_em.approximation.adjoint import WarpAdjoint, build_bases

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from panther_em.approximation.adjoint import AdjointBases
    from panther_em.coordinates.transform_base import CoordinateTransform
    from panther_em.coordinates.warp_config import WarpConfig
    from panther_em.decomposition.result import DecompositionBase

BUILD_CHUNK = 64
"""Kernels built per polar-to-Cartesian call."""


def _as_cells(cells: torch.Tensor | Iterable[Sequence[int]]) -> list[tuple[int, int]]:
    """``(n, 2)`` tensors or iterables of pairs as Python ``(k, m)`` tuples."""
    if isinstance(cells, torch.Tensor):
        if cells.dim() != 2 or cells.shape[1] != 2:
            raise ValueError(f"cells must have shape (n, 2), got {tuple(cells.shape)}")
        return [(int(k), int(m)) for k, m in cells.tolist()]
    return [(int(c[0]), int(c[1])) for c in cells]


class Reconstructor:
    """The forward model of one decomposition under one warp.

    Parameters
    ----------
    result : DecompositionBase
        The (dense or sparse) decomposition with its coordinate transform.
    warp : WarpConfig, optional
        Interpolation settings of the polar-to-Cartesian warp; defaults to the
        decomposition's own (``result.warp_config``).
    device : torch.device or str
        Where features, projections, kernels and the adjoint are built.
    """

    def __init__(
        self,
        result: DecompositionBase,
        warp: WarpConfig | None = None,
        *,
        device: torch.device | str = "cpu",
    ) -> None:
        if result.coordinate_transform is None:
            raise ValueError("the decomposition must include a coordinate transform")
        self.result = result
        self.warp: WarpConfig = warp or result.warp_config
        self.device = torch.device(device)
        shape = result.coordinate_transform.cartesian_shape
        self.box_shape: tuple[int, int] = (int(shape[0]), int(shape[1]))

    @property
    def transform(self) -> CoordinateTransform:
        """The decomposition's polar coordinate transform."""
        transform = self.result.coordinate_transform
        assert transform is not None  # checked in __init__
        return transform

    def _warp(self, polar: torch.Tensor) -> torch.Tensor:
        """``transform.to_cartesian`` of ``polar`` under :attr:`warp`."""
        out: torch.Tensor = self.transform.to_cartesian(
            polar, **self.warp.cartesian_kwargs()
        )
        return out

    def _require_real(self) -> None:
        if self.result.is_complex_projection:
            raise NotImplementedError(
                "kernels and the adjoint need real-valued projections"
            )

    # -- basis features --------------------------------------------------------------

    def angular_phase_component(self, k: int, phase_shift: float = 0.0) -> torch.Tensor:
        """``(A,)`` complex angular basis vector ``e^(2 pi i k n / A) / sqrt(A)``.

        Parameters
        ----------
        k : int
            Angular frequency: ``[0, k_max]`` real-valued, ``[-k_max, k_max]``
            complex-valued decompositions.
        phase_shift : float
            In-plane rotation in degrees, applied as the phase ``k * phase_shift``.
        """
        return self._angular_phase_components(
            k, torch.tensor([float(phase_shift)], device=self.device)
        )[0]

    def _angular_phase_components(
        self, k: int, phase_shifts: torch.Tensor
    ) -> torch.Tensor:
        """``(B, A)`` angular basis vectors for ``(B,)`` phase shifts in degrees."""
        n_angle = self.result.num_angular_components
        n = torch.arange(n_angle, device=self.device, dtype=torch.float32)
        base = 2.0 * np.pi * k * n / n_angle
        angles = base[None, :] + k * phase_shifts.to(self.device)[:, None] * (
            np.pi / 180.0
        )
        out: torch.Tensor = torch.exp(1j * angles) / np.sqrt(n_angle)
        return out

    def polar_feature(
        self, k: int, m: int, in_plane_rotation: float = 0.0
    ) -> torch.Tensor:
        """``(A, R)`` complex polar feature ``a_k (x) Vh_km``."""
        angular = self.angular_phase_component(k, in_plane_rotation)
        _, _, vh = self.result.get_svd_tensors(np.array([[k, m]]), self.device)
        return torch.outer(angular, vh[0])

    def cartesian_feature(
        self, k: int, m: int, in_plane_rotation: float = 0.0
    ) -> torch.Tensor:
        """``(kH, kW)`` complex feature: :meth:`polar_feature` warped to the box."""
        return self._warp(self.polar_feature(k, m, in_plane_rotation))

    # -- projections -----------------------------------------------------------------

    def _component_groups(self, num_components: int | None) -> dict[int, list[int]]:
        """Eigenvector indices of the leading components, grouped by ``k``."""
        r = self.result
        n_blocks = r.k_max * 2 if r.is_complex_projection else r.k_max
        total = n_blocks * r.eig_max
        n = total if num_components is None else min(int(num_components), total)
        groups: dict[int, list[int]] = {}
        # include_negative=False: a real decomposition's -k blocks are conjugates
        for k, m in r.get_top_n(n, include_negative=False):
            groups.setdefault(int(k), []).append(int(m))
        return groups

    def reconstruct(
        self,
        orientation: int,
        fourier_filter: int = 0,
        volume: int = 0,
        *,
        in_plane_rotation: float = 0.0,
        num_components: int | None = None,
        polar: bool = False,
    ) -> torch.Tensor:
        """One template projection from its leading components.

        Parameters
        ----------
        orientation, fourier_filter, volume : int
            The template row.
        in_plane_rotation : float
            In-plane rotation in degrees, applied analytically as a phase.
        num_components : int, optional
            Components by singular value (``None``: all stored).
        polar : bool
            Return the ``(A, R)`` polar image instead of the ``(kH, kW)`` warp.
        """
        return self.reconstruct_batch(
            [(orientation, fourier_filter, in_plane_rotation)],
            volume=volume,
            num_components=num_components,
            polar=polar,
        )[0]

    def reconstruct_batch(
        self,
        queries: Sequence[tuple[int, int, float]],
        *,
        volume: int = 0,
        num_components: int | None = None,
        polar: bool = False,
    ) -> torch.Tensor:
        """Template projections of ``(orientation, fourier_filter, rotation)`` queries.

        Returns ``(B, kH, kW)`` complex Cartesian images, or ``(B, A, R)`` polar ones
        when ``polar``; see :meth:`reconstruct` for the arguments.
        """
        r = self.result
        orientations = [int(q[0]) for q in queries]
        filters = [int(q[1]) for q in queries]
        for f in filters:
            if not 0 <= f < r.num_fourier_filters:
                raise ValueError(
                    f"fourier_filter {f} out of range [0, {r.num_fourier_filters})"
                )
        for o in orientations:
            if not 0 <= o < r.num_orientations:
                raise ValueError(
                    f"orientation {o} out of range [0, {r.num_orientations})"
                )
        if not 0 <= volume < r.num_volumes:
            raise ValueError(f"volume {volume} out of range [0, {r.num_volumes})")

        o_idx = torch.tensor(orientations, device=self.device)
        f_idx = torch.tensor(filters, device=self.device)
        shifts = torch.tensor(
            [float(q[2]) for q in queries], device=self.device, dtype=torch.float32
        )
        n_angle = r.num_angular_components
        out = torch.zeros(
            (len(queries), n_angle, r.num_radial_components),
            dtype=torch.complex64,
            device=self.device,
        )
        for k, ms in sorted(self._component_groups(num_components).items()):
            indices = np.column_stack([np.full(len(ms), k), ms])
            u, s, vh = r.get_svd_tensors(indices, self.device)
            radial = (u[volume, f_idx, o_idx, :] * s[None, :]) @ vh  # (B, R)
            term = (
                self._angular_phase_components(k, shifts)[:, :, None]
                * radial[:, None, :]
            )
            out += term
            # a real decomposition's -k block is the conjugate of the +k block (DC and
            # Nyquist are their own mirror)
            nyquist = n_angle % 2 == 0 and k == n_angle // 2
            if not r.is_complex_projection and k > 0 and not nyquist:
                out += term.conj()
        return out if polar else self._warp(out)

    # -- kernels ---------------------------------------------------------------------

    def kernels(self, cells: torch.Tensor | Iterable[Sequence[int]]) -> torch.Tensor:
        """``(n, kH, kW)`` complex64 kernels of ``(k, m)`` cells, in request order.

        Built on every call, grouped by ``k`` (the angular phase is shared and the
        radial vectors are gathered once per chunk).
        """
        self._require_real()
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
        indices = np.column_stack([np.full(len(ms), k), np.asarray(ms)])
        _, _, vh = self.result.get_svd_tensors(indices, self.device)  # (M, R)
        angular = self.angular_phase_component(k)  # (A,)
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
        if self.device.type != "cuda" or self.warp.mode != "constant":
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
        order = self.warp.order
        # the generic warp pads the periodic axis by 4 * order rows (the prefilter's
        # boundary error decays like |pole|**rows); mirror it exactly or the fast and
        # generic kernels differ at the 0/360 seam by ~1e-3
        wrap = (
            4 * order if (self.warp.wrap_angular_axis and tf.has_periodic_axis) else 0
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
        cval = float(self.warp.cval)
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
            a = self.angular_phase_component(k).to(torch.complex64)
            if wrap:
                a = torch.nn.functional.pad(
                    a[None, None], (wrap, wrap), mode="circular"
                )[0, 0]
            ang = (filt(a.real), filt(a.imag))  # (A',)
            sep["angular"][k] = ang
        indices = np.column_stack([np.full(len(ms), k), np.asarray(ms)])
        _, _, vh = self.result.get_svd_tensors(indices, self.device)  # (M, R)
        vh = vh.to(torch.complex64)
        if self.warp.preserve_energy and sep["jac"] is not None:
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

        See :func:`~panther_em.approximation.adjoint.unsupported_reason`.
        """
        self._require_real()
        return WarpAdjoint.build(self.transform, self.warp, self.device)

    def adjoint_bases(self, rect: Sequence[int]) -> AdjointBases:
        """Prefiltered polar vectors of ``rect`` for :attr:`adjoint` (built anew)."""
        adjoint = self.adjoint
        if adjoint is None:
            raise ValueError("this warp has no adjoint (see unsupported_reason)")
        key = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
        return build_bases(adjoint, self, key)

    # -- the generic warp ------------------------------------------------------------

    def to_cartesian(self, polar: torch.Tensor) -> torch.Tensor:
        """``(..., A, R)`` polar images warped to ``(..., kH, kW)`` under :attr:`warp`.

        Real images come back float32, complex ones complex64.
        """
        out = self._warp(polar)
        return out.to(torch.complex64 if out.is_complex() else torch.float32)

    @cached_property
    def support(self) -> torch.Tensor:
        """``(kH, kW)`` bool CPU support: where a warped image of ones exceeds 1/2."""
        kwargs = self.warp.cartesian_kwargs()
        kwargs["preserve_energy"] = False
        warped = self.transform.to_cartesian(
            torch.ones(self.transform.polar_shape), **kwargs
        )
        return warped > 0.5

    def release(self) -> None:
        """Drop the adjoint and the separable build's device state."""
        for name in ("adjoint", "_separable"):
            self.__dict__.pop(name, None)
