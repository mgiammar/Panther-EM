"""Hypothesis space of the search: defocus x Fourier filter x orientation.

The in-plane angle ``psi`` is never enumerated -- it is recovered by the inverse FFT
over angular frequency inside the statistics kernels -- so it is not part of the space
here, only of the *decoding* of a best index.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from panther_em.decomposition.result import DecompositionResult


def psi_degrees(n_psi: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Uniformly spaced in-plane angles in degrees, ``(n_psi,)`` float32."""
    return 360.0 * torch.arange(n_psi, device=device, dtype=torch.float32) / n_psi


@dataclass(frozen=True, eq=False)
class HypothesisSpace:
    """The full ``(defocus, Fourier filter, orientation)`` space.

    Parameters
    ----------
    n_defocus, n_filters, n_orientations : int
        Axis sizes ``D``, ``FF``, ``O``.
    phi_deg, theta_deg : torch.Tensor, optional
        ``(O,)`` out-of-plane Euler angles (degrees) from the decomposition.
    defocus_values : torch.Tensor, optional
        ``(D,)`` caller-supplied labels for the filter planes (e.g. relative defocus
        in Angstrom).
    """

    n_defocus: int
    n_filters: int
    n_orientations: int
    phi_deg: torch.Tensor | None = None
    theta_deg: torch.Tensor | None = None
    defocus_values: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Validate axis sizes and table shapes."""
        for name in ("n_defocus", "n_filters", "n_orientations"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        for name, n in (
            ("phi_deg", self.n_orientations),
            ("theta_deg", self.n_orientations),
            ("defocus_values", self.n_defocus),
        ):
            t = getattr(self, name)
            if t is not None and tuple(t.shape) != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {tuple(t.shape)}")

    @classmethod
    def from_result(
        cls,
        result: DecompositionResult,
        n_defocus: int = 1,
        defocus_values: torch.Tensor | None = None,
    ) -> HypothesisSpace:
        """Build the space from a decomposition's filter/orientation axes."""
        phi = result.phi_values
        theta = result.theta_values
        return cls(
            n_defocus=int(n_defocus),
            n_filters=int(result.num_fourier_filters),
            n_orientations=int(result.num_orientations),
            phi_deg=None if phi is None else torch.as_tensor(phi, dtype=torch.float32),
            theta_deg=None
            if theta is None
            else torch.as_tensor(theta, dtype=torch.float32),
            defocus_values=None
            if defocus_values is None
            else torch.as_tensor(defocus_values, dtype=torch.float32),
        )

    @property
    def n_templates(self) -> int:
        """``FF * O``: template rows of the weight matrix."""
        return self.n_filters * self.n_orientations

    def decode(
        self, global_id: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(defocus, filter, orientation)`` of global ids; ``-1`` stays ``-1``."""
        g = torch.as_tensor(global_id, dtype=torch.long)
        valid = g >= 0
        safe = torch.where(valid, g, torch.zeros_like(g))
        o = safe % self.n_orientations
        rest = safe // self.n_orientations
        f = rest % self.n_filters
        d = rest // self.n_filters
        neg = torch.full_like(g, -1)
        return (
            torch.where(valid, d, neg),
            torch.where(valid, f, neg),
            torch.where(valid, o, neg),
        )


def _sorted_unique_ids(
    ids: torch.Tensor | None, upper: int, name: str
) -> torch.Tensor | None:
    if ids is None:
        return None
    t = torch.as_tensor(ids, dtype=torch.long).cpu().reshape(-1)
    if t.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if int(t.min()) < 0 or int(t.max()) >= upper:
        raise ValueError(
            f"{name} must lie in [0, {upper}), got range "
            f"[{int(t.min())}, {int(t.max())}]"
        )
    if not bool(torch.all(t[1:] > t[:-1])):
        t = torch.unique(t)  # sorted + unique
    return t


def template_rows_for(
    space: HypothesisSpace,
    orientation_ids: torch.Tensor,
    filter_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Template rows ``f * O + o`` of the given orientations x (default: all) filters."""
    o = torch.as_tensor(orientation_ids, dtype=torch.long).reshape(-1)
    f = (
        torch.arange(space.n_filters)
        if filter_ids is None
        else torch.as_tensor(filter_ids, dtype=torch.long).reshape(-1)
    )
    return (f[:, None] * space.n_orientations + o[None, :]).reshape(-1)


@dataclass(frozen=True)
class HypBatch:
    """One contiguous batch of virtual hypothesis ids for one defocus plane.

    Attributes
    ----------
    d_local : int
        Position of the defocus plane within the stage's selected planes.
    j0, j1 : int
        Half-open range of selected-template positions.
    virtual_offset : int
        ``d_local * N_sel + j0``: the batch's first virtual id, handed to the
        statistics kernels as the hypothesis offset of the batch.
    """

    d_local: int
    j0: int
    j1: int
    virtual_offset: int

    @property
    def size(self) -> int:
        """Hypotheses in the batch."""
        return self.j1 - self.j0


@dataclass(frozen=True, eq=False)
class HypothesisSet:
    """One stage's hypotheses, enumerated into kernel-ready batches.

    Virtual ids run ``d_local * N_sel + j`` over the selected defocus planes (outer)
    and selected template rows (inner); :meth:`decode` maps them back to global ids
    ``d * FF * O + template``.

    Parameters
    ----------
    space : HypothesisSpace
        The full space.
    template_ids : torch.Tensor, optional
        Rows of the ``FF * O`` weight matrix to search (``None`` = all); kept
        sorted-unique on the CPU.
    defocus_ids : torch.Tensor, optional
        Defocus planes to search (``None`` = all); sorted-unique on the CPU.
    hyp_batch : int
        Hypotheses per contraction.
    n_psi : int
        In-plane samples recovered per hypothesis.
    packed_index_bits : int, optional
        Width of the fused kernel's packed ``(hypothesis, psi)`` index. Every
        ``(virtual id, psi)`` of the stage must fit it, so
        ``n_virtual * n_psi < 2 ** packed_index_bits`` (8.4M hypotheses at
        ``n_psi = 512`` for the default 32); split the CTF planes across stages
        when a stage would exceed it. Defaults to 32.
    """

    space: HypothesisSpace
    template_ids: torch.Tensor | None
    defocus_ids: torch.Tensor | None
    hyp_batch: int
    n_psi: int
    packed_index_bits: int = 32

    @classmethod
    def create(
        cls,
        space: HypothesisSpace,
        template_ids: torch.Tensor | None = None,
        defocus_ids: torch.Tensor | None = None,
        *,
        hyp_batch: int,
        n_psi: int,
        packed_index_bits: int = 32,
    ) -> HypothesisSet:
        """Build a set from raw ids: any int tensor, any device, unsorted, repeated."""
        return cls(
            space,
            _sorted_unique_ids(template_ids, space.n_templates, "template_ids"),
            _sorted_unique_ids(defocus_ids, space.n_defocus, "defocus_ids"),
            int(hyp_batch),
            int(n_psi),
            int(packed_index_bits),
        )

    def __post_init__(self) -> None:
        """Validate (never coerce): ids sorted-unique CPU int64, budget respected."""
        if self.hyp_batch < 1:
            raise ValueError("hyp_batch must be >= 1")
        if self.n_psi < 2:
            raise ValueError("n_psi must be >= 2")
        for name, ids, upper in (
            ("template_ids", self.template_ids, self.space.n_templates),
            ("defocus_ids", self.defocus_ids, self.space.n_defocus),
        ):
            if ids is None:
                continue
            normalised = _sorted_unique_ids(ids, upper, name)
            if (
                normalised is None
                or ids.device.type != "cpu"
                or ids.dtype is not torch.long
                or not torch.equal(ids, normalised)
            ):
                raise ValueError(
                    f"{name} must be a sorted, unique CPU int64 tensor; build the set "
                    "with HypothesisSet.create() to normalise raw ids"
                )
        limit = 1 << self.packed_index_bits
        if self.n_virtual * self.n_psi >= limit:
            raise ValueError(
                f"{self.n_virtual} hypotheses x {self.n_psi} in-plane angles exceed "
                f"the kernel's {self.packed_index_bits}-bit packed index; split the "
                "CTF planes (or the orientations) across stages"
            )

    @cached_property
    def batches(self) -> tuple[HypBatch, ...]:
        """Kernel-ready batches: selected planes outer, template rows inner."""
        n_sel = self.n_selected
        return tuple(
            HypBatch(d_local, j0, min(j0 + self.hyp_batch, n_sel), d_local * n_sel + j0)
            for d_local in range(self.n_defocus_selected)
            for j0 in range(0, n_sel, self.hyp_batch)
        )

    # -- sizes ---------------------------------------------------------------

    @property
    def n_selected(self) -> int:
        """Selected template rows ``N_sel``."""
        return (
            self.space.n_templates
            if self.template_ids is None
            else int(self.template_ids.numel())
        )

    @property
    def n_defocus_selected(self) -> int:
        """Selected defocus planes ``D_sel``."""
        return (
            self.space.n_defocus
            if self.defocus_ids is None
            else int(self.defocus_ids.numel())
        )

    @property
    def n_virtual(self) -> int:
        """``D_sel * N_sel``: hypotheses the kernels enumerate."""
        return self.n_defocus_selected * self.n_selected

    @property
    def n_evaluated(self) -> int:
        """``(hypothesis, psi)`` pairs per pixel: ``n_virtual * n_psi``."""
        return self.n_virtual * self.n_psi

    @property
    def is_contiguous(self) -> bool:
        """Whether the selected templates are the full ``FF * O`` range."""
        return self.template_ids is None

    @property
    def signature(self) -> tuple:
        """Hashable identity for caches (graphs, prepared weights)."""
        t = (
            None
            if self.template_ids is None
            else (
                int(self.template_ids.numel()),
                int(self.template_ids[0]),
                int(self.template_ids[-1]),
                hash(self.template_ids.numpy().tobytes()),
            )
        )
        d = None if self.defocus_ids is None else tuple(self.defocus_ids.tolist())
        return (
            self.space.n_defocus,
            self.space.n_filters,
            self.space.n_orientations,
            t,
            d,
            self.hyp_batch,
            self.n_psi,
        )

    # -- decoding ------------------------------------------------------------

    def template_rows(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """``(N_sel,)`` rows of the ``FF * O`` weight matrix used by this plan."""
        if self.template_ids is None:
            return torch.arange(self.space.n_templates, device=device)
        return self.template_ids.to(device)

    def defocus_planes(self) -> tuple[int, ...]:
        """Global defocus plane indices, in ``d_local`` order."""
        if self.defocus_ids is None:
            return tuple(range(self.space.n_defocus))
        return tuple(int(x) for x in self.defocus_ids.tolist())

    def decode(self, best_virtual: torch.Tensor) -> torch.Tensor:
        """Map virtual ids to global ids (``-1`` passes through); device ops only."""
        v = torch.as_tensor(best_virtual, dtype=torch.long)
        valid = v >= 0
        safe = torch.where(valid, v, torch.zeros_like(v))
        n_sel = self.n_selected
        d_local = safe // n_sel
        j = safe % n_sel
        template = j if self.template_ids is None else self.template_ids.to(v.device)[j]
        d = (
            d_local
            if self.defocus_ids is None
            else self.defocus_ids.to(v.device)[d_local]
        )
        g = d * self.space.n_templates + template
        return torch.where(valid, g, torch.full_like(v, -1))


# --------------------------------------------------------------------------- #
# HEALPix orientation grids
# --------------------------------------------------------------------------- #


def healpix_orientations(nside: int) -> tuple[np.ndarray, np.ndarray]:
    """``(phi_deg, theta_deg)`` of every HEALPix pixel at ``nside``, NESTED order.

    Float32, in the argument order of ``PolarProjectionDecomposer``; orientation
    index == NESTED pixel id, which is what :func:`healpix_level_ids` relies on.
    """
    from torch_so3.hierarchical_s2_grid import (  # type: ignore[import-untyped]
        HierarchicalS2Grid,
    )

    theta, phi = HierarchicalS2Grid(nside, 1).get_level_orientations(0)
    return phi.astype(np.float32), theta.astype(np.float32)


def _unit_vectors(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    theta, phi = np.radians(theta_deg), np.radians(phi_deg)
    return np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1
    )


def healpix_ipix(
    space: HypothesisSpace, *, tol_deg: float = 1e-3, max_nside: int = 4096
) -> tuple[int, np.ndarray] | None:
    """``(nside, ipix)`` when the space's orientations are HEALPix pixel centers.

    The smallest ``nside`` whose NESTED pixel centers coincide with every orientation
    (within ``tol_deg``, all distinct) wins; ``None`` when there is none.
    """
    if space.phi_deg is None or space.theta_deg is None:
        return None
    from torch_so3.hierarchical_s2_grid import HierarchicalS2Grid

    phi = space.phi_deg.cpu().numpy().astype(np.float64)
    theta = space.theta_deg.cpu().numpy().astype(np.float64)
    vec = _unit_vectors(theta, phi)
    nside = 1
    while nside <= max_nside:
        grid = HierarchicalS2Grid(nside, 1)
        if grid.get_level(0).npix >= phi.size:
            ipix = grid.index_from_angle(theta, phi, level=0)
            if np.unique(ipix).size == ipix.size:
                t2, p2 = grid.angle_from_index(ipix, level=0)
                cos = np.clip((vec * _unit_vectors(t2, p2)).sum(-1), -1.0, 1.0)
                if np.degrees(np.arccos(cos)).max() <= tol_deg:
                    return nside, np.asarray(ipix, dtype=np.int64)
        nside *= 2
    return None


def healpix_level_ids(ipix: np.ndarray, level: int) -> torch.Tensor:
    """Orientation ids forming a sampling ``level`` HEALPix levels coarser."""
    if level < 0:
        raise ValueError("level must be >= 0")
    ipix = np.asarray(ipix, dtype=np.int64)
    keep = np.nonzero(ipix % (4**level) == 0)[0]
    return torch.from_numpy(keep.astype(np.int64))
