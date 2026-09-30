"""Hypothesis space of the search: defocus x Fourier filter x orientation.

The in-plane angle ``psi`` is never enumerated -- it is recovered by the inverse FFT
over angular frequency inside the statistics kernels -- so it is not part of the space
here, only of the *decoding* of a best index.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import torch

if TYPE_CHECKING:
    from panther_em.decomposition.result import DecompositionResult


class HypothesisIndex(NamedTuple):
    """Decoded components of a global hypothesis id (``-1`` passes through)."""

    defocus: torch.Tensor
    filter: torch.Tensor
    orientation: torch.Tensor


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

    @property
    def global_size(self) -> int:
        """``D * FF * O``."""
        return self.n_defocus * self.n_templates

    def encode(
        self, defocus: torch.Tensor, filter: torch.Tensor, orientation: torch.Tensor
    ) -> torch.Tensor:
        """Global ids from component indices (broadcasting, int64)."""
        d = torch.as_tensor(defocus, dtype=torch.long)
        f = torch.as_tensor(filter, dtype=torch.long)
        o = torch.as_tensor(orientation, dtype=torch.long)
        return (d * self.n_filters + f) * self.n_orientations + o

    def decode(self, global_id: torch.Tensor) -> HypothesisIndex:
        """Component indices of global ids; ``-1`` decodes to ``(-1, -1, -1)``."""
        g = torch.as_tensor(global_id, dtype=torch.long)
        valid = g >= 0
        safe = torch.where(valid, g, torch.zeros_like(g))
        o = safe % self.n_orientations
        rest = safe // self.n_orientations
        f = rest % self.n_filters
        d = rest // self.n_filters
        neg = torch.full_like(g, -1)
        return HypothesisIndex(
            torch.where(valid, d, neg),
            torch.where(valid, f, neg),
            torch.where(valid, o, neg),
        )

    def psi_degrees(
        self, n_psi: int, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        """See :func:`psi_degrees`."""
        return psi_degrees(n_psi, device)

    def angles(
        self, global_id: torch.Tensor, best_psi: torch.Tensor, n_psi: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(phi, theta, psi)`` in degrees for decoded hypotheses; NaN for id -1."""
        if self.phi_deg is None or self.theta_deg is None:
            raise ValueError("this HypothesisSpace carries no phi/theta tables")
        idx = self.decode(global_id)
        valid = idx.orientation >= 0
        o = torch.where(valid, idx.orientation, torch.zeros_like(idx.orientation))
        psi_tab = self.psi_degrees(n_psi, device=torch.as_tensor(best_psi).device)
        p = torch.as_tensor(best_psi, dtype=torch.long)
        p_safe = torch.where(p >= 0, p, torch.zeros_like(p))
        nan = torch.full(o.shape, float("nan"), dtype=torch.float32)
        phi = torch.where(valid, self.phi_deg.to(o.device)[o], nan.to(o.device))
        theta = torch.where(valid, self.theta_deg.to(o.device)[o], nan.to(o.device))
        psi = torch.where(p >= 0, psi_tab[p_safe], nan.to(p.device))
        return phi, theta, psi


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


@dataclass(frozen=True, eq=False)
class HypothesisSubset:
    """Restriction of a :class:`HypothesisSpace` to some templates/defocus planes.

    ``None`` on either axis means "all". Ids are kept sorted-unique on the CPU.
    """

    template_ids: torch.Tensor | None = None
    defocus_ids: torch.Tensor | None = None

    @classmethod
    def strided(
        cls, space: HypothesisSpace, stride: int, phase: int = 0
    ) -> HypothesisSubset:
        """Every ``stride``-th template row (a scout pass), all defocus planes."""
        if stride < 1:
            raise ValueError("stride must be >= 1")
        return cls(template_ids=torch.arange(phase, space.n_templates, stride))

    @classmethod
    def orientations(
        cls,
        space: HypothesisSpace,
        orientation_ids: torch.Tensor,
        filter_ids: torch.Tensor | None = None,
    ) -> HypothesisSubset:
        """Product of the given orientations with the given (default: all) filters."""
        o = torch.as_tensor(orientation_ids, dtype=torch.long).reshape(-1)
        f = (
            torch.arange(space.n_filters)
            if filter_ids is None
            else torch.as_tensor(filter_ids, dtype=torch.long).reshape(-1)
        )
        rows = (f[:, None] * space.n_orientations + o[None, :]).reshape(-1)
        return cls(template_ids=rows)

    @classmethod
    def from_mask(cls, space: HypothesisSpace, mask: torch.Tensor) -> HypothesisSubset:
        """Templates where an ``(FF, O)`` bool mask is True."""
        if tuple(mask.shape) != (space.n_filters, space.n_orientations):
            raise ValueError(
                "mask must have shape (FF, O) = "
                f"{(space.n_filters, space.n_orientations)}"
            )
        rows = torch.nonzero(mask.reshape(-1), as_tuple=False).squeeze(-1)
        return cls(template_ids=rows)

    @classmethod
    def defocus(
        cls, space: HypothesisSpace, defocus_ids: torch.Tensor
    ) -> HypothesisSubset:
        """A subset of the defocus planes, all templates."""
        return cls(defocus_ids=torch.as_tensor(defocus_ids, dtype=torch.long))

    def resolve(
        self, space: HypothesisSpace
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Validated ``(template_ids, defocus_ids)`` against ``space`` (None = all)."""
        return (
            _sorted_unique_ids(self.template_ids, space.n_templates, "template_ids"),
            _sorted_unique_ids(self.defocus_ids, space.n_defocus, "defocus_ids"),
        )


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
        ``d_local * N_sel + j0``: the batch's first virtual id.
    kernel_offset : int
        ``virtual_offset - epoch_base``: the offset actually handed to the fused
        kernel, kept small enough that ``(kernel_offset + N_b) * n_psi`` fits the
        kernel's 32-bit packed index.
    epoch : int
        Index of the epoch this batch belongs to (see :class:`HypothesisPlan`).
    """

    d_local: int
    j0: int
    j1: int
    virtual_offset: int
    kernel_offset: int
    epoch: int

    @property
    def size(self) -> int:
        """Hypotheses in the batch."""
        return self.j1 - self.j0


@dataclass(frozen=True)
class Epoch:
    """A run of batches sharing a base virtual offset for the packed index."""

    base: int
    batches: slice  # into HypothesisPlan.batches


@dataclass(frozen=True, eq=False)
class HypothesisPlan:
    """Enumeration of one stage's hypotheses into kernel-ready batches.

    Parameters
    ----------
    space : HypothesisSpace
        The full space.
    subset : HypothesisSubset, optional
        Restriction for this stage (``None`` = everything).
    hyp_batch : int
        Hypotheses per contraction.
    n_psi : int
        In-plane samples; sets the packed-index budget per epoch.
    packed_index_bits : int, optional
        Width of the kernel's packed ``(hyp, psi)`` index. Defaults to 32.
    """

    space: HypothesisSpace
    subset: HypothesisSubset | None
    hyp_batch: int
    n_psi: int
    packed_index_bits: int = 32
    template_ids: torch.Tensor | None = field(init=False, default=None)
    defocus_ids: torch.Tensor | None = field(init=False, default=None)
    batches: tuple[HypBatch, ...] = field(init=False, default=())
    epochs: tuple[Epoch, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        """Enumerate batches and epochs."""
        if self.hyp_batch < 1:
            raise ValueError("hyp_batch must be >= 1")
        if self.n_psi < 2:
            raise ValueError("n_psi must be >= 2")
        t_ids, d_ids = (
            (None, None) if self.subset is None else self.subset.resolve(self.space)
        )
        object.__setattr__(self, "template_ids", t_ids)
        object.__setattr__(self, "defocus_ids", d_ids)

        n_sel = self.n_selected
        limit = 1 << self.packed_index_bits
        batches: list[HypBatch] = []
        epochs: list[Epoch] = []
        epoch_base = 0
        epoch_start = 0
        for d_local in range(self.n_defocus_selected):
            for j0 in range(0, n_sel, self.hyp_batch):
                j1 = min(j0 + self.hyp_batch, n_sel)
                v = d_local * n_sel + j0
                if (v - epoch_base + (j1 - j0)) * self.n_psi >= limit:
                    if len(batches) > epoch_start:
                        epochs.append(
                            Epoch(epoch_base, slice(epoch_start, len(batches)))
                        )
                        epoch_start = len(batches)
                    epoch_base = v
                if (j1 - j0) * self.n_psi >= limit:
                    raise ValueError(
                        f"hyp_batch * n_psi = {(j1 - j0) * self.n_psi} exceeds the "
                        f"{self.packed_index_bits}-bit packed index; reduce hyp_batch"
                    )
                batches.append(
                    HypBatch(d_local, j0, j1, v, v - epoch_base, len(epochs))
                )
        epochs.append(Epoch(epoch_base, slice(epoch_start, len(batches))))
        object.__setattr__(self, "batches", tuple(batches))
        object.__setattr__(self, "epochs", tuple(epochs))

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
