"""Rectangular tilings of ``(k, m)`` SVD feature space and its persistent store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

# (k_start, m_start, k_extent, m_extent)
Rectangle = tuple[int, int, int, int]


def _contract_region(
    y: torch.Tensor,  # (P, num_k, num_m)
    w: torch.Tensor,  # (N, num_k, num_m)
    conjugate: bool = True,
) -> torch.Tensor:  # (P, N, num_k)
    """Contract one rectangular block over the eigenvector axis ``m``."""
    # (num_k, num_m) -> (1, num_k, num_m)
    if y.ndim == 2:
        y = y.unsqueeze(0)

    # (num_k, num_m) -> (1, num_k, num_m)
    if w.ndim == 2:
        w = w.unsqueeze(0)

    w = w.conj() if conjugate else w
    y_b = y.permute(1, 0, 2)  # (num_k, P, num_m)
    w_b = w.permute(1, 2, 0)  # (num_k, num_m, N)

    return torch.bmm(y_b, w_b).permute(1, 2, 0)  # (num_k, P, N) -> (P, N, num_k)


@dataclass
class RectangularFeatureRegion:
    """Single rectangle in ``(k, m)`` component space with helpers for contraction.

    Attributes
    ----------
    k_start : int
        Starting angular frequency index (inclusive).
    k_stop : int
        Stopping angular frequency index (exclusive).
    m_start : int
        Starting eigenvector index (inclusive).
    m_stop : int
        Stopping eigenvector index (exclusive).
    indices : torch.Tensor
        Long tensor of shape ``(num_k * num_m,)`` mapping all selected ``(k, m)`` cells
        to their index in flattened feature dimension of length ``r``. Row-major order.
        Assigned by the owning :class:`FeatureTiling`.
    """

    k_start: int
    k_stop: int
    m_start: int
    m_stop: int
    indices: torch.Tensor | None = None  # (num_k * num_m,) long, set by FeatureTiling

    @property
    def num_k(self) -> int:
        """Number of angular frequency cells."""
        return self.k_stop - self.k_start

    @property
    def num_m(self) -> int:
        """Number of eigenvector cells."""
        return self.m_stop - self.m_start

    @property
    def num_features(self) -> int:
        """Total number of features (cells) in the rectangle."""
        return self.num_k * self.num_m

    @property
    def extent(self) -> tuple[int, int]:
        """Helper to get rectangle's extent."""
        return (self.num_k, self.num_m)

    def cells(self, device: torch.device | str) -> torch.Tensor:
        """``(num_features, 2)`` long tensor of the ``(k, m)`` cells, row-major.

        Parameters
        ----------
        device : torch.device or str
            Device for the built tensor.

        Returns
        -------
        torch.Tensor
            Long tensor of shape ``(num_features, 2)`` giving the global ``(k, m)``
            indices of every slot in the feature dimension.
        """
        k = torch.arange(self.k_start, self.k_stop, device=device)
        m = torch.arange(self.m_start, self.m_stop, device=device)

        return torch.stack(torch.meshgrid(k, m, indexing="ij"), dim=-1).reshape(-1, 2)

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract this region's ``(..., num_k, num_m)`` block from a flat feature axis.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor whose last axis is the feature dimension addressed by
            :attr:`indices`.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(..., num_k, num_m)``.
        """
        if self.indices is None:
            raise ValueError(
                "region has no feature indices; build it through a FeatureTiling."
            )
        idx = self.indices.to(tensor.device)
        block = tensor.index_select(-1, idx)  # (..., num_k * num_m)

        return block.reshape(*block.shape[:-1], self.num_k, self.num_m)

    @torch.no_grad()
    def accumulate_into(
        self,
        Y_flat: torch.Tensor,
        W_flat: torch.Tensor,
        out: torch.Tensor,
        conjugate: bool = True,
    ) -> torch.Tensor:
        r"""Accumulate ``C[:, :, k] += sum_m Y[:, k, m] * conj(W[:, k, m])``.

        Parameters
        ----------
        Y_flat : torch.Tensor
            Complex featurized image ``(P, r)``; ``P`` pixels, ``r`` the feature axis.
        W_flat : torch.Tensor
            Complex contraction weights ``(N, r)`` in the same feature layout.
        out : torch.Tensor
            Complex output spectrum ``(P, N, n_freq)`` accumulated into in place; the
            region writes the contiguous slice ``[..., k_start:k_stop]``.
        conjugate : bool, optional
            Apply the matched-filter conjugation on ``W`` (``W = U * S`` is stored
            un-conjugated). Defaults to ``True``.

        Returns
        -------
        torch.Tensor
            The same ``out`` tensor, accumulated in place (returned for chaining).
        """
        y = self.gather(Y_flat)
        w = self.gather(W_flat)
        out[:, :, self.k_start : self.k_stop] += _contract_region(y, w, conjugate)

        return out


class FeatureTiling:
    """Group of feature regions which tile some portion of ``(k, m)`` space.

    Attributes
    ----------
    regions : list[RectangularFeatureRegion]
        The rectangles in the tiling, in the order they occupy the feature dimension.
    k_indices : torch.Tensor
        Long tensor of shape ``(num_features,)`` giving the global angular frequency
        index of every slot in the feature dimension.
    m_indices : torch.Tensor
        Long tensor of shape ``(num_features,)`` giving the global eigenvector index of
        every slot in the feature dimension.
    is_compacted : bool
        ``True`` when the tiling defines a compacted layout (sub-regions selected)
        and ``False`` when the tiling addresses an uncropped ``K * num_m`` grid. Will
        generally be ``False`` (i.e. should select sub-regions of feature space).

    Methods
    -------
    from_extents(extents, device, validate=True) : FeatureTiling
        Build a tiling over a compacted feature axis of length ``num_features``. Should
        generally be the method use to construct a tiling.
    on_full_grid(extents, num_m, device, validate=True) : FeatureTiling
        Build a tiling addressing an uncropped feature axis of length ``K * num_m``.
        Avoid for large feature spaces typical for SVD-2DTM.
    validate_disjoint() : None
        Raises an error if any ``(k, m)`` cell is covered by more than one region. To
        avoid double counting features.
    run(Y_flat, W_flat, n_freq, out=None) : torch.Tensor
        Accumulate every region into the dense spectrum ``out[P, N, n_freq]`` where the
        last dimension is the frequency dimension corresponding to ``k`` of each
        feature.
    new_slots(other) : torch.Tensor
        Returns the slots of *this* tiling whose cells are absent from ``other``.
    new_cells(other) : torch.Tensor
        Returns the ``(n_new, 2)`` ``(k, m)`` cells this tiling needs that ``other``
        lacks.
    gather_from(other) : tuple[torch.Tensor, torch.Tensor]
        Returns the index pair re-ordering ``other``'s feature axis into this tiling's
        layout. The returned ``src`` and ``dst`` are long tensors of positions in
        ``other`` and *this* tiling's feature dimension, respectively. Used for copying
        already computed features into a new tiling layout.
    """

    regions: list[RectangularFeatureRegion]
    # Global (k, m) of every slot in the feature dimension, in the same sequential
    # row-major order as the regions.
    k_indices: torch.Tensor  # (num_features,)
    m_indices: torch.Tensor  # (num_features,)
    is_compacted: bool  # False for an uncropped `K * num_m` layout

    def __init__(
        self,
        regions: list[RectangularFeatureRegion],
        device: torch.device | str,
        num_m: int | None = None,
        validate: bool = True,
    ) -> None:
        """Build a tiling; prefer :meth:`from_extents` / :meth:`on_full_grid`.

        Parameters
        ----------
        regions : list[RectangularFeatureRegion]
            Cell-disjoint rectangles, in the order they occupy the feature dimension.
        device : torch.device or str
            Device for the built index tensors.
        num_m : int, optional
            Eigenvector-axis stride of the target feature layout. When omitted the
            tiling defines its own compacted layout (regions packed back-to-back).
        validate : bool, optional
            Reject overlapping rectangles. Defaults to ``True``.
        """
        if len(regions) == 0:
            raise ValueError("FeatureTiling must have at least one region.")

        self.regions = list(regions)

        cells = torch.cat(
            [region.cells(device=device) for region in self.regions], dim=0
        )
        cells = cells.to(torch.long)
        self.k_indices = cells[:, 0]
        self.m_indices = cells[:, 1]

        if validate:
            self.validate_disjoint()

        # One (k, m) -> slot mapping for the whole tiling, sliced out to the regions
        self.is_compacted = num_m is None
        if self.is_compacted:
            indices = torch.arange(self.num_features, dtype=torch.long, device=device)
        else:
            assert num_m is not None
            indices = cells[:, 0].to(torch.long) * int(num_m) + cells[:, 1]
            indices = indices.to(torch.long).to(device)

        # Set the region indices on the flattened feature dimension
        offset = 0
        for region in self.regions:
            region.indices = indices[offset : offset + region.num_features]
            offset += region.num_features

    @classmethod
    def from_extents(
        cls,
        extents: Sequence[Rectangle],
        device: torch.device | str,
        validate: bool = True,
    ) -> FeatureTiling:
        """Tiling over a compacted feature axis of length :attr:`num_features`.

        Parameters
        ----------
        extents : Sequence[Rectangle]
            ``(k_start, m_start, k_extent, m_extent)`` blocks of feature space. This is
            the form an external feature selector emits (rank/singular-value driven;
            out of scope for this module).
        device : torch.device or str
            Device for the built index tensors.
        validate : bool, optional
            Reject overlapping rectangles through raise of ValueError. Defaults to
            ``True``.

        Returns
        -------
        FeatureTiling
        """
        return cls(cls._regions_from_extents(extents), device=device, validate=validate)

    @classmethod
    def on_full_grid(
        cls,
        extents: Sequence[Rectangle],
        num_m: int,
        device: torch.device | str,
        validate: bool = True,
    ) -> FeatureTiling:
        """Tiling addressing an *uncropped* feature axis of length ``K * num_m``.

        Use this to contract tensors that were built over the whole ``(k, m)`` grid
        rather than compacted down to the selected rectangles.

        Parameters
        ----------
        extents : Sequence[Rectangle]
            ``(k_start, m_start, k_extent, m_extent)`` blocks of feature space.
        num_m : int
            Eigenvector-axis size of the full grid.
        device : torch.device or str
            Device for the built index tensors.
        validate : bool, optional
            Reject overlapping rectangles. Defaults to ``True``.

        Returns
        -------
        FeatureTiling
        """
        return cls(
            cls._regions_from_extents(extents),
            device=device,
            num_m=num_m,
            validate=validate,
        )

    @staticmethod
    def _regions_from_extents(
        extents: Sequence[Rectangle],
    ) -> list[RectangularFeatureRegion]:
        return [
            RectangularFeatureRegion(
                k_start=int(k0),
                k_stop=int(k0) + int(k_ext),
                m_start=int(m0),
                m_stop=int(m0) + int(m_ext),
            )
            for (k0, m0, k_ext, m_ext) in extents
        ]

    @property
    def layout(self) -> torch.Tensor:
        """``(num_features, 2)`` long tensor of the global ``(k, m)`` of every slot."""
        return torch.stack((self.k_indices, self.m_indices), dim=-1)

    @property
    def num_features(self) -> int:
        """Total number of SVD features across all tiled regions."""
        return sum(region.num_features for region in self.regions)

    @property
    def k_stop(self) -> int:
        """Smallest angular-frequency axis length that can hold this tiling."""
        return max(region.k_stop for region in self.regions)

    @property
    def device(self) -> torch.device:
        """Device of the tiling's index tensors."""
        return self.k_indices.device

    def validate_disjoint(self) -> None:
        """Raise if any ``(k, m)`` cell is covered by more than one region."""
        keys = self._keys(self.k_indices, self.m_indices)
        if int(torch.unique(keys).numel()) != int(keys.numel()):
            raise ValueError(
                "Regions overlap; rectangles within a tiling must be pairwise "
                "cell-disjoint or the contraction would double count."
            )

    # -- contraction -------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        Y_flat: torch.Tensor,
        W_flat: torch.Tensor,
        n_freq: int,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Accumulate every region into the dense spectrum ``C[P, N, n_freq]``.

        Parameters
        ----------
        Y_flat : torch.Tensor
            Featurized image ``(P, r)`` in this tiling's feature layout.
        W_flat : torch.Tensor
            Contraction weights ``(N, r)`` in the same feature layout.
        n_freq : int
            Size of the output angular-frequency axis (``result.k_max``). Ignored when
            ``out`` is supplied.
        out : torch.Tensor, optional
            Pre-allocated ``(P, N, n_freq)`` complex accumulator. A fresh zero tensor
            is allocated when omitted.

        Returns
        -------
        torch.Tensor
            Complex angular-frequency spectrum ``C`` of shape ``(P, N, n_freq)``.
            Recover the real ``\psi``-resolved correlogram with an ``irfft`` over the
            last axis.
        """
        if out is None:
            out = torch.zeros(
                (Y_flat.shape[0], W_flat.shape[0], int(n_freq)),
                dtype=Y_flat.dtype,
                device=Y_flat.device,
            )

        for region in self.regions:
            region.accumulate_into(Y_flat, W_flat, out, conjugate=True)

        return out

    # -- incremental search: diffing and re-layout -------------------------

    @staticmethod
    def _keys(
        k_indices: torch.Tensor, m_indices: torch.Tensor, base: int = 1 << 20
    ) -> torch.Tensor:
        """Collapse global ``(k, m)`` pairs to scalar keys for set operations."""
        return k_indices.to(torch.long) * base + m_indices.to(torch.long)

    def _match(self, other: FeatureTiling) -> tuple[torch.Tensor, torch.Tensor]:
        """Locate this tiling's cells inside ``other``.

        Returns
        -------
        found : torch.Tensor
            Bool tensor ``(num_features,)``; ``found[i]`` is ``True`` when slot ``i``
            of this tiling is already present in ``other``.
        src : torch.Tensor
            Long tensor of the matching slot in ``other``, one entry per ``True`` in
            ``found`` (same order).
        """
        s_keys = self._keys(self.k_indices, self.m_indices).cpu()
        o_keys = self._keys(other.k_indices, other.m_indices).cpu()

        order = torch.argsort(o_keys)
        sorted_keys = o_keys[order]
        pos = torch.searchsorted(sorted_keys, s_keys).clamp_(
            max=sorted_keys.numel() - 1
        )
        found = sorted_keys[pos] == s_keys

        return found, order[pos[found]]

    def new_slots(self, other: FeatureTiling) -> torch.Tensor:
        """Slots of *this* tiling whose cells are absent from ``other``.

        Returns
        -------
        torch.Tensor
            Long tensor of positions in this tiling's feature dimension that must be
            filled by a fresh featurization.
        """
        found, _ = self._match(other)

        return torch.nonzero(~found, as_tuple=False).squeeze(-1)

    def new_cells(self, other: FeatureTiling) -> torch.Tensor:
        """``(n_new, 2)`` ``(k, m)`` cells this tiling needs that ``other`` lacks."""
        return self.layout[self.new_slots(other)]

    def gather_from(self, other: FeatureTiling) -> tuple[torch.Tensor, torch.Tensor]:
        """Index pair re-ordering ``other``'s feature axis into this tiling's layout.

        Returns
        -------
        src : torch.Tensor
            Long positions in ``other``'s feature dimension.
        dst : torch.Tensor
            Long positions in *this* tiling's feature dimension.

        Notes
        -----
        Reuse is a single vectorised copy; the remaining slots are
        :meth:`new_slots`::

            Y_new = torch.empty(P, new.num_features, dtype=..., device=...)
            src, dst = new.gather_from(old)
            Y_new[:, dst] = Y_old[:, src]
            Y_new[:, new.new_slots(old)] = featurize(new.new_cells(old))
        """
        found, src = self._match(other)
        dst = torch.nonzero(found, as_tuple=False).squeeze(-1)

        return src, dst


class FeaturizedImageStore:
    r"""Featurized image ``Y`` held in the layout of the current :class:`FeatureTiling`.

    Parameters
    ----------
    num_pixels : int
        Number of valid cross-correlation output pixels ``P`` (``out_h * out_w``).
    device : torch.device or str
        Device the featurized image is held on.
    dtype : torch.dtype, optional
        Complex dtype of the stored features. Defaults to ``torch.complex64``.

    Attributes
    ----------
    tiling : FeatureTiling or None
        The layout ``Y`` is currently stored in. ``None`` before the first
        :meth:`relayout`.
    Y : torch.Tensor or None
        Complex ``(P, tiling.num_features)`` features in ``tiling``'s slot order.
    """

    def __init__(
        self,
        num_pixels: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        self.num_pixels = int(num_pixels)
        self.device = torch.device(device)
        self.dtype = dtype
        self.tiling: FeatureTiling | None = None
        self.Y: torch.Tensor | None = None

    def missing_cells(self, tiling: FeatureTiling) -> torch.Tensor:
        """``(n_new, 2)`` cells of ``tiling`` not already featurized in the store."""
        if self.tiling is None:
            return tiling.layout.clone()

        return tiling.new_cells(self.tiling)

    def relayout(self, tiling: FeatureTiling, new_features: torch.Tensor) -> None:
        """Adopt ``tiling``'s layout, reusing carried-over columns.

        Parameters
        ----------
        tiling : FeatureTiling
            The layout to move to.
        new_features : torch.Tensor
            Complex features of shape ``(n_new, P)`` for exactly the cells returned by
            ``missing_cells(tiling)``, in that order (the channel-major layout of
            :func:`panther_em.inference.search.utils.featurize_cells`).
        """
        if not tiling.is_compacted:
            raise ValueError(
                "FeaturizedImageStore holds a compacted feature axis; build the "
                "tiling with FeatureTiling.from_extents, not on_full_grid."
            )

        Y = torch.empty(
            (self.num_pixels, tiling.num_features),
            dtype=self.dtype,
            device=self.device,
        )

        # Set carried over features from old tiling, if any.
        if self.tiling is None or self.Y is None:
            slots = torch.arange(tiling.num_features, dtype=torch.long)
        else:
            src, dst = tiling.gather_from(self.tiling)
            Y[:, dst.to(self.device)] = self.Y[:, src.to(self.device)]
            slots = tiling.new_slots(self.tiling)

        if int(slots.numel()) != int(new_features.shape[0]):
            raise ValueError(
                f"new_features has {new_features.shape[0]} rows but the re-layout "
                f"needs {int(slots.numel())} freshly featurized cells."
            )

        # Fill the new features based on slot ordering.
        if slots.numel():
            Y[:, slots.to(self.device)] = new_features.transpose(0, 1).to(
                self.device, self.dtype
            )

        self.Y = Y
        self.tiling = tiling

    def image_view(self, pixel_index: torch.Tensor | None = None) -> torch.Tensor:
        """Return the featurized image, optionally row-selected to a pixel subset.

        Parameters
        ----------
        pixel_index : torch.Tensor, optional
            Long indices into the ``P`` pixel axis (e.g. a shrinking follow-up mask or
            batch indices). When omitted the full ``(P, num_features)`` image is
            returned.

        Returns
        -------
        torch.Tensor
            Complex ``(P_sel, num_features)`` view in the store's current layout.
        """
        if self.Y is None:
            raise ValueError("store is empty; call relayout() first.")
        if pixel_index is None:
            return self.Y

        return self.Y[pixel_index.to(self.Y.device)]
