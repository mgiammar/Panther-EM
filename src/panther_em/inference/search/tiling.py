"""Rectangular tilings of ``(k, m)`` SVD feature space and its persistent store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

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


# Contraction precisions. "fp32" is cuBLAS cgemm; "tf32" the same with TF32 tensor
# cores enabled around the call; "fp16" the real-valued "4M" formulation below on
# FP16 tensor cores (~4x the cgemm rate on Ada, ~4e-4 relative error).
Precision = Literal["fp32", "tf32", "fp16"]


def _weights_fp16_4m(w_conj: torch.Tensor) -> torch.Tensor:
    r"""Interleave a pre-conjugated ``(N, num_k, num_m)`` block for the fp16 4M GEMM.

    The complex product ``C = Y @ w`` (``w`` already conjugated) is computed as ONE real
    GEMM per ``k``:  ``[Yr | Yi] (P, 2 num_m) @ B (2 num_m, 2N)`` with the columns of
    ``B`` interleaved so that ``out[:, 2n] = Cr[:, n]`` and ``out[:, 2n+1] = Ci[:, n]``::

        Cr = Yr wr - Yi wi      ->  B[:num_m, 2n] =  wr[:, n],  B[num_m:, 2n]   = -wi[:, n]
        Ci = Yr wi + Yi wr      ->  B[:num_m, 2n+1] = wi[:, n], B[num_m:, 2n+1] =  wr[:, n]

    The real ``(num_k, P, 2N)`` output is therefore bit-identical in memory to a
    ``(num_k, P, N)`` complex tensor -- exactly the layout the fused reduce kernels
    consume, with no transposes or copies.

    Parameters
    ----------
    w_conj : torch.Tensor
        Pre-conjugated complex weights ``(N, num_k, num_m)``.

    Returns
    -------
    torch.Tensor
        float16 ``(num_k, 2 num_m, 2N)``, contiguous.
    """
    n, num_k, num_m = w_conj.shape
    wk = w_conj.permute(1, 2, 0)  # (num_k, num_m, N)
    b = torch.empty((num_k, 2 * num_m, 2 * n), dtype=torch.float16, device=w_conj.device)
    b[:, :num_m, 0::2] = wk.real
    b[:, num_m:, 0::2] = -wk.imag
    b[:, :num_m, 1::2] = wk.imag
    b[:, num_m:, 1::2] = wk.real
    return b


def _features_fp16_4m(y: torch.Tensor) -> torch.Tensor:
    """``(P, num_k, num_m)`` complex features -> fp16 ``(num_k, P, 2 num_m)`` ``[Yr | Yi]``."""
    yk = y.permute(1, 0, 2)  # (num_k, P, num_m)
    return torch.cat((yk.real, yk.imag), dim=-1).to(torch.float16)


def _contract_region_fp16(
    a16: torch.Tensor,  # (num_k, P, 2 num_m) fp16, from _features_fp16_4m
    b16: torch.Tensor,  # (num_k, 2 num_m, 2N) fp16, from _weights_fp16_4m
    accumulate_fp32: bool,
) -> torch.Tensor:  # (P, N, num_k) complex view (complex32, or complex64 if accumulate_fp32)
    """fp16 tensor-core contraction; output viewed as the ``(P, N, num_k)`` spectrum."""
    num_k, p, _ = a16.shape
    n = b16.shape[-1] // 2
    if accumulate_fp32:
        out = torch.bmm(a16, b16, out_dtype=torch.float32)  # (num_k, P, 2N) fp32
    else:
        out = torch.bmm(a16, b16)  # (num_k, P, 2N) fp16
    return torch.view_as_complex(out.view(num_k, p, n, 2)).permute(1, 2, 0)


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
    # (start, stop) when `indices` is a contiguous arange (compacted layouts), so
    # `gather` can slice instead of index_select. Set by FeatureTiling.
    slice_bounds: tuple[int, int] | None = None

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
        if self.slice_bounds is not None:
            # Compacted layouts give every region a contiguous slot range: slice
            # instead of gathering (a view, no copy).
            start, stop = self.slice_bounds
            block = tensor[..., start:stop]
        else:
            idx = self.indices.to(tensor.device)
            block = tensor.index_select(-1, idx)  # (..., num_k * num_m)

        return block.reshape(*block.shape[:-1], self.num_k, self.num_m)

    def gather_conjugated_weights(self, W_flat: torch.Tensor) -> torch.Tensor:
        """Pre-gather and conjugate this region's ``W`` block.

        Parameters
        ----------
        W_flat : torch.Tensor
            Complex contraction weights ``(N, r)`` in this region's feature layout.

        Returns
        -------
        torch.Tensor
            Pre-conjugated ``(N, num_k, num_m)`` block, ready to pass as
            ``prepared_w`` to :meth:`accumulate_into`.
        """
        return self.gather(W_flat).conj()

    @torch.no_grad()
    def accumulate_into(
        self,
        Y_flat: torch.Tensor,
        W_flat: torch.Tensor | None,
        out: torch.Tensor,
        conjugate: bool = True,
        prepared_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Accumulate ``C[:, :, k] += sum_m Y[:, k, m] * conj(W[:, k, m])``.

        Parameters
        ----------
        Y_flat : torch.Tensor
            Complex featurized image ``(P, r)``; ``P`` pixels, ``r`` the feature axis.
        W_flat : torch.Tensor or None
            Complex contraction weights ``(N, r)`` in the same feature layout. May be
            ``None`` when ``prepared_w`` is supplied instead.
        out : torch.Tensor
            Complex output spectrum ``(P, N, n_freq)`` accumulated into in place; the
            region writes the contiguous slice ``[..., k_start:k_stop]``.
        conjugate : bool, optional
            Apply the matched-filter conjugation on ``W`` (``W = U * S`` is stored
            un-conjugated). Ignored when ``prepared_w`` is supplied (it is already
            conjugated). Defaults to ``True``.
        prepared_w : torch.Tensor, optional
            Pre-gathered, pre-conjugated weights from :meth:`gather_conjugated_weights`
            / :meth:`FeatureTiling.prepare_weights`, reused across many calls.

        Returns
        -------
        torch.Tensor
            The same ``out`` tensor, accumulated in place (returned for chaining).
        """
        y = self.gather(Y_flat)
        if prepared_w is not None:
            w, conjugate = prepared_w, False
        else:
            assert W_flat is not None
            w = self.gather(W_flat)
        out[:, :, self.k_start : self.k_stop] += _contract_region(y, w, conjugate)

        return out


@dataclass
class KInterval:
    """A run ``[k_start, k_stop)`` of angular frequencies covered by a fixed region set.

    See :meth:`FeatureTiling._build_k_intervals`. ``indices`` (``(num_k * num_m,)``,
    row-major in ``(k, m)``) addresses the concatenated ``m``-ranges of the covering
    regions in the tiling's flat feature layout; ``slice_bounds`` is set when that is a
    contiguous run (so gathering is a view).
    """

    k_start: int
    k_stop: int
    num_m: int
    indices: torch.Tensor
    slice_bounds: tuple[int, int] | None = None

    @property
    def num_k(self) -> int:
        """Number of angular frequencies in the run."""
        return self.k_stop - self.k_start

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """``(..., num_k, num_m)`` block of a tensor whose last axis is the feature axis."""
        if self.slice_bounds is not None:
            start, stop = self.slice_bounds
            block = tensor[..., start:stop]
        else:
            block = tensor.index_select(-1, self.indices.to(tensor.device))
        return block.reshape(*block.shape[:-1], self.num_k, self.num_m)


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
    prepare_weights(W_flat, precision) : list[torch.Tensor]
        Pre-gather and conjugate every k-interval's ``W`` block once, for reuse across
        many :meth:`run` calls that share the same hypothesis batch.
    prepare_features(Y_flat, precision) : list[torch.Tensor]
        Per-k-interval blocks of the featurized image, once per pixel batch.
    """

    regions: list[RectangularFeatureRegion]
    # Global (k, m) of every slot in the feature dimension, in the same sequential
    # row-major order as the regions.
    k_indices: torch.Tensor  # (num_features,)
    m_indices: torch.Tensor  # (num_features,)
    is_compacted: bool  # False for an uncropped `K * num_m` layout
    # Contraction plan: k-runs with a constant covering-region set (see
    # _build_k_intervals); prepare_weights / prepare_features / run work per interval.
    k_intervals: list[KInterval]

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
            if self.is_compacted:
                region.slice_bounds = (offset, offset + region.num_features)
            offset += region.num_features

        self.k_intervals = self._build_k_intervals(device)

    def _build_k_intervals(self, device: torch.device | str) -> list[KInterval]:
        """Partition the ``k`` axis into runs with a constant set of covering regions.

        Within one run every covering region contributes a disjoint ``m``-range (the
        tiling is cell-disjoint), so the contraction over the run is ONE GEMM whose
        inner dimension is the concatenation of those ``m``-ranges. That turns any
        multi-rectangle tiling into a handful of independent GEMMs, each written
        straight into its own ``k``-slab of the spectrum -- no accumulation pass, no
        zero-fill except for ``k`` no rectangle covers. A single full-height rectangle
        yields exactly one interval, the classic fast path.
        """
        bounds = sorted({r.k_start for r in self.regions} | {r.k_stop for r in self.regions})
        intervals: list[KInterval] = []
        for k_start, k_stop in zip(bounds[:-1], bounds[1:], strict=True):
            members = [
                r for r in self.regions if r.k_start <= k_start and r.k_stop >= k_stop
            ]
            if not members:
                continue
            num_k = k_stop - k_start
            num_m = sum(r.num_m for r in members)
            # slot(region, k, m) = region_offset + (k - k_start_r) * num_m_r + (m - m_start_r)
            rows = []
            for k in range(k_start, k_stop):
                parts = []
                for r in members:
                    assert r.indices is not None
                    base = (k - r.k_start) * r.num_m
                    parts.append(r.indices[base : base + r.num_m])
                rows.append(torch.cat(parts))
            indices = torch.stack(rows).reshape(-1).to(device=device, dtype=torch.long)
            first = int(indices[0])
            is_slice = bool(
                torch.equal(
                    indices,
                    torch.arange(
                        first, first + indices.numel(), device=indices.device
                    ),
                )
            )
            intervals.append(
                KInterval(
                    k_start=k_start,
                    k_stop=k_stop,
                    num_m=num_m,
                    indices=indices,
                    slice_bounds=(first, first + int(indices.numel())) if is_slice else None,
                )
            )
        return intervals

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

    def prepare_weights(
        self, W_flat: torch.Tensor, precision: Precision = "fp32"
    ) -> list[torch.Tensor]:
        """Pre-gather and conjugate every region's ``W`` block once.

        Parameters
        ----------
        W_flat : torch.Tensor
            Contraction weights ``(N, r)`` in this tiling's feature layout.
        precision : {"fp32", "tf32", "fp16"}, optional
            Contraction precision :meth:`run` will be called with. For ``"fp16"`` the
            blocks are returned pre-interleaved for the real 4M GEMM (see
            :func:`_weights_fp16_4m`), ``(num_k, 2 num_m, 2N)`` float16.

        Returns
        -------
        list[torch.Tensor]
            One block per :attr:`k_intervals` entry: pre-conjugated ``(N, num_k, num_m)``
            complex for fp32/tf32 (``num_m`` = the interval's concatenated eigenvector
            count), or the fp16 4M layout ``(num_k, 2 num_m, 2N)``.
        """
        blocks = [interval.gather(W_flat).conj() for interval in self.k_intervals]
        if precision == "fp16":
            return [_weights_fp16_4m(block) for block in blocks]
        return blocks

    def prepare_features(
        self, Y_flat: torch.Tensor, precision: Precision = "fp32"
    ) -> list[torch.Tensor]:
        """Slice (and for fp16, convert) every region's block of the featurized image.

        Doing this once per pixel batch, outside the hypothesis loop, means the
        per-hypothesis-batch work in :meth:`run` is exactly one ``bmm`` per region.

        Parameters
        ----------
        Y_flat : torch.Tensor
            Featurized image ``(P, r)`` in this tiling's feature layout.
        precision : {"fp32", "tf32", "fp16"}, optional
            See :meth:`prepare_weights`.

        Returns
        -------
        list[torch.Tensor]
            Per :attr:`k_intervals` entry: ``(P, num_k, num_m)`` complex (fp32/tf32) or
            fp16 ``(num_k, P, 2 num_m)`` ``[Yr | Yi]`` (fp16).
        """
        blocks = [interval.gather(Y_flat) for interval in self.k_intervals]
        if precision == "fp16":
            return [_features_fp16_4m(block) for block in blocks]
        return blocks

    @torch.no_grad()
    def run(
        self,
        Y_flat: torch.Tensor | None,
        W_flat: torch.Tensor | None,
        n_freq: int,
        out: torch.Tensor | None = None,
        prepared_weights: list[torch.Tensor] | None = None,
        prepared_features: list[torch.Tensor] | None = None,
        precision: Precision = "fp32",
    ) -> torch.Tensor:
        r"""Accumulate every region into the dense spectrum ``C[P, N, n_freq]``.

        Parameters
        ----------
        Y_flat : torch.Tensor or None
            Featurized image ``(P, r)`` in this tiling's feature layout. May be ``None``
            when ``prepared_features`` is supplied instead.
        W_flat : torch.Tensor or None
            Contraction weights ``(N, r)`` in the same feature layout. May be ``None``
            when ``prepared_weights`` is supplied instead.
        n_freq : int
            Size of the output angular-frequency axis. Should be ``self.k_stop`` (the
            spectrum carries no content above it, downstream ``irfft`` zero-pads  rest),
            which also enables the single-region assignment fast path. Ignored when
            ``out`` is supplied.
        out : torch.Tensor, optional
            Pre-allocated ``(P, N, n_freq)`` complex accumulator. A fresh zero tensor
            is allocated when omitted.
        prepared_weights : list[torch.Tensor], optional
            Output of :meth:`prepare_weights` (with the same ``precision``), reused
            across many calls that share the same hypothesis batch instead of
            re-gathering/re-conjugating ``W_flat`` here. Takes precedence over
            ``W_flat`` when supplied. Required for ``precision="fp16"``.
        prepared_features : list[torch.Tensor], optional
            Output of :meth:`prepare_features` (same ``precision``), reused across the
            hypothesis batches of one pixel batch. Takes precedence over ``Y_flat``.
        precision : {"fp32", "tf32", "fp16"}, optional
            Contraction precision. ``"fp16"`` runs the real 4M GEMM on FP16 tensor
            cores; the single-full-rectangle fast path then returns a **complex32**
            spectrum (the fused reduce kernel consumes it directly), while the
            multi-region path accumulates in fp32 and returns complex64.

        Returns
        -------
        torch.Tensor
            Complex angular-frequency spectrum ``C`` of shape ``(P, N, n_freq)``.
            Recover the real ``\psi``-resolved correlogram with an ``irfft`` over the
            last axis.
        """
        if precision == "fp16":
            return self._run_fp16(Y_flat, n_freq, out, prepared_weights, prepared_features)
        if precision == "tf32":
            prev = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
            try:
                return self.run(
                    Y_flat, W_flat, n_freq, out, prepared_weights, prepared_features
                )
            finally:
                torch.backends.cuda.matmul.allow_tf32 = prev

        if Y_flat is None and prepared_features is None:
            raise ValueError("run() needs Y_flat or prepared_features")

        if out is not None:
            # Caller-provided accumulator: original per-region accumulate-into semantics.
            for region in self.regions:
                region.accumulate_into(Y_flat, W_flat, out, conjugate=True)  # type: ignore[arg-type]
            return out

        if prepared_weights is None:
            assert W_flat is not None
            prepared_weights = self.prepare_weights(W_flat)
        if prepared_features is None:
            assert Y_flat is not None
            prepared_features = self.prepare_features(Y_flat)

        # Fast path: one interval spanning the whole frequency axis (e.g. a single
        # full-height rectangle) -- the bmm output IS the spectrum, no fill or copy.
        iv0 = self.k_intervals[0]
        if len(self.k_intervals) == 1 and iv0.k_start == 0 and iv0.k_stop == int(n_freq):
            return _contract_region(prepared_features[0], prepared_weights[0], False)

        # General: frequency-major (n_freq, P, N) buffer handed out as the (P, N, n_freq)
        # permuted view (the layout the fused reduce kernels take zero-copy). Every
        # interval's GEMM is written straight into its contiguous k-slab via out=; only
        # k no rectangle covers is zero-filled.
        p = prepared_features[0].shape[0]
        n_hyp = prepared_weights[0].shape[0]
        slab_base = torch.empty(
            (int(n_freq), p, n_hyp),
            dtype=prepared_features[0].dtype,
            device=prepared_features[0].device,
        )
        covered = [False] * int(n_freq)
        for iv, y, w in zip(self.k_intervals, prepared_features, prepared_weights, strict=True):
            torch.bmm(
                y.permute(1, 0, 2),  # (num_k, P, num_m)
                w.permute(1, 2, 0),  # (num_k, num_m, N)
                out=slab_base[iv.k_start : iv.k_stop],
            )
            for k in range(iv.k_start, iv.k_stop):
                covered[k] = True
        for k, is_covered in enumerate(covered):
            if not is_covered:
                slab_base[k].zero_()
        return slab_base.permute(1, 2, 0)

    def _run_fp16(
        self,
        Y_flat: torch.Tensor | None,
        n_freq: int,
        out: torch.Tensor | None,
        prepared_weights: list[torch.Tensor] | None,
        prepared_features: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        """fp16 4M contraction; see :meth:`run` (``precision="fp16"``)."""
        if prepared_weights is None:
            raise ValueError('precision="fp16" requires prepared_weights')
        if prepared_features is None:
            if Y_flat is None:
                raise ValueError("run() needs Y_flat or prepared_features")
            prepared_features = self.prepare_features(Y_flat, precision="fp16")

        if out is not None:
            # Caller-provided (complex64) accumulator: fp32-accumulated per interval.
            for iv, a16, b16 in zip(
                self.k_intervals, prepared_features, prepared_weights, strict=True
            ):
                out[:, :, iv.k_start : iv.k_stop] += _contract_region_fp16(
                    a16, b16, accumulate_fp32=True
                )
            return out

        iv0 = self.k_intervals[0]
        if len(self.k_intervals) == 1 and iv0.k_start == 0 and iv0.k_stop == int(n_freq):
            # complex32 (P, N, n_freq) view of the fp16 GEMM output, zero copies.
            return _contract_region_fp16(
                prepared_features[0], prepared_weights[0], accumulate_fp32=False
            )

        # General: complex32 frequency-major slabs, one fp16 GEMM per interval written
        # directly (as its (num_k, P, 2N) real view) into its k-slab.
        p = prepared_features[0].shape[1]
        n = prepared_weights[0].shape[-1] // 2
        slab_base = torch.empty(
            (int(n_freq), p, n), dtype=torch.complex32, device=prepared_features[0].device
        )
        covered = [False] * int(n_freq)
        for iv, a16, b16 in zip(
            self.k_intervals, prepared_features, prepared_weights, strict=True
        ):
            slab_real = torch.view_as_real(slab_base[iv.k_start : iv.k_stop]).view(
                iv.num_k, p, 2 * n
            )
            torch.bmm(a16, b16, out=slab_real)
            for k in range(iv.k_start, iv.k_stop):
                covered[k] = True
        for k, is_covered in enumerate(covered):
            if not is_covered:
                slab_base[k].zero_()
        return slab_base.permute(1, 2, 0)

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

        # Fast path: already laid out exactly as `tiling` wants
        if (
            self.tiling is not None
            and self.Y is not None
            and new_features.numel() == 0
            and self.tiling.num_features == tiling.num_features
            and torch.equal(self.tiling.k_indices, tiling.k_indices)
            and torch.equal(self.tiling.m_indices, tiling.m_indices)
        ):
            self.tiling = tiling
            return

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
