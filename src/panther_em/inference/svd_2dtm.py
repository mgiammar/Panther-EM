r"""Symmetry exploiting SVD-2DTM search including in-plane rotations.

Pre-processing Stage
--------------------
1.  Given a "SVD result" object, select the top-r features each with
    ``(k_idx, eig_idx)`` plus the associated singular values, ``s``. Then, reorder the
    indices of the top-r features such that they appear in increasing ``k_idx`` order
    (for efficiency in a downstream grouped sum reduction stage).
2.  Using the associated "reconstruction" object, produce the complex-valued Cartesian
    template feature kernels for the top-r selected features (from
    :func:`build_block_kernels`).
3.  If appropriate (decomposition- and image-dependent), apply Fourier filters to the
    the template feature kernels. Assume the image has already been pre-processed.


Image Featurization Stage
-------------------------
1.  Do a complex-valued cross-correlation of the image (shape ``[H, W]``) with the stack
    of kernels (shape ``[r, h, w]``) into an output featurized image for the valid
    cross-correlation region, denoted as variable ``Z``, with shape
    ``[H - h + 1, W - w + 1, r]``. NOTE: last (feature) dimension is contiguous for
    efficient per-pixel memory access.
2.  Based on an optional real-space correlation pixel mask, ``M``, with shape
    ``[H - h + 1, W - w + 1]``, apply the mask to the featurized image ``Z``.
3.  Then, flatten the featurized pixel image into a 2D tensor of shape ``[P, r]`` where
    ``P = (H - h + 1) * (W - w + 1)`` for no mask or ``P = sum(M)`` for a mask. Refer to
    this flattened featurized 2D tensor as ``Y``.

NOTE: Arbitrarily batched images are also supported for shapes ``(..., H, W)``. The
      featurization stage still collapses the spatial dimension into a single pixel axis
      but tracking input shape permits reconstruction.


Inner Product Search Stage
--------------------------
1.  Construct the per-hypotheses weights from the "SVD result" object (using
    :func:`build_block_weights`) which represent ``U\Sigma`` for the selected ``r``
    features. These weights have shape ``(N', r)`` where ``N'`` indexes out-of-plane
    orientation and possibly defocus hypotheses.
2.  Choose batch dimensions across flattened pixel space ``PX_BATCH_SIZE`` and the
    hypothesis space ``HY_BATCH_SIZE``.
3.  Do a partial sum reduction inner product for a batch combination (across only
    ``eig_idx`` for each ``k_idx``) in a tiled kernel via:
    a.  Extract a ``[PX_BLOCK_DIM, FT_BLOCK_DIM]`` tile from the the flattened
        featurized image as ``y``.
    b.  Extract a ``[HY_BLOCK_DIM, FT_BLOCK_DIM]`` tile from the reconstruction weights
        as ``w``.
    c.  Compute batched element-wise multiplication between ``y`` and ``w`` along the
        feature dimension reducing (sum-wise) to an output tile ``c`` of shape
        ``[PX_BLOCK_DIM, HY_BLOCK_DIM, NUM_K]`` based on the associated angular
        frequency ``k_idx`` of each feature.
        NOTE: Need to figure out how to tile ``c`` plus the layout of angular
              frequencies along the feature dimension to maximize memory efficiency.
    d.  Accumulate results from each tile into a final output tensor, ``C``, for the
        batch combination with shape ``[PX_BATCH_SIZE, HY_BATCH_SIZE, NUM_K]``.
    e.  Apply a inverse real-valued FFT across the angular frequency dimension (-1) to
        produce a real-valued correlogram for the batch combination with shape
        ``[PX_BATCH_SIZE, HY_BATCH_SIZE, N_psi]``.
    f.  Apply statistics tracking and reduction (online mean/variance update, max/argmax
        tracking across hypothesis and psi dimensions) to the correlogram.
4.  Repeat step 3 for all combinations across batch sizes to process full images.


A Multi-Precision Approach
--------------------------
Leveraging knowledge of SVD truncation rank (selected top-r features) and reconstruction
accuracy, can begin with a much lower-precision correlogram reconstruction and
progressively increase reconstruction rank. Based on current iteration result and the
reconstruction accuracy (from singular values, plus/minus some bounds), can iterate on
the algorithm as:
1.  Partition reconstructed cross-correlogram values into following three categories:
    a.  Accepted detections (based on reconstruction accuracy, far above expectation)
    b.  Rejected detections (based on reconstruction accuracy, far below expectation)
    c.  Image positions for further follow-up.
2.  Record accepted and rejected image locations plus associated statistics, update
    image mask to select for only follow-up positions.
3.  Repeat though pre-processing stage with the updated image mask and a selected
    ``r_{i+1} > r_{i}``. NOTE: Can leverage pre-calculated data, like previous kernels,
    if stored appropriately.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from tqdm import tqdm

from panther_em.inference.correlation import (
    build_cartesian_kernels,
    compute_feature_stack,
)
from panther_em.inference.projection_reconstruction import ProjectionReconstructor

# A rectangle is a contiguous block of feature space,
# ``(k_start, m_start, k_extent, m_extent)``, for selecting regions of SVD features.
Rectangle = tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Shape / index helpers
# ---------------------------------------------------------------------------


def _ensure_bhw(image: torch.Tensor) -> torch.Tensor:
    """Helper to coerce ``(H, W)`` or ``(B, H, W)`` to ``(B, H, W)``, error otherwise."""
    if image.dim() == 2:
        return image.unsqueeze(0)
    if image.dim() == 3:
        return image
    raise ValueError(
        f"image must have shape (H, W) or (B, H, W), got {tuple(image.shape)}"
    )


def psi_degrees(n_psi: int, device: torch.device | str) -> torch.Tensor:
    """Uniformly spaced in-plane angles (degrees) from 0 to 360."""
    return 360.0 * torch.arange(n_psi, device=device, dtype=torch.float32) / n_psi


# ---------------------------------------------------------------------------
# Tracking search statistics over hypotheses for some pixels
# ---------------------------------------------------------------------------


class PixelStats:
    """Helper for tracing statistics over hypothesis space in a small pixel batch.

    Attributes
    ----------
    hypothesis_count : int
        The number of hypotheses processed.
    num_pixels : int
        The number of pixels in the batch.
    corr_sum : torch.Tensor
        The sum of correlation values.
    corr_sum2 : torch.Tensor
        The sum of squared correlation values.
    best_corr : torch.Tensor
        The best correlation value found.
    best_hypothesis : torch.Tensor
        Global index corresponding to the hypothesis which produced the best correlation
        value.
    best_psi_angle : torch.Tensor
        The in-plane rotation angle (also a hypothesis, but separate axis) which
        produced the best correlation value.

    Methods
    -------
    clear : None
        Clear all tracked statistics.
    update(corr: torch.Tensor, hyp_global_idx: torch.Tensor) : None
        Update tracked statistics with new corr values and hypothesis indices.
    finalize : dict[str, torch.Tensor]
        Return a dictionary of the final statistics (maximum intensity projection /
        maximum inner product, z-score, mean, variance, best hypothesis index,
        best psi index).
    """

    hypothesis_count: int
    num_pixels: int
    corr_sum: torch.Tensor
    corr_sum2: torch.Tensor
    best_corr: torch.Tensor
    best_hypothesis: torch.Tensor
    best_psi_angle: torch.Tensor

    def __init__(
        self,
        num_pixels: int,
        device: torch.device,
        accumulate_dtype: torch.dtype = torch.float32,
    ):
        """Initialize a new pixel statistics tracker."""
        self.hypothesis_count = 0
        self.num_pixels = num_pixels
        self.corr_sum = torch.zeros(num_pixels, device=device, dtype=accumulate_dtype)
        self.corr_sum2 = torch.zeros(num_pixels, device=device, dtype=accumulate_dtype)
        self.best_corr = torch.full(
            (num_pixels,), float("-inf"), device=device, dtype=accumulate_dtype
        )
        self.best_hypothesis = torch.full(
            (num_pixels,), -1, device=device, dtype=torch.long
        )
        self.best_psi_angle = torch.full(
            (num_pixels,), -1, device=device, dtype=torch.long
        )

    @torch.no_grad()
    def clear(self) -> None:
        """Clear all tracked statistics."""
        self.hypothesis_count = 0
        self.corr_sum.zero_()
        self.corr_sum2.zero_()
        self.best_corr.fill_(float("-inf"))
        self.best_hypothesis.fill_(-1)
        self.best_psi_angle.fill_(-1)

    @torch.no_grad()
    def update(self, corr: torch.Tensor, hyp_global_idx: torch.Tensor) -> None:
        """Update tracked statistics with new corr values and hypothesis indices.

        Parameters
        ----------
        corr : torch.Tensor
            Tensor with correlation values with shape (num_pixels, hyp_batch, num_psi)
            where `num_pixels` must match held in `self.num_pixels`, `hyp_batch` is
            number of hypotheses in the batch, and `num_psi` are all in-plane angles.
        hyp_global_idx : torch.Tensor
            Integer tensor referencing which indices these hypotheses correspond to,
            shape (hyp_batch,).
        """
        corr_cast = corr.to(self.best_corr.dtype)
        num_psi = corr.shape[2]  # in-plane angle axis
        total_hypotheses = corr.shape[1] * corr.shape[2]  # num(other) * num(in-plane)

        ### 1. Update sum and squared sum statistics
        self.corr_sum += corr_cast.sum(dim=(1, 2))
        self.corr_sum2 += (corr_cast**2).sum(dim=(1, 2))
        self.hypothesis_count += total_hypotheses

        ### 2. Conditional update on best correlation and hypotheses
        corr_flat = corr_cast.reshape(-1, total_hypotheses)
        vmax, amax = corr_flat.max(dim=1)
        mask = vmax > self.best_corr

        if not mask.any():  # Short circuit if none of the indices improve
            return

        # Decode indices for all pixels to undo flattening of last dim
        n_local = torch.div(amax, num_psi, rounding_mode="floor")
        psi = amax % num_psi
        global_hyp = hyp_global_idx[n_local]

        self.best_corr[mask] = vmax[mask]
        self.best_hypothesis[mask] = global_hyp[mask]
        self.best_psi_angle[mask] = psi[mask]

    @torch.no_grad()
    def finalize(self) -> dict[str, torch.Tensor]:
        """Finalize tracked statistics into summary dictionary.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the final tracked statistics. Tensors exists on device
            used to initialize the PixelStats object.
        """
        mean = self.corr_sum / self.hypothesis_count
        variance = (self.corr_sum2 / self.hypothesis_count) - (mean**2)
        variance.clamp_min_(0.0)
        std = variance.sqrt()
        zscore = (self.best_corr - mean) / std.clamp_min(1e-12)

        return {
            "mip": self.best_corr,
            "zscore": zscore,
            "mean": mean,
            "variance": variance,
            "best_index": self.best_hypothesis,
            "best_psi": self.best_psi_angle,
        }


# ---------------------------------------------------------------------------
# Rectangular tiling of ``(k, m)`` feature space
# ---------------------------------------------------------------------------


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
        return self.k_stop - self.k_start

    @property
    def num_m(self) -> int:
        return self.m_stop - self.m_start

    @property
    def num_features(self) -> int:
        return self.num_k * self.num_m

    @property
    def extent(self) -> tuple[int, int]:
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

        return tensor.index_select(-1, idx).unflatten(-1, (self.num_k, self.num_m))

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
            unconjugated). Defaults to ``True``.

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
        return sum(region.num_features for region in self.regions)

    @property
    def k_stop(self) -> int:
        """Smallest angular-frequency axis length that can hold this tiling."""
        return max(region.k_stop for region in self.regions)

    @property
    def device(self) -> torch.device:
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
            last axis (see :func:`contract_weights_to_correlations`).
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

    Holds a complex tensor of shape ``(P, tiling.num_features)`` -- one column per
    ``(k, m)`` cell of the tiling (the valid cross-correlation of the image against
    that cell's Cartesian kernel, flattened over the ``P`` output pixels). Enables easy
    re-layout of the feature axis when tiling changes, such as in an incremental search
    where only the new ``(k, m)`` cells should be featureized.

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
            :func:`featurize_cells`).
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
            Long indices into the ``P`` pixel axis (e.g. a shrinking follow-up mask).
            When omitted the full ``(P, num_features)`` image is returned.

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


# ---------------------------------------------------------------------------
# Helpers tying the store/tiling to the reconstructor (image features + weights)
# ---------------------------------------------------------------------------


def build_block_kernels(
    reconstructor: ProjectionReconstructor,
    indices: torch.Tensor,
    **polar_to_cart_kwargs: dict[str, Any],
) -> torch.Tensor:
    r"""Build Cartesian feature kernels for arbitrary (k_idx, eig_idx) pairs.

    Parameters
    ----------
    reconstructor : ProjectionReconstructor
        Supplies the polar transform and stored SVD tensors.
    indices : torch.Tensor
        Shape ``(r, 2)`` integer array; each row is ``(k_idx, eig_idx)``.
        For real-valued decompositions, ``k_idx`` must be in ``[0, k_max]``.
    **polar_to_cart_kwargs
        Forwarded to
        :meth:`ProjectionReconstructor.construct_cartesian_feature` (e.g. ``order``,
        ``mode``, ``preserve_energy``).

    Returns
    -------
    torch.Tensor
        Complex64 kernels of shape ``(r, kH, kW)`` on the device, indexed by
        the input ``indices`` array.

    Raises
    ------
    NotImplementedError
        If the result is a complex-valued projection.
    """
    if reconstructor.result.is_complex_projection:
        raise NotImplementedError("Complex-valued projections not yet supported")

    return build_cartesian_kernels(reconstructor, indices, **polar_to_cart_kwargs)


def build_multichannel_correlogram(
    image: torch.Tensor,  # shape (B, H, W)
    kernels: torch.Tensor,  # shape (r, kH, kW)
    *,
    out: torch.Tensor | None = None,  # shape (B, r, out_H, out_W)
    feature_chunk: int | None = None,
) -> torch.Tensor:  # shape (B, r, out_H, out_W)
    r"""Cross-correlate an image against a Cartesian feature stack.

    Parameters
    ----------
    image : torch.Tensor
        Real image of shape ``(B, H, W)``.
    kernels : torch.Tensor
        Cartesian feature kernels of shape ``(r, kH, kW)`` as returned by
        :func:`build_block_kernels`.
    out : torch.Tensor, optional
        Optional pre-allocated output tensor of shape ``(B, r, out_H, out_W)``.
        NOTE: May be a different device than ``image`` or ``kernels`` to manage large
        tensors.
    feature_chunk : int, optional
        Number of kernels to correlate per FFT batch, bounding transient FFT memory.
        Defaults to all kernels at once.

    Returns
    -------
    torch.Tensor
        Complex64 stack ``Z`` of shape ``(B, r, out_H, out_W)`` on the kernels'
        device, where ``out_H = H - kH + 1`` and ``out_W = W - kW + 1``.

    Raises
    ------
    ValueError
        If ``image`` is smaller than the kernel box (empty valid correlation).
    """
    compute_device = kernels.device

    if kernels.dim() != 3:
        raise ValueError(
            f"kernels must have shape (r, kH, kW), got {tuple(kernels.shape)}"
        )

    r, k_h, k_w = kernels.shape

    # Compute expected shapes output shapes
    image = _ensure_bhw(image).to(compute_device)
    b = image.shape[0]
    out_h = image.shape[-2] - k_h + 1
    out_w = image.shape[-1] - k_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError(
            f"image {tuple(image.shape[-2:])} smaller than kernel {(k_h, k_w)}; "
            "valid cross-correlation is empty."
        )

    # Check if out is provided and compatible
    if out is not None:
        if out.shape != (b, r, out_h, out_w):
            raise ValueError(
                f"out has shape {tuple(out.shape)}, expected {(b, r, out_h, out_w)}"
            )
        if out.dtype != torch.complex64:
            raise ValueError(f"out must have dtype complex64, got {out.dtype}")
        z_flat = out
    else:
        z_flat = torch.empty(
            (b, r, out_h, out_w), dtype=torch.complex64, device=compute_device
        )

    chunk = feature_chunk if feature_chunk is not None else r
    for start in tqdm(
        range(0, r, chunk),
        desc="Building multi-channel corr",
        unit="kernel",
    ):
        stop = min(start + chunk, r)
        tmp = compute_feature_stack(image, kernels[start:stop])
        z_flat[:, start:stop] = tmp.to(z_flat.device)  # NOTE: may be different device

    return z_flat


def build_block_feature_stack(
    image: torch.Tensor,
    reconstructor: ProjectionReconstructor,
    *,
    indices: torch.Tensor,
    out: torch.Tensor | None = None,
    feature_chunk: int | None = None,
    **polar_to_cart_kwargs: dict[str, Any],
) -> torch.Tensor:
    r"""Convenience: build kernels and the multi-channel correlogram in one call.

    Equivalent to :func:`build_block_kernels` followed by
    :func:`build_multichannel_correlogram`. Use the two-step form when the kernels
    should be cached across images.

    Parameters
    ----------
    image : torch.Tensor
        Real image of shape ``(H, W)`` or ``(B, H, W)``.
    reconstructor : ProjectionReconstructor
        Builds the Cartesian kernels and supplies the polar transform / device.
    indices : torch.Tensor
        Shape ``(r, 2)`` integer array; each row is ``(k_idx, eig_idx)``.
    out : torch.Tensor, optional
        Optional pre-allocated output tensor of shape ``(B, r, out_H, out_W)``. May
        be on different device than the kernels or image to manage large tensors.
    feature_chunk : int, optional
        Kernel correlation chunk size (see
        :func:`build_multichannel_correlogram`).
    **polar_to_cart_kwargs
        Forwarded to kernel construction.

    Returns
    -------
    torch.Tensor
        Complex64 stack ``(B, r, out_H, out_W)``, can be different device than input
        image if ``out`` is provided and on a different device.
    """
    kernels = build_block_kernels(reconstructor, indices, **polar_to_cart_kwargs)
    return build_multichannel_correlogram(
        image, kernels, feature_chunk=feature_chunk, out=out
    )


def featurize_cells(
    image: torch.Tensor,
    reconstructor: ProjectionReconstructor,
    cells: torch.Tensor,
    *,
    feature_chunk: int | None = None,
    **polar_to_cart_kwargs: Any,
) -> torch.Tensor:
    """Cross-correlate an image against the kernels for a set of feature cells.

    Parameters
    ----------
    image : torch.Tensor
        Real image of shape ``(H, W)`` (single image; ``P = out_h * out_w``).
    reconstructor : ProjectionReconstructor
        Builds the kernels and supplies the polar transform / device.
    cells : torch.Tensor
        ``(n_cells, 2)`` long tensor of ``(k, m)`` cells to featurize, e.g. from
        :meth:`FeaturizedImageStore.missing_cells`.
    feature_chunk : int, optional
        Kernel-correlation chunk size (see :func:`build_multichannel_correlogram`).
    **polar_to_cart_kwargs
        Forwarded to kernel construction.

    Returns
    -------
    torch.Tensor
        Complex features of shape ``(n_cells, P)``.
    """
    # TODO: Add support for >1 batch dimension (e.g. multiple image classes)
    z = build_block_feature_stack(
        image,
        reconstructor,
        indices=cells.to(torch.long),
        feature_chunk=feature_chunk,
        **polar_to_cart_kwargs,
    )

    _b, n_cells, out_h, out_w = z.shape

    return z.reshape(n_cells, out_h * out_w)


def build_block_weights(
    reconstructor: ProjectionReconstructor,
    indices: torch.Tensor,
) -> torch.Tensor:
    r"""Build contraction weights ``W = U * S`` for arbitrary (k_idx, eig_idx) pairs.

    Parameters
    ----------
    reconstructor : ProjectionReconstructor
        Provides the on-device ``U``/``S`` tensors and the result metadata.
    indices : torch.Tensor
        Shape ``(r, 2)`` integer array; each row is ``(k_idx, eig_idx)``.
        For real-valued decompositions, ``k_idx`` must be in ``[0, k_max]``.

    Returns
    -------
    torch.Tensor
        Complex64 weights of shape ``(FF, O, r)`` on ``reconstructor.device``.

    Raises
    ------
    NotImplementedError
        If the result is a complex-valued projection.
    """
    result = reconstructor.result

    if result.is_complex_projection:
        raise NotImplementedError("Complex-valued projections not yet supported")

    U, S, _ = result.get_svd_tensors(indices.cpu().numpy(), reconstructor.device)
    # U: (FF, O, r),  S: (r,)
    return U * S[None, None, :]


def build_layout_weights(
    reconstructor: ProjectionReconstructor,
    tiling: FeatureTiling,
    hypothesis_indexes: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Build ``W = U * S`` weights laid out to match a tiling's feature dimension.

    Parameters
    ----------
    reconstructor : ProjectionReconstructor
        Supplies the on-device ``U`` / ``S`` tensors.
    tiling : FeatureTiling
        Defines the compacted feature dimension via :attr:`FeatureTiling.layout`.
    hypothesis_indexes : torch.Tensor, optional
        Long indices selecting a subset of the flattened ``(FF * O)`` template rows
        (the hypothesis batch). Defaults to all rows.

    Returns
    -------
    torch.Tensor
        Complex64 weights of shape ``(N, r)`` where ``r = tiling.num_features`` and
        ``N`` is the number of selected hypotheses.
    """
    weights = build_block_weights(reconstructor, tiling.layout)  # (FF, O, r)

    ff, o, r = weights.shape
    w_flat = weights.reshape(ff * o, r)

    if hypothesis_indexes is not None:
        w_flat = w_flat[hypothesis_indexes.to(w_flat.device)]

    return w_flat


# ---------------------------------------------------------------------------
# Incremental (multi-precision) search driver
# ---------------------------------------------------------------------------


@torch.no_grad()
def progressive_search(
    image: torch.Tensor,
    reconstructor: ProjectionReconstructor,
    stage_rectangles: Sequence[Sequence[Rectangle]],
    *,
    store: FeaturizedImageStore | None = None,
    hypothesis_indexes: torch.Tensor | None = None,
    pixel_index: torch.Tensor | None = None,
    pixel_batch: int | None = None,
    hyp_batch: int | None = None,
    n_psi: int | None = None,
    feature_chunk: int | None = None,
    follow_up_fn: (
        Callable[[dict[str, torch.Tensor], torch.Tensor], torch.Tensor | None] | None
    ) = None,
    **polar_to_cart_kwargs: Any,
) -> Any:  # TODO: Get proper return type for generator
    r"""Incremental SVD-2DTM search; yields per-pixel statistics per stage.

    TODO: Integrate the error-aware decision process into the follow up function for
          both pixel and hypothesis selection. Error-aware decision process not yet
          implemented.

    TODO: Split long, complex logic int smaller helper functions for clarity.

    TODO: Less implicit parameters passing, that is, for batching and numbers require
          their inclusion.

    Implements the multi-precision strategy: each stage selects a (typically
    higher-rank) set of contiguous feature-space rectangles, reusing the image
    features computed in earlier stages.  The nested loops, outermost to innermost,
    are:

    1.  **Stages** -- ``stage_rectangles`` (increasing rank). Between stages an
        external caller narrows the pixel/hypothesis sets; that selection logic lives
        in ``follow_up_fn`` and is intentionally out of scope here.
    2.  **Pixel batches** -- bound the transient ``(P_b, N, N_\psi)`` correlogram.
    3.  **Hypothesis batches** -- ``W`` is rebuilt per batch; statistics accumulate
        online via :class:`PixelStats`.
    4.  **Regions (innermost)** -- :meth:`FeatureTiling.run` sweeps the rectangular
        regions, each a batched ``bmm`` accumulating into the spectrum ``C``.

    The expensive image featurization happens only for cells genuinely missing from
    ``store``; everything downstream of the store (weights ``W``, the spectrum ``C``,
    the correlogram, and the reductions) is recomputed every stage.

    Parameters
    ----------
    image : torch.Tensor
        Real image of shape ``(H, W)``.
    reconstructor : ProjectionReconstructor
        Holds the SVD tensors, polar transform, and device.
    stage_rectangles : Sequence[Sequence[Rectangle]]
        One list of ``(k_start, m_start, k_extent, m_extent)`` rectangles per stage.
        Rectangles within a stage must be pairwise cell-disjoint; overshoot (extra
        low-value cells) is fine and is not zeroed.
    store : FeaturizedImageStore, optional
        Persistent feature store to reuse across calls. A fresh one sized to the
        image's valid-correlation grid is created when omitted.
    hypothesis_indexes : torch.Tensor, optional
        Long indices into the flattened ``(FF * O)`` template space to search.
        Defaults to all templates.
    pixel_index : torch.Tensor, optional
        Long indices into the ``P`` valid-correlation pixels for the *first* stage
        (e.g. an initial mask). Defaults to all pixels. Subsequent stages use the
        mask returned by ``follow_up_fn``.
    pixel_batch, hyp_batch : int, optional
        Batch sizes for the pixel and hypothesis loops. Default to all-at-once.
    n_psi : int, optional
        Number of in-plane samples. Defaults to ``result.num_angular_components``.
    feature_chunk : int, optional
        Stage-1 kernel-correlation chunk size for featurizing missing cells.
    follow_up_fn : callable, optional
        ``(stats, pixel_index) -> next_pixel_index | None`` mapping a stage's
        finalized statistics and the pixels it ran on to the follow-up pixel mask for
        the next stage. Returning ``None`` (the default behaviour when omitted) keeps
        the same pixel set.
    **polar_to_cart_kwargs
        Forwarded to kernel construction when featurizing cells.

    Yields
    ------
    dict[str, torch.Tensor]
        Per stage, the finalized :meth:`PixelStats.finalize` maps (``"mip"``,
        ``"zscore"``, ``"mean"``, ``"variance"``, ``"best_index"``, ``"best_psi"``)
        each of shape ``(P_stage,)``, indexed by the stage's pixel set.
    """
    result = reconstructor.result
    device = reconstructor.device
    if n_psi is None:
        n_psi = result.num_angular_components

    # Valid cross-correlation grid -> number of searchable pixels P.
    k_h, k_w = reconstructor.image_shape
    out_h = image.shape[-2] - k_h + 1
    out_w = image.shape[-1] - k_w + 1
    n_px = out_h * out_w

    if store is None:
        store = FeaturizedImageStore(n_px, device=device)

    if hypothesis_indexes is None:
        hypothesis_indexes = torch.arange(
            result.num_fourier_filters * result.num_orientations, device=device
        )
    if pixel_index is None:
        pixel_index = torch.arange(n_px, device=device)

    px = pixel_index
    for rects in stage_rectangles:  # (1) stages -- increasing rank
        tiling = FeatureTiling.from_extents(rects, device=device)

        # -- image-feature reuse: featurize ONLY genuinely new cells, then move the
        #    store into this stage's contiguous layout --
        todo = store.missing_cells(tiling)
        if todo.numel():
            feats = featurize_cells(
                image,
                reconstructor,
                todo,
                feature_chunk=feature_chunk,
                **polar_to_cart_kwargs,
            )
        else:
            feats = torch.empty((0, n_px), dtype=store.dtype, device=store.device)
        store.relayout(tiling, feats)

        w_layout = build_layout_weights(reconstructor, tiling)  # (FF*O, r)

        # Accumulate this stage's statistics over its pixel set.
        p_step = pixel_batch if pixel_batch is not None else int(px.numel())
        h_step = hyp_batch if hyp_batch is not None else int(hypothesis_indexes.numel())

        stage_stats: list[dict[str, torch.Tensor]] = []
        for p0 in range(0, int(px.numel()), p_step):  # (2) pixel batches
            px_b = px[p0 : p0 + p_step]
            Y_flat = store.image_view(px_b)  # (P_b, r), no recompute
            stats = PixelStats(int(px_b.numel()), device)

            for h0 in range(
                0, int(hypothesis_indexes.numel()), h_step
            ):  # (3) hyp batches
                hyp_b = hypothesis_indexes[h0 : h0 + h_step]
                W_flat = w_layout[hyp_b]  # (N_b, r)
                # (4) regions -> spectrum (P_b, N_b, k_max)
                C = tiling.run(Y_flat, W_flat, result.k_max)
                corr = torch.fft.irfft(
                    C.conj(), n=n_psi, dim=-1, norm="forward"
                )  # (P_b, N_b, N_psi)
                stats.update(corr, hyp_b)

            stage_stats.append(stats.finalize())

        # Stitch the pixel batches back into stage-level maps.
        stage_maps = {
            key: torch.cat([s[key] for s in stage_stats], dim=0)
            for key in stage_stats[0]
        }

        yield stage_maps

        # Hand the follow-up mask (selection logic) back to the next stage.
        if follow_up_fn is not None:
            nxt = follow_up_fn(stage_maps, px)
            if nxt is not None:
                px = nxt


# ---------------------------------------------------------------------------
# Stage 0 -- SVD weights and Cartesian kernels (built once, reused)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stage 1 -- multi-channel cross-correlogram
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stage 2+3 (fused) -- weighted contraction + kappa->psi inverse transform
# ---------------------------------------------------------------------------


# def contract_weights_to_correlations(
#     z: torch.Tensor,  # shape (..., L)
#     weights: torch.Tensor,  # shape (N, L)
#     k_indices: torch.Tensor,  # shape (L,), angular-frequency block of each channel
#     *,
#     k_max: int,
#     n_psi: int,
#     weights_indices: torch.Tensor | None = None,  # (M,) subset of rows; default all
# ) -> torch.Tensor:  # shape (..., M, n_psi)
#     r"""Fuse the SVD weight contraction with the ``\kappa -> \psi`` inverse transform.

#     For each selected template row ``n`` and each pixel in the leading batch,
#     contracts the ``L`` feature channels against the weights, groups the products
#     by angular frequency ``k`` into ``k_max`` blocks, and inverse-FFTs over ``k`` to
#     recover the matched-filter correlation at all ``n_psi`` in-plane angles in one
#     shot::

#         c_n(psi_a) = sum_k [ sum_{l : k_l = k} z_l * conj(W[n, l]) ] * e^{-i k psi_a}

#     The ``sum_l`` is a vectorised ``index_add_`` into the ``k`` axis (no sorting of
#     ``k_indices`` required), and the ``sum_k e^{-i k psi}`` is realised by an
#     :func:`torch.fft.irfft` over the ``k_max`` non-negative angular-frequency bins
#     (real-valued decompositions only).

#     Parameters
#     ----------
#     z : torch.Tensor
#         Complex per-pixel features of shape ``(..., L)``; leading dims are an
#         arbitrary pixel batch.
#     weights : torch.Tensor
#         Complex contraction weights ``W = U * S`` of shape ``(N, L)`` (the
#         ``(FF, O, L)`` output of :func:`build_block_weights` flattened to
#         ``N = FF * O`` rows).  **Not** pre-conjugated.
#     k_indices : torch.Tensor
#         Shape ``(L,)`` mapping each feature channel to its angular-frequency block
#         in ``[0, k_max)`` (i.e. the ``k_idx`` column of ``indices``).
#     k_max : int
#         Number of non-negative angular-frequency blocks (``result.k_max``).
#     n_psi : int
#         Number of in-plane samples ``N_\psi`` to synthesise (``= 2*(k_max-1)`` for
#         a band-limited real decomposition, but passed explicitly for odd lengths).
#     weights_indices : torch.Tensor, optional
#         Shape ``(M,)`` selecting a subset of the ``N`` template rows to evaluate
#         (sub-blocking the search). Defaults to all ``N`` rows.

#     Returns
#     -------
#     torch.Tensor
#         Real float32 correlations of shape ``(..., M, n_psi)``; output index ``a``
#         along the last axis is in-plane angle ``\psi_a = 360 * a / n_psi`` degrees.
#     """
#     w = weights if weights_indices is None else weights[weights_indices]  # (M, L)
#     k_idx = k_indices.to(device=z.device, dtype=torch.long)

#     # (..., 1, L) * (M, L) -> (..., M, L); conj forms the matched filter (W = U*S).
#     prod = z.unsqueeze(-2) * w.conj()

#     # Group features by angular frequency into k_max blocks: (..., M, k_max).
#     c_hat = torch.zeros((*prod.shape[:-1], k_max), dtype=prod.dtype, device=prod.device)
#     c_hat.index_add_(-1, k_idx, prod)

#     # kappa -> psi: mf(psi_a) = sum_k c_hat_k e^{-i k psi_a}.  Conjugating the
#     # input flips irfft's ``+`` analysis sign to the required ``-`` and Hermitian-
#     # extends k<0; forward norm leaves the kernels' 1/sqrt(N_psi) intact.
#     correlations = torch.fft.irfft(c_hat.conj(), n=n_psi, dim=-1, norm="forward")
#     return correlations


# ---------------------------------------------------------------------------
# Stage 4 -- per-pixel 2DTM search statistics
# ---------------------------------------------------------------------------

# Per-pixel statistic maps, grouped by output dtype. ``search_2dtm`` pre-allocates
# one tensor per key and feeds pixel-batch slices to ``compute_search_statistics``.
STAT_FLOAT_KEYS: tuple[str, ...] = ("mip", "mean", "variance", "std", "zscore")
STAT_INDEX_KEYS: tuple[str, ...] = ("best_ff", "best_orientation", "best_psi")


def allocate_search_maps(
    batch_shape: tuple[int, ...],
    device: torch.device | str,
    *,
    real_dtype: torch.dtype = torch.float32,
    index_dtype: torch.dtype = torch.long,
) -> dict[str, torch.Tensor]:
    """Pre-allocate the per-pixel 2DTM statistic maps written by the search.

    Parameters
    ----------
    batch_shape : tuple[int, ...]
        Leading (pixel) dimensions of the maps, e.g. ``(B, n_px)``.
    device : torch.device or str
        Device for the allocated maps.
    real_dtype : torch.dtype, optional
        Dtype for the float statistics (``mip``/``mean``/...). Defaults float32.
    index_dtype : torch.dtype, optional
        Dtype for the argmax index maps (``best_*``). Defaults int64.

    Returns
    -------
    dict[str, torch.Tensor]
        One empty tensor of shape ``batch_shape`` per key in
        :data:`STAT_FLOAT_KEYS` and :data:`STAT_INDEX_KEYS`.
    """
    maps: dict[str, torch.Tensor] = {}
    for key in STAT_FLOAT_KEYS:
        maps[key] = torch.empty(batch_shape, dtype=real_dtype, device=device)
    for key in STAT_INDEX_KEYS:
        maps[key] = torch.empty(batch_shape, dtype=index_dtype, device=device)
    return maps


def compute_search_statistics(
    c_psi: torch.Tensor,
    *,
    n_search_dims: int = 3,
    score: str = "real",
    out: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    r"""Reduce a pixel batch over the search space into 2DTM statistics (Stage 4).

    The last ``n_search_dims`` axes of ``c_psi`` are the search space (by
    convention ``(FF, O, \psi)``); every leading axis is a pixel-batch axis that
    is preserved.  For each pixel this computes the maximum-intensity projection
    (MIP), the ``argmax`` location, and the mean / variance of the search
    distribution (for SNR / z-scoring).

    When ``out`` is supplied the reductions are written **in place** into the
    provided tensors (e.g. pixel-batch slices of maps from
    :func:`allocate_search_maps`), so the hot loop allocates no new output
    storage -- the body is a static, ``torch.compile``-friendly tensor program.

    Parameters
    ----------
    c_psi : torch.Tensor
        ``(..., FF, O, \psi)`` correlation (real, or complex from a complex
        decomposition -- reduced to a real score via ``score``).
    n_search_dims : int, optional
        Number of trailing axes that constitute the search space. Defaults to 3
        (``FF``, ``O``, ``\psi``).
    score : {"real", "abs"}, optional
        How to turn a complex correlation into a real score. Ignored for real input.
        Defaults to ``"real"``.
    out : dict[str, torch.Tensor], optional
        Destination tensors keyed as below, each shaped like the leading
        (pixel-batch) dims of ``c_psi``.  Every key present is written via
        ``copy_``; keys are a subset of those returned in the allocating path.
        When omitted, fresh tensors are allocated and returned.

    Returns
    -------
    dict[str, torch.Tensor]
        Either ``out`` (in-place path), or freshly allocated per-pixel maps:
        ``"mip"`` (max), ``"mean"``, ``"variance"``, ``"std"``,
        ``"zscore"`` ``= (mip - mean) / std``, ``"count"`` (scalar search size),
        and the unravelled argmax ``"best_ff"`` / ``"best_orientation"`` /
        ``"best_psi"`` when ``n_search_dims == 3``, else ``"best_index"`` (flat).
    """
    if c_psi.is_complex():
        c = c_psi.real if score == "real" else c_psi.abs()
    else:
        c = c_psi

    if c.dim() < n_search_dims:
        raise ValueError(
            f"c_psi has {c.dim()} dims, need at least n_search_dims={n_search_dims}"
        )
    search_shape = c.shape[-n_search_dims:]
    batch_shape = c.shape[:-n_search_dims]
    flat = c.reshape(*batch_shape, -1)
    n = flat.shape[-1]

    mip, arg = flat.max(dim=-1)
    mean = flat.mean(dim=-1)
    variance = flat.var(dim=-1, unbiased=False)
    std = variance.sqrt()
    # Guard against a degenerate (zero-variance) search distribution.
    zscore = (mip - mean) / std.clamp_min(torch.finfo(std.dtype).tiny)

    values: dict[str, torch.Tensor] = {
        "mip": mip,
        "mean": mean,
        "variance": variance,
        "std": std,
        "zscore": zscore,
    }
    if n_search_dims == 3:
        _ff, o, psi = (int(s) for s in search_shape)
        values["best_ff"] = arg // (o * psi)
        values["best_orientation"] = (arg // psi) % o
        values["best_psi"] = arg % psi
    else:
        values["best_index"] = arg

    if out is not None:
        for key, dest in out.items():
            dest.copy_(values[key])
        return out

    values["count"] = torch.tensor(n, device=c.device)
    return values
