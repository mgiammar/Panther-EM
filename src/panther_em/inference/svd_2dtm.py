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


Inner Product Search Stage
--------------------------
1.  Construct the per-hypotheses weights from the "SVD result" object (using
    :func:`build_block_weights`) which represent ``U\Sigma`` for the selected top-r
    features. These weights have shape ``(N', r)`` where ``N'`` indexes out-of-plane
    orientation and possibly defocus index.
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

# A feature cell is an angular-frequency / radial-eigenvector pair ``(k, m)`` mapping to
# a single Cartesian kernel (hence one column of the featurized image and one column of
# the contraction weights).  A rectangle is a contiguous block of feature space,
# ``(k_start, m_start, k_extent, m_extent)``, covering every cell ``(k, m)`` with
# ``k_start <= k < k_start + k_extent`` and ``m_start <= m < m_start + m_extent``.
Cell = tuple[int, int]
Rectangle = tuple[int, int, int, int]


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


# Residual scalar between the \kappa->\psi inverse transform and the matched-filter
# convention of `reconstruct_projection`/`compute_feature_stack`.  The analytic
# Fourier mode already carries ``1/sqrt(N_\psi)``, so the recovery uses an unscaled
# forward-normalised irfft and this scale is 1.0 (empirically pinned against the
# brute-force matched filter in ``tests/test_svd_2dtm.py``).
_PSI_IRFFT_SCALE = 1.0


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


def psi_degrees(n_psi: int, device: torch.device | str | None = None) -> torch.Tensor:
    """Uniformly spaced in-plane angles (degrees) from 0 to 360."""
    return 360.0 * torch.arange(n_psi, device=device, dtype=torch.float32) / n_psi


@dataclass
class RectangularFeatureGroup:
    r"""Batched-MMA group of rectangles sharing ``(k_extent, m_extent)``.

    # A "group" collects one or more feature-space rectangles that have the *same*
    # eigenvector extent ``m_extent`` (the dimension contracted over) and the same
    # angular-frequency extent ``k_extent``, so every angular-frequency row in the
    # group contracts an equal-length ``m`` run and the whole group is a single
    # batched matrix multiplication (``torch.bmm``).

    # The contraction performed for each row ``r`` (one angular frequency ``k``) is
    # ``C[:, :, k] += sum_m Y[:, col(k, m)] * conj(W[:, col(k, m)])``, where the
    # ``sum_m`` is the inner dimension of the ``bmm`` and the scatter onto the output
    # ``k`` axis is an :func:`torch.Tensor.index_add_`.

    Attributes
    ----------
    k_extent : int
        Number of angular frequencies spanned by each rectangle in the group.
    m_extent : int
        Number of eigenvectors (the axis reduced over) in each rectangle.
    feature_indices : torch.Tensor
        Long tensor of shape ``(R, m_extent)`` mapping each ``(row, m)`` pair to its
        column in the flattened feature dimension ``r`` of ``Y_flat`` / ``W_flat``.
        ``R`` is ``num_rectangles_in_group * k_extent``; each row is one angular
        frequency.
    output_k_indices : torch.Tensor
        Long tensor of shape ``(R,)`` giving the destination angular-frequency bin
        in the dense output spectrum for each row of ``feature_indices``.
    """

    k_extent: int  # number of angular frequencies
    m_extent: int  # number of eigenvectors (accumulate over)
    feature_indices: torch.Tensor  # (R, m_extent) column idx along flattened r-dim
    output_k_indices: torch.Tensor  # (R,) angular frequency bin to accumulate into

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
            Complex-valued featurized image of shape ``(P, r)`` where ``P`` is the
            number of pixels under consideration and ``r`` the total feature
            dimension size (number of stored ``(k, m)`` columns).
        W_flat : torch.Tensor
            Complex-valued contraction weights of shape ``(N, r)`` where ``N`` is the
            number of hypotheses in the batch, in the same column layout as
            ``Y_flat``.
        out : torch.Tensor
            Complex-valued output spectrum of shape ``(P, N, K)`` accumulated into in
            place; ``K`` indexes angular-frequency bins (see ``output_k_indices``).
        conjugate : bool, optional
            Apply the matched-filter conjugation on ``W`` (``W = U * S`` is stored
            unconjugated). Defaults to ``True``.

        Returns
        -------
        torch.Tensor
            The same ``out`` tensor, accumulated in place (returned for chaining).
        """
        idx = self.feature_indices.to(Y_flat.device)
        out_k = self.output_k_indices.to(out.device)

        # Gather the (k, m) columns for this group: (P, R, m_extent), (N, R, m_extent).
        feat_group = Y_flat[:, idx]
        weights_group = W_flat[:, idx]
        weights_group = weights_group.conj() if conjugate else weights_group

        # Batch over the angular-frequency rows R: bmm contracts the m_extent axis.
        feat_group = feat_group.permute(1, 0, 2)  # (R, P, m_extent)
        weights_group = weights_group.permute(1, 0, 2)  # (R, N, m_extent)
        out_batch = torch.bmm(feat_group, weights_group.transpose(-2, -1))  # (R, P, N)

        # Scatter each row onto its destination angular-frequency bin.
        out.index_add_(2, out_k, out_batch.permute(1, 2, 0))  # (P, N, R) -> dim 2

        return out


class FeatureTiling:
    r"""A single rectangular tiling of ``(k, m)`` feature space and its contraction.

    # Built from a list of cell-disjoint rectangles plus a column layout
    # (``cell -> column``) supplied by a :class:`FeaturizedImageStore`.  Rectangles are
    # bucketed by ``(k_extent, m_extent)`` into :class:`RectangularFeatureGroup`
    # objects so each bucket is one batched ``bmm``; :meth:`run` accumulates every
    # group into the dense angular-frequency spectrum ``C[P, N, K]``.

    # Overshoot is expected and intentional: a rectangle may include ``(k, m)`` cells
    # whose singular values are negligible.  Those extra cells simply contribute small
    # terms to the contraction and are *not* zeroed.  The only requirement is that the
    # rectangles in one tiling be pairwise cell-disjoint (no ``(k, m)`` covered twice),
    # so the ``index_add_`` scatter does not double count.

    Parameters
    ----------
    rectangles : Sequence[Rectangle]
        ``(k_start, m_start, k_extent, m_extent)`` blocks of feature space.
    column_index : dict[Cell, int]
        Mapping from cell ``(k, m)`` to its column in ``Y_flat`` / ``W_flat``.
    n_freq : int
        Number of output angular-frequency bins ``K`` (``result.k_max`` for a
        real-valued decomposition), i.e. the size of the last axis of ``C``.
    device : torch.device or str, optional
        Device for the built index tensors. Defaults to CPU.
    """

    def __init__(
        self,
        rectangles: Sequence[Rectangle],
        column_index: dict[Cell, int],
        n_freq: int,
        device: torch.device | str | None = None,
    ) -> None:
        self.rectangles: list[Rectangle] = [tuple(r) for r in rectangles]
        self.column_index = column_index
        self.n_freq = int(n_freq)
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.groups: list[RectangularFeatureGroup] = self.build_groups()

    @staticmethod
    def cells_for_rectangles(rectangles: Sequence[Rectangle]) -> list[Cell]:
        """Return the de-duplicated union of all ``(k, m)`` cells in ``rectangles``.

        Parameters
        ----------
        rectangles : Sequence[Rectangle]
            ``(k_start, m_start, k_extent, m_extent)`` blocks.

        Returns
        -------
        list[Cell]
            Sorted unique ``(k, m)`` cells covered by the rectangles.
        """
        cells: set[Cell] = set()
        for k0, m0, k_ext, m_ext in rectangles:
            for k in range(k0, k0 + k_ext):
                for m in range(m0, m0 + m_ext):
                    cells.add((int(k), int(m)))
        return sorted(cells)

    def required_cells(self) -> list[Cell]:
        """Cells this tiling needs featurized (see :meth:`cells_for_rectangles`)."""
        return self.cells_for_rectangles(self.rectangles)

    def build_groups(self) -> list[RectangularFeatureGroup]:
        """Bucket rectangles by ``(k_extent, m_extent)`` into batched-MMA groups.

        # Validates that the rectangles are pairwise cell-disjoint, then for each
        # ``(k_extent, m_extent)`` bucket builds the ``(R, m_extent)`` column-index
        # tensor and the ``(R,)`` output-frequency tensor consumed by
        # :meth:`RectangularFeatureGroup.accumulate_into`.

        Returns
        -------
        list[RectangularFeatureGroup]
            One group per distinct ``(k_extent, m_extent)`` extent.

        Raises
        ------
        ValueError
            If two rectangles cover the same ``(k, m)`` cell, or if a covered cell
            is absent from ``column_index``.
        """
        # 1. Reject overlap within the tiling (overshoot is fine; double counting is
        #    not -- the index_add_ scatter would sum a shared cell twice).
        seen: set[Cell] = set()
        for k0, m0, k_ext, m_ext in self.rectangles:
            for k in range(k0, k0 + k_ext):
                for m in range(m0, m0 + m_ext):
                    cell = (int(k), int(m))
                    if cell in seen:
                        raise ValueError(
                            f"Rectangles overlap at cell {cell}; rectangles within a "
                            "tiling must be pairwise cell-disjoint."
                        )
                    seen.add(cell)

        # 2. Group rectangles sharing the same (k_extent, m_extent) so each bucket is
        #    one batched bmm. Within a bucket every rectangle contributes k_extent rows.
        buckets: dict[tuple[int, int], list[Rectangle]] = {}
        for rect in self.rectangles:
            _, _, k_ext, m_ext = rect
            buckets.setdefault((int(k_ext), int(m_ext)), []).append(rect)

        groups: list[RectangularFeatureGroup] = []
        for (k_ext, m_ext), rects in buckets.items():
            feature_rows: list[list[int]] = []
            output_k: list[int] = []
            for k0, m0, _, _ in rects:
                for k in range(k0, k0 + k_ext):
                    row = [self._column(k, m0 + j) for j in range(m_ext)]
                    feature_rows.append(row)
                    output_k.append(int(k))
            groups.append(
                RectangularFeatureGroup(
                    k_extent=k_ext,
                    m_extent=m_ext,
                    feature_indices=torch.tensor(
                        feature_rows, dtype=torch.long, device=self.device
                    ),
                    output_k_indices=torch.tensor(
                        output_k, dtype=torch.long, device=self.device
                    ),
                )
            )
        return groups

    def _column(self, k: int, m: int) -> int:
        """Look up the feature column for cell ``(k, m)``, erroring if unfeaturized."""
        try:
            return self.column_index[(int(k), int(m))]
        except KeyError as exc:
            raise ValueError(
                f"Cell {(int(k), int(m))} is required by the tiling but has not been "
                "featurized into the column layout."
            ) from exc

    @torch.no_grad()
    def run(
        self,
        Y_flat: torch.Tensor,
        W_flat: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Accumulate every group into the dense spectrum ``C[P, N, K]``.

        Parameters
        ----------
        Y_flat : torch.Tensor
            Featurized image ``(P, r)`` in this tiling's column layout.
        W_flat : torch.Tensor
            Contraction weights ``(N, r)`` in the same column layout.
        out : torch.Tensor, optional
            Pre-allocated ``(P, N, n_freq)`` complex accumulator. A fresh zero tensor is
            allocated when omitted.

        Returns
        -------
        torch.Tensor
            Complex angular-frequency spectrum ``C`` of shape ``(P, N, n_freq)``.
            Recover the real ``\psi``-resolved correlogram with
            :func:`correlogram_from_spectrum`.
        """
        n_px = Y_flat.shape[0]
        n_hyp = W_flat.shape[0]

        if out is None:
            out = torch.zeros(
                (n_px, n_hyp, self.n_freq), dtype=Y_flat.dtype, device=Y_flat.device
            )

        for group in self.groups:
            group.accumulate_into(Y_flat, W_flat, out, conjugate=True)

        return out


class FeaturizedImageStore:
    r"""Persistent store of the featurized image across multi-stage search.

    Holds the complex featurized image ``Y`` of shape ``(P, num_cols)`` -- one column
    per featurized cell ``(k, m)`` (the valid cross-correlation of the image against
    that cell's Cartesian kernel, flattened over the ``P`` output pixels).  The store
    persists across stages of :func:`progressive_search` so that the expensive
    image-featurization is performed once per cell: when a later, higher-rank tiling
    needs new cells, only the missing ones are computed and appended.

    Parameters
    ----------
    num_pixels : int
        Number of valid cross-correlation output pixels ``P`` (``out_h * out_w``).
    n_freq : int
        Number of output angular-frequency bins handed to tilings built against this
        store (``result.k_max`` for a real-valued decomposition).
    device : torch.device or str, optional
        Device the featurized image is held on. Defaults to CPU.
    dtype : torch.dtype, optional
        Complex dtype of the stored features. Defaults to ``torch.complex64``.
    """

    def __init__(
        self,
        num_pixels: int,
        n_freq: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        self.num_pixels = int(num_pixels)
        self.n_freq = int(n_freq)
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.dtype = dtype
        self._cells: list[Cell] = []  # column -> cell
        self._columns: dict[Cell, int] = {}  # cell -> column
        self._Y: torch.Tensor | None = None  # (P, num_cols) complex

    @property
    def column_index(self) -> dict[Cell, int]:
        """Current ``cell -> column`` layout (a copy, safe to hand to a tiling)."""
        return dict(self._columns)

    @property
    def cells(self) -> list[Cell]:
        """Featurized cells in column order."""
        return list(self._cells)

    @property
    def num_columns(self) -> int:
        """Number of featurized cells currently stored."""
        return len(self._cells)

    def missing(self, cells: Sequence[Cell]) -> list[Cell]:
        """Return the subset of ``cells`` not yet featurized (de-duplicated, ordered).

        Parameters
        ----------
        cells : Sequence[Cell]
            Candidate ``(k, m)`` cells (e.g. ``tiling.required_cells()``).

        Returns
        -------
        list[Cell]
            Cells absent from the store, preserving first-seen order.
        """
        out: list[Cell] = []
        seen: set[Cell] = set()
        for cell in cells:
            cell = (int(cell[0]), int(cell[1]))
            if cell not in self._columns and cell not in seen:
                out.append(cell)
                seen.add(cell)
        return out

    def add(self, cells: Sequence[Cell], features: torch.Tensor) -> None:
        """Append newly featurized cells as columns of the stored image.

        Parameters
        ----------
        cells : Sequence[Cell]
            The ``(k, m)`` cells being added; must all be currently missing.
        features : torch.Tensor
            Complex features of shape ``(len(cells), P)`` -- row ``i`` is the
            flattened valid cross-correlation for ``cells[i]`` (the natural
            channel-major layout of :func:`featurize_cells`).
        """
        cells = [(int(k), int(m)) for (k, m) in cells]
        if features.shape[0] != len(cells):
            raise ValueError(
                f"features has {features.shape[0]} rows but {len(cells)} cells given."
            )
        cols = features.transpose(0, 1).to(self.device, self.dtype)  # (P, len(cells))
        if self._Y is None:
            self._Y = cols.contiguous()
        else:
            self._Y = torch.cat([self._Y, cols], dim=1)
        start = len(self._cells)
        for offset, cell in enumerate(cells):
            self._columns[cell] = start + offset
            self._cells.append(cell)

    def reindex_contiguous(self) -> None:
        """Reorder columns into sorted ``(k, m)`` order.

        TODO: figure out how to intertwine this re-ordering with teh feature tiling.
        """
        if self._Y is None:
            return
        order = sorted(range(len(self._cells)), key=lambda c: self._cells[c])
        self._Y = self._Y[:, order].contiguous()
        self._cells = [self._cells[i] for i in order]
        self._columns = {cell: i for i, cell in enumerate(self._cells)}

    def image_view(self, pixel_index: torch.Tensor | None = None) -> torch.Tensor:
        """Return the featurized image, optionally row-selected to a pixel subset.

        Parameters
        ----------
        pixel_index : torch.Tensor, optional
            Long indices into the ``P`` pixel axis (e.g. a shrinking follow-up mask).
            When omitted the full ``(P, num_cols)`` image is returned.

        Returns
        -------
        torch.Tensor
            Complex ``(P_sel, num_cols)`` view in the store's current column layout.
        """
        if pixel_index is None:
            return self._Y

        return self._Y[pixel_index.to(self._Y.device)]


# ---------------------------------------------------------------------------
# Helpers tying the store/tiling to the reconstructor (image features + weights)
# ---------------------------------------------------------------------------


def featurize_cells(
    image: torch.Tensor,
    reconstructor: ProjectionReconstructor,
    cells: Sequence[Cell],
    *,
    feature_chunk: int | None = None,
    **polar_to_cart_kwargs: Any,
) -> torch.Tensor:
    """Cross-correlate an image against the kernels for a set of feature cells.

    # The expensive image operation behind the store: builds the Cartesian kernel for
    # each ``(k, m)`` cell and cross-correlates the image, returning the result in the
    # channel-major ``(len(cells), P)`` layout consumed by
    # :meth:`FeaturizedImageStore.add`.

    Parameters
    ----------
    image : torch.Tensor
        Real image of shape ``(H, W)`` (single image; ``P = out_h * out_w``).
    reconstructor : ProjectionReconstructor
        Builds the kernels and supplies the polar transform / device.
    cells : Sequence[Cell]
        ``(k, m)`` cells to featurize.
    feature_chunk : int, optional
        Kernel-correlation chunk size (see :func:`build_multichannel_correlogram`).
    **polar_to_cart_kwargs
        Forwarded to kernel construction.

    Returns
    -------
    torch.Tensor
        Complex features of shape ``(len(cells), P)``.
    """
    indices = torch.tensor([[int(k), int(m)] for (k, m) in cells], dtype=torch.long)
    z = build_block_feature_stack(
        image,
        reconstructor,
        indices=indices,
        feature_chunk=feature_chunk,
        **polar_to_cart_kwargs,
    )

    _b, n_cells, out_h, out_w = z.shape

    return z.reshape(n_cells, out_h * out_w)


def build_layout_weights(
    reconstructor: ProjectionReconstructor,
    column_index: dict[Cell, int],
    hypotheses: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Build ``W = U * S`` weights laid out to match a store's column layout.

    Produces the contraction weights for every featurized cell, ordered so column
    ``c`` of the returned tensor corresponds to the cell at column ``c`` of the
    store's featurized image -- ready to pair with :meth:`FeaturizedImageStore.image_view`
    inside :meth:`FeatureTiling.run`.

    Parameters
    ----------
    reconstructor : ProjectionReconstructor
        Supplies the on-device ``U`` / ``S`` tensors.
    column_index : dict[Cell, int]
        ``cell -> column`` layout from :attr:`FeaturizedImageStore.column_index`.
    hypotheses : torch.Tensor, optional
        Long indices selecting a subset of the flattened ``(FF * O)`` template rows
        (the hypothesis batch). Defaults to all rows.

    Returns
    -------
    torch.Tensor
        Complex64 weights of shape ``(N, r)`` where ``r = len(column_index)`` and
        ``N`` is the number of selected hypotheses.
    """
    # Cells in column order so column c of W aligns with column c of the image.
    ordered = sorted(column_index, key=lambda cell: column_index[cell])
    indices = torch.tensor([[k, m] for (k, m) in ordered], dtype=torch.long)
    weights = build_block_weights(reconstructor, indices)  # (FF, O, r)

    ff, o, r = weights.shape
    w_flat = weights.reshape(ff * o, r)

    if hypotheses is not None:
        w_flat = w_flat[hypotheses.to(w_flat.device)]

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
    hypotheses: torch.Tensor | None = None,
    pixel_index: torch.Tensor | None = None,
    pixel_batch: int | None = None,
    hyp_batch: int | None = None,
    n_psi: int | None = None,
    feature_chunk: int | None = None,
    follow_up_fn: (
        Callable[[dict[str, torch.Tensor], torch.Tensor], torch.Tensor | None] | None
    ) = None,
    **polar_to_cart_kwargs: Any,
):
    r"""Incremental SVD-2DTM search; yields per-pixel statistics per stage.

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
    4.  **Tiles (innermost)** -- :meth:`FeatureTiling.run` sweeps the rectangular
        groups, each a batched ``bmm`` accumulating into the spectrum ``C``.

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
    hypotheses : torch.Tensor, optional
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
        store = FeaturizedImageStore(n_px, result.k_max, device=device)

    if hypotheses is None:
        hypotheses = torch.arange(
            result.num_fourier_filters * result.num_orientations, device=device
        )
    if pixel_index is None:
        pixel_index = torch.arange(n_px, device=device)

    px = pixel_index
    for rects in stage_rectangles:  # (1) stages -- increasing rank
        # -- image-feature reuse: featurize ONLY genuinely new cells --
        needed = FeatureTiling.cells_for_rectangles(rects)
        todo = store.missing(needed)
        if todo:
            feats = featurize_cells(
                image,
                reconstructor,
                todo,
                feature_chunk=feature_chunk,
                **polar_to_cart_kwargs,
            )
            store.add(todo, feats)
            store.reindex_contiguous()

        tiling = FeatureTiling(rects, store.column_index, store.n_freq, device=device)
        w_layout = build_layout_weights(reconstructor, store.column_index)  # (FF*O, r)

        # Accumulate this stage's statistics over its pixel set.
        p_step = pixel_batch if pixel_batch is not None else int(px.numel())
        h_step = hyp_batch if hyp_batch is not None else int(hypotheses.numel())

        stage_stats: list[dict[str, torch.Tensor]] = []
        for p0 in range(0, int(px.numel()), p_step):  # (2) pixel batches
            px_b = px[p0 : p0 + p_step]
            Y_flat = store.image_view(px_b)  # (P_b, r), no recompute
            stats = PixelStats(int(px_b.numel()), device)

            for h0 in range(0, int(hypotheses.numel()), h_step):  # (3) hyp batches
                hyp_b = hypotheses[h0 : h0 + h_step]
                W_flat = w_layout[hyp_b]  # (N_b, r)
                C = tiling.run(Y_flat, W_flat)  # (4) tiles -> spectrum (P_b, N_b, K)
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


def build_block_weights(
    reconstructor: ProjectionReconstructor,
    indices: torch.Tensor,
) -> torch.Tensor:
    r"""Build contraction weights ``W = U * S`` for arbitrary (k_idx, eig_idx) pairs.

    # The matched-filter conjugation lands on ``W`` inside
    # :func:`contract_weights_to_correlations`, so these weights are **not**
    # pre-conjugated here.

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


# ---------------------------------------------------------------------------
# Stage 1 -- multi-channel cross-correlogram
# ---------------------------------------------------------------------------


def build_multichannel_correlogram(
    image: torch.Tensor,  # shape (B, H, W)
    kernels: torch.Tensor,  # shape (r, kH, kW)
    *,
    out: torch.Tensor | None = None,  # shape (B, r, out_H, out_W)
    feature_chunk: int | None = None,
) -> torch.Tensor:  # shape (B, r, out_H, out_W)
    r"""Cross-correlate an image against a Cartesian feature stack.

    # For every kernel ``g_l`` in the stack, computes the valid cross-correlation
    # ``Z_l(i,j) = (X \star g_l)(i,j)``. This is the only stage that operates at
    # full image dimensions.

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


# ---------------------------------------------------------------------------
# Stage 2+3 (fused) -- weighted contraction + kappa->psi inverse transform
# ---------------------------------------------------------------------------


def contract_weights_to_correlations(
    z: torch.Tensor,  # shape (..., L)
    weights: torch.Tensor,  # shape (N, L)
    k_indices: torch.Tensor,  # shape (L,), angular-frequency block of each channel
    *,
    k_max: int,
    n_psi: int,
    weights_indices: torch.Tensor | None = None,  # (M,) subset of rows; default all
) -> torch.Tensor:  # shape (..., M, n_psi)
    r"""Fuse the SVD weight contraction with the ``\kappa -> \psi`` inverse transform.

    For each selected template row ``n`` and each pixel in the leading batch,
    contracts the ``L`` feature channels against the weights, groups the products
    by angular frequency ``k`` into ``k_max`` blocks, and inverse-FFTs over ``k`` to
    recover the matched-filter correlation at all ``n_psi`` in-plane angles in one
    shot::

        c_n(psi_a) = sum_k [ sum_{l : k_l = k} z_l * conj(W[n, l]) ] * e^{-i k psi_a}

    The ``sum_l`` is a vectorised ``index_add_`` into the ``k`` axis (no sorting of
    ``k_indices`` required), and the ``sum_k e^{-i k psi}`` is realised by an
    :func:`torch.fft.irfft` over the ``k_max`` non-negative angular-frequency bins
    (real-valued decompositions only).

    Parameters
    ----------
    z : torch.Tensor
        Complex per-pixel features of shape ``(..., L)``; leading dims are an
        arbitrary pixel batch.
    weights : torch.Tensor
        Complex contraction weights ``W = U * S`` of shape ``(N, L)`` (the
        ``(FF, O, L)`` output of :func:`build_block_weights` flattened to
        ``N = FF * O`` rows).  **Not** pre-conjugated.
    k_indices : torch.Tensor
        Shape ``(L,)`` mapping each feature channel to its angular-frequency block
        in ``[0, k_max)`` (i.e. the ``k_idx`` column of ``indices``).
    k_max : int
        Number of non-negative angular-frequency blocks (``result.k_max``).
    n_psi : int
        Number of in-plane samples ``N_\psi`` to synthesise (``= 2*(k_max-1)`` for
        a band-limited real decomposition, but passed explicitly for odd lengths).
    weights_indices : torch.Tensor, optional
        Shape ``(M,)`` selecting a subset of the ``N`` template rows to evaluate
        (sub-blocking the search). Defaults to all ``N`` rows.

    Returns
    -------
    torch.Tensor
        Real float32 correlations of shape ``(..., M, n_psi)``; output index ``a``
        along the last axis is in-plane angle ``\psi_a = 360 * a / n_psi`` degrees.
    """
    w = weights if weights_indices is None else weights[weights_indices]  # (M, L)
    k_idx = k_indices.to(device=z.device, dtype=torch.long)

    # (..., 1, L) * (M, L) -> (..., M, L); conj forms the matched filter (W = U*S).
    prod = z.unsqueeze(-2) * w.conj()

    # Group features by angular frequency into k_max blocks: (..., M, k_max).
    c_hat = torch.zeros((*prod.shape[:-1], k_max), dtype=prod.dtype, device=prod.device)
    c_hat.index_add_(-1, k_idx, prod)

    # kappa -> psi: mf(psi_a) = sum_k c_hat_k e^{-i k psi_a}.  Conjugating the
    # input flips irfft's ``+`` analysis sign to the required ``-`` and Hermitian-
    # extends k<0; forward norm leaves the kernels' 1/sqrt(N_psi) intact.
    correlations = torch.fft.irfft(c_hat.conj(), n=n_psi, dim=-1, norm="forward")
    return correlations * _PSI_IRFFT_SCALE


# ---------------------------------------------------------------------------
# Stage 4 -- per-pixel 2DTM search statistics
# ---------------------------------------------------------------------------

# Per-pixel statistic maps, grouped by output dtype. ``search_2dtm`` pre-allocates
# one tensor per key and feeds pixel-batch slices to ``compute_search_statistics``.
STAT_FLOAT_KEYS: tuple[str, ...] = ("mip", "mean", "variance", "std", "zscore")
STAT_INDEX_KEYS: tuple[str, ...] = ("best_ff", "best_orientation", "best_psi")


def allocate_search_maps(
    batch_shape: tuple[int, ...],
    *,
    device: torch.device | str | None = None,
    real_dtype: torch.dtype = torch.float32,
    index_dtype: torch.dtype = torch.long,
) -> dict[str, torch.Tensor]:
    """Pre-allocate the per-pixel 2DTM statistic maps written by the search.

    Parameters
    ----------
    batch_shape : tuple[int, ...]
        Leading (pixel) dimensions of the maps, e.g. ``(B, n_px)``.
    device : torch.device or str, optional
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


# ---------------------------------------------------------------------------
# High-level driver -- Stages 2-4 streamed over pixel batches
# ---------------------------------------------------------------------------


# def _search_pixel_batch(
#     feats: torch.Tensor,
#     weights_flat: torch.Tensor,
#     k_indices: torch.Tensor,
#     *,
#     k_max: int,
#     n_psi: int,
#     n_ff: int,
#     n_or: int,
#     score: str,
#     out: dict[str, torch.Tensor],
# ) -> None:
#     r"""Run Stages 3-4 for one pixel batch, writing stats into ``out`` slices.

#     This is the per-batch tensor program that :func:`search_2dtm` streams over the
#     spatial axis: the fused contraction + ``\kappa -> \psi`` transform
#     (:func:`contract_weights_to_correlations`) followed by the 2DTM reduction
#     (:func:`compute_search_statistics`) -- all writing into the pre-allocated
#     ``out`` views.

#     Parameters
#     ----------
#     feats : torch.Tensor
#         Pixel-batch features ``(B, npx, L)``.
#     weights_flat : torch.Tensor
#         SVD contraction weights flattened to ``(FF * O, L)``.
#     k_indices : torch.Tensor
#         Shape ``(L,)`` angular-frequency block of each feature channel.
#     k_max, n_psi : int
#         Block count and number of in-plane samples (see
#         :func:`contract_weights_to_correlations`).
#     n_ff, n_or : int
#         Fourier-filter and orientation counts, to unflatten the template axis
#         back to ``(FF, O)`` for the statistics reduction.
#     score : str
#         Real-score reduction for complex decompositions.
#     out : dict[str, torch.Tensor]
#         Pixel-batch slices of the statistic maps to write in place.
#     """
#     # (B, npx, FF*O, n_psi)
#     c_psi = contract_weights_to_correlations(
#         feats, weights_flat, k_indices, k_max=k_max, n_psi=n_psi
#     )
#     # (B, npx, FF, O, n_psi) so compute_search_statistics can decode best_ff/orient.
#     c_psi = c_psi.unflatten(-2, (n_ff, n_or))
#     compute_search_statistics(c_psi, score=score, out=out)


# def search_2dtm(
#     image: torch.Tensor,
#     reconstructor: ProjectionReconstructor,
#     *,
#     indices: torch.Tensor | None = None,
#     kernels: torch.Tensor | None = None,
#     feature_chunk: int | None = None,
#     feature_device: torch.device | str | None = None,
#     pixel_batch: int | None = None,
#     score: str = "real",
#     **polar_to_cart_kwargs: object,
# ) -> dict[str, torch.Tensor]:
#     r"""Full SVD-2DTM search: per-pixel statistic maps over ``(FF, O, \psi)``.

#     Runs Stage 1 once (:func:`build_multichannel_correlogram`), then streams
#     Stages 2-4 over ``pixel_batch`` pixels at a time, accumulating per-pixel
#     2DTM statistics.  The full ``(FF, O, N_\psi, H, W)`` correlogram is never
#     materialised.

#     Parameters
#     ----------
#     image : torch.Tensor
#         Real image of shape ``(H, W)`` or ``(B, H, W)``.
#     reconstructor : ProjectionReconstructor
#         Holds the SVD tensors, polar transform, and device.
#     indices : torch.Tensor, optional
#         Shape ``(L, 2)`` integer array of ``(k_idx, eig_idx)`` pairs. If omitted,
#         uses all available components (full dense result).
#     kernels : torch.Tensor, optional
#         Pre-built Cartesian feature kernels ``(L, kH, kW)`` (e.g. from
#         :func:`build_block_kernels`) to reuse across images.  When omitted they
#         are built from ``reconstructor`` (with ``indices`` / ``polar_to_cart_kwargs``).
#     feature_chunk : int, optional
#         Stage-1 kernel correlation chunk size.
#     feature_device : torch.device or str, optional
#         Device for the Stage-1 correlogram. Allowed to be different than image device
#         for memory management. Data transfers are managed automatically. Defaults to the
#         image's device.
#     pixel_batch : int, optional
#         Number of output pixels processed per Stage 2-4 pass.  Bounds the
#         transient ``(B, pixel_batch, FF, O, N_\psi)`` memory.  Defaults to all
#         pixels at once.
#     score : {"real", "abs"}, optional
#         Real-score reduction for complex decompositions (see
#         :func:`compute_search_statistics`).
#     **polar_to_cart_kwargs
#         Forwarded to kernel construction (ignored if ``kernels`` is given).

#     Returns
#     -------
#     dict[str, torch.Tensor]
#         Per-pixel maps of shape ``(B, out_H, out_W)`` (or ``(out_H, out_W)`` for
#         2-D input): ``"mip"``, ``"best_ff"``, ``"best_orientation"``,
#         ``"best_psi"``, ``"mean"``, ``"variance"``, ``"std"``, ``"zscore"``,
#         plus scalar ``"count"``.  Decode ``best_psi`` with :func:`psi_degrees`
#         and ``best_orientation`` with ``result.phi_values`` / ``theta_values``.
#     """
#     device = image.device
#     result = reconstructor.result
#     n_psi = result.num_angular_components

#     if image.dim() == 2:
#         had_batch = False
#         image = image.unsqueeze(0)
#     elif image.dim() == 3:
#         had_batch = True
#     else:
#         raise ValueError(
#             f"image must have shape (H, W) or (B, H, W), got {tuple(image.shape)}"
#         )
#     batch_size = image.shape[0]

#     if kernels is None:
#         kernels = build_block_kernels(reconstructor, indices, **polar_to_cart_kwargs)

#     out_h = image.shape[-2] - kernels.shape[-2] + 1
#     out_w = image.shape[-1] - kernels.shape[-1] + 1

#     # Handle different storage device for correlogram
#     if feature_device is not None:
#         feature_device = torch.device(feature_device)
#         if feature_device != device:
#             z = torch.empty(
#                 (batch_size, kernels.shape[0], out_h, out_w),
#                 dtype=torch.complex64,
#                 device=feature_device,
#             )
#             out = z
#         else:
#             out = None
#     else:
#         out = None

#     z = build_multichannel_correlogram(
#         image,
#         kernels,
#         feature_chunk=feature_chunk,
#         out=out,
#     )
#     b, L, _out_h, _out_w = z.shape
#     weights = build_block_weights(reconstructor, indices)  # (FF, O, L)
#     n_ff, n_or = weights.shape[0], weights.shape[1]
#     weights_flat = weights.reshape(n_ff * n_or, L)  # (FF*O, L)
#     k_indices = indices[:, 0].to(device)

#     n_px = out_h * out_w
#     # (B, ) === Outer image batch dimension
#     # (P, ) === Flattened pixel batch dimension (the search-space outer batch)
#     z_flat = z.reshape(b, L, n_px)  # (B, L, P)

#     # Pre-allocate the per-pixel statistic maps once.
#     maps = allocate_search_maps((b, n_px), device=device)

#     batch = pixel_batch if pixel_batch is not None else n_px
#     for p0 in tqdm(range(0, n_px, batch), unit="px", desc="iterating pixels"):
#         p1 = min(p0 + batch, n_px)
#         # (B, L, npx) -> (B, npx, L)
#         feats = z_flat[..., p0:p1].to(device).permute(0, 2, 1)
#         out_slices = {key: maps[key][:, p0:p1] for key in maps}
#         _search_pixel_batch(
#             feats,
#             weights_flat,
#             k_indices,
#             k_max=result.k_max,
#             n_psi=n_psi,
#             n_ff=n_ff,
#             n_or=n_or,
#             score=score,
#             out=out_slices,
#         )

#     # Reshape the flattened pixel maps back to the spatial grid.
#     search_maps: dict[str, torch.Tensor] = {}
#     for key, value in maps.items():
#         spatial = value.view(b, out_h, out_w)
#         search_maps[key] = spatial if had_batch else spatial[0]

#     # Scalar size of the searched (FF, O, psi) space, shared by every pixel.
#     search_maps["count"] = torch.tensor(n_ff * n_or * n_psi, device=device)

#     return search_maps


# # ---------------------------------------------------------------------------
# # Explicit psi-resolved correlogram (memory-hungry; small problems / inspection)
# # ---------------------------------------------------------------------------


# def compute_psi_correlogram(
#     image: torch.Tensor,
#     reconstructor: ProjectionReconstructor,
#     *,
#     indices: torch.Tensor | None = None,
#     kernels: torch.Tensor | None = None,
#     feature_chunk: int | None = None,
#     **polar_to_cart_kwargs: object,
# ) -> torch.Tensor:
#     r"""Materialise the full explicit \psi-resolved correlogram ``c(\omega', \psi, i, j)``.

#     Routes through the same :func:`contract_weights_to_correlations` core as
#     :func:`search_2dtm`, but keeps every ``(FF, O, \psi)`` value rather than
#     reducing to statistics.

#     .. warning::
#         The output has shape ``(B, FF, O, N_\psi, out_H, out_W)`` and scales as
#         ``O(FF * O * N_\psi * out^2)`` -- tens of GB for realistic search spaces.
#         For full searches prefer :func:`search_2dtm` (reduced statistics).  This
#         function is intended for small problems, verification, and inspection.

#     Parameters
#     ----------
#     image : torch.Tensor
#         Real image of shape ``(H, W)`` or ``(B, H, W)``.
#     reconstructor : ProjectionReconstructor
#         Holds the SVD tensors, polar transform, and device.
#     indices : torch.Tensor, optional
#         Shape ``(L, 2)`` integer array of ``(k_idx, eig_idx)`` pairs. If omitted,
#         uses all available components.
#     kernels : torch.Tensor, optional
#         Pre-built Cartesian feature kernels ``(L, kH, kW)`` to reuse.
#     feature_chunk : int, optional
#         Stage-1 kernel correlation chunk size.
#     **polar_to_cart_kwargs
#         Forwarded to kernel construction (ignored if ``kernels`` is given).

#     Returns
#     -------
#     torch.Tensor
#         ``(B, FF, O, N_\psi, out_H, out_W)`` if ``image`` was batched, else
#         ``(FF, O, N_\psi, out_H, out_W)``.  Real float32.  Output index ``a`` along
#         the \psi axis corresponds to ``\psi_a = 360 * a / N_\psi`` degrees (see
#         :func:`psi_degrees`).
#     """
#     device = reconstructor.device
#     result = reconstructor.result
#     n_psi = result.num_angular_components
#     had_batch = image.dim() == 3

#     if kernels is None:
#         kernels = build_block_kernels(reconstructor, indices, **polar_to_cart_kwargs)

#     z = build_multichannel_correlogram(image, kernels, feature_chunk=feature_chunk)
#     b, L, out_h, out_w = z.shape
#     weights = build_block_weights(reconstructor, indices)  # (FF, O, L)
#     n_ff, n_or = weights.shape[0], weights.shape[1]
#     weights_flat = weights.reshape(n_ff * n_or, L)
#     k_indices = indices[:, 0].to(device)

#     # (B, P, L) -> core -> (B, P, FF*O, n_psi)
#     z_flat = z.permute(0, 2, 3, 1).reshape(b, out_h * out_w, L)
#     c = contract_weights_to_correlations(
#         z_flat, weights_flat, k_indices, k_max=result.k_max, n_psi=n_psi
#     )
#     # (B, P, FF*O, n_psi) -> (B, FF, O, n_psi, out_H, out_W)
#     c = c.reshape(b, out_h, out_w, n_ff, n_or, n_psi).permute(0, 3, 4, 5, 1, 2)
#     return c if had_batch else c[0]


# # ---------------------------------------------------------------------------
# # Memory planning utility
# # ---------------------------------------------------------------------------


# def estimate_search_memory_bytes(
#     result: DecompositionResult,
#     image_shape: tuple[int, ...],
#     *,
#     indices: torch.Tensor | None = None,
#     pixel_batch: int | None = None,
# ) -> dict[str, int]:
#     r"""Estimate peak memory for a :func:`search_2dtm` pass.

#     All figures are approximate and exclude allocator overhead.  Use this to
#     pick ``pixel_batch`` (and to compare against the full
#     :func:`compute_psi_correlogram` tensor).

#     Parameters
#     ----------
#     result : DecompositionResult
#         Provides shape metadata only; nothing is allocated.
#     image_shape : tuple[int, ...]
#         ``(B, H, W)`` or ``(H, W)``.
#     indices : torch.Tensor, optional
#         Shape ``(L, 2)`` integer array of ``(k_idx, eig_idx)`` pairs. If omitted,
#         uses all available components.
#     pixel_batch : int, optional
#         Pixels per Stage 2-4 pass; defaults to all output pixels.

#     Returns
#     -------
#     dict[str, int]
#         Byte estimates keyed by ``"feature_stack"`` (resident ``Z``),
#         ``"chat_batch"`` (one Stage-2 pixel batch), ``"psi_batch"`` (one Stage-3
#         pixel batch), ``"stat_maps"`` (the resident output maps),
#         ``"full_correlogram"`` (the dense :func:`compute_psi_correlogram`
#         tensor, for comparison), and ``"total_peak"`` (``Z`` + one ``chat`` +
#         one ``psi`` batch + maps).
#     """
#     cplx = 8  # complex64 bytes/element
#     real = 4  # float32 bytes/element

#     # Determine L from indices or full result
#     if indices is None:
#         nb = result.k_max * 2 if result.is_complex_projection else result.k_max
#         L = nb * result.eig_max
#     else:
#         L = indices.shape[0]

#     n_ff = result.num_fourier_filters
#     n_or = result.num_orientations
#     n_psi = result.num_angular_components

#     b = image_shape[0] if len(image_shape) == 3 else 1
#     k = result.num_radial_components
#     out_h = image_shape[-2] - k + 1
#     out_w = image_shape[-1] - k + 1
#     out_px = max(out_h, 0) * max(out_w, 0)

#     npx = pixel_batch if pixel_batch is not None else out_px
#     npx = min(npx, out_px) if out_px else 0
#     psi_bpe = cplx if result.is_complex_projection else real

#     feature_stack = b * L * out_px * cplx
#     chat_batch = b * npx * n_ff * n_or * result.k_max * cplx
#     psi_batch = b * npx * n_ff * n_or * n_psi * psi_bpe
#     stat_maps = b * out_px * real * 8
#     full_correlogram = b * n_ff * n_or * n_psi * out_px * psi_bpe

#     return {
#         "feature_stack": feature_stack,
#         "chat_batch": chat_batch,
#         "psi_batch": psi_batch,
#         "stat_maps": stat_maps,
#         "full_correlogram": full_correlogram,
#         "total_peak": feature_stack + chat_batch + psi_batch + stat_maps,
#     }
