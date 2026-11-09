"""``U * S`` weights of a decomposition, device-resident once, served as blocks.

:class:`TensorWeights` is any ``(N, K, M)`` set of template weights held as one
device tensor and served as rectangular blocks (the interface the correlation reads:
``block``, ``n_templates``, ``device``); :class:`WeightSource` loads a decomposition's
``U * S`` into one. Other sets (the cell-mean seed templates of
:mod:`~panther_em.search.seeding`) subclass :class:`TensorWeights` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    from panther_em.decomposition.rectangles import Rectangle
    from panther_em.decomposition.result import DecompositionBase
    from panther_em.search.correlation.layout import FeatureLayout


class TensorWeights:
    """Template weights as one ``(N, K, M)`` complex64 tensor, served as blocks.

    Notes
    -----
    Row ``n = f * n_orientations + o`` holds Fourier filter ``f`` of orientation
    ``o``; the tensor covers the ``(k, m)`` box ``[k_start, k_stop) x [m_start,
    m_stop)`` of feature space.

    Parameters
    ----------
    us : torch.Tensor
        ``(N, K, M)`` complex weights (moved to complex64 on its device).
    n_filters, n_orientations : int
        ``FF`` and ``O`` with ``FF * O == N``.
    k_start, m_start : int
        Feature-space origin of the tensor's box.
    key : Hashable
        Identity of the set for caches that may hold several sets; ``None`` marks a
        correlator's own (primary) weights.
    """

    def __init__(
        self,
        us: torch.Tensor,
        *,
        n_filters: int,
        n_orientations: int,
        k_start: int = 0,
        m_start: int = 0,
        key: Hashable = None,
    ) -> None:
        if us.dim() != 3:
            raise ValueError(f"weights must be (N, K, M), got {tuple(us.shape)}")
        self.n_filters = int(n_filters)
        self.n_orientations = int(n_orientations)
        if self.n_filters * self.n_orientations != int(us.shape[0]):
            raise ValueError(
                f"{self.n_filters} filters x {self.n_orientations} orientations != "
                f"{int(us.shape[0])} weight rows"
            )
        self.us: torch.Tensor = us.to(torch.complex64)
        self.device = self.us.device
        self.k_start, self.m_start = int(k_start), int(m_start)
        self.k_stop = self.k_start + int(us.shape[1])
        self.m_stop = self.m_start + int(us.shape[2])
        self.key = key

    @property
    def n_templates(self) -> int:
        """``FF * O``."""
        return self.n_filters * self.n_orientations

    @property
    def nbytes(self) -> int:
        """Resident bytes."""
        return self.us.numel() * self.us.element_size()

    def block(
        self, rect: Rectangle, template_rows: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``(N_sel, k_extent, m_extent)`` complex64 block of ``rect``."""
        k0, m0, k_ext, m_ext = (int(v) for v in rect)
        if (
            k0 < self.k_start
            or m0 < self.m_start
            or k0 + k_ext > self.k_stop
            or m0 + m_ext > self.m_stop
        ):
            raise ValueError(f"rectangle {rect} lies outside the resident box")
        view = self.us[
            :,
            k0 - self.k_start : k0 - self.k_start + k_ext,
            m0 - self.m_start : m0 - self.m_start + m_ext,
        ]
        if template_rows is None:
            return view
        return view.index_select(0, template_rows.to(self.device))

    def layout_weights(
        self, layout: FeatureLayout, template_rows: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``(N_sel, r)`` weights in ``layout``'s compacted slot order."""
        blocks = [
            self.block(rect, template_rows).reshape(-1, rect[2] * rect[3])
            for rect in layout.rectangles
        ]
        return torch.cat(blocks, dim=1)


class WeightSource(TensorWeights):
    """A decomposition's ``U * S`` weights, device-resident once.

    Parameters
    ----------
    result : DecompositionBase
        Dense or sparse single-volume decomposition (sparse pairs absent from the
        stored set are zeros, as in :meth:`~DecompositionBase.get_component`).
    device : torch.device or str
        Where the weights live.
    cells_union : Rectangle or sequence of Rectangle, optional
        Bounding box of every rectangle any stage will contract; only that box of
        ``(k, m)`` space is resident. Defaults to the whole stored grid.
    """

    def __init__(
        self,
        result: DecompositionBase,
        device: torch.device | str,
        cells_union: Rectangle | Sequence[Rectangle] | None = None,
    ) -> None:
        if result.is_complex_projection:
            raise NotImplementedError("Complex-valued projections not yet supported")
        if result.num_volumes != 1:
            raise NotImplementedError(
                f"the search supports one volume per decomposition, got "
                f"{result.num_volumes}"
            )
        self.result = result
        n_filters = int(result.num_fourier_filters)
        n_orientations = int(result.num_orientations)
        k_blocks, e_max = int(result.k_max), int(result.eig_max)
        if cells_union is None:
            k0, m0, k1, m1 = 0, 0, k_blocks, e_max
        else:
            rects: list[Rectangle] = (
                [cells_union]  # type: ignore[list-item]
                if isinstance(cells_union[0], int)
                else list(cells_union)  # type: ignore[arg-type]
            )
            k0 = min(r[0] for r in rects)
            m0 = min(r[1] for r in rects)
            k1 = max(r[0] + r[2] for r in rects)
            m1 = max(r[1] + r[3] for r in rects)
            if k1 > k_blocks or m1 > e_max:
                raise ValueError(
                    f"cells_union {(k0, m0, k1, m1)} exceeds the stored "
                    f"(k_max={k_blocks}, "
                    f"eig_max={e_max}) grid"
                )

        ks, ms = np.meshgrid(np.arange(k0, k1), np.arange(m0, m1), indexing="ij")
        u, s, _ = result.get_component(ks.reshape(-1), ms.reshape(-1), return_vh=False)
        assert u is not None and s is not None
        us = (u * s[None, None, :]).reshape(
            n_filters * n_orientations, k1 - k0, m1 - m0
        )
        super().__init__(
            torch.as_tensor(us, dtype=torch.complex64).to(torch.device(device)),
            n_filters=n_filters,
            n_orientations=n_orientations,
            k_start=k0,
            m_start=m0,
        )
