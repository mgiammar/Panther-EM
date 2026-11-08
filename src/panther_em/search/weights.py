"""``U * S`` weights of a decomposition, device-resident once, served as blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from panther_em.decomposition.rectangles import Rectangle
    from panther_em.decomposition.result import DecompositionBase
    from panther_em.search.correlation.layout import FeatureLayout


class WeightSource:
    """``U * S`` weights, device-resident once, served as rectangular blocks.

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
        self.device = torch.device(device)
        self.n_filters = int(result.num_fourier_filters)
        self.n_orientations = int(result.num_orientations)
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
        self.k_start, self.m_start, self.k_stop, self.m_stop = k0, m0, k1, m1

        ks, ms = np.meshgrid(np.arange(k0, k1), np.arange(m0, m1), indexing="ij")
        u, s, _ = result.get_component(ks.reshape(-1), ms.reshape(-1), return_vh=False)
        assert u is not None and s is not None
        us = (u * s[None, None, :]).reshape(
            self.n_filters * self.n_orientations, k1 - k0, m1 - m0
        )
        self.us: torch.Tensor = torch.as_tensor(us, dtype=torch.complex64).to(
            self.device
        )

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
        """``(N_sel, k_extent, m_extent)`` complex64 block of ``U * S`` for ``rect``."""
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
