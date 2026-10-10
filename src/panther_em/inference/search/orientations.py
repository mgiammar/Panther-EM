"""Orientation neighbourhoods for pair dilation, via ``torch_so3``'s HEALPix grid.

The decomposition only stores Euler angles. When those angles are HEALPix pixel
centers (any order, subsets included) :meth:`S2Neighborhood.from_space` recovers the
grid and pixel ids, and ring neighbourhoods come straight from
``HierarchicalS2Grid.neighbor_indices``. :func:`healpix_orientations` generates such
angles for a new decomposition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch_so3.hierarchical_s2_grid import (  # type: ignore[import-untyped]
    HierarchicalS2Grid,
)

if TYPE_CHECKING:
    from panther_em.inference.search.hypotheses import HypothesisSpace


def healpix_orientations(nside: int) -> tuple[np.ndarray, np.ndarray]:
    """``(phi_deg, theta_deg)`` of every HEALPix pixel at ``nside``, NESTED order.

    Float32, in the argument order of ``PolarProjectionDecomposer``; orientation
    index == NESTED pixel id.
    """
    theta, phi = HierarchicalS2Grid(nside, 1).get_level_orientations(0)
    return phi.astype(np.float32), theta.astype(np.float32)


def _unit_vectors(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    theta, phi = np.radians(theta_deg), np.radians(phi_deg)
    return np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1
    )


class S2Neighborhood:
    """Ring neighbourhoods of a template's orientations on a single-level grid.

    Parameters
    ----------
    grid : HierarchicalS2Grid
        One-level grid (``n_levels=1``).
    ipix : np.ndarray
        ``(O,)`` NESTED pixel id of every orientation.
    """

    def __init__(self, grid: HierarchicalS2Grid, ipix: np.ndarray) -> None:
        self.grid = grid
        self.ipix = np.asarray(ipix, dtype=np.int64)
        self._orientation_of = np.full(grid.get_level(0).npix, -1, dtype=np.int64)
        self._orientation_of[self.ipix] = np.arange(self.ipix.size)
        self._tables: dict[int, torch.Tensor] = {}

    @classmethod
    def from_space(
        cls, space: HypothesisSpace, *, tol_deg: float = 1e-3, max_nside: int = 4096
    ) -> S2Neighborhood | None:
        """Recover the grid from the space's ``(phi, theta)``; ``None`` if not HEALPix.

        The smallest ``nside`` whose pixel centers coincide with every orientation
        (within ``tol_deg``, all distinct) wins.
        """
        if space.phi_deg is None or space.theta_deg is None:
            return None
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
                        return cls(grid, ipix)
            nside *= 2
        return None

    @property
    def n_orientations(self) -> int:
        """``O``."""
        return int(self.ipix.size)

    def neighbor_table(
        self, rings: int, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """``(O, n_max)`` int64: orientations within ``rings`` hops of each one.

        Rows are sorted and ``-1`` padded, include the orientation itself, and skip
        grid pixels that are not among the template's orientations. Cached per
        ``rings``.
        """
        table = self._tables.get(rings)
        if table is None:
            rows = []
            for pix in self.ipix:
                o = self._orientation_of[self.grid.neighbor_indices(pix, 0, rings)]
                rows.append(torch.from_numpy(np.sort(o[o >= 0])))
            table = torch.nn.utils.rnn.pad_sequence(
                rows, batch_first=True, padding_value=-1
            )
            self._tables[rings] = table
        return table if device is None else table.to(device)
