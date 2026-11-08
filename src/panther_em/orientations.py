"""Orientation grids: HEALPix out-of-plane orientations and uniform in-plane angles.

A template's out-of-plane orientations are HEALPix pixel centers in NESTED order
(:func:`healpix_orientations`), so that an orientation's index is its pixel id and a
coarser sampling level is a stride over the ids (:func:`healpix_level_ids`). The
in-plane angle ``psi`` is sampled uniformly (:func:`psi_degrees`).

* :func:`healpix_orientations` -- ``(phi, theta)`` of every pixel at ``nside``.
* :func:`healpix_ipix` -- recognize arbitrary orientations as HEALPix pixel centers.
* :func:`healpix_level_ids` -- the orientations of a coarser sampling level.
* :func:`ring_neighborhoods` -- each pixel's neighbors out to a number of rings.
* :func:`psi_degrees` -- uniform in-plane angles.
"""

from __future__ import annotations

import healpy as hp
import numpy as np
import torch

__all__ = [
    "healpix_ipix",
    "healpix_level_ids",
    "healpix_orientations",
    "psi_degrees",
    "ring_neighborhoods",
]


def psi_degrees(n_psi: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Uniformly spaced in-plane angles in degrees, ``(n_psi,)`` float32."""
    return 360.0 * torch.arange(n_psi, device=device, dtype=torch.float32) / n_psi


def healpix_orientations(nside: int) -> tuple[np.ndarray, np.ndarray]:
    """``(phi_deg, theta_deg)`` of every HEALPix pixel at ``nside``, NESTED order.

    Float32, in the argument order of ``PolarProjectionDecomposer``; orientation
    index == NESTED pixel id, which is what :func:`healpix_level_ids` relies on.
    """
    from torch_so3.hierarchical_s2_grid import HierarchicalS2Grid

    theta, phi = HierarchicalS2Grid(nside, 1).get_level_orientations(0)
    return phi.astype(np.float32), theta.astype(np.float32)


def _unit_vectors(theta_deg: np.ndarray, phi_deg: np.ndarray) -> np.ndarray:
    theta, phi = np.radians(theta_deg), np.radians(phi_deg)
    return np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1
    )


def _as_float64(values: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64).reshape(-1)


def healpix_ipix(
    phi_deg: np.ndarray | torch.Tensor,
    theta_deg: np.ndarray | torch.Tensor,
    *,
    tol_deg: float = 1e-3,
    max_nside: int = 4096,
) -> tuple[int, np.ndarray] | None:
    """``(nside, ipix)`` when the orientations are HEALPix pixel centers.

    The smallest ``nside`` whose NESTED pixel centers coincide with every orientation
    (within ``tol_deg``, all distinct) wins; ``None`` when there is none.

    Parameters
    ----------
    phi_deg, theta_deg : numpy.ndarray or torch.Tensor
        ``(O,)`` orientation angles in degrees.
    tol_deg : float
        Largest allowed angle between an orientation and its pixel center.
    max_nside : int
        Largest ``nside`` tried (powers of two from 1).
    """
    from torch_so3.hierarchical_s2_grid import HierarchicalS2Grid

    phi = _as_float64(phi_deg)
    theta = _as_float64(theta_deg)
    if phi.shape != theta.shape:
        raise ValueError(
            f"phi_deg and theta_deg differ in shape: {phi.shape} vs {theta.shape}"
        )
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


def unique_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row sorted unique non-negative values of a 2-D array (``-1`` = absent).

    Parameters
    ----------
    values : numpy.ndarray
        ``(N, M)`` int64 array of values, where ``-1`` marks absent values.

    Returns
    -------
    table : numpy.ndarray
        ``(N, W)`` int64 array of each row's distinct values in ascending order, padded
        by repeating the first value.
    counts : numpy.ndarray
        ``(N,)`` int64 array of each row's distinct value count (0 = no value).
    """
    big = np.iinfo(np.int64).max
    s = np.sort(np.where(values < 0, big, values), axis=1)
    dup = np.zeros_like(s, dtype=bool)
    dup[:, 1:] = s[:, 1:] == s[:, :-1]
    s = np.sort(np.where(dup, big, s), axis=1)
    counts = (s != big).sum(axis=1)
    width = max(1, int(counts.max()))
    s = s[:, :width]
    first = np.where(counts > 0, s[:, 0], -1)
    return np.where(s == big, first[:, None], s), counts


def ring_neighborhoods(nside: int, rings: int) -> np.ndarray:
    """Every NESTED pixel's neighbors out to ``rings`` rings at ``nside``.

    Returns
    -------
    numpy.ndarray
        ``(12 nside^2, W)`` int64 pixel ids of each pixel's neighborhood (itself
        included) in ascending order, padded by repeating the first id.
    """
    if rings < 0:
        raise ValueError("rings must be >= 0")
    npix = 12 * int(nside) ** 2
    ring = np.arange(npix, dtype=np.int64)[:, None]
    for _ in range(int(rings)):
        # healpy spells it this way
        nb = hp.get_all_neighbours(int(nside), ring.reshape(-1), nest=True)
        nb = np.asarray(nb, dtype=np.int64).T.reshape(npix, -1)
        ring, _ = unique_rows(np.concatenate([ring, nb], axis=1))
    return ring
