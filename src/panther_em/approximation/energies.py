r"""Truncation errors of a decomposition from its spectrum alone (the cheap tier).

A template row's polar image :math:`x` (``A`` angles) is stored as its angular rfft
blocks :math:`X_k` (``norm="ortho"``), so by Parseval

.. math:: \lVert x \rVert^2 = \sum_k w_k \lVert X_k \rVert^2, \qquad
   w_0 = w_{A/2} = 1, \; w_k = 2 \text{ otherwise}

(``w_{A/2}`` only for even ``A``). The SVD of block ``k`` writes :math:`X_{i,k}` on the
orthonormal right singular vectors ``Vh[k, e]`` with the coefficients
``U[i, k, e] * S[k, e]``, so the energy a set of kept cells captures is
:math:`\sum w_k |U S|^2` over the cells, and what a row loses is its energy
(:attr:`~panther_em.decomposition.DecompositionResult.block_energy`, recorded before
the ``k_max`` crop) minus that. The relative error splits in quadrature,
:math:`\rho^2 = \rho_k^2 + \rho_m^2`:

* :math:`\rho_k` -- the angular truncation: the energy of the blocks no kept cell
  lies in (including every block past ``k_max``, which the decomposition never
  stored);
* :math:`\rho_m` -- the eigenvector truncation: what the blocks the cells reach lose
  to the components left out.

These are errors of the polar images. The warp to Cartesian preserves energy only
approximately, so the exact Cartesian errors
(:func:`~panther_em.approximation.errors.compute_errors`) also carry the warp's own
round-trip error; the cheap tier is meant for comparing and choosing ranks
(:func:`~panther_em.approximation.selection.suggest_rectangles`). Being a difference
of float32 energies, it resolves :math:`\rho` down to about ``1e-3``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from panther_em.decomposition.rectangles import as_rectangles

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from panther_em.decomposition.rectangles import Rectangle
    from panther_em.decomposition.result import DecompositionBase, DecompositionResult


def parseval_weights(num_angle: int, num_blocks: int) -> np.ndarray:
    """``(num_blocks,)`` float64 weights of the angular rfft blocks of ``num_angle``.

    ``1`` for ``k = 0`` and (even ``num_angle``) the Nyquist block ``num_angle / 2``,
    ``2`` for the others, which stand for a frequency and its negative.
    """
    w = np.full(num_blocks, 2.0)
    w[0] = 1.0
    if num_angle % 2 == 0 and num_blocks > num_angle // 2:
        w[num_angle // 2] = 1.0
    return w


def _checked(result: DecompositionBase) -> tuple[DecompositionResult, np.ndarray]:
    """The dense result and its ``(rows, blocks)`` block energies; raise if absent."""
    if result.is_complex_projection:
        raise NotImplementedError(
            "truncation errors are implemented for real projections only"
        )
    if result.block_energy is None:
        raise ValueError(
            "the decomposition records no block energies (decomposed before they "
            "were stored); decompose it again"
        )
    blocks = np.asarray(result.block_energy, dtype=np.float64)
    return result.to_dense(), blocks.reshape(-1, blocks.shape[-1])


def _weights(result: DecompositionBase) -> np.ndarray:
    return parseval_weights(result.num_angular_components, result.num_angular_blocks)


def _captured_chunks(
    result: DecompositionResult, cells: np.ndarray, chunk: int
) -> Iterator[tuple[slice, np.ndarray]]:
    """``(rows, (n, K, E) float64)``: ``w_k |U S|^2`` of the kept ``cells``, chunked.

    ``cells`` is the ``(K, E)`` 0/1 mask of the kept cells over the stored grid's
    leading corner; rows run over ``(volume, filter, orientation)`` in C order.
    """
    k_stop, e_stop = cells.shape
    w = _weights(result)[:k_stop, None] * cells
    weight = w * np.asarray(result.S, dtype=np.float64)[:k_stop, :e_stop] ** 2
    n_or = result.num_orientations
    for v in range(result.num_volumes):
        for f in range(result.num_fourier_filters):
            base = (v * result.num_fourier_filters + f) * n_or
            for o0 in range(0, n_or, chunk):
                o1 = min(o0 + chunk, n_or)
                u = np.asarray(result.U[v, f, o0:o1, :k_stop, :e_stop])
                power = u.real.astype(np.float64) ** 2 + u.imag.astype(np.float64) ** 2
                yield slice(base + o0, base + o1), power * weight


def _cell_mask(
    result: DecompositionBase, rects: tuple[Rectangle, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """``(K, E)`` mask of the kept cells, ``(blocks,)`` mask of the reached blocks."""
    k_stop = max(k0 + ke for k0, _, ke, _ in rects)
    e_stop = max(m0 + me for _, m0, _, me in rects)
    if k_stop > result.k_max or e_stop > result.eig_max:
        raise ValueError(
            f"rectangles reach ({k_stop}, {e_stop}) beyond the stored "
            f"(k_max={result.k_max}, eig_max={result.eig_max})"
        )
    cells = np.zeros((k_stop, e_stop))
    reached = np.zeros(result.num_angular_blocks, dtype=bool)
    for k0, m0, ke, me in rects:
        cells[k0 : k0 + ke, m0 : m0 + me] = 1.0
        reached[k0 : k0 + ke] = True
    return cells, reached


def _sqrt_ratio(loss: np.ndarray, energy: np.ndarray) -> np.ndarray:
    out: np.ndarray = np.sqrt(loss / energy)
    return out


@dataclass(frozen=True)
class TruncationError:
    """Per-row polar truncation error of a selection of feature rectangles.

    Rows run over ``(volume, Fourier filter, orientation)`` in C order (the error
    tables' template rows for one volume).

    Parameters
    ----------
    rectangles : tuple of Rectangle
        The kept ``(k_start, m_start, k_extent, m_extent)`` regions.
    energy : numpy.ndarray
        ``(rows,)`` energy of each row's polar image.
    angular_loss : numpy.ndarray
        ``(rows,)`` energy of the blocks no rectangle reaches.
    eigen_loss : numpy.ndarray
        ``(rows,)`` energy the reached blocks lose to the left-out components.
    """

    rectangles: tuple[Rectangle, ...]
    energy: np.ndarray
    angular_loss: np.ndarray
    eigen_loss: np.ndarray

    @property
    def rho(self) -> np.ndarray:
        """``(rows,)`` relative error ``sqrt(rho_k^2 + rho_m^2)``."""
        return _sqrt_ratio(self.angular_loss + self.eigen_loss, self.energy)

    @property
    def rho_k(self) -> np.ndarray:
        """``(rows,)`` angular part of :attr:`rho`."""
        return _sqrt_ratio(self.angular_loss, self.energy)

    @property
    def rho_m(self) -> np.ndarray:
        """``(rows,)`` eigenvector part of :attr:`rho`."""
        return _sqrt_ratio(self.eigen_loss, self.energy)

    def summary(self) -> dict[str, float]:
        """Pooled errors (``sqrt`` of total loss over total energy), the worst row."""
        total = float(self.energy.sum())
        angular, eigen = float(self.angular_loss.sum()), float(self.eigen_loss.sum())
        return {
            "rho": float(np.sqrt((angular + eigen) / total)),
            "rho_k": float(np.sqrt(angular / total)),
            "rho_m": float(np.sqrt(eigen / total)),
            "rho_max": float(self.rho.max()),
        }


def truncation_error(
    result: DecompositionBase, rectangles: Sequence[Rectangle], *, chunk: int = 256
) -> TruncationError:
    """Per-row truncation error of ``rectangles`` (see the module docstring).

    Parameters
    ----------
    result : DecompositionBase
        A real-projection decomposition with recorded ``block_energy``. A sparse
        result counts only its stored components.
    rectangles : sequence of Rectangle
        Kept ``(k_start, m_start, k_extent, m_extent)`` regions (overlaps count once).
    chunk : int
        Orientations read from ``U`` at a time. Default 256.
    """
    dense, blocks = _checked(result)
    rects = as_rectangles(rectangles)
    cells, reached = _cell_mask(dense, rects)
    weighted = blocks * _weights(dense)
    captured = np.zeros(blocks.shape[0])
    for rows, power in _captured_chunks(dense, cells, chunk):
        captured[rows] = power.sum(axis=(1, 2))
    return TruncationError(
        rectangles=rects,
        energy=weighted.sum(axis=1),
        angular_loss=weighted[:, ~reached].sum(axis=1),
        eigen_loss=np.maximum(weighted[:, reached].sum(axis=1) - captured, 0.0),
    )


@dataclass(frozen=True)
class EnergySpectrum:
    """Energies pooled over every row, for the errors of leading-corner rectangles.

    Parameters
    ----------
    captured : numpy.ndarray
        ``(k_max, eig_max)`` energy ``sum_rows w_k |U S|^2`` of each stored cell.
    blocks : numpy.ndarray
        ``(num_angular_blocks,)`` energy ``sum_rows w_k block_energy`` of each block.
    """

    captured: np.ndarray
    blocks: np.ndarray

    @property
    def total(self) -> float:
        """Energy of every row together."""
        return float(self.blocks.sum())

    def rho_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pooled ``(rho, rho_k, rho_m)`` of every rectangle ``(0, 0, K, M)``.

        Each is ``(k_max, eig_max)``, entry ``[K - 1, M - 1]`` the rectangle's.
        """
        total = self.total
        kept = self.captured.cumsum(axis=1).cumsum(axis=0)
        reached = self.blocks.cumsum()[: kept.shape[0], None]
        rho = np.sqrt(np.maximum(total - kept, 0.0) / total)
        rho_k = np.sqrt(np.maximum(total - reached, 0.0) / total)
        rho_m = np.sqrt(np.maximum(reached - kept, 0.0) / total)
        return rho, np.broadcast_to(rho_k, rho.shape), rho_m


def energy_spectrum(result: DecompositionBase, *, chunk: int = 256) -> EnergySpectrum:
    """The pooled :class:`EnergySpectrum` of ``result`` (one pass over ``U``)."""
    dense, blocks = _checked(result)
    cells = np.ones((dense.k_max, dense.eig_max))
    captured = np.zeros(cells.shape)
    for _, power in _captured_chunks(dense, cells, chunk):
        captured += power.sum(axis=0)
    return EnergySpectrum(
        captured=captured, blocks=(blocks * _weights(dense)).sum(axis=0)
    )
