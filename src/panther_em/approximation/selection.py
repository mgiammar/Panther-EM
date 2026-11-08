"""Rank suggestions: which feature rectangle meets an error target or a budget.

The suggestions read the cheap tier (:mod:`panther_em.approximation.energies`): the
pooled polar truncation error of every leading-corner rectangle ``(0, 0, K, M)`` from
one pass over the decomposition. They know nothing of the search's noise or cost;
check a chosen rectangle per row with
:func:`~panther_em.approximation.energies.truncation_error` and exactly with
:func:`~panther_em.approximation.errors.compute_errors`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from panther_em.approximation.energies import EnergySpectrum, energy_spectrum

if TYPE_CHECKING:
    from panther_em.decomposition.rectangles import Rectangle
    from panther_em.decomposition.result import DecompositionBase


@dataclass(frozen=True)
class RectangleCandidate:
    """One suggested rectangle and its pooled truncation errors.

    Parameters
    ----------
    rectangle : Rectangle
        ``(0, 0, k_extent, m_extent)``.
    components : int
        Kept cells, ``k_extent * m_extent``.
    rho, rho_k, rho_m : float
        Pooled relative error and its angular and eigenvector parts.
    """

    rectangle: Rectangle
    components: int
    rho: float
    rho_k: float
    rho_m: float


def suggest_rectangles(
    result: DecompositionBase | EnergySpectrum,
    *,
    max_error: float | None = None,
    max_components: int | None = None,
    max_k_extent: int | None = None,
) -> list[RectangleCandidate]:
    """Leading-corner rectangles on the error-versus-size frontier.

    Parameters
    ----------
    result : DecompositionBase or EnergySpectrum
        A decomposition with recorded block energies, or its
        :func:`~panther_em.approximation.energies.energy_spectrum` (to suggest
        several times from one pass).
    max_error : float, optional
        Target pooled relative error ``rho``.
    max_components : int, optional
        Budget of kept cells ``k_extent * m_extent``.
    max_k_extent : int, optional
        Largest ``k_extent`` to consider. Default the stored ``k_max``.

    Returns
    -------
    list of RectangleCandidate
        Increasing in ``components``, decreasing in ``rho``; empty when nothing meets
        the targets.
    """
    if max_error is None and max_components is None:
        raise ValueError("give max_error, max_components or both")
    if max_error is not None and max_error <= 0:
        raise ValueError(f"max_error must be positive, got {max_error}")
    if max_components is not None and max_components < 1:
        raise ValueError(f"max_components must be >= 1, got {max_components}")

    spectrum = result if isinstance(result, EnergySpectrum) else energy_spectrum(result)
    rho, rho_k, rho_m = spectrum.rho_grid()
    k_max, eig_max = rho.shape
    k_limit = k_max if max_k_extent is None else min(max_k_extent, k_max)

    candidates: list[RectangleCandidate] = []
    for k in range(1, k_limit + 1):
        m_limit = eig_max
        if max_components is not None:
            m_limit = min(eig_max, max_components // k)
        if m_limit < 1:
            break
        m = m_limit
        if max_error is not None:
            meets = (rho[k - 1, :m_limit] <= max_error).nonzero()[0]
            if meets.size == 0:
                continue
            m = int(meets[0]) + 1
        i, j = k - 1, m - 1
        candidates.append(
            RectangleCandidate(
                rectangle=(0, 0, k, m),
                components=k * m,
                rho=float(rho[i, j]),
                rho_k=float(rho_k[i, j]),
                rho_m=float(rho_m[i, j]),
            )
        )

    frontier: list[RectangleCandidate] = []
    for c in sorted(candidates, key=lambda c: (c.components, c.rho)):
        if not frontier or c.rho < frontier[-1].rho:
            frontier.append(c)
    return frontier
