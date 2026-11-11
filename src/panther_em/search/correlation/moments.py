"""Closed-form per-pixel moments over a pass's whole hypothesis set.

The hypothesis loop's moment sums over ``(plane, template, psi)`` are, by Parseval,
quadratic forms of the pixel's features. With ``C_dnk = sum_m Y_dkm conj(W_nkm)``
and the reduction's convention (the DC bin and a Nyquist bin enter by their real
parts only, every other bin twice)::

    sum_{n, psi} corr   = n_psi sum_d sum_n Re C_dn0              = n_psi sum_d v_d0 . u
    sum_{n, psi} corr^2 = n_psi sum_d sum_n [(Re C_dn0)^2 + sum_{k>=1} a_k |C_dnk|^2]
                        = n_psi sum_d sum_k v_dk^T Q_k v_dk

where ``v_dk`` is the real row of the features the contraction engine feeds its
GEMM (``[Re Y | Im Y]`` for the fp16 engine, interleaved ``(re, im)`` pairs for the
complex engines), ``a_k`` is 2 (1 at a Nyquist bin, real part only) and ``Q_k``,
``u`` are sums over the template set of outer products of the weights' matching
real columns. The forms are built once per template set (:meth:`MomentForms.build`)
and cost one pass over a store (``num_k x inner^2`` multiply-adds per pixel and
plane); they make the mean and variance of the *whole* set available without
evaluating it, which is what lets a local stage score only a neighborhood.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from panther_em.search.correlation.contraction import ContractionEngine
    from panther_em.search.correlation.layout import FeatureLayout
    from panther_em.search.correlation.loop import LoopPlan
    from panther_em.search.weights import WeightSource


def real_weight_columns(
    w: torch.Tensor, engine: ContractionEngine
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real ``(num_k, N, inner)`` columns ``(a, b)``: ``Re C = v.a``, ``Im C = v.b``.

    ``w`` are the un-conjugated ``(N, num_k, num_m)`` weights and ``v`` the engine's
    real operand row (``inner = 2 num_m``): ``[Re Y | Im Y]`` for the fp16 engine,
    ``(Re Y_0, Im Y_0, Re Y_1, ...)`` for the complex engines.
    """
    wr, wi = w.real.to(torch.float32), w.imag.to(torch.float32)
    if engine.precision == "fp16":
        a = torch.cat([wr, wi], dim=-1)
        b = torch.cat([-wi, wr], dim=-1)
    else:
        n, num_k, num_m = w.shape
        a = torch.stack([wr, wi], dim=-1).reshape(n, num_k, 2 * num_m)
        b = torch.stack([-wi, wr], dim=-1).reshape(n, num_k, 2 * num_m)
    return a.permute(1, 0, 2).contiguous(), b.permute(1, 0, 2).contiguous()


def real_operand(a: torch.Tensor, engine: ContractionEngine) -> torch.Tensor:
    """The float32 ``(num_k, P, inner)`` real rows of a GEMM A operand (any engine)."""
    if engine.precision == "fp16":
        return a.to(torch.float32)
    return torch.view_as_real(a).reshape(a.shape[0], a.shape[1], -1).to(torch.float32)


@dataclass(frozen=True, eq=False)
class RunForms:
    """The forms of one k-run of a layout.

    Attributes
    ----------
    k_start : int
        First angular frequency of the run.
    q : torch.Tensor
        ``(num_k, inner, inner)`` float32 symmetric forms with the bin weights folded
        in: ``sum_n (Re C_n0)^2`` at ``k = 0``, ``sum_n (Re C_nk)^2`` at a Nyquist bin
        and ``2 sum_n |C_nk|^2`` elsewhere.
    u : torch.Tensor or None
        ``(inner,)`` float32 ``sum_n a_n0`` when the run holds ``k = 0``; else ``None``.
    """

    k_start: int
    q: torch.Tensor
    u: torch.Tensor | None


class MomentForms:
    """Quadratic forms of one template set for one layout, engine and ``n_psi``.

    Build with :meth:`build`; evaluate a store block with :func:`block_moments`.

    Parameters
    ----------
    runs : sequence of RunForms
        One entry per ``layout.k_runs`` entry, in order.
    n_templates : int
        Template rows the forms sum over (``N_sel``).
    n_psi : int
        In-plane samples (the sums carry the factor and its Nyquist convention).
    """

    def __init__(self, runs: Sequence[RunForms], n_templates: int, n_psi: int) -> None:
        self.runs = tuple(runs)
        self.n_templates = int(n_templates)
        self.n_psi = int(n_psi)
        if not self.runs:
            raise ValueError("MomentForms needs at least one run")

    @property
    def nbytes(self) -> int:
        """Bytes held by the forms."""
        return sum(
            r.q.numel() * 4 + (0 if r.u is None else r.u.numel() * 4) for r in self.runs
        )

    @classmethod
    @torch.no_grad()
    def build(
        cls,
        weights: WeightSource,
        layout: FeatureLayout,
        engine: ContractionEngine,
        n_psi: int,
        *,
        template_rows: torch.Tensor | None = None,
        chunk: int = 4096,
    ) -> MomentForms:
        """Sum the forms over ``template_rows`` (``None`` = every row) of ``weights``.

        Parameters
        ----------
        weights : WeightSource
            The template's ``U * S`` weights.
        layout : FeatureLayout
            The stage's rectangles (one form set per k-run).
        engine : ContractionEngine
            Fixes the real operand ordering the forms must match.
        n_psi : int
            In-plane samples: the bin ``n_psi // 2`` (when the layout reaches it) is
            the Nyquist bin.
        template_rows : torch.Tensor, optional
            Template rows of the set.
        chunk : int, optional
            Template rows per partial sum.
        """
        rows = (
            torch.arange(weights.n_templates, device=weights.device)
            if template_rows is None
            else template_rows.to(weights.device, torch.long)
        )
        n_sel = int(rows.numel())
        nyquist = int(n_psi) // 2 if int(n_psi) % 2 == 0 else -1
        runs: list[RunForms] = []
        for run in layout.k_runs:
            inner = 2 * run.num_m
            q_re = torch.zeros(run.num_k, inner, inner, device=weights.device)
            q_im = torch.zeros_like(q_re)
            u = torch.zeros(inner, device=weights.device) if run.k_start == 0 else None
            for j0 in range(0, n_sel, int(chunk)):
                sub = rows[j0 : j0 + int(chunk)]
                # per part, as the operand concatenates each part's real row
                # (fp16: [Re Y_1 | Im Y_1 | Re Y_2 | Im Y_2])
                columns = [
                    real_weight_columns(
                        weights.block(
                            (
                                run.k_start,
                                part.m_start,
                                run.num_k,
                                part.m_stop - part.m_start,
                            ),
                            sub,
                        ),
                        engine,
                    )
                    for part in run.parts
                ]
                a = torch.cat([c[0] for c in columns], dim=-1)  # (num_k, n, inner)
                b = torch.cat([c[1] for c in columns], dim=-1)
                q_re += torch.bmm(a.transpose(1, 2), a)
                q_im += torch.bmm(b.transpose(1, 2), b)
                if u is not None:
                    u += a[0].sum(dim=0)
            k = torch.arange(run.k_start, run.k_stop, device=weights.device)
            real_only = (k == 0) | (k == nyquist)
            weight_re = torch.where(real_only, 1.0, 2.0)
            weight_im = torch.where(real_only, 0.0, 2.0)
            q = weight_re[:, None, None] * q_re + weight_im[:, None, None] * q_im
            runs.append(RunForms(run.k_start, q.contiguous(), u))
        return cls(runs, n_sel, n_psi)

    @torch.no_grad()
    def accumulate(
        self,
        operands: Sequence[torch.Tensor],
        engine: ContractionEngine,
        s1: torch.Tensor,
        s2: torch.Tensor,
        *,
        chunk: int = 4096,
    ) -> None:
        """Add one plane's moment sums into ``s1`` / ``s2`` (``(P,)`` float32).

        ``operands[i]`` is the ``(num_k, P, inner)`` GEMM A operand of run ``i``
        (parts already concatenated along ``inner``), as the hypothesis loop feeds it.
        """
        for forms, a in zip(self.runs, operands, strict=True):
            n_pixels = int(a.shape[1])
            for p0 in range(0, n_pixels, int(chunk)):
                p1 = min(p0 + int(chunk), n_pixels)
                v = real_operand(a[:, p0:p1], engine)  # (K, Pc, inner)
                quad = (torch.bmm(v, forms.q) * v).sum(dim=(0, 2))  # (Pc,)
                s2[p0:p1] += self.n_psi * quad
                if forms.u is not None:
                    s1[p0:p1] += self.n_psi * (v[0] @ forms.u)


@torch.no_grad()
def block_moments(
    plan: LoopPlan,
    forms: MomentForms,
    operands: Mapping[int, torch.Tensor],
    n_valid: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(s1, s2)`` ``(n_valid,)`` of one store block over the pass's planes and runs.

    Assembles the operands as the hypothesis loop does (``loop._loop_body``), so the
    sums cover exactly the hypotheses the loop would enumerate for the block.
    """
    engine = plan.engine
    device = next(iter(operands.values())).device
    s1 = torch.zeros(n_valid, dtype=torch.float32, device=device)
    s2 = torch.zeros_like(s1)
    for d_local in range(plan.hyps.n_defocus_selected):
        per_run = []
        for run in plan.layout.k_runs:
            parts = []
            for part in run.parts:
                gid, local_plane = plan.plane_lookup[part.rect][d_local]
                k_start = plan.layout.rectangles[part.rect][0]
                parts.append(
                    engine.operand(
                        operands[gid],
                        local_plane,
                        run.k_start - k_start,
                        run.k_stop - k_start,
                        0,
                        n_valid,
                    )
                )
            per_run.append(parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1))
        forms.accumulate(per_run, engine, s1, s2)
    return s1, s2


def moments_to_stats(
    s1: torch.Tensor, s2: torch.Tensor, n_evaluated: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(mean, variance)`` from moment sums over ``n_evaluated`` correlations."""
    mean = s1 / float(n_evaluated)
    variance = (s2 / float(n_evaluated) - mean * mean).clamp_min(0.0)
    return mean, variance
