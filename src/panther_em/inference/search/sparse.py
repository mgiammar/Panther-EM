"""Sparse evaluation of ``(pixel, template)`` pairs at full rank (algorithm 3, stage 2).

Only the pairs a dense stage emitted are scored, so the per-pixel moments of the
dense loop have no counterpart here: the session reuses the emitting stage's mean and
variance when it standardises the sparse scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from panther_em.inference.search.store import FeatureStore
from panther_em.inference.search.types import PixelSelection

if TYPE_CHECKING:
    from collections.abc import Sequence

    from panther_em.inference.search.basis import Template
    from panther_em.inference.search.featurize import Featurizer
    from panther_em.inference.search.pairs import PairList
    from panther_em.inference.search.schedule import StageSpec
    from panther_em.inference.search.store import ColumnGroup
    from panther_em.inference.search.types import ImageSet


@dataclass(eq=False)
class PairScores:
    """Per-pair maxima over psi and defocus planes, aligned with ``pairs``.

    Attributes
    ----------
    pairs : PairList
    value : torch.Tensor
        ``(M,)`` float32 correlation maxima (dense-``mip`` units).
    psi : torch.Tensor
        ``(M,)`` int64 in-plane index (dense convention, ``psi_degrees(n_psi)``).
    plane : torch.Tensor
        ``(M,)`` int64 global defocus plane of the maximum.
    n_psi : int
    """

    pairs: PairList
    value: torch.Tensor
    psi: torch.Tensor
    plane: torch.Tensor
    n_psi: int

    def best_per_pixel(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(pixels (U,), index (U,))``: each pixel's best pair (first on ties)."""
        order = torch.argsort(self.value, descending=True, stable=True)
        pixel = self.pairs.pixel[order]
        pixels, first = torch.unique(pixel, return_inverse=True)
        best = torch.full_like(pixels, int(order.numel())).scatter_reduce_(
            0, first, torch.arange(order.numel(), device=order.device), "amin"
        )
        return pixels, order[best]

    def global_ids(self) -> torch.Tensor:
        """Global hypothesis id ``plane * n_templates + template`` of every pair."""
        return self.plane * self.pairs.n_templates + self.pairs.template


class SparsePairEvaluator:
    """Scores pairs through the ordinary featurizer and ``U * S`` weights.

    Pixels are processed in chunks (one single-block :class:`FeatureStore` per chunk
    and plane, dropped afterwards) and pairs in sub-chunks: each pair costs one
    gathered feature column (``r`` complex values) and one dot product per angular
    frequency, then an inverse FFT over psi.

    Parameters
    ----------
    template : Template
    featurizer : Featurizer
    pixel_chunk : int
        Pixels featurized per chunk (bounds the chunk store: ``r * elem`` per pixel).
    pair_chunk : int
        Pairs contracted per call (bounds the gathered operands: ``2 r elem`` each).
    cell_chunk : int
        Forwarded to the featurizer.
    """

    def __init__(
        self,
        template: Template,
        featurizer: Featurizer,
        *,
        pixel_chunk: int = 2048,
        pair_chunk: int = 1024,
        cell_chunk: int = 64,
    ) -> None:
        self.template = template
        self.featurizer = featurizer
        self.pixel_chunk = int(pixel_chunk)
        self.pair_chunk = int(pair_chunk)
        self.cell_chunk = int(cell_chunk)

    @torch.no_grad()
    def evaluate(
        self,
        images: ImageSet,
        pairs: PairList,
        spec: StageSpec,
        *,
        planes: Sequence[int] | None = None,
    ) -> PairScores:
        """Score ``pairs`` with ``spec``'s rectangles over ``planes`` (default: all)."""
        from panther_em.inference.search.types import SearchGeometry

        device = self.template.device
        geometry = SearchGeometry.from_images(images, self.template.template_shape)
        planes = tuple(range(images.n_defocus)) if planes is None else tuple(planes)
        pairs = pairs.to(device)
        value = torch.full((len(pairs),), float("-inf"), device=device)
        psi = torch.zeros(len(pairs), dtype=torch.long, device=device)
        plane_of = torch.zeros(len(pairs), dtype=torch.long, device=device)
        weights = [
            self.template.weights.block(rect) for rect in spec.rectangles
        ]  # (N, ke, me) complex64 each
        pixels = pairs.pixels()
        for p0 in range(0, int(pixels.numel()), self.pixel_chunk):
            chunk = pixels[p0 : p0 + self.pixel_chunk]
            selection = PixelSelection(chunk.cpu(), geometry)
            schedule = self.featurizer.schedule(geometry, selection)
            column_of = torch.full((geometry.n_pixels,), -1, dtype=torch.long)
            column_of[schedule.column_index] = torch.arange(len(selection))
            in_chunk = torch.nonzero(torch.isin(pairs.pixel, chunk)).squeeze(-1)
            cols = column_of.to(device)[pairs.pixel[in_chunk]]
            for plane in planes:
                store = FeatureStore(
                    geometry,
                    selection,
                    schedule.column_index,
                    block_pixels=len(selection),
                    n_defocus=images.n_defocus,
                    device=device,
                )
                groups = []
                for rect in spec.rectangles:
                    group = store.add_group(rect, spec.precision, planes=(plane,))
                    self.featurizer.produce(
                        images,
                        group,
                        store.writer(group),
                        schedule,
                        chunk=self.cell_chunk,
                    )
                    groups.append(group)
                for q0 in range(0, int(in_chunk.numel()), self.pair_chunk):
                    sub = in_chunk[q0 : q0 + self.pair_chunk]
                    spectrum = torch.zeros(
                        (int(sub.numel()), spec.k_stop),
                        dtype=torch.complex64,
                        device=device,
                    )
                    for group, w in zip(groups, weights, strict=True):
                        k0, _, ke, _ = group.rect
                        spectrum[:, k0 : k0 + ke] += self._contract(
                            store,
                            group,
                            cols[q0 : q0 + self.pair_chunk],
                            w,
                            pairs.template[sub],
                        )
                    corr = torch.fft.irfft(
                        spectrum, n=spec.n_psi, dim=-1, norm="forward"
                    )
                    best, amax = corr.max(dim=1)
                    better = best > value[sub]
                    value[sub] = torch.where(better, best, value[sub])
                    psi[sub] = torch.where(
                        better, (spec.n_psi - amax) % spec.n_psi, psi[sub]
                    )
                    plane_of[sub] = torch.where(
                        better, torch.full_like(amax, plane), plane_of[sub]
                    )
        return PairScores(pairs, value, psi, plane_of, spec.n_psi)

    @staticmethod
    def _contract(
        store: FeatureStore,
        group: ColumnGroup,
        cols: torch.Tensor,
        weights: torch.Tensor,
        templates: torch.Tensor,
    ) -> torch.Tensor:
        """``(M, ke)`` complex64 spectrum of ``(cols, templates)`` within ``group``."""
        engine = group.engine
        a = store.gather_columns(group, cols)[0].permute(1, 0, 2)  # (M, ke, inner)
        b = engine.prepare_weights(
            weights
        )  # (ke, inner, N * c): c = 1 (complex) or 2 (fp16 re/im)
        c = int(b.shape[-1]) // int(weights.shape[0])
        b = b.view(b.shape[0], b.shape[1], -1, c)[:, :, templates]  # (ke, inner, M, c)
        out = torch.einsum("mki,kimc->mkc", a, b)
        if c == 1:
            return out[..., 0].to(torch.complex64)
        return torch.view_as_complex(out.to(torch.float32).contiguous())
