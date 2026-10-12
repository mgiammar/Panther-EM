"""Sparse evaluation of ``(pixel, template)`` pairs at full rank (algorithm 3, stage 2).

Only the pairs a dense stage emitted are scored, so the per-pixel moments of the
dense loop have no counterpart here: the session reuses the emitting stage's mean and
variance when it standardises the sparse scores.

Survivors are a small fraction of the pixels, so their features are not produced by
the tiled FFT correlation (which would re-correlate whole tiles per chunk): each
survivor's template-sized patch is cut out of the plane-filtered image and the
center-mode :class:`~panther_em.inference.search.featurize.InnerProductFeaturizer`
contracts the patches with the kernels as if they were a particle stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from panther_em.inference.search.featurize import (
    InnerProductFeaturizer,
    _FilteredImageCache,
)
from panther_em.inference.search.store import FeatureStore
from panther_em.inference.search.types import (
    ImageSet,
    PixelSelection,
    Precision,
    SearchGeometry,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from panther_em.inference.search.basis import Template
    from panther_em.inference.search.pairs import PairList
    from panther_em.inference.search.schedule import StageSpec
    from panther_em.inference.search.store import ColumnGroup


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


def survivor_patches(
    images: ImageSet,
    geometry: SearchGeometry,
    pixel_ids: torch.Tensor,
    plane: int,
    template_shape: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    *,
    cache: _FilteredImageCache | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``(n, kH, kW)`` template-sized patches of the plane-filtered images at pixels.

    Pixel ``(b, r, c)`` of the valid-correlation grid owns the patch
    ``image_b[r:r+kH, c:c+kW]`` (the inner product with a kernel is that pixel's
    feature). Center mode crops the center of each image. ``cache`` supplies the
    plane-filtered images (filtered once per plane when shared across calls) and
    ``out`` receives the patches in place.
    """
    cache = cache or _FilteredImageCache(images, plane, device)
    kh, kw = template_shape
    per_image = geometry.n_pixels_per_image
    pixel_ids = pixel_ids.to(device)
    image_of = pixel_ids // per_image
    n = int(pixel_ids.numel())
    if out is None:
        out = torch.empty((n, kh, kw), device=device, dtype=dtype)
    elif tuple(out.shape) != (n, kh, kw):
        raise ValueError(f"out must have shape {(n, kh, kw)}")
    for image in torch.unique(image_of).tolist():
        where = torch.nonzero(image_of == image).squeeze(-1)
        filtered = cache[image]
        if geometry.mode == "center":
            h, w = filtered.shape
            r0, c0 = (h - kh) // 2, (w - kw) // 2
            out[where] = filtered[r0 : r0 + kh, c0 : c0 + kw].to(out.dtype)
        else:
            ow = geometry.grid[1]
            rem = pixel_ids[where] % per_image
            windows = filtered.unfold(0, kh, 1).unfold(1, kw, 1)  # (oh, ow, kH, kW)
            # the gather materialises float32 patches before the cast (1 MiB each):
            # sub-chunks of 1024 keep that transient at 1 GiB instead of 8 GiB
            for s0 in range(0, int(where.numel()), 1024):
                sel = where[s0 : s0 + 1024]
                r = rem[s0 : s0 + 1024]
                out[sel] = windows[r // ow, r % ow].to(out.dtype)
    return out


def _ceil8(n: int) -> int:
    return -(-n // 8) * 8


class SparsePairEvaluator:
    """Scores pairs through patch inner products and the ``U * S`` weights.

    Pixels are processed in chunks that span **every requested defocus plane**: the
    survivors' patches of all planes form one center-mode image set, the
    inner-product featurizer fills a single-block :class:`FeatureStore` (dropped
    afterwards) with one GEMM pass over the kernels, and pairs are contracted in
    sub-chunks: one gathered feature column (``r`` complex values) per plane and one
    dot product per angular frequency per pair against the template's weight rows,
    gathered once and shared by the planes, then an inverse FFT over psi and a
    maximum over psi and planes. Only one plane survives per pair.

    Parameters
    ----------
    template : Template
    pixel_chunk : int
        Upper bound on survivor pixels per chunk (the chunk's GEMM has
        ``pixels x planes`` columns).
    pair_chunk : int
        Pair-plane products contracted per call (gathered operands: ``2 r elem``
        bytes each; the pair sub-chunk is ``pair_chunk / planes``).
    cell_chunk : int
        Kernels per inner-product GEMM; each GEMM re-reads the chunk's patches, so
        larger chunks raise its arithmetic intensity (512: 1 GiB of kernel rows).
    patch_budget : int
        Bytes of patches per chunk across planes; caps ``pixel_chunk`` at
        ``budget / (planes * patch_bytes)``.
    """

    def __init__(
        self,
        template: Template,
        *,
        pixel_chunk: int = 2048,
        pair_chunk: int = 8192,
        cell_chunk: int = 512,
        patch_budget: int = 8 << 30,
    ) -> None:
        self.template = template
        self.pixel_chunk = int(pixel_chunk)
        self.pair_chunk = int(pair_chunk)
        self.cell_chunk = int(cell_chunk)
        self.patch_budget = int(patch_budget)

    def chunk_pixels(self, n_planes: int, patch_dtype: torch.dtype) -> int:
        """Survivor pixels per chunk under the patch budget."""
        kh, kw = self.template.template_shape
        patch_bytes = kh * kw * torch.empty((), dtype=patch_dtype).element_size()
        return max(1, min(self.pixel_chunk, self.patch_budget // (n_planes * patch_bytes)))

    @torch.no_grad()
    def evaluate(
        self,
        images: ImageSet,
        pairs: PairList,
        spec: StageSpec,
        *,
        planes: Sequence[int] | None = None,
        trace: Callable[[str], None] | None = None,
    ) -> PairScores:
        """Score ``pairs`` with ``spec``'s rectangles over ``planes`` (default: all)."""
        device = self.template.device
        geometry = SearchGeometry.from_images(images, self.template.template_shape)
        planes = tuple(range(images.n_defocus)) if planes is None else tuple(planes)
        n_planes = len(planes)
        pairs = pairs.to(device)
        value = torch.full((len(pairs),), float("-inf"), device=device)
        psi = torch.zeros(len(pairs), dtype=torch.long, device=device)
        plane_of = torch.zeros(len(pairs), dtype=torch.long, device=device)
        weights = [
            self.template.weights.block(rect) for rect in spec.rectangles
        ]  # (N, ke, me) complex64 each
        pixels = pairs.pixels()
        # extracted straight into the GEMM operand dtype: an fp32 copy of 8192 patches
        # is 8.6 GB, the fp16 one 4.3 GB
        patch_dtype = torch.float16 if spec.precision is Precision.FP16 else torch.float32
        chunk_px = self.chunk_pixels(n_planes, patch_dtype)
        kh, kw = self.template.template_shape
        # plane-filtered images once per plane, shared by every chunk
        caches = [_FilteredImageCache(images, plane, device) for plane in planes]
        # GEMM column counts (pixels x planes per chunk) stay multiples of 8: an
        # unaligned leading dimension drops cuBLAS off the tensor-core path (8190
        # columns ran at 64 TFLOPS against 93 for 8192); the pad columns are zero
        # patches whose features nothing reads
        capacity = _ceil8(chunk_px * n_planes)
        featurizer = InnerProductFeaturizer(self.template.kernels, image_batch=capacity)
        patches = torch.zeros((capacity, kh, kw), device=device, dtype=patch_dtype)
        plane_ids = torch.tensor(planes, device=device)
        pair_sub = max(1, self.pair_chunk // n_planes)
        for c, p0 in enumerate(range(0, int(pixels.numel()), chunk_px)):
            chunk = pixels[p0 : p0 + chunk_px]
            n = int(chunk.numel())
            in_chunk = torch.nonzero(torch.isin(pairs.pixel, chunk)).squeeze(-1)
            cols = torch.searchsorted(chunk, pairs.pixel[in_chunk])  # chunk order
            # box j*n + i is pixel i of the chunk under plane j
            for j, cache in enumerate(caches):
                survivor_patches(
                    images, geometry, chunk, planes[j], (kh, kw), device, patch_dtype,
                    cache=cache, out=patches[j * n : (j + 1) * n],
                )
            if trace is not None:
                trace(f"sparse:chunk{c}:patches")
            n_cols = _ceil8(n * n_planes)
            if n_cols > n * n_planes:
                patches[n * n_planes : n_cols].zero_()
            boxes = ImageSet.from_boxes(patches[:n_cols])
            box_geometry = SearchGeometry.from_images(boxes, (kh, kw))
            selection = PixelSelection.all(box_geometry)
            schedule = featurizer.schedule(box_geometry, selection)
            store = FeatureStore(
                box_geometry,
                selection,
                schedule.column_index,
                block_pixels=len(selection),
                n_defocus=1,
                device=device,
            )
            groups = []
            if trace is not None:
                trace(f"sparse:chunk{c}:boxes")
            for rect in spec.rectangles:
                group = store.add_group(rect, spec.precision, planes=(0,))
                featurizer.produce(
                    boxes, group, store.writer(group), schedule, chunk=self.cell_chunk
                )
                groups.append(group)
            del boxes
            if trace is not None:
                trace(f"sparse:chunk{c}:featurized")
            for q0 in range(0, int(in_chunk.numel()), pair_sub):
                sub = in_chunk[q0 : q0 + pair_sub]
                m = int(sub.numel())
                # all planes of the sub-chunk's pairs: column j*n + col for plane j
                all_cols = (
                    cols[q0 : q0 + pair_sub].unsqueeze(0) + n * torch.arange(n_planes, device=device).unsqueeze(1)
                ).reshape(-1)  # (planes * m,), plane-major
                templates = pairs.template[sub]
                spectrum = torch.zeros(
                    (n_planes, m, spec.k_stop), dtype=torch.complex64, device=device
                )
                for group, w in zip(groups, weights, strict=True):
                    k0, _, ke, _ = group.rect
                    spectrum[:, :, k0 : k0 + ke] += self._contract(
                        store, group, all_cols, w, templates, n_planes
                    )
                corr = torch.fft.irfft(spectrum, n=spec.n_psi, dim=-1, norm="forward")
                # one plane survives per pair: the maximum over psi and planes
                best_psi, amax = corr.max(dim=-1)  # (planes, m)
                best, which = best_psi.max(dim=0)  # (m,)
                amax = amax.gather(0, which.unsqueeze(0)).squeeze(0)
                value[sub] = best
                psi[sub] = (spec.n_psi - amax) % spec.n_psi
                plane_of[sub] = plane_ids[which]
            del store, groups
            if trace is not None:
                trace(f"sparse:chunk{c}:contracted")
        return PairScores(pairs, value, psi, plane_of, spec.n_psi)

    @staticmethod
    def _contract(
        store: FeatureStore,
        group: ColumnGroup,
        cols: torch.Tensor,
        weights: torch.Tensor,
        templates: torch.Tensor,
        n_planes: int,
    ) -> torch.Tensor:
        """``(planes, M, ke)`` complex64 spectra of ``(cols, templates)`` within ``group``.

        Per pair: its feature column (gathered from the block, ``ke x m`` complex, one
        per plane) times the conjugate of its template's weight rows, summed over
        ``m``. Both operands are row gathers on contiguous last axes and the
        reduction is one fused multiply-sum, so the pair count sets the cost, not a
        batched GEMM's padding; the weight rows are gathered once and broadcast over
        the planes.
        """
        a = store.gather_columns(group, cols)[0].permute(1, 0, 2)  # (planes*M, ke, inner)
        num_m = int(weights.shape[-1])
        if a.is_complex():
            a_c = a
        else:  # fp16 layout [Yr | Yi]
            a_c = torch.complex(a[..., :num_m].float(), a[..., num_m:].float())
        a_c = a_c.to(torch.complex64).reshape(n_planes, -1, *a_c.shape[1:])
        w = weights.index_select(0, templates.to(weights.device))  # (M, ke, m)
        return (a_c * w.conj().unsqueeze(0)).sum(-1)
