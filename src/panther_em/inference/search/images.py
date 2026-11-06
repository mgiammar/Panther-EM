"""The image side of a search: what is searched, and where.

* :class:`SearchImages` -- a stack of preprocessed images plus optional per-image
  filters (CTF planes) and masks;
* :class:`SearchGeometry` -- the grid of template positions over the stack;
* :class:`PixelSelection` -- a subset of that grid (sorted flat indices).

Plain values: no device work at construction beyond cheap validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, eq=False)
class SearchImages:
    """A batch of already-preprocessed images plus optional per-image inputs.

    The low-level input of a search, without a noise model: enough for stages decided
    by the plain threshold (and for particle boxes). A micrograph whose stages run the
    threshold test is searched as a :class:`~.noise_model.ImageSet`, which the session
    turns into one of these.

    Parameters
    ----------
    images : torch.Tensor
        Real images of shape ``(B, H, W)``; a single ``(H, W)`` image is promoted to
        ``B = 1``. Whitening, normalization and contrast convention are the caller's
        responsibility (Panther-EM applies no preprocessing).
    filters : torch.Tensor, optional
        Fourier filters in rfft layout applied inside featurization, one per defocus
        hypothesis plane: ``(D, H, W // 2 + 1)`` shared by every image, or
        ``(B, D, H, W // 2 + 1)`` per image. Real or complex.
    mask : torch.Tensor, optional
        Boolean selection of valid-correlation pixels to search: ``(out_h, out_w)``
        shared, or ``(B, out_h, out_w)`` per image; in center-only mode ``(B,)``.
        Validated against the template box by :meth:`SearchGeometry.from_images`.
    center_only : bool, optional
        Force the 1x1 (center pixel only) evaluation mode. ``None`` (default) selects
        it automatically when the image box equals the template box.
    """

    images: torch.Tensor
    filters: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    center_only: bool | None = None

    def __post_init__(self) -> None:
        """Validate (never coerce): a real ``(B, H, W)`` stack and matching filters."""
        images = self.images
        if images.dim() != 3:
            raise ValueError(
                f"images must have shape (B, H, W), got {tuple(images.shape)}; use "
                "SearchImages.single() for one (H, W) image"
            )
        if images.is_complex():
            raise ValueError("images must be real-valued")

        b, h, w = images.shape
        if self.filters is not None:
            f = self.filters
            if f.dim() == 3:
                expected = (h, w // 2 + 1)
                if tuple(f.shape[1:]) != expected:
                    raise ValueError(
                        "filters must have shape (D, H, W//2+1) = "
                        f"(D, {h}, {w // 2 + 1}), got {tuple(f.shape)}"
                    )
            elif f.dim() == 4:
                expected4 = (b, h, w // 2 + 1)
                if (f.shape[0], f.shape[2], f.shape[3]) != expected4:
                    raise ValueError(
                        "filters must have shape (B, D, H, W//2+1) = "
                        f"({b}, D, {h}, {w // 2 + 1}), got {tuple(f.shape)}"
                    )
            else:
                raise ValueError(
                    "filters must have shape (D, H, W//2+1) or (B, D, H, W//2+1), got "
                    f"{tuple(f.shape)}"
                )
            if f.shape[-3] < 1:
                raise ValueError("filters must have at least one defocus plane")

    @property
    def n_images(self) -> int:
        """Number of images ``B``."""
        return int(self.images.shape[0])

    @property
    def image_shape(self) -> tuple[int, int]:
        """``(H, W)`` of every image."""
        return int(self.images.shape[-2]), int(self.images.shape[-1])

    @property
    def n_defocus(self) -> int:
        """Number of defocus (filter) planes ``D``; ``1`` when no filters are given."""
        if self.filters is None:
            return 1
        return int(self.filters.shape[-3])

    @property
    def per_image_filters(self) -> bool:
        """Whether ``filters`` carries a leading image axis."""
        return self.filters is not None and self.filters.dim() == 4

    def filter_plane(self, d: int, b: int = 0) -> torch.Tensor | None:
        """The rfft filter for defocus plane ``d`` of image ``b``, or ``None``."""
        if self.filters is None:
            return None
        if self.per_image_filters:
            return self.filters[b, d]
        return self.filters[d]

    def filtered_planes(
        self, b: int, planes: Sequence[int], device: torch.device | str
    ) -> torch.Tensor:
        """``(len(planes), H, W)`` float32 image ``b`` under each plane's filter."""
        image = self.images[b].to(device=device, dtype=torch.float32)
        out = torch.empty(
            (len(planes), *image.shape), dtype=torch.float32, device=device
        )
        for i, d in enumerate(planes):
            filt = self.filter_plane(int(d), b)
            if filt is None:
                out[i] = image
            else:
                out[i] = torch.fft.irfft2(
                    torch.fft.rfft2(image) * filt.to(device), s=tuple(image.shape)
                )
        return out

    @classmethod
    def single(
        cls,
        image: torch.Tensor,
        *,
        filters: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        center_only: bool | None = None,
    ) -> SearchImages:
        """One ``(H, W)`` image."""
        if image.dim() != 2:
            raise ValueError(
                f"single() expects an (H, W) image, got {tuple(image.shape)}"
            )
        return cls(
            image.unsqueeze(0),
            filters=filters,
            mask=mask,
            center_only=center_only,
        )

    @classmethod
    def from_stack(
        cls,
        images: torch.Tensor,
        *,
        filters: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        center_only: bool | None = None,
    ) -> SearchImages:
        """A ``(B, H, W)`` stack of same-size images (micrographs or particle boxes)."""
        if images.dim() != 3:
            raise ValueError(
                f"from_stack() expects a (B, H, W) stack, got {tuple(images.shape)}"
            )
        return cls(images, filters=filters, mask=mask, center_only=center_only)

    # Alias with the particle-stack vocabulary.
    from_boxes = from_stack


@dataclass(frozen=True)
class SearchGeometry:
    """Valid-correlation grid of an image set against a template box.

    Attributes
    ----------
    n_images : int
        ``B``.
    image_shape, template_shape : tuple[int, int]
        ``(H, W)`` and ``(kH, kW)``.
    grid : tuple[int, ...]
        ``(out_h, out_w)`` in grid mode or ``(1,)`` in center-only mode.
    mode : {"grid", "center"}
        Whether the whole valid-correlation grid or only the center pixel of each
        image is evaluated.
    """

    n_images: int
    image_shape: tuple[int, int]
    template_shape: tuple[int, int]
    grid: tuple[int, ...]
    mode: Literal["grid", "center"]

    @classmethod
    def from_images(
        cls, images: SearchImages, template_shape: tuple[int, int]
    ) -> SearchGeometry:
        """Derive the geometry (and evaluation mode) for ``images``."""
        h, w = images.image_shape
        k_h, k_w = int(template_shape[0]), int(template_shape[1])
        if h < k_h or w < k_w:
            raise ValueError(
                f"image {(h, w)} smaller than the template box {(k_h, k_w)}; the valid "
                "cross-correlation is empty."
            )
        center = (
            images.center_only
            if images.center_only is not None
            else (h, w) == (k_h, k_w)
        )
        if center:
            grid: tuple[int, ...] = (1,)
            mode: Literal["grid", "center"] = "center"
        else:
            grid = (h - k_h + 1, w - k_w + 1)
            mode = "grid"
        geometry = cls(
            n_images=images.n_images,
            image_shape=(h, w),
            template_shape=(k_h, k_w),
            grid=grid,
            mode=mode,
        )
        if images.mask is not None:
            geometry.validate_mask(images.mask)
        return geometry

    @property
    def n_pixels_per_image(self) -> int:
        """Pixels evaluated per image."""
        return math.prod(self.grid)

    @property
    def n_pixels(self) -> int:
        """Total pixels across the image set (batch-major flat axis)."""
        return self.n_images * self.n_pixels_per_image

    @property
    def dense_shape(self) -> tuple[int, ...]:
        """``(B, *grid)`` shape of a dense per-pixel map."""
        return (self.n_images, *self.grid)

    def validate_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Broadcast a shared/per-image mask to ``(B, *grid)`` bool."""
        if mask.dtype != torch.bool:
            raise ValueError(f"mask must be a bool tensor, got {mask.dtype}")
        if tuple(mask.shape) == self.dense_shape:
            return mask
        if tuple(mask.shape) == self.grid:
            return mask.unsqueeze(0).expand(self.dense_shape)
        if self.mode == "center" and tuple(mask.shape) == (self.n_images,):
            return mask.reshape(self.dense_shape)
        raise ValueError(
            f"mask shape {tuple(mask.shape)} is neither the shared grid "
            f"{self.grid} nor "
            f"the per-image {self.dense_shape}"
        )

    def flat_index(self, mask: torch.Tensor | None) -> torch.Tensor:
        """Sorted flat int64 indices of the selected pixels (all if no ``mask``)."""
        if mask is None:
            return torch.arange(self.n_pixels, dtype=torch.long)
        dense = self.validate_mask(mask).reshape(-1)
        return torch.nonzero(dense, as_tuple=False).squeeze(-1).to(torch.long)

    def positions(
        self, flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(image, row, col)`` top-left box origins of flat pixel indices.

        Center mode: pixel ``b`` is the box centred on image ``b``.
        """
        flat = flat.to(torch.long)
        if self.mode == "center":
            (h, w), (k_h, k_w) = self.image_shape, self.template_shape
            rows = torch.full_like(flat, (h - k_h) // 2)
            cols = torch.full_like(flat, (w - k_w) // 2)
            return flat, rows, cols
        out_h, out_w = self.grid
        within = flat % (out_h * out_w)
        return flat // (out_h * out_w), within // out_w, within % out_w

    def unflatten(self, flat: torch.Tensor) -> torch.Tensor:
        """Reshape a ``(n_pixels, ...)`` tensor to ``(B, *grid, ...)``."""
        if int(flat.shape[0]) != self.n_pixels:
            raise ValueError(
                f"expected leading axis of {self.n_pixels}, got {tuple(flat.shape)}"
            )
        return flat.reshape(*self.dense_shape, *flat.shape[1:])


@dataclass(frozen=True, eq=False)
class PixelSelection:
    """A subset of a geometry's pixels, as sorted flat indices.

    Attributes
    ----------
    index : torch.Tensor
        ``(P,)`` int64 flat indices into ``geometry`` (batch-major), sorted ascending.
    geometry : SearchGeometry
        The grid the indices address.
    """

    index: torch.Tensor
    geometry: SearchGeometry

    @classmethod
    def create(cls, index: torch.Tensor, geometry: SearchGeometry) -> PixelSelection:
        """Build a selection from any integer index tensor (cast to int64)."""
        return cls(torch.as_tensor(index).reshape(-1).to(torch.long), geometry)

    def __post_init__(self) -> None:
        """Validate (never coerce): a 1-D int64 index."""
        if self.index.dim() != 1 or self.index.dtype is not torch.long:
            raise ValueError(
                f"index must be a 1-D int64 tensor, got {tuple(self.index.shape)} "
                f"{self.index.dtype}; see PixelSelection.create()"
            )

    @classmethod
    def all(
        cls, geometry: SearchGeometry, device: torch.device | str = "cpu"
    ) -> PixelSelection:
        """Every pixel of the geometry."""
        return cls(
            torch.arange(geometry.n_pixels, dtype=torch.long, device=device), geometry
        )

    @classmethod
    def from_mask(
        cls, geometry: SearchGeometry, mask: torch.Tensor | None
    ) -> PixelSelection:
        """Pixels where ``mask`` is True (all pixels when ``mask`` is None)."""
        return cls(geometry.flat_index(mask), geometry)

    def __len__(self) -> int:
        """Number of selected pixels."""
        return int(self.index.numel())

    @property
    def device(self) -> torch.device:
        """Device of the index tensor."""
        return self.index.device

    @property
    def is_full(self) -> bool:
        """Whether every pixel of the geometry is selected (cheap size check)."""
        return len(self) == self.geometry.n_pixels

    def to(self, device: torch.device | str) -> PixelSelection:
        """Copy with the index on another device."""
        return PixelSelection(self.index.to(device), self.geometry)

    def subset(self, keep: torch.Tensor) -> PixelSelection:
        """Keep pixels where the ``(P,)`` bool ``keep`` is True (order preserved)."""
        if tuple(keep.shape) != (len(self),) or keep.dtype != torch.bool:
            raise ValueError(f"keep must be a ({len(self)},) bool tensor")
        return PixelSelection(self.index[keep.to(self.index.device)], self.geometry)

    def scatter(self, values: torch.Tensor, fill: float | int) -> torch.Tensor:
        """Scatter ``(P, ...)`` values into a dense ``(B, *grid, ...)`` map."""
        if int(values.shape[0]) != len(self):
            raise ValueError(
                f"values must have leading axis {len(self)}, got {tuple(values.shape)}"
            )
        dense = torch.full(
            (self.geometry.n_pixels, *values.shape[1:]),
            fill,
            dtype=values.dtype,
            device=values.device,
        )
        dense[self.index.to(values.device)] = values
        return self.geometry.unflatten(dense)
