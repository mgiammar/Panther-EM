"""Staged, pinned, double-buffered host-to-device transfer of featurized pixels."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator
    from concurrent.futures import Future

    from panther_em.inference.search.tiling import FeaturizedImageStore

DEFAULT_STAGE_BYTES = 2 * 1024**3  # 2 GiB (x2 since double-buffering)


class PixelStager:
    """Stream ``store``'s selected pixel rows to ``compute_device`` in large chunks.

    Parameters
    ----------
    store : FeaturizedImageStore
        Persistent feature store (``relayout`` already called at least once).
    pixel_index : torch.Tensor
        Long indices into ``store``'s pixel axis, in the exact order the caller wants
        them delivered. Need not be sorted or contiguous.
    compute_device : torch.device
        Device the yielded chunks should live on.
    stage_bytes : int, optional
        Byte budget per staging buffer (there are two, for double buffering). Defaults
        to :data:`DEFAULT_STAGE_BYTES` (2 GiB, i.e. ~4 GiB total pinned + ~4 GiB total
        device memory across both buffers).
    """

    def __init__(
        self,
        store: FeaturizedImageStore,
        pixel_index: torch.Tensor,
        compute_device: torch.device,
        stage_bytes: int = DEFAULT_STAGE_BYTES,
    ) -> None:
        if store.Y is None:
            raise ValueError("store is empty; call relayout() first.")

        self.store = store
        self.pixel_index = pixel_index
        self.compute_device = compute_device
        self.n_pixels = int(pixel_index.numel())
        self.num_features = int(store.Y.shape[1])

        self._needs_staging = (
            compute_device.type == "cuda" and store.device != compute_device
        )
        self._pinned = self._needs_staging and store.device.type == "cpu"

        row_bytes = max(1, self.num_features * store.Y.element_size())
        self.stage_pixels = max(1, min(self.n_pixels, stage_bytes // row_bytes))

        self._bounds = (
            [
                (s, min(s + self.stage_pixels, self.n_pixels))
                for s in range(0, max(self.n_pixels, 1), self.stage_pixels)
            ]
            if self.n_pixels > 0
            else []
        )

        if not self._needs_staging:
            return

        self._n_slots = min(2, len(self._bounds)) or 1
        self._host_buffers: list[torch.Tensor] | None = (
            [
                torch.empty(
                    (self.stage_pixels, self.num_features),
                    dtype=store.Y.dtype,
                    pin_memory=True,
                )
                for _ in range(self._n_slots)
            ]
            if self._pinned
            else None
        )
        self._device_buffers = [
            torch.empty(
                (self.stage_pixels, self.num_features),
                dtype=store.Y.dtype,
                device=compute_device,
            )
            for _ in range(self._n_slots)
        ]
        self._copy_stream = torch.cuda.Stream(device=compute_device)

        # Fires once a slot's H2D copy has landed (device buffer safe to read).
        self._h2d_done = [torch.cuda.Event() for _ in range(self._n_slots)]

        # Fires once compute has finished reading a slot (buffers safe to overwrite).
        self._compute_done = [torch.cuda.Event() for _ in range(self._n_slots)]
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pixel-stager-prefetch"
        )
        self._futures: list[Future] = [
            self._executor.submit(self._prefetch, stage_idx)
            for stage_idx in range(self._n_slots)
        ]

    def _prefetch(self, stage_idx: int) -> None:
        """Gather + issue the async copy for ``stage_idx``, if it exists."""
        assert self.store.Y is not None  # For mypy, should never be None when called

        if stage_idx >= len(self._bounds):
            return

        slot = stage_idx % self._n_slots
        start, end = self._bounds[stage_idx]
        n = end - start
        idx = self.pixel_index[start:end].to(self.store.Y.device)

        # CUDA's "current device" is thread-local, so expressly set it to compute device
        with torch.cuda.device(self.compute_device):
            # Block until the *previous* occupant has actually finished being read.
            if stage_idx >= self._n_slots:
                self._h2d_done[slot].synchronize()

            if self._host_buffers is not None:
                host_buf = self._host_buffers[slot][:n]
                torch.index_select(self.store.Y, 0, idx, out=host_buf)
                src = host_buf
            else:
                src = self.store.Y.index_select(0, idx)

            with torch.cuda.stream(self._copy_stream):
                if stage_idx >= self._n_slots:
                    self._copy_stream.wait_event(self._compute_done[slot])
                self._device_buffers[slot][:n].copy_(src, non_blocking=True)
                self._h2d_done[slot].record(self._copy_stream)

    def stages(self) -> Iterator[tuple[int, int, torch.Tensor]]:
        """Yield ``(start, end, Y_stage)`` chunks covering ``pixel_index`` in order."""
        if not self._needs_staging:
            for start, end in self._bounds:
                idx = self.pixel_index[start:end]
                yield start, end, self.store.image_view(idx)
            return

        compute_stream = torch.cuda.current_stream(self.compute_device)
        try:
            for stage_idx, (start, end) in enumerate(self._bounds):
                slot = stage_idx % self._n_slots
                n = end - start

                self._futures[slot].result()
                compute_stream.wait_event(self._h2d_done[slot])
                yield start, end, self._device_buffers[slot][:n]

                self._compute_done[slot].record(compute_stream)
                self._futures[slot] = self._executor.submit(
                    self._prefetch, stage_idx + self._n_slots
                )
        finally:
            self._executor.shutdown(wait=True)
