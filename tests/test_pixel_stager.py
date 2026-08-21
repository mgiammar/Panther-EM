"""Unit tests for PixelStager (staged, pinned, double-buffered H2D transfer).

Covers
------
- The no-staging fallback (store already on compute_device): every machine, no GPU
  required.
- The staged CUDA path: pinned double-buffered transfer from a CPU-resident store,
  forced through several stages (small stage_bytes) and buffer-slot reuse, checked
  bit-exact against a plain `index_select` reference -- including a non-contiguous,
  shuffled pixel_index to make sure ordering survives staging.
"""

from __future__ import annotations

import pytest
import torch

from panther_em.inference.search.staging import PixelStager
from panther_em.inference.search.tiling import FeaturizedImageStore


def _make_store(
    num_pixels: int, num_features: int, device: str, seed: int = 0
) -> FeaturizedImageStore:
    store = FeaturizedImageStore(num_pixels, device=torch.device(device))
    gen = torch.Generator().manual_seed(seed)
    real = torch.randn(num_pixels, num_features, generator=gen)
    imag = torch.randn(num_pixels, num_features, generator=gen)
    store.Y = torch.complex(real, imag).to(torch.complex64).to(device)
    return store


def _collect(stager: PixelStager) -> torch.Tensor:
    n = stager.n_pixels
    out = torch.empty(
        (n, stager.num_features), dtype=torch.complex64, device=stager.compute_device
    )
    for start, end, chunk in stager.stages():
        out[start:end] = chunk
    return out


def test_no_staging_fallback_matches_direct_index_select():
    store = _make_store(37, 5, device="cpu")
    pixel_index = torch.arange(37)
    stager = PixelStager(store, pixel_index, torch.device("cpu"), stage_bytes=1024)

    result = _collect(stager)
    expected = store.Y.index_select(0, pixel_index)
    assert torch.equal(result, expected)


def test_no_staging_fallback_handles_shuffled_subset():
    store = _make_store(50, 4, device="cpu")
    pixel_index = torch.randperm(50)[:23]
    stager = PixelStager(store, pixel_index, torch.device("cpu"), stage_bytes=64)

    result = _collect(stager)
    expected = store.Y.index_select(0, pixel_index)
    assert torch.equal(result, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
class TestStagedCudaTransfer:
    def test_matches_reference_across_many_stages(self):
        num_pixels, num_features = 401, 6
        store = _make_store(num_pixels, num_features, device="cpu")
        pixel_index = torch.arange(num_pixels)

        # row_bytes = num_features * 8 (complex64); force ~ (401 / 8) => many
        # small stages so buffer slots (only 2) get reused several times over.
        row_bytes = num_features * 8
        stager = PixelStager(
            store,
            pixel_index,
            torch.device("cuda"),
            stage_bytes=row_bytes * 8,
        )
        assert len(stager._bounds) > 4, "test should exercise multiple stage reuses"

        result = _collect(stager)
        expected = store.Y.to("cuda").index_select(0, pixel_index.to("cuda"))
        assert torch.equal(result, expected)

    def test_matches_reference_with_shuffled_pixel_index(self):
        num_pixels, num_features = 200, 3
        store = _make_store(num_pixels, num_features, device="cpu", seed=1)
        pixel_index = torch.randperm(num_pixels)[:150]

        row_bytes = num_features * 8
        stager = PixelStager(
            store,
            pixel_index,
            torch.device("cuda"),
            stage_bytes=row_bytes * 10,
        )

        result = _collect(stager)
        expected = store.Y.to("cuda").index_select(0, pixel_index.to("cuda"))
        assert torch.equal(result, expected)

    def test_matches_reference_single_stage(self):
        """stage_bytes large enough that everything fits in one stage/slot."""
        num_pixels, num_features = 64, 4
        store = _make_store(num_pixels, num_features, device="cpu", seed=2)
        pixel_index = torch.arange(num_pixels)

        stager = PixelStager(
            store, pixel_index, torch.device("cuda"), stage_bytes=1024**3
        )
        assert len(stager._bounds) == 1

        result = _collect(stager)
        expected = store.Y.to("cuda").index_select(0, pixel_index.to("cuda"))
        assert torch.equal(result, expected)

    def test_store_already_on_compute_device_is_noop_passthrough(self):
        num_pixels, num_features = 30, 3
        store = _make_store(num_pixels, num_features, device="cuda", seed=3)
        pixel_index = torch.arange(num_pixels)

        stager = PixelStager(store, pixel_index, torch.device("cuda"), stage_bytes=1024)
        assert not stager._needs_staging

        result = _collect(stager)
        expected = store.Y.index_select(0, pixel_index.to("cuda"))
        assert torch.equal(result, expected)
