"""Tests for :mod:`panther_em.inference.search.tracking`.

Covers the pure `classify_zscore_interval` helper, `MultiPrecisionPixelTracker`'s
per-stage bookkeeping (status transitions, frozen stats, absorption via dilation,
`finalize()`'s schema) in isolation with hand-built stage maps, and one end-to-end
run wired into a real (stubbed-featurizer) `incremental_search`. Also covers the
`incremental_search` empty-pixel-set early exit this feature relies on.

Reuses the `_make_reconstructor` / `_stub_featurize_cells` pattern from
`tests/test_staged_search_integration.py` to bypass the real polar-to-Cartesian
kernel construction, which is irrelevant to what these tests target.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import panther_em.inference.search.compressed as compressed_mod
from panther_em.coordinates.offset_polar import OffsetPolarTransform
from panther_em.decomposition.result import DecompositionResult
from panther_em.inference.projection_reconstruction import ProjectionReconstructor
from panther_em.inference.search import incremental_search
from panther_em.inference.search.incremental import compute_zscore_error_intervals
from panther_em.inference.search.tracking import (
    MultiPrecisionPixelTracker,
    PixelLabel,
    classify_zscore_interval,
)

NUM_FF = 1
NUM_OR = 6
K_MAX = 4
EIG_MAX = 5
NUM_R = 8
NUM_ANG = 24
CARTESIAN_SIZE = 20
IMAGE_SIZE = 30  # gives a (11, 11) = 121-pixel valid-correlation grid


def _make_reconstructor(device: torch.device) -> ProjectionReconstructor:
    rng = np.random.default_rng(0)
    num_freq_blocks = K_MAX

    U = (
        rng.standard_normal((NUM_FF, NUM_OR, num_freq_blocks, EIG_MAX)).astype(
            np.float32
        )
        + 1j
        * rng.standard_normal((NUM_FF, NUM_OR, num_freq_blocks, EIG_MAX)).astype(
            np.float32
        )
    ).astype(np.complex64)
    S = rng.random((num_freq_blocks, EIG_MAX)).astype(np.float32)
    Vh = (
        rng.standard_normal((num_freq_blocks, EIG_MAX, NUM_R)).astype(np.float32)
        + 1j * rng.standard_normal((num_freq_blocks, EIG_MAX, NUM_R)).astype(np.float32)
    ).astype(np.complex64)
    transform = OffsetPolarTransform.from_image(
        image_shape=(CARTESIAN_SIZE, CARTESIAN_SIZE),
        num_angle=NUM_ANG,
        num_radius=NUM_R,
    )
    result = DecompositionResult(
        U=U,
        S=S,
        Vh=Vh,
        k_max=K_MAX,
        eig_max=EIG_MAX,
        is_complex_projection=False,
        num_fourier_filters=NUM_FF,
        num_orientations=NUM_OR,
        num_angular_components=NUM_ANG,
        num_radial_components=NUM_R,
        coordinate_transform=transform,
    )
    return ProjectionReconstructor(result=result, device=device)


def _stub_featurize_cells(
    image, reconstructor, cells, *, feature_chunk=None, feature_store_device=None, **_
):
    """Bypass the real polar-to-cartesian kernel construction (see
    `tests/test_staged_search_integration.py` for the original rationale)."""
    image_bhw = image if image.dim() == 3 else image.unsqueeze(0)
    b, h, w = image_bhw.shape
    k_h, k_w = reconstructor.image_shape
    n_px = b * (h - k_h + 1) * (w - k_w + 1)
    gen = torch.Generator().manual_seed(1)
    real = torch.randn(int(cells.shape[0]), n_px, generator=gen)
    imag = torch.randn(int(cells.shape[0]), n_px, generator=gen)
    feats = torch.complex(real, imag).to(torch.complex64)
    return feats.to(feature_store_device or reconstructor.device)


def _stage_maps(mip, mean, variance, best_index=None, best_psi=None):
    mip = torch.as_tensor(mip, dtype=torch.float32)
    n = mip.numel()
    return {
        "mip": mip,
        "mean": torch.as_tensor(mean, dtype=torch.float32).expand(n).clone(),
        "variance": torch.as_tensor(variance, dtype=torch.float32).expand(n).clone(),
        "best_index": torch.as_tensor(
            best_index if best_index is not None else torch.arange(n), dtype=torch.int64
        ),
        "best_psi": torch.as_tensor(
            best_psi if best_psi is not None else torch.zeros(n), dtype=torch.int64
        ),
    }


# ---------------------------------------------------------------------------
# classify_zscore_interval
# ---------------------------------------------------------------------------


def test_classify_zscore_interval_three_states():
    intervals = torch.tensor(
        [
            [1.0, 2.0],  # entirely below threshold -> REJECTED
            [3.0, 7.0],  # straddles threshold -> UNCERTAIN
            [8.0, 9.0],  # entirely above threshold -> ACCEPTED
            [1.0, 5.0],  # upper == threshold -> UNCERTAIN (strict inequality)
            [5.0, 9.0],  # lower == threshold -> UNCERTAIN (strict inequality)
        ]
    )
    labels = classify_zscore_interval(intervals, threshold=5.0)
    expected = torch.tensor(
        [
            PixelLabel.REJECTED,
            PixelLabel.UNCERTAIN,
            PixelLabel.ACCEPTED,
            PixelLabel.UNCERTAIN,
            PixelLabel.UNCERTAIN,
        ],
        dtype=torch.int8,
    )
    assert torch.equal(labels, expected)
    assert labels.dtype == torch.int8


def test_classify_zscore_interval_composes_with_error_intervals():
    # error = 0 collapses the interval to a point at the exact z-score.
    corr = torch.tensor([10.0, 1.0, 5.0])
    mu = torch.zeros(3)
    sigma = torch.ones(3)
    interval = compute_zscore_error_intervals(0.0, corr, mu, sigma)
    labels = classify_zscore_interval(interval, threshold=5.0)
    assert torch.equal(
        labels,
        torch.tensor(
            [PixelLabel.ACCEPTED, PixelLabel.REJECTED, PixelLabel.UNCERTAIN],
            dtype=torch.int8,
        ),
    )

    # A modest error widens each interval but doesn't flip any pixel's decision here;
    # the already-uncertain pixel (index 2) stays uncertain either way.
    wide_interval = compute_zscore_error_intervals(0.1, corr, mu, sigma)
    wide_labels = classify_zscore_interval(wide_interval, threshold=5.0)
    assert torch.equal(labels, wide_labels)
    assert (
        wide_interval[:, 1] - wide_interval[:, 0] > interval[:, 1] - interval[:, 0]
    ).all()


# ---------------------------------------------------------------------------
# MultiPrecisionPixelTracker.step
# ---------------------------------------------------------------------------


def test_tracker_step_partitions_and_narrows():
    out_shape = (1, 4, 4)  # 16 pixels
    tracker = MultiPrecisionPixelTracker(
        stage_errors=[0.0, 0.0],
        zscore_threshold=5.0,
        dilation_radius=0,
        out_shape=out_shape,
    )

    accepted_idx = torch.arange(0, 5)
    rejected_idx = torch.arange(5, 10)
    uncertain_idx = torch.arange(10, 16)
    mip = torch.empty(16)
    mip[accepted_idx] = 10.0
    mip[rejected_idx] = 1.0
    mip[uncertain_idx] = (
        5.0  # boundary -> uncertain, since error=0 means interval=[z,z]
    )

    stage1 = _stage_maps(mip=mip, mean=0.0, variance=1.0)
    next_px = tracker.step(stage1, torch.arange(16))

    assert torch.equal(
        tracker.status[accepted_idx],
        torch.full((5,), int(PixelLabel.ACCEPTED), dtype=torch.int8),
    )
    assert torch.equal(
        tracker.status[rejected_idx],
        torch.full((5,), int(PixelLabel.REJECTED), dtype=torch.int8),
    )
    assert torch.equal(
        tracker.status[uncertain_idx],
        torch.full((6,), int(PixelLabel.UNCERTAIN), dtype=torch.int8),
    )
    assert torch.equal(torch.sort(next_px).values, uncertain_idx)
    assert torch.all(tracker.last_stage[accepted_idx] == 0)

    # Stage 2 only touches the 6 still-uncertain pixels; resolve half of them.
    mip2 = torch.tensor([10.0, 10.0, 10.0, 1.0, 1.0, 5.0])
    stage2 = _stage_maps(mip=mip2, mean=0.0, variance=1.0)
    next_px2 = tracker.step(stage2, uncertain_idx)

    assert tracker.status[10] == PixelLabel.ACCEPTED
    assert tracker.status[11] == PixelLabel.ACCEPTED
    assert tracker.status[12] == PixelLabel.ACCEPTED
    assert tracker.status[13] == PixelLabel.REJECTED
    assert tracker.status[14] == PixelLabel.REJECTED
    assert tracker.status[15] == PixelLabel.UNCERTAIN  # still ambiguous at final stage
    assert torch.equal(next_px2, torch.tensor([15]))

    # Pixels decided in stage 1 are frozen: stats/last_stage untouched by stage 2.
    assert tracker.mip[0] == 10.0
    assert tracker.last_stage[0] == 0
    assert torch.all(
        tracker.status[16:] == 0
    )  # nothing beyond index 15 in a 16-pixel grid


def test_tracker_step_raises_after_stage_errors_exhausted():
    tracker = MultiPrecisionPixelTracker(
        stage_errors=[0.0, 0.0],
        zscore_threshold=5.0,
        out_shape=(1, 1, 1),
    )
    stage_maps = _stage_maps(mip=[0.0], mean=0.0, variance=1.0)
    tracker.step(stage_maps, torch.tensor([0]))
    tracker.step(stage_maps, torch.tensor([0]))
    with pytest.raises(IndexError):
        tracker.step(stage_maps, torch.tensor([0]))


# ---------------------------------------------------------------------------
# Absorption / dilation
# ---------------------------------------------------------------------------


def test_absorption_dilates_accepted_mask_by_radius():
    tracker = MultiPrecisionPixelTracker(
        stage_errors=[0.0],
        zscore_threshold=5.0,
        dilation_radius=1,
        out_shape=(1, 5, 5),
    )
    tracker.status[:] = int(PixelLabel.UNCERTAIN)
    center = 2 * 5 + 2  # (row=2, col=2)
    tracker.status[center] = int(PixelLabel.ACCEPTED)

    tracker._absorb_uncertain_near_accepted()

    grid = tracker.status.reshape(5, 5)
    four_neighbors = [(1, 2), (3, 2), (2, 1), (2, 3)]
    diagonal_neighbors = [(1, 1), (1, 3), (3, 1), (3, 3)]
    for r, c in four_neighbors:
        assert grid[r, c] == PixelLabel.ABSORBED, (r, c)
    for r, c in diagonal_neighbors:
        assert grid[r, c] == PixelLabel.UNCERTAIN, (r, c)
    assert grid[2, 2] == PixelLabel.ACCEPTED
    assert grid[0, 0] == PixelLabel.UNCERTAIN


def test_absorption_does_not_cross_batch_boundary():
    tracker = MultiPrecisionPixelTracker(
        stage_errors=[0.0],
        zscore_threshold=5.0,
        dilation_radius=3,  # large enough to cover an entire 3x3 image
        out_shape=(2, 3, 3),
    )
    tracker.status[:] = int(PixelLabel.UNCERTAIN)
    tracker.status[8] = int(PixelLabel.ACCEPTED)  # image 0, position (2, 2)
    # index 9 is image 1, position (0, 0) -- adjacent in the flat index, but a
    # different image; must never be absorbed.

    tracker._absorb_uncertain_near_accepted()

    assert tracker.status[9] == PixelLabel.UNCERTAIN
    grid1 = tracker.status.reshape(2, 3, 3)[1]
    assert torch.all(grid1 == int(PixelLabel.UNCERTAIN))


def test_absorption_disabled_when_radius_is_zero():
    tracker = MultiPrecisionPixelTracker(
        stage_errors=[0.0],
        zscore_threshold=5.0,
        dilation_radius=0,
        out_shape=(1, 3, 3),
    )
    stage_maps = _stage_maps(
        mip=[10.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], mean=0.0, variance=1.0
    )
    tracker.step(stage_maps, torch.arange(9))
    assert tracker.status[0] == PixelLabel.ACCEPTED
    assert torch.all(tracker.status[1:] == int(PixelLabel.UNCERTAIN))


# ---------------------------------------------------------------------------
# finalize()
# ---------------------------------------------------------------------------


def test_finalize_schema_and_pending_pixels():
    tracker = MultiPrecisionPixelTracker(
        stage_errors=[0.0],
        zscore_threshold=5.0,
        out_shape=(1, 2, 2),
    )
    stage_maps = _stage_maps(mip=[10.0, 1.0], mean=0.0, variance=1.0)
    tracker.step(stage_maps, torch.tensor([0, 1]))  # pixels 2, 3 stay PENDING

    result = tracker.finalize()
    expected_keys = {
        "status",
        "mip",
        "mean",
        "std",
        "zscore",
        "best_index",
        "best_psi",
        "last_stage",
    }
    assert set(result.keys()) == expected_keys
    for key, value in result.items():
        assert value.shape == (4,), key

    assert result["status"][0] == PixelLabel.ACCEPTED
    assert result["status"][1] == PixelLabel.REJECTED
    assert torch.all(result["status"][2:] == int(PixelLabel.PENDING))
    assert torch.all(result["last_stage"][2:] == -1)
    assert torch.all(result["best_index"][2:] == -1)
    assert torch.all(torch.isnan(result["mip"][2:]))

    reshaped = tracker.finalize(reshape=True)
    assert reshaped["status"].shape == (1, 2, 2)


# ---------------------------------------------------------------------------
# Full integration with a real (stubbed-featurizer) incremental_search
# ---------------------------------------------------------------------------


def test_tracker_wired_into_incremental_search(monkeypatch):
    monkeypatch.setattr(compressed_mod, "featurize_cells", _stub_featurize_cells)

    device = torch.device("cpu")
    reconstructor = _make_reconstructor(device)
    gen = torch.Generator().manual_seed(0)
    image = torch.randn(IMAGE_SIZE, IMAGE_SIZE, generator=gen)
    stage_rectangles = [[(0, 0, 2, 3)], [(0, 0, K_MAX, EIG_MAX)]]

    tracker = MultiPrecisionPixelTracker(
        stage_errors=[0.2, 0.2],
        zscore_threshold=1.0,
        dilation_radius=0,
        reconstructor=reconstructor,
        image=image,
        device=device,
    )

    stage_sizes = []
    for stage_maps in incremental_search(
        image,
        reconstructor,
        stage_rectangles,
        pixel_batch=32,
        hyp_batch=5,
        n_psi=16,
        feature_chunk=8,
        compute_device=device,
        feature_store_device=device,
        use_fused_kernel=False,
        follow_up_fn=tracker.step,
    ):
        stage_sizes.append(stage_maps["mip"].numel())

    assert stage_sizes[0] == tracker.n_px_total
    assert all(b <= a for a, b in zip(stage_sizes, stage_sizes[1:], strict=False))
    assert len(stage_sizes) < 2 or stage_sizes[1] < stage_sizes[0]

    result = tracker.finalize()
    assert not torch.any(result["status"] == int(PixelLabel.PENDING))


def test_incremental_search_stops_once_pixel_set_is_empty(monkeypatch):
    monkeypatch.setattr(compressed_mod, "featurize_cells", _stub_featurize_cells)

    device = torch.device("cpu")
    reconstructor = _make_reconstructor(device)
    gen = torch.Generator().manual_seed(0)
    image = torch.randn(IMAGE_SIZE, IMAGE_SIZE, generator=gen)
    stage_rectangles = [
        [(0, 0, 2, 3)],
        [(0, 0, K_MAX, EIG_MAX)],
        [(0, 0, K_MAX, EIG_MAX)],
    ]

    def empty_after_first_stage(stage_maps, pixel_index):
        return torch.empty(0, dtype=torch.long)

    stages = list(
        incremental_search(
            image,
            reconstructor,
            stage_rectangles,
            pixel_batch=32,
            hyp_batch=5,
            n_psi=16,
            feature_chunk=8,
            compute_device=device,
            feature_store_device=device,
            use_fused_kernel=False,
            follow_up_fn=empty_after_first_stage,
        )
    )
    assert len(stages) == 1
