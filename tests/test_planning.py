"""Stage planning: bands, plane passes and the engine, without running anything."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from panther_em.search.config import ExecutionConfig, Tuning
from panther_em.search.correlation.contraction import BatchShape
from panther_em.search.featurization import StripSpec
from panther_em.search.hypotheses import HypothesisSpace
from panther_em.search.images import (
    PixelSelection,
    SearchGeometry,
    SearchImages,
)
from panther_em.search.planning import (
    DeviceInfo,
    feature_bytes_per_pixel,
    plan_stage,
)
from panther_em.search.stage import Stage

CPU = DeviceInfo(torch.device("cpu"))
RECT = (0, 0, 4, 5)
EXEC = ExecutionConfig(tuning=Tuning(batch=BatchShape(8, 5)))


@pytest.fixture
def setup(synthetic_result):
    space = HypothesisSpace.from_result(synthetic_result(), n_defocus=3)
    images = SearchImages.single(torch.zeros(64, 64), filters=torch.ones(3, 64, 33))
    geometry = SearchGeometry.from_images(images, (20, 20))  # 45 x 45 pixels
    return space, geometry, PixelSelection.all(geometry)


def plan(setup, stage, execution=EXEC, pixels=None, free_bytes=None):
    space, geometry, every = setup
    return plan_stage(
        0,
        stage,
        every if pixels is None else pixels,
        geometry,
        space=space,
        device=CPU,
        execution=execution,
        free_bytes=free_bytes,
    )


def test_device_info_on_cpu():
    info = DeviceInfo.probe("cpu")
    assert not info.is_cuda and not info.fused_kernel and info.free_bytes() is None


def test_one_band_and_plane_passes(setup):
    stage = Stage.create([RECT], n_psi=16, planes=(0, 2))
    p = plan(setup, stage)
    assert p.mode == "fft" and len(p.bands) == 1 and p.n_pixels == 45 * 45
    assert [step.planes for step in p.bands[0].passes] == [(0,), (2,)]
    assert [step.memory.hyps.defocus_planes() for step in p.bands[0].passes] == [
        (0,),
        (2,),
    ]
    assert p.hyps.defocus_planes() == (0, 2) and p.n_passes == 2
    assert torch.equal(p.pixel_index, torch.arange(45 * 45))
    # plane-major off: one pass holds every plane
    joint = plan(
        setup, stage, replace(EXEC, tuning=replace(EXEC.tuning, plane_major=False))
    )
    assert [step.planes for step in joint.bands[0].passes] == [(0, 2)]
    assert joint.bands[0].passes[0].memory is not None


def test_engine_choice_and_override(setup):
    geometry = setup[1]
    stage = Stage.create([RECT], n_psi=16)
    sparse = PixelSelection.create(torch.tensor([3, 50, 99]), geometry)
    assert plan(setup, stage, pixels=sparse).mode == "direct"
    assert plan(setup, stage).mode == "fft"
    forced = replace(EXEC, featurize_mode="direct")
    assert plan(setup, stage, forced).mode == "direct"


def test_fft_bands_are_whole_rows_that_fit(setup):
    stage = Stage.create([RECT], n_psi=16, planes=(1,))
    per_pixel = feature_bytes_per_pixel([(RECT, stage.precision, 1)])
    assert per_pixel == 4 * 5 * 8  # complex64 features of one plane
    budget = 300 * per_pixel  # a band holds 300 pixels -> 6 whole rows of 45
    p = plan(setup, stage, free_bytes=int(budget / EXEC.device_store_fraction))
    assert p.mode == "fft"
    sizes = [len(b.pixels) for b in p.bands]
    # 45 rows in the fewest equal bands of at most 6 rows: 8 bands of 6, the rest 3
    assert sizes == [6 * 45] * 7 + [3 * 45]
    assert all(s * per_pixel <= budget for s in sizes)
    assert torch.equal(p.pixel_index, torch.arange(45 * 45))  # bands in row order
    geometry = setup[1]
    seen: set[int] = set()
    for band in p.bands:
        rows = set(geometry.positions(band.pixels.index)[1].tolist())
        assert not rows & seen  # no row is shared between bands
        seen |= rows
        (step,) = band.passes
        assert step.planes == (1,)
        assert step.memory.block_pixels >= min(len(band.pixels), 8)
    # a budget the stage fits in keeps one band
    assert len(plan(setup, stage, free_bytes=10**9).bands) == 1


def test_fft_bands_hold_whole_strips(setup):
    """With fixed FFT heights a band holds the rows of whole strips."""
    space, geometry, every = setup
    stage = Stage.create([RECT], n_psi=16, planes=(1,))
    per_pixel = feature_bytes_per_pixel([(RECT, stage.precision, 1)])
    strips = StripSpec((20, 20), heights=(24, 32))  # 5 and 13 rows per strip
    p = plan_stage(
        0,
        stage,
        every,
        geometry,
        space=space,
        device=CPU,
        execution=EXEC,
        free_bytes=int(15 * 45 * per_pixel / EXEC.device_store_fraction),
        strips=strips,
    )
    # up to 15 rows fit: bands of one 32-high strip (13 rows), the 6 left in a
    # 24-high one, 3 x 32 + 24 = 120 rows of FFT, less than 3 equal bands of 15
    # rows (each 32 + 24)
    assert [len(b.pixels) // 45 for b in p.bands] == [13, 13, 13, 6]
    # a masked selection keeps the bands on the same rows
    keep = geometry.positions(every.index)[1] % 4 != 1
    masked = PixelSelection.create(every.index[keep], geometry)
    q = plan_stage(
        0,
        stage,
        masked,
        geometry,
        space=space,
        device=CPU,
        execution=EXEC,
        free_bytes=int(15 * 45 * per_pixel / EXEC.device_store_fraction),
        strips=strips,
    )
    for band, full in zip(q.bands, p.bands, strict=True):
        rows = geometry.positions(band.pixels.index)[1]
        full_rows = geometry.positions(full.pixels.index)[1]
        assert int(rows.min()) >= int(full_rows.min())
        assert int(rows.max()) <= int(full_rows.max())


def test_direct_bands_are_whole_gemm_batches(setup):
    stage = Stage.create([RECT], n_psi=16, planes=(1,))
    per_pixel = feature_bytes_per_pixel([(RECT, stage.precision, 1)])
    budget = 300 * per_pixel  # a band holds 300 pixels -> 296 (whole GEMM batches)
    direct = replace(EXEC, featurize_mode="direct")
    p = plan(setup, stage, direct, free_bytes=int(budget / EXEC.device_store_fraction))
    sizes = [len(b.pixels) for b in p.bands]
    # 7 bands of equal size, rounded up to whole GEMM batches; the last takes the rest
    assert sizes == [296] * 6 + [45 * 45 - 6 * 296]
    assert torch.equal(p.pixel_index, torch.arange(45 * 45))


def test_execution_config_validation():
    with pytest.raises(ValueError):
        ExecutionConfig(featurize_mode="tile")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ExecutionConfig(device_store_fraction=0.0)
    with pytest.raises(ValueError):
        ExecutionConfig(tuning=None)  # type: ignore[arg-type]
