"""Stage planning: bands, plane passes and the engine, without running anything."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from panther_em.inference.search.config import ExecutionConfig, Tuning
from panther_em.inference.search.hypotheses import HypothesisSpace
from panther_em.inference.search.planning import (
    DeviceInfo,
    feature_bytes_per_pixel,
    plan_stage,
)
from panther_em.inference.search.stage import Stage
from panther_em.inference.search.types import (
    BatchShape,
    PixelSelection,
    SearchGeometry,
    SearchImages,
)

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


def test_bands_cover_the_pixels_and_fit(setup):
    stage = Stage.create([RECT], n_psi=16, planes=(1,))
    per_pixel = feature_bytes_per_pixel([(RECT, stage.precision, 1)])
    assert per_pixel == 4 * 5 * 8  # complex64 features of one plane
    budget = 300 * per_pixel  # a band holds 300 pixels -> 296 (whole GEMM batches)
    p = plan(setup, stage, free_bytes=int(budget / EXEC.device_store_fraction))
    sizes = [len(b.pixels) for b in p.bands]
    # 7 bands of equal size, rounded up to whole GEMM batches; the last takes the rest
    assert sizes == [296] * 6 + [45 * 45 - 6 * 296]
    assert all(s * per_pixel <= budget for s in sizes)
    assert torch.equal(p.pixel_index, torch.arange(45 * 45))  # bands in row order
    for band in p.bands:
        (step,) = band.passes
        assert step.planes == (1,)
        assert step.memory.block_pixels >= min(len(band.pixels), 8)
    # a budget the stage fits in keeps one band
    assert len(plan(setup, stage, free_bytes=10**9).bands) == 1


def test_execution_config_validation():
    with pytest.raises(ValueError):
        ExecutionConfig(featurize_mode="tile")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ExecutionConfig(device_store_fraction=0.0)
    with pytest.raises(ValueError):
        ExecutionConfig(tuning=None)  # type: ignore[arg-type]
