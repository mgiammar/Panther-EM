"""Stage construction/validation and the JSON form of a stage and its decision."""

from __future__ import annotations

import pytest
import torch
from noise_case import noise_case

from panther_em.inference.search.decision import Decision
from panther_em.inference.search.noise_model import Calibration, ThresholdTest
from panther_em.inference.search.stage import Stage, stage_from_json, stage_to_json
from panther_em.inference.search.types import Precision

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*__array_wrap__ must accept.*:DeprecationWarning",
    "ignore:.*__array__ implementation doesn't accept.*:DeprecationWarning",
)


def test_stage_create_and_validation():
    stage = Stage.create([(0, 0, 8, 4), (8, 0, 4, 4)], precision="fp16", n_psi=32)
    assert stage.precision is Precision.FP16 and stage.k_stop == 12
    assert stage.num_features == 48 and stage.layout().num_features == 48
    single = Stage.create((0, 0, 4, 4), planes=[2, 0, 2], orientations=[5, 1])
    assert single.rectangles == ((0, 0, 4, 4),) and single.planes == (0, 2)
    assert single.orientations.tolist() == [5, 1] and single.orientations.dtype is torch.long
    with pytest.raises(ValueError):
        Stage.create([])
    with pytest.raises(ValueError):
        Stage.create([(0, 0, 8, 4), (4, 2, 8, 4)])  # overlap
    with pytest.raises(ValueError):
        Stage.create([(0, 0, 20, 4)], n_psi=32)  # k_stop 20 > 17
    with pytest.raises(ValueError):
        Stage.create([(0, 0, 0, 4)])
    with pytest.raises(ValueError):
        Stage.create([(0, 0, 4, 4)], orientations=-1)
    with pytest.raises(ValueError):
        Stage.create([(0, 0, 4, 4)], planes=[])
    # the plain constructor validates and points at create()
    with pytest.raises(ValueError, match="create"):
        Stage([(0, 0, 4, 4)])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="create"):
        Stage(((0, 0, 4, 4),), precision="fp16")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Stage(((0, 0, 4, 4),), planes=(2, 0))


def test_stage_json_round_trip():
    table = noise_case().table([(0, 0, 2, 3)])
    test = ThresholdTest(table, Calibration(alpha=1e-3, zeta=0.9, b_factor=10.0))
    stage = Stage.create(
        [(0, 0, 4, 4), (4, 0, 2, 2)],
        precision="tf32",
        n_psi=64,
        orientations=1,
        planes=(1, 3),
        decision=Decision(7.8, test, accept=False, absorb_radius=2),
        name="coarse",
    )
    data = stage_to_json(stage)
    assert data["orientations"] == 1 and data["planes"] == [1, 3]
    assert data["decision"] == {
        "threshold": 7.8,
        "accept": False,
        "absorb_radius": 2,
        "test": {
            "alpha": 1e-3,
            "zeta": 0.9,
            "eps_rho": 0.1,
            "b_factor": 10.0,
            "table": table.summary(),
        },
    }
    assert data["decision"]["test"]["table"]["rectangles"] == [[0, 0, 2, 3]]
    back = stage_from_json(data)
    assert back.rectangles == stage.rectangles and back.precision is Precision.TF32
    assert back.orientations == 1 and back.planes == (1, 3) and back.name == "coarse"
    assert back.decision is not None and back.decision.threshold == 7.8
    # the test itself is not restored (its table lives in its own file)
    assert back.decision.test is None and not back.decision.accept
    assert back.decision.absorb_radius == 2
    plain = Stage.create([(0, 0, 2, 2)], decision=Decision(3.0))
    assert stage_from_json(stage_to_json(plain)).decision == Decision(3.0)
    explicit = Stage.create([(0, 0, 2, 2)], orientations=[3, 4])
    assert stage_from_json(stage_to_json(explicit)).orientations.tolist() == [3, 4]
    assert stage_from_json(stage_to_json(Stage.create([(0, 0, 2, 2)]))).decision is None

