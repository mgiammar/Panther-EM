"""Stage construction/validation and the spectral error model."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from panther_em.decomposition.result import DecompositionResult
from panther_em.inference.search.decision import Decision, ErrorModel
from panther_em.inference.search.stage import (
    FixedError,
    Stage,
    stage_from_json,
    stage_to_json,
)
from panther_em.inference.search.types import Precision


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
    bound = FixedError(0.05)
    stage = Stage.create(
        [(0, 0, 4, 4), (4, 0, 2, 2)],
        precision="tf32",
        n_psi=64,
        orientations=1,
        planes=(1, 3),
        decision=Decision(7.8, bound, absorb_radius=2),
        name="coarse",
    )
    data = stage_to_json(stage)
    assert data["orientations"] == 1 and data["planes"] == [1, 3]
    assert data["decision"] == {"threshold": 7.8, "absorb_radius": 2, "rho": 0.05}
    back = stage_from_json(data)
    assert back.rectangles == stage.rectangles and back.precision is Precision.TF32
    assert back.orientations == 1 and back.planes == (1, 3) and back.name == "coarse"
    assert back.decision is not None and back.decision.threshold == 7.8
    assert back.decision.bound.rho(stage.rectangles) == 0.05
    explicit = Stage.create([(0, 0, 2, 2)], orientations=[3, 4])
    assert stage_from_json(stage_to_json(explicit)).orientations.tolist() == [3, 4]
    assert stage_from_json(stage_to_json(Stage.create([(0, 0, 2, 2)]))).decision is None


def _result(k=6, e=5, n_ang=24, complex_proj=False):
    rng = np.random.default_rng(0)
    blocks = 2 * k if complex_proj else k
    S = rng.random((blocks, e)).astype(np.float32)
    U = rng.standard_normal((1, 3, blocks, e)).astype(np.complex64)
    Vh = rng.standard_normal((blocks, e, 8)).astype(np.complex64)
    return DecompositionResult(
        U=U,
        S=S,
        Vh=Vh,
        k_max=k,
        eig_max=e,
        is_complex_projection=complex_proj,
        num_fourier_filters=1,
        num_orientations=3,
        num_angular_components=n_ang,
        num_radial_components=8,
        coordinate_transform=None,
    )


def test_error_model_spectral_matches_brute_force():
    result = _result()
    model = ErrorModel.from_result(result)
    S2 = result.S.astype(np.float64) ** 2
    w = np.full(S2.shape[0], 2.0)
    w[0] = 1.0  # DC; N_ang = 24 -> Nyquist k = 12 not stored
    e_all = float((S2 * w[:, None]).sum())
    for k_ext, m_ext in [(1, 1), (3, 2), (6, 5), (4, 5)]:
        e_sel = float((S2[:k_ext, :m_ext] * w[:k_ext, None]).sum())
        expected = math.sqrt(1.0 - e_sel / e_all)
        assert model.rho([(0, 0, k_ext, m_ext)]) == pytest.approx(expected, abs=1e-12)
    assert model.rho([(0, 0, 6, 5)]) == pytest.approx(0.0, abs=1e-12)
    # a multi-rectangle stage sums the rectangles' energies
    stage = Stage.create([(0, 0, 3, 5), (3, 0, 3, 2)])
    e_sel = float((S2[:3, :] * w[:3, None]).sum() + (S2[3:6, :2] * w[3:6, None]).sum())
    assert model.rho(stage.rectangles) == pytest.approx(math.sqrt(1 - e_sel / e_all), abs=1e-12)


def test_error_model_tail_is_monotone_and_nyquist_weight():
    result = _result(k=13, n_ang=24)  # stores k = 12 = Nyquist
    m0 = ErrorModel.from_result(result)
    m1 = ErrorModel.from_result(result, tail=0.1)
    assert m1.parseval[12] == 1.0 and m1.parseval[1] == 2.0 and m1.parseval[0] == 1.0
    assert m1.rho([(0, 0, 4, 3)]) > m0.rho([(0, 0, 4, 3)])
    assert m1.rho([(0, 0, 13, 5)]) == pytest.approx(math.sqrt(0.1), abs=1e-12)
    cplx = ErrorModel.from_result(_result(complex_proj=True))
    assert torch.all(cplx.parseval == 1.0)
    with pytest.raises(ValueError):
        ErrorModel(m0.S2, m0.parseval, tail=1.0)
    with pytest.raises(ValueError):
        ErrorModel(m0.S2, m0.parseval[:2])
