"""StageSpec / SearchSchedule / ErrorModel / nested_schedule."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from panther_em.decomposition.result import DecompositionResult
from panther_em.inference.search.decisions import NoDecision
from panther_em.inference.search.schedule import (
    ErrorModel,
    SearchSchedule,
    StageSpec,
    nested_schedule,
)
from panther_em.inference.search.types import Precision


def test_stage_spec_validation():
    spec = StageSpec.create([(0, 0, 8, 4), (8, 0, 4, 4)], precision="fp16", n_psi=32)
    assert (
        spec.precision is Precision.FP16
        and spec.k_stop == 12
        and spec.num_features == 48
    )
    assert spec.layout().num_features == 48
    with pytest.raises(ValueError):
        StageSpec.create([])
    with pytest.raises(ValueError):
        StageSpec.create([(0, 0, 8, 4), (4, 2, 8, 4)])  # overlap
    with pytest.raises(ValueError):
        StageSpec.create([(0, 0, 20, 4)], n_psi=32)  # k_stop 20 > 17
    with pytest.raises(ValueError):
        StageSpec.create([(0, 0, 4, 4)], error=1.5)
    with pytest.raises(ValueError):
        StageSpec.create([(0, 0, 0, 4)])


def test_schedule_defaults_and_length_check():
    s = SearchSchedule((StageSpec.create([(0, 0, 4, 4)]), StageSpec.create([(0, 0, 8, 8)])))
    assert len(s) == 2 and all(isinstance(d, NoDecision) for d in s.decisions)
    assert not s.requires_errors()
    with pytest.raises(ValueError):
        SearchSchedule((StageSpec.create([(0, 0, 4, 4)]),), (NoDecision(), NoDecision()))
    with pytest.raises(ValueError):
        SearchSchedule(())


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


def test_error_model_spectral_matches_brute_force_and_composes_in_quadrature():
    result = _result()
    model = ErrorModel.from_result(result)
    S2 = result.S.astype(np.float64) ** 2
    w = np.full(S2.shape[0], 2.0)
    w[0] = 1.0  # DC; N_ang = 24 -> Nyquist k = 12 not stored
    e_all = float((S2 * w[:, None]).sum())
    for k_ext, m_ext in [(1, 1), (3, 2), (6, 5), (4, 5)]:
        e_sel = float((S2[:k_ext, :m_ext] * w[:k_ext, None]).sum())
        expected = math.sqrt(1.0 - e_sel / e_all)
        assert model.rho((0, 0, k_ext, m_ext)) == pytest.approx(expected, abs=1e-12)
        rho, rho_k, rho_m = model.decompose(k_ext, m_ext)
        assert rho == pytest.approx(expected, abs=1e-12)
        assert rho**2 == pytest.approx(rho_k**2 + rho_m**2, abs=1e-12)
    assert model.rho((0, 0, 6, 5)) == pytest.approx(0.0, abs=1e-12)
    # multi-rectangle stage and StageSpec input agree
    spec = StageSpec.create([(0, 0, 3, 5), (3, 0, 3, 2)])
    e_sel = float((S2[:3, :] * w[:3, None]).sum() + (S2[3:6, :2] * w[3:6, None]).sum())
    assert model.rho(spec) == pytest.approx(math.sqrt(1 - e_sel / e_all), abs=1e-12)


def test_error_model_tail_is_monotone_and_nyquist_weight():
    result = _result(k=13, n_ang=24)  # stores k = 12 = Nyquist
    m0 = ErrorModel.from_result(result)
    m1 = ErrorModel.from_result(result, tail=0.1)
    assert m1.parseval[12] == 1.0 and m1.parseval[1] == 2.0 and m1.parseval[0] == 1.0
    assert m1.rho((0, 0, 4, 3)) > m0.rho((0, 0, 4, 3))
    assert m1.rho((0, 0, 13, 5)) == pytest.approx(math.sqrt(0.1), abs=1e-12)
    cplx = ErrorModel.from_result(_result(complex_proj=True))
    assert torch.all(cplx.parseval == 1.0)


def test_error_model_table_lookup(tmp_path):
    csv_path = tmp_path / "errors.csv"
    csv_path.write_text(
        "k_extent,m_extent,recon_error_used,recon_error_mean\n8,8,0.70,0.71\n16,16,0.52,0.53\n"
    )
    model = ErrorModel.from_table(csv_path)
    assert model.rho((0, 0, 16, 16)) == pytest.approx(0.52)
    assert ErrorModel.from_table(csv_path, column="recon_error_mean").rho(
        StageSpec.create([(0, 0, 8, 8)])
    ) == pytest.approx(0.71)
    with pytest.raises(KeyError):
        model.rho((0, 0, 4, 4))
    with pytest.raises(ValueError):
        model.rho((2, 0, 8, 8))


def test_nested_schedule_broadcasts_and_fills_errors():
    result = _result()
    model = ErrorModel.from_result(result)
    sched = nested_schedule(
        [(2, 2), (4, 3), (6, 5)],
        precision=["fp16", "fp16", "fp32"],
        n_psi=16,
        errors=[0.3, None, None],
        error_model=model,
    )
    assert [s.precision for s in sched.stages] == [
        Precision.FP16,
        Precision.FP16,
        Precision.FP32,
    ]
    assert sched.stages[0].error == 0.3  # explicit error kept
    assert sched.stages[1].error == pytest.approx(model.rho((0, 0, 4, 3)))
    assert sched.stages[2].error == pytest.approx(0.0, abs=1e-12)
    assert sched.stages[0].orientations is None and sched.stages[0].planes is None
    with pytest.raises(ValueError):
        nested_schedule([(2, 2), (4, 4)], precision=["fp16"])
    with pytest.raises(ValueError):
        nested_schedule([])
