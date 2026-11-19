"""The FDR detection rule (``SearchConfig.fdr``, default 0.05)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from leopard_em.analysis.fdr import bh_cutoff, nominal_null_rate
from scipy.special import ndtri

from panther_em.managers import SearchConfig, search
from panther_em.managers.fdr import (
    bh_threshold,
    calibrated_null_rate,
    expected_false_peaks,
    null_cut_1fp,
)
from panther_em.search import PixelLabel, SearchResult

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "threshold_test_pipeline.py"


def test_expected_false_peaks_is_the_formula():
    z = torch.tensor([7.0, 8.0])
    lam = expected_false_peaks(z, 1000, 10**9)
    t1 = float(-ndtri(1.0 / (1000 * 10**9)))  # the 1-FP threshold
    assert float(expected_false_peaks(torch.tensor([t1]), 1000, 10**9)[0]) == (
        pytest.approx(1.0, rel=1e-6)
    )
    assert float(lam[0]) > float(lam[1])


def test_bh_threshold_step_up():
    n_pos, n_hyp = 1000, 10**9
    t1 = float(-ndtri(1.0 / (n_pos * n_hyp)))
    # 40 strong peaks well above the 1-FP cut, then noise-like peaks below it
    strong = torch.full((40,), t1 + 1.0)
    weak = torch.linspace(t1 - 0.05, t1 - 2.0, 200)
    t = bh_threshold(torch.cat([strong, weak]), n_pos, n_hyp, 0.05)
    assert t is not None
    k = int((torch.cat([strong, weak]) >= t).sum())
    lam = float(expected_false_peaks(torch.tensor([t]), n_pos, n_hyp)[0])
    assert lam <= 0.05 * k + 1e-9  # the rule holds at the threshold
    # step-up: no lower peak also satisfies it
    lower = torch.cat([strong, weak])
    lower = lower[lower < t]
    for z in lower[:20]:
        kz = int((torch.cat([strong, weak]) >= z).sum())
        assert float(expected_false_peaks(z[None], n_pos, n_hyp)[0]) > 0.05 * kz
    assert t < t1  # many true peaks: the FDR cut sits below the 1-FP cut


def test_bh_threshold_no_picks_and_validation():
    assert bh_threshold(torch.tensor([3.0, 2.0]), 1000, 10**9, 0.05) is None
    assert bh_threshold(torch.zeros(0), 1000, 10**9, 0.05) is None
    with pytest.raises(ValueError):
        bh_threshold(torch.tensor([9.0]), 1000, 10**9, 0.0)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("threshold_test_pipeline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path_factory.mktemp("pipeline")
    module.run(out)
    return module, out


def test_thresholds_lowered_only_when_active(pipeline, tmp_path):
    _, out = pipeline
    base = SearchConfig.from_yaml(out / "search.yaml")
    assert base.fdr == 0.05
    on = search._Run(base.model_copy(update={"output_dir": tmp_path}))
    off = search._Run(base.model_copy(update={"output_dir": tmp_path, "fdr": None}))
    n_pixels = on.n_positions((96, 96))
    for spec in base.stages:
        if spec.decision.threshold == "auto":
            reach = base.fdr_reach if spec is base.stages[-1] else base.fdr_gate_reach
            assert on.threshold(spec, n_pixels) == pytest.approx(
                off.threshold(spec, n_pixels) - reach
            )
            # the anchor moves every auto threshold, only when the rule is on
            assert on.threshold(spec, n_pixels, -0.2) == pytest.approx(
                off.threshold(spec, n_pixels) - reach - 0.2
            )
            assert off.threshold(spec, n_pixels, -0.2) == off.threshold(spec, n_pixels)
    assert on.fdr_active and not off.fdr_active
    on.release()
    off.release()


def test_fdr_search(pipeline, tmp_path):
    module, out = pipeline
    base = SearchConfig.from_yaml(out / "search.yaml")
    done = search.run(base.model_copy(update={"output_dir": tmp_path / "fdr"}))
    summary = json.loads((tmp_path / "fdr" / "search_summary.json").read_text())
    for name in module.NAMES:
        rep = summary[name]["fdr"]
        assert rep["q"] == 0.05
        result = SearchResult.load(done[name])
        accepted = result.label[0] == int(PixelLabel.ACCEPTED)
        if rep["threshold"] is None:
            assert not bool(accepted.any())
            continue
        assert rep["threshold"] >= rep["floor"]
        assert bool((result.zscore[0][accepted] >= rep["threshold"] - 1e-6).all())
    # the synthetic particle is still picked
    p = module.PARTICLE
    r, c = p["top_left"]
    result = SearchResult.load(done[p["micrograph"]])
    assert int(result.label[0, r, c]) == int(PixelLabel.ACCEPTED)

    # fdr None: no report, thresholds as before
    plain = search.run(
        base.model_copy(update={"output_dir": tmp_path / "plain", "fdr": None})
    )
    summary = json.loads((tmp_path / "plain" / "search_summary.json").read_text())
    assert all("fdr" not in summary[name] for name in module.NAMES)
    assert sorted(plain) == list(module.NAMES)


def test_null_cut_1fp():
    """The z at which a null rate falls to one false peak, and when there is none."""

    def exponential(z):
        return 1000.0 * np.exp(-3.0 * (np.asarray(z, float) - 6.0))

    assert null_cut_1fp(exponential) == pytest.approx(
        6.0 + np.log(1000.0) / 3.0, abs=1e-4
    )
    assert null_cut_1fp(lambda z: np.full(np.shape(z), 0.5)) is None  # never above one
    assert null_cut_1fp(lambda z: np.full(np.shape(z), 5.0)) is None  # never below one
    # infinite below the curve (a summary's rate), still a crossing
    cut = null_cut_1fp(lambda z: np.where(np.asarray(z) < 7.0, np.inf, 0.1))
    assert cut == pytest.approx(7.0, abs=2e-3)


def test_bh_threshold_is_the_shared_rule():
    """Panther-EM's cut is Leopard-EM's engine-agnostic BH rule under its null."""
    rng = np.random.default_rng(0)
    z = torch.from_numpy(
        np.concatenate([rng.normal(9, 1, 50), 6 + rng.exponential(0.3, 3000)])
    )
    n_pos, n_hyp = 9000, 10**8
    shared = bh_cutoff(z.numpy(), nominal_null_rate(n_pos * n_hyp), 0.05)
    assert bh_threshold(z, n_pos, n_hyp, 0.05) == shared.bh_threshold

    def lower(t):  # a calibrated null below the formula: a lower cut
        return 0.2 * nominal_null_rate(n_pos * n_hyp)(t)

    assert bh_threshold(z, n_pos, n_hyp, 0.05, lower) <= shared.bh_threshold


def _null_summary(path, name, valid_area, scale):
    """A ``panther-em null`` summary: one micrograph's analytic and counted nulls."""
    t = np.round(np.arange(2.0, 10.0, 0.01), 3)
    rate = scale * 1e9 * 0.5 * np.exp(-0.5 * t**2) / 2.5066 / t  # ~ N Q(t)
    path.write_text(
        json.dumps(
            {
                "micrographs": {
                    name: {
                        "valid_area": valid_area,
                        "analytic": {"thresholds": t.tolist(), "rate": rate.tolist()},
                        "rate": {
                            "thresholds": t[400:].tolist(),
                            "rate": rate[400:].tolist(),
                        },
                    }
                }
            }
        )
    )
    return t, rate


def test_calibrated_null_rate(tmp_path):
    path = tmp_path / "null_summary.json"
    t, rate = _null_summary(path, "m0", 1000, 1.0)
    rate_fn, record = calibrated_null_rate(path, "m0", 2000, "analytic")
    assert rate_fn(np.array([t[300]]))[0] == pytest.approx(2 * rate[300])
    assert record["null"] == "analytic" and record["area_scale"] == 2.0
    counted, _ = calibrated_null_rate(path, "m0", 1000, "empirical")
    assert np.isinf(counted(np.array([5.0]))[0])  # below the copies' table
    assert counted(np.array([t[500]]))[0] == pytest.approx(rate[500])
    assert calibrated_null_rate(path, "nope", 1000) is None


def test_fdr_search_with_a_calibrated_null(pipeline, tmp_path):
    """``fdr_null``: a micrograph's cut under its calibrated null, the rest formula."""
    module, out = pipeline
    base = SearchConfig.from_yaml(out / "search.yaml")
    name = module.PARTICLE["micrograph"]
    summary_path = tmp_path / "null_summary.json"
    _null_summary(summary_path, name, 9000, 1e-3)  # a null far below the formula
    done = search.run(
        base.model_copy(
            update={"output_dir": tmp_path / "cal", "fdr_null": summary_path}
        )
    )
    summary = json.loads((tmp_path / "cal" / "search_summary.json").read_text())
    rep = summary[name]["fdr"]
    assert rep["null"] == "analytic" and rep["null_summary"] == str(summary_path)
    if rep["threshold"] is not None:
        assert rep["expected_false_picks"] < rep["expected_false_picks_formula"]
    others = [n for n in module.NAMES if n != name]
    assert all(summary[n]["fdr"]["null"] == "formula" for n in others)
    assert all("note" in summary[n]["fdr"] for n in others)
    result = SearchResult.load(done[name])
    r, c = module.PARTICLE["top_left"]
    assert int(result.label[0, r, c]) == int(PixelLabel.ACCEPTED)


def test_fdr_null_analytic_is_computed_by_the_search(pipeline, tmp_path, monkeypatch):
    """The default ``fdr_null: analytic``: each micrograph's own analytic null, no
    separate null run; ``formula`` keeps the formula; a search the model cannot cover
    falls back to the formula and says why."""
    module, out = pipeline
    base = SearchConfig.from_yaml(out / "search.yaml")
    assert base.fdr_null == "analytic"
    name = module.PARTICLE["micrograph"]
    one = base.model_copy(
        update={
            "micrographs": [m for m in base.micrographs if m.key == name],
            "output_dir": tmp_path / "analytic",
        }
    )
    search.run(one)
    rep = json.loads((tmp_path / "analytic" / "search_summary.json").read_text())[name][
        "fdr"
    ]
    assert rep["null"] == "analytic"
    assert np.isfinite(rep["null_cut_1fp"]) and rep["null_seconds"] > 0
    if rep["threshold"] is not None:
        assert rep["expected_false_picks"] < rep["expected_false_picks_formula"]
    # the reach is measured from the null's own 1-FP cut (fdr_anchor: null)
    assert rep["anchor"] == "null"
    assert rep["anchor_cut_1fp"] == pytest.approx(rep["null_cut_1fp"], abs=0.01)
    assert rep["anchor_shift"] == pytest.approx(
        rep["anchor_cut_1fp"] - rep["formula_cut_1fp"]
    )
    assert rep["floor"] == pytest.approx(rep["anchor_cut_1fp"] - base.fdr_reach)
    assert isinstance(rep["floor_limited"], bool)

    search.run(
        one.model_copy(
            update={"output_dir": tmp_path / "formula_anchor", "fdr_anchor": "formula"}
        )
    )
    rep = json.loads((tmp_path / "formula_anchor" / "search_summary.json").read_text())[
        name
    ]["fdr"]
    assert rep["null"] == "analytic" and rep["anchor"] == "formula"
    assert rep["floor"] == pytest.approx(rep["formula_cut_1fp"] - base.fdr_reach)

    search.run(
        one.model_copy(
            update={"output_dir": tmp_path / "formula", "fdr_null": "formula"}
        )
    )
    rep = json.loads((tmp_path / "formula" / "search_summary.json").read_text())[name][
        "fdr"
    ]
    assert rep["null"] == "formula"
    assert rep["expected_false_picks"] == rep["expected_false_picks_formula"]
    assert rep["anchor"] == "formula"  # nothing to anchor on
    assert rep["floor"] == pytest.approx(rep["formula_cut_1fp"] - base.fdr_reach)

    from panther_em.managers import null as null_program

    def unavailable(*args, **kwargs):
        raise NotImplementedError("a final stage over a subset of orientations")

    monkeypatch.setattr(null_program, "predict_search_null", unavailable)
    search.run(one.model_copy(update={"output_dir": tmp_path / "fallback"}))
    rep = json.loads((tmp_path / "fallback" / "search_summary.json").read_text())[name][
        "fdr"
    ]
    assert rep["null"] == "formula" and "subset of orientations" in rep["note"]
