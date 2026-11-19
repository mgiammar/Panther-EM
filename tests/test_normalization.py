"""After-CTF normalization (``SearchConfig.normalization: after_ctf``).

The factors equalize the predicted noise variance of every ``(plane, template)``
hypothesis; the error tables, template weights and CTF planes carry them
consistently; a search runs with them and still finds the synthetic particle.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from noise_case import noise_case

from panther_em.managers import SearchConfig, search
from panther_em.search import SearchResult, ThresholdTest
from panther_em.search.normalization import (
    AfterCtfNormalization,
    hypothesis_variance,
    plane_scales,
    relative_sd,
    row_scales,
    scale_table,
    template_power,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

RECTS = [(0, 0, 3, 4)]
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "threshold_test_pipeline.py"


@pytest.fixture(scope="module")
def case():
    case = noise_case()
    table = case.table(RECTS)
    return case, table, case.images(table)


def test_separable_variance_is_equalized_exactly():
    gen = torch.Generator().manual_seed(0)
    per_plane = 1.0 + 0.1 * torch.rand(5, generator=gen, dtype=torch.float64)
    per_row = 1.0 + 0.3 * torch.rand(40, generator=gen, dtype=torch.float64)
    e = per_plane[:, None] * per_row[None, :]
    a = row_scales(e)
    b = plane_scales(e * a.square()[None, :])
    scaled = e * a.square()[None, :] * b.square()[:, None]
    assert relative_sd(e) > 0.05
    assert relative_sd(scaled) < 1e-12
    assert float(scaled.mean()) == pytest.approx(float(e.mean()), rel=1e-12)


def test_interaction_is_what_remains():
    gen = torch.Generator().manual_seed(1)
    e = (1.0 + 0.1 * torch.rand(5, 1, generator=gen, dtype=torch.float64)) * (
        1.0 + 0.3 * torch.rand(1, 40, generator=gen, dtype=torch.float64)
    )
    e = e * (1.0 + 0.01 * torch.randn(5, 40, generator=gen, dtype=torch.float64))
    a = row_scales(e)
    after_rows = e * a.square()[None, :]
    after = after_rows * plane_scales(after_rows).square()[:, None]
    assert relative_sd(after) < relative_sd(after_rows) < relative_sd(e)
    assert relative_sd(after) == pytest.approx(0.01, rel=0.3)


def test_scaled_table_scales_the_threshold_test_consistently(case):
    _, table, images = case
    a = torch.linspace(0.8, 1.25, table.n_templates, dtype=torch.float64)
    before = ThresholdTest(table).scales(images)
    after = ThresholdTest(scale_table(table, a)).scales(images)
    a2 = a.square()[None, :]
    torch.testing.assert_close(after.q_kept, before.q_kept * a2, rtol=1e-5, atol=0)
    torch.testing.assert_close(after.q_perp, before.q_perp * a2, rtol=1e-4, atol=1e-12)
    torch.testing.assert_close(after.lam, before.lam, rtol=1e-5, atol=1e-9)
    torch.testing.assert_close(after.rho, before.rho, rtol=1e-5, atol=1e-9)
    with pytest.raises(ValueError, match="template rows"):
        scale_table(table, a[:-1])


def test_q_kept_is_the_hypothesis_variance(case):
    _, table, images = case
    q = ThresholdTest(table).scales(images).q_kept
    e = hypothesis_variance(images.noise, template_power(table))
    torch.testing.assert_close(e, q, rtol=1e-5, atol=0)


def test_apply_equalizes_the_micrographs_variance(case):
    _, table, images = case
    norm = AfterCtfNormalization.create(table, images.noise)
    scaled, report = norm.apply(images)
    a2 = norm.row_scale.square()[None, :]
    e = hypothesis_variance(scaled.noise, template_power(table)) * a2
    v = report["variance_rel_sd"]
    assert v["after"] == pytest.approx(relative_sd(e), rel=1e-6)
    assert v["after"] <= v["rows"] <= v["before"]
    assert v["after"] < 0.5 * v["before"]
    b = torch.tensor(report["plane_scale"], dtype=torch.float32)
    torch.testing.assert_close(scaled.ctf, images.ctf * b[:, None, None])
    torch.testing.assert_close(
        scaled.noise, images.noise * b.double().square()[:, None], rtol=1e-6, atol=0
    )
    torch.testing.assert_close(scaled.image, images.image)
    torch.testing.assert_close(scaled.level, images.level)


def test_template_scale_rows(case):
    c, _, _ = case
    template = c.template
    base = template.weights.us.clone()
    a = torch.linspace(0.5, 2.0, template.n_templates, dtype=torch.float64)
    template.scale_rows(a)
    torch.testing.assert_close(template.weights.us, base * a.float()[:, None, None])
    template.scale_rows(torch.ones_like(a))  # replaces, does not compound
    torch.testing.assert_close(template.weights.us, base)
    with pytest.raises(ValueError, match="positive"):
        template.scale_rows(-a)
    with pytest.raises(ValueError, match="template rows"):
        template.scale_rows(a[:-1])


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("threshold_test_pipeline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path_factory.mktemp("pipeline")
    module.run(out)
    return module, out


def test_default_is_after_ctf(pipeline):
    _, out = pipeline
    config = SearchConfig.from_yaml(out / "search.yaml")
    assert config.normalization == "after_ctf"
    data = config.model_dump()
    del data["normalization"]  # a config written before the option existed
    assert SearchConfig.model_validate(data).normalization == "after_ctf"


def test_before_ctf_runs_no_new_code(pipeline, tmp_path):
    _, out = pipeline
    config = SearchConfig.from_yaml(out / "search.yaml")
    config = config.model_copy(
        update={"output_dir": tmp_path, "normalization": "before_ctf"}
    )
    state = search._Run(config)
    assert state.normalization is None and state.template.row_scale is None
    for tag, table in state.tables.items():
        raw = search.PerSpatialFrequencyErrorTable.load(config.tables[tag])
        torch.testing.assert_close(
            table.reconstruction_autocorr, raw.reconstruction_autocorr
        )
    state.release()


def test_after_ctf_search(pipeline, tmp_path):
    module, out = pipeline
    base = SearchConfig.from_yaml(out / "search.yaml")
    config = base.model_copy(
        update={"output_dir": tmp_path / "after", "normalization": "after_ctf"}
    )
    state = search._Run(config)
    a = state.normalization.row_scale
    assert state.template.row_scale is not None
    torch.testing.assert_close(state.template.row_scale, a)
    for tag, table in state.tables.items():  # every table carries a
        raw = search.PerSpatialFrequencyErrorTable.load(config.tables[tag])
        torch.testing.assert_close(
            table.reconstruction_autocorr,
            raw.reconstruction_autocorr * a.float().square()[:, None],
        )
    state.release()

    done = search.run(config)
    assert sorted(done) == list(module.NAMES)
    summary = json.loads((tmp_path / "after" / "search_summary.json").read_text())
    old_way = search.run(
        base.model_copy(
            update={"output_dir": tmp_path / "before", "normalization": "before_ctf"}
        )
    )
    for name in module.NAMES:
        v = summary[name]["normalization"]["variance_rel_sd"]
        assert v["after"] <= v["rows"] + 1e-12 and v["after"] < v["before"]

    # the synthetic particle is found as before: same label, orientation and plane,
    # z within a few percent (its hypothesis's factor a_n b_d is close to 1)
    p = module.PARTICLE
    r, c = p["top_left"]
    before = SearchResult.load(old_way[p["micrograph"]])
    after = SearchResult.load(done[p["micrograph"]])
    assert int(after.label[0, r, c]) == int(before.label[0, r, c])
    for key in ("orientation_index", "defocus_index"):
        assert int(after.decode()[key][0, r, c]) == int(before.decode()[key][0, r, c])
    z0, z1 = float(before.zscore[0, r, c]), float(after.zscore[0, r, c])
    assert z1 == pytest.approx(z0, rel=0.1)
