"""null: phase-randomised copies, the analytic null model and the ``null`` program.

A module fixture runs the synthetic pipeline (``scripts/threshold_test_pipeline.py``,
whose step 7 is ``null``) in process; the tests reuse its files and configs.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import mrcfile
import numpy as np
import pytest
import roma
import torch
import yaml
from click.testing import CliRunner
from leopard_em.analysis.analytic_null import PowerWeighting
from leopard_em.analysis.empirical_null import distinct_peaks
from leopard_em.utils.phase_randomization import phase_randomize_image
from noise_case import colored, micrographs, noise_case

from panther_em.cli import main
from panther_em.filters import ctf_planes
from panther_em.managers import null, search
from panther_em.managers._common import read_config, read_micrograph, read_volume
from panther_em.managers.decompose import DecomposeConfig
from panther_em.orientations import healpix_orientations
from panther_em.search import Searcher, SearchResult
from panther_em.search.normalization import hypothesis_variance, template_power
from panther_em.search.null_model import (
    TemplateProjector,
    healpix_nside,
    lag_grid_planes,
    reference_grid,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "threshold_test_pipeline.py"

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """``(script module, output dir)`` after one run of the pipeline."""
    spec = importlib.util.spec_from_file_location("threshold_test_pipeline", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path_factory.mktemp("pipeline")
    module.run(out)
    return module, out


def search_variant(out: Path, path: Path, **changes) -> Path:
    """The pipeline's search config with ``changes``, written to ``path``."""
    data = yaml.safe_load((out / "search.yaml").read_text())
    data.update(changes)
    data["output_dir"] = str(path.parent / f"{path.stem}_search")
    path.write_text(yaml.safe_dump(data))
    return path


def low_defocus(out: Path, path: Path, **changes) -> Path:
    """The pipeline's search at 1000 / 2000 A defocus, where the 96 px micrograph
    samples its CTF finely enough for the template's lag grid (1-2 um is aliased)."""
    data = yaml.safe_load((out / "search.yaml").read_text())
    for entry in data["micrographs"]:
        entry["defocus_u"] = entry["defocus_v"] = 1500.0
    return search_variant(
        out,
        path,
        micrographs=data["micrographs"],
        defocus_offsets=[-500.0, 500.0],
        **changes,
    )


# --------------------------------------------------------------------------- #
# Phase-randomised copies
# --------------------------------------------------------------------------- #


def test_phase_randomized_copy_passes_the_whitening_check():
    """A copy has its micrograph's amplitude spectrum, so the micrograph's whitening
    row applies to it; another micrograph's copy is refused."""
    whitening = noise_case().whitening
    images = dict(micrographs())
    for name, image in images.items():
        copy = phase_randomize_image(image, seed=7)
        whitening.check_image(name, copy)
        assert not torch.allclose(copy, image)
    with pytest.raises(ValueError, match="differs"):
        whitening.check_image("m0", phase_randomize_image(images["m2"], seed=7))


def test_null_image_is_the_copy_as_the_search_reads_it(tmp_path):
    """``null_image`` equals reading a phase-randomised copy written to an MRC file
    (bit for bit), negated with ``invert_contrast``; seeds give different copies."""
    source = tmp_path / "m.mrc"
    mrcfile.write(str(source), (colored((64, 80), 3) + 5.0).numpy())
    data = np.asarray(mrcfile.read(str(source)), dtype=np.float32)
    copy = tmp_path / "copy.mrc"
    mrcfile.write(str(copy), phase_randomize_image(data, 11).numpy())
    expected = read_micrograph(copy)
    assert torch.equal(null.null_image(source, 11), expected)
    assert torch.equal(null.null_image(source, 11, invert_contrast=True), -expected)
    assert not torch.equal(null.null_image(source, 12), expected)


# --------------------------------------------------------------------------- #
# The analytic null model
# --------------------------------------------------------------------------- #


def test_lag_grid_planes():
    """Exact where the lag grid divides the image grid; interpolated otherwise."""
    planes = ctf_planes((96, 96), [10000.0, 15000.0], pixel_size=1.5)
    exact = lag_grid_planes(planes, (96, 96), (48, 48))
    torch.testing.assert_close(exact, planes[:, ::2, ::2], rtol=0, atol=0)
    fine = ctf_planes((1000, 1000), [10000.0, 15000.0], pixel_size=1.5)
    coarse = ctf_planes((40, 40), [10000.0, 15000.0], pixel_size=1.5)
    sampled = lag_grid_planes(fine, (1000, 1000), (40, 40))
    torch.testing.assert_close(sampled, coarse, rtol=0, atol=1e-4)
    with pytest.raises(ValueError, match="planes must be"):
        lag_grid_planes(planes, (96, 94), (48, 48))


def model_variances(config_path: Path, normalization: str):
    """The model's ``(plane, row)`` noise variance at the grid rows, and Panther's."""
    config = search.SearchConfig.from_yaml(config_path).model_copy(
        update={"normalization": normalization}
    )
    state = search._Run(config)
    micrograph = state.prepare(search._read(config, config.micrographs[1]))
    result = state.template.result
    box = state.template.template_shape
    weighting = PowerWeighting(
        torch.fft.rfft2(micrograph.image), micrograph.image_shape, box
    )
    reference = (
        None
        if state.reference_noise is None
        else reference_grid(*state.reference_noise, box)
    )
    projector = TemplateProjector(
        read_volume(read_config(config.decomposition, DecomposeConfig).volume),
        state.whitening.template_filter(box),
        weighting,
        lag_grid_planes(micrograph.ctf, micrograph.image_shape, weighting.lag_shape),
        support=state.template.reconstructor.support,
        reference=reference,
    )
    angles = torch.stack(
        [
            torch.as_tensor(result.phi_values),
            torch.as_tensor(result.theta_values),
            torch.zeros(len(result.phi_values)),
        ],
        dim=-1,
    )
    q = projector(roma.euler_to_rotmat("ZYZ", angles.float(), degrees=True))
    model = weighting.inner(q, q).double()
    table = state.tables[state._final_table_tag()]  # rows scaled by a^2 (after_ctf)
    panther = hypothesis_variance(micrograph.noise, template_power(table))
    state.release()
    return model, panther


def test_model_variance_follows_the_noise_model(pipeline, tmp_path):
    """At the grid rows, the model's templates have the noise variance per plane and
    row that Panther's own noise model gives them; after-CTF evens them out."""
    _, out = pipeline
    path = low_defocus(out, tmp_path / "low.yaml")
    model, panther = model_variances(path, "before_ctf")
    rel = lambda v: v / v.mean()  # noqa: E731
    assert float(np.corrcoef(model.flatten(), panther.flatten())[0, 1]) > 0.99
    # within the polar warp (not modelled): 5 % on this coarse 32 x 16 grid
    torch.testing.assert_close(rel(model), rel(panther), rtol=0.08, atol=0)
    before = float(model.std() / model.mean())
    model, panther = model_variances(path, "after_ctf")
    torch.testing.assert_close(
        rel(model.mean(1)), torch.ones(2, dtype=torch.float64), rtol=0, atol=0.02
    )
    assert float(model.std() / model.mean()) < 0.25 * before


def test_healpix_nside():
    assert healpix_nside(*healpix_orientations(4)) == 4
    with pytest.raises(ValueError, match="HEALPix"):
        healpix_nside(np.array([0.0, 10.0]), np.array([30.0, 40.0]))


def test_predicted_null_matches_measured_nulls(pipeline, tmp_path):
    """One dense stage (no gate), after-CTF: the predicted 1- and 3-false-positive
    thresholds agree with those counted on 30 phase-randomised copies."""
    _, out = pipeline
    data = yaml.safe_load((out / "search.yaml").read_text())
    full = data["stages"][1]
    dense = {
        "name": "dense",
        "rectangles": full["rectangles"],
        "table": full["table"],
        "n_psi": full["n_psi"],
        "decision": {"threshold": "auto", "accept": True},
    }
    path = low_defocus(
        out, tmp_path / "dense.yaml", stages=[dense], noise_table=full["table"]
    )
    summary = null.run(
        null.NullConfig(
            search=path,
            output_dir=tmp_path / "nulls",
            mode="both",
            micrographs=["m1"],
            seeds=list(range(1, 31)),
            false_positives=[1.0, 3.0],
            peak_floor=3.0,
            analytic=null.AnalyticSpec(
                n_mean_samples=500, n_kappa_pairs=16, n_lattice_samples=48
            ),
        )
    )
    record = summary["micrographs"]["m1"]
    assert record["gate"]["floor"] is None  # one stage: nothing to drop
    for cut in record["cuts"]:
        assert cut["complete"]
        assert abs(cut["analytic"] - cut["empirical"]) < 0.15, cut
        assert cut["analytic"] < cut["formula"]  # correlated tests: below the formula
    analytic = record["analytic"]
    assert 0.0 < analytic["kappa"] < 0.1 and 0.0 <= analytic["rho_bar"] < 0.1
    assert analytic["details"]["normalization"] == "after_ctf"
    rate = np.asarray(analytic["rate"])
    assert bool(np.all(np.diff(rate) <= 1e-12))


def test_predicted_refined_null(pipeline, tmp_path):
    """With a refine_template config, the analytic mode also predicts the threshold
    on refined z-scores, from the search's own templates and refinement's exact ones."""
    _, out = pipeline
    path = low_defocus(out, tmp_path / "low.yaml")
    refine = tmp_path / "refine.yaml"
    refine.write_text(
        yaml.safe_dump(
            {
                "orientation_refinement_config": {
                    "enabled": True,
                    "psi_step_coarse": 10.0,
                    "psi_step_fine": 5.0,
                    "theta_step_coarse": 10.0,
                    "theta_step_fine": 5.0,
                },
                "defocus_refinement_config": {
                    "enabled": True,
                    "defocus_min": -200.0,
                    "defocus_max": 200.0,
                    "defocus_step": 200.0,
                },
            }
        )
    )
    summary = null.run(
        null.NullConfig(
            search=path,
            output_dir=tmp_path / "nulls",
            mode="analytic",
            micrographs=["m1"],
            false_positives=[1.0],
            analytic=null.AnalyticSpec(
                n_mean_samples=200, n_kappa_pairs=8, n_lattice_samples=16
            ),
            refine=null.RefineSpec(
                config=refine, candidate_threshold=3.0, n_candidates=2, n_draws=4000
            ),
        )
    )
    record = summary["micrographs"]["m1"]
    cut = record["cuts"][0]
    # Leopard-EM re-scores exactly what the search approximated: the refined cut may
    # fall below the search's (here it does, on a coarse decomposition)
    assert np.isfinite(cut["analytic_refined"]) and np.isfinite(cut["analytic"])
    refined = record["analytic_refined"]
    assert 0.5 < refined["details"]["captured_power"] < 1.5
    assert bool(np.all(np.diff(np.asarray(refined["rate"])) <= 1e-9))
    assert refined["candidate_threshold"] == 3.0
    assert refined["gain"]["n_windows"] == 2
    assert refined["details"]["defocus_offsets"] == [-200.0, 0.0, 200.0]
    assert refined["details"]["normalization"] == "after_ctf"
    survival = np.asarray(refined["gain"]["survival"])
    assert bool(np.all(np.diff(survival, axis=1) <= 1e-12))
    with pytest.raises(ValueError, match="refine needs mode"):
        null.NullConfig(
            search=path,
            output_dir=tmp_path / "x",
            refine=null.RefineSpec(config=refine, candidate_threshold=3.0),
        )


# --------------------------------------------------------------------------- #
# The program
# --------------------------------------------------------------------------- #


def test_pipeline_null_step(pipeline):
    """The pipeline's ``null`` step: copies of m1 (seeds 1001-1003: m1 is the second
    micrograph), its peaks file and summary agree, the gate's floor is recorded."""
    module, out = pipeline
    config = null.NullConfig.from_yaml(out / "null.yaml")
    summary = json.loads(config.summary_path.read_text())
    record = summary["micrographs"]["m1"]
    assert list(summary["micrographs"]) == ["m1"]
    assert [c["seed"] for c in record["copies"]] == [1001, 1002, 1003]
    records, settings = null.read_null_peaks(config.peaks_path)
    assert settings["peak_radius"] == 10.0 and settings["border"] == 0
    peaks = null.null_peaks(records["m1"], record["peak_floor"], 10.0)
    assert peaks.n_copies == len(module.NULL_SEEDS)
    threshold = record["thresholds"][-1]
    assert record["peak_floor"] == pytest.approx(threshold - null.DEFAULT_FLOOR_OFFSET)
    stages = search.SearchConfig.from_yaml(out / "search.yaml").stages
    zeta = stages[0].decision.calibration.zeta
    assert record["gate"]["floor"] == pytest.approx(zeta * record["thresholds"][0])
    for cut, fp in zip(record["cuts"], (1.0, 5.0), strict=True):
        assert cut["false_positives"] == fp
        assert cut["empirical"] == pytest.approx(peaks.zscore_cutoff(fp))
        assert cut["complete"] == (cut["empirical"] >= record["gate"]["floor"])
        assert np.isfinite(cut["analytic"])
    first = record["rate"]["thresholds"][0]  # the floor, rounded up to the step
    assert record["peak_floor"] <= first < record["peak_floor"] + 0.05
    pooled = np.concatenate([c.z for c in records["m1"].values()])
    assert record["rate"]["counts"][0] == int((pooled >= first).sum())
    assert len(pooled) == sum(c["peaks"] for c in record["copies"])
    assert record["normalization"]["plane_scale"]  # after-CTF: the source's b_d


def test_null_program(pipeline, tmp_path, monkeypatch):
    """``panther-em null``: peaks only where the final stage scored, results kept on
    request, a rerun searches only the missing copies, inconsistent reruns refused."""
    _, out = pipeline
    config = null.NullConfig(
        search=out / "search.yaml",
        output_dir=tmp_path / "nulls",
        micrographs=["m2"],
        seeds=[4, 5],
        keep_results=True,
        border=3,
    )
    config.to_yaml(tmp_path / "null.yaml")
    ran = CliRunner().invoke(main, ["null", str(tmp_path / "null.yaml")])
    assert ran.exit_code == 0, ran.output
    summary = json.loads(config.summary_path.read_text())
    record = summary["micrographs"]["m2"]
    assert "analytic" not in record and "analytic" not in record["cuts"][0]
    assert [c["seed"] for c in record["copies"]] == [2004, 2005]
    records, _ = null.read_null_peaks(config.peaks_path)
    for seed in (2004, 2005):
        result = SearchResult.load(config.result_path("m2", seed))
        assert read_config(config.result_path("m2", seed), null.NullConfig) == config
        z, masked = null.final_zscore(result)
        final = result.last_stage[0].numpy() == len(result.stages) - 1
        assert bool(np.all(np.isneginf(z[~final])))
        assert (~final).any()  # the gate dropped pixels before the final stage
        found = distinct_peaks(z, record["peak_floor"], 10.0, 3, masked)
        copy = records["m2"][seed]
        np.testing.assert_array_equal(copy.z, found.zscore)
        assert bool(final[copy.y, copy.x].all())
        assert copy.valid_area == (result.zscore.shape[1] - 6) ** 2

    # a rerun with one more seed searches only that copy
    searched = []
    run = Searcher.run
    monkeypatch.setattr(
        Searcher,
        "run",
        lambda self, m, **kw: searched.append(m.name) or run(self, m, **kw),
    )
    more = config.model_copy(update={"seeds": [4, 5, 6], "keep_results": False})
    record = null.run(more)["micrographs"]["m2"]
    assert searched == ["m2"]
    assert [c["seed"] for c in record["copies"]] == [2004, 2005, 2006]

    with pytest.raises(ValueError, match="other peak settings"):
        null.run(more.model_copy(update={"border": 0}))
    with pytest.raises(ValueError, match="above the floor"):
        null.run(more.model_copy(update={"peak_floor": 0.0}))
    with pytest.raises(ValueError, match="search's output directory"):
        null.run(more.model_copy(update={"output_dir": out}))
    with pytest.raises(ValueError, match="not in the search config"):
        null.run(more.model_copy(update={"micrographs": ["nope"]}))
    with h5py.File(config.peaks_path, "r") as f:
        assert sorted(f["m2"]) == ["s2004", "s2005", "s2006"]


def test_null_config_validation(tmp_path):
    base = {"search": "s.yaml", "output_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="distinct"):
        null.NullConfig(**base, seeds=[1, 1])
    with pytest.raises(ValueError, match="seed_stride"):
        null.NullConfig(**base, seeds=[1, 5000])
    with pytest.raises(ValueError, match="positive"):
        null.NullConfig(**base, false_positives=[0.0])
    config = null.NullConfig(**base, mode="analytic")
    assert config.predicted and not config.empirical
