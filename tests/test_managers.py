"""managers and cli: configs, provenance, resuming, the Leopard-EM export, the CLI.

A module fixture runs the synthetic pipeline (``scripts/threshold_test_pipeline.py``)
in process; the tests reuse its files and configs.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import h5py
import mrcfile
import numpy as np
import pytest
import torch
import yaml
from click.testing import CliRunner
from leopard_em.pydantic_models.data_structures.particle_stack import (
    ParticleStackHDF5,
)
from leopard_em.pydantic_models.formats import (
    LARGE_STD_PADDING,
    MATCH_TEMPLATE_DF_COLUMN_ORDER,
    STATISTIC_MAP_PATH_TO_HDF5_DATASET,
)
from leopard_em.pydantic_models.results.match_template_result import (
    MatchTemplateResultHDF5,
)
from pydantic import ValidationError

from panther_em.cli import main
from panther_em.managers import (
    CalibrateConfig,
    DecomposeConfig,
    ExportConfig,
    SearchConfig,
    WhiteningConfig,
    calibrate,
    leopard,
    search,
)
from panther_em.managers._common import (
    read_config,
    read_micrograph,
    read_volume,
    save_output,
)
from panther_em.search import (
    ExecutionConfig,
    Micrograph,
    PixelLabel,
    Searcher,
    SearchResult,
)
from panther_em.search.correlation.contraction import BatchShape

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "threshold_test_pipeline.py"
CONFIGS = {
    "whitening": WhiteningConfig,
    "decompose": DecomposeConfig,
    "calibrate": CalibrateConfig,
    "search": SearchConfig,
    "export": ExportConfig,
}

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


def test_yaml_round_trip_and_typos(pipeline):
    _, out = pipeline
    for program, cls in CONFIGS.items():
        config = cls.from_yaml(out / f"{program}.yaml")
        text = config.to_yaml()
        assert cls.model_validate(yaml.safe_load(text)) == config
        data = yaml.safe_load(text)
        data["not_a_field"] = 1
        with pytest.raises(ValidationError, match="not_a_field"):
            cls.model_validate(data)


def test_config_validation(pipeline):
    _, out = pipeline
    data = yaml.safe_load((out / "search.yaml").read_text())
    data["stages"][0]["table"] = "missing"
    with pytest.raises(ValidationError, match="unknown table"):
        SearchConfig.model_validate(data)
    data["stages"][0]["table"] = None
    with pytest.raises(ValidationError, match="needs a table"):
        SearchConfig.model_validate(data)
    with pytest.raises(ValidationError, match="unique"):
        WhiteningConfig(
            micrographs=["a/m.mrc", "b/m.mrc"], output="w.h5", band_cutoff=0.3
        )
    with pytest.raises(ValidationError, match="tables, suggest"):
        CalibrateConfig(
            volume="v.mrc", decomposition="d.h5", whitening="w.h5", output_dir="."
        )
    with pytest.raises(ValidationError):  # a rectangle needs a positive extent
        CalibrateConfig(
            volume="v.mrc",
            decomposition="d.h5",
            whitening="w.h5",
            output_dir=".",
            tables={"a": [(0, 0, 0, 4)]},
        )


def test_execution_config():
    config = search.execution_config(
        {"direct_precision": "fp32", "tuning": {"batch": [256, 128], "cell_chunk": 8}}
    )
    assert isinstance(config, ExecutionConfig)
    assert config.direct_precision == "fp32"
    assert config.tuning.batch == BatchShape(256, 128)
    assert config.tuning.cell_chunk == 8
    with pytest.raises(TypeError):
        search.execution_config({"no_such_field": 1})


def test_outputs_record_their_configs(pipeline):
    _, out = pipeline
    decompose = read_config(out / "decomposition.h5", DecomposeConfig)
    assert decompose == DecomposeConfig.from_yaml(out / "decompose.yaml")
    assert read_config(out / "decomposition.h5", SearchConfig) is None
    assert read_config(out / "whitening.h5", WhiteningConfig) is not None
    assert read_config(out / "tables_full.h5", CalibrateConfig) is not None
    with h5py.File(out / "m1_result.h5", "r") as f:
        assert f.attrs["panther_em_program"] == "search"
        assert f.attrs["panther_em_micrograph"] == "m1"
    assert not list(out.glob(".*partial*"))
    report = yaml.safe_load((out / "calibration.json").read_text())
    assert report["tables"]["full"]["truncation_error"]["rho"] >= 0
    assert report["suggestions"][0]["candidates"]


def test_save_output_is_atomic(tmp_path, pipeline):
    _, out = pipeline
    config = ExportConfig.from_yaml(out / "export.yaml")

    def fails(path):
        Path(path).write_bytes(b"half")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        save_output(fails, tmp_path / "x.h5", config)
    assert not (tmp_path / "x.h5").exists()


def test_mrc_readers(tmp_path):
    image = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    mrcfile.write(str(tmp_path / "m.mrc"), image)
    read = read_micrograph(tmp_path / "m.mrc")
    assert read.shape == (3, 4) and abs(float(read.mean())) < 1e-6
    mrcfile.write(str(tmp_path / "v.mrc"), np.zeros((4, 4, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="cubic"):
        read_volume(tmp_path / "v.mrc")


def test_search_resumes(pipeline, tmp_path, monkeypatch):
    _, out = pipeline
    config = SearchConfig.from_yaml(out / "search.yaml")
    for name in ("m0", "m1", "m2"):
        for kind in ("result", "micrograph"):
            shutil.copy(out / f"{name}_{kind}.h5", tmp_path / f"{name}_{kind}.h5")
    config = config.model_copy(update={"output_dir": tmp_path})
    (tmp_path / "m1_result.h5").unlink()

    searched = []
    run = Searcher.run
    monkeypatch.setattr(
        Searcher, "run", lambda self, m: searched.append(m.name) or run(self, m)
    )
    monkeypatch.setattr(  # the cached micrograph is reused, not prepared again
        Micrograph, "prepare", classmethod(lambda *a, **k: pytest.fail("prepared"))
    )
    done = search.run(config)
    assert searched == ["m1"]
    assert sorted(done) == ["m0", "m1", "m2"]
    # the summary breaks the micrograph's time down by pipeline step
    record = json.loads((tmp_path / "search_summary.json").read_text())["m1"]
    steps = ("prepare_s", "search_s", "save_result_s", "leopard_maps_s")
    assert all(record[k] >= 0.0 for k in steps)
    assert record["elapsed_s"] == pytest.approx(sum(record[k] for k in steps))
    for stage in record["stages"]:
        parts = stage["featurize_s"] + stage["execute_s"] + stage["other_s"]
        assert parts == pytest.approx(stage["elapsed_s"])
    original = h5py.File(out / "m1_result.h5", "r")
    again = h5py.File(tmp_path / "m1_result.h5", "r")
    with original, again:
        np.testing.assert_array_equal(original["label"][()], again["label"][()])


def test_export_stack(pipeline):
    """The particle stack Leopard-EM reads: one row per particle, maps by path."""
    module, out = pipeline
    stack = ParticleStackHDF5(hdf5_path=str(out / "particles.h5"))
    df = stack.get_dataframe_copy()
    assert set(MATCH_TEMPLATE_DF_COLUMN_ORDER) <= set(df.columns)
    assert set(leopard.EXTRA_COLUMNS) <= set(df.columns)
    assert stack.extracted_box_size == (module.BOX + 16, module.BOX + 16)
    assert stack.original_template_size == (module.BOX, module.BOX)
    assert list(df["particle_index"]) == list(range(len(df)))
    assert df["particle_id"].is_unique

    # the planted particle, at its pose, from m0 only
    r, c = module.PARTICLE["top_left"]
    first = df.iloc[0]
    assert (first["pos_y"], first["pos_x"]) == (r, c)
    assert first["pos_x_img"] == c + module.BOX // 2
    assert first["relative_defocus"] == pytest.approx(5000.0)
    assert first["panther_label"] == "ACCEPTED"
    assert first["total_correlations"] == 2 * 12 * module.N_PSI
    assert first["micrograph_path"].endswith("m0.mrc")
    assert first["template_path"].endswith("volume.mrc")
    assert set(df["micrograph_path"].map(lambda p: Path(p).stem)) == {"m0"}
    for column in STATISTIC_MAP_PATH_TO_HDF5_DATASET:
        assert set(df[column]) == {str(out / "m0_result.h5")}

    # one particle per accepted region: every accepted pixel lies near a particle
    result = SearchResult.load(out / "m0_result.h5")
    accepted = torch.nonzero(result.label[0] == int(PixelLabel.ACCEPTED)).float()
    picks = torch.tensor(df[["pos_y", "pos_x"]].to_numpy().copy(), dtype=torch.float32)
    assert float(torch.cdist(accepted, picks).min(1).values.max()) <= 10.0
    assert len(df) < len(accepted)

    # Leopard-EM reads the statistic maps through the path columns
    maps = stack.get_local_stat_maps(use_stored=False)
    assert all(v.shape == (len(df), 17, 17) for v in maps.values())


def test_result_is_a_leopard_match_template_result(pipeline):
    _, out = pipeline
    path = out / "m0_result.h5"
    leo = MatchTemplateResultHDF5.from_hdf5(path)
    result = SearchResult.load(path)  # still a Panther-EM result
    searched = result.last_stage[0] >= 0
    assert bool(searched.all())
    # correlations in units of the median standard deviation (Leopard-EM's scale);
    # the z-score is scale-free
    scale = leopard.correlation_scale(result)
    with h5py.File(path, "r") as f:
        assert f.attrs["panther_em_correlation_scale"] == pytest.approx(scale)
    std = result.variance[0].float().sqrt()
    torch.testing.assert_close(leo.scaled_mip, result.zscore[0].float())
    torch.testing.assert_close(leo.mip, result.mip[0].float() / scale)
    torch.testing.assert_close(leo.correlation_average, result.mean[0].float() / scale)
    torch.testing.assert_close(leo.correlation_variance, std / scale)
    assert float(leo.correlation_variance.median()) == pytest.approx(1.0, abs=1e-3)
    z = (leo.mip - leo.correlation_average) / leo.correlation_variance
    torch.testing.assert_close(z, leo.scaled_mip, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(leo.orientation_psi, result.psi_degrees()[0])
    assert leo.total_defocus == 2 and leo.total_projections == 2 * 12 * 64


def test_unsearched_pixels_are_padded(pipeline):
    _, out = pipeline
    result = SearchResult.load(out / "m1_result.h5")
    result.last_stage[0, :3] = -1  # pretend the first rows were masked
    maps = leopard.match_template_maps(result)
    assert bool((maps["scaled_mip"][:3] == 0).all())
    assert bool((maps["correlation_variance"][:3] == LARGE_STD_PADDING).all())
    assert bool(torch.isfinite(maps["orientation_phi"]).all())


def test_cli(pipeline, tmp_path):
    _, out = pipeline
    runner = CliRunner()
    listing = runner.invoke(main, ["--help"])
    assert listing.exit_code == 0
    for program in CONFIGS:
        assert program in listing.output

    example = runner.invoke(main, ["search", "--example"])
    assert example.exit_code == 0
    assert yaml.safe_load(example.output)["decomposition"] == "<required>"

    config = yaml.safe_load((out / "export.yaml").read_text())
    config["output"] = str(tmp_path / "again.h5")
    (tmp_path / "export.yaml").write_text(yaml.safe_dump(config))
    ran = runner.invoke(main, ["export", str(tmp_path / "export.yaml")])
    assert ran.exit_code == 0, ran.output
    again = ParticleStackHDF5(hdf5_path=str(tmp_path / "again.h5"))
    before = ParticleStackHDF5(hdf5_path=str(out / "particles.h5"))
    assert again.get_dataframe_copy().equals(before.get_dataframe_copy())

    config["typo"] = 1
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump(config))
    bad = runner.invoke(main, ["export", str(tmp_path / "bad.yaml")])
    assert bad.exit_code != 0 and "typo" in bad.output
    assert runner.invoke(main, ["export"]).exit_code != 0


def test_calibrate_needs_energies_for_suggestions(pipeline, tmp_path):
    _, out = pipeline
    with h5py.File(out / "decomposition.h5", "r") as f:
        assert "block_energy" in f
    shutil.copy(out / "decomposition.h5", tmp_path / "d.h5")
    with h5py.File(tmp_path / "d.h5", "r+") as f:
        del f["block_energy"]
    config = CalibrateConfig.from_yaml(out / "calibrate.yaml").model_copy(
        update={
            "decomposition": tmp_path / "d.h5",
            "output_dir": tmp_path,
            "tables": {},
        }
    )
    with pytest.raises(ValueError, match="block energies"):
        calibrate.run(config)
