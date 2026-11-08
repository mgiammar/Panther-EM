"""managers and cli: configs, provenance, resuming, export and the command line.

A module fixture runs the synthetic pipeline (``scripts/threshold_test_pipeline.py``)
in process; the tests reuse its files and configs.
"""

from __future__ import annotations

import csv
import importlib.util
import shutil
from pathlib import Path

import h5py
import mrcfile
import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from pydantic import ValidationError

from panther_em.cli import main
from panther_em.managers import (
    CalibrateConfig,
    DecomposeConfig,
    ExportConfig,
    SearchConfig,
    WhiteningConfig,
    calibrate,
    export,
    search,
)
from panther_em.managers._common import (
    read_config,
    read_micrograph,
    read_volume,
    save_output,
)
from panther_em.search import ExecutionConfig, Micrograph, Searcher
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
    original = h5py.File(out / "m1_result.h5", "r")
    again = h5py.File(tmp_path / "m1_result.h5", "r")
    with original, again:
        np.testing.assert_array_equal(original["label"][()], again["label"][()])


def test_export_rows(pipeline):
    module, out = pipeline
    with (out / "particles.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0]) == export.COLUMNS
    assert all(r["panther_label"] == "ACCEPTED" for r in rows)
    assert [int(r["particle_index"]) for r in rows] == list(range(len(rows)))
    r, c = module.PARTICLE["top_left"]
    (particle,) = [x for x in rows if (int(x["pos_y"]), int(x["pos_x"])) == (r, c)]
    assert particle["micrograph_path"].endswith("m0.mrc")
    assert particle["template_path"].endswith("volume.mrc")
    assert float(particle["relative_defocus"]) == pytest.approx(5000.0)
    assert int(particle["pos_x_img"]) == c + module.BOX // 2
    assert int(particle["total_correlations"]) == 2 * 12 * module.N_PSI
    assert float(particle["scaled_mip"]) > 5.0


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
    config["output"] = str(tmp_path / "again.csv")
    (tmp_path / "export.yaml").write_text(yaml.safe_dump(config))
    ran = runner.invoke(main, ["export", str(tmp_path / "export.yaml")])
    assert ran.exit_code == 0, ran.output
    assert (tmp_path / "again.csv").read_text() == (out / "particles.csv").read_text()

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
