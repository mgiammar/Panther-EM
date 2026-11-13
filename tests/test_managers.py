"""managers and cli: configs, provenance, resuming, the Leopard-EM export, the CLI.

A module fixture runs the synthetic pipeline (``scripts/threshold_test_pipeline.py``)
in process; the tests reuse its files and configs.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import shutil
import threading
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
from PIL import Image
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
    read_mask,
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
    zscore_threshold,
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
        Searcher,
        "run",
        lambda self, m, **kw: searched.append(m.name) or run(self, m, **kw),
    )
    monkeypatch.setattr(  # the cached micrograph is reused, not prepared again
        Micrograph, "prepare", classmethod(lambda *a, **k: pytest.fail("prepared"))
    )
    done = search.run(config)
    assert searched == ["m1"]
    assert sorted(done) == ["m0", "m1", "m2"]
    # the summary breaks the micrograph's time down by pipeline step
    record = json.loads((tmp_path / "search_summary.json").read_text())["m1"]
    steps = (
        "read_s",
        "prepare_s",
        "search_s",
        "handoff_s",
        "save_result_s",
        "leopard_maps_s",
    )
    assert all(record[k] >= 0.0 for k in steps)
    assert record["elapsed_s"] == pytest.approx(sum(record[k] for k in steps))
    # without prefetch the search waits for all of its read
    assert record["read_wait_s"] >= record["read_s"]
    for stage in record["stages"]:
        parts = stage["featurize_s"] + stage["execute_s"] + stage["other_s"]
        assert parts == pytest.approx(stage["elapsed_s"])
    original = h5py.File(out / "m1_result.h5", "r")
    again = h5py.File(tmp_path / "m1_result.h5", "r")
    with original, again:
        np.testing.assert_array_equal(original["label"][()], again["label"][()])


def test_read_mask(tmp_path):
    mask = np.zeros((3, 4), dtype=np.float32)
    mask[1, 2] = 1.0
    mrcfile.write(str(tmp_path / "m.mrc"), mask[None])
    np.save(tmp_path / "m.npy", mask.astype(np.uint8))
    for name in ("m.mrc", "m.npy"):
        read = read_mask(tmp_path / name, (3, 4))
        assert read.dtype is torch.bool and read.device.type == "cpu"
        assert torch.nonzero(read).tolist() == [[1, 2]]
    # PNG: 1-bit, 8-bit gray, 16-bit gray, palette and RGB, plain or gzipped
    gray = (mask * 255).astype(np.uint8)
    pngs = {
        "1": Image.fromarray(gray).convert("1"),
        "L": Image.fromarray(gray),
        "I;16": Image.fromarray(mask.astype(np.uint16) * 1000),
        "P": Image.fromarray(gray).convert("P"),
        "RGB": Image.fromarray(gray).convert("RGB"),
    }
    for mode, image in pngs.items():
        assert image.mode == mode
        image.save(tmp_path / f"{mode}.png", optimize=True)
        with gzip.open(tmp_path / f"{mode}.png.gz", "wb") as f:
            image.save(f, format="PNG")
        for name in (f"{mode}.png", f"{mode}.png.gz"):
            read = read_mask(tmp_path / name, (3, 4))
            assert read.dtype is torch.bool, name
            assert torch.nonzero(read).tolist() == [[1, 2]], name
    with pytest.raises(ValueError, match="micrograph's"):
        read_mask(tmp_path / "L.png", (4, 3))
    with pytest.raises(ValueError, match="micrograph's"):
        read_mask(tmp_path / "m.npy", (4, 3))
    with pytest.raises(ValueError, match=r"\.png\.gz file"):
        read_mask(tmp_path / "m.txt", (3, 4))


def masked_search(pipeline, tmp_path):
    """The pipeline's search with m0 under a partial mask and m1 fully masked."""
    module, out = pipeline
    config = SearchConfig.from_yaml(out / "search.yaml")
    for name in ("m0", "m1", "m2"):
        shutil.copy(out / f"{name}_micrograph.h5", tmp_path / f"{name}_micrograph.h5")
    shape = (module.MICROGRAPH, module.MICROGRAPH)
    image_mask = np.zeros(shape, dtype=np.float32)
    image_mask[:60, :70] = 1.0  # keeps the planted particle's box center
    mrcfile.write(str(tmp_path / "m0_mask.mrc"), image_mask)
    np.save(tmp_path / "m1_mask.npy", np.zeros(shape, dtype=bool))
    entries = [
        entry.model_copy(update={"mask": mask})
        for entry, mask in zip(
            config.micrographs,
            [tmp_path / "m0_mask.mrc", tmp_path / "m1_mask.npy", None],
            strict=True,
        )
    ]
    config = config.model_copy(update={"output_dir": tmp_path, "micrographs": entries})
    return config, torch.from_numpy(image_mask != 0)


def test_masked_search(pipeline, tmp_path):
    """Only positions under a micrograph's mask are searched; the auto threshold
    counts only them; a fully masked micrograph is skipped."""
    module, _ = pipeline
    config, image_mask = masked_search(pipeline, tmp_path)
    done = search.run(config)
    assert sorted(done) == ["m0", "m2"]
    assert not (tmp_path / "m1_result.h5").exists()

    result = SearchResult.load(tmp_path / "m0_result.h5")
    grid = result.geometry.grid_mask(image_mask)
    assert torch.equal(result.label[0] == int(PixelLabel.MASKED), ~grid)
    assert (result.last_stage[0][~grid] == -1).all()
    assert (result.last_stage[0][grid] >= 0).all()

    summary = json.loads((tmp_path / "search_summary.json").read_text())
    n_searchable = int(grid.sum())
    n_hypotheses = 2 * 12 * module.N_PSI
    m0, m1, m2 = summary["m0"], summary["m1"], summary["m2"]
    assert m0["mask"] == str(tmp_path / "m0_mask.mrc")
    assert m0["n_searchable"] == n_searchable < grid.numel()
    assert m0["final"]["MASKED"] == grid.numel() - n_searchable
    assert m0["thresholds"] == pytest.approx(
        [zscore_threshold(n_searchable, n_hypotheses)] * 2
    )
    assert m2["mask"] is None and m2["n_searchable"] == grid.numel()
    assert m2["thresholds"] == pytest.approx(
        [zscore_threshold(grid.numel(), n_hypotheses)] * 2
    )
    assert m0["thresholds"][0] < m2["thresholds"][0]
    assert m1["n_searchable"] == 0 and "skipped" in m1

    # Leopard-EM: masked positions are padded, never picked, and not in the scale
    maps = leopard.match_template_maps(result)
    assert bool((maps["scaled_mip"][~grid] == 0).all())
    assert bool((maps["correlation_variance"][~grid] == LARGE_STD_PADDING).all())
    std = result.variance[0].double().clamp_min(0.0).sqrt()
    assert leopard.correlation_scale(result) == pytest.approx(float(std[grid].median()))
    pos_y, pos_x = leopard.pick_particles(result, radius=10.0)
    assert len(pos_y) > 0 and grid[pos_y, pos_x].all()
    assert (int(pos_y[0]), int(pos_x[0])) == module.PARTICLE["top_left"]


def test_masked_search_mask_overrides_the_cache(pipeline, tmp_path):
    """The entry's mask replaces a cached micrograph's, or drops it."""
    config, image_mask = masked_search(pipeline, tmp_path)
    m0, _, m2 = config.micrographs
    state = search._Run(config)
    # m2 cached with a mask, but its entry has none
    cached = Micrograph.load(tmp_path / "m2_micrograph.h5").with_mask(~image_mask)
    cached.save(tmp_path / "m2_micrograph.h5")
    assert state.micrograph(m2).mask is None
    # m0 cached without a mask, its entry has one
    assert torch.equal(state.micrograph(m0).mask, image_mask)
    # prepared afresh, the cache records the entry's mask
    fresh = config.model_copy(update={"overwrite": True})
    prepared = search._Run(fresh).micrograph(m0)
    assert torch.equal(prepared.mask, image_mask)
    assert torch.equal(Micrograph.load(tmp_path / "m0_micrograph.h5").mask, image_mask)


def test_search_write_error_surfaces_after_pending_writes(
    pipeline, tmp_path, monkeypatch
):
    """A failed background write raises from ``run`` once the other writes finish."""
    _, out = pipeline
    config = SearchConfig.from_yaml(out / "search.yaml")
    for name in ("m0", "m1", "m2"):
        shutil.copy(out / f"{name}_micrograph.h5", tmp_path / f"{name}_micrograph.h5")
    config = config.model_copy(update={"output_dir": tmp_path, "pending_writes": 1})
    write = leopard.write_match_template_maps

    def fails_on_m1(path, result, maps=None):
        if "m1" in Path(path).name:
            raise RuntimeError("disk full")
        write(path, result, maps)

    monkeypatch.setattr(leopard, "write_match_template_maps", fails_on_m1)
    with pytest.raises(RuntimeError, match="disk full"):
        search.run(config)
    assert sorted(p.name for p in tmp_path.glob("*_result.h5")) == [
        "m0_result.h5",
        "m2_result.h5",
    ]
    summary = json.loads((tmp_path / "search_summary.json").read_text())
    assert sorted(summary) == ["m0", "m2"]


@pytest.mark.parametrize("cached", [False, True], ids=["fresh", "cached"])
def test_search_prefetch_matches_serial(pipeline, tmp_path, cached):
    """Reading ahead changes no result, prepared afresh or from the cache."""
    runs = {}
    for prefetch in (0, 1, 2):
        out = tmp_path / str(prefetch)
        out.mkdir()
        config, _ = masked_search(pipeline, out)
        config = config.model_copy(
            update={"prefetch": prefetch, "overwrite": not cached}
        )
        done = search.run(config)
        assert sorted(done) == ["m0", "m2"]
        runs[prefetch] = (out, json.loads((out / "search_summary.json").read_text()))
    serial_dir, serial = runs[0]
    for prefetch in (1, 2):
        out, summary = runs[prefetch]
        assert sorted(summary) == sorted(serial) == ["m0", "m1", "m2"]
        for key in ("n_searchable", "skipped"):  # m1 is fully masked
            assert summary["m1"][key] == serial["m1"][key]
        for name in ("m0", "m2"):
            for key in ("n_searchable", "thresholds", "final"):
                assert summary[name][key] == serial[name][key], (prefetch, name, key)
            want = SearchResult.load(serial_dir / f"{name}_result.h5")
            got = SearchResult.load(out / f"{name}_result.h5")
            assert torch.equal(got.label, want.label)
            assert torch.equal(got.last_stage, want.last_stage)
            torch.testing.assert_close(
                got.mip, want.mip, rtol=0, atol=0, equal_nan=True
            )


def test_search_prefetch_reads_during_the_search(pipeline, tmp_path, monkeypatch):
    """With ``prefetch``, the next micrograph is read on the reader thread while the
    current one is searched."""
    _, out = pipeline
    config = SearchConfig.from_yaml(out / "search.yaml")
    for name in ("m0", "m1", "m2"):
        shutil.copy(out / f"{name}_micrograph.h5", tmp_path / f"{name}_micrograph.h5")
    config = config.model_copy(update={"output_dir": tmp_path, "prefetch": 1})

    events = {name: threading.Event() for name in ("m0", "m1", "m2")}
    threads = {}
    read = search._read

    def recorded_read(config, entry):
        loaded = read(config, entry)
        threads[entry.key] = threading.current_thread().name
        events[entry.key].set()
        return loaded

    order = ["m0", "m1", "m2"]
    run = Searcher.run

    def waits_for_next_read(self, micrograph, **kw):
        following = order[order.index(micrograph.name) + 1 :]
        if following:  # the next micrograph is read before this search ends
            assert events[following[0]].wait(timeout=30)
        return run(self, micrograph, **kw)

    monkeypatch.setattr(search, "_read", recorded_read)
    monkeypatch.setattr(Searcher, "run", waits_for_next_read)
    assert sorted(search.run(config)) == order
    assert sorted(threads) == order
    assert all(name.startswith("reader") for name in threads.values())


def test_search_prefetch_read_error_surfaces_in_turn(pipeline, tmp_path):
    """A micrograph that fails to read raises when its turn comes: the micrographs
    before it are searched and written, the ones after it are not."""
    _, out = pipeline
    config = SearchConfig.from_yaml(out / "search.yaml")
    m0, m1, m2 = config.micrographs
    entries = [m0, m1.model_copy(update={"path": tmp_path / "missing.mrc"}), m2]
    config = config.model_copy(
        update={
            "output_dir": tmp_path,
            "micrographs": entries,
            "prefetch": 2,
            "cache_micrographs": False,
        }
    )
    with pytest.raises(FileNotFoundError, match="missing"):
        search.run(config)
    assert [p.name for p in tmp_path.glob("*_result.h5")] == ["m0_result.h5"]
    assert not list(tmp_path.glob("*_micrograph.h5"))  # caching is opt-in
    summary = json.loads((tmp_path / "search_summary.json").read_text())
    assert sorted(summary) == ["m0"]


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
