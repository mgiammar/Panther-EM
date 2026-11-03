"""The synthetic end-to-end pipeline (whitening, decomposition, tables, spectra,
two-stage search with the threshold test) runs and every file it writes reloads."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from panther_em.inference.search import (
    ImageSpectra,
    NoiseScales,
    PixelLabel,
    RadialTables,
    SearchResult,
)
from panther_em.spectra import WhiteningFilter

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "threshold_test_pipeline.py"


def test_threshold_test_pipeline_runs_end_to_end(tmp_path):
    proc = subprocess.run(
        [sys.executable, "-W", "ignore", str(SCRIPT), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["threshold"] > 4.0 and summary["z_alpha"] > 2.0

    whitening = WhiteningFilter.load(tmp_path / "whitening.h5")
    assert whitening.names == ("m0", "m1", "m2")
    tables = {
        tag: RadialTables.load(tmp_path / f"tables_{tag}.h5")
        for tag in ("coarse", "full")
    }
    assert tables["coarse"].signal_source == "projection"
    assert tables["coarse"].rectangles == ((0, 0, 6, 4),)
    assert tables["full"].n_templates == 12

    for name in whitening.names:
        spectra = ImageSpectra.load(tmp_path / f"{name}_spectra.h5")
        scales = NoiseScales.load(tmp_path / f"{name}_scales_coarse.h5")
        assert spectra.noise.shape == (1, 2, spectra.n_bins)  # two CTF planes
        assert spectra.level.shape == (1, 96, 96)
        assert scales.q_kept.shape == (1, 2, 12)
        assert scales.rectangles == ((0, 0, 6, 4),)
        result = SearchResult.load(tmp_path / f"{name}_result.h5")
        assert result.spectra is not None
        torch.testing.assert_close(result.spectra.level, spectra.level)
        assert len(result.stage_results) == 2
        searched = result.last_stage >= 0
        assert searched.all()  # every pixel was decided by one of the stages
        assert torch.isfinite(result.score).all()
        record = summary["micrographs"][name]
        coarse, full = record["stages"]
        assert coarse["ACCEPTED"] == 0 and coarse["REJECTED"] > 0  # reject or abstain
        assert full["n_pixels"] == coarse["AMBIGUOUS"]
        assert not (result.label == int(PixelLabel.AMBIGUOUS)).all()
        first = result.stage_results[0]
        assert torch.isfinite(first.score).all() and first.n_pixels == 65 * 65

    # the injected particle survives both stages and is accepted at its pose
    particle = summary["micrographs"]["m0"]["particle"]
    assert particle["label"] == "ACCEPTED"
    assert particle["orientation"] == 3 and particle["defocus"] == 1
    assert particle["zscore"] > summary["threshold"]
