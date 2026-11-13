"""The synthetic end-to-end pipeline runs and every file it writes reloads.

The five programs (whitening, decomposition, error tables, the two-stage search -- a
full stage 0, a local stage 1 -- with the threshold test, and the export) on MRC
inputs, as ``panther-em`` runs them from YAML.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from panther_em.approximation import PerSpatialFrequencyErrorTable
from panther_em.decomposition.result import DecompositionResult
from panther_em.filters import WhiteningFilter
from panther_em.search import PixelLabel, SearchResult
from panther_em.search.template import Template

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
    assert summary["particles_exported"] >= 1  # particles.h5: one row per particle

    whitening = WhiteningFilter.load(tmp_path / "whitening.h5")
    assert whitening.names == ("m0", "m1", "m2")
    template = Template(
        DecompositionResult.load(tmp_path / "decomposition.h5"), device="cpu"
    )
    whitening.check_template_filter(template.result.fourier_filters)
    tables = {
        tag: PerSpatialFrequencyErrorTable.load(tmp_path / f"tables_{tag}.h5")
        for tag in ("coarse", "full")
    }
    assert tables["coarse"].rectangles == ((0, 0, 6, 4),)
    assert tables["full"].n_templates == 12  # the nside-1 HEALPix grid
    # stage 1 scores each pixel over a 1-ring neighborhood, not all 12 orientations
    assert 1 < summary["local_orientations"][1] < 12
    for table in tables.values():
        table.check(template.reconstructor)

    for name in whitening.names:
        record = summary["micrographs"][name]
        assert record["noise_shape"] == [2, tables["coarse"].n_bins]  # 2 CTF planes
        lo, hi = record["level_range"]
        assert 0.0 < lo < 1.0 < hi
        result = SearchResult.load(tmp_path / f"{name}_result.h5")
        assert len(result.stage_results) == 2
        assert (result.last_stage >= 0).all()  # every pixel decided by a stage
        assert torch.isfinite(result.score).all()
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
    # the empty micrographs accept nothing; a fixed-seed statistical outcome, see the
    # note at NOISE_SEED in the script before changing either
    for name in ("m1", "m2"):
        assert summary["micrographs"][name]["final"]["ACCEPTED"] == 0
