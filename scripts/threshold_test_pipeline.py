"""The coloured-noise threshold test end to end on synthetic data.

The steps an application drives around a Panther-EM search, on a random smooth
volume and three synthetic micrographs (one carries a particle):

1. select the micrographs: ``(name, image)`` pairs of pre-processed real-space images;
2. ``WhiteningFilter.from_images``: the averaged whitening ``W_avg``, every
   micrograph's correction ``C_i`` and the template band;
3. decompose the volume under ``whitening.template_filter`` (``W_avg * band``);
4. per feature selection ``compute_error_tables``;
5. per micrograph ``ImageSet.from_micrograph`` (search image ``W_avg C_i^2``, noise and
   signal curves, level map), the two-stage search and ``SearchResult.save``:

   * stage 0, the coarse features over every orientation on one CTF plane, rejects
     or abstains;
   * stage 1, the full features, is *local*: each pixel is scored over the
     orientations within one HEALPix ring of its stage-0 orientation, on both
     planes, with the mean and variance of the whole set from the closed-form
     moments; it rejects, abstains or accepts.

The template's orientations are the nside-1 HEALPix pixel centers (12).

Every file is written to ``--out`` and read back. CPU, well under a minute.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch_ctf import calculate_ctf_2d
from torch_grid_utils.fftfreq_grid import fftfreq_grid

from panther_em.coordinates import OffsetPolarTransform
from panther_em.decomposition import PolarProjectionDecomposer
from panther_em.decomposition.pipeline_projections import (
    generate_projection_batch,
    precompute_volume_dft,
)
from panther_em.decomposition.result import DecompositionResult
from panther_em.inference.search import (
    Calibration,
    Decision,
    ImageSet,
    PerSpatialFrequencyErrorTable,
    PixelLabel,
    Searcher,
    SearchResult,
    Stage,
    ThresholdTest,
    WhiteningFilter,
    compute_error_tables,
    healpix_orientations,
    zscore_threshold,
)
from panther_em.inference.template import Template

if TYPE_CHECKING:
    from collections.abc import Iterator

BOX = 32
MICROGRAPH = 96
NSIDE = 1
N_ORIENTATIONS = 12 * NSIDE**2
LOCAL_RINGS = 1
N_ANGLE, N_RADIUS = 32, 16
EIG_MAX = 8
N_PSI = 64
COARSE = (0, 0, 6, 4)
FULL = (0, 0, N_ANGLE // 2 + 1, EIG_MAX)
BAND = (0.35, 0.05)  # cycles / px: the template band's cutoff and falloff
PIXEL_SIZE = 1.5  # A
DEFOCUS_UM = (1.0, 2.0)  # the two CTF planes
PARTICLE = {"micrograph": "m0", "orientation": 3, "plane": 1, "top_left": (20, 44)}
NAMES = ("m0", "m1", "m2")


def synthetic_volume(seed: int = 0) -> np.ndarray:
    """A random smooth ``BOX^3`` volume."""
    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.standard_normal((BOX, BOX, BOX)), 2.0).astype(np.float32)


def orientations() -> tuple[np.ndarray, np.ndarray]:
    """``(phi, theta)`` in degrees of the template's orientations (HEALPix, NESTED)."""
    return healpix_orientations(NSIDE)


def ctf_planes(shape: tuple[int, int]) -> torch.Tensor:
    """``(2, H, W // 2 + 1)`` CTF filters of the two defocus planes."""
    return calculate_ctf_2d(
        defocus=torch.tensor(DEFOCUS_UM),
        astigmatism=0.0,
        astigmatism_angle=0.0,
        voltage=300.0,
        spherical_aberration=2.7,
        amplitude_contrast=0.07,
        phase_shift=0.0,
        pixel_size=PIXEL_SIZE,
        image_shape=shape,
        rfft=True,
        fftshift=False,
    ).to(torch.float32)


def coloured_noise(shape: tuple[int, int], scale: float, seed: int) -> torch.Tensor:
    """Noise whose amplitude falls as ``1 / (1 + 8 f)``, at ``scale`` per pixel."""
    gen = torch.Generator().manual_seed(seed)
    white = torch.randn(shape, generator=gen)
    f = fftfreq_grid(image_shape=shape, rfft=True, fftshift=False, norm=True)
    x = torch.fft.irfft2(torch.fft.rfft2(white) / (1.0 + 8.0 * f), s=shape)
    return scale * x / x.std()


def particle_image(volume: np.ndarray, shape: tuple[int, int]) -> torch.Tensor:
    """The particle's projection, CTF-filtered on its plane, placed in a blank image."""
    dft, edge_value_scaled, pad = precompute_volume_dft(torch.as_tensor(volume))
    phi, theta = orientations()
    o = PARTICLE["orientation"]
    angle = lambda a: torch.tensor([float(a[o])])  # noqa: E731
    proj = generate_projection_batch(
        dft, edge_value_scaled, pad, angle(phi), angle(theta), torch.zeros(1)
    )[0].to(torch.float32)
    proj = (proj - proj.mean()) / proj.std()
    canvas = torch.zeros(shape)
    r, c = PARTICLE["top_left"]
    canvas[r : r + BOX, c : c + BOX] = proj
    ctf = ctf_planes(shape)[PARTICLE["plane"]]
    return torch.fft.irfft2(torch.fft.rfft2(canvas) * ctf, s=shape)


def raw_micrographs(volume: np.ndarray) -> Iterator[tuple[str, torch.Tensor]]:
    """Step 1: the pre-processed real-space micrographs, one at a time."""
    shape = (MICROGRAPH, MICROGRAPH)
    for i, (name, scale) in enumerate(zip(NAMES, (1.0, 0.7, 1.4), strict=True)):
        image = coloured_noise(shape, scale, seed=10 + i)
        if name == PARTICLE["micrograph"]:
            image = image + 4.0 * particle_image(volume, shape)
        yield name, image


def decompose(volume: np.ndarray, whitening: WhiteningFilter) -> DecompositionResult:
    """Step 3: the decomposition under ``W_avg * band``."""
    phi, theta = orientations()
    transform = OffsetPolarTransform.from_image(
        image_shape=(BOX, BOX), num_angle=N_ANGLE, num_radius=N_RADIUS
    )
    filters = whitening.template_filter((BOX, BOX))[None]
    return PolarProjectionDecomposer(
        volume, phi, theta, coordinate_transform=transform, fourier_filters=filters
    ).do_decomposition(eig_max=EIG_MAX, normalize_projections=True, svd_method="dense")


def label_counts(labels: torch.Tensor) -> dict[str, int]:
    """Pixels per label name."""
    return {lab.name: int((labels == int(lab)).sum()) for lab in PixelLabel}


def run(out: Path, device: str = "cpu") -> dict:
    """Run the pipeline into ``out``; returns a summary (also written as JSON)."""
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    volume = synthetic_volume()

    # steps 1 + 2: the whitening filter streams the micrographs
    whitening = WhiteningFilter.from_images(
        raw_micrographs(volume), band_cutoff=BAND[0], band_falloff=BAND[1]
    )
    whitening.save(out / "whitening.h5")
    whitening = WhiteningFilter.load(out / "whitening.h5")

    # step 3
    decompose(volume, whitening).save(out / "decomposition.h5")
    template = Template(
        DecompositionResult.load(out / "decomposition.h5"), device=device
    )

    # step 4: one table per feature selection
    tables = {}
    for tag, rect in (("coarse", COARSE), ("full", FULL)):
        compute_error_tables(volume, template, [rect], whitening).save(
            out / f"tables_{tag}.h5"
        )
        tables[tag] = PerSpatialFrequencyErrorTable.load(out / f"tables_{tag}.h5")

    calibration = Calibration(alpha=5e-3, zeta=0.9, eps_rho=0.1, b_factor=20.0)
    n_hypotheses = len(DEFOCUS_UM) * N_ORIENTATIONS * N_PSI
    grid = MICROGRAPH - BOX + 1
    threshold = zscore_threshold(grid * grid, n_hypotheses)
    tests = {tag: ThresholdTest(table, calibration) for tag, table in tables.items()}
    stages = [
        Stage.create(
            [COARSE],
            n_psi=N_PSI,
            planes=(1,),
            decision=Decision(threshold, tests["coarse"], accept=False),
            name="coarse",
        ),
        Stage.create(
            [FULL],
            n_psi=N_PSI,
            local_rings=LOCAL_RINGS,
            decision=Decision(threshold, tests["full"], absorb_radius=2),
            name="local",
        ),
    ]
    searcher = Searcher(
        template, stages, n_defocus=len(DEFOCUS_UM), keep_stage_results=True
    )
    ctf = ctf_planes((MICROGRAPH, MICROGRAPH))

    # step 5, per micrograph
    summary: dict = {
        "threshold": threshold,
        "z_alpha": calibration.z_alpha,
        "micrographs": {},
    }
    if 1 in searcher.local:  # orientations per neighbourhood, of N_ORIENTATIONS
        counts = searcher.local[1].counts
        summary["local_orientations"] = [
            int(counts[counts > 0].min()),
            int(counts.max()),
        ]
    for name, raw in raw_micrographs(volume):
        images = ImageSet.from_micrograph(
            raw.to(device),
            name,
            whitening,
            ctf,
            tables["coarse"],
            pixel_size=PIXEL_SIZE,
        )
        scales = tests["coarse"].scales(images)
        search_result = searcher.run(images)
        search_result.save(out / f"{name}_result.h5")
        loaded = SearchResult.load(out / f"{name}_result.h5")
        tested = loaded.last_stage >= 0
        record = {
            "stages": [
                {
                    "name": s.stage.name,
                    "n_pixels": s.n_pixels,
                    **label_counts(s.labels_after),
                }
                for s in loaded.stage_results
            ],
            "final": label_counts(loaded.label),
            "score_range": [
                float(loaded.score[tested].min()),
                float(loaded.score[tested].max()),
            ],
            "zscore_max": float(loaded.zscore[tested].max()),
            "rho_range": [float(scales.rho.min()), float(scales.rho.max())],
            "level_range": [float(images.level.min()), float(images.level.max())],
            "noise_shape": list(images.noise.shape),
        }
        if name == PARTICLE["micrograph"]:
            r, c = PARTICLE["top_left"]
            record["particle"] = {
                "label": PixelLabel(int(loaded.label[0, r, c])).name,
                "zscore": float(loaded.zscore[0, r, c]),
                "orientation": int(loaded.decode()["orientation_index"][0, r, c]),
                "defocus": int(loaded.decode()["defocus_index"][0, r, c]),
            }
        summary["micrographs"][name] = record
    summary["elapsed_s"] = time.perf_counter() - t0
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    """Run into ``--out`` and print the per-stage label counts."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    summary = run(args.out, args.device)
    print(
        f"threshold T = {summary['threshold']:.3f}, z_alpha = {summary['z_alpha']:.3f}"
    )
    for name, rec in summary["micrographs"].items():
        for s in rec["stages"]:
            print(
                f"{name} {s['name']:6s}: {s['n_pixels']:5d} px -> "
                f"{s['REJECTED']:5d} rejected, {s['AMBIGUOUS']:5d} ambiguous, "
                f"{s['ACCEPTED']:4d} accepted, {s['ABSORBED']:4d} absorbed"
            )
        lo, hi = rec["score_range"]
        print(f"{name}: score {lo:.2f} .. {hi:.2f}, max z {rec['zscore_max']:.2f}")
        if "particle" in rec:
            print(f"{name}: particle pixel {rec['particle']}")
    print(f"{summary['elapsed_s']:.1f} s; files in {args.out}")


if __name__ == "__main__":
    main()
