"""Program ``whitening``: the dataset's whitening filter from its micrographs."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import Field, field_validator

from panther_em.filters import WhiteningFilter
from panther_em.managers._common import ProgramConfig, read_micrograph, save_output


class WhiteningConfig(ProgramConfig):
    """Configuration of :func:`run`.

    Parameters
    ----------
    micrographs : list of Path
        Pre-processed (motion-corrected) MRC micrographs; each is named by its file
        stem, the name the search looks its correction ``C_i`` up by.
    output : Path
        The whitening filter's HDF5 file.
    band_cutoff, band_falloff : float
        The template band's cutoff and falloff, cycles per pixel.
    max_freq : float
        Highest frequency of the spectra, cycles per pixel. Default 0.5.
    device : str
        Where the spectra are computed. Default ``"cpu"``.
    """

    program: ClassVar[str] = "whitening"

    micrographs: list[Path] = Field(min_length=1)
    output: Path
    band_cutoff: float
    band_falloff: float = 0.0
    max_freq: float = 0.5
    device: str = "cpu"

    @field_validator("micrographs")
    @classmethod
    def _unique_names(cls, paths: list[Path]) -> list[Path]:
        stems = [p.stem for p in paths]
        if len(set(stems)) != len(stems):
            raise ValueError("micrograph file names (stems) must be unique")
        return paths


def run(config: WhiteningConfig) -> WhiteningFilter:
    """Estimate the whitening filter and write it to ``config.output``."""
    whitening = WhiteningFilter.from_images(
        ((p.stem, read_micrograph(p, config.device)) for p in config.micrographs),
        band_cutoff=config.band_cutoff,
        band_falloff=config.band_falloff,
        max_freq=config.max_freq,
    )
    save_output(whitening.save, config.output, config)
    return whitening
