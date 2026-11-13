"""Shared pieces of the programs: the config base, provenance, atomic saves, MRC I/O."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Self, TypeVar

import h5py
import mrcfile
import numpy as np
import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Callable

Rectangle = tuple[
    Annotated[int, Field(ge=0)],
    Annotated[int, Field(ge=0)],
    Annotated[int, Field(ge=1)],
    Annotated[int, Field(ge=1)],
]
"""A ``(k_start, m_start, k_extent, m_extent)`` feature rectangle."""


class Model(BaseModel):
    """Frozen pydantic model that rejects unknown keys (a typo is an error)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProgramConfig(Model):
    """The configuration of one program: pure data, read from YAML.

    Relative paths are taken relative to the working directory.
    """

    program: ClassVar[str]

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Validate the YAML file at ``path``."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path | None = None) -> str:
        """The config as YAML text, also written to ``path`` when given."""
        text = yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)
        if path is not None:
            Path(path).write_text(text)
        return text


# --------------------------------------------------------------------------- #
# Provenance: every output records the config that wrote it
# --------------------------------------------------------------------------- #

PROGRAM_ATTR = "panther_em_program"
CONFIG_ATTR = "panther_em_config"

C = TypeVar("C", bound=ProgramConfig)


def save_output(
    save: Callable[[Path], object],
    path: Path,
    config: ProgramConfig,
    **attrs: str,
) -> Path:
    """Write ``path`` with ``save`` and record ``config`` (and ``attrs``) in it.

    The file is written under a temporary name and moved into place once complete,
    so an interrupted program never leaves a file that looks finished.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.stem}.partial{path.suffix}")
    save(partial)
    with h5py.File(partial, "a") as f:
        f.attrs[PROGRAM_ATTR] = config.program
        f.attrs[CONFIG_ATTR] = config.model_dump_json()
        for key, value in attrs.items():
            f.attrs[f"panther_em_{key}"] = value
    partial.replace(path)
    return path


def read_attr(path: str | Path, key: str) -> str | None:
    """A ``panther_em_<key>`` attribute recorded by :func:`save_output`, if any."""
    with h5py.File(path, "r") as f:
        value = f.attrs.get(f"panther_em_{key}")
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return None if value is None else str(value)


def read_config(path: str | Path, cls: type[C]) -> C | None:
    """The ``cls`` config recorded in ``path``; ``None`` if another program wrote it."""
    if read_attr(path, "program") != cls.program:
        return None
    text = read_attr(path, "config")
    return None if text is None else cls.model_validate_json(text)


# --------------------------------------------------------------------------- #
# MRC input
# --------------------------------------------------------------------------- #


def read_micrograph(
    path: str | Path, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """A 2-D MRC image as float32 on ``device``, its mean subtracted."""
    data = np.asarray(mrcfile.read(str(path)), dtype=np.float32)
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(f"'{path}' is not a 2-D image: shape {data.shape}")
    image = torch.from_numpy(data.copy())
    return (image - image.mean()).to(device)


def read_volume(path: str | Path) -> np.ndarray:
    """A cubic 3-D MRC volume as float32."""
    data = np.asarray(mrcfile.read(str(path)), dtype=np.float32)
    if data.ndim != 3 or len(set(data.shape)) != 1:
        raise ValueError(f"'{path}' is not a cubic volume: shape {data.shape}")
    return data
