"""Format tags of the HDF5 files Panther-EM writes.

Every file carries the attributes ``panther_em_format`` (what it holds) and
``format_version`` (the layout version of that kind). A loader rejects a file of
another kind or of a newer version than it reads; files written before the tags
existed carry neither and load as version 0.

:func:`create_dataset` writes the result maps: integer and boolean arrays (labels,
indices) are gzip-compressed, float arrays (correlations, angles) are not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    import h5py

FORMAT_VERSIONS = {
    "decomposition": 1,
    "whitening_filter": 1,
    "error_table": 1,
    "micrograph": 1,
    "search_result": 1,
}
"""The layout version each kind of file is written with (and the newest read)."""

INTEGER_COMPRESSION = {"compression": "gzip", "compression_opts": 1, "shuffle": True}


def create_dataset(
    group: h5py.Group, name: str, data: np.ndarray | np.generic
) -> h5py.Dataset:
    """Write ``data`` as ``group[name]``, compressing integer and boolean arrays."""
    data = np.asarray(data)
    compress = data.dtype.kind in "biu" and data.ndim > 0 and data.size > 0
    return group.create_dataset(
        name, data=data, **(INTEGER_COMPRESSION if compress else {})
    )


def write_format(f: h5py.File, kind: str) -> None:
    """Tag an open file as ``kind`` at its current layout version."""
    f.attrs["panther_em_format"] = kind
    f.attrs["format_version"] = FORMAT_VERSIONS[kind]


def check_format(f: h5py.File, kind: str, path: str | Path) -> int:
    """The layout version of an open ``kind`` file; raise for another kind or newer.

    Untagged files (written before the tags existed) are accepted as version 0.
    """
    found = f.attrs.get("panther_em_format")
    if found is None:
        return 0
    if isinstance(found, bytes):
        found = found.decode("utf-8")
    if found != kind:
        raise ValueError(f"'{path}' holds a Panther-EM {found}, not a {kind}")
    version = int(f.attrs.get("format_version", 0))
    newest = FORMAT_VERSIONS[kind]
    if version > newest:
        raise ValueError(
            f"'{path}' is a {kind} file of format version {version}; this Panther-EM "
            f"reads up to version {newest} (upgrade Panther-EM to read it)"
        )
    return version
