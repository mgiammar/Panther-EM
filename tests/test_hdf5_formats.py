"""Format tags of the HDF5 files: kind, version, and untagged (older) files."""

from __future__ import annotations

import h5py
import pytest

from panther_em._hdf5 import FORMAT_VERSIONS, check_format, write_format
from panther_em.approximation import PerSpatialFrequencyErrorTable
from panther_em.decomposition import DecompositionBase, DecompositionResult
from panther_em.filters import WhiteningFilter
from panther_em.search import Micrograph, SearchResult


def test_every_kind_has_a_version():
    assert set(FORMAT_VERSIONS) == {
        "decomposition",
        "whitening_filter",
        "error_table",
        "micrograph",
        "search_result",
    }


def test_check_format_accepts_its_kind_and_untagged_files(tmp_path):
    path = tmp_path / "x.h5"
    with h5py.File(path, "w") as f:
        assert check_format(f, "micrograph", path) == 0  # untagged
        write_format(f, "micrograph")
        assert check_format(f, "micrograph", path) == FORMAT_VERSIONS["micrograph"]
        with pytest.raises(ValueError, match="holds a Panther-EM micrograph"):
            check_format(f, "error_table", path)
        f.attrs["format_version"] = FORMAT_VERSIONS["micrograph"] + 1
        with pytest.raises(ValueError, match="upgrade Panther-EM"):
            check_format(f, "micrograph", path)


def test_loaders_check_the_tag(synthetic_result, tmp_path):
    path = tmp_path / "decomposition.h5"
    synthetic_result().save(path)
    with h5py.File(path, "r") as f:
        assert f.attrs["panther_em_format"] == "decomposition"
    assert isinstance(DecompositionBase.load(path), DecompositionResult)

    for loader in (
        WhiteningFilter.load,
        PerSpatialFrequencyErrorTable.load,
        Micrograph.load,
        SearchResult.load,
    ):
        with pytest.raises(ValueError, match="holds a Panther-EM decomposition"):
            loader(path)

    with h5py.File(path, "r+") as f:
        f.attrs["format_version"] = FORMAT_VERSIONS["decomposition"] + 1
    with pytest.raises(ValueError, match="format version"):
        DecompositionBase.load(path)

    with h5py.File(path, "r+") as f:  # written before the tags existed
        del f.attrs["panther_em_format"], f.attrs["format_version"]
    assert isinstance(DecompositionBase.load(path), DecompositionResult)
