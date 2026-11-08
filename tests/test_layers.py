"""The package's layering: every module imports only from its own layer or below.

Bottom to top: ``_hdf5`` -> ``coordinates`` -> ``orientations`` -> ``filters`` ->
``decomposition`` -> ``approximation`` -> ``search`` -> ``managers`` -> ``cli``.
Imports under ``TYPE_CHECKING`` and inside functions count too.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import panther_em

LAYERS = (
    "_hdf5",
    "coordinates",
    "orientations",
    "filters",
    "decomposition",
    "approximation",
    "search",
    "managers",
    "cli",
)
ROOT = Path(panther_em.__file__).parent


def _modules() -> list[Path]:
    return sorted(p for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _layer(dotted: str) -> str | None:
    """The layer of ``panther_em.<layer>...`` (``None``: the package root itself)."""
    parts = dotted.split(".")
    return parts[1] if len(parts) > 1 and parts[0] == "panther_em" else None


def _package_of(path: Path) -> list[str]:
    rel = path.relative_to(ROOT.parent).with_suffix("")
    parts = list(rel.parts)
    return parts if path.name == "__init__.py" else parts[:-1]


def _imports(path: Path) -> list[str]:
    """Absolute dotted names of every ``panther_em`` import in ``path``."""
    tree = ast.parse(path.read_text(), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = _package_of(path)
                base = package[: len(package) - node.level + 1]
                module = ".".join([*base, node.module] if node.module else base)
            else:
                module = node.module or ""
            names.append(module)
    return [n for n in names if n == "panther_em" or n.startswith("panther_em.")]


def test_every_top_level_module_has_a_layer():
    tops = {
        p.relative_to(ROOT).parts[0].removesuffix(".py")
        for p in _modules()
        if p != ROOT / "__init__.py"
    }
    assert tops == set(LAYERS)


@pytest.mark.parametrize(
    "path", _modules(), ids=lambda p: str(p.relative_to(ROOT.parent))
)
def test_imports_point_down(path):
    if path == ROOT / "__init__.py":  # the package root may import anything
        return
    own = _layer(".".join([*_package_of(path), path.stem]))
    assert own is not None
    rank = LAYERS.index(own)
    for name in _imports(path):
        target = _layer(name)
        assert target is not None, f"{path.name} imports the package root ({name})"
        assert LAYERS.index(target) <= rank, f"{own} imports {name} from above"
