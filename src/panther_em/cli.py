"""The ``panther-em`` command: run one pipeline program on a YAML config.

::

    panther-em whitening whitening.yaml
    panther-em decompose decompose.yaml
    panther-em calibrate calibrate.yaml
    panther-em search search.yaml
    panther-em export export.yaml

``panther-em <program> --example`` prints a config with every field (defaults
filled, required ones marked) to start from.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import yaml
from pydantic import ValidationError

from panther_em.managers import (
    calibrate,
    decompose,
    export,
    search,
    whitening,
)

if TYPE_CHECKING:
    from types import ModuleType

    from panther_em.managers._common import ProgramConfig

PROGRAMS: dict[str, tuple[ModuleType, type[ProgramConfig]]] = {
    "whitening": (whitening, whitening.WhiteningConfig),
    "decompose": (decompose, decompose.DecomposeConfig),
    "calibrate": (calibrate, calibrate.CalibrateConfig),
    "search": (search, search.SearchConfig),
    "export": (export, export.ExportConfig),
}


def example(config: type[ProgramConfig]) -> dict[str, Any]:
    """Every top-level field of ``config``: its default, or ``"<required>"``."""
    out: dict[str, Any] = {}
    for name, field in config.model_fields.items():
        if field.is_required():
            out[name] = "<required>"
        else:
            default = field.get_default(call_default_factory=True)
            dump = getattr(default, "model_dump", None)
            out[name] = dump(mode="json") if dump else default
    return out


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="panther-em")
def main() -> None:
    """Panther-EM: SVD-accelerated 2D template matching programs."""


def _command(name: str) -> click.Command:
    module, config_cls = PROGRAMS[name]

    @click.command(name=name, help=(module.__doc__ or "").split("\n\n")[0])
    @click.argument(
        "config",
        required=False,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
    )
    @click.option(
        "--example", "show_example", is_flag=True, help="Print an example config."
    )
    def command(config: Path | None, show_example: bool) -> None:
        if show_example:
            click.echo(yaml.safe_dump(example(config_cls), sort_keys=False), nl=False)
            return
        if config is None:
            raise click.UsageError("give a CONFIG file (or --example)")
        try:
            parsed = config_cls.from_yaml(config)
        except ValidationError as err:
            raise click.ClickException(f"{config}:\n{err}") from None
        module.run(parsed)

    return command


for _name in PROGRAMS:
    main.add_command(_command(_name))
