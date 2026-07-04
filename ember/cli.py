from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.logging import RichHandler

from ember import __version__
from ember.utils import imports

if TYPE_CHECKING:
    from ember.run import EmberRunner

app = typer.Typer(
    help="A minimalist Fabric-based framework for machine learning",
    add_completion=False,
    rich_markup_mode="rich",
)

ScriptPath = Annotated[
    Path,
    typer.Argument(
        help="Path to the script containing the runner",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
]

ConfigPath = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="Path to the configuration file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
]

Verbosity = Annotated[
    int,
    typer.Option(
        "--verbose",
        "-v",
        count=True,
        help="Global verbosity level.",
    ),
]


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"ember version: {__version__}")
        raise typer.Exit()


VersionFlag = Annotated[
    bool | None,
    typer.Option(
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
]


@app.command()
def main(
    script_path: ScriptPath,
    cfg_path: ConfigPath = None,
    verbosity: Verbosity = 0,
    version: VersionFlag = None,
) -> None:
    logging.basicConfig(
        level=[logging.WARNING, logging.INFO, logging.DEBUG][min(verbosity, 2)],
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
        format="%(message)s",
        datefmt="[%X]",
    )

    try:
        module = imports.load_script_module(script_path)
    except Exception as exc:
        typer.echo(f"[ember] failed to load script {script_path}: {exc}", err=True)
        raise typer.Exit(1) from exc

    script_dir = script_path.parent
    project_root = imports.find_project_root(script_dir)
    try:
        runner: EmberRunner = imports.detect_runner_instance(
            module,
            script_dir,
            cfg_path,
            verbosity,
            project_root=project_root,
        ) or imports.discover_runner_subclass(
            module,
            script_dir,
            cfg_path,
            verbosity,
            project_root=project_root,
        )
    except (TypeError, RuntimeError) as exc:
        typer.echo(f"[ember] runner discovery failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    if not runner:
        typer.echo(
            "[ember] runner discovery returned no runner; ensure your script "
            "exports a runner or a single EmberRunner subclass",
            err=True,
        )
        raise typer.Exit(1)

    runner.run()
