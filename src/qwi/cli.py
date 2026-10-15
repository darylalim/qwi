"""Command-line entry point. Commands land milestone by milestone (PLAN.md §5 CLI reference)."""

import logging
from typing import Annotated

import typer

from qwi import logs, storage
from qwi.errors import QwiError

app = typer.Typer(no_args_is_help=True, help="Local text-to-image pipeline for Apple Silicon.")
models_app = typer.Typer(no_args_is_help=True, help="Model weights (the only download path).")
app.add_typer(models_app, name="models")

MARK = {"pass": "ok  ", "warn": "WARN", "fail": "FAIL"}


@app.callback()
def main() -> None:
    """qwi: guard → rewrite → generate N seeds → rerank."""


def _fail(exc: QwiError) -> typer.Exit:
    logs.event(logging.ERROR, None, "cli_error", str(exc), type=type(exc).__name__)
    typer.echo(str(exc), err=True)
    return typer.Exit(exc.exit_code)


@app.command()
def doctor() -> None:
    """Read-only health check: disk, models, $QWI_HOME, runs, offline mode, torch (§5.7)."""
    logs.app_log()
    checks = storage.doctor()
    width = max(len(c.name) for c in checks)
    for check in checks:
        typer.echo(f"{MARK[check.status]}  {check.name:<{width}}  {check.detail}")
    problems = {c.name: c.status for c in checks if c.status != "pass"}
    logs.event(logging.INFO, None, "doctor", "checked", problems=problems)
    if any(c.status == "fail" for c in checks):
        raise typer.Exit(1)


@models_app.command("fetch")
def models_fetch(
    role: Annotated[str, typer.Argument(help="A role from models.toml, e.g. image.")],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Check size and free disk only.")
    ] = False,
) -> None:
    """Download a role's weights at the pinned revision, if ≥ 20 GB stays free (§5.7)."""
    logs.app_log()
    try:
        path = storage.fetch(role, dry_run=dry_run)
    except QwiError as exc:
        raise _fail(exc) from exc
    typer.echo(
        f"{role}: enough disk; nothing downloaded (--dry-run)" if path is None else str(path)
    )
