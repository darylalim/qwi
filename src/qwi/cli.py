"""Command-line entry point. Commands land milestone by milestone (PLAN.md §5 CLI reference)."""

import logging
from typing import Annotated

import typer

from qwi import errors, logs, storage
from qwi.config import DEFAULTS
from qwi.errors import PipelineBusy, QwiError
from qwi.events import CandidateReady, DenoiseStep, Event, RunStarted, StageFinished, StageStarted
from qwi.pipeline import GenerateRequest, Pipeline

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


def _render(event: Event) -> None:
    match event:
        case RunStarted(run_dir, seeds):
            typer.echo(f"run {run_dir.name} (seeds {seeds[0]}…{seeds[-1]})")
        case StageStarted(stage):
            typer.echo(f"{stage}: loading")
        case DenoiseStep(candidate, step, total):
            typer.echo(f"\r  candidate {candidate + 1}: step {step}/{total}", nl=False)
        case CandidateReady(index, seed, path):
            typer.echo(f"\r  candidate {index + 1} (seed {seed}): {path}")
        case StageFinished(stage, seconds, peak_gb):
            typer.echo(f"{stage}: {seconds:.1f} s, peak {peak_gb:.1f} GB")


@app.command()
def generate(
    prompt: Annotated[str, typer.Argument(help="What to draw.")],
    n: Annotated[int, typer.Option("-n", min=1, help="Candidates (seeds).")] = DEFAULTS.n,
    steps: Annotated[int, typer.Option(min=1, help="Denoising steps.")] = DEFAULTS.steps,
    width: Annotated[int, typer.Option(min=16)] = DEFAULTS.width,
    height: Annotated[int, typer.Option(min=16)] = DEFAULTS.height,
    seed: Annotated[int | None, typer.Option(help="Base seed (default: random).")] = None,
    allow_swap: Annotated[
        bool, typer.Option("--allow-swap", help="Run even if pre-flight expects swap.")
    ] = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="DEBUG logs.")] = False,
) -> None:
    """Generate N candidates for a prompt; writes images + run.json under $QWI_HOME/runs."""
    logs.app_log(verbose)
    request = GenerateRequest(
        prompt,
        n=n,
        steps=steps,
        width=width,
        height=height,
        seed=seed,
        allow_swap=allow_swap,
        verbose=verbose,
    )
    try:
        rec = Pipeline().run(request, _render)
    except PipelineBusy as exc:
        raise _fail(exc) from exc
    error = rec.error or {}
    if rec.status == "ok":
        typer.echo(f"ok: {len(rec.candidates)} candidates")
    else:
        typer.echo(f"{rec.status}: {error.get('message', '')}".rstrip(": "), err=True)
    raise typer.Exit(errors.exit_code_for(rec.status, error.get("type")))


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
