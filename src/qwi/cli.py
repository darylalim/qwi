"""Command-line entry point. Commands land milestone by milestone (PLAN.md §5 CLI reference)."""

import logging
import os
import signal
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Annotated

import typer

from qwi import errors, lock, logs, paths, record, sessions, storage
from qwi.config import DEFAULTS
from qwi.errors import PipelineBusy, QwiError
from qwi.evaluation import EvalRequest, EvalRunner
from qwi.events import CandidateReady, DenoiseStep, Event, RunStarted, StageFinished, StageStarted
from qwi.pipeline import GenerateRequest, Pipeline

app = typer.Typer(no_args_is_help=True, help="Local text-to-image pipeline for Apple Silicon.")
models_app = typer.Typer(no_args_is_help=True, help="Model weights (the only download path).")
runs_app = typer.Typer(no_args_is_help=True, help="Past runs under $QWI_HOME/runs.")
eval_app = typer.Typer(help="Eval set runs and blind review sessions (§5.4).")
app.add_typer(models_app, name="models")
app.add_typer(eval_app, name="eval")
app.add_typer(runs_app, name="runs")

MARK = {"pass": "ok  ", "warn": "WARN", "fail": "FAIL"}


def _version(value: bool) -> None:
    if value:
        from importlib.metadata import version

        typer.echo(f"qwi {version('qwi')}")
        raise typer.Exit()


@app.callback()
def main(
    _: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version,
            is_eager=True,
            help="Show the version (from the package metadata) and exit.",
        ),
    ] = False,
) -> None:
    """qwi: guard → rewrite → generate N seeds → rerank."""


def _fail(exc: QwiError) -> typer.Exit:
    logs.event(logging.ERROR, None, "cli_error", str(exc), type=type(exc).__name__)
    typer.echo(str(exc), err=True)
    return typer.Exit(exc.exit_code)


@contextmanager
def cancel_on_signals(cancel: threading.Event) -> Generator[None]:
    """First Ctrl-C / SIGTERM: cancel cleanly at the next step. Second: exit now (§5.11).

    Exiting now skips the run's cleanup; the OS frees its memory and lock, and its `running`
    record is repaired as `interrupted` the next time it's read.
    """

    def handle(signum: int, frame: FrameType | None) -> None:
        if cancel.is_set():
            typer.echo("\nstopping now", err=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(130)
        cancel.set()
        typer.echo("\ncancelling at the next step (Ctrl-C again to stop now)", err=True)

    previous = {sig: signal.signal(sig, handle) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


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
        with cancel_on_signals(request.cancel):
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


def _per_stage(values: dict[str, float], unit: str) -> str:
    return ", ".join(f"{stage} {value:g} {unit}" for stage, value in values.items()) or "-"


def _show_record(run_id: str, rec: record.RunRecord) -> None:
    cfg = rec.config
    typer.echo(f"run      {run_id}  [{rec.status}]  started {rec.started_at}  pid {rec.pid}")
    typer.echo(f"prompt   {rec.user_prompt}")
    if cfg:
        size = f"{cfg.get('width')}x{cfg.get('height')}"
        typer.echo(
            f"config   {size}, {cfg.get('steps')} steps, n={cfg.get('n')}, "
            f"encoder {cfg.get('encoder')}, seeds {rec.seeds}"
        )
    for c in rec.candidates:
        typer.echo(
            f"  cand {c['index']}  seed {c['seed']}  {c['path']}  "
            f"{c.get('seconds', 0):g} s  peak {c.get('peak_gb', 0):g} GB"
        )
    typer.echo(f"time     {_per_stage(rec.timings_s, 's')}")
    typer.echo(f"peak     {_per_stage(rec.peak_mem_gb, 'GB')}")
    typer.echo(f"swap     {_per_stage(rec.swap_gb, 'GB')}")
    for stage, check in rec.preflight.items():
        typer.echo(
            f"preflight {stage}: est {check['est_gb']} GB, avail {check['avail_gb']} GB, "
            f"limit {check['limit_gb']} GB -> {check['decision']}"
        )
    if rec.error:
        typer.echo(f"error    {rec.error['type']} in {rec.error['stage']}: {rec.error['message']}")


def _show_log_line(line: dict) -> None:
    fixed = {"ts", "level", "stage", "event", "msg", "traceback"}
    context = " ".join(f"{k}={v}" for k, v in line.items() if k not in fixed)
    head = f"{line['ts']} {line['level']:<7} {line.get('stage') or '-':<9} {line['event']}"
    typer.echo(f"{head}: {line['msg']}  {context}".rstrip())
    if "traceback" in line:
        typer.echo("    " + line["traceback"].replace("\n", "\n    "))


@runs_app.command("show")
def runs_show(
    run_id: Annotated[str, typer.Argument(help="Run folder name, or a unique prefix of it.")],
    log: Annotated[bool, typer.Option("--log", help="Also print run.log.")] = False,
    level: Annotated[
        str, typer.Option(help="Lowest log level to print: debug, info, warning, error.")
    ] = "info",
) -> None:
    """A run's summary from run.json, and optionally its JSON-lines log (§5.11)."""
    try:
        run = storage.find_run(run_id)
    except QwiError as exc:
        raise _fail(exc) from exc
    rec = record.repair_if_interrupted(run, lock_held=lock.is_held())
    _show_record(run.name, rec)
    if log:
        log_path = run / "run.log"
        if not log_path.exists():
            typer.echo("(no run.log)")
            return
        typer.echo("")
        for line in logs.read(log_path, level):
            _show_log_line(line)


@eval_app.callback(invoke_without_command=True)
def eval_run(
    ctx: typer.Context,
    variant: Annotated[
        str | None, typer.Option(help="Name for this variant, e.g. baseline (required).")
    ] = None,
    seeds: Annotated[str, typer.Option(help="Comma-separated seeds.")] = ",".join(
        map(str, DEFAULTS.eval_seeds)
    ),
    only: Annotated[
        str | None, typer.Option(help="Comma-separated prompt ids (smoke runs).")
    ] = None,
    steps: Annotated[int, typer.Option(min=1)] = DEFAULTS.steps,
    width: Annotated[int, typer.Option(min=16)] = DEFAULTS.width,
    height: Annotated[int, typer.Option(min=16)] = DEFAULTS.height,
    allow_swap: Annotated[bool, typer.Option("--allow-swap")] = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Generate the eval set (12 prompts x seeds) for one variant (§5.4 "Protocol")."""
    if ctx.invoked_subcommand is not None:
        return
    if variant is None:
        raise typer.BadParameter("--variant is required", param_hint="--variant")
    logs.app_log(verbose)
    request = EvalRequest(
        variant,
        seeds=[int(s) for s in seeds.split(",")],
        only=only.split(",") if only else None,
        width=width,
        height=height,
        steps=steps,
        allow_swap=allow_swap,
        verbose=verbose,
    )
    try:
        with cancel_on_signals(request.cancel):
            result = EvalRunner().run(request, _render)
    except QwiError as exc:
        raise _fail(exc) from exc
    typer.echo(f"{result.status}: {len(result.rows)} images")
    typer.echo(f"  results  {result.results_dir}")
    typer.echo(f"  images   {result.images_dir}")
    if result.error:
        typer.echo(f"{result.error['type']}: {result.error['message']}", err=True)
    raise typer.Exit(errors.exit_code_for(result.status, (result.error or {}).get("type")))


def _session_dir(arg: str) -> Path:
    path = Path(arg)
    return path if (path / "manifest.json").is_file() else paths.EVAL_SET / "sessions" / arg


@eval_app.command("session")
def eval_session(
    a: Annotated[list[str], typer.Option("--a", help="Variant A results (repeatable).")],
    b: Annotated[list[str], typer.Option("--b", help="Variant B results (repeatable).")],
    seed: Annotated[int | None, typer.Option(help="Shuffle seed (default: random).")] = None,
    extend: Annotated[
        str | None, typer.Option(help="Keep this session's judgments; add the new pairs.")
    ] = None,
) -> None:
    """Create a blind paired session: anonymous ids, shuffled order, random sides (§5.4)."""
    try:
        folder = sessions.create(
            [sessions.results_dir(x) for x in a],
            [sessions.results_dir(x) for x in b],
            seed=seed,
            extend=_session_dir(extend) if extend else None,
        )
    except QwiError as exc:
        raise _fail(exc) from exc
    typer.echo(f"session {folder.name}: judge review.csv, then `qwi eval compare {folder.name}`")


@eval_app.command("compare")
def eval_compare(
    session: Annotated[str, typer.Argument(help="Session folder, or its name in eval/sessions.")],
    noise_floor: Annotated[int, typer.Option(min=0, help="From an A-vs-A session (§5.4).")] = 0,
) -> None:
    """Unblind a fully judged session, score it, and write report.md (§5.4 decision rule)."""
    try:
        report, path = sessions.write_report(_session_dir(session), noise_floor)
    except QwiError as exc:
        raise _fail(exc) from exc
    o = report.overall
    typer.echo(f"A = {report.a}, B = {report.b}: {o.wins} wins, {o.losses} losses, {o.ties} ties")
    typer.echo(f"B is {report.verdict}: {report.reason}")
    typer.echo(f"report {path}")
