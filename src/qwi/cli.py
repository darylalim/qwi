"""Command-line entry point. Commands land milestone by milestone (PLAN.md §5 CLI reference)."""

import logging
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from functools import partial
from importlib.util import find_spec
from pathlib import Path
from types import FrameType
from typing import Annotated

import typer

from qwi import bestof, errors, lock, logs, paths, record, sessions, storage
from qwi.config import DEFAULTS
from qwi.errors import PipelineBusy, QwiError
from qwi.evaluation import RERANK_PRESETS, EvalRequest, EvalRunner, rescore, run_guard_allow
from qwi.events import (
    CandidateReady,
    DenoiseStep,
    Event,
    GuardVerdict,
    PreflightWarning,
    RerankDone,
    RewriteReady,
    RunStarted,
    StageFinished,
    StageStarted,
)
from qwi.pipeline import GenerateRequest, Pipeline, continue_seeds
from qwi.stages.guard import GuardMode
from qwi.stages.rewrite import RewriteStyle

app = typer.Typer(no_args_is_help=True, help="Local text-to-image pipeline for Apple Silicon.")
models_app = typer.Typer(no_args_is_help=True, help="Model weights (the only download path).")
runs_app = typer.Typer(no_args_is_help=True, help="Past runs under $QWI_HOME/runs.")
eval_app = typer.Typer(help="Eval set runs and blind review sessions (§5.4).")
app.add_typer(models_app, name="models")
app.add_typer(eval_app, name="eval")
app.add_typer(runs_app, name="runs")

MARK = {"pass": "ok  ", "warn": "WARN", "fail": "FAIL"}
GUARD_MODE = GuardMode(DEFAULTS.guard_mode)
REWRITE_STYLE = RewriteStyle(DEFAULTS.rewrite_style)
# What --continue-seeds takes from its source run, so these can't be given with it (§5.3).
FROM_SOURCE = {
    "prompt": "PROMPT",
    "seed": "--seed",
    "steps": "--steps",
    "width": "--width",
    "height": "--height",
    "no_rewrite": "--no-rewrite",
    "rewrite_style": "--rewrite-style",
}


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


def ranked_table(event: RerankDone) -> list[str]:
    """Rank, seed, prob, margin and file per candidate, best first (§5.3 "Selection")."""
    width = max(len("seed"), *(len(str(seed)) for seed in event.seeds))
    lines = [f"  rank  {'seed':>{width}}  prob  margin  file"]
    order = sorted(range(len(event.ranks)), key=lambda i: event.ranks[i])
    for i in order:
        lines.append(
            f"  {event.ranks[i]:>4}  {event.seeds[i]:>{width}}  {event.probs[i]:.2f}"
            f"  {event.margins[i]:>+6.2f}  {event.paths[i].name}"
        )
    return lines


def _render(event: Event, show_rewrite: bool = False) -> None:
    match event:
        case RunStarted(run_dir, seeds):
            typer.echo(f"run {run_dir.name} (seeds {seeds[0]}…{seeds[-1]})")
        case StageStarted(stage):
            typer.echo(f"{stage}: loading")
        case PreflightWarning(stage, est_gb, avail_gb):
            typer.echo(
                f"swap risk before {stage}: needs ~{est_gb:.1f} GB, {avail_gb:.1f} GB available; "
                "continuing (the swap watchdog stops the run if macOS starts swapping)",
                err=True,
            )
        case GuardVerdict(decision="warn", summary=summary):
            typer.echo(f"guard warning: {summary}", err=True)
        case GuardVerdict(decision="block", summary=summary):
            typer.echo(f"Blocked by guard: {summary}", err=True)
        case RewriteReady(text, fallback_reason, style):
            if show_rewrite:
                typer.echo(f"rewrite ({style}): {text or '(empty)'}")
            if fallback_reason:
                typer.echo(
                    f"rewrite not used ({fallback_reason}): generating from your prompt", err=True
                )
        case DenoiseStep(candidate, step, total):
            typer.echo(f"\r  candidate {candidate + 1}: step {step}/{total}", nl=False)
        case CandidateReady(index, seed, path):
            typer.echo(f"\r  candidate {index + 1} (seed {seed}): {path}")
        case StageFinished(stage, seconds, peak_gb):
            typer.echo(f"{stage}: {seconds:.1f} s, peak {peak_gb:.1f} GB")
        case RerankDone(best_index, best_path, low_confidence=low, min_score=floor):
            for line in ranked_table(event):
                typer.echo(line)
            prob = event.probs[best_index]
            typer.echo(f"best: candidate {best_index + 1} (prob {prob:.2f}): {best_path}")
            if low:
                typer.echo(
                    f"low confidence: no candidate scored above {floor:.2f}; results are saved. "
                    f"More candidates: qwi generate --continue-seeds {best_path.parent.name}",
                    err=True,
                )


def _on_command_line(ctx: typer.Context, name: str) -> bool:
    """Whether the user typed this parameter (not its default).

    Typer bundles its own click, so compare the source's name, not `click.core.ParameterSource`.
    """
    source = ctx.get_parameter_source(name)
    return source is not None and source.name == "COMMANDLINE"


@app.command()
def generate(
    ctx: typer.Context,
    prompt: Annotated[str | None, typer.Argument(help="What to draw.")] = None,
    n: Annotated[int, typer.Option("-n", min=1, help="Candidates (seeds).")] = DEFAULTS.n,
    steps: Annotated[int, typer.Option(min=1, help="Denoising steps.")] = DEFAULTS.steps,
    width: Annotated[int, typer.Option(min=16)] = DEFAULTS.width,
    height: Annotated[int, typer.Option(min=16)] = DEFAULTS.height,
    seed: Annotated[int | None, typer.Option(help="Base seed (default: random).")] = None,
    guard_mode: Annotated[
        GuardMode,
        typer.Option(help="How Controversial prompts are handled. Unsafe always blocks."),
    ] = GUARD_MODE,
    no_rewrite: Annotated[
        bool, typer.Option("--no-rewrite", help="Generate from the prompt as given.")
    ] = False,
    rewrite_style: Annotated[
        RewriteStyle, typer.Option(help="Rewriter system prompt: 40-120 or 400-500 words.")
    ] = REWRITE_STYLE,
    show_rewrite: Annotated[
        bool, typer.Option("--show-rewrite", help="Print the rewrite before generating.")
    ] = False,
    min_score: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Warn when the best candidate's reranker prob is below this.",
        ),
    ] = DEFAULTS.rerank_min_score,
    rerank_instruction: Annotated[
        str, typer.Option(help="The reranker's <Instruct> text (PLAN.md §5.3).")
    ] = DEFAULTS.rerank_instruction,
    allow_swap: Annotated[
        bool,
        typer.Option(
            "--allow-swap",
            help="Let a stage swap up to 6 GiB (default 2 GiB) before it is stopped.",
        ),
    ] = False,
    continue_from: Annotated[
        str | None,
        typer.Option(
            "--continue-seeds",
            metavar="RUN_ID",
            help="N more seeds after this run's last one: its prompt, size and steps, no rewrite.",
        ),
    ] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose", help="DEBUG logs.")] = False,
) -> None:
    """Generate N candidates for a prompt; writes images + run.json under $QWI_HOME/runs."""
    if not rerank_instruction.strip():
        raise typer.BadParameter("must not be empty", param_hint="--rerank-instruction")
    logs.app_log(verbose)
    request = GenerateRequest(
        prompt or "",
        n=n,
        steps=steps,
        width=width,
        height=height,
        seed=seed,
        guard_mode=guard_mode.value,
        rewrite=not no_rewrite,
        rewrite_style=rewrite_style.value,
        rerank_instruction=rerank_instruction,
        min_score=min_score,
        allow_swap=allow_swap,
        verbose=verbose,
    )
    if continue_from is not None:
        given = [hint for name, hint in FROM_SOURCE.items() if _on_command_line(ctx, name)]
        if given:
            raise typer.BadParameter(
                f"comes from the source run; don't also give {', '.join(given)}",
                param_hint="--continue-seeds",
            )
        try:
            run = storage.find_run(continue_from)
            request = continue_seeds(run, request, n if _on_command_line(ctx, "n") else None)
        except QwiError as exc:
            raise _fail(exc) from exc
        typer.echo(f"continuing {run.name}: same prompt, no rewrite")
    elif prompt is None:
        raise typer.BadParameter("give a prompt, or --continue-seeds RUN_ID", param_hint="PROMPT")
    try:
        with cancel_on_signals(request.cancel):
            rec = Pipeline().run(request, partial(_render, show_rewrite=show_rewrite))
    except PipelineBusy as exc:
        raise _fail(exc) from exc
    error = rec.error or {}
    if rec.status == "ok":
        typer.echo(f"ok: {len(rec.candidates)} candidates")
    else:
        typer.echo(f"{rec.status}: {error.get('message', '')}".rstrip(": "), err=True)
    raise typer.Exit(errors.exit_code_for(rec.status, error.get("type")))


def _render_rewrite(event: Event) -> None:
    """`qwi rewrite`'s progress, all on stderr: stdout is only the prompt (§5.6)."""
    match event:
        case StageStarted(stage):
            typer.echo(f"{stage}: loading", err=True)
        case StageFinished(stage, seconds, peak_gb):
            typer.echo(f"{stage}: {seconds:.1f} s, peak {peak_gb:.1f} GB", err=True)
        case RewriteReady(fallback_reason=reason) if reason:
            typer.echo(f"rewrite not used ({reason}): printing your prompt instead", err=True)
        case _:
            _render(event)  # guard verdicts and swap risk already go to stderr


@app.command()
def rewrite(
    prompt: Annotated[str, typer.Argument(help="What to draw.")],
    guard_mode: Annotated[
        GuardMode,
        typer.Option(help="How Controversial prompts are handled. Unsafe always blocks."),
    ] = GUARD_MODE,
    rewrite_style: Annotated[
        RewriteStyle, typer.Option(help="Rewriter system prompt: 40-120 or 400-500 words.")
    ] = REWRITE_STYLE,
) -> None:
    """Guard and rewrite a prompt, and print it to edit before `qwi generate --no-rewrite` (§5.6).

    Prints what a run would generate from: the rewrite, or your prompt on a fallback.
    """
    logs.app_log()
    request = GenerateRequest(
        prompt, guard_mode=guard_mode.value, rewrite_style=rewrite_style.value
    )
    try:
        with cancel_on_signals(request.cancel):
            result = Pipeline().rewrite_only(request, _render_rewrite)
    except PipelineBusy as exc:
        raise _fail(exc) from exc
    error = result.error or {}
    if result.status == "ok":
        typer.echo(result.prompt)
    elif result.status != "blocked":  # a block was already explained by its verdict
        typer.echo(f"{result.status}: {error.get('message', '')}".rstrip(": "), err=True)
    raise typer.Exit(errors.exit_code_for(result.status, error.get("type")))


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
    if rec.continued_from:
        typer.echo(f"continue from {rec.continued_from}: its prompt, no new rewrite")
    if rec.guard:
        typer.echo(f"guard    {rec.guard['decision']} ({rec.guard['mode']})", nl=False)
        typer.echo(f": {rec.guard['reason']}" if rec.guard["reason"] else "")
    if rec.rewritten_prompt:
        used = f"not used: {rec.rewrite_fallback_reason}" if rec.rewrite_fallback_reason else "used"
        typer.echo(f"rewrite  [{used}] {rec.rewritten_prompt}")
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


def _mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def _clip(text: str, width: int = 60) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 1] + "…"


@runs_app.command("list")
def runs_list() -> None:
    """Runs, newest first: id, date, status, star, size and prompt (§5.7 "Retention")."""
    held = lock.is_held()
    rows = []
    for run, _ in storage.run_records():
        rec = record.repair_if_interrupted(run, lock_held=held)
        started = rec.started_at[:16].replace("T", " ")
        star = "*" if rec.starred else " "
        size = _mb(storage.tree_bytes(run))
        rows.append((run.name, started, rec.status, star, size, _clip(rec.user_prompt)))
    if not rows:
        typer.echo("(no runs)")
        return
    widths = [max(len(row[i]) for row in rows) for i in range(5)]
    for row in rows:
        cells = [row[0].ljust(widths[0]), row[1].ljust(widths[1]), row[2].ljust(widths[2])]
        typer.echo("  ".join([*cells, row[3], row[4].rjust(widths[4]), row[5]]))


@runs_app.command("star")
def runs_star(
    run_id: Annotated[str, typer.Argument(help="Run folder name, or a unique prefix of it.")],
) -> None:
    """Toggle a run's star; `qwi runs prune` never deletes a starred run (§5.7)."""
    try:
        run = storage.find_run(run_id)
        rec = record.repair_if_interrupted(run, lock_held=lock.is_held())
        rec = record.set_starred(run, not rec.starred)
    except QwiError as exc:
        raise _fail(exc) from exc
    typer.echo(f"{'starred' if rec.starred else 'unstarred'} {run.name}")


@runs_app.command("prune")
def runs_prune(
    keep_days: Annotated[
        int | None, typer.Option(min=0, help="Keep runs started in the last N days.")
    ] = None,
    keep_last: Annotated[int | None, typer.Option(min=0, help="Keep the newest N runs.")] = None,
    eval_images: Annotated[
        bool,
        typer.Option("--eval", help="Prune eval image folders ($QWI_HOME/eval) instead of runs."),
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Only list what would be deleted.")
    ] = False,
) -> None:
    """Delete old run folders (or eval image folders); starred runs are never deleted (§5.7)."""
    if (keep_days is None) == (keep_last is None):
        raise typer.BadParameter(
            "give exactly one of --keep-days, --keep-last", param_hint="--keep-days/--keep-last"
        )
    logs.app_log()
    try:
        result = storage.prune(
            keep_days=keep_days, keep_last=keep_last, eval_images=eval_images, dry_run=dry_run
        )
    except QwiError as exc:
        raise _fail(exc) from exc
    verb = "would remove" if dry_run else "removed"
    for name, size in result.removed:
        typer.echo(f"{verb}  {name}  {_mb(size)}")
    for name in result.kept_starred:
        typer.echo(f"kept (starred)  {name}")
    what = "eval folders" if eval_images else "runs"
    total = _mb(sum(size for _, size in result.removed))
    typer.echo(f"{verb} {len(result.removed)} {what} ({total})")


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
    guard_mode: Annotated[GuardMode, typer.Option(help="Guard policy column.")] = GUARD_MODE,
    no_rewrite: Annotated[
        bool, typer.Option("--no-rewrite", help="Generate the prompts as written.")
    ] = False,
    rewrite_style: Annotated[
        RewriteStyle, typer.Option(help="Rewriter system prompt.")
    ] = REWRITE_STYLE,
    rerank_instruction: Annotated[
        str, typer.Option(help="The reranker's <Instruct> text for the scoring pass.")
    ] = DEFAULTS.rerank_instruction,
    allow_swap: Annotated[
        bool,
        typer.Option(
            "--allow-swap",
            help="Let a stage swap up to 6 GiB (default 2 GiB) before it is stopped.",
        ),
    ] = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Generate the eval set (12 prompts x seeds) for one variant (§5.4 "Protocol").

    Runs guard → rewrite → guard → encode → generate → rerank, each model loaded once.
    """
    if ctx.invoked_subcommand is not None:
        return
    if variant is None:
        raise typer.BadParameter("--variant is required", param_hint="--variant")
    if not rerank_instruction.strip():
        raise typer.BadParameter("must not be empty", param_hint="--rerank-instruction")
    logs.app_log(verbose)
    request = EvalRequest(
        variant,
        seeds=[int(s) for s in seeds.split(",")],
        only=only.split(",") if only else None,
        width=width,
        height=height,
        steps=steps,
        guard_mode=guard_mode.value,
        rewrite=not no_rewrite,
        rewrite_style=rewrite_style.value,
        rerank_instruction=rerank_instruction,
        allow_swap=allow_swap,
        verbose=verbose,
    )
    try:
        with cancel_on_signals(request.cancel):
            result = EvalRunner().run(request, _render)
    except QwiError as exc:
        raise _fail(exc) from exc
    typer.echo(f"{result.status}: {len(result.rows)} images")
    if result.blocked:
        typer.echo(f"  blocked  {', '.join(result.blocked)} (no images)", err=True)
    typer.echo(f"  results  {result.results_dir}")
    typer.echo(f"  images   {result.images_dir}")
    if result.error:
        typer.echo(f"{result.error['type']}: {result.error['message']}", err=True)
    raise typer.Exit(errors.exit_code_for(result.status, (result.error or {}).get("type")))


@eval_app.command("guard")
def eval_guard(
    guard_mode: Annotated[
        GuardMode, typer.Option(help="Policy column to check against.")
    ] = GUARD_MODE,
) -> None:
    """Run the guard false-positive set (eval/guard_allow.jsonl): pass = not blocked (§5.4)."""
    logs.app_log()
    try:
        rows = run_guard_allow(guard_mode.value)
    except QwiError as exc:
        raise _fail(exc) from exc
    width = max(len(row.id) for row in rows)
    for row in rows:
        mark = MARK["pass"] if row.passed else MARK["fail"]
        verdict = row.result.input.describe()
        typer.echo(f"{mark}  {row.id:<{width}}  {row.result.decision:<5}  {verdict}")
    failed = [row.id for row in rows if not row.passed]
    typer.echo(f"{len(rows) - len(failed)}/{len(rows)} not blocked (--guard-mode {guard_mode})")
    if failed:
        raise typer.Exit(1)


@eval_app.command("rescore")
def eval_rescore(
    results: Annotated[str, typer.Argument(help="A results folder name, e.g. 2026-10-17_bo4.")],
    label: Annotated[
        str, typer.Option(help="Names the output: rerank_<label>.jsonl. Presets: task, generic.")
    ],
    rerank_instruction: Annotated[
        str | None,
        typer.Option(help="The <Instruct> text (default: the preset named by --label)."),
    ] = None,
) -> None:
    """Re-score a variant's saved images under another reranker instruction (§5.4)."""
    instruction = rerank_instruction or RERANK_PRESETS.get(label)
    if instruction is None or not instruction.strip():
        raise typer.BadParameter(
            f"no preset {label!r}; pass --rerank-instruction", param_hint="--rerank-instruction"
        )
    logs.app_log()
    cancel = threading.Event()
    try:
        with cancel_on_signals(cancel):
            out = rescore(results, label, instruction, cancel=cancel)
    except QwiError as exc:
        raise _fail(exc) from exc
    typer.echo(f"ok: {out}")


@eval_app.command("session")
def eval_session(
    a: Annotated[
        list[str] | None, typer.Option("--a", help="Variant A results (repeatable).")
    ] = None,
    b: Annotated[
        list[str] | None, typer.Option("--b", help="Variant B results (repeatable).")
    ] = None,
    bo4: Annotated[
        str | None,
        typer.Option("--bo4", help="One variant's results: a best-of-N calibration session."),
    ] = None,
    seed: Annotated[int | None, typer.Option(help="Shuffle seed (default: random).")] = None,
    extend: Annotated[
        str | None, typer.Option(help="Keep this session's judgments; add the new pairs.")
    ] = None,
) -> None:
    """Create a blind session: A vs B pairs (--a/--b), or best-of-N candidates (--bo4) (§5.4)."""
    if bo4 is not None:
        if a or b or extend:
            raise typer.BadParameter("--bo4 takes no --a, --b or --extend", param_hint="--bo4")
        try:
            folder = bestof.create(sessions.results_dir(bo4), seed=seed)
        except QwiError as exc:
            raise _fail(exc) from exc
        typer.echo(f"session {folder.name}: judge it with `qwi eval review {folder.name}`")
        return
    if not a or not b:
        raise typer.BadParameter("give --a and --b, or --bo4", param_hint="--a/--b")
    try:
        folder = sessions.create(
            [sessions.results_dir(x) for x in a],
            [sessions.results_dir(x) for x in b],
            seed=seed,
            extend=sessions.session_dir(extend) if extend else None,
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
        report, path = sessions.write_report(sessions.session_dir(session), noise_floor)
    except QwiError as exc:
        raise _fail(exc) from exc
    o = report.overall
    typer.echo(f"A = {report.a}, B = {report.b}: {o.wins} wins, {o.losses} losses, {o.ties} ties")
    typer.echo(f"B is {report.verdict}: {report.reason}")
    typer.echo(f"report {path}")


@eval_app.command("calibrate")
def eval_calibrate(
    session: Annotated[str, typer.Argument(help="A judged best-of-N session folder, or its name.")],
) -> None:
    """Score each reranker instruction's top picks against the human's; write report.md (§5.4)."""
    try:
        report, path = bestof.write_calibration(sessions.session_dir(session))
    except QwiError as exc:
        raise _fail(exc) from exc
    for r in report.instructions:
        typer.echo(
            f"{r.label}: top-1 = best {r.agreement}/{report.items}, "
            f"top-1 acceptable {r.top_acceptable}/{report.with_acceptable}"
        )
    typer.echo(f"no reranker (base seed): top-1 = best {report.base_seed_agreement}/{report.items}")
    typer.echo(f"instruction: {report.winner}; --min-score {report.min_score:g}")
    typer.echo(f"report {path}")


@eval_app.command("review")
def eval_review(
    session: Annotated[str, typer.Argument(help="Session folder, or its name in eval/sessions.")],
    port: Annotated[int, typer.Option(help="Local port for the app.")] = DEFAULTS.review_port,
) -> None:
    """Judge a session blind in the browser; each Save writes review.csv (§5.4)."""
    folder = sessions.session_dir(session).resolve()
    try:
        # Open it here so a bad session fails now, not in the browser.
        if sessions.load_manifest(folder).get("kind") == sessions.BESTOF_KIND:
            bestof.open_session(folder)
        else:
            sessions.open_session(folder)
    except QwiError as exc:
        raise _fail(exc) from exc
    # The full UI on its own port, opened on the Eval review page with this session picked.
    env = {sessions.REVIEW_SESSION_ENV: str(folder)}
    _run_streamlit("streamlit_app.py", port, f"reviewing {folder.name}", "The review app", env)


@app.command()
def ui(
    port: Annotated[int, typer.Option(help="Local port for the app.")] = DEFAULTS.ui_port,
) -> None:
    """The Streamlit UI: generate, watch progress, cancel (§5.6)."""
    _run_streamlit("streamlit_app.py", port, "qwi", "The UI")


def _run_streamlit(
    page: str, port: int, what: str, needs: str, env: dict[str, str] | None = None
) -> None:
    """Run `ui/<page>` with this interpreter's Streamlit until Ctrl-C; exits with its code."""
    if find_spec("streamlit") is None:
        typer.echo(f"{needs} needs the ui extra: uv sync --all-extras", err=True)
        raise typer.Exit(1)
    app_path = Path(__file__).parent / "ui" / page
    command = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    typer.echo(f"{what} at http://localhost:{port} (Ctrl-C to stop)")
    try:
        # From the repo root, so Streamlit reads .streamlit/config.toml (localhost only).
        code = subprocess.call(command, env={**os.environ, **(env or {})}, cwd=paths.REPO)
    except KeyboardInterrupt:  # the app got the same Ctrl-C and is shutting down
        code = 0
    raise typer.Exit(code)
