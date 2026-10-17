import logging
import math
import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from qwi import cli, logs, paths, record, storage
from qwi.config import DEFAULTS
from qwi.errors import InsufficientDisk
from qwi.events import (
    CandidateReady,
    DenoiseStep,
    GuardVerdict,
    RerankDone,
    RewriteReady,
    RunStarted,
    StageFinished,
    StageStarted,
)
from qwi.record import RunRecord

runner = CliRunner()


def test_doctor_prints_a_row_per_check_and_exits_0(monkeypatch: pytest.MonkeyPatch) -> None:
    checks = [
        storage.Check("disk", "pass", "46.8 GB free"),
        storage.Check("runs size", "warn", "2.1 GB"),
    ]
    monkeypatch.setattr(storage, "doctor", lambda: checks)
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0
    assert result.output.splitlines() == [
        "ok    disk       46.8 GB free",
        "WARN  runs size  2.1 GB",
    ]


def test_doctor_exits_1_on_any_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "doctor", lambda: [storage.Check("disk", "fail", "5 GB free")])
    assert runner.invoke(cli.app, ["doctor"]).exit_code == 1


def test_models_fetch_reports_insufficient_disk_with_exit_6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(role: str, *, dry_run: bool) -> None:
        raise InsufficientDisk(role, 10 * 10**9, 29 * 10**9, 20 * 10**9)

    monkeypatch.setattr(storage, "fetch", refuse)
    result = runner.invoke(cli.app, ["models", "fetch", "rerank"])
    assert result.exit_code == 6
    assert "Not enough disk for rerank" in result.output


def test_models_fetch_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []

    def fake(role: str, *, dry_run: bool) -> None:
        seen.append(dry_run)

    monkeypatch.setattr(storage, "fetch", fake)
    result = runner.invoke(cli.app, ["models", "fetch", "image", "--dry-run"])
    assert (result.exit_code, seen) == (0, [True])
    assert "nothing downloaded" in result.output


class FakePipeline:
    def __init__(self, status: str, error: dict | None = None) -> None:
        self.status, self.error = status, error
        self.requests: list = []

    def __call__(self) -> "FakePipeline":
        return self

    def run(self, request, on_event) -> RunRecord:
        self.requests.append(request)
        on_event(RunStarted(Path("/runs/20261015-093005-abcdef"), [7, 8]))
        on_event(StageStarted("generate"))
        on_event(DenoiseStep(0, 1, 2))
        on_event(CandidateReady(0, 7, Path("/runs/x/cand_0_seed7.png")))
        on_event(StageFinished("generate", 251.6, 13.8))
        rec = RunRecord(request.prompt, status=self.status, error=self.error)
        rec.candidates = [{"index": 0}]
        return rec


def test_generate_renders_events_and_exits_0(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakePipeline("ok")
    monkeypatch.setattr(cli, "Pipeline", fake)
    result = runner.invoke(cli.app, ["generate", "a red apple", "-n", "2", "--seed", "7"])
    assert result.exit_code == 0
    assert "run 20261015-093005-abcdef (seeds 7…8)" in result.output
    assert "candidate 1 (seed 7): /runs/x/cand_0_seed7.png" in result.output
    assert "generate: 251.6 s, peak 13.8 GB" in result.output
    (request,) = fake.requests
    assert (request.prompt, request.n, request.seed, request.width) == ("a red apple", 2, 7, 768)


def done(low: bool = False, margins: tuple[float, ...] = (0.5, 2.0, -1.0)) -> RerankDone:
    probs = [round(1 / (1 + math.exp(-m)), 4) for m in margins]
    ranks = [sorted(margins, reverse=True).index(m) + 1 for m in margins]
    seeds = [7 + i for i in range(len(margins))]
    paths_ = [Path(f"/runs/x/cand_{i}_seed{s}.png") for i, s in enumerate(seeds)]
    best = ranks.index(1)
    return RerankDone(
        best, Path("/runs/x/best.png"), seeds, paths_, list(margins), probs, ranks, low, 0.5
    )


def test_ranked_table_lists_the_best_first() -> None:
    assert cli.ranked_table(done()) == [
        "  rank  seed  prob  margin  file",
        "     1     8  0.88   +2.00  cand_1_seed8.png",
        "     2     7  0.62   +0.50  cand_0_seed7.png",
        "     3     9  0.27   -1.00  cand_2_seed9.png",
    ]


def test_ranked_table_widens_for_long_seeds() -> None:
    event = done(margins=(1.0,))
    event = RerankDone(0, event.best_path, [1234567], event.paths, [1.0], [0.73], [1], False, 0.5)
    assert cli.ranked_table(event)[1] == "     1  1234567  0.73   +1.00  cand_0_seed7.png"


def test_rerank_prints_the_table_then_the_best(capsys: pytest.CaptureFixture[str]) -> None:
    cli._render(done())
    out = capsys.readouterr()
    lines = out.out.splitlines()
    assert lines[:4] == cli.ranked_table(done())
    assert lines[4] == "best: candidate 2 (prob 0.88): /runs/x/best.png"
    assert out.err == ""


def test_low_confidence_warns_on_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    cli._render(done(low=True, margins=(-0.5,)))
    out = capsys.readouterr()
    assert out.out.splitlines()[-1] == "best: candidate 1 (prob 0.38): /runs/x/best.png"
    assert out.err == (
        "low confidence: no candidate scored above 0.50; results are saved. "
        "More candidates: qwi generate --continue-seeds x\n"
    )


def test_rerank_flags_reach_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakePipeline("ok")
    monkeypatch.setattr(cli, "Pipeline", fake)
    assert runner.invoke(cli.app, ["generate", "p"]).exit_code == 0
    args = ["generate", "p", "--min-score", "0.7", "--rerank-instruction", "Is it a cube?"]
    assert runner.invoke(cli.app, args).exit_code == 0
    first, second = fake.requests
    assert (first.min_score, first.rerank_instruction) == (0.5, DEFAULTS.rerank_instruction)
    assert (second.min_score, second.rerank_instruction) == (0.7, "Is it a cube?")


@pytest.mark.parametrize(
    "args",
    [["--min-score", "1.5"], ["--min-score", "-0.1"], ["--rerank-instruction", "  "]],
    ids=["above-1", "below-0", "blank-instruction"],
)
def test_bad_rerank_flags_are_usage_errors(
    monkeypatch: pytest.MonkeyPatch, args: list[str]
) -> None:
    fake = FakePipeline("ok")
    monkeypatch.setattr(cli, "Pipeline", fake)
    result = runner.invoke(cli.app, ["generate", "p", *args])
    assert result.exit_code == 2
    assert fake.requests == []


def test_generate_exit_code_follows_the_record(monkeypatch: pytest.MonkeyPatch) -> None:
    error = {"type": "InsufficientMemory", "message": "Not enough free memory for generate"}
    monkeypatch.setattr(cli, "Pipeline", FakePipeline("error", error))
    result = runner.invoke(cli.app, ["generate", "p"])
    assert result.exit_code == 4
    assert "error: Not enough free memory for generate" in result.output


def test_guard_mode_reaches_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakePipeline("ok")
    monkeypatch.setattr(cli, "Pipeline", fake)
    assert runner.invoke(cli.app, ["generate", "p"]).exit_code == 0
    assert runner.invoke(cli.app, ["generate", "p", "--guard-mode", "strict"]).exit_code == 0
    assert [r.guard_mode for r in fake.requests] == ["default", "strict"]


def test_an_unknown_guard_mode_is_a_usage_error() -> None:
    assert runner.invoke(cli.app, ["generate", "p", "--guard-mode", "off"]).exit_code == 2


class GuardedPipeline:
    """Emits one guard verdict; a block ends the run like the real pipeline."""

    def __init__(self, decision: str) -> None:
        self.decision = decision

    def __call__(self) -> "GuardedPipeline":
        return self

    def run(self, request, on_event) -> RunRecord:
        summary = "input prompt is Controversial (Violent) (--guard-mode default)"
        on_event(GuardVerdict("input", "Controversial", ("Violent",), self.decision, summary))
        status = "blocked" if self.decision == "block" else "ok"
        return RunRecord(request.prompt, status=status)


def test_a_guard_warning_goes_to_stderr_and_the_run_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "Pipeline", GuardedPipeline("warn"))
    result = runner.invoke(cli.app, ["generate", "p"])
    assert result.exit_code == 0
    assert "guard warning: input prompt is Controversial (Violent)" in result.stderr
    assert "guard warning" not in result.stdout


def test_a_guard_block_prints_why_and_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "Pipeline", GuardedPipeline("block"))
    result = runner.invoke(cli.app, ["generate", "p"])
    assert result.exit_code == 2
    assert "Blocked by guard: input prompt is Controversial (Violent)" in result.stderr


def test_rewrite_flags_reach_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakePipeline("ok")
    monkeypatch.setattr(cli, "Pipeline", fake)
    assert runner.invoke(cli.app, ["generate", "p"]).exit_code == 0
    args = ["generate", "p", "--no-rewrite", "--rewrite-style", "long"]
    assert runner.invoke(cli.app, args).exit_code == 0
    args = ["generate", "p", "--rewrite-style", "short"]
    assert runner.invoke(cli.app, args).exit_code == 0
    assert [(r.rewrite, r.rewrite_style) for r in fake.requests] == [
        (True, "long"),
        (False, "long"),
        (True, "short"),
    ]


class RewritingPipeline:
    def __init__(self, fallback: str | None) -> None:
        self.fallback = fallback

    def __call__(self) -> "RewritingPipeline":
        return self

    def run(self, request, on_event) -> RunRecord:
        on_event(RewriteReady("A glossy red apple on an oak table.", self.fallback, "short"))
        return RunRecord(request.prompt, status="ok")


def test_show_rewrite_prints_the_rewrite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "Pipeline", RewritingPipeline(None))
    shown = runner.invoke(cli.app, ["generate", "p", "--show-rewrite"])
    assert "rewrite (short): A glossy red apple on an oak table." in shown.stdout
    hidden = runner.invoke(cli.app, ["generate", "p"])
    assert "rewrite (short)" not in hidden.output


def test_a_rewrite_fallback_is_always_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "Pipeline", RewritingPipeline("quoted_text_missing"))
    result = runner.invoke(cli.app, ["generate", "p"])
    assert result.exit_code == 0
    assert "rewrite not used (quoted_text_missing): generating from your prompt" in result.stderr


def test_generate_reports_busy_with_exit_3(monkeypatch: pytest.MonkeyPatch) -> None:
    from qwi.errors import PipelineBusy

    class Busy:
        def run(self, request, on_event) -> None:
            raise PipelineBusy("2026-10-15T09:30:05")

    monkeypatch.setattr(cli, "Pipeline", Busy)
    result = runner.invoke(cli.app, ["generate", "p"])
    assert (result.exit_code, "Another run is in progress" in result.output) == (3, True)


def _saved_run(name: str, **fields: Any) -> Path:
    run = paths.runs_dir() / name
    run.mkdir(parents=True)
    rec = RunRecord(
        "a red apple",
        seeds=[7, 8],
        config={"width": 768, "height": 768, "n": 2, "steps": 20, "encoder": "A-q4"},
        **fields,
    )
    record.write(rec, run)
    return run


def test_runs_show_prints_the_summary() -> None:
    _saved_run(
        "20261015-093005-abcdef",
        status="ok",
        candidates=[
            {"index": 0, "seed": 7, "path": "cand_0_seed7.png", "seconds": 125.8, "peak_gb": 13.8}
        ],
        timings_s={"generate": 251.6},
        peak_mem_gb={"generate": 13.8},
        swap_gb={"generate": 0.0},
    )
    result = runner.invoke(cli.app, ["runs", "show", "20261015-0930"])  # a unique prefix
    assert result.exit_code == 0
    out = result.output
    assert "run      20261015-093005-abcdef  [ok]" in out
    assert "768x768, 20 steps, n=2, encoder A-q4, seeds [7, 8]" in out
    assert "cand 0  seed 7  cand_0_seed7.png  125.8 s  peak 13.8 GB" in out
    assert "swap     generate 0 GB" in out


def test_runs_show_prints_the_guard_and_the_rewrite() -> None:
    _saved_run(
        "20261015-093005-abcdef",
        status="ok",
        guard={"mode": "default", "decision": "warn", "reason": "input:Controversial:Violent"},
        rewritten_prompt="A knight on a misty battlefield.",
    )
    _saved_run(
        "20261015-093006-abcdef",
        status="ok",
        guard={"mode": "strict", "decision": "allow", "reason": None},
        rewritten_prompt="A storefront.",
        rewrite_fallback_reason="quoted_text_missing",
    )
    first = runner.invoke(cli.app, ["runs", "show", "20261015-093005"]).output
    assert "guard    warn (default): input:Controversial:Violent" in first
    assert "rewrite  [used] A knight on a misty battlefield." in first
    second = runner.invoke(cli.app, ["runs", "show", "20261015-093006"]).output
    assert "guard    allow (strict)\n" in second
    assert "rewrite  [not used: quoted_text_missing] A storefront." in second


def test_runs_show_repairs_an_interrupted_run() -> None:
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    _saved_run("20261015-093005-abcdef", pid=child.pid)
    result = runner.invoke(cli.app, ["runs", "show", "20261015-093005-abcdef"])
    assert "[interrupted]" in result.output


def test_runs_show_log_filters_by_level_and_prints_tracebacks() -> None:
    run = _saved_run("20261015-093005-abcdef", status="error")
    with logs.run_log(run):
        logs.event(logging.INFO, "generate", "candidate_ready", "cand_0_seed7.png", seed=7)
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            logs.exception("generate", "error", "Unexpected error", type="RuntimeError")
    result = runner.invoke(cli.app, ["runs", "show", run.name, "--log", "--level", "warning"])
    assert result.exit_code == 0
    assert "candidate_ready" not in result.output
    assert "ERROR   generate  error: Unexpected error  type=RuntimeError" in result.output
    assert "    RuntimeError: boom" in result.output


@pytest.mark.parametrize("run_id", ["nope", "20261015"])
def test_runs_show_unknown_or_ambiguous_id_exits_1(run_id: str) -> None:
    _saved_run("20261015-093005-abcdef")
    _saved_run("20261015-093010-123456")
    result = runner.invoke(cli.app, ["runs", "show", run_id])
    assert result.exit_code == 1
    assert f"Run {run_id!r}" in result.output


def test_version_comes_from_the_package_metadata() -> None:
    import tomllib
    from importlib.metadata import version

    from .conftest import ROOT

    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert result.output == f"qwi {version('qwi')}\n"
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert version("qwi") == declared  # the installed metadata matches pyproject.toml


def plain(text: str) -> str:
    """Drop ANSI styling and Rich's error box, and join its wrapped lines into one."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)  # CI advertises color
    return " ".join(text.replace("│", " ").split())


def source_run(status: str = "ok") -> Path:
    run = paths.new_run_dir()
    rec = RunRecord(
        "a red apple",
        status=status,
        rewritten_prompt="A glossy red apple",
        prompt_source="rewrite",
        seeds=[7, 8],
        candidates=[{"index": i, "seed": 7 + i, "path": f"cand_{i}.png"} for i in range(2)],
        config={
            "width": 512,
            "height": 512,
            "n": 2,
            "steps": 4,
            "encoder": "A-q4",
            "rewrite": "long",
        },
    )
    record.write(rec, run)
    return run


def test_continue_seeds_runs_the_source_prompt_from_its_next_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakePipeline("ok")
    monkeypatch.setattr(cli, "Pipeline", fake)
    run = source_run()
    args = ["generate", "--continue-seeds", run.name, "--min-score", "0.7"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert f"continuing {run.name}: same prompt, no rewrite" in result.output
    (request,) = fake.requests
    assert (request.prompt, request.seed, request.n, request.width, request.steps) == (
        "a red apple",
        9,
        2,
        512,
        4,
    )
    assert request.continuation.prompt == "A glossy red apple"
    assert request.continuation.run_id == run.name
    assert request.min_score == 0.7


def test_continue_seeds_takes_n_and_a_run_id_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakePipeline("ok")
    monkeypatch.setattr(cli, "Pipeline", fake)
    run = source_run()
    result = runner.invoke(cli.app, ["generate", "--continue-seeds", run.name[:-2], "-n", "6"])
    assert result.exit_code == 0, result.output
    (request,) = fake.requests
    assert (request.seed, request.n) == (9, 6)


@pytest.mark.parametrize(
    ("extra", "named"),
    [
        (["a red apple"], "PROMPT"),
        (["--seed", "3"], "--seed"),
        (["--steps", "8", "--width", "256"], "--steps, --width"),
        (["--no-rewrite"], "--no-rewrite"),
        (["--rewrite-style", "short"], "--rewrite-style"),
    ],
    ids=["prompt", "seed", "size", "no-rewrite", "style"],
)
def test_continue_seeds_refuses_what_comes_from_the_source(
    monkeypatch: pytest.MonkeyPatch, extra: list[str], named: str
) -> None:
    fake = FakePipeline("ok")
    monkeypatch.setattr(cli, "Pipeline", fake)
    result = runner.invoke(cli.app, ["generate", *extra, "--continue-seeds", source_run().name])
    assert result.exit_code == 2
    assert f"don't also give {named}" in plain(result.output)
    assert fake.requests == []


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: "nope", "Run 'nope': no match"),
        (lambda: source_run("interrupted").name, "it is interrupted"),
    ],
    ids=["unknown", "interrupted"],
)
def test_continue_seeds_refuses_a_run_it_cannot_continue(
    monkeypatch: pytest.MonkeyPatch, make: Any, message: str
) -> None:
    fake = FakePipeline("ok")
    monkeypatch.setattr(cli, "Pipeline", fake)
    result = runner.invoke(cli.app, ["generate", "--continue-seeds", make()])
    assert result.exit_code == 1
    assert message in result.output
    assert fake.requests == []


def test_generate_needs_a_prompt_or_continue_seeds() -> None:
    result = runner.invoke(cli.app, ["generate"])
    assert result.exit_code == 2
    assert "give a prompt, or --continue-seeds RUN_ID" in plain(result.output)


def test_runs_show_names_the_source_of_a_continued_run() -> None:
    run = source_run()
    rec = record.read(run)
    rec.continued_from = "20261017-090000-abcdef"
    record.write(rec, run)
    result = runner.invoke(cli.app, ["runs", "show", run.name])
    assert result.exit_code == 0, result.output
    assert "continue from 20261017-090000-abcdef: its prompt, no new rewrite" in result.output
