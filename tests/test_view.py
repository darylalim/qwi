"""`qwi.ui.view`: what the Generate page shows for a job (PLAN.md §5.6, §5.11)."""

import logging
from pathlib import Path
from typing import Any

import pytest

from qwi import errors, logs, paths
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
from qwi.pipeline import GenerateRequest
from qwi.ui import view
from qwi.ui.jobs import JobSnapshot

RUN = Path("/runs/20261018-101500-abcdef")
SAFE = GuardVerdict("input", "Safe", (), "allow", "input: Safe")


def snapshot(
    events: list[Event],
    status: str = "running",
    error: BaseException | None = None,
    traceback: str | None = None,
    **request: Any,
) -> JobSnapshot:
    fields: dict[str, Any] = {"prompt": "a red apple", "n": 2}
    req = GenerateRequest(**(fields | request))
    kind = type(error).__name__ if error else None
    message = str(error) if error else None
    return JobSnapshot(req, status, tuple(events), None, kind, message, traceback)


def started(*stages: str) -> list[Event]:
    events: list[Event] = [RunStarted(RUN, [7, 8])]
    for stage in stages:
        events += [StageStarted(stage), StageFinished(stage, 1.25, 3.5)]
    return events


def test_finished_and_running_stages() -> None:
    run = view.summarize(snapshot([*started("guard"), StageStarted("rewrite")]))
    assert [(s.name, s.state) for s in run.stages] == [
        ("guard", "complete"),
        ("rewrite", "running"),
    ]
    assert run.stages[0].label == "Guard: 1.2 s, peak 3.5 GB"
    assert run.stages[1].label == "Rewrite"
    assert run.run_dir == RUN


def test_a_stage_still_running_when_the_job_ended_is_an_error() -> None:
    run = view.summarize(snapshot([*started("guard"), StageStarted("generate")], "cancelled"))
    assert [s.state for s in run.stages] == ["complete", "error"]


def test_progress_while_generating_counts_every_step_of_every_candidate() -> None:
    events = [*started("encode"), StageStarted("generate"), DenoiseStep(1, 5, 20)]
    progress = view.summarize(snapshot(events)).progress
    assert progress is not None
    assert progress.text == "Candidate 2 of 2, step 5 of 20"
    assert progress.fraction == 25 / 40


def test_no_progress_once_generate_finishes_or_the_job_ends() -> None:
    steps: list[Event] = [StageStarted("generate"), DenoiseStep(1, 20, 20)]
    finished = [*started("encode"), *steps, StageFinished("generate", 1.0, 1.0)]
    assert view.summarize(snapshot(finished)).progress is None
    assert view.summarize(snapshot([*started("encode"), *steps], "error")).progress is None


def test_guard_warnings_and_a_block() -> None:
    warn = GuardVerdict("input", "Controversial", ("Politics",), "warn", "input: warn")
    block = GuardVerdict("rewrite", "Unsafe", ("Violent",), "block", "rewrite: block")
    run = view.summarize(snapshot([*started("guard"), SAFE, warn, block], "blocked"))
    assert run.warnings == (warn,)
    assert run.block == block
    assert view.summarize(snapshot([SAFE])).block is None


def test_candidates_are_unranked_until_rerank_then_carry_its_scores() -> None:
    a, b = Path("/runs/x/cand_0_seed7.png"), Path("/runs/x/cand_1_seed8.png")
    events: list[Event] = [*started("generate"), CandidateReady(0, 7, a), CandidateReady(1, 8, b)]
    before = view.summarize(snapshot(events))
    assert [(c.seed, c.prob, c.best) for c in before.candidates] == [
        (7, None, False),
        (8, None, False),
    ]
    assert before.candidates[0].caption == "Seed 7"

    done = RerankDone(
        1, RUN / "best.png", [7, 8], [a, b], [0.5, 2.0], [0.3, 0.8], [2, 1], True, 0.5
    )
    after = view.summarize(snapshot([*events, done], "ok"))
    assert [c.best for c in after.candidates] == [False, True]
    assert after.candidates[1].caption == "Seed 8 · prob 0.80 · margin +2.00"
    assert after.low_confidence


def test_the_rewrite_is_kept() -> None:
    rewrite = RewriteReady("A detailed apple.", None, "long")
    assert view.summarize(snapshot([*started("rewrite"), rewrite])).rewrite == rewrite


def test_swap_risks_are_kept() -> None:
    risk = PreflightWarning("generate", 14.97, 9.87)
    assert view.summarize(snapshot([*started("encode"), risk])).swap_risks == (risk,)


# --- Errors (§5.11) ---------------------------------------------------------------------------


def test_retries_offer_only_cheaper_settings() -> None:
    def labels(**request: Any) -> list[str]:
        fields: dict[str, Any] = {"prompt": "x"} | request
        return [r.label for r in view.retries(GenerateRequest(**fields))]

    assert labels() == ["Retry at 512²", "Retry with N=2"]  # 768², N=4
    assert labels(width=512, height=512) == ["Retry with N=2"]
    assert labels(n=2) == ["Retry at 512²"]
    assert labels(width=512, height=512, n=1) == []


def test_a_retry_is_the_same_request_at_the_cheaper_setting_with_a_fresh_cancel() -> None:
    request = GenerateRequest("x", seed=5, guard_mode="strict")
    request.cancel.set()
    smaller = view.retry_request(request, "size")
    assert (smaller.width, smaller.height, smaller.n) == (512, 512, 4)
    assert (smaller.prompt, smaller.seed, smaller.guard_mode) == ("x", 5, "strict")
    assert not smaller.cancel.is_set()
    assert view.retry_request(request, "count").n == 2


@pytest.mark.parametrize(
    ("error", "title", "retry"),
    [
        (errors.PipelineBusy("2026-10-18T10:00:00"), "Another run is in progress", False),
        (errors.InsufficientMemory("generate", 30.0, 22.9), "Not enough free memory", True),
        (errors.DeviceOutOfMemory("generate", "swap"), "Ran out of GPU memory", True),
        (errors.ModelUnavailable("image", "missing"), "Model not available", False),
        (errors.DiskFull(10**9), "Disk full", False),
        (errors.InsufficientDisk("a run", 0, 10**9, 20 * 10**9), "Not enough disk", False),
    ],
)
def test_each_error_class_has_its_banner(error: Exception, title: str, retry: bool) -> None:
    banner = view.error_banner(snapshot([], "error", error))
    assert banner is not None
    assert (banner.title, banner.body) == (title, str(error))
    assert bool(banner.retries) == retry


def test_an_unexpected_error_points_at_the_traceback() -> None:
    logged = view.error_banner(snapshot([], "error", RuntimeError("boom")))
    raised = view.error_banner(snapshot([], "error", RuntimeError("boom"), traceback="Traceback"))
    assert logged is not None
    assert raised is not None
    assert logged.title == raised.title == "Unexpected error"
    assert logged.body == "RuntimeError: boom. Traceback in the log below."
    assert raised.body == "RuntimeError: boom. Traceback below."
    assert logged.retries == ()


def test_only_errors_get_a_banner() -> None:
    for status in ("running", "ok", "cancelled", "blocked"):
        assert view.error_banner(snapshot([], status)) is None


def test_log_lines_are_readable_and_capped() -> None:
    run = paths.new_run_dir()
    assert view.log_lines(run) == []
    with logs.run_log(run):
        for i in range(5):
            logs.event(logging.INFO, "generate", "step", f"line {i}")
        logs.event(logging.WARNING, None, "end", "done")
    lines = view.log_lines(run, limit=2)
    assert len(lines) == 2
    assert lines[0].endswith("INFO     generate  line 4")
    assert lines[1].endswith("WARNING  -         done")


def test_log_lines_show_a_traceback_under_its_line() -> None:
    run = paths.new_run_dir()
    with logs.run_log(run):
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            logs.exception("generate", "error", "boom")
    first, *trace = view.log_lines(run)
    assert first.endswith("ERROR    generate  boom")
    assert trace[0] == "    Traceback (most recent call last):"
    assert trace[-1] == "    RuntimeError: boom"
