"""`qwi.ui.view.summarize`: what the Generate page shows for a job's events (PLAN.md §5.6)."""

from pathlib import Path
from typing import Any

from qwi.events import (
    CandidateReady,
    DenoiseStep,
    Event,
    GuardVerdict,
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


def snapshot(events: list[Event], status: str = "running", **request: Any) -> JobSnapshot:
    fields: dict[str, Any] = {"prompt": "a red apple", "n": 2}
    req = GenerateRequest(**(fields | request))
    return JobSnapshot(req, status, tuple(events), None, None, None, None)


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
