"""`qwi.ui.history`: what the History page shows for each run (PLAN.md §5.6, §5.7, §5.11)."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from qwi import errors, lock, paths, record
from qwi.record import RunRecord
from qwi.ui import history


def dead_pid() -> int:
    proc = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(proc.stdout)


def write_run(**fields: Any) -> Path:
    run = paths.new_run_dir()
    defaults: dict[str, Any] = {"user_prompt": "a red apple", "status": "ok"}
    record.write(RunRecord(**(defaults | fields)), run)
    return run


FINISHED: dict[str, Any] = {
    "seeds": [7, 8],
    "candidates": [
        {
            "index": 0,
            "seed": 7,
            "path": "cand_0_seed7.png",
            "prob": 0.41,
            "margin": -0.4,
            "rank": 2,
        },
        {"index": 1, "seed": 8, "path": "cand_1_seed8.png", "prob": 0.88, "margin": 2.0, "rank": 1},
    ],
    "best_index": 1,
    "timings_s": {"guard": 1.5, "generate": 300.0, "rerank": 20.25},
    "peak_mem_gb": {"guard": 2.0, "generate": 21.5, "rerank": 9.0},
    "swap_gb": {"generate": 0.0},
    "config": {"width": 768, "height": 768, "steps": 20, "n": 2, "encoder": "A-q4"},
    "guard": {
        "mode": "default",
        "input": {"level": "Controversial", "categories": ["Politically Sensitive Topics"]},
        "rewrite": {"level": "Safe", "categories": []},
        "decision": "warn",
        "reason": "Controversial input",
    },
}


def test_runs_are_listed_newest_first_with_their_folders() -> None:
    older = write_run(user_prompt="first")
    newer = write_run(user_prompt="second")
    entries = history.load()
    assert {e.run_dir for e in entries} == {older, newer}
    assert [e.record.user_prompt for e in entries] == [
        r.user_prompt
        for _, r in sorted(((p, record.read(p)) for p in (older, newer)), reverse=True)
    ]
    assert entries[0].run_id == entries[0].run_dir.name


def test_an_empty_history() -> None:
    assert history.load() == []


def test_a_row_has_the_best_prob_total_seconds_and_star() -> None:
    write_run(starred=True, **FINISHED)
    (entry,) = history.load()
    row = entry.row()
    assert row["Prompt"] == "a red apple"
    assert row["Status"] == "ok"
    assert row["Best prob"] == 0.88
    assert row["Seconds"] == pytest.approx(321.75)
    assert row["Starred"] is True
    assert len(row["Started"]) == 19


def test_an_unranked_run_has_no_best_prob_or_seconds() -> None:
    write_run(status="cancelled")
    (entry,) = history.load()
    assert entry.best_prob is None
    assert entry.total_seconds is None


def test_candidates_resolve_inside_the_run_and_keep_their_ranking() -> None:
    run = write_run(**FINISHED)
    (entry,) = history.load()
    cands = entry.candidates()
    assert [c.path for c in cands] == [run / "cand_0_seed7.png", run / "cand_1_seed8.png"]
    assert [c.best for c in cands] == [False, True]
    assert cands[1].caption == "Seed 8 · prob 0.88 · margin +2.00"


def test_guard_lines_show_each_pass_and_the_decision() -> None:
    write_run(**FINISHED)
    (entry,) = history.load()
    assert entry.guard_lines() == [
        "Input: Controversial (Politically Sensitive Topics)",
        "Rewrite: Safe",
        "Decision: warn (default mode): Controversial input",
    ]


def test_stage_rows_and_config_line() -> None:
    write_run(**FINISHED)
    (entry,) = history.load()
    assert entry.stage_rows() == [
        {"Stage": "Guard", "Seconds": 1.5, "Peak GB": 2.0, "Swap GB": None},
        {"Stage": "Generate", "Seconds": 300.0, "Peak GB": 21.5, "Swap GB": 0.0},
        {"Stage": "Rerank", "Seconds": 20.25, "Peak GB": 9.0, "Swap GB": None},
    ]
    assert entry.config_line() == "768x768, 20 steps, N=2, encoder A-q4, seeds 7, 8"


def test_a_record_with_no_guard_or_config_shows_neither() -> None:
    write_run()
    (entry,) = history.load()
    assert entry.guard_lines() == []
    assert entry.config_line() is None
    assert entry.stage_rows() == []


def test_a_dead_running_run_is_listed_and_repaired_as_interrupted() -> None:
    run = write_run(status="running", pid=dead_pid())
    (entry,) = history.load()
    assert entry.record.status == "interrupted"
    assert entry.status_color == "orange"
    assert record.read(run).status == "interrupted"


def test_a_live_run_holding_the_lock_stays_running() -> None:
    run = write_run(status="running", pid=os.getpid())
    with lock.RunLock():
        (entry,) = history.load()
    assert entry.record.status == "running"
    assert record.read(run).status == "running"


def test_each_status_has_a_badge_color() -> None:
    for status in ("ok", "running", "interrupted", "cancelled", "blocked", "error"):
        assert status in history.STATUS_COLORS


def test_the_selected_row_maps_to_the_run_id_it_was_drawn_with() -> None:
    ids = ["20261019-120000-bbbbbb", "20261019-110000-aaaaaa"]
    assert history.selected_run([1], ids) == "20261019-110000-aaaaaa"
    assert history.selected_run([], ids) is None
    assert history.selected_run([5], ids) is None


def test_set_starred_toggles_the_flag_on_disk() -> None:
    run = write_run()
    assert record.set_starred(run, True).starred
    assert record.read(run).starred
    assert not record.set_starred(run, False).starred
    assert not record.read(run).starred
    assert not (run / "run.json.tmp").exists()


def test_a_running_run_cannot_be_starred() -> None:
    run = write_run(status="running")
    with pytest.raises(errors.RunStillRunning, match="still running"):
        record.set_starred(run, True)
    assert not record.read(run).starred
