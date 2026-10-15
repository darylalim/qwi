import errno
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from qwi import errors, models, paths, record
from qwi.record import RunRecord


@pytest.fixture
def run() -> Path:
    return paths.new_run_dir()


def test_write_then_read_round_trips(run: Path) -> None:
    rec = RunRecord("a red apple", seeds=[7, 8], config={"width": 768})
    path = record.write(rec, run)
    assert path == run / "run.json"
    assert not (run / "run.json.tmp").exists()
    assert record.read(run) == rec


def test_a_new_record_is_running_with_pid_and_start_time() -> None:
    rec = RunRecord("p")
    assert (rec.status, rec.pid, rec.rerank_query) == ("running", os.getpid(), "p")
    assert rec.started_at[:2] == "20"


def test_read_ignores_fields_a_newer_version_added(run: Path) -> None:
    record.write(RunRecord("p"), run)
    data = json.loads((run / "run.json").read_text())
    (run / "run.json").write_text(json.dumps({**data, "added_later": 1}))
    assert record.read(run).user_prompt == "p"


def test_failed_replace_keeps_the_previous_record(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record.write(RunRecord("first"), run)

    def crash(self: Path, target: Path) -> Path:
        raise OSError("crash mid-write")

    monkeypatch.setattr(Path, "replace", crash)
    with pytest.raises(OSError, match="crash"):
        record.write(RunRecord("second"), run)
    assert json.loads((run / "run.json").read_text())["user_prompt"] == "first"


def test_enospc_raises_disk_full_and_keeps_the_previous_record(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record.write(RunRecord("first"), run)

    def full(fd: int) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "fsync", full)
    with pytest.raises(errors.DiskFull):
        record.write(RunRecord("second"), run)
    assert not (run / "run.json.tmp").exists()
    assert record.read(run).user_prompt == "first"


def test_model_revisions_are_repo_at_revision() -> None:
    image = models.spec("image")
    assert record.model_revisions(["image"]) == {"image": f"{image.repo}@{image.revision}"}


@pytest.mark.parametrize(
    ("exc", "status", "has_error"),
    [
        (None, "ok", False),
        (errors.GuardBlocked("unsafe"), "blocked", False),
        (errors.GenerationCancelled(1, 4), "cancelled", False),
        (errors.DeviceOutOfMemory("generate", "swapped"), "error", True),
        (ValueError("boom"), "error", True),
    ],
)
def test_finish_sets_status_and_error(
    exc: BaseException | None, status: str, has_error: bool
) -> None:
    rec = RunRecord("p")
    record.finish(rec, exc, "generate")
    assert rec.status == status
    assert (rec.error is not None) == has_error
    if exc is not None and rec.error is not None:
        assert rec.error == {
            "type": type(exc).__name__,
            "message": str(exc),
            "stage": "generate",
            "log": "run.log",
        }


def _dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_pid_alive() -> None:
    assert record.pid_alive(os.getpid())
    assert not record.pid_alive(_dead_pid())


def test_dead_running_record_is_repaired_as_interrupted(run: Path) -> None:
    record.write(RunRecord("p", pid=_dead_pid()), run)
    assert record.repair_if_interrupted(run, lock_held=False).status == "interrupted"
    assert record.read(run).status == "interrupted"


def test_live_run_holding_the_lock_is_still_running(run: Path) -> None:
    record.write(RunRecord("p"), run)
    assert record.repair_if_interrupted(run, lock_held=True).status == "running"
    assert record.read(run).status == "running"


def test_free_lock_means_interrupted_even_if_the_pid_was_reused(run: Path) -> None:
    record.write(RunRecord("p"), run)  # our own pid: alive, but no run holds the lock
    assert record.repair_if_interrupted(run, lock_held=False).status == "interrupted"


def test_finished_records_are_left_alone(run: Path) -> None:
    rec = RunRecord("p", pid=_dead_pid())
    record.finish(rec)
    record.write(rec, run)
    assert record.repair_if_interrupted(run, lock_held=False).status == "ok"
