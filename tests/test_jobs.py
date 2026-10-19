"""`JobManager` with `FakePipeline` (PLAN.md §5.6 "Tests"): no models, no Streamlit."""

from typing import Any

import pytest

from qwi import lock, logs, paths, record
from qwi.errors import PipelineBusy
from qwi.events import CandidateReady, DenoiseStep, RerankDone, RunStarted
from qwi.pipeline import GenerateRequest
from qwi.ui.fake import FakePipeline, Hold
from qwi.ui.jobs import Busy, JobManager

TIMEOUT = 10


def request(**overrides: Any) -> GenerateRequest:
    fields: dict[str, Any] = {"prompt": "a red apple", "n": 2, "steps": 3, "seed": 7}
    return GenerateRequest(**(fields | overrides))


def finish(jobs: JobManager) -> None:
    assert jobs.wait(TIMEOUT)


def test_nothing_submitted_has_no_snapshot() -> None:
    jobs = JobManager(FakePipeline())
    assert jobs.snapshot() is None
    assert not jobs.cancel()


def test_submit_runs_on_a_worker_and_keeps_events_and_the_record() -> None:
    fake = FakePipeline()
    jobs = JobManager(fake)
    jobs.submit(request())
    finish(jobs)

    snap = jobs.snapshot()
    assert snap is not None
    assert (snap.status, snap.running, snap.error_type) == ("ok", False, None)
    assert isinstance(snap.events[0], RunStarted)
    assert isinstance(snap.events[-1], RerankDone)
    assert sum(isinstance(e, DenoiseStep) for e in snap.events) == 6  # 2 candidates x 3 steps
    assert [e.seed for e in snap.events if isinstance(e, CandidateReady)] == [7, 8]
    assert snap.record is not None
    assert snap.record.best_index == 1  # the fake's margins: candidate 1 wins
    run = snap.events[0].run_dir
    assert record.read(run) == snap.record
    assert (run / "best.png").exists()
    assert not lock.is_held()


def test_a_second_submit_while_running_is_busy_then_accepted_after() -> None:
    hold = Hold()
    fake = FakePipeline(hold=hold)
    jobs = JobManager(fake)
    jobs.submit(request())
    assert hold.reached.wait(TIMEOUT)

    snap = jobs.snapshot()
    assert snap is not None
    assert snap.running
    with pytest.raises(Busy):
        jobs.submit(request(prompt="another"))
    assert len(fake.requests) == 1  # the refused request never reached the pipeline

    hold.release()
    finish(jobs)
    jobs.submit(request(prompt="another"))
    finish(jobs)
    assert [r.prompt for r in fake.requests] == ["a red apple", "another"]


def test_snapshots_are_copies_the_worker_never_changes() -> None:
    hold = Hold()
    jobs = JobManager(FakePipeline(hold=hold))
    jobs.submit(request())
    assert hold.reached.wait(TIMEOUT)
    before = jobs.snapshot()
    assert before is not None
    count = len(before.events)
    hold.release()
    finish(jobs)
    assert len(before.events) == count
    assert before.status == "running"


def test_cancel_mid_run_ends_cancelled_and_evicts() -> None:
    hold = Hold(candidate=1, step=2)
    fake = FakePipeline(hold=hold)
    jobs = JobManager(fake)
    jobs.submit(request())
    assert hold.reached.wait(TIMEOUT)
    assert jobs.cancel()
    hold.release()
    finish(jobs)

    snap = jobs.snapshot()
    assert snap is not None
    assert snap.status == "cancelled"
    assert snap.record is not None
    assert [c["seed"] for c in snap.record.candidates] == [7]  # the finished candidate is kept
    assert fake.evictions == 4  # guard, rewrite, encode, and generate in `finally`
    assert not lock.is_held()
    assert not jobs.cancel()  # nothing running any more


def test_a_worker_that_raises_is_an_error_with_its_traceback() -> None:
    fake = FakePipeline(raises=RuntimeError("boom in the pipeline"))
    jobs = JobManager(fake)
    logs.app_log()
    jobs.submit(request())
    finish(jobs)

    snap = jobs.snapshot()
    assert snap is not None
    assert snap.status == "error"
    assert (snap.error_type, snap.message) == ("RuntimeError", "boom in the pipeline")
    assert snap.traceback is not None
    assert "RuntimeError: boom in the pipeline" in snap.traceback
    assert snap.record is None
    assert not lock.is_held()
    (line,) = [x for x in logs.read(paths.logs_dir() / "qwi.log") if x["event"] == "job_error"]
    assert "boom in the pipeline" in line["traceback"]

    fake.raises = None
    jobs.submit(request())  # the next submit is accepted
    finish(jobs)
    snap = jobs.snapshot()
    assert snap is not None
    assert snap.status == "ok"


def test_another_process_holding_the_lock_is_an_error_not_a_stuck_job() -> None:
    jobs = JobManager(FakePipeline())
    with lock.RunLock():
        jobs.submit(request())
        finish(jobs)
    snap = jobs.snapshot()
    assert snap is not None
    assert (snap.status, snap.error_type) == ("error", PipelineBusy.__name__)
    assert snap.message is not None
    assert snap.message.startswith("Another run is in progress")


def test_an_error_record_carries_its_type_and_message() -> None:
    class Failing(FakePipeline):
        def run(self, request, on_event):
            rec = super().run(request, on_event)
            record.finish(rec, MemoryError("out"), "generate")
            return rec

    jobs = JobManager(Failing())
    jobs.submit(request())
    finish(jobs)
    snap = jobs.snapshot()
    assert snap is not None
    assert (snap.status, snap.error_type, snap.message) == ("error", "MemoryError", "out")


def test_sync_mode_runs_inside_submit() -> None:
    jobs = JobManager(FakePipeline(), sync=True)
    jobs.submit(request())
    snap = jobs.snapshot()
    assert snap is not None
    assert snap.status == "ok"
    assert jobs.wait(0)


def test_a_rewrite_only_job_keeps_its_result_and_no_record() -> None:
    fake = FakePipeline()
    jobs = JobManager(fake)
    jobs.submit(request(), kind="rewrite_only")
    finish(jobs)

    snap = jobs.snapshot()
    assert snap is not None
    assert (snap.kind, snap.status, snap.record) == ("rewrite_only", "ok", None)
    assert snap.rewrite is not None
    assert snap.rewrite.prompt == "A detailed photograph of a red apple."
    assert [type(e).__name__ for e in snap.events][-1] == "RewriteReady"
    assert (len(fake.rewrites), fake.requests) == (1, [])
    assert not paths.runs_dir().exists()
    assert not lock.is_held()


def test_a_blocked_rewrite_only_job_ends_blocked() -> None:
    jobs = JobManager(FakePipeline(guard="block"))
    jobs.submit(request(), kind="rewrite_only")
    finish(jobs)
    snap = jobs.snapshot()
    assert snap is not None
    assert snap.status == "blocked"
    assert snap.rewrite is not None
    assert snap.rewrite.rewritten_prompt == ""


def test_each_submit_gets_the_next_job_number() -> None:
    jobs = JobManager(FakePipeline(), sync=True)
    jobs.submit(request(), kind="rewrite_only")
    first = jobs.snapshot()
    jobs.submit(request())
    second = jobs.snapshot()
    assert first is not None
    assert second is not None
    assert (first.number, second.number, second.kind) == (1, 2, "generate")


def test_an_unknown_job_kind_is_refused() -> None:
    jobs = JobManager(FakePipeline())
    with pytest.raises(ValueError, match="unknown job kind"):
        jobs.submit(request(), kind="upscale")
    assert jobs.snapshot() is None
