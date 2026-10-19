"""One pipeline job at a time on a background thread (PLAN.md §5.6 "Execution model").

Streamlit reruns the whole script on every interaction, so a run inside the script would stop
partway on a refresh or a click. `JobManager` runs `Pipeline.run` on its own worker thread
instead; pages read `snapshot()` from a polling fragment and never touch the job directly. No
`st` imports here: the worker thread never calls `st.*`, and plain pytest can test all of it.

Only the manager is cached in Streamlit (`st.cache_resource`), never models: the pipeline still
loads and evicts each stage itself (§4).
"""

import logging
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from qwi import errors, logs
from qwi.events import Event
from qwi.pipeline import GenerateRequest, Pipeline
from qwi.record import RunRecord

PIPELINE_ENV = "QWI_PIPELINE"  # "fake": the scripted `FakePipeline` (tests, CI)
FAKE_SCRIPT_ENV = "QWI_FAKE_SCRIPT"  # which `FakePipeline.scripted()` run (ok, warn, ...)
SYNC_ENV = "QWI_JOBS_SYNC"  # "1": run jobs inside `submit()` (AppTest, §5.6 "Tests")


class Runner(Protocol):
    def run(self, request: GenerateRequest, on_event: Callable[[Event], None]) -> RunRecord: ...


def make_pipeline(kind: str | None, script: str | None = None) -> Runner:
    """The real `Pipeline`, or the scripted fake when `kind` is "fake" (`QWI_PIPELINE`)."""
    if kind == "fake":
        from qwi.ui.fake import FakePipeline

        return FakePipeline.scripted(script or "ok")
    return Pipeline()


class Busy(Exception):
    """`submit()` while a job is running: this process runs one job at a time."""

    def __init__(self) -> None:
        super().__init__("A job is already running")


@dataclass
class Job:
    request: GenerateRequest
    status: str = "running"  # running, then the record's status, or `error` if run() raised
    events: list[Event] = field(default_factory=list)
    record: RunRecord | None = None  # None until the run returns, or when run() raised
    error_type: str | None = None  # set when run() raised instead of returning a record
    message: str | None = None
    traceback: str | None = None


@dataclass(frozen=True)
class JobSnapshot:
    """A consistent copy of the current or last job, safe to read while the worker runs."""

    request: GenerateRequest
    status: str
    events: tuple[Event, ...]
    record: RunRecord | None
    error_type: str | None
    message: str | None
    traceback: str | None

    @property
    def running(self) -> bool:
        return self.status == "running"


class JobManager:
    """Owns one worker thread and the current job; the last job stays readable after it ends.

    `sync=True` runs the job inside `submit()` instead (`QWI_JOBS_SYNC=1`): AppTest doesn't
    advance background threads or `run_every` timers (§5.6 "Tests").
    """

    def __init__(self, pipeline: Runner, *, sync: bool = False) -> None:
        self._pipeline = pipeline
        self._sync = sync
        self._lock = threading.Lock()
        self._job: Job | None = None
        self._worker: threading.Thread | None = None

    def submit(self, request: GenerateRequest) -> None:
        with self._lock:
            if self._job is not None and self._job.status == "running":
                raise Busy()
            job = self._job = Job(request)
        if self._sync:
            self._work(job)
            return
        # A daemon thread: quitting the server mustn't wait minutes for a run; the run it kills
        # is shown as `interrupted` later (§5.11).
        self._worker = threading.Thread(target=self._work, args=(job,), name="qwi-job", daemon=True)
        self._worker.start()

    def cancel(self) -> bool:
        """Ask the running job to stop at its next check (§5.6); False when nothing is running."""
        with self._lock:
            if self._job is None or self._job.status != "running":
                return False
            self._job.request.cancel.set()
            return True

    def snapshot(self) -> JobSnapshot | None:
        with self._lock:
            job = self._job
            if job is None:
                return None
            return JobSnapshot(
                job.request,
                job.status,
                tuple(job.events),
                job.record,
                job.error_type,
                job.message,
                job.traceback,
            )

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the worker ends; True if no job is still running."""
        if self._worker is not None:
            self._worker.join(timeout)
        return self._worker is None or not self._worker.is_alive()

    def _emit(self, job: Job, event: Event) -> None:
        with self._lock:
            job.events.append(event)

    def _work(self, job: Job) -> None:
        # Catches BaseException too: a job left `running` would refuse every later submit.
        try:
            rec = self._pipeline.run(job.request, lambda event: self._emit(job, event))
        except BaseException as exc:
            text = traceback.format_exc()
            logs.exception(None, "job_error", str(exc), type=type(exc).__name__)
            with self._lock:
                job.status = errors.status(exc) or "error"
                job.error_type = type(exc).__name__
                job.message = str(exc)
                job.traceback = text
            return
        logs.event(logging.INFO, None, "job_finished", rec.status, status=rec.status)
        with self._lock:
            job.record = rec
            job.status = rec.status
            if rec.error is not None:
                job.error_type = rec.error.get("type")
                job.message = rec.error.get("message")
