"""What the Generate page shows, derived from a job's snapshot (PLAN.md §5.6). No `st` here.

The page renders a `RunView` and nothing else, so everything it decides (stage states, progress,
guard verdicts, the ranking) is tested with plain pytest, not AppTest.
"""

import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from qwi import logs
from qwi.config import DEFAULTS
from qwi.events import (
    CandidateReady,
    DenoiseStep,
    GuardVerdict,
    PreflightWarning,
    RerankDone,
    RewriteReady,
    RunStarted,
    StageFinished,
    StageStarted,
)
from qwi.pipeline import GenerateRequest
from qwi.ui.jobs import JobSnapshot

StageState = Literal["running", "complete", "error"]  # `st.status` states

STAGE_LABELS = {
    "guard": "Guard",
    "rewrite": "Rewrite",
    "encode": "Encode",
    "generate": "Generate",
    "rerank": "Rerank",
}


@dataclass(frozen=True)
class StageView:
    name: str
    state: StageState
    seconds: float | None = None
    peak_gb: float | None = None

    @property
    def label(self) -> str:
        name = STAGE_LABELS.get(self.name, self.name.capitalize())
        if self.seconds is None:
            return name
        return f"{name}: {self.seconds:.1f} s, peak {self.peak_gb:.1f} GB"


@dataclass(frozen=True)
class Progress:
    candidate: int  # 0-based
    step: int  # 1-based
    total: int
    n: int

    @property
    def fraction(self) -> float:
        return min(1.0, (self.candidate * self.total + self.step) / (self.n * self.total))

    @property
    def text(self) -> str:
        return f"Candidate {self.candidate + 1} of {self.n}, step {self.step} of {self.total}"


@dataclass(frozen=True)
class CandidateView:
    index: int
    seed: int
    path: Path
    prob: float | None = None  # None: not ranked (yet, or ever after a failure, §5.11)
    margin: float | None = None
    rank: int | None = None

    @property
    def best(self) -> bool:
        return self.rank == 1

    @property
    def caption(self) -> str:
        parts = [f"Seed {self.seed}"]
        if self.prob is not None and self.margin is not None:
            parts += [f"prob {self.prob:.2f}", f"margin {self.margin:+.2f}"]
        return " · ".join(parts)


@dataclass(frozen=True)
class RunView:
    status: str
    n: int
    run_dir: Path | None
    stages: tuple[StageView, ...]
    progress: Progress | None  # only while generate runs
    rewrite: RewriteReady | None
    verdicts: tuple[GuardVerdict, ...]  # each guard pass, in order
    swap_risks: tuple[PreflightWarning, ...]  # stages whose pre-check warned (§5.11)
    candidates: tuple[CandidateView, ...]
    rerank: RerankDone | None

    @property
    def warnings(self) -> tuple[GuardVerdict, ...]:
        return tuple(v for v in self.verdicts if v.decision == "warn")

    @property
    def block(self) -> GuardVerdict | None:
        return next((v for v in self.verdicts if v.decision == "block"), None)

    @property
    def low_confidence(self) -> bool:
        return self.rerank is not None and self.rerank.low_confidence


def summarize(snap: JobSnapshot) -> RunView:
    run_dir: Path | None = None
    stages: dict[str, StageView] = {}
    step: DenoiseStep | None = None
    rewrite: RewriteReady | None = None
    verdicts: list[GuardVerdict] = []
    swap_risks: list[PreflightWarning] = []
    candidates: dict[int, CandidateView] = {}
    rerank: RerankDone | None = None
    for event in snap.events:
        match event:
            case RunStarted():
                run_dir = event.run_dir
            case StageStarted():
                stages[event.stage] = StageView(event.stage, "running")
            case StageFinished():
                stages[event.stage] = StageView(
                    event.stage, "complete", event.seconds, event.peak_gb
                )
            case DenoiseStep():
                step = event
            case RewriteReady():
                rewrite = event
            case GuardVerdict():
                verdicts.append(event)
            case PreflightWarning():
                swap_risks.append(event)
            case CandidateReady():
                candidates[event.index] = CandidateView(event.index, event.seed, event.path)
            case RerankDone():
                rerank = event
                for i, c in candidates.items():
                    candidates[i] = CandidateView(
                        c.index, c.seed, c.path, event.probs[i], event.margins[i], event.ranks[i]
                    )
    if not snap.running:  # a stage still running when the job ended is where it stopped
        stages = {
            k: StageView(k, "error") if s.state == "running" else s for k, s in stages.items()
        }
    generating = snap.running and stages.get("generate", StageView("", "error")).state == "running"
    progress = (
        Progress(step.candidate, step.step, step.total, snap.request.n)
        if generating and step is not None
        else None
    )
    return RunView(
        snap.status,
        snap.request.n,
        run_dir,
        tuple(stages.values()),
        progress,
        rewrite,
        tuple(verdicts),
        tuple(swap_risks),
        tuple(candidates[i] for i in sorted(candidates)),
        rerank,
    )


# --- Errors (§5.11 "Error types"): one banner per class, with its action -------------------

ERROR_TITLES = {
    "PipelineBusy": "Another run is in progress",
    "InsufficientMemory": "Not enough free memory",
    "DeviceOutOfMemory": "Ran out of GPU memory",
    "ModelUnavailable": "Model not available",
    "DiskFull": "Disk full",
    "InsufficientDisk": "Not enough disk",
}
MEMORY_ERRORS = {"InsufficientMemory", "DeviceOutOfMemory"}  # retrying smaller can help


@dataclass(frozen=True)
class Retry:
    kind: str  # size | count
    label: str


@dataclass(frozen=True)
class ErrorBanner:
    title: str
    body: str
    retries: tuple[Retry, ...]  # one-click cheaper settings; no automatic retries (§5.11)


def retries(request: GenerateRequest) -> tuple[Retry, ...]:
    """The cheaper settings that differ from this request's (§5.11 "Retry at 512²", "N=2")."""
    found = []
    if max(request.width, request.height) > DEFAULTS.retry_size:
        found.append(Retry("size", f"Retry at {DEFAULTS.retry_size}²"))
    if request.n > DEFAULTS.retry_n:
        found.append(Retry("count", f"Retry with N={DEFAULTS.retry_n}"))
    return tuple(found)


def retry_request(request: GenerateRequest, kind: str) -> GenerateRequest:
    """The same request at the cheaper setting, with a fresh cancel flag."""
    changes: dict[str, int] = (
        {"width": DEFAULTS.retry_size, "height": DEFAULTS.retry_size}
        if kind == "size"
        else {"n": DEFAULTS.retry_n}
    )
    return replace(request, cancel=threading.Event(), **changes)


def error_banner(snap: JobSnapshot) -> ErrorBanner | None:
    if snap.status != "error":
        return None
    kind = snap.error_type or "error"
    message = snap.message or "The run stopped with an error."
    if kind not in ERROR_TITLES:
        where = "below" if snap.traceback else "in the log below"
        if snap.kind == "rewrite_only" and not snap.traceback:  # no run folder, so no run.log
            where = "in $QWI_HOME/logs/qwi.log"
        return ErrorBanner("Unexpected error", f"{kind}: {message}. Traceback {where}.", ())
    # The cheaper settings shrink generation; they don't help a rewrite.
    found = retries(snap.request) if kind in MEMORY_ERRORS and snap.kind == "generate" else ()
    return ErrorBanner(ERROR_TITLES[kind], message, found)


def log_lines(run_dir: Path, limit: int = 200, min_level: str = "DEBUG") -> list[str]:
    """The run's last `limit` log entries at `min_level` or above, readable, each with its
    traceback indented below. Always the run's own `run.log`: `qwi.log` has other runs' lines."""
    path = run_dir / "run.log"
    if not path.exists():
        return []
    out = []
    for line in logs.read(path, min_level)[-limit:]:
        out.append(
            f"{line['ts'][11:19]}  {line['level']:<7}  {line.get('stage') or '-':<8}  {line['msg']}"
        )
        if trace := line.get("traceback"):
            out += [f"    {row}" for row in trace.splitlines()]
    return out
