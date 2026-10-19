"""What the Generate page shows, derived from a job's snapshot (PLAN.md §5.6). No `st` here.

The page renders a `RunView` and nothing else, so everything it decides (stage states, progress,
guard verdicts, the ranking) is tested with plain pytest, not AppTest.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
        tuple(candidates[i] for i in sorted(candidates)),
        rerank,
    )
