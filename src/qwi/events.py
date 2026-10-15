"""Typed pipeline events (PLAN.md §5.6): `Pipeline.run` emits them; the CLI and UI render them.

M2 adds `GuardVerdict` and `RewriteReady`, M3 adds `RerankDone`.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunStarted:
    run_dir: Path
    seeds: list[int]


@dataclass(frozen=True)
class StageStarted:
    stage: str


@dataclass(frozen=True)
class StageFinished:
    stage: str
    seconds: float
    peak_gb: float


@dataclass(frozen=True)
class GuardVerdict:
    check: str  # "input" | "rewrite": which pass (§5.2)
    level: str | None  # Safe | Unsafe | Controversial; None when unparseable
    categories: tuple[str, ...]
    decision: str  # allow | warn | block: this pass's own action
    summary: str  # one line: pass, level, categories, mode


@dataclass(frozen=True)
class RewriteReady:
    text: str  # the cleaned rewrite, also when it isn't used
    fallback_reason: str | None  # None: generation uses the rewrite (§5.1)
    style: str


@dataclass(frozen=True)
class DenoiseStep:
    candidate: int  # index into the run's seeds
    step: int  # 1-based
    total: int


@dataclass(frozen=True)
class CandidateReady:
    index: int
    seed: int
    path: Path


Event = (
    RunStarted
    | StageStarted
    | StageFinished
    | GuardVerdict
    | RewriteReady
    | DenoiseStep
    | CandidateReady
)
