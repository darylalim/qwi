"""Typed pipeline events (PLAN.md §5.6): `Pipeline.run` emits them; the CLI and UI render them.

M2 adds `GuardVerdict` (and `RewriteReady` with the rewrite stage), M3 adds `RerankDone`.
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
    decision: str  # allow | warn | block, after both passes so far
    summary: str  # one line: pass, level, categories, mode


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


Event = RunStarted | StageStarted | StageFinished | GuardVerdict | DenoiseStep | CandidateReady
