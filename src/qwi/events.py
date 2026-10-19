"""Typed pipeline events (PLAN.md §5.6): `Pipeline.run` emits them; the CLI and UI render them.

M2 added `GuardVerdict` and `RewriteReady`, M3 `RerankDone`, M5a `PreflightWarning`.
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
class PreflightWarning:
    """A stage's memory pre-check says swap risk; the run continues (§5.11, §8 2026-10-17)."""

    stage: str
    est_gb: float
    avail_gb: float


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


@dataclass(frozen=True)
class RerankDone:
    best_index: int
    best_path: Path  # best.png, linked to the winning candidate
    seeds: list[int]  # the lists below are all aligned with the run's seeds
    paths: list[Path]  # each candidate's image
    margins: list[float]
    probs: list[float]
    ranks: list[int]  # 1 = best
    low_confidence: bool  # the best prob is below min_score (§5.3)
    min_score: float


Event = (
    RunStarted
    | StageStarted
    | StageFinished
    | PreflightWarning
    | GuardVerdict
    | RewriteReady
    | DenoiseStep
    | CandidateReady
    | RerankDone
)
