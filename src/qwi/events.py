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
class DenoiseStep:
    candidate: int  # index into the run's seeds
    step: int  # 1-based
    total: int


@dataclass(frozen=True)
class CandidateReady:
    index: int
    seed: int
    path: Path


Event = RunStarted | StageStarted | StageFinished | DenoiseStep | CandidateReady
