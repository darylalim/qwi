"""Pre-flight memory check before each stage loads (PLAN.md §5.11).

Budget: `min(wired limit, available memory) - headroom`. The estimate includes the stage's
transient peak (for generate, the VAE decode) and is the larger of the static estimate below and
the largest peak measured for that stage in the last matching runs.

- estimate > wired limit → `InsufficientMemory` (the run can't work at all);
- estimate > available → `SwapRisk`, unless `allow_swap`, which also raises the swap watchdog's
  limit (§8 2026-10-15 `--allow-swap`);
- otherwise → ok.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qwi import logs, memory, paths
from qwi.config import DEFAULTS
from qwi.errors import InsufficientMemory, SwapRisk

HISTORY_RUNS = 10

# Generate peak on the q4 pack with the 2 GiB cache cap, fitted as `a + b * megapixels` through
# two measurements with the q4 encoder still loaded: 17.78 GB at 768² (M0b encoder residency) and
# 37.92 GB at 2048² (M1 memory-limit test). The evicted encoder is 3.97 GB (§4: 13.82 GB at 768²).
GENERATE_BASE_GB = 14.48
GENERATE_GB_PER_MPX = 5.59
ENCODER_Q4_GB = 3.97

# Weights only (§4) until M2/M3 measure each stage's peak.
STATIC_GB = {"guard": 1.1, "rewrite": 6.5, "rerank": 2.5}


def generate_gb(width: int, height: int, encoder_resident: bool = True) -> float:
    gb = GENERATE_BASE_GB + GENERATE_GB_PER_MPX * width * height / 1e6
    return gb if encoder_resident else gb - ENCODER_Q4_GB


def history_gb(stage: str, config: dict[str, Any], runs: Path | None = None) -> float:
    """Largest `peak_mem_gb[stage]` in the last `HISTORY_RUNS` records whose `config` matches."""
    runs = runs or paths.runs_dir()
    peaks: list[float] = []
    for run in sorted(runs.glob("*/run.json"), reverse=True):  # names start with a timestamp
        try:
            data = json.loads(run.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # unreadable records don't block a run
        if all(data.get("config", {}).get(k) == v for k, v in config.items()):
            if stage in data.get("peak_mem_gb", {}):
                peaks.append(float(data["peak_mem_gb"][stage]))
            if len(peaks) == HISTORY_RUNS:
                break
    return max(peaks, default=0.0)


def wired_limit_bytes() -> int:
    import mlx.core as mx

    return int(mx.device_info()["max_recommended_working_set_size"])


@dataclass(frozen=True)
class Check:
    est_gb: float
    avail_gb: float
    limit_gb: float
    decision: str  # ok | swap_risk_allowed

    def as_record(self) -> dict[str, Any]:
        """The `run.json` `preflight[stage]` entry."""
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in asdict(self).items()}


def check(
    stage: str,
    est_gb: float,
    *,
    allow_swap: bool = False,
    limit_bytes: int | None = None,
    available_bytes: int | None = None,
    headroom_bytes: int = DEFAULTS.preflight_headroom_bytes,
) -> Check:
    if limit_bytes is None:
        limit_bytes = wired_limit_bytes()
    if available_bytes is None:
        available_bytes = memory.vm_stat().available_bytes
    limit_gb = (limit_bytes - headroom_bytes) / memory.GB
    avail_gb = (available_bytes - headroom_bytes) / memory.GB
    context = {"est_gb": round(est_gb, 2), "avail_gb": round(avail_gb, 2)}
    context["limit_gb"] = round(limit_gb, 2)
    if est_gb > limit_gb:
        logs.event(logging.ERROR, stage, "preflight", "refused: over the memory limit", **context)
        raise InsufficientMemory(stage, est_gb, limit_gb)
    if est_gb > avail_gb and not allow_swap:
        logs.event(logging.WARNING, stage, "preflight", "refused: swap risk", **context)
        raise SwapRisk(stage, est_gb, avail_gb)
    decision = "swap_risk_allowed" if est_gb > avail_gb else "ok"
    lvl = logging.WARNING if decision != "ok" else logging.INFO
    logs.event(lvl, stage, "preflight", decision, **context)
    return Check(est_gb, avail_gb, limit_gb, decision)


def swap_limit_bytes(allow_swap: bool) -> int:
    """The `SwapWatchdog` limit for a run (§8 2026-10-15 `--allow-swap`)."""
    return DEFAULTS.allow_swap_limit_bytes if allow_swap else DEFAULTS.swap_limit_bytes
