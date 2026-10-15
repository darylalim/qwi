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

# Measured generate peaks (megapixels, GB) on the q4 pack with the q4 encoder still loaded:
# 256² 2 steps 11.32 GiB (M0b weight source), 768² 17.78 GB (M0b encoder residency), 2048² 37.92 GB
# (M1 memory-limit test). The peak isn't linear in pixels (~10.7 GB/Mpx below 768², ~5.6 above),
# so estimates interpolate between neighbours and extend the end segments (§8 2026-10-15).
GENERATE_PEAKS_GB = ((0.065536, 12.15), (0.589824, 17.78), (4.194304, 37.92))
# Evicting the encoder lowers the *pipeline's* generate peak by 2.83 GB, not the encoder's full
# 3.97 GB: 17.78 resident vs 14.95 GB evicted at 768², the max over the 28 images of the M1 exit
# runs (§8 2026-10-15). M0b's 13.82 GB came from a bare generate call, not the pipeline path.
EVICTED_SAVING_GB = 2.83

# Rewrite and rerank: weights only (§4) until M2/M3 measure their pipeline peaks. Guard: the
# M0b smoke test's peak, 1.62 GiB, for load + one short generation (§8 2026-10-15).
STATIC_GB = {"guard": 1.74, "rewrite": 6.5, "rerank": 2.5}


def generate_gb(width: int, height: int, encoder_resident: bool = True) -> float:
    mpx = width * height / 1e6
    points = GENERATE_PEAKS_GB
    segment = next((i for i in range(1, len(points) - 1) if mpx <= points[i][0]), len(points) - 1)
    (x0, y0), (x1, y1) = points[segment - 1], points[segment]
    gb = y0 + (y1 - y0) * (mpx - x0) / (x1 - x0)
    return gb if encoder_resident else gb - EVICTED_SAVING_GB


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
        # WARNING: the caller logs the resulting error once, as ERROR with its traceback.
        logs.event(logging.WARNING, stage, "preflight", "refused: over the memory limit", **context)
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
