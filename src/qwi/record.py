"""The run record, `run.json` (PLAN.md §5 `RunRecord`, §5.11 "Crash-safe run records").

`write()` is atomic (tmp → fsync → `os.replace`), so a crash or a full disk never leaves a
half-written record. A run starts as `running` and is rewritten at each stage boundary and at the
end; `repair_if_interrupted()` turns a `running` record whose process is gone into `interrupted`.
"""

import errno
import json
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

from qwi import errors, models, paths

RUN_JSON = "run.json"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class RunRecord:
    user_prompt: str
    status: str = "running"  # running | ok | blocked | cancelled | error | interrupted
    guard: dict[str, Any] = field(default_factory=dict)  # §5.2
    rewritten_prompt: str = ""
    rewrite_fallback_reason: str | None = None
    prompt_source: str = "user"  # user | rewrite | edited_rewrite
    rerank_query: str = ""  # defaults to user_prompt (§5.3)
    seeds: list[int] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)  # §5.3; paths relative
    rerank: dict[str, Any] = field(default_factory=dict)  # §5.3
    best_index: int | None = None  # None until rerank picks one
    error: dict[str, Any] | None = None  # {type, message, stage, log} when status == "error"
    preflight: dict[str, Any] = field(default_factory=dict)  # {stage: {est_gb, ...}}
    pid: int = field(default_factory=os.getpid)
    started_at: str = field(default_factory=_now)
    model_revisions: dict[str, str] = field(default_factory=dict)  # role → repo@revision
    starred: bool = False
    timings_s: dict[str, float] = field(default_factory=dict)
    peak_mem_gb: dict[str, float] = field(default_factory=dict)
    swap_gb: dict[str, float] = field(default_factory=dict)  # system swap-out delta per stage
    config: dict[str, Any] = field(default_factory=dict)  # steps, size, n, encoder, ...
    continued_from: str | None = None  # --continue-seeds: the source run's id (§5.3)

    def __post_init__(self) -> None:
        self.rerank_query = self.rerank_query or self.user_prompt

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def model_revisions(roles: list[str]) -> dict[str, str]:
    return {role: models.spec(role).pinned for role in roles}


def write(record: RunRecord, run: Path) -> Path:
    target = run / RUN_JSON
    tmp = run / f"{RUN_JSON}.tmp"
    text = json.dumps(asdict(record), indent=2, ensure_ascii=False) + "\n"
    try:
        with tmp.open("w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(target)
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            tmp.unlink(missing_ok=True)
            raise errors.DiskFull(paths.free_bytes()) from exc
        raise
    return target


def read(run: Path) -> RunRecord:
    return RunRecord.from_dict(json.loads((run / RUN_JSON).read_text(encoding="utf-8")))


def finish(record: RunRecord, exc: BaseException | None = None, stage: str | None = None) -> None:
    """Set the final status from the outcome; `error` is filled only for status `error`."""
    if exc is None:
        record.status = "ok"
        return
    record.status = errors.status(exc) or "error"
    if record.status == "error":
        record.error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "stage": stage,
            "log": "run.log",
        }


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by someone else
        return True
    return True


def repair_if_interrupted(run: Path, lock_held: bool) -> RunRecord:
    """Mark a `running` record `interrupted` (on disk too) if its run can't still be going.

    Runs hold `$QWI_HOME/.lock` for their whole lifetime, so a free lock or a dead pid means the
    run died (force-quit, panic, closed terminal). Either is enough: a free lock also catches a
    dead run whose pid was reused (§8 2026-10-15 interrupted rule).
    """
    record = read(run)
    if record.status == "running" and not (lock_held and pid_alive(record.pid)):
        record.status = "interrupted"
        write(record, run)
    return record
