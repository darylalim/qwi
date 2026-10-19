"""Typed pipeline errors (PLAN.md §5.11 "Error types").

Each class carries the `RunRecord.status` it produces and the CLI exit code. Anything that isn't
a `QwiError` after `classify()` is an unexpected error: status `error`, exit code 1.
"""

import errno
from typing import ClassVar

from qwi import paths

UNEXPECTED_EXIT = 1

# Message fragments of MLX allocation failures, pinned against tests/fixtures/mlx_errors.json.
OOM_PATTERNS = (
    "[metal::malloc]",
    "[malloc] Unable to allocate",
    "[METAL] Command buffer execution failed",
)
_SMALLER = "close other apps, or use 512² or fewer candidates"
_RUNS = "$QWI_HOME/runs"


class QwiError(Exception):
    """Base class for every error the pipeline raises on purpose."""

    status: ClassVar[str | None] = "error"  # None: no run record is created
    exit_code: ClassVar[int] = UNEXPECTED_EXIT


class GuardBlocked(QwiError):
    """The guard's decision is `block` (§5.2). Not a failure: the run ends as `blocked`."""

    status = "blocked"
    exit_code = 2

    def __init__(self, detail: str) -> None:
        super().__init__(f"Blocked by guard: {detail}")
        self.detail = detail


class PipelineBusy(QwiError):
    """`$QWI_HOME/.lock` is held by another run."""

    status = None
    exit_code = 3

    def __init__(self, started_at: str | None = None) -> None:
        since = f" (started {started_at})" if started_at else ""
        super().__init__(f"Another run is in progress{since}")
        self.started_at = started_at


class InsufficientMemory(QwiError):
    """The pre-flight estimate for a stage exceeds the wired memory limit: it can't run at all."""

    exit_code = 4

    def __init__(self, stage: str, need_gb: float, have_gb: float) -> None:
        super().__init__(
            f"Not enough free memory for {stage}: needs {need_gb:.1f} GB, "
            f"{have_gb:.1f} GB within the GPU memory limit; {_SMALLER}"
        )
        self.stage = stage
        self.need_gb = need_gb
        self.have_gb = have_gb


class DeviceOutOfMemory(QwiError):
    """MLX/Metal ran out of memory, or the machine started swapping, during a stage."""

    exit_code = 4

    def __init__(self, stage: str, detail: str) -> None:
        super().__init__(f"Ran out of GPU memory during {stage}: {detail}; {_SMALLER}")
        self.stage = stage
        self.detail = detail


class ModelUnavailable(QwiError):
    """A role's weights are missing at the pinned revision, or failed to load offline."""

    exit_code = 5

    def __init__(self, role: str, detail: str) -> None:
        super().__init__(f"{role} model not available: {detail}. Run `qwi models fetch {role}`.")
        self.role = role
        self.detail = detail


class DiskFull(QwiError):
    """A write failed with ENOSPC."""

    exit_code = 6

    def __init__(self, free_bytes: int) -> None:
        super().__init__(f"Disk full ({free_bytes / 1e9:.1f} GB free). Run `qwi runs prune`.")
        self.free_bytes = free_bytes


class InsufficientDisk(QwiError):
    """Free disk would fall below the 20 GB rule (§5.7): checked before downloads and runs."""

    exit_code = 6

    def __init__(self, what: str, need_bytes: int, free_bytes: int, keep_bytes: int) -> None:
        super().__init__(
            f"Not enough disk for {what}: needs {need_bytes / 1e9:.1f} GB with "
            f"{free_bytes / 1e9:.1f} GB free, and {keep_bytes / 1e9:.0f} GB must stay free. "
            "Free space first (`qwi runs prune`, `hf cache`)."
        )
        self.need_bytes = need_bytes
        self.free_bytes = free_bytes


class RunNotFound(QwiError):
    """No run folder matches the id given to `qwi runs show` (exact name or a unique prefix)."""

    status = None

    def __init__(self, run_id: str, matches: list[str]) -> None:
        detail = f"matches {len(matches)} runs: {', '.join(matches[:5])}" if matches else "no match"
        super().__init__(f"Run {run_id!r}: {detail}. See {_RUNS}.")
        self.run_id = run_id


class RunNotContinuable(QwiError):
    """`qwi generate --continue-seeds` names a run that can't be continued (§5.3)."""

    status = None

    def __init__(self, run_id: str, reason: str) -> None:
        super().__init__(f"Can't continue run {run_id!r}: {reason}.")
        self.run_id = run_id


class RunStillRunning(QwiError):
    """A star change names a run that is still `running`: the run's next write would drop it."""

    status = None

    def __init__(self, run_id: str) -> None:
        super().__init__(f"Run {run_id!r} is still running; star it when it ends.")
        self.run_id = run_id


class EvalSetError(QwiError):
    """`eval/prompts.jsonl` is malformed, or `qwi eval --only` names an unknown prompt id."""

    status = None


class EvalExists(QwiError):
    """The variant already has `results.jsonl`: eval results are never overwritten."""

    status = None


class SessionError(QwiError):
    """A review session can't be created or read (bad results folders, manifest, values)."""

    status = None


class SessionIncomplete(QwiError):
    """`qwi eval compare` refuses to reveal the variants until every check is judged (§5.4)."""

    status = None

    def __init__(self, missing: int, examples: list[str]) -> None:
        super().__init__(
            f"{missing} checks are not judged yet (e.g. {', '.join(examples)}); "
            "variants stay hidden until every check is pass/fail"
        )
        self.missing = missing


class GenerationCancelled(QwiError):
    """Cancel in the UI, or the first Ctrl-C in the CLI."""

    status = "cancelled"
    exit_code = 130

    def __init__(self, done: int | None = None, total: int | None = None) -> None:
        after = f" after {done} of {total} candidates" if done is not None and total else ""
        super().__init__(f"Cancelled{after}")
        self.done = done
        self.total = total


def classify(exc: BaseException, stage: str) -> BaseException:
    """Map a known library failure to its `QwiError`; return anything else unchanged.

    Unknown `RuntimeError`s stay unexpected errors: never guess (§5.11).
    """
    if isinstance(exc, QwiError):
        return exc
    if isinstance(exc, KeyboardInterrupt):  # outside mflux's step loop; inside, see below
        return GenerationCancelled()
    if isinstance(exc, RuntimeError) and any(p in str(exc) for p in OOM_PATTERNS):
        return DeviceOutOfMemory(stage, str(exc))
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return DiskFull(paths.free_bytes())
    # mflux turns a KeyboardInterrupt inside its step loop into this (§2); matched by name so
    # classifying never imports mflux.
    kind = type(exc)
    if kind.__name__ == "StopImageGenerationException" and kind.__module__.startswith("mflux."):
        return GenerationCancelled()
    return exc


def status(exc: BaseException) -> str | None:
    return exc.status if isinstance(exc, QwiError) else "error"


def exit_code(exc: BaseException) -> int:
    return exc.exit_code if isinstance(exc, QwiError) else UNEXPECTED_EXIT


def exit_code_for(status: str, error_type: str | None = None) -> int:
    """The CLI exit code for a finished `RunRecord` (`status`, `error["type"]`)."""
    if status == "ok":
        return 0
    if status == "error":
        kind = _by_name().get(error_type or "")
        return kind.exit_code if kind else UNEXPECTED_EXIT
    by_status = {cls.status: cls.exit_code for cls in (GuardBlocked, GenerationCancelled)}
    return by_status.get(status, UNEXPECTED_EXIT)


def _by_name(cls: type[QwiError] = QwiError) -> dict[str, type[QwiError]]:
    found = {cls.__name__: cls}
    for sub in cls.__subclasses__():
        found |= _by_name(sub)
    return found
