"""One pipeline run at a time, across processes (PLAN.md §5.6): `flock` on `$QWI_HOME/.lock`.

The CLI and the UI resolve the same path through `qwi.paths`, so they exclude each other. The OS
drops the lock when its process dies, so stale locks can't happen (§5.11).
"""

import fcntl
import json
import os
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import IO, Self

from qwi import paths
from qwi.errors import PipelineBusy


def _try_flock(f: IO[str]) -> bool:
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _started_at(holder: str) -> str | None:
    try:
        return json.loads(holder).get("started_at")
    except ValueError:  # empty, or caught mid-write by the holder
        return None


class RunLock:
    """`with RunLock():` holds the lock for the block, or raises `PipelineBusy` at once."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or paths.lock_path()
        self._file: IO[str] | None = None

    def __enter__(self) -> Self:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        f = self.path.open("a+", encoding="utf-8")
        if not _try_flock(f):
            f.seek(0)
            holder = f.read()
            f.close()
            raise PipelineBusy(_started_at(holder))
        f.seek(0)
        f.truncate()
        f.write(
            json.dumps({"pid": os.getpid(), "started_at": datetime.now().isoformat("T", "seconds")})
        )
        f.flush()
        self._file = f
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._file is not None:
            self._file.close()  # closing the descriptor releases the flock
            self._file = None


def is_held(path: Path | None = None) -> bool:
    """Whether some run holds the lock now, in this process or another (for interrupted repair)."""
    path = path or paths.lock_path()
    if not path.exists():
        return False
    with path.open("a+", encoding="utf-8") as f:
        if _try_flock(f):
            fcntl.flock(f, fcntl.LOCK_UN)
            return False
        return True
