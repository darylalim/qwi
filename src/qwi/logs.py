"""JSON-lines logs: `run.log` per run and a rotating app log (PLAN.md §5.11 "Logs").

Log through `event()`, so every line has `ts`, `level`, `stage`, `event` and `msg`, plus any
context (`seed`, `step`, `est_gb`, ...). An ERROR logged with `exc_info` carries `traceback`.
"""

import json
import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from qwi import paths
from qwi.config import DEFAULTS

LOGGER = logging.getLogger("qwi")
# qwi's logs go only to its files: no propagation, and a NullHandler so logging's last-resort
# handler never prints them to stderr when no log file is attached.
# The logger passes everything; each file handler filters at its own level.
LOGGER.propagate = False
LOGGER.addHandler(logging.NullHandler())
LOGGER.setLevel(logging.DEBUG)

APP_LOG_BYTES = 5 * 2**20
APP_LOG_FILES = 5  # qwi.log + 4 rotated (5 x 5 MB)

_RECORD_FIELDS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonLines(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created)
            .astimezone()
            .isoformat("T", "milliseconds"),
            "level": record.levelname,
            "stage": getattr(record, "stage", None),
            "event": getattr(record, "event", None),
            "msg": record.getMessage(),
        }
        line |= {k: v for k, v in vars(record).items() if k not in _RECORD_FIELDS | line.keys()}
        if record.exc_info:
            line["traceback"] = self.formatException(record.exc_info)
        return json.dumps(line, ensure_ascii=False, default=str)


def level(verbose: bool = False) -> int:
    """DEBUG with `-v`, else `QWI_LOG_LEVEL`, else the config default; unknown names → INFO."""
    if verbose:
        return logging.DEBUG
    name = os.environ.get("QWI_LOG_LEVEL") or DEFAULTS.log_level
    return logging.getLevelNamesMapping().get(name.upper(), logging.INFO)


def event(lvl: int, stage: str | None, name: str, msg: str, **context: Any) -> None:
    LOGGER.log(lvl, msg, extra={"stage": stage, "event": name, **context})


def exception(stage: str | None, name: str, msg: str, **context: Any) -> None:
    """An ERROR line with the current exception's full traceback (call it in `except`)."""
    LOGGER.error(msg, exc_info=True, extra={"stage": stage, "event": name, **context})


def _attach(handler: logging.Handler, lvl: int) -> None:
    handler.setFormatter(JsonLines())
    handler.setLevel(lvl)
    LOGGER.addHandler(handler)


@contextmanager
def run_log(run: Path, verbose: bool = False) -> Generator[Path]:
    """Write `run/run.log` for the duration of the block (one handler per run)."""
    path = run / "run.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    _attach(handler, level(verbose))
    try:
        yield path
    finally:
        LOGGER.removeHandler(handler)
        handler.close()


def app_log(verbose: bool = False) -> Path:
    """Attach `$QWI_HOME/logs/qwi.log` once per process, for anything outside a run."""
    path = paths.logs_dir() / "qwi.log"
    for handler in LOGGER.handlers:
        if isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename) == path:
            return path
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        path, maxBytes=APP_LOG_BYTES, backupCount=APP_LOG_FILES - 1, encoding="utf-8"
    )
    _attach(handler, level(verbose))
    return path


def read(path: Path, min_level: str = "DEBUG") -> list[dict[str, Any]]:
    """The log's lines at `min_level` or above, for `qwi runs show --log --level ...`."""
    floor = logging.getLevelNamesMapping()[min_level.upper()]
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [line for line in lines if logging.getLevelNamesMapping()[line["level"]] >= floor]
