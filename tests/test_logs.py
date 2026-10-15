import logging
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from qwi import logs, paths


@pytest.fixture(autouse=True)
def clean_logger(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.delenv("QWI_LOG_LEVEL", raising=False)
    before = list(logs.LOGGER.handlers)
    yield
    for handler in logs.LOGGER.handlers[:]:
        if handler not in before:
            logs.LOGGER.removeHandler(handler)
            handler.close()


def test_run_log_writes_json_lines_with_context() -> None:
    run = paths.new_run_dir()
    with logs.run_log(run) as path:
        logs.event(logging.INFO, "generate", "candidate", "seed done", seed=7, seconds=125.8)
        logs.event(logging.WARNING, "generate", "preflight", "swap risk", est_gb=17.8)
    lines = logs.read(path)
    assert [line["event"] for line in lines] == ["candidate", "preflight"]
    first = lines[0]
    assert {k: first[k] for k in ("level", "stage", "msg", "seed", "seconds")} == {
        "level": "INFO",
        "stage": "generate",
        "msg": "seed done",
        "seed": 7,
        "seconds": 125.8,
    }
    assert first["ts"][:2] == "20"


def test_exception_lines_carry_the_traceback() -> None:
    run = paths.new_run_dir()
    with logs.run_log(run) as path:
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            logs.exception("generate", "error", "unexpected error")
    (line,) = logs.read(path)
    assert line["level"] == "ERROR"
    assert "RuntimeError: boom" in line["traceback"]


def test_debug_is_dropped_by_default_and_kept_with_verbose() -> None:
    quiet, loud = paths.new_run_dir(), paths.new_run_dir()
    with logs.run_log(quiet) as path:
        logs.event(logging.DEBUG, None, "evict", "freed")
    assert logs.read(path) == []
    with logs.run_log(loud, verbose=True) as path:
        logs.event(logging.DEBUG, None, "evict", "freed")
    assert len(logs.read(path)) == 1


def test_read_filters_by_minimum_level() -> None:
    run = paths.new_run_dir()
    with logs.run_log(run) as path:
        logs.event(logging.INFO, None, "a", "info")
        logs.event(logging.WARNING, None, "b", "warning")
        logs.event(logging.ERROR, None, "c", "error")
    assert [line["event"] for line in logs.read(path, "warning")] == ["b", "c"]


def test_run_log_detaches_when_the_run_ends() -> None:
    run = paths.new_run_dir()
    with logs.run_log(run) as path:
        pass
    logs.event(logging.ERROR, None, "late", "after the run")
    assert logs.read(path) == []


def test_level_from_env_verbose_and_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    assert logs.level() == logging.INFO
    monkeypatch.setenv("QWI_LOG_LEVEL", "debug")
    assert logs.level() == logging.DEBUG
    monkeypatch.setenv("QWI_LOG_LEVEL", "chatty")
    assert logs.level() == logging.INFO
    assert logs.level(verbose=True) == logging.DEBUG


def test_app_log_rotates_and_attaches_once(qwi_home: Path) -> None:
    path = logs.app_log()
    assert path == qwi_home / "logs" / "qwi.log"
    assert logs.app_log() == path
    rotating = [h for h in logs.LOGGER.handlers if isinstance(h, RotatingFileHandler)]
    assert len(rotating) == 1
    assert (rotating[0].maxBytes, rotating[0].backupCount) == (5 * 2**20, 4)


def test_nothing_reaches_stderr_without_a_log_file(capsys: pytest.CaptureFixture[str]) -> None:
    logs.event(logging.WARNING, "generate", "preflight", "swap risk")
    assert capsys.readouterr().err == ""
