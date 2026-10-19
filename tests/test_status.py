"""The sidebar's doctor line: which checks it runs, its summary, and how the app shows it."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from qwi import storage
from qwi.storage import Check
from qwi.ui import jobs, status

APP = str(Path(__file__).resolve().parents[1] / "src" / "qwi" / "ui" / "streamlit_app.py")

OK = Check("disk", "pass", "46.8 GB free")
WARN = Check("torch", "warn", "installed (transformers will import it): `uv sync`")
FAIL = Check("model image", "fail", "missing: run `qwi models fetch image`")


def test_all_passing_is_ok_and_green() -> None:
    line = status.summarize([OK, OK])
    assert (line.status, line.label, line.color, line.problems) == (
        "pass",
        "Doctor: ok",
        "green",
        [],
    )


def test_warnings_are_counted_and_orange() -> None:
    line = status.summarize([OK, WARN, WARN])
    assert (line.status, line.label, line.color) == ("warn", "Doctor: 2 warnings", "orange")


def test_a_failure_wins_and_is_listed_first() -> None:
    line = status.summarize([WARN, OK, FAIL])
    assert (line.status, line.label, line.color) == ("fail", "Doctor: 1 failure", "red")
    assert line.problems == [
        "model image: missing: run `qwi models fetch image`",
        "torch: installed (transformers will import it): `uv sync`",
    ]


def test_quick_checks_skip_the_slow_scans(monkeypatch: pytest.MonkeyPatch) -> None:
    def slow() -> Check:
        raise AssertionError("the sidebar must not walk $QWI_HOME")

    for name in ["check_runs_size", "check_eval_size", "check_run_index"]:
        monkeypatch.setattr(storage, name, slow)
    monkeypatch.setattr(storage, "check_models", lambda: [Check("model image", "pass", "x")])
    names = [c.name for c in status.quick_checks()]
    assert names == ["disk", "model image", "QWI_HOME writable", "HF_HUB_OFFLINE", "torch"]


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(jobs.PIPELINE_ENV, "fake")
    monkeypatch.setenv(jobs.SYNC_ENV, "1")
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


def sidebar(monkeypatch: pytest.MonkeyPatch, checks: list[Check]) -> AppTest:
    monkeypatch.setattr(status, "quick_checks", lambda: checks)
    at = AppTest.from_file(APP, default_timeout=10).run()
    assert not at.exception
    return at


@pytest.mark.usefixtures("app")
def test_sidebar_shows_a_green_badge_when_all_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    at = sidebar(monkeypatch, [OK])
    badges = [m.value for m in at.sidebar.markdown if "-badge[" in m.value]
    assert len(badges) == 1
    assert badges[0].startswith(":green-badge[")
    assert "Doctor: ok" in badges[0]
    assert not [c.value for c in at.sidebar.caption if "qwi doctor" in c.value]


@pytest.mark.usefixtures("app")
def test_sidebar_names_the_problems_on_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    at = sidebar(monkeypatch, [OK, FAIL])
    badges = [m.value for m in at.sidebar.markdown if "-badge[" in m.value]
    assert badges[0].startswith(":red-badge[")
    assert "Doctor: 1 failure" in badges[0]
    captions = [c.value for c in at.sidebar.caption]
    assert any("model image: missing" in c and "Run `qwi doctor`" in c for c in captions)


@pytest.mark.usefixtures("app")
def test_sidebar_line_is_cached_between_reruns(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def counted() -> list[Check]:
        calls.append(1)
        return [OK]

    monkeypatch.setattr(status, "quick_checks", counted)
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.run()
    assert (not at.exception, len(calls)) == (True, 1)
