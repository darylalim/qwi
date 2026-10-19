"""The History page with AppTest, on real run folders written by the fake pipeline (§5.6)."""

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from qwi import errors, lock, logs, paths, record
from qwi.pipeline import GenerateRequest
from qwi.record import RunRecord
from qwi.ui import jobs
from qwi.ui.fake import FakePipeline
from tests.test_history import FINISHED, dead_pid, write_run

APP = str(Path(__file__).resolve().parents[1] / "src" / "qwi" / "ui" / "streamlit_app.py")
PAGE = "app_pages/history.py"
TIMEOUT = 10


@pytest.fixture(autouse=True)
def fake_jobs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(jobs.PIPELINE_ENV, "fake")
    monkeypatch.setenv(jobs.SYNC_ENV, "1")
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def fake_run(pipeline: FakePipeline | None = None, prompt: str = "a red apple") -> Path:
    """A finished run folder, as the fake pipeline writes it; its path."""
    events: list = []
    with contextlib.suppress(errors.QwiError):  # a failed run still leaves its folder
        (pipeline or FakePipeline()).run(GenerateRequest(prompt, n=4), events.append)
    return events[0].run_dir


def history(run: Path | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    if run is not None:
        at.query_params["run"] = run.name
    at.run()
    at.switch_page(PAGE).run()
    assert not at.exception
    return at


def table(at: AppTest) -> list[dict]:
    return at.dataframe[0].value.to_dict("records")


def test_no_runs_yet() -> None:
    at = history()
    assert [i.title for i in at.info] == ["No runs yet"]
    assert not at.dataframe


def test_the_table_lists_every_run_and_asks_for_a_selection() -> None:
    fake_run(prompt="first")
    fake_run(prompt="second")
    at = history()
    rows = table(at)
    assert sorted(r["Prompt"] for r in rows) == ["first", "second"]
    assert {r["Status"] for r in rows} == {"ok"}
    assert all(r["Best prob"] == pytest.approx(0.9) for r in rows)
    assert "Select a run" in at.main.caption[-1].value
    assert not at.get("image")


def test_a_selected_run_shows_its_images_prompts_stages_and_log() -> None:
    run = fake_run()
    at = history(run)
    assert at.subheader[0].value == run.name
    assert ":green-badge[ok]" in [m.value for m in at.markdown]
    images = list(at.get("image"))
    assert len(images) == 4
    assert images[0].proto.imgs[0].alt == "a red apple (seed 1000)"
    assert at.code[0].value == "a red apple"
    assert [e.label for e in at.expander] == ["Rewritten prompt", "Log"]
    log = at.code[-1].value
    assert "INFO" in log
    assert "DEBUG" not in log


def test_the_detail_shows_guard_config_and_a_stage_table() -> None:
    at = history(write_run(**FINISHED))
    captions = [c.value for c in at.main.caption]
    assert "Guard · Decision: warn (default mode): Controversial input" in captions
    assert "768x768, 20 steps, N=2, encoder A-q4, seeds 7, 8" in captions
    assert [r["Stage"] for r in at.dataframe[1].value.to_dict("records")] == [
        "Guard",
        "Generate",
        "Rerank",
    ]
    assert len(at.get("image")) == 0  # the record names candidates that were never written
    assert "cand_0_seed7.png is missing" in captions


def test_the_log_filters_by_level() -> None:
    at = history(fake_run())
    at.button_group(key="hist_log_level").set_value("ERROR").run()
    assert "No log lines at this level." in [c.value for c in at.caption]


def test_the_log_is_the_runs_own_not_the_app_log() -> None:
    logs.app_log()  # as in the UI and the CLI: it copies every run's lines too (§9)
    first = fake_run(prompt="first")
    fake_run(prompt="second")
    at = history(first)
    shown = at.code[-1].value.splitlines()
    assert len(shown) == len(logs.read(first / "run.log", "INFO"))
    assert len(logs.read(paths.logs_dir() / "qwi.log", "INFO")) > len(shown)


def test_star_toggles_run_json_and_the_table() -> None:
    run = fake_run()
    at = history(run)
    at.button(key="hist_star").click().run()
    assert not at.exception
    assert record.read(run).starred
    assert table(at)[0]["Starred"] is True
    assert at.button(key="hist_star").label == "Unstar"
    at.button(key="hist_star").click().run()
    assert not record.read(run).starred


def test_a_killed_run_shows_as_interrupted() -> None:
    run = paths.new_run_dir()
    record.write(RunRecord("a red apple", pid=dead_pid()), run)
    at = history(run)
    assert table(at)[0]["Status"] == "interrupted"
    assert [w.title for w in at.warning] == ["Interrupted"]
    assert record.read(run).status == "interrupted"


def test_a_live_run_shows_as_running_and_cannot_be_starred() -> None:
    run = paths.new_run_dir()
    record.write(RunRecord("a red apple", pid=os.getpid()), run)
    with lock.RunLock():
        at = history(run)
    assert table(at)[0]["Status"] == "running"
    assert [i.title for i in at.info] == ["Running"]
    assert at.button(key="hist_star").disabled


def test_an_error_run_shows_its_error() -> None:
    run = fake_run(FakePipeline(fails=errors.DeviceOutOfMemory("generate", "test")))
    at = history(run)
    assert [e.title for e in at.error] == ["Error in generate"]
    assert "DeviceOutOfMemory" in at.error[0].value


def test_an_unknown_run_in_the_link_selects_nothing() -> None:
    fake_run()
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["run"] = "20000101-000000-gone00"
    at.run()
    at.switch_page(PAGE).run()
    assert not at.exception
    assert "Select a run" in at.main.caption[-1].value
