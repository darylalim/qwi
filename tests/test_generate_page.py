"""The Generate page with AppTest and the fake pipeline (PLAN.md §5.6 "Tests")."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from qwi import errors, lock
from qwi.ui import components, jobs
from qwi.ui.fake import FakePipeline, Hold

APP = str(Path(__file__).resolve().parents[1] / "src" / "qwi" / "ui" / "streamlit_app.py")
TIMEOUT = 10


@pytest.fixture(autouse=True)
def fake_jobs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A fresh shared `JobManager` per test, running the fake pipeline inline."""
    monkeypatch.setenv(jobs.PIPELINE_ENV, "fake")
    monkeypatch.setenv(jobs.SYNC_ENV, "1")
    monkeypatch.delenv(jobs.FAKE_SCRIPT_ENV, raising=False)
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception
    return at


def generate(at: AppTest, prompt: str = "a red apple") -> AppTest:
    at.text_area(key="gen_prompt").set_value(prompt)
    at.form_submit_button[0].click().run()
    assert not at.exception
    return at


def images(at: AppTest) -> list:
    return list(at.get("image"))


def badges(at: AppTest) -> list[str]:
    return [m.value for m in at.markdown if "badge[" in m.value]


def test_the_page_opens_with_an_empty_result_area() -> None:
    at = app()
    assert "Results appear here." in [c.value for c in at.caption]
    assert not at.form_submit_button[0].disabled
    assert not images(at)


def test_a_run_shows_every_candidate_with_alt_text_one_best_badge_and_downloads() -> None:
    at = generate(app())
    shown = images(at)
    assert len(shown) == 4
    assert [i.proto.imgs[0].alt for i in shown] == [
        f"a red apple (seed {s})" for s in range(1000, 1004)
    ]
    assert badges(at) == [":violet-badge[:material/star: Best]"]
    assert len(at.get("download_button")) == 4
    assert [s.proto.label for s in at.get("status")] == [
        f"{name}: 0.0 s, peak 1.0 GB"
        for name in ("Guard", "Rewrite", "Encode", "Generate", "Rerank")
    ]
    assert [e.label for e in at.expander] == ["Rewritten prompt"]
    assert not lock.is_held()


def test_the_form_values_reach_the_request() -> None:
    at = app()
    at.number_input(key="gen_n").set_value(2)
    at.number_input(key="gen_steps").set_value(3)
    at.button_group(key="gen_size").set_value(512)
    at.number_input(key="gen_seed").set_value(5)
    at.button_group(key="gen_guard").set_value("strict")
    at.toggle(key="gen_rewrite").set_value(False)
    generate(at, "  a blue cube  ")

    snap = components.job_manager().snapshot()
    assert snap is not None
    r = snap.request
    assert (r.prompt, r.n, r.steps, r.width, r.height, r.seed) == ("a blue cube", 2, 3, 512, 512, 5)
    assert (r.guard_mode, r.rewrite) == ("strict", False)
    assert len(images(at)) == 2
    assert not at.expander  # no rewrite


def test_an_empty_prompt_is_refused_before_any_run() -> None:
    at = generate(app(), "   ")
    assert [w.value for w in at.warning] == ["Enter a prompt first."]
    assert components.job_manager().snapshot() is None


@pytest.mark.parametrize(
    ("script", "title"),
    [("warn", "Guard warning"), ("low", "No candidate clearly matched the prompt")],
)
def test_warnings_keep_the_results(
    script: str, title: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(jobs.FAKE_SCRIPT_ENV, script)
    at = generate(app())
    assert title in [w.title for w in at.warning]
    assert len(images(at)) == 4


def test_a_guard_block_shows_the_error_and_no_images(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(jobs.FAKE_SCRIPT_ENV, "block")
    at = generate(app())
    (error,) = at.error
    assert error.title == "Blocked by guard"
    assert "Unsafe (Violent)" in error.value
    assert not images(at)


def test_while_a_job_runs_the_form_is_disabled_and_cancel_stops_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hold = Hold(candidate=1, step=2)
    monkeypatch.delenv(jobs.SYNC_ENV)
    monkeypatch.setattr(jobs, "make_pipeline", lambda kind, script=None: FakePipeline(hold=hold))
    at = generate(app())
    assert hold.reached.wait(TIMEOUT)
    at.run()

    assert at.form_submit_button[0].disabled
    (bar,) = at.get("progress")
    assert bar.proto.text == "Candidate 2 of 4, step 2 of 20"
    next(b for b in at.button if b.label == "Cancel").click().run()
    assert next(b for b in at.button if b.label == "Cancelling…").disabled

    hold.release()
    assert components.job_manager().wait(TIMEOUT)
    at.run()
    assert not at.exception
    assert not at.form_submit_button[0].disabled
    assert [i.title for i in at.info] == ["Cancelled"]
    assert "Cancelled after 1 of 4 candidates." in at.info[0].value
    assert [c.value for c in at.caption if c.value.startswith("Seed")] == ["Seed 1000 · not ranked"]


def test_a_pipeline_that_raises_shows_an_unexpected_error_and_its_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boom = FakePipeline(raises=RuntimeError("boom"))
    monkeypatch.setattr(jobs, "make_pipeline", lambda kind, script=None: boom)
    at = generate(app())
    (error,) = at.error
    assert (error.title, error.value) == (
        "Unexpected error",
        "RuntimeError: boom. Traceback below.",
    )
    assert [e.label for e in at.expander][-2:] == ["Traceback", "View log"]
    assert not retry_buttons(at)
    assert not lock.is_held()


def retry_buttons(at: AppTest) -> list[str]:
    return [b.label for b in at.button if b.label.startswith("Retry")]


@pytest.mark.parametrize(
    ("error", "title", "retries"),
    [
        (errors.InsufficientMemory("generate", 30.0, 22.9), "Not enough free memory", True),
        (errors.DeviceOutOfMemory("generate", "swap"), "Ran out of GPU memory", True),
        (errors.ModelUnavailable("image", "missing"), "Model not available", False),
        (errors.DiskFull(10**9), "Disk full", False),
        (errors.InsufficientDisk("a run", 0, 10**9, 20 * 10**9), "Not enough disk", False),
        (RuntimeError("unknown"), "Unexpected error", False),
    ],
)
def test_each_error_class_shows_its_banner_actions_and_log(
    error: Exception, title: str, retries: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jobs, "make_pipeline", lambda kind, script=None: FakePipeline(fails=error))
    at = generate(app())
    (banner,) = at.error
    assert banner.title == title
    expected = ["Retry at 512²", "Retry with N=2"] if retries else []
    assert retry_buttons(at) == expected
    log = next(e for e in at.expander if e.label == "View log")
    assert f"{type(error).__name__}: " in log.code[0].value  # the ERROR line's traceback
    assert "Traceback" not in [e.label for e in at.expander]  # it's in the log instead


@pytest.mark.parametrize(
    ("label", "size", "n"), [("Retry at 512²", 512, 4), ("Retry with N=2", 768, 2)]
)
def test_a_retry_resubmits_with_the_cheaper_setting(
    label: str, size: int, n: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakePipeline(fails=errors.DeviceOutOfMemory("generate", "swap"))
    monkeypatch.setattr(jobs, "make_pipeline", lambda kind, script=None: fake)
    at = generate(app())
    fake.fails = None
    next(b for b in at.button if b.label == label).click().run()
    assert not at.exception

    first, second = fake.requests
    assert (second.width, second.height, second.n) == (size, size, n)
    assert (second.prompt, second.seed) == (first.prompt, first.seed)
    assert not at.error
    assert len(images(at)) == n


def test_a_run_held_by_another_process_shows_busy() -> None:
    at = app()
    with lock.RunLock():
        generate(at)
    (banner,) = at.error
    assert banner.title == "Another run is in progress"
    assert not [e.label for e in at.expander if e.label == "View log"]  # no run was created


def test_a_swap_risk_shows_a_notice_and_the_run_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(jobs.FAKE_SCRIPT_ENV, "swap")
    at = generate(app())
    (warning,) = at.warning
    assert warning.title == "Swap risk"
    assert "Generate needs about 15.0 GB and 9.9 GB is free" in warning.value
    assert len(images(at)) == 4


# --- Rewrite, then edit (§5.6) -------------------------------------------------------------


def button(at: AppTest, label: str):
    return next(b for b in at.form_submit_button if b.label == label)


def rewrite_only(at: AppTest, prompt: str = "a red apple") -> AppTest:
    at.text_area(key="gen_prompt").set_value(prompt)
    button(at, "Rewrite only").click().run()
    assert not at.exception
    return at


def edit_area(at: AppTest) -> list:
    return [t for t in at.text_area if t.key == "gen_edit"]


def test_rewrite_only_fills_the_editable_prompt_and_generates_from_the_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakePipeline()
    monkeypatch.setattr(jobs, "make_pipeline", lambda kind, script=None: fake)
    at = rewrite_only(app())
    (area,) = edit_area(at)
    assert area.value == "A detailed photograph of a red apple."
    assert "Guard: input: Safe" in [c.value for c in at.caption]
    assert not images(at)  # nothing generated yet
    assert (len(fake.rewrites), fake.requests) == (1, [])

    area.set_value("  A green apple on slate  ")
    at.number_input(key="gen_n").set_value(2)
    button(at, "Generate from this prompt").click().run()
    assert not at.exception

    (request,) = fake.requests
    assert (request.prompt, request.rewrite, request.n) == ("A green apple on slate", False, 2)
    assert (request.rerank_query, request.prompt_source) == ("a red apple", "edited_rewrite")
    assert len(images(at)) == 2
    assert edit_area(at)[0].value == "  A green apple on slate  "  # kept to try again


def test_clear_discards_the_edit() -> None:
    at = rewrite_only(app())
    button(at, "Clear").click().run()
    assert not edit_area(at)
    at.run()  # the finished rewrite job doesn't fill it again
    assert not edit_area(at)
    assert not at.form_submit_button[0].disabled


def test_a_blocked_rewrite_shows_the_error_and_no_editable_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(jobs.FAKE_SCRIPT_ENV, "block")
    at = rewrite_only(app())
    (error,) = at.error
    assert error.title == "Blocked by guard"
    assert not edit_area(at)


def test_a_guard_warning_shows_above_the_editable_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(jobs.FAKE_SCRIPT_ENV, "warn")
    at = rewrite_only(app())
    assert edit_area(at)
    assert [w.title for w in at.warning] == ["Guard warning", "Guard warning"]  # form + results


def test_rewrite_only_needs_a_prompt() -> None:
    at = rewrite_only(app(), "  ")
    assert [w.value for w in at.warning] == ["Enter a prompt first."]
    assert components.job_manager().snapshot() is None
