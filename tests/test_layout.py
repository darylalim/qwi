"""The responsive layout with AppTest (PLAN.md §5.6 "Tests"): the sidebar's Thumbnail size and
Stacked layout controls, the panes they switch, and the shared candidate cards.

AppTest doesn't measure layout, and its elements keep only their own proto, not the width and
height config around it. So card widths are checked through a spy on `candidate_card`, the
one-column flow through the absence of columns, and the URL only in the reading direction: a
bound widget's write to the URL happens in the browser, which AppTest doesn't run.
"""

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from qwi import sessions
from qwi.ui import components, jobs
from tests.test_eval_review_page import variant
from tests.test_history_page import fake_run

APP = str(Path(__file__).resolve().parents[1] / "src" / "qwi" / "ui" / "streamlit_app.py")
TIMEOUT = 10


@pytest.fixture(autouse=True)
def fake_jobs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(jobs.PIPELINE_ENV, "fake")
    monkeypatch.setenv(jobs.SYNC_ENV, "1")
    monkeypatch.delenv(jobs.FAKE_SCRIPT_ENV, raising=False)
    monkeypatch.delenv(sessions.REVIEW_SESSION_ENV, raising=False)
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def widths(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The width each `candidate_card` call got."""
    seen: list[int] = []
    card = components.candidate_card

    def spy(c, width: int, **kwargs) -> None:
        seen.append(width)
        card(c, width, **kwargs)

    monkeypatch.setattr(components, "candidate_card", spy)
    return seen


def app(**params: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    for key, value in params.items():
        at.query_params[key] = value
    at.run()
    assert not at.exception
    return at


def generate(at: AppTest, prompt: str = "a red apple") -> AppTest:
    at.text_area(key="gen_prompt").set_value(prompt)
    at.button(key="FormSubmitter:generate-Generate").click().run()
    assert not at.exception
    return at


def test_every_page_has_the_layout_controls() -> None:
    at = app()
    for page in (None, "app_pages/history.py", "app_pages/eval_review.py"):
        if page:
            at.switch_page(page).run()
        assert [s.label for s in at.sidebar.segmented_control] == ["Thumbnail size"]
        assert [t.label for t in at.sidebar.toggle] == ["Stacked layout"]
    assert at.sidebar.segmented_control[0].value == "M"
    assert not at.sidebar.toggle[0].value


@pytest.mark.parametrize(("size", "width"), [("S", 256), ("M", 384), ("L", 512)])
def test_the_thumbnail_size_in_the_link_sets_the_card_width(
    size: str, width: int, widths: list[int]
) -> None:
    generate(app(thumb=size))
    assert widths == [width] * 4


def test_an_unknown_thumbnail_size_falls_back_to_m(widths: list[int]) -> None:
    generate(app(thumb="XL"))
    assert widths == [384] * 4


def test_changing_the_thumbnail_size_resizes_the_cards(
    widths: list[int],
) -> None:
    at = generate(app())
    widths.clear()
    at.sidebar.segmented_control[0].set_value("L").run()
    assert widths == [512] * 4


def test_history_cards_follow_the_thumbnail_size(widths: list[int]) -> None:
    run = fake_run()
    at = app(thumb="S", run=run.name)
    at.switch_page("app_pages/history.py").run()
    assert not at.exception
    assert widths == [256] * 4
    assert len(at.get("download_button")) == 4


def test_generate_has_two_panes_with_the_form_first() -> None:
    at = generate(app())
    assert len(at.columns) == 2
    form, results = at.columns
    assert form.text_area(key="gen_prompt")
    assert not form.get("image")
    assert len(results.get("image")) == 4


def test_stacked_layout_puts_generate_in_one_column() -> None:
    at = app()
    at.sidebar.toggle[0].set_value(True).run()
    generate(at)
    assert not at.columns
    order = [n.type for n in at.main if n.type in ("text_area", "status", "image")]
    assert order[0] == "text_area"
    assert order.index("status") < order.index("image")  # form → status → grid


def test_the_status_and_cancel_sit_under_the_form() -> None:
    at = generate(app())
    left = at.columns[0]
    assert len(left.get("status")) == 5
    assert not at.columns[1].get("status")


def test_history_puts_the_table_beside_the_detail_unless_stacked() -> None:
    run = fake_run()
    at = app(run=run.name)
    at.switch_page("app_pages/history.py").run()
    table, detail = at.columns
    assert table.dataframe
    assert not table.get("image")
    assert len(detail.get("image")) == 4
    at.sidebar.toggle[0].set_value(True).run()
    assert not at.columns
    order = [n.type for n in at.main if n.type in ("dataframe", "image")]
    assert order[0] == "dataframe"  # table first, then the run's images


def test_paired_review_has_three_panes_unless_stacked() -> None:
    session = sessions.create([variant("alpha")], [variant("beta")], seed=7, day=date(2026, 10, 15))
    at = app(review=f"session/{session.name}")
    at.switch_page("app_pages/eval_review.py").run()
    assert not at.exception
    assert len(at.columns) == 3
    at.sidebar.toggle[0].set_value(True).run()
    assert not at.columns
    assert len(at.get("image")) == 2


def test_view_opens_the_full_size_image() -> None:
    at = generate(app())
    at.button(key=f"view_{_first_image(at)}").click().run()
    assert not at.exception
    images = at.get("image")
    assert len(images) == 5  # the four cards and the dialog's
    assert images[-1].proto.imgs[0].alt == "a red apple (seed 1000)"


def test_the_rewrite_is_collapsed_and_wrapped() -> None:
    at = generate(app())
    (rewrite,) = at.expander
    assert rewrite.label == "Rewritten prompt"
    assert rewrite.proto.expanded is False
    assert rewrite.code[0].proto.wrap_lines


def _first_image(at: AppTest) -> str:
    key = next(b.key for b in at.button if b.key and b.key.startswith("view_"))
    return key.removeprefix("view_")
