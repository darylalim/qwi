"""The Eval review page (PLAN.md §5.6) with AppTest: paired, best-of-N and single-variant modes.

Also the page's two data helpers without Streamlit: `qwi.ui.review` (what can be opened) and
`qwi.variant_review` (a results folder's `review.csv`).
"""

from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from qwi import bestof, evaluation, paths, sessions, variant_review
from qwi.bestof import BestOfJudgment
from qwi.errors import SessionError
from qwi.evaluation import load_prompts
from qwi.sessions import Judgment
from qwi.ui import jobs, review
from qwi.variant_review import VariantJudgment

from .test_bestof import SEEDS, bo4
from .test_review import full, variant

APP = str(Path(__file__).resolve().parents[1] / "src" / "qwi" / "ui" / "streamlit_app.py")
PAGE = "app_pages/eval_review.py"
TIMEOUT = 10


@pytest.fixture(autouse=True)
def fake_jobs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The Generate page renders first without `QWI_REVIEW_SESSION`: keep it model-free."""
    monkeypatch.setenv(jobs.PIPELINE_ENV, "fake")
    monkeypatch.setenv(jobs.SYNC_ENV, "1")
    monkeypatch.delenv(sessions.REVIEW_SESSION_ENV, raising=False)
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def paired() -> Path:
    return sessions.create([variant("alpha")], [variant("beta")], seed=7, day=date(2026, 10, 15))


@pytest.fixture
def best_of() -> Path:
    return bestof.create(bo4(), seed=7, day=date(2026, 10, 17))


@pytest.fixture
def results() -> Path:
    """A results folder with the review sheet `qwi eval` writes (24 images at two seeds)."""
    folder = variant("gamma")
    images = paths.eval_dir() / folder.name
    rows = evaluation.read_rows(folder / "results.jsonl")
    extra: list[dict[str, Any]] = []
    for row in rows:  # the second seed's images, copied from the first
        image = str(row["image"]).replace("1000", "1001")
        (images / image).write_bytes((images / row["image"]).read_bytes())
        extra.append({**row, "seed": 1001, "image": image})
    evaluation.write_rows(folder / "results.jsonl", rows + extra)
    evaluation.write_review(folder / "review.csv", load_prompts(), rows + extra)
    return folder


def launched(session: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    """The UI as `qwi eval review <session>` starts it: on this page, the session picked."""
    monkeypatch.setenv(sessions.REVIEW_SESSION_ENV, str(session))
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception
    return at


def opened(key: str | None = None) -> AppTest:
    """The page reached from `qwi ui`'s navigation, with `?review=<key>` when given."""
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    if key is not None:
        at.query_params["review"] = key
    at.run()
    at.switch_page(PAGE).run()
    assert not at.exception
    return at


def texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    return "\n".join(str(p) for p in parts)


def page_through(at: AppTest, steps: int) -> None:
    for _ in range(steps):
        at.button(key="next").click().run()


def _prompt(prompt_id: str) -> str:
    return {p.id: p.prompt for p in load_prompts()}[prompt_id]


# --- Sources (no Streamlit) ----------------------------------------------------------------


def test_sources_list_sessions_then_results_with_a_review_sheet(
    paired: Path, best_of: Path, results: Path
) -> None:
    variant("delta")  # results without review.csv: nothing to review there
    found = review.sources()
    assert [(s.kind, s.folder.name) for s in found] == [
        ("bestof", best_of.name),
        ("paired", paired.name),
        ("variant", results.name),
    ]
    assert [s.key for s in found] == [
        f"session/{best_of.name}",
        f"session/{paired.name}",
        f"results/{results.name}",
    ]
    assert [s.label for s in found] == ["best-of-N", "paired", "single variant"]


def test_a_session_outside_eval_sessions_is_listed_first(paired: Path, tmp_path: Path) -> None:
    moved = paired.rename(tmp_path / paired.name)
    assert review.sources() == []
    assert review.sources(moved) == [review.Source("paired", moved)]


# --- Single-variant data layer (no Streamlit) ----------------------------------------------


def test_open_results_lists_images_in_sheet_order_with_hints(results: Path) -> None:
    opened_ = variant_review.open_results(results)
    assert len(opened_.items) == 24
    first = opened_.items[0]
    prompt = load_prompts()[0]
    assert (first.item, first.prompt_id, first.seed) == (f"{prompt.id}_1000", prompt.id, 1000)
    assert first.checks == prompt.checks
    assert first.prompt == prompt.prompt
    assert first.image == paths.eval_dir() / results.name / f"{prompt.id}_1000.png"
    assert first.hints == {"rewrite": f"gamma rewrite of {prompt.id}", "seconds": 124.0}


def test_variant_save_round_trips_and_touches_only_that_image(results: Path) -> None:
    first, second = variant_review.open_results(results).items[:2]
    saved = VariantJudgment(dict.fromkeys(first.checks, "pass"), "a crisp sign")
    variant_review.save_judgment(results, first, saved)
    after = variant_review.read_judgments(results)
    assert after[first.item] == saved
    assert after[first.item].complete(first.checks)
    assert not after[second.item].complete(second.checks)
    assert not (results / "review.csv.tmp").exists()
    lines = (results / "review.csv").read_text().splitlines()
    assert sum("a crisp sign" in line for line in lines) == 1  # notes on the first row only


def test_variant_save_rejects_bad_values_and_unknown_images(results: Path) -> None:
    item = variant_review.open_results(results).items[0]
    with pytest.raises(SessionError, match="pass/fail"):
        variant_review.save_judgment(results, item, VariantJudgment({"x": "maybe"}))
    stranger = variant_review.VariantItem("nope_1", "nope", 1, (), Path(), "", {})
    with pytest.raises(SessionError, match="nope_1"):
        variant_review.save_judgment(results, stranger, VariantJudgment({}))


def test_open_results_needs_a_review_sheet() -> None:
    with pytest.raises(SessionError, match=r"no review\.csv"):
        variant_review.open_results(variant("delta"))


# --- Picking a source ----------------------------------------------------------------------


def test_with_nothing_to_review_the_page_says_where_it_comes_from() -> None:
    at = opened()
    assert at.info[0].value.startswith("Sessions from `qwi eval session`")


def test_the_page_waits_for_a_pick(paired: Path) -> None:
    at = opened()
    assert at.selectbox(key="review").value is None
    assert not at.image


def test_the_query_parameter_opens_a_source(results: Path) -> None:
    at = opened(f"results/{results.name}")
    assert at.selectbox(key="review").value == f"results/{results.name}"
    assert "Image 1 of 24" in texts(at)


def test_picking_a_source_opens_it(paired: Path) -> None:
    at = opened()
    at.selectbox(key="review").select(f"session/{paired.name}").run()
    assert not at.exception
    assert "Item 1 of 12" in texts(at)


def test_qwi_eval_review_lands_on_its_session(
    paired: Path, best_of: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = launched(paired, monkeypatch)
    assert at.selectbox(key="review").value == f"session/{paired.name}"
    assert "Item 1 of 12" in texts(at)


def test_an_unreadable_session_shows_an_error(paired: Path) -> None:
    (paired / "review.csv").unlink()
    at = opened(f"session/{paired.name}")
    assert at.error[0].value.startswith(f"Can't open `{paired}`")


# --- Paired mode ---------------------------------------------------------------------------


def fill_pair(at: AppTest, item: sessions.Item, left: str = "pass", right: str = "fail") -> None:
    for n, _ in enumerate(item.checks):
        at.segmented_control(key=f"{item.item}-{n}-left").set_value(left)
        at.segmented_control(key=f"{item.item}-{n}-right").set_value(right)


def test_the_first_item_shows_two_unlabeled_images(
    paired: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = launched(paired, monkeypatch)
    item = sessions.open_session(paired).items[0]
    prompt = _prompt(item.prompt_id)
    assert "Item 1 of 12" in texts(at)
    assert at.button(key="previous").disabled
    assert [img.proto.imgs[0].alt for img in at.image] == [f"Left: {prompt}", f"Right: {prompt}"]
    assert [h.value for h in at.subheader] == ["Left", "Right"]
    assert len(at.main.segmented_control) == 2 * len(item.checks) + 1  # + prefer
    assert at.progress[0].value == 0


def test_nothing_reveals_the_variant_before_saving(
    paired: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = launched(paired, monkeypatch)
    assert not at.expander  # no hints yet
    assert "rewrite of" not in texts(at)


def test_paging_stays_within_bounds(paired: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = launched(paired, monkeypatch)
    page_through(at, 11)
    assert "Item 12 of 12" in texts(at)
    assert at.button(key="next").disabled
    at.button(key="previous").click().run()
    assert "Item 11 of 12" in texts(at)


def test_an_incomplete_pair_is_not_saved(paired: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = launched(paired, monkeypatch)
    item = sessions.open_session(paired).items[0]
    before = (paired / "review.csv").read_text()
    at.segmented_control(key=f"{item.item}-0-left").set_value("pass")
    at.form_submit_button[0].click().run()
    assert at.warning[0].value == "Mark every check pass or fail on both sides."
    assert (paired / "review.csv").read_text() == before


def test_saving_a_pair_writes_review_csv_and_reveals_the_hints(
    paired: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = launched(paired, monkeypatch)
    item = sessions.open_session(paired).items[0]
    fill_pair(at, item)
    at.segmented_control(key=f"{item.item}-prefer").set_value("left")
    at.text_input(key=f"{item.item}-notes").set_value("crisper sign")
    at.form_submit_button[0].click().run()
    assert not at.exception
    assert sessions.read_judgments(paired)[item.item] == Judgment(
        dict.fromkeys(item.checks, "pass"),
        dict.fromkeys(item.checks, "fail"),
        "left",
        "crisper sign",
    )
    assert ":green-badge[:material/check: Saved]" in texts(at)
    assert len(at.expander) == 2  # hints, one per side
    assert "rewrite of" in texts(at)
    assert at.progress[0].value == round(100 / 12)


def test_reopening_a_session_starts_at_the_first_unjudged_item(
    paired: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = sessions.open_session(paired).items[0]
    sessions.save_judgment(paired, first, full(first))
    at = launched(paired, monkeypatch)
    assert "Item 2 of 12" in texts(at)


# --- Best-of-N mode ------------------------------------------------------------------------


def fill_bestof(
    at: AppTest, item: bestof.BestOfItem, marks: dict[int, str], best: int | None
) -> None:
    for n, mark in marks.items():
        at.segmented_control(key=f"{item.item}-{n}-acceptable").set_value(mark)
    at.segmented_control(key=f"{item.item}-best").set_value(best)


def test_a_best_of_prompt_shows_its_candidates_and_nothing_else(
    best_of: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = launched(best_of, monkeypatch)
    session = bestof.open_session(best_of)
    item = session.items[0]
    prompt = session.prompts[item.prompt_id]
    assert "Prompt 1 of 12" in texts(at)
    assert f"**Prompt:** {prompt}" in texts(at)
    assert [img.proto.imgs[0].alt for img in at.image] == [
        f"Candidate {n}: {prompt}" for n in item.candidates
    ]
    assert [h.value for h in at.subheader] == [f"Candidate {n}" for n in item.candidates]
    assert len(at.main.segmented_control) == 4 + 1  # acceptable per candidate, then best
    shown = texts(at)
    assert not any(str(s) in shown for s in SEEDS)  # no seeds ...
    assert "0.9" not in shown  # ... and no reranker scores
    assert at.progress[0].value == 0


def test_the_best_must_be_acceptable_when_any_is(
    best_of: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = launched(best_of, monkeypatch)
    item = bestof.open_session(best_of).items[0]
    before = (best_of / "review.csv").read_text()
    fill_bestof(at, item, {1: "yes", 2: "no"}, 1)
    at.form_submit_button[0].click().run()
    assert at.warning[0].value == "Mark every candidate acceptable or not."
    fill_bestof(at, item, {1: "yes", 2: "no", 3: "no", 4: "no"}, 2)
    at.form_submit_button[0].click().run()
    assert at.warning[0].value == "The best candidate must be an acceptable one, when any is."
    assert (best_of / "review.csv").read_text() == before


def test_saving_a_best_of_prompt_writes_review_csv(
    best_of: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = launched(best_of, monkeypatch)
    item = bestof.open_session(best_of).items[0]
    fill_bestof(at, item, {1: "no", 2: "yes", 3: "yes", 4: "no"}, 3)
    at.text_input(key=f"{item.item}-notes").set_value("crisp apples")
    at.form_submit_button[0].click().run()
    assert not at.exception
    assert bestof.read_judgments(best_of)[item.item] == BestOfJudgment(
        {1: "no", 2: "yes", 3: "yes", 4: "no"}, 3, "crisp apples"
    )
    assert ":green-badge[:material/check: Saved]" in texts(at)
    assert at.progress[0].value == round(100 / 12)


def test_each_source_keeps_its_own_place(paired: Path, best_of: Path) -> None:
    at = opened(f"session/{paired.name}")
    page_through(at, 2)
    at.selectbox(key="review").select(f"session/{best_of.name}").run()
    assert "Prompt 1 of 12" in texts(at)
    at.selectbox(key="review").select(f"session/{paired.name}").run()
    assert "Item 3 of 12" in texts(at)


# --- Single-variant mode -------------------------------------------------------------------


def test_an_image_shows_with_its_hints_in_view(results: Path) -> None:
    at = opened(f"results/{results.name}")
    item = variant_review.open_results(results).items[0]
    assert "Image 1 of 24" in texts(at)
    assert [img.proto.imgs[0].alt for img in at.image] == [item.prompt]
    assert f"gamma rewrite of {item.prompt_id}" in texts(at)  # nothing here is blind
    assert [c.label for c in at.main.segmented_control] == list(item.checks)
    assert at.progress[0].value == 0


def test_paging_covers_every_image(results: Path) -> None:
    at = opened(f"results/{results.name}")
    page_through(at, 23)
    assert "Image 24 of 24" in texts(at)
    assert at.button(key="next").disabled


def test_saving_an_image_writes_review_csv(results: Path) -> None:
    at = opened(f"results/{results.name}")
    item = variant_review.open_results(results).items[0]
    before = (results / "review.csv").read_text()
    at.segmented_control(key=f"{item.item}-0-result").set_value("pass")
    at.form_submit_button[0].click().run()
    assert at.warning[0].value == "Mark every check pass or fail."
    assert (results / "review.csv").read_text() == before

    for n, _ in enumerate(item.checks):
        at.segmented_control(key=f"{item.item}-{n}-result").set_value("fail")
    at.text_input(key=f"{item.item}-notes").set_value("sign misspelled")
    at.form_submit_button[0].click().run()
    assert not at.exception
    assert variant_review.read_judgments(results)[item.item] == VariantJudgment(
        dict.fromkeys(item.checks, "fail"), "sign misspelled"
    )
    assert ":green-badge[:material/check: Saved]" in texts(at)
    assert at.progress[0].value == round(100 / 24)
