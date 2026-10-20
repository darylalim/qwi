"""Eval review page (PLAN.md §5.4, §5.6): judge a session or a variant's images, one at a time.

Three modes, picked by the source: a **paired** session (blind Left / Right, hints only after
saving), a **best-of-N** session (candidates side by side; no seed or score ever shown) and a
**single variant** (`eval/results/<variant>/review.csv`, hints in view). Every Save writes that
source's `review.csv`, which is re-read on each rerun as the source of truth. The source is bound
to the `review` query parameter, so each has a link; `qwi eval review <session>` preselects its
session. All file logic lives in `qwi.sessions`, `qwi.bestof` and `qwi.variant_review`.
"""

import os
from collections.abc import Sequence
from pathlib import Path

import streamlit as st

from qwi import bestof, sessions, variant_review
from qwi.errors import SessionError
from qwi.ui import components, review
from qwi.ui.review import Source

HINT_LABELS = {
    "rewrite": "Rewrite",
    "ocr": "OCR",
    "ocr_exact": "OCR exact match",
    "ocr_match": "OCR match, ignoring case",
    "rerank_prob": "Reranker prob",
    "white_border_share": "White border share",
    "seconds": "Seconds",
}
OPEN_ERRORS = (SessionError, OSError, KeyError, ValueError)
# Fixed sizes, not the sidebar's Thumbnail size: every item is judged at the same scale (§5.6).
PAIR_WIDTH = components.THUMB_SIZES["M"]
VARIANT_WIDTH = components.THUMB_SIZES["L"]


def move(key: str, step: int, total: int) -> None:
    st.session_state[key] = min(max(st.session_state[key] + step, 0), total - 1)


def toolbar(source: Source, judged: Sequence[bool], unit: str, units: str) -> int:
    """Previous / `n of N` / Next, the Saved badge and progress; the index shown.

    Each source keeps its own place, starting at the first item still to judge."""
    key = f"review_index:{source.key}"
    total = len(judged)
    if key not in st.session_state:
        st.session_state[key] = next((n for n, done in enumerate(judged) if not done), 0)
    index = st.session_state[key]
    with st.container(horizontal=True, vertical_alignment="center"):
        st.button(
            "Previous",
            icon=":material/chevron_left:",
            key="previous",
            on_click=move,
            args=(key, -1, total),
            disabled=index == 0,
        )
        st.markdown(f"{unit} {index + 1} of {total}")
        st.button(
            "Next",
            icon=":material/chevron_right:",
            key="next",
            on_click=move,
            args=(key, 1, total),
            disabled=index == total - 1,
        )
        if judged[index]:
            st.badge("Saved", icon=":material/check:", color="green")
    done = sum(judged)
    st.progress(done / total, text=f"{done} of {total} {units} judged")
    return index


def show_image(image: Path, alt: str, width: int) -> None:
    """The image at `width` px, or less when its pane is narrower."""
    with st.container(width=width):
        if image.is_file():
            st.image(str(image), width="stretch", alt=alt)
        else:
            st.warning(f"`{image}`", title="Image not found")


def show_hints(hints: dict) -> None:
    for key, value in hints.items():
        st.markdown(f"**{HINT_LABELS.get(key, key)}:** {value}")


def show_paired(source: Source) -> None:
    """One pair labeled only Left / Right; each side's hints appear once the pair is saved."""
    session = sessions.open_session(source.folder)
    judgments = sessions.read_judgments(session.folder)

    def judged(item: sessions.Item) -> bool:
        found = judgments.get(item.item)
        return found is not None and found.complete(item.checks)

    index = toolbar(source, [judged(i) for i in session.items], "Item", "pairs")
    item = session.items[index]
    judgment = judgments.get(item.item) or sessions.Judgment({}, {})

    image_left, image_right, form_column = components.panes([1, 1, 1])
    for column, side in ((image_left, "left"), (image_right, "right")):
        shown = session.sides[(item.item, side)]
        with column:
            st.subheader(side.capitalize())
            show_image(shown.image, f"{side.capitalize()}: {shown.prompt}", PAIR_WIDTH)
            if judged(item):  # hints only after the pair is saved (§5.4)
                with st.expander("Hints"):
                    show_hints(shown.hints)

    with form_column, components.reading():
        st.markdown(f"**Prompt:** {session.sides[(item.item, 'left')].prompt}")
        with st.form(key=f"form-{item.item}", border=False):
            for n, check in enumerate(item.checks):
                st.markdown(check)
                with st.container(horizontal=True):
                    for side, saved in (("left", judgment.left), ("right", judgment.right)):
                        st.segmented_control(
                            side.capitalize(),
                            sessions.RESULTS,
                            default=saved.get(check) or None,
                            format_func=str.capitalize,
                            key=f"{item.item}-{n}-{side}",
                        )
            st.segmented_control(
                "Prefer (optional)",
                sessions.PREFER,
                default=judgment.prefer or None,
                format_func=str.capitalize,
                key=f"{item.item}-prefer",
            )
            st.text_input("Notes", value=judgment.notes, key=f"{item.item}-notes")
            submitted = st.form_submit_button("Save", icon=":material/save:", type="primary")

        if submitted:
            state = st.session_state
            update = sessions.Judgment(
                {c: state[f"{item.item}-{n}-left"] or "" for n, c in enumerate(item.checks)},
                {c: state[f"{item.item}-{n}-right"] or "" for n, c in enumerate(item.checks)},
                state[f"{item.item}-prefer"] or "",
                state[f"{item.item}-notes"],
            )
            if update.complete(item.checks):
                sessions.save_judgment(session.folder, item, update)
                st.rerun()  # re-read review.csv: the badge and hints appear
            st.warning("Mark every check pass or fail on both sides.", title="Not saved")


def show_bestof(source: Source) -> None:
    """One prompt's candidates in a shuffled order; the reranker's view never shows (§5.4)."""
    session = bestof.open_session(source.folder)
    judgments = bestof.read_judgments(session.folder)

    def judged(item: bestof.BestOfItem) -> bool:
        found = judgments.get(item.item)
        return found is not None and found.problem(item) is None

    index = toolbar(source, [judged(i) for i in session.items], "Prompt", "prompts")
    item = session.items[index]
    judgment = judgments.get(item.item) or bestof.BestOfJudgment({})
    prompt = session.prompts[item.prompt_id]
    st.markdown(f"**Prompt:** {prompt}")

    with st.form(key=f"form-{item.item}", border=False):
        # Up to N candidates: a wrapping row, not N columns (§5.6 allows 3 per row).
        with st.container(horizontal=True, gap="medium"):
            for n in item.candidates:
                with st.container(width=PAIR_WIDTH):
                    st.subheader(f"Candidate {n}")
                    show_image(
                        session.images[(item.item, n)], f"Candidate {n}: {prompt}", PAIR_WIDTH
                    )
                    st.segmented_control(
                        "Acceptable",
                        bestof.ACCEPTABLE,
                        default=judgment.acceptable.get(n) or None,
                        format_func=str.capitalize,
                        key=f"{item.item}-{n}-acceptable",
                    )
        st.segmented_control(
            "Best",
            list(item.candidates),
            default=judgment.best,
            format_func=lambda n: f"Candidate {n}",
            key=f"{item.item}-best",
        )
        st.text_input("Notes", value=judgment.notes, key=f"{item.item}-notes")
        submitted = st.form_submit_button("Save", icon=":material/save:", type="primary")

    if submitted:
        state = st.session_state
        update = bestof.BestOfJudgment(
            {n: state[f"{item.item}-{n}-acceptable"] or "" for n in item.candidates},
            state[f"{item.item}-best"],
            state[f"{item.item}-notes"],
        )
        problem = update.problem(item)
        if problem is None:
            bestof.save_judgment(session.folder, item, update)
            st.rerun()  # re-read review.csv: the badge appears
        st.warning(problem, title="Not saved")


def show_variant(source: Source) -> None:
    """One image of one variant with its automated hints in view: nothing here is blind."""
    results = variant_review.open_results(source.folder)
    if not results.items:
        st.info(f"`{source.folder.name}/review.csv` lists no images.", title="Nothing to review")
        return
    judgments = variant_review.read_judgments(results.folder)

    def judged(item: variant_review.VariantItem) -> bool:
        found = judgments.get(item.item)
        return found is not None and found.complete(item.checks)

    index = toolbar(source, [judged(i) for i in results.items], "Image", "images")
    item = results.items[index]
    judgment = judgments.get(item.item) or variant_review.VariantJudgment({})

    image_column, form_column = components.panes([1, 1])
    with image_column:
        show_image(item.image, item.prompt, VARIANT_WIDTH)
    with form_column, components.reading():
        st.markdown(f"**Prompt:** {item.prompt}")
        st.caption(f"`{item.prompt_id}` · seed {item.seed}")
        show_hints(item.hints)
        with st.form(key=f"form-{item.item}", border=False):
            for n, check in enumerate(item.checks):
                st.segmented_control(
                    check,
                    sessions.RESULTS,
                    default=judgment.results.get(check) or None,
                    format_func=str.capitalize,
                    key=f"{item.item}-{n}-result",
                )
            st.text_input("Notes", value=judgment.notes, key=f"{item.item}-notes")
            submitted = st.form_submit_button("Save", icon=":material/save:", type="primary")

        if submitted:
            state = st.session_state
            update = variant_review.VariantJudgment(
                {c: state[f"{item.item}-{n}-result"] or "" for n, c in enumerate(item.checks)},
                state[f"{item.item}-notes"],
            )
            if update.complete(item.checks):
                variant_review.save_judgment(results.folder, item, update)
                st.rerun()  # re-read review.csv: the badge appears
            st.warning("Mark every check pass or fail.", title="Not saved")


launched = os.environ.get(sessions.REVIEW_SESSION_ENV)
found = review.sources(Path(launched) if launched else None)
by_key = {s.key: s for s in found}
if launched and "review" not in st.session_state and "review" not in st.query_params:
    st.session_state.review = review.session_source(Path(launched)).key  # before the widget

st.title("Eval review")
if not found:
    st.info(
        "Sessions from `qwi eval session` and results from `qwi eval` appear here.",
        title="Nothing to review yet",
    )
    st.stop()
picked = components.reading().selectbox(
    "Review",
    list(by_key),
    index=None,
    placeholder="Pick a session or a variant's results",
    key="review",
    bind="query-params",
)
if picked is None:
    st.caption(
        "A **paired** session compares two variants blind; a **best-of-N** session picks the "
        "best candidate per prompt; a **single variant** checks one results folder's images."
    )
    st.stop()

source = by_key[picked]
st.caption(source.label.capitalize())
show = {"paired": show_paired, "bestof": show_bestof, "variant": show_variant}[source.kind]
try:
    show(source)
except OPEN_ERRORS as exc:
    st.error(f"Can't open `{source.folder}`: {exc}", title="Can't review this")
