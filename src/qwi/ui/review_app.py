"""Blind review of a paired session: `qwi eval review <session>` (PLAN.md §5.4, §5.6).

One pair at a time, labeled only Left / Right. Each check is marked pass/fail per side, with an
optional preference and notes; **Save** writes the item's rows to `review.csv`. Variant names never
appear, and the per-side hints (rewrite, OCR, reranker score, timing) stay hidden until the pair
is saved. All file logic lives in `qwi.sessions`; this page only renders it. M5b folds it into
the main UI as the Eval review page; the theme comes in M5c.
"""

import os
from pathlib import Path

import streamlit as st

from qwi import sessions
from qwi.errors import SessionError

HINT_LABELS = {
    "rewrite": "Rewrite",
    "ocr": "OCR",
    "ocr_exact": "OCR exact match",
    "ocr_match": "OCR match, ignoring case",
    "rerank_prob": "Reranker prob",
    "white_border_share": "White border share",
    "seconds": "Seconds",
}

st.set_page_config(page_title="Eval review", page_icon=":material/compare:", layout="wide")

folder = Path(os.environ.get(sessions.REVIEW_SESSION_ENV, ""))
try:
    session = sessions.open_session(folder)
except (SessionError, OSError, KeyError, ValueError) as exc:
    st.error(f"Can't open the session at `{folder}`: {exc}", title="No review session")
    st.stop()

# review.csv is the source of truth: re-read it on every rerun.
judgments = sessions.read_judgments(session.folder)
items = session.items
total = len(items)


def judged(item: sessions.Item) -> bool:
    found = judgments.get(item.item)
    return found is not None and found.complete(item.checks)


if "index" not in st.session_state:  # start at the first item still to judge
    st.session_state.index = next((n for n, i in enumerate(items) if not judged(i)), 0)


def move(step: int) -> None:
    st.session_state.index = min(max(st.session_state.index + step, 0), total - 1)


index = st.session_state.index
item = items[index]
judgment = judgments.get(item.item) or sessions.Judgment({}, {})
done = sum(judged(i) for i in items)

st.title("Eval review")
st.caption(f"Session `{session.name}`")
with st.container(horizontal=True, vertical_alignment="center"):
    st.button(
        "Previous",
        icon=":material/chevron_left:",
        key="previous",
        on_click=move,
        args=(-1,),
        disabled=index == 0,
    )
    st.markdown(f"Item {index + 1} of {total}")
    st.button(
        "Next",
        icon=":material/chevron_right:",
        key="next",
        on_click=move,
        args=(1,),
        disabled=index == total - 1,
    )
    if judged(item):
        st.badge("Saved", icon=":material/check:", color="green")
st.progress(done / total, text=f"{done} of {total} pairs judged")

image_left, image_right, form_column = st.columns([1, 1, 1], gap="large")
for column, side in ((image_left, "left"), (image_right, "right")):
    shown = session.sides[(item.item, side)]
    with column:
        st.subheader(side.capitalize())
        if shown.image.is_file():
            st.image(str(shown.image), width="stretch", alt=f"{side.capitalize()}: {shown.prompt}")
        else:
            st.warning(f"`{shown.image}`", title="Image not found")
        if judged(item):  # hints only after the pair is saved (§5.4)
            with st.expander("Hints"):
                for key, value in shown.hints.items():
                    st.markdown(f"**{HINT_LABELS.get(key, key)}:** {value}")

with form_column:
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
