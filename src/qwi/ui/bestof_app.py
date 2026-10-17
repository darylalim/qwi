"""Blind best-of-N review: `qwi eval review <bo4 session>` (PLAN.md §5.4 "Best-of-4 calibration").

One prompt at a time with its candidates side by side, numbered in a shuffled order. Each is
marked acceptable or not, and one is picked as best; **Save** writes the item's rows to
`review.csv`. No reranker score or seed is ever shown: the calibration compares the reranker's
picks with these. All file logic lives in `qwi.bestof`; this page only renders it.
"""

import os
from pathlib import Path

import streamlit as st

from qwi import bestof, sessions
from qwi.errors import SessionError

st.set_page_config(page_title="Best-of review", page_icon=":material/grid_view:", layout="wide")

folder = Path(os.environ.get(sessions.REVIEW_SESSION_ENV, ""))
try:
    session = bestof.open_session(folder)
except (SessionError, OSError, KeyError, ValueError) as exc:
    st.error(f"Can't open the session at `{folder}`: {exc}", title="No review session")
    st.stop()

# review.csv is the source of truth: re-read it on every rerun.
judgments = bestof.read_judgments(session.folder)
items = session.items
total = len(items)


def judged(item: bestof.BestOfItem) -> bool:
    found = judgments.get(item.item)
    return found is not None and found.problem(item) is None


if "index" not in st.session_state:  # start at the first item still to judge
    st.session_state.index = next((n for n, i in enumerate(items) if not judged(i)), 0)


def move(step: int) -> None:
    st.session_state.index = min(max(st.session_state.index + step, 0), total - 1)


index = st.session_state.index
item = items[index]
judgment = judgments.get(item.item) or bestof.BestOfJudgment({})
prompt = session.prompts[item.prompt_id]
done = sum(judged(i) for i in items)

st.title("Best-of review")
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
    st.markdown(f"Prompt {index + 1} of {total}")
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
st.progress(done / total, text=f"{done} of {total} prompts judged")
st.markdown(f"**Prompt:** {prompt}")

with st.form(key=f"form-{item.item}", border=False):
    for column, n in zip(st.columns(len(item.seeds), gap="medium"), item.candidates, strict=True):
        with column:
            st.subheader(f"Candidate {n}")
            image = session.images[(item.item, n)]
            if image.is_file():
                st.image(str(image), width="stretch", alt=f"Candidate {n}: {prompt}")
            else:
                st.warning(f"`{image}`", title="Image not found")
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
