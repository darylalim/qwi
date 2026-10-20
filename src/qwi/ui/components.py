"""Streamlit pieces shared by every page (PLAN.md §5.6): the job manager, the sidebar's doctor
line and layout controls, and the responsive layout helpers and candidate cards.

Layout adapts in the browser, never by detecting the viewport: columns stack at ≤ 640 px, and
fixed-width cards in a horizontal container wrap onto new rows and shrink to a narrower parent.
"""

import os
from collections.abc import Sequence
from typing import Literal

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from qwi import paths
from qwi.config import DEFAULTS
from qwi.ui import jobs, status
from qwi.ui.jobs import JobManager
from qwi.ui.status import StatusLine
from qwi.ui.view import CandidateView

THUMB_SIZES = DEFAULTS.thumb_sizes  # Thumbnail size → card width in px


def job_manager() -> JobManager:
    """The one `JobManager` all tabs and sessions share (`QWI_PIPELINE`, `QWI_JOBS_SYNC`)."""
    return _job_manager(
        os.environ.get(jobs.PIPELINE_ENV),
        os.environ.get(jobs.FAKE_SCRIPT_ENV),
        os.environ.get(jobs.SYNC_ENV) == "1",
    )


# Only the manager is cached, never models: the pipeline loads and evicts each stage (§4).
@st.cache_resource(max_entries=1, show_spinner=False)
def _job_manager(pipeline: str | None, script: str | None, sync: bool) -> JobManager:
    return JobManager(jobs.make_pipeline(pipeline, script), sync=sync)


def doctor_status() -> None:
    """The sidebar's `qwi doctor` line: a badge, and what's wrong when something is (§5.6)."""
    line = _doctor_line(str(paths.home()))
    detail = "\n\n".join(line.problems) or "Disk, models, $QWI_HOME, offline mode and torch pass."
    st.sidebar.badge(line.label, icon=line.icon, color=line.color, help=detail)
    if line.problems:
        st.sidebar.caption("  \n".join([*line.problems, "Run `qwi doctor` for details."]))


# Plain data, refreshed each minute; keyed by $QWI_HOME so tests with their own homes don't share.
@st.cache_data(ttl="1m", show_spinner=False)
def _doctor_line(home: str) -> StatusLine:
    return status.summarize(status.quick_checks())


def layout_controls() -> None:
    """Thumbnail size and Stacked layout, in the URL so a bookmark or reload keeps them.

    Drawn by the entrypoint before the page runs, so every page reads the current values."""
    st.sidebar.segmented_control(
        "Thumbnail size",
        list(THUMB_SIZES),
        default=DEFAULTS.thumb_size,
        required=True,
        key="thumb",
        bind="query-params",
    )
    st.sidebar.toggle(
        "Stacked layout",
        key="stacked",
        bind="query-params",
        help="One column at any width, for half-screen windows.",
    )


def thumb_width() -> int:
    return THUMB_SIZES.get(st.session_state.get("thumb"), THUMB_SIZES[DEFAULTS.thumb_size])


def stacked() -> bool:
    return bool(st.session_state.get("stacked"))


def panes(
    spec: Sequence[int], gap: Literal["small", "medium", "large"] = "large"
) -> list[DeltaGenerator]:
    """Side-by-side panes, or one container per pane in the same order when stacked."""
    if stacked():
        return [st.container() for _ in spec]
    return st.columns(spec, gap=gap, vertical_alignment="top")


def reading() -> DeltaGenerator:
    """A block for text and forms, never wider than the reading width; it shrinks to its parent."""
    return st.container(width=DEFAULTS.reading_width)


def long_text(label: str, text: str) -> None:
    """Long or variable text (a rewrite of up to 500 words) collapsed, at a bounded height."""
    with st.expander(label):
        st.code(text, language=None, wrap_lines=True, height=320)


@st.dialog("Full view", width="large")
def full_view_dialog(c: CandidateView, alt: str) -> None:
    """The image at its native size, capped by the dialog, so inspection works on every screen."""
    st.image(str(c.path), width="content", output_format="PNG", alt=alt)
    st.caption(c.caption)


def candidate_card(
    c: CandidateView, width: int, *, alt: str, name: str, running: bool = False
) -> None:
    """One bordered card: image, caption, Best badge, View and Download.

    `name` prefixes the download's file name (the run folder); `running` says a missing score is
    still to come rather than never coming."""
    with st.container(border=True, width=width):
        if not c.path.exists():
            st.caption(f"{c.path.name} is missing")
            return
        st.image(
            str(c.path),
            width="stretch",
            output_format="PNG",  # shown as saved, not re-encoded as JPEG
            alt=alt,
        )
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(c.caption if c.prob is not None or running else f"{c.caption} · not ranked")
            if c.best:
                st.badge("Best", color="violet", icon=":material/star:")
        with st.container(horizontal=True):
            if st.button("View", icon=":material/zoom_in:", key=f"view_{c.path}"):
                full_view_dialog(c, alt)
            st.download_button(
                "Download",
                data=c.path.read_bytes,
                file_name=f"{name}_{c.path.name}",
                mime="image/png",
                on_click="ignore",
                icon=":material/download:",
                key=f"download_{c.path}",
            )


def candidate_grid(
    cands: Sequence[CandidateView], width: int, *, prompt: str, name: str, running: bool = False
) -> None:
    """Cards in seed order, as many per row as fit, so positions stay put while results stream."""
    with st.container(horizontal=True, gap="medium"):
        for c in cands:
            candidate_card(c, width, alt=f"{prompt} (seed {c.seed})", name=name, running=running)
