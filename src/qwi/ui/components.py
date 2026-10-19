"""Streamlit pieces shared by every page (PLAN.md §5.6): the job manager and the sidebar's doctor
line. M5c adds the card and grid components."""

import os

import streamlit as st

from qwi import paths
from qwi.ui import jobs, status
from qwi.ui.jobs import JobManager
from qwi.ui.status import StatusLine


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
