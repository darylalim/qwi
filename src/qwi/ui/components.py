"""Streamlit pieces shared by every page (PLAN.md §5.6). M5c adds the card and grid components."""

import os

import streamlit as st

from qwi.ui import jobs
from qwi.ui.jobs import JobManager


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
