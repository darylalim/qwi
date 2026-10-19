"""qwi's Streamlit UI (PLAN.md §5.6): `qwi ui`. Page config and navigation; pages in app_pages/.

The UI is a thin client of `Pipeline.run`: pages submit requests to the shared `JobManager` and
render its snapshots. No model code, policy logic or file layout lives here.
"""

import os
from importlib.metadata import version

import streamlit as st

from qwi import logs, sessions
from qwi.ui import components

# Jobs and rewrite-only runs log outside any run.log; attached once per process (§5.11).
logs.app_log()

st.set_page_config(
    page_title="qwi",
    page_icon=":material/photo_library:",
    layout="wide",
    initial_sidebar_state="auto",
)

page = st.navigation(
    [
        st.Page("app_pages/generate.py", title="Generate", icon=":material/auto_awesome:"),
        st.Page("app_pages/history.py", title="History", icon=":material/history:"),
        st.Page(
            "app_pages/eval_review.py",
            title="Eval review",
            icon=":material/rate_review:",
            # `qwi eval review <session>` opens here; `qwi ui` opens on Generate.
            default=sessions.REVIEW_SESSION_ENV in os.environ,
        ),
    ]
)
components.doctor_status()
st.sidebar.caption(f"qwi {version('qwi')}")
page.run()
