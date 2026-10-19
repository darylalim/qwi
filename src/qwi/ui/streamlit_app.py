"""qwi's Streamlit UI (PLAN.md §5.6): `qwi ui`. Page config and navigation; pages in app_pages/.

The UI is a thin client of `Pipeline.run`: pages submit requests to the shared `JobManager` and
render its snapshots. No model code, policy logic or file layout lives here.
"""

from importlib.metadata import version

import streamlit as st

st.set_page_config(
    page_title="qwi",
    page_icon=":material/photo_library:",
    layout="wide",
    initial_sidebar_state="auto",
)

page = st.navigation(
    [st.Page("app_pages/generate.py", title="Generate", icon=":material/auto_awesome:")]
)
st.sidebar.caption(f"qwi {version('qwi')}")
page.run()
