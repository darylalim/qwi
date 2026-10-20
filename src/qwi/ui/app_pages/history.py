"""History page (PLAN.md §5.6): every run in `$QWI_HOME/runs`, newest first, and one run's detail.

The selected run lives in the `run` query parameter, not in a row index, so a run has a link and
the selection stays on it when newer runs appear above it. The page is read-only except the star
(§8 2026-10-19), which goes through `record.set_starred`, the same atomic write as every record.
"""

import streamlit as st

from qwi import errors, record
from qwi.ui import components, history, view
from qwi.ui.history import RunEntry

LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]


def select() -> None:
    """Map the table's selected row to its run id, using the ids that table was drawn with."""
    rows = st.session_state.hist_table["selection"]["rows"]
    if run_id := history.selected_run(rows, st.session_state.hist_ids):
        st.query_params["run"] = run_id
    else:
        st.query_params.pop("run", None)


def toggle_star(entry: RunEntry) -> None:
    try:
        record.set_starred(entry.run_dir, not entry.record.starred)
    except errors.RunStillRunning as exc:
        st.session_state.hist_notice = str(exc)


def show_candidates(entry: RunEntry) -> None:
    if candidates := entry.candidates():
        components.candidate_grid(
            candidates,
            components.thumb_width(),
            prompt=entry.record.user_prompt,
            name=entry.run_id,
        )


def show_log(entry: RunEntry) -> None:
    """The run's own `run.log`, never `qwi.log`, which also has other runs' lines (§9)."""
    with st.expander("Log", icon=":material/description:"):
        level = st.segmented_control(
            "Lowest level", LOG_LEVELS, default="INFO", required=True, key="hist_log_level"
        )
        lines = view.log_lines(entry.run_dir, min_level=level or "INFO")
        if lines:
            st.code("\n".join(lines), language=None, height=320)
        else:
            st.caption("No log lines at this level.")


def show_run(entry: RunEntry) -> None:
    with components.reading():
        show_summary(entry)
    show_candidates(entry)
    if rows := entry.stage_rows():
        st.dataframe(
            rows,
            hide_index=True,
            width="content",
            column_config={
                "Seconds": st.column_config.NumberColumn(format="%.1f"),
                "Peak GB": st.column_config.NumberColumn(format="%.1f"),
                "Swap GB": st.column_config.NumberColumn(format="%.2f"),
            },
            alt="Time, peak memory and swap per stage",
        )
    show_log(entry)


def show_summary(entry: RunEntry) -> None:
    """The run's status, notices, prompts, guard and settings: the text above its images."""
    rec = entry.record
    running = rec.status == "running"
    with st.container(horizontal=True, vertical_alignment="center"):
        st.subheader(entry.run_id)
        st.badge(rec.status, color=entry.status_color)
        st.button(
            "Unstar" if rec.starred else "Star",
            icon=":material/star:" if rec.starred else ":material/star_outline:",
            on_click=toggle_star,
            args=(entry,),
            disabled=running,
            help="A run can be starred once it ends." if running else "Prune never deletes it.",
            key="hist_star",
        )
    if notice := st.session_state.pop("hist_notice", None):
        st.warning(notice)
    st.caption(f"Started {entry.started} · pid {rec.pid}")
    if running:
        st.info("The run is still going. Reload the page to see its progress.", title="Running")
    elif rec.status == "interrupted":
        st.warning(
            "The run's process ended before it finished (quit, crash, or closed terminal). "
            "Its finished candidates are kept.",
            title="Interrupted",
        )
    elif rec.status == "cancelled":
        st.info(f"Cancelled after {len(rec.candidates)} candidates.", title="Cancelled")
    if rec.error:
        st.error(
            f"{rec.error.get('type')}: {rec.error.get('message')}",
            title=f"Error in {rec.error.get('stage') or 'the run'}",
        )

    st.caption("Prompt")
    st.code(rec.user_prompt, language=None, wrap_lines=True)
    if rec.continued_from:
        st.caption(f"More seeds for run {rec.continued_from}: its prompt, no new rewrite.")
    if rec.rewritten_prompt:
        components.long_text("Rewritten prompt", rec.rewritten_prompt)
        if rec.rewrite_fallback_reason:
            st.caption(f"Generated from the user's prompt instead: {rec.rewrite_fallback_reason}")
    if rec.prompt_source == "edited_rewrite":
        st.caption("Generated from an edited rewrite, ranked against the prompt above.")
    for line in entry.guard_lines():
        st.caption(f"Guard · {line}")
    if config := entry.config_line():
        st.caption(config)


entries = history.load()
if not entries:
    st.info("Runs from the Generate page and `qwi generate` appear here.", title="No runs yet")
    st.stop()

ids = [e.run_id for e in entries]
st.session_state.hist_ids = ids
selected = next((e for e in entries if e.run_id == st.query_params.get("run")), None)
table, detail = components.panes([2, 3])
table.dataframe(
    [e.row() for e in entries],
    key="hist_table",
    on_select=select,
    selection_mode="single-row",
    selection_default={"selection": {"rows": [ids.index(selected.run_id)]}} if selected else None,
    hide_index=True,
    column_config={
        "Prompt": st.column_config.TextColumn(width="large"),
        "Best prob": st.column_config.NumberColumn(format="%.2f"),
        "Seconds": st.column_config.NumberColumn(format="%.0f"),
        "Starred": st.column_config.CheckboxColumn(),
    },
    width="stretch",
    height=320 if components.stacked() else 480,  # stacked, the detail sits below the table
    placeholder="-",
    alt="Pipeline runs, newest first",
)
with detail:
    if selected is None:
        st.caption("Select a run to see its images, prompts, guard verdicts, timings and log.")
    else:
        show_run(selected)
