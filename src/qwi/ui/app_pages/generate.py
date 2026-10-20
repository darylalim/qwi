"""Generate page (PLAN.md §5.6): the form, live status while a job runs, and the candidates.

Submitting runs in the button's callback, before the script, so the same rerun already shows
the form disabled. Two fragments, the status under the form and the candidates beside it (§5.6),
poll the job once a second only while it runs, and rerun the page when it ends so the form comes
back. Errors get their §5.11 banner, one-click
cheaper retries, and the run's log.

Rewrite, then edit (§5.6): **Rewrite only** runs a short `rewrite_only` job; its result fills the
editable **Prompt to generate**, and **Generate from this prompt** runs it as given, still ranked
against the original prompt. The area stays until **Clear**, so an edit can be tried again.
"""

from typing import Any

import streamlit as st

from qwi.config import DEFAULTS
from qwi.pipeline import GenerateRequest, RewriteResult
from qwi.ui import components, view
from qwi.ui.jobs import Busy, JobSnapshot

SIZES = [512, 768, 1024]
GUARD_MODES = ["strict", "default", "loose"]

EDIT_KEYS = ("gen_edit", "gen_edit_result", "gen_edit_verdicts")  # the editable rewrite

jobs = components.job_manager()


def form_request(prompt: str, **changes: Any) -> GenerateRequest:
    """A request from the form's settings; `changes` override them."""
    state = st.session_state
    settings: dict[str, Any] = {
        "n": state.gen_n,
        "steps": state.gen_steps,
        "width": state.gen_size,
        "height": state.gen_size,
        "seed": state.gen_seed,
        "guard_mode": state.gen_guard,
        "rewrite": state.gen_rewrite,
    }
    return GenerateRequest(prompt, **(settings | changes))


def send(request: GenerateRequest, kind: str = "generate") -> None:
    try:
        jobs.submit(request, kind)
    except Busy:
        st.session_state.gen_notice = "A run is already in progress."


def entered_prompt(key: str) -> str | None:
    prompt = st.session_state[key].strip()
    if not prompt:
        st.session_state.gen_notice = "Enter a prompt first."
        return None
    return prompt


def submit() -> None:
    if prompt := entered_prompt("gen_prompt"):
        send(form_request(prompt))


def rewrite_only() -> None:
    if prompt := entered_prompt("gen_prompt"):
        send(form_request(prompt), "rewrite_only")


def submit_edited() -> None:
    result: RewriteResult = st.session_state.gen_edit_result
    if prompt := entered_prompt("gen_edit"):
        send(
            form_request(
                prompt,
                rewrite=False,
                rerank_query=result.user_prompt,
                prompt_source="edited_rewrite",
            )
        )


def clear_edit() -> None:
    for key in EDIT_KEYS:
        st.session_state.pop(key, None)


def load_rewrite(snap: JobSnapshot | None) -> None:
    """Fill the editable prompt once per finished rewrite-only job, before the widget exists."""
    state = st.session_state
    if snap is None or snap.running or snap.kind != "rewrite_only":
        return
    if state.get("gen_edit_job") == snap.number:  # already loaded, maybe edited or cleared since
        return
    state.gen_edit_job = snap.number
    if snap.status == "ok" and snap.rewrite is not None:
        state.gen_edit = snap.rewrite.prompt
        state.gen_edit_result = snap.rewrite
        state.gen_edit_verdicts = view.summarize(snap).verdicts


def cancel() -> None:
    jobs.cancel()


def retry(request: GenerateRequest, kind: str) -> None:
    try:
        jobs.submit(view.retry_request(request, kind))
    except Busy:
        st.session_state.gen_notice = "A run is already in progress."


def show_error(snap: JobSnapshot, run: view.RunView) -> None:
    """The §5.11 banner for the error class, its one-click retries, and where to look next."""
    banner = view.error_banner(snap)
    if banner is None:
        return
    st.error(banner.body, title=banner.title)
    if banner.retries:
        with st.container(horizontal=True):
            for option in banner.retries:
                st.button(
                    option.label,
                    icon=":material/refresh:",
                    on_click=retry,
                    args=(snap.request, option.kind),
                    key=f"retry_{option.kind}",
                )
    if snap.traceback:
        with st.expander("Traceback"):
            st.code(snap.traceback, language=None)
    if run.run_dir is not None and (lines := view.log_lines(run.run_dir)):
        with st.expander("View log", icon=":material/description:"):
            st.code("\n".join(lines), language=None, height=320)


def show_outcome(snap: JobSnapshot, run: view.RunView) -> None:
    if run.block is not None:
        st.error(run.block.summary, title="Blocked by guard")
    elif run.status == "cancelled" and snap.kind == "rewrite_only":
        st.info("The rewrite was cancelled.", title="Cancelled")
    elif run.status == "cancelled":
        st.info(f"Cancelled after {len(run.candidates)} of {run.n} candidates.", title="Cancelled")
    show_error(snap, run)


snapshot = jobs.snapshot()
busy = snapshot is not None and snapshot.running
load_rewrite(snapshot)


def follow(snap: JobSnapshot | None) -> None:
    """Rerun the page when a job started or ended since it ran, so the form follows."""
    if snap is not None and snap.running != busy:
        st.rerun()


@st.fragment(run_every="1s" if busy else None)
def job_status() -> None:
    """Under the form: stages, progress, Cancel, guard and swap notices, the rewrite, outcome."""
    snap = jobs.snapshot()
    follow(snap)
    if snap is None:
        return
    run = view.summarize(snap)
    if snap.running:
        cancelling = snap.request.cancel.is_set()
        st.button(
            "Cancelling…" if cancelling else "Cancel",
            icon=":material/cancel:",
            on_click=cancel,
            disabled=cancelling,
        )
    for stage in run.stages:
        st.status(stage.label, state=stage.state)
    if run.progress is not None:
        st.progress(run.progress.fraction, text=run.progress.text)
    for verdict in run.warnings:
        st.warning(verdict.summary, title="Guard warning")
    for risk in run.swap_risks:
        st.warning(
            f"{view.STAGE_LABELS.get(risk.stage, risk.stage)} needs about {risk.est_gb:.1f} GB "
            f"and {risk.avail_gb:.1f} GB is free, so macOS may swap. The run continues; it stops "
            "if swapping starts. Closing other apps helps.",
            title="Swap risk",
        )
    if snap.kind == "rewrite_only":
        if snap.status == "ok":
            st.caption("The rewrite is ready to edit under **Prompt to generate**.")
    elif run.rewrite is not None:
        components.long_text("Rewritten prompt", run.rewrite.text)
        if run.rewrite.fallback_reason:
            st.caption(f"Generated from your prompt instead: {run.rewrite.fallback_reason}")
    show_outcome(snap, run)


@st.fragment(run_every="1s" if busy else None)
def gallery() -> None:
    """Beside the form: the candidates as they arrive, and the low-confidence warning."""
    snap = jobs.snapshot()
    follow(snap)
    if snap is None:
        st.caption("Results appear here.")
        return
    run = view.summarize(snap)
    if run.low_confidence and run.rerank is not None:
        best = max(run.rerank.probs)
        st.warning(
            f"The best candidate's probability is {best:.2f}, below {run.rerank.min_score}.",
            title="No candidate clearly matched the prompt",
        )
    if run.candidates:
        components.candidate_grid(
            run.candidates,
            components.thumb_width(),
            prompt=snap.request.prompt,
            name=run.run_dir.name if run.run_dir else "qwi",
            running=snap.running,
        )


left, right = components.panes([1, 2])
with left, components.reading():
    form = st.form("generate", border=False)
    job_status()
with form:
    st.text_area("Prompt", key="gen_prompt", height=140, placeholder="Describe the image")
    # Right under the prompt, so they stay in reach without scrolling (§5.6).
    with st.container(horizontal=True):
        st.form_submit_button(
            "Generate",
            type="primary",
            icon=":material/auto_awesome:",
            on_click=submit,
            disabled=busy,
        )
        st.form_submit_button(
            "Rewrite only",
            icon=":material/edit_note:",
            on_click=rewrite_only,
            disabled=busy,
        )
    with st.container(horizontal=True):
        st.number_input("Candidates", min_value=1, max_value=8, value=DEFAULTS.n, key="gen_n")
        st.number_input("Steps", min_value=1, max_value=100, value=DEFAULTS.steps, key="gen_steps")
    st.segmented_control(
        "Size",
        SIZES,
        default=DEFAULTS.width,
        required=True,
        format_func=lambda size: f"{size}²",
        key="gen_size",
    )
    st.number_input(
        "Seed",
        min_value=0,
        max_value=2**31 - 1,
        value=None,
        placeholder="Random",
        key="gen_seed",
    )
    st.segmented_control(
        "Guard mode", GUARD_MODES, default=DEFAULTS.guard_mode, required=True, key="gen_guard"
    )
    st.toggle("Rewrite the prompt", value=True, key="gen_rewrite")
    if (result := st.session_state.get("gen_edit_result")) is not None:
        st.divider()
        for verdict in st.session_state.gen_edit_verdicts:
            if verdict.decision == "warn":
                st.warning(verdict.summary, title="Guard warning")
            else:
                st.caption(f"Guard: {verdict.summary}")
        if result.fallback_reason:
            st.caption(f"Rewrite not used ({result.fallback_reason}): this is your prompt.")
        st.text_area("Prompt to generate", key="gen_edit", height=220)
        with st.container(horizontal=True):
            st.form_submit_button(
                "Generate from this prompt",
                type="primary",
                icon=":material/auto_awesome:",
                on_click=submit_edited,
                disabled=busy,
            )
            st.form_submit_button("Clear", icon=":material/close:", on_click=clear_edit)
    if notice := st.session_state.pop("gen_notice", None):
        st.warning(notice)

with right:
    gallery()
