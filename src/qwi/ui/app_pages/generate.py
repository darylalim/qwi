"""Generate page (PLAN.md §5.6): the form, live status while a job runs, and the candidates.

Submitting runs in the button's callback, before the script, so the same rerun already shows
the form disabled. The results fragment polls the job once a second only while it runs, and
reruns the page when it ends so the form comes back. Errors get their §5.11 banner, one-click
cheaper retries, and the run's log.
"""

import streamlit as st

from qwi.config import DEFAULTS
from qwi.pipeline import GenerateRequest
from qwi.ui import components, view
from qwi.ui.jobs import Busy, JobSnapshot

SIZES = [512, 768, 1024]
GUARD_MODES = ["strict", "default", "loose"]
CARD_WIDTH = 384  # px: the "M" thumbnail, 768 device pixels on Retina (§5.6); sizes come in M5c

jobs = components.job_manager()


def submit() -> None:
    state = st.session_state
    prompt = state.gen_prompt.strip()
    if not prompt:
        state.gen_notice = "Enter a prompt first."
        return
    request = GenerateRequest(
        prompt,
        n=state.gen_n,
        steps=state.gen_steps,
        width=state.gen_size,
        height=state.gen_size,
        seed=state.gen_seed,
        guard_mode=state.gen_guard,
        rewrite=state.gen_rewrite,
    )
    try:
        jobs.submit(request)
    except Busy:
        state.gen_notice = "A run is already in progress."


def cancel() -> None:
    jobs.cancel()


def retry(request: GenerateRequest, kind: str) -> None:
    try:
        jobs.submit(view.retry_request(request, kind))
    except Busy:
        st.session_state.gen_notice = "A run is already in progress."


def show_candidates(run: view.RunView, prompt: str, running: bool) -> None:
    if not run.candidates:
        return
    with st.container(horizontal=True, gap="medium"):
        for c in run.candidates:
            with st.container(border=True, width=CARD_WIDTH):
                st.image(
                    str(c.path),
                    width="stretch",
                    output_format="PNG",  # shown as saved, not re-encoded as JPEG
                    alt=f"{prompt} (seed {c.seed})",
                )
                caption = (
                    c.caption if c.prob is not None or running else f"{c.caption} · not ranked"
                )
                st.caption(caption)
                with st.container(horizontal=True, vertical_alignment="center"):
                    if c.best:
                        st.badge("Best", color="violet", icon=":material/star:")
                    st.download_button(
                        "Download",
                        data=c.path.read_bytes,
                        file_name=f"{run.run_dir.name if run.run_dir else 'qwi'}_{c.path.name}",
                        mime="image/png",
                        on_click="ignore",
                        icon=":material/download:",
                        key=f"download_{c.path}",
                    )


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
    elif run.status == "cancelled":
        st.info(f"Cancelled after {len(run.candidates)} of {run.n} candidates.", title="Cancelled")
    show_error(snap, run)
    if run.low_confidence and run.rerank is not None:
        best = max(run.rerank.probs)
        st.warning(
            f"The best candidate's probability is {best:.2f}, below {run.rerank.min_score}.",
            title="No candidate clearly matched the prompt",
        )


snapshot = jobs.snapshot()
busy = snapshot is not None and snapshot.running


@st.fragment(run_every="1s" if busy else None)
def results() -> None:
    snap = jobs.snapshot()
    if snap is None:
        st.caption("Results appear here.")
        return
    if snap.running != busy:  # it started or ended since the page ran: update the form too
        st.rerun()
    run = view.summarize(snap)
    for stage in run.stages:
        st.status(stage.label, state=stage.state)
    if run.progress is not None:
        st.progress(run.progress.fraction, text=run.progress.text)
    if snap.running:
        cancelling = snap.request.cancel.is_set()
        st.button(
            "Cancelling…" if cancelling else "Cancel",
            icon=":material/cancel:",
            on_click=cancel,
            disabled=cancelling,
        )
    for verdict in run.warnings:
        st.warning(verdict.summary, title="Guard warning")
    for risk in run.swap_risks:
        st.warning(
            f"{view.STAGE_LABELS.get(risk.stage, risk.stage)} needs about {risk.est_gb:.1f} GB "
            f"and {risk.avail_gb:.1f} GB is free, so macOS may swap. The run continues; it stops "
            "if swapping starts. Closing other apps helps.",
            title="Swap risk",
        )
    if run.rewrite is not None:
        with st.expander("Rewritten prompt"):
            st.code(run.rewrite.text, language=None, wrap_lines=True)
        if run.rewrite.fallback_reason:
            st.caption(f"Generated from your prompt instead: {run.rewrite.fallback_reason}")
    show_outcome(snap, run)
    show_candidates(run, snap.request.prompt, snap.running)


left, right = st.columns([1, 2], gap="large")
with left, st.form("generate", border=False):
    st.text_area("Prompt", key="gen_prompt", height=140, placeholder="Describe the image")
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
    st.form_submit_button(
        "Generate",
        type="primary",
        icon=":material/auto_awesome:",
        on_click=submit,
        disabled=busy,
    )
    if notice := st.session_state.pop("gen_notice", None):
        st.warning(notice)

with right:
    results()
