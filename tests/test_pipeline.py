"""`Pipeline.run` with fake stages: lifecycle, events, cancel, failures, lock."""

import errno
import threading
from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from qwi import errors, lock, logs, memory, paths, preflight, record
from qwi.config import DEFAULTS, GiB
from qwi.errors import (
    DeviceOutOfMemory,
    GenerationCancelled,
    InsufficientDisk,
    ModelUnavailable,
    PipelineBusy,
    RunNotContinuable,
)
from qwi.events import (
    CandidateReady,
    DenoiseStep,
    Event,
    GuardVerdict,
    PreflightWarning,
    RerankDone,
    RewriteReady,
    RunStarted,
    StageFinished,
    StageStarted,
)
from qwi.pipeline import GenerateRequest, Pipeline, RewriteResult, continue_seeds
from qwi.record import RunRecord
from qwi.stages import guard, rerank, rewrite
from qwi.stages.encode import PromptEmbeds
from qwi.stages.generate import Candidate

LIMIT = 26800603136  # max_recommended_working_set_size on the M2 Max (24.96 GiB)
PAGE = 16384


def vm(free_bytes: int, swapouts: int = 0) -> memory.VmStat:
    """`swapouts` is a page count, like vm_stat's counter."""
    pages = {"Pages free": free_bytes // PAGE, "Pages inactive": 0, "Pages purgeable": 0}
    return memory.VmStat(PAGE, {**pages, "Swapouts": swapouts})


class FakeStage:
    """Calls the registered hooks like mflux does, and can raise at a given candidate."""

    def __init__(self, callbacks: Sequence[Any]) -> None:
        self.callbacks = list(callbacks)
        self.loaded = self.unloaded = False
        self.fail_at: tuple[int, BaseException] | None = None
        self.encoded: tuple[list[str], str] | None = None

    def load(self) -> None:
        self.loaded = True

    def encode(self, prompts: Sequence[str], encoder: str = "A-q4") -> list[PromptEmbeds]:
        import mlx.core as mx

        self.encoded = (list(prompts), encoder)
        return [PromptEmbeds(p, mx.zeros((1, 12, 8)), mx.ones((1, 12)), encoder) for p in prompts]

    def __call__(
        self, prompt: str, seeds: Sequence[int], *, width: int, height: int, steps: int
    ) -> Generator[Candidate]:
        for index, seed in enumerate(seeds):
            if self.fail_at and self.fail_at[0] == index:
                raise self.fail_at[1]
            config = SimpleNamespace(num_inference_steps=steps)
            for t in range(steps):
                for hook in self.callbacks:
                    hook.call_in_loop(
                        t=t, seed=seed, prompt=prompt, latents=None, config=config, time_steps=None
                    )
            yield Candidate(seed, Image.new("RGB", (width, height)), 1.5, 13.8 + index / 10)

    def unload(self) -> None:
        self.unloaded = True


class FakeGuard:
    """Returns a fixed guard reply, parsed like the real stage does."""

    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.seen: list[str] = []
        self.loaded = self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def __call__(self, text: str) -> guard.Verdict:
        self.seen.append(text)
        return guard.parse(self.raw)

    def unload(self) -> None:
        self.unloaded = True


SAFE = "Safety: Safe\nCategories: None"
REWRITE = "A glossy red apple on a wooden table, soft window light from the left"


class FakeRewrite:
    """Returns a fixed model reply, post-processed like the real stage."""

    def __init__(self, raw: str, style: str) -> None:
        self.raw, self.style = raw, style
        self.seen: list[str] = []
        self.loaded = self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def __call__(self, prompt: str) -> rewrite.Rewritten:
        self.seen.append(prompt)
        text = rewrite.clean(self.raw)
        final, reason = rewrite.postprocess(prompt, text, 200)
        return rewrite.Rewritten(final, text, reason, self.style)

    def unload(self) -> None:
        self.unloaded = True


class FakeRerank:
    """Returns the next margin from a fixed list, one per image, like the real stage's Score."""

    def __init__(self, margins: Sequence[float], instruction: str) -> None:
        self.margins = list(margins)
        self.instruction = instruction
        self.seen: list[tuple[str, tuple[int, int]]] = []
        self.loaded = self.unloaded = False

    def load(self) -> None:
        self.loaded = True

    def __call__(self, query: str, image: Image.Image) -> rerank.Score:
        self.seen.append((query, image.size))
        margin = self.margins[len(self.seen) - 1]
        return rerank.Score(margin, rerank.sigmoid(margin), False)

    def unload(self) -> None:
        self.unloaded = True


class Harness:
    def __init__(self) -> None:
        self.stages: list[FakeStage] = []
        self.events: list[Event] = []
        self.configure: Callable[[FakeStage], None] = lambda stage: None
        self.on_event: Callable[[Event], None] = lambda event: None
        self.resolved: list[str] = []
        self.startup: Callable[[list[str]], None] = lambda roles: None
        self.guard_reply = SAFE
        self.guards: list[FakeGuard] = []
        self.rewrite_reply = REWRITE
        self.rewriters: list[FakeRewrite] = []
        self.margins = [0.5, 2.0, -1.0, 1.0]  # candidate 1 wins
        self.rerankers: list[FakeRerank] = []
        self.pipeline = Pipeline(
            self._make,
            resolve=self._resolve,
            startup=lambda r: self.startup(r),
            make_guard=self._make_guard,
            make_rewrite=self._make_rewrite,
            make_rerank=self._make_rerank,
        )

    def _make_rerank(self, model_path: Path, instruction: str) -> FakeRerank:
        self.rerankers.append(FakeRerank(self.margins, instruction))
        return self.rerankers[-1]

    def _make_rewrite(self, model_path: Path, style: str) -> FakeRewrite:
        self.rewriters.append(FakeRewrite(self.rewrite_reply, style))
        return self.rewriters[-1]

    def _make_guard(self, model_path: Path) -> FakeGuard:
        self.guards.append(FakeGuard(self.guard_reply))
        return self.guards[-1]

    def _resolve(self, role: str) -> Path:
        self.resolved.append(role)
        return Path("/weights") / role

    def _make(self, model_path: Path, callbacks: Sequence[object], _: object) -> FakeStage:
        stage = FakeStage(callbacks)
        self.configure(stage)
        self.stages.append(stage)
        return stage

    def _emit(self, event: Event) -> None:
        self.events.append(event)
        self.on_event(event)

    def run(self, **overrides: Any) -> RunRecord:
        fields: dict[str, Any] = {
            "prompt": "a red apple",
            "n": 3,
            "steps": 2,
            "width": 64,
            "height": 64,
            "seed": 100,
        }
        request = GenerateRequest(**(fields | overrides))
        return self.pipeline.run(request, self._emit)

    def kinds(self) -> list[str]:
        return [type(e).__name__ for e in self.events]


@pytest.fixture(autouse=True)
def machine(monkeypatch: pytest.MonkeyPatch) -> None:
    """24.96 GiB wired limit, 30 GB free memory, no swap, unless a test changes them."""
    monkeypatch.setattr(preflight, "wired_limit_bytes", lambda: LIMIT)
    monkeypatch.setattr(memory, "vm_stat", lambda: vm(30 * 10**9))


@pytest.fixture
def h() -> Harness:
    return Harness()


def run_dir(rec: RunRecord) -> Path:
    (path,) = paths.runs_dir().iterdir()
    assert record.read(path) == rec  # what's on disk is what run() returned
    return path


def test_a_run_writes_candidates_and_a_final_record(h: Harness) -> None:
    rec = h.run()
    run = run_dir(rec)
    assert rec.status == "ok"
    assert rec.seeds == [100, 101, 102]
    assert [c["path"] for c in rec.candidates] == [
        "cand_0_seed100.png",
        "cand_1_seed101.png",
        "cand_2_seed102.png",
    ]
    assert all((run / c["path"]).is_file() for c in rec.candidates)
    assert rec.config == {
        "width": 64,
        "height": 64,
        "n": 3,
        "steps": 2,
        "encoder": "A-q4",
        "rewrite": "long",
        "base_seed": 100,
        "allow_swap": False,
    }
    assert rec.preflight["generate"]["decision"] == "ok"
    assert rec.peak_mem_gb["generate"] == pytest.approx(14.0)
    assert rec.swap_gb == {"guard": 0.0, "rewrite": 0.0, "generate": 0.0, "rerank": 0.0}
    assert set(rec.timings_s) == {"guard", "rewrite", "encode", "generate", "rerank"}
    assert set(rec.model_revisions) == {"guard", "rewrite", "image", "rerank"}
    assert h.resolved == ["guard", "rewrite", "image", "rerank"]
    assert (rec.rewritten_prompt, rec.rewrite_fallback_reason) == (REWRITE, None)
    assert (rec.prompt_source, rec.rerank_query) == ("rewrite", "a red apple")
    assert rec.guard["decision"] == "allow"
    assert rec.guard["rewrite"]["level"] == "Safe"
    assert h.guards[0].seen == ["a red apple", REWRITE]  # input pass, then the rewrite pass
    assert h.stages[0].encoded == ([REWRITE], "A-q4")  # generation uses the rewrite
    assert (h.guards[0].loaded, h.guards[0].unloaded) == (True, True)
    assert (h.rewriters[0].loaded, h.rewriters[0].unloaded) == (True, True)
    assert h.stages[0].loaded
    assert h.stages[0].unloaded
    assert (h.rerankers[0].loaded, h.rerankers[0].unloaded) == (True, True)
    assert rec.best_index == 1


def test_events_arrive_in_order(h: Harness) -> None:
    h.run(n=2)
    assert h.kinds() == [
        "RunStarted",
        "StageStarted",
        "StageFinished",
        "GuardVerdict",
        "StageStarted",
        "StageFinished",
        "RewriteReady",
        "GuardVerdict",
        "StageStarted",
        "StageFinished",
        "StageStarted",
        *["DenoiseStep"] * 2,
        "CandidateReady",
        *["DenoiseStep"] * 2,
        "CandidateReady",
        "StageFinished",
        "StageStarted",
        "StageFinished",
        "RerankDone",
    ]
    steps = [e for e in h.events if isinstance(e, DenoiseStep)]
    assert [(s.candidate, s.step, s.total) for s in steps] == [
        (0, 1, 2),
        (0, 2, 2),
        (1, 1, 2),
        (1, 2, 2),
    ]
    assert isinstance(h.events[0], RunStarted)
    assert h.events[0].seeds == [100, 101]
    stages = ["guard", "rewrite", "encode", "generate", "rerank"]
    assert [e.stage for e in h.events if isinstance(e, StageStarted)] == stages
    assert [e.stage for e in h.events if isinstance(e, StageFinished)] == stages
    ready = [e for e in h.events if isinstance(e, CandidateReady)]
    assert [(e.index, e.seed, e.path.name) for e in ready][1] == (1, 101, "cand_1_seed101.png")
    (generated,) = [e for e in h.events if isinstance(e, StageFinished) and e.stage == "generate"]
    assert generated.peak_gb == pytest.approx(13.9)
    assert isinstance(h.events[-1], RerankDone)


def test_random_base_seed_is_recorded(h: Harness) -> None:
    rec = h.run(seed=None)
    base = rec.config["base_seed"]
    assert rec.seeds == [base, base + 1, base + 2]


def test_run_log_records_the_run(h: Harness) -> None:
    rec = h.run()
    lines = logs.read(run_dir(rec) / "run.log")
    names = [line["event"] for line in lines]
    assert names[0] == "run_started"
    assert names[-1] == "run_finished"
    assert names.count("candidate_ready") == 3
    assert "preflight" in names


def test_cancel_on_a_denoising_step_keeps_finished_candidates(h: Harness) -> None:
    cancel = threading.Event()

    def on_event(event: Event) -> None:
        if isinstance(event, DenoiseStep) and (event.candidate, event.step) == (1, 1):
            cancel.set()

    h.on_event = on_event
    rec = h.run(cancel=cancel)
    assert rec.status == "cancelled"
    assert rec.error is None
    assert [c["seed"] for c in rec.candidates] == [100]
    assert h.stages[0].unloaded
    assert not lock.is_held()


def test_cancel_between_candidates(h: Harness) -> None:
    cancel = threading.Event()

    def on_event(event: Event) -> None:
        if isinstance(event, CandidateReady):
            cancel.set()

    h.on_event = on_event
    h.configure = lambda stage: setattr(stage, "callbacks", [])  # no step hooks: per-candidate only
    rec = h.run(cancel=cancel)
    assert (rec.status, len(rec.candidates)) == ("cancelled", 1)
    logs_ = logs.read(run_dir(rec) / "run.log", "warning")
    assert any(line["msg"] == "Cancelled after 1 of 3 candidates" for line in logs_)


def test_cancel_before_loading_never_builds_the_stage(h: Harness) -> None:
    cancel = threading.Event()
    cancel.set()
    rec = h.run(cancel=cancel)
    assert (rec.status, rec.candidates, h.stages) == ("cancelled", [], [])


def test_mlx_oom_mid_stage_is_device_out_of_memory(h: Harness) -> None:
    oom = RuntimeError("[metal::malloc] Resource limit (499000) exceeded.")
    h.configure = lambda stage: setattr(stage, "fail_at", (1, oom))
    rec = h.run()
    assert rec.status == "error"
    assert rec.error is not None
    assert (rec.error["type"], rec.error["stage"]) == ("DeviceOutOfMemory", "generate")
    assert errors.exit_code_for(rec.status, rec.error["type"]) == 4
    assert len(rec.candidates) == 1
    assert h.stages[0].unloaded
    (line,) = logs.read(run_dir(rec) / "run.log", "error")
    assert "RuntimeError: [metal::malloc]" in line["traceback"]


def test_unexpected_errors_exit_1_and_still_evict(h: Harness) -> None:
    h.configure = lambda stage: setattr(stage, "fail_at", (0, ValueError("boom")))
    rec = h.run()
    assert rec.error is not None
    assert (rec.status, rec.error["type"], rec.error["message"]) == ("error", "ValueError", "boom")
    assert errors.exit_code_for(rec.status, rec.error["type"]) == 1
    assert h.stages[0].unloaded
    assert not lock.is_held()


def test_ctrl_c_inside_the_stage_is_cancelled(h: Harness) -> None:
    h.configure = lambda stage: setattr(stage, "fail_at", (2, KeyboardInterrupt()))
    rec = h.run()
    assert (rec.status, len(rec.candidates)) == ("cancelled", 2)


def test_swap_risk_runs_and_records_the_decision(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory, "vm_stat", lambda: vm(12 * 10**9))
    rec = h.run(width=768, height=768)  # 14.95 GB estimate, ~9.9 GB available after headroom
    assert rec.status == "ok"
    assert rec.preflight["generate"]["decision"] == "swap_risk"
    (warning,) = [e for e in h.events if isinstance(e, PreflightWarning)]  # generate only
    assert warning.stage == "generate"
    assert (round(warning.est_gb, 2), round(warning.avail_gb, 2)) == (
        rec.preflight["generate"]["est_gb"],
        rec.preflight["generate"]["avail_gb"],
    )


def test_no_swap_risk_sends_no_warning(h: Harness) -> None:
    h.run()
    assert not [e for e in h.events if isinstance(e, PreflightWarning)]


def test_swap_during_the_stage_is_recorded(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    # guard, rewrite: pre-flight, watchdog start, delta; generate: pre-flight, start, then swap.
    readings = iter([vm(30 * 10**9, swapouts=0)] * 8)
    swapped = vm(30 * 10**9, swapouts=6104)  # pages: 6104 x 16 KiB = 0.1 GB
    monkeypatch.setattr(memory, "vm_stat", lambda: next(readings, swapped))
    rec = h.run()
    assert rec.swap_gb["generate"] == pytest.approx(0.1)


def test_a_second_run_while_one_holds_the_lock_is_busy(h: Harness) -> None:
    with lock.RunLock(), pytest.raises(PipelineBusy):
        h.run()
    assert not paths.runs_dir().exists()  # no run was created


def test_the_lock_is_held_during_the_run_and_released_after(h: Harness) -> None:
    held: list[bool] = []
    h.on_event = lambda event: (
        held.append(lock.is_held()) if isinstance(event, StageStarted) else None
    )
    h.run()
    assert held == [True] * 5  # guard, rewrite, encode, generate and rerank
    assert not lock.is_held()


def test_a_worker_thread_run_is_busy_to_others_and_cancels_from_outside(h: Harness) -> None:
    """How the UI drives it (§5.6): run on a worker, submit and cancel from another thread."""
    cancel, denoising, resume = threading.Event(), threading.Event(), threading.Event()
    result: list[RunRecord] = []

    def on_event(event: Event) -> None:
        if isinstance(event, DenoiseStep) and (event.candidate, event.step) == (0, 1):
            denoising.set()
            assert resume.wait(5)

    h.on_event = on_event
    worker = threading.Thread(target=lambda: result.append(h.run(cancel=cancel)))
    worker.start()
    assert denoising.wait(5)
    assert lock.is_held()
    with pytest.raises(PipelineBusy):
        Harness().run()
    cancel.set()
    resume.set()
    worker.join(5)

    (rec,) = result
    assert rec.status == "cancelled"
    assert rec.candidates == []
    assert h.stages[0].unloaded
    assert not lock.is_held()
    assert "run_finished" in [line["event"] for line in logs.read(run_dir(rec) / "run.log")]


# --- Fault injection (§5.11 "Tests"): every failure path the M1 pipeline has ---------------


def _raise(exc: BaseException) -> Callable[..., Any]:
    def fail(*_: Any) -> Any:
        raise exc

    return fail


def _fail_at(index: int, exc: BaseException) -> Callable[[Harness, pytest.MonkeyPatch], None]:
    def setup(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
        h.configure = lambda stage: setattr(stage, "fail_at", (index, exc))

    return setup


def _startup(exc: BaseException) -> Callable[[Harness, pytest.MonkeyPatch], None]:
    def setup(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
        h.startup = _raise(exc)

    return setup


def _resolve_missing(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    h.pipeline._resolve = _raise(ModelUnavailable("image", "not in the local HF cache"))


def _tiny_gpu(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight, "wired_limit_bytes", lambda: 8 * 10**9)


def _disk_full_on_second_save(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    saves = iter([None])
    real_save = Image.Image.save

    def save(self: Image.Image, fp: Any, *args: Any, **kwargs: Any) -> None:
        if next(saves, "full") == "full":
            raise OSError(errno.ENOSPC, "No space left on device")
        real_save(self, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", save)


OOM = "[metal::malloc] Attempting to allocate 20100677632 bytes which is greater than the maximum"
FAULTS = [
    # id, setup, status, error type, exit code, candidates kept, stage built
    (
        "startup-model",
        _startup(ModelUnavailable("image", "missing")),
        "error",
        "ModelUnavailable",
        5,
        0,
        False,
    ),
    (
        "startup-disk",
        _startup(InsufficientDisk("a run", 0, 5 * 10**9, 20 * 10**9)),
        "error",
        "InsufficientDisk",
        6,
        0,
        False,
    ),
    ("resolve-model", _resolve_missing, "error", "ModelUnavailable", 5, 0, False),
    ("preflight-limit", _tiny_gpu, "error", "InsufficientMemory", 4, 0, False),
    ("mlx-oom", _fail_at(1, RuntimeError(OOM)), "error", "DeviceOutOfMemory", 4, 1, True),
    (
        "swap-watchdog",
        _fail_at(2, DeviceOutOfMemory("generate", "swapped 2.1 GB")),
        "error",
        "DeviceOutOfMemory",
        4,
        2,
        True,
    ),
    ("disk-full", _disk_full_on_second_save, "error", "DiskFull", 6, 1, True),
    (
        "unexpected",
        _fail_at(1, RuntimeError("[gather] index out of bounds")),
        "error",
        "RuntimeError",
        1,
        1,
        True,
    ),
    ("cancelled", _fail_at(1, GenerationCancelled()), "cancelled", None, 130, 1, True),
    ("ctrl-c", _fail_at(0, KeyboardInterrupt()), "cancelled", None, 130, 0, True),
]


@pytest.mark.parametrize(
    ("setup", "status", "error_type", "code", "kept", "built"),
    [pytest.param(*fault[1:], id=fault[0]) for fault in FAULTS],
)
def test_every_failure_path(
    h: Harness,
    monkeypatch: pytest.MonkeyPatch,
    setup: Callable[[Harness, pytest.MonkeyPatch], None],
    status: str,
    error_type: str | None,
    code: int,
    kept: int,
    built: bool,
) -> None:
    setup(h, monkeypatch)
    rec = h.run(width=768, height=768)
    run = run_dir(rec)  # also checks run.json on disk == the returned record

    assert rec.status == status
    assert (rec.error or {}).get("type") == error_type
    assert errors.exit_code_for(rec.status, error_type) == code
    assert len(rec.candidates) == kept
    assert all((run / c["path"]).is_file() for c in rec.candidates)
    assert [stage.unloaded for stage in h.stages] == ([True] if built else [])
    assert not lock.is_held()

    error_lines = logs.read(run / "run.log", "error")
    if status == "error":
        (line,) = error_lines
        assert line["type"] == error_type
        assert "Traceback" in line["traceback"]
    else:
        assert error_lines == []


# --- Guard (§5.2): the input pass before any other model loads -----------------------------


def test_a_guard_block_ends_the_run_with_exit_2_before_generate(h: Harness) -> None:
    h.guard_reply = "Safety: Unsafe\nCategories: Violent"
    rec = h.run()
    run = run_dir(rec)
    assert (rec.status, rec.error, rec.candidates) == ("blocked", None, [])
    assert errors.exit_code_for(rec.status) == 2
    assert rec.guard == {
        "mode": "default",
        "input": {"level": "Unsafe", "categories": ["Violent"], "raw": h.guard_reply},
        "rewrite": None,
        "decision": "block",
        "reason": "input:Unsafe:Violent",
    }
    assert h.stages == []  # the image model never loaded
    assert h.resolved == ["guard"]
    assert h.guards[0].unloaded
    assert not lock.is_held()
    (verdict,) = [e for e in h.events if isinstance(e, GuardVerdict)]
    assert (verdict.check, verdict.decision) == ("input", "block")
    assert verdict.summary == "input prompt is Unsafe (Violent) (--guard-mode default)"
    (line,) = logs.read(run / "run.log", "warning")
    assert line["event"] == "blocked"
    assert line["msg"] == f"Blocked by guard: {verdict.summary}"


def test_a_guard_warning_is_logged_and_the_run_continues(h: Harness) -> None:
    h.guard_reply = "Safety: Controversial\nCategories: Violent"
    rec = h.run()
    assert rec.status == "ok"
    assert (rec.guard["decision"], rec.guard["reason"]) == ("warn", "input:Controversial:Violent")
    assert len(rec.candidates) == 3
    lines = logs.read(run_dir(rec) / "run.log", "warning")
    # The fake guard answers both passes the same way: one warning per pass.
    assert [(x["event"], x["msg"], x["check"]) for x in lines] == [
        ("guard_verdict", "warn", "input"),
        ("guard_verdict", "warn", "rewrite"),
    ]
    assert lines[1]["reason"] == "input:Controversial:Violent"  # the input wins a tie


def test_guard_mode_selects_the_policy_column(h: Harness) -> None:
    h.guard_reply = "Safety: Controversial\nCategories: Violent"
    rec = h.run(guard_mode="strict")
    assert (rec.status, rec.guard["mode"], rec.guard["decision"]) == ("blocked", "strict", "block")


def test_unparseable_guard_output_fails_closed(h: Harness) -> None:
    h.guard_reply = "I'm sorry, I can't help with that."
    rec = h.run(guard_mode="loose")
    assert rec.status == "blocked"
    assert rec.guard["input"]["level"] is None
    assert rec.guard["reason"] == "input:guard_parse_error"


def test_a_guard_failure_is_an_error_in_the_guard_stage(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: FakeGuard, text: str) -> guard.Verdict:
        raise RuntimeError("[metal::malloc] Resource limit (499000) exceeded.")

    monkeypatch.setattr(FakeGuard, "__call__", broken)
    rec = h.run()
    assert rec.error is not None
    assert (rec.error["type"], rec.error["stage"]) == ("DeviceOutOfMemory", "guard")
    assert h.guards[0].unloaded
    assert h.stages == []


# --- Rewrite (§5.1) and the guard's second pass (§5.2) -------------------------------------


def test_the_guard_stays_loaded_through_the_rewrite(h: Harness) -> None:
    loaded_during_rewrite: list[bool] = []
    real_call = FakeRewrite.__call__

    def spy(self: FakeRewrite, prompt: str) -> rewrite.Rewritten:
        loaded_during_rewrite.append(not h.guards[0].unloaded)
        return real_call(self, prompt)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(FakeRewrite, "__call__", spy)
        rec = h.run()
    assert rec.status == "ok"
    assert loaded_during_rewrite == [True]
    assert (h.guards[0].unloaded, h.rewriters[0].unloaded) == (
        True,
        True,
    )  # both gone before encoding


def test_no_rewrite_skips_the_stage_and_the_second_pass(h: Harness) -> None:
    rec = h.run(rewrite=False)
    assert rec.status == "ok"
    assert h.rewriters == []
    assert h.resolved == ["guard", "image", "rerank"]
    assert set(rec.model_revisions) == {"guard", "image", "rerank"}
    assert rec.config["rewrite"] == "none"
    assert (rec.rewritten_prompt, rec.prompt_source, rec.guard["rewrite"]) == ("", "user", None)
    assert h.guards[0].seen == ["a red apple"]
    assert h.stages[0].encoded == (["a red apple"], "A-q4")
    assert "rewrite" not in rec.timings_s
    assert not any(isinstance(e, RewriteReady) for e in h.events)


def test_rewrite_style_reaches_the_stage_and_the_record(h: Harness) -> None:
    rec = h.run(rewrite_style="long")
    assert h.rewriters[0].style == "long"
    assert rec.config["rewrite"] == "long"


def test_a_fallback_generates_from_the_user_prompt_without_a_second_pass(h: Harness) -> None:
    h.rewrite_reply = "Prompt: An apple, lovingly described."
    request_prompt = 'a crate stamped "FUJI"'
    rec = h.pipeline.run(
        GenerateRequest(request_prompt, n=1, steps=1, width=64, height=64, seed=1), h._emit
    )
    assert rec.status == "ok"
    assert rec.rewrite_fallback_reason == "quoted_text_missing"
    assert (rec.rewritten_prompt, rec.prompt_source) == ("An apple, lovingly described.", "user")
    assert h.guards[0].seen == [request_prompt]  # the input was already checked
    assert rec.guard["rewrite"] is None
    assert h.stages[0].encoded == ([request_prompt], "A-q4")
    (ready,) = [e for e in h.events if isinstance(e, RewriteReady)]
    assert (ready.fallback_reason, ready.style) == ("quoted_text_missing", "long")
    (line,) = logs.read(run_dir(rec) / "run.log", "warning")
    assert (line["event"], line["msg"]) == ("rewrite_ready", "fallback: quoted_text_missing")


def test_a_block_on_the_rewrite_pass_ends_the_run(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    def judge(self: FakeGuard, text: str) -> guard.Verdict:
        self.seen.append(text)
        unsafe = "Safety: Unsafe\nCategories: Violent"
        return guard.parse(SAFE if text == "a red apple" else unsafe)

    monkeypatch.setattr(FakeGuard, "__call__", judge)
    rec = h.run()
    assert rec.status == "blocked"
    assert errors.exit_code_for(rec.status) == 2
    assert (rec.guard["decision"], rec.guard["reason"]) == ("block", "rewrite:Unsafe:Violent")
    assert rec.guard["input"]["level"] == "Safe"
    assert rec.rewritten_prompt == REWRITE  # recorded before the block, for the record
    assert h.stages == []
    assert (h.guards[0].unloaded, h.rewriters[0].unloaded) == (True, True)
    verdicts = [(e.check, e.decision) for e in h.events if isinstance(e, GuardVerdict)]
    assert verdicts == [("input", "allow"), ("rewrite", "block")]


def test_an_input_block_never_builds_the_rewriter(h: Harness) -> None:
    h.guard_reply = "Safety: Unsafe\nCategories: Jailbreak"
    rec = h.run()
    assert (rec.status, h.rewriters, h.resolved) == ("blocked", [], ["guard"])


def test_a_rewrite_failure_is_an_error_in_the_rewrite_stage(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: FakeRewrite, prompt: str) -> rewrite.Rewritten:
        raise RuntimeError("[metal::malloc] Resource limit (499000) exceeded.")

    monkeypatch.setattr(FakeRewrite, "__call__", broken)
    rec = h.run()
    assert rec.error is not None
    assert (rec.error["type"], rec.error["stage"]) == ("DeviceOutOfMemory", "rewrite")
    assert (h.guards[0].unloaded, h.rewriters[0].unloaded) == (True, True)
    assert h.stages == []


# --- Rewrite only, then generate from the edited rewrite (§5.6) ----------------------------


def rewrite_only(h: Harness, **overrides: Any) -> RewriteResult:
    return h.pipeline.rewrite_only(GenerateRequest("a red apple", **overrides), h._emit)


def test_rewrite_only_guards_rewrites_and_evicts_without_a_run(h: Harness) -> None:
    held: list[bool] = []
    h.on_event = lambda event: (
        held.append(lock.is_held()) if isinstance(event, StageStarted) else None
    )
    startup: list[list[str]] = []
    h.startup = startup.append
    result = rewrite_only(h, rewrite=False, rewrite_style="short")  # the toggle doesn't apply

    assert (result.status, result.error, result.style) == ("ok", None, "short")
    assert (result.user_prompt, result.rewritten_prompt, result.fallback_reason) == (
        "a red apple",
        REWRITE,
        None,
    )
    assert result.prompt == REWRITE
    assert (result.guard["decision"], result.guard["rewrite"]["level"]) == ("allow", "Safe")
    assert h.guards[0].seen == ["a red apple", REWRITE]
    assert (h.guards[0].unloaded, h.rewriters[0].unloaded) == (True, True)
    assert h.rewriters[0].style == "short"
    assert (h.stages, h.rerankers) == ([], [])
    assert startup == [["guard", "rewrite"]]
    assert held == [True, True]
    assert not lock.is_held()
    assert not paths.runs_dir().exists()  # nothing generated, no run folder
    assert h.kinds() == [
        "StageStarted",
        "StageFinished",
        "GuardVerdict",
        "StageStarted",
        "StageFinished",
        "RewriteReady",
        "GuardVerdict",
    ]


def test_rewrite_only_on_a_fallback_offers_the_user_prompt(h: Harness) -> None:
    h.rewrite_reply = ""
    result = rewrite_only(h)
    assert (result.status, result.fallback_reason, result.prompt) == ("ok", "empty", "a red apple")
    assert result.guard["rewrite"] is None  # no second pass on a fallback


def test_rewrite_only_blocked_on_input_never_builds_the_rewriter(h: Harness) -> None:
    h.guard_reply = "Safety: Unsafe\nCategories: Violent"
    result = rewrite_only(h)
    assert (result.status, result.error, result.rewritten_prompt) == ("blocked", None, "")
    assert result.guard["reason"] == "input:Unsafe:Violent"
    assert h.rewriters == []
    assert h.guards[0].unloaded
    assert not lock.is_held()


def test_rewrite_only_blocked_on_the_rewrite_pass(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    def judge(self: FakeGuard, text: str) -> guard.Verdict:
        return guard.parse(SAFE if text == "a red apple" else "Safety: Unsafe\nCategories: Violent")

    monkeypatch.setattr(FakeGuard, "__call__", judge)
    result = rewrite_only(h)
    assert (result.status, result.guard["reason"]) == ("blocked", "rewrite:Unsafe:Violent")
    assert (h.guards[0].unloaded, h.rewriters[0].unloaded) == (True, True)


def test_rewrite_only_cancelled_before_the_rewrite(h: Harness) -> None:
    cancel = threading.Event()
    h.on_event = lambda event: cancel.set() if isinstance(event, GuardVerdict) else None
    result = h.pipeline.rewrite_only(GenerateRequest("a red apple", cancel=cancel), h._emit)
    assert result.status == "cancelled"
    assert h.rewriters == []  # the rewrite pre-flight is the last chance to cancel
    assert h.guards[0].unloaded


def test_rewrite_only_failure_is_an_error_and_evicts(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: FakeRewrite, prompt: str) -> rewrite.Rewritten:
        raise RuntimeError("[metal::malloc] Resource limit (499000) exceeded.")

    monkeypatch.setattr(FakeRewrite, "__call__", broken)
    logs.app_log()
    result = rewrite_only(h)
    assert result.status == "error"
    assert result.error is not None
    assert (result.error["type"], result.error["stage"]) == ("DeviceOutOfMemory", "rewrite")
    assert "log" not in result.error  # no run.log: it's in the app log
    assert (h.guards[0].unloaded, h.rewriters[0].unloaded) == (True, True)
    assert not lock.is_held()
    (line,) = [x for x in logs.read(paths.logs_dir() / "qwi.log") if x["level"] == "ERROR"]
    assert line["stage"] == "rewrite"


def test_rewrite_only_while_a_run_holds_the_lock_is_busy(h: Harness) -> None:
    with lock.RunLock(), pytest.raises(PipelineBusy):
        rewrite_only(h)
    assert h.guards == []


EDITED = "A glossy green apple on a slate table"


def test_an_edited_rewrite_generates_as_given_and_reranks_against_the_original(
    h: Harness,
) -> None:
    rec = h.run(
        prompt=EDITED, rewrite=False, rerank_query="a red apple", prompt_source="edited_rewrite"
    )
    assert rec.status == "ok"
    assert h.guards[0].seen == [EDITED]  # the input pass checks the edited text
    assert h.rewriters == []
    assert h.stages[0].encoded == ([EDITED], "A-q4")
    assert h.rerankers[0].seen == [("a red apple", (64, 64))] * 3
    assert (rec.user_prompt, rec.rewritten_prompt, rec.prompt_source, rec.rerank_query) == (
        "a red apple",
        EDITED,
        "edited_rewrite",
        "a red apple",
    )
    assert rec.config["rewrite"] == "none"
    assert record.read(run_dir(rec)) == rec


def test_rerank_query_defaults_to_the_prompt(h: Harness) -> None:
    rec = h.run(rewrite=False, rerank_query=None)
    assert (rec.prompt_source, rec.rerank_query) == ("user", "a red apple")


def test_continue_seeds_of_an_edited_rewrite_keeps_the_edit(h: Harness) -> None:
    h.run(prompt=EDITED, rewrite=False, rerank_query="a red apple", prompt_source="edited_rewrite")
    rec = h.pipeline.run(continue_seeds(last_run(h), GenerateRequest("")), h._emit)
    assert h.stages[-1].encoded == ([EDITED], "A-q4")
    assert h.rerankers[-1].seen[0][0] == "a red apple"
    assert (rec.user_prompt, rec.prompt_source) == ("a red apple", "edited_rewrite")


@pytest.mark.parametrize(
    "overrides",
    [{"prompt_source": "rewrite"}, {"prompt_source": "edited_rewrite"}],
    ids=["unknown-source", "edited-with-rewrite-on"],
)
def test_a_request_refuses_an_inconsistent_prompt_source(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"prompt_source|rewrite=False"):
        GenerateRequest("p", **overrides)


# --- Rerank (§5.3): score every candidate against the user's prompt, link best.png ----------


def test_rerank_records_scores_and_links_best(h: Harness) -> None:
    rec = h.run()
    run = run_dir(rec)
    assert [c["margin"] for c in rec.candidates] == [0.5, 2.0, -1.0]
    assert [c["prob"] for c in rec.candidates] == [0.6225, 0.8808, 0.2689]
    assert [c["rank"] for c in rec.candidates] == [2, 1, 3]
    assert [c["resized"] for c in rec.candidates] == [False] * 3
    assert rec.best_index == 1
    assert rec.rerank == {
        "instruction": DEFAULTS.rerank_instruction,
        "query": "user_prompt",
        "min_score": 0.5,
        "low_confidence": False,
    }
    best = run / "best.png"
    assert best.stat().st_ino == (run / "cand_1_seed101.png").stat().st_ino  # a hard link
    assert rec.preflight["rerank"]["est_gb"] == preflight.STATIC_GB["rerank"]
    done = h.events[-1]
    assert isinstance(done, RerankDone)
    assert (done.best_index, done.best_path, done.ranks) == (1, best, [2, 1, 3])
    assert done.seeds == [100, 101, 102]
    assert done.paths == [run / c["path"] for c in rec.candidates]
    assert done.probs == pytest.approx([0.6225, 0.8808, 0.2689], abs=1e-4)


def test_rerank_scores_the_user_prompt_and_the_images_in_memory(h: Harness) -> None:
    h.run()
    # The original prompt, not the rewrite (§5.3 "Inputs"); every decoded image, in seed order.
    assert h.rerankers[0].seen == [("a red apple", (64, 64))] * 3


def test_rerank_options_reach_the_stage_and_the_record(h: Harness) -> None:
    rec = h.run(rerank_instruction="Is this an apple?", min_score=0.9)
    assert h.rerankers[0].instruction == "Is this an apple?"
    assert (rec.rerank["instruction"], rec.rerank["min_score"]) == ("Is this an apple?", 0.9)
    assert rec.rerank["low_confidence"]  # the best prob, 0.88, is below 0.9


def test_low_confidence_warns_but_keeps_the_results(h: Harness) -> None:
    h.margins = [-2.0, -0.5, -1.0]
    rec = h.run()
    run = run_dir(rec)
    assert (rec.status, rec.best_index, rec.rerank["low_confidence"]) == ("ok", 1, True)
    assert (run / "best.png").is_file()
    (line,) = [x for x in logs.read(run / "run.log", "warning") if x["event"] == "rerank_done"]
    assert line["best_prob"] == pytest.approx(0.3775)
    done = h.events[-1]
    assert isinstance(done, RerankDone)
    assert done.low_confidence


def test_a_generation_failure_skips_rerank(h: Harness) -> None:
    h.configure = lambda stage: setattr(stage, "fail_at", (2, ValueError("boom")))
    rec = h.run()
    run = run_dir(rec)
    assert (rec.status, len(rec.candidates), rec.best_index) == ("error", 2, None)
    assert h.rerankers == []  # never built: the kept candidates stay unranked (§5.11)
    assert "rerank" not in rec.preflight
    assert not (run / "best.png").exists()


def test_cancel_during_rerank_leaves_the_run_unranked(h: Harness) -> None:
    cancel = threading.Event()
    h.on_event = lambda event: (
        cancel.set() if isinstance(event, StageStarted) and event.stage == "rerank" else None
    )
    rec = h.run(cancel=cancel)
    run = run_dir(rec)
    assert (rec.status, len(rec.candidates), rec.best_index) == ("cancelled", 3, None)
    assert len(h.rerankers[0].seen) == 1  # stopped after the first image
    assert h.rerankers[0].unloaded
    assert not (run / "best.png").exists()
    assert not lock.is_held()


def test_swap_during_rerank_stops_the_stage(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    real = memory.SwapWatchdog
    readings = iter([0, 3 * GiB])  # rerank: the baseline, then 3 GiB swapped after image 1

    def watchdog(stage: str, limit: int = DEFAULTS.swap_limit_bytes, *_: Any) -> Any:
        if stage != "rerank":
            return real(stage, limit, lambda: 0)
        return real(stage, limit, lambda: next(readings, 3 * GiB))

    monkeypatch.setattr(memory, "SwapWatchdog", watchdog)
    rec = h.run()
    assert rec.error is not None
    assert (rec.status, rec.error["type"], rec.error["stage"]) == (
        "error",
        "DeviceOutOfMemory",
        "rerank",
    )
    assert len(h.rerankers[0].seen) == 1
    assert h.rerankers[0].unloaded
    assert (rec.best_index, rec.swap_gb["rerank"]) == (None, 3.22)


def test_allow_swap_raises_the_rerank_watchdog_limit(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    limits: dict[str, int] = {}
    real = memory.SwapWatchdog

    def watchdog(stage: str, limit: int = DEFAULTS.swap_limit_bytes, *_: Any) -> Any:
        limits[stage] = limit
        return real(stage, limit, lambda: 0)

    monkeypatch.setattr(memory, "SwapWatchdog", watchdog)
    h.run(allow_swap=True)
    assert limits["rerank"] == DEFAULTS.allow_swap_limit_bytes
    assert limits["guard"] == DEFAULTS.swap_limit_bytes


# --continue-seeds (§5.3 "Low-confidence warning")


def last_run(h: Harness) -> Path:
    return [e for e in h.events if isinstance(e, RunStarted)][-1].run_dir


def test_continue_seeds_runs_more_seeds_from_the_source_prompt(h: Harness) -> None:
    source = h.run(rewrite_style="long")
    source_dir = last_run(h)
    rec = h.pipeline.run(continue_seeds(source_dir, GenerateRequest("")), h._emit)
    assert rec.status == "ok"
    assert (rec.seeds, rec.continued_from) == ([103, 104, 105], source_dir.name)
    assert [c["seed"] for c in rec.candidates] == [103, 104, 105]  # only its own candidates
    assert [c["rank"] for c in rec.candidates] == [2, 1, 3]
    assert len(h.rewriters) == 1  # no new rewrite
    assert h.guards[-1].seen == ["a red apple"]  # the input guard still runs
    assert h.stages[-1].encoded == ([REWRITE], "A-q4")
    assert h.rerankers[-1].seen == [("a red apple", (64, 64))] * 3
    assert (rec.user_prompt, rec.rewritten_prompt, rec.prompt_source) == (
        "a red apple",
        REWRITE,
        "rewrite",
    )
    assert rec.rerank_query == source.rerank_query
    assert set(rec.model_revisions) == {"guard", "image", "rerank"}
    assert "rewrite" not in rec.timings_s
    keys = ("width", "height", "n", "steps", "rewrite", "base_seed")
    assert {k: rec.config[k] for k in keys} == {
        "width": 64,
        "height": 64,
        "n": 3,
        "steps": 2,
        "rewrite": "long",
        "base_seed": 103,
    }
    assert record.read(last_run(h)) == rec
    assert record.read(source_dir) == source  # the source is never edited


def test_continue_seeds_takes_n_and_the_caller_options(h: Harness) -> None:
    h.run()
    options = GenerateRequest("", guard_mode="strict", rerank_instruction="Apples?", min_score=0.8)
    request = continue_seeds(last_run(h), options, n=2)
    assert (request.prompt, request.n, request.seed, request.steps) == ("a red apple", 2, 103, 2)
    assert (request.guard_mode, request.rerank_instruction, request.min_score) == (
        "strict",
        "Apples?",
        0.8,
    )


@pytest.mark.parametrize(
    ("overrides", "prompt"),
    [({"rewrite": False}, "a red apple"), ({}, REWRITE)],
    ids=["no-rewrite", "rewrite"],
)
def test_continue_seeds_generates_from_what_the_source_generated_from(
    h: Harness, overrides: dict[str, Any], prompt: str
) -> None:
    source = h.run(**overrides)
    rec = h.pipeline.run(continue_seeds(last_run(h), GenerateRequest("")), h._emit)
    assert h.stages[-1].encoded == h.stages[0].encoded == ([prompt], "A-q4")
    assert rec.config["rewrite"] == source.config["rewrite"]


def test_continue_seeds_after_a_fallback_keeps_the_user_prompt(h: Harness) -> None:
    h.rewrite_reply = ""  # empty: the user's prompt is generated from
    h.run()
    rec = h.pipeline.run(continue_seeds(last_run(h), GenerateRequest("")), h._emit)
    assert (rec.prompt_source, rec.rewrite_fallback_reason) == ("user", "empty")
    assert h.stages[-1].encoded == (["a red apple"], "A-q4")


def source_run(**fields: Any) -> Path:
    run = paths.new_run_dir()
    config = {"width": 64, "height": 64, "n": 2, "steps": 2, "encoder": "A-q4", "rewrite": "none"}
    defaults: dict[str, Any] = {"seeds": [1, 2], "candidates": [{"index": 0}], "config": config}
    record.write(RunRecord("a red apple", **(defaults | fields)), run)
    return run


def test_continue_seeds_accepts_a_cancelled_run_with_candidates() -> None:
    request = continue_seeds(source_run(status="cancelled"), GenerateRequest(""))
    assert (request.seed, request.n) == (3, 2)


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"status": "running"}, "it is interrupted"),  # the lock is free: it died
        ({"status": "interrupted"}, "it is interrupted"),
        ({"status": "blocked", "candidates": []}, "no candidates"),
        ({"status": "error", "candidates": []}, "no candidates"),
        ({"status": "ok", "config": {"encoder": "B"}}, "encoder B, not A-q4"),
    ],
    ids=["dead-running", "interrupted", "blocked", "error-before-generate", "other-encoder"],
)
def test_continue_seeds_refuses_a_run_it_cannot_continue(
    fields: dict[str, Any], reason: str
) -> None:
    run = source_run(**fields)
    with pytest.raises(RunNotContinuable, match=reason):
        continue_seeds(run, GenerateRequest(""))


def test_continue_seeds_refuses_a_run_still_in_progress() -> None:
    run = source_run(status="running")  # this process, holding the lock
    with lock.RunLock(), pytest.raises(RunNotContinuable, match="it is running"):
        continue_seeds(run, GenerateRequest(""))
