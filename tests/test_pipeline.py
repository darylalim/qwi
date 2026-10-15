"""`Pipeline.run` with a fake generate stage: lifecycle, events, cancel, failures, lock."""

import threading
from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from qwi import errors, lock, logs, memory, paths, preflight, record
from qwi.errors import PipelineBusy
from qwi.events import CandidateReady, DenoiseStep, Event, RunStarted, StageFinished, StageStarted
from qwi.pipeline import GenerateRequest, Pipeline
from qwi.record import RunRecord
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

    def load(self) -> None:
        self.loaded = True

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


class Harness:
    def __init__(self) -> None:
        self.stages: list[FakeStage] = []
        self.events: list[Event] = []
        self.configure: Callable[[FakeStage], None] = lambda stage: None
        self.on_event: Callable[[Event], None] = lambda event: None
        self.resolved: list[str] = []
        self.pipeline = Pipeline(self._make, resolve=self._resolve)

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
        fields: dict[str, Any] = {"n": 3, "steps": 2, "width": 64, "height": 64, "seed": 100}
        request = GenerateRequest("a red apple", **(fields | overrides))
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
        "base_seed": 100,
        "allow_swap": False,
    }
    assert rec.preflight["generate"]["decision"] == "ok"
    assert rec.peak_mem_gb["generate"] == pytest.approx(14.0)
    assert rec.swap_gb == {"generate": 0.0}
    assert "generate" in rec.timings_s
    assert set(rec.model_revisions) == {"image"}
    assert h.resolved == ["image"]
    assert h.stages[0].loaded
    assert h.stages[0].unloaded


def test_events_arrive_in_order(h: Harness) -> None:
    h.run(n=2)
    assert h.kinds() == [
        "RunStarted",
        "StageStarted",
        *["DenoiseStep"] * 2,
        "CandidateReady",
        *["DenoiseStep"] * 2,
        "CandidateReady",
        "StageFinished",
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
    assert isinstance(h.events[1], StageStarted)
    assert h.events[1].stage == "generate"
    ready = [e for e in h.events if isinstance(e, CandidateReady)]
    assert [(e.index, e.seed, e.path.name) for e in ready][1] == (1, 101, "cand_1_seed101.png")
    assert isinstance(h.events[-1], StageFinished)
    assert h.events[-1].peak_gb == pytest.approx(13.9)


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


def test_swap_risk_is_refused_before_loading(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "vm_stat", lambda: vm(12 * 10**9))
    rec = h.run(width=768, height=768)  # 17.8 GB estimate, ~9.9 GB available after headroom
    assert rec.error is not None
    assert rec.error["type"] == "SwapRisk"
    assert h.stages == []
    assert errors.exit_code_for(rec.status, rec.error["type"]) == 4


def test_allow_swap_runs_and_records_the_decision(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory, "vm_stat", lambda: vm(12 * 10**9))
    rec = h.run(width=768, height=768, allow_swap=True)
    assert rec.status == "ok"
    assert rec.preflight["generate"]["decision"] == "swap_risk_allowed"


def test_swap_during_the_stage_is_recorded(h: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    readings = iter([vm(30 * 10**9, swapouts=0), vm(30 * 10**9, swapouts=0)])
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
    assert held == [True]
    assert not lock.is_held()
