"""Orchestrates the stages (PLAN.md §1, §4, §5.6, §5.11): `Pipeline().run(request, on_event)`.

One run at a time (`RunLock`). Each stage: pre-flight check → load → run → evict in `finally`.
`run()` always returns the `RunRecord`; its `status` and `error` say how it ended, and the
traceback of any error is in `run.log`. Only `PipelineBusy` is raised, because no run exists then.

Stages: guard (input pass, §5.2) → encode → generate. The rewrite stage (M2) goes between guard
and encode and adds the guard's second pass; M3 adds rerank after generate.
"""

import logging
import random
import threading
import time
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from qwi import errors, logs, memory, models, paths, preflight, record, storage
from qwi.config import DEFAULTS
from qwi.errors import GenerationCancelled, GuardBlocked
from qwi.events import (
    CandidateReady,
    DenoiseStep,
    Event,
    GuardVerdict,
    RunStarted,
    StageFinished,
    StageStarted,
)
from qwi.lock import RunLock
from qwi.record import RunRecord
from qwi.stages import guard
from qwi.stages.encode import PromptEmbeds
from qwi.stages.generate import Candidate, Generate

ENCODER = "A-q4"  # the q4 pack's own encoder through mflux's recipe (§3, §8)
HISTORY_KEYS = ("width", "height", "n", "encoder")  # what pre-flight history must match (§5.11)
ROLES = ["guard", "image"]  # models this run loads; rewrite and rerank join in M2/M3


@dataclass
class GenerateRequest:
    prompt: str
    n: int = DEFAULTS.n
    width: int = DEFAULTS.width
    height: int = DEFAULTS.height
    steps: int = DEFAULTS.steps
    seed: int | None = None  # base seed; random when None, recorded either way
    guard_mode: str = DEFAULTS.guard_mode
    allow_swap: bool = False
    verbose: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)


class GuardStage(Protocol):
    def load(self) -> None: ...

    def __call__(self, text: str) -> guard.Verdict: ...

    def unload(self) -> None: ...


class GenerateStage(Protocol):
    def load(self) -> None: ...

    def encode(self, prompts: Sequence[str], encoder: str = ...) -> list[PromptEmbeds]: ...

    def __call__(
        self, prompt: str, seeds: Sequence[int], *, width: int, height: int, steps: int
    ) -> Generator[Candidate]: ...

    def unload(self) -> None: ...


StageFactory = Callable[[Path, Sequence[object], memory.SwapWatchdog], GenerateStage]
GuardFactory = Callable[[Path], GuardStage]


def default_generate_stage(
    model_path: Path, callbacks: Sequence[object], watchdog: memory.SwapWatchdog
) -> GenerateStage:
    return Generate(model_path, callbacks=callbacks, watchdog=watchdog)


def default_guard_stage(model_path: Path) -> GuardStage:
    return guard.Guard(model_path)


def ignore_event(_: Event) -> None:
    pass


class _ProgressHook:
    """mflux step hook: emits `DenoiseStep` and cancels at the next step (§5.6)."""

    def __init__(
        self, seeds: Sequence[int], cancel: threading.Event, emit: Callable[[Event], None]
    ) -> None:
        self._index = {seed: i for i, seed in enumerate(seeds)}
        self._cancel = cancel
        self._emit = emit

    def call_in_loop(self, *, t: int, seed: int, config: Any, **_: Any) -> None:
        self._emit(DenoiseStep(self._index[seed], t + 1, config.num_inference_steps))
        if self._cancel.is_set():
            raise GenerationCancelled()


class Pipeline:
    def __init__(
        self,
        make_generate: StageFactory = default_generate_stage,
        resolve: Callable[[str], Path] = models.resolve,
        startup: Callable[[list[str]], None] = storage.startup,
        make_guard: GuardFactory = default_guard_stage,
    ) -> None:
        self._make_generate = make_generate
        self._make_guard = make_guard
        self._resolve = resolve
        self._startup = startup
        self._stage: str | None = None

    def run(
        self, request: GenerateRequest, on_event: Callable[[Event], None] = ignore_event
    ) -> RunRecord:
        with RunLock():
            return self._run(request, on_event)

    def _run(self, request: GenerateRequest, on_event: Callable[[Event], None]) -> RunRecord:
        base = request.seed if request.seed is not None else random.randrange(2**31)
        seeds = [base + i for i in range(request.n)]
        run = paths.new_run_dir()
        rec = RunRecord(
            request.prompt,
            seeds=seeds,
            model_revisions=record.model_revisions(ROLES),
            config={
                "width": request.width,
                "height": request.height,
                "n": request.n,
                "steps": request.steps,
                "encoder": ENCODER,
                "base_seed": base,
                "allow_swap": request.allow_swap,
            },
        )
        self._stage = None
        with logs.run_log(run, request.verbose):
            record.write(rec, run)
            logs.event(logging.INFO, None, "run_started", "run started", run=run.name, seeds=seeds)
            on_event(RunStarted(run, seeds))
            try:
                self._stage = "startup"
                self._startup(ROLES)  # disk rule, $QWI_HOME writable, models cached (§5.11)
                self._guard(request, rec, run, on_event)
                self._generate(request, rec, run, on_event)
            except (Exception, KeyboardInterrupt) as exc:
                failure = errors.classify(exc, self._stage or "setup")
                if isinstance(failure, GenerationCancelled):
                    failure = GenerationCancelled(len(rec.candidates), request.n)
                record.finish(rec, failure, self._stage)
                if rec.status == "error":
                    logs.exception(self._stage, "error", str(failure), type=type(failure).__name__)
                else:
                    logs.event(logging.WARNING, self._stage, rec.status, str(failure))
            else:
                record.finish(rec)
            finally:
                record.write(rec, run)
            logs.event(logging.INFO, None, "run_finished", rec.status, status=rec.status)
        return rec

    def _guard(
        self,
        request: GenerateRequest,
        rec: RunRecord,
        run: Path,
        on_event: Callable[[Event], None],
    ) -> None:
        """Stage 1, input pass (§5.2): a block ends the run before any other model loads."""
        stage = self._stage = "guard"
        policy = guard.load_policy()  # a broken table fails before anything loads
        rec.preflight[stage] = preflight.check(
            stage, preflight.STATIC_GB[stage], allow_swap=request.allow_swap
        ).as_record()
        record.write(rec, run)
        if request.cancel.is_set():
            raise GenerationCancelled()

        watchdog = memory.SwapWatchdog(stage)
        watchdog.start()
        model = self._make_guard(self._resolve("guard"))
        on_event(StageStarted(stage))
        logs.event(logging.INFO, stage, "stage_started", "loading", mode=request.guard_mode)
        started = time.perf_counter()
        try:
            with memory.track_peak() as peak:
                model.load()
                verdict = model(request.prompt)
        finally:
            model.unload()  # the rewrite stage will keep it loaded for the second pass (§4)
            rec.swap_gb[stage] = round(watchdog.delta_bytes / memory.GB, 2)
        self._finish_stage(rec, stage, started, peak.gb, on_event)

        result = guard.decide(request.guard_mode, verdict, policy=policy)
        rec.guard = result.as_record()
        record.write(rec, run)
        summary = result.summary()
        on_event(GuardVerdict("input", verdict.level, verdict.categories, result.decision, summary))
        logs.event(
            logging.WARNING if result.decision == "warn" else logging.INFO,
            stage,
            "guard_verdict",
            result.decision,
            check="input",
            level=verdict.level,
            categories=list(verdict.categories),
            reason=result.reason,
        )
        if result.decision == "block":
            raise GuardBlocked(summary)

    def _generate(
        self,
        request: GenerateRequest,
        rec: RunRecord,
        run: Path,
        on_event: Callable[[Event], None],
    ) -> None:
        stage = self._stage = "generate"
        history = {k: rec.config[k] for k in HISTORY_KEYS}
        est = max(
            # The encoder is evicted before denoising, so the generate peak excludes it (§4).
            preflight.generate_gb(request.width, request.height, encoder_resident=False),
            preflight.history_gb(stage, history),
        )
        rec.preflight[stage] = preflight.check(
            stage, est, allow_swap=request.allow_swap
        ).as_record()
        record.write(rec, run)
        if request.cancel.is_set():
            raise GenerationCancelled()

        watchdog = memory.SwapWatchdog(stage, preflight.swap_limit_bytes(request.allow_swap))
        watchdog.start()  # baseline before the model loads (§8 watchdog method)
        hook = _ProgressHook(rec.seeds, request.cancel, on_event)
        generator = self._make_generate(self._resolve("image"), [hook], watchdog)
        on_event(StageStarted("encode"))
        logs.event(logging.INFO, "encode", "stage_started", "loading", est_gb=round(est, 2))
        peaks: list[float] = []
        candidates: Generator[Candidate] | None = None
        started: float | None = None  # set once denoising starts
        try:
            self._stage = "encode"
            encode_started = time.perf_counter()
            with memory.track_peak() as encode_peak:
                generator.load()
                (embeds,) = generator.encode([request.prompt], ENCODER)
            self._finish_stage(rec, "encode", encode_started, encode_peak.gb, on_event)
            logs.event(logging.INFO, "encode", "encoded", "encoder evicted", tokens=embeds.tokens)
            record.write(rec, run)
            if request.cancel.is_set():
                raise GenerationCancelled()

            self._stage = stage
            on_event(StageStarted(stage))
            started = time.perf_counter()
            candidates = generator(
                request.prompt,
                rec.seeds,
                width=request.width,
                height=request.height,
                steps=request.steps,
            )
            for index, candidate in enumerate(candidates):
                path = run / paths.candidate_name(index, candidate.seed)
                candidate.image.save(path)
                peaks.append(candidate.peak_gb)
                rec.candidates.append(
                    {
                        "index": index,
                        "seed": candidate.seed,
                        "path": paths.relative(path, run),
                        "seconds": round(candidate.seconds, 2),
                        "peak_gb": round(candidate.peak_gb, 2),
                    }
                )
                record.write(rec, run)
                on_event(CandidateReady(index, candidate.seed, path))
                logs.event(
                    logging.INFO,
                    stage,
                    "candidate_ready",
                    path.name,
                    seed=candidate.seed,
                    seconds=round(candidate.seconds, 2),
                    peak_gb=round(candidate.peak_gb, 2),
                )
                if request.cancel.is_set() and index < len(rec.seeds) - 1:
                    raise GenerationCancelled()
        finally:
            if candidates is not None:
                candidates.close()  # a cancel between candidates leaves it suspended
            generator.unload()
            rec.swap_gb[stage] = round(watchdog.delta_bytes / memory.GB, 2)
            if started is not None:
                self._finish_stage(rec, stage, started, max(peaks, default=0.0), on_event)

    @staticmethod
    def _finish_stage(
        rec: RunRecord,
        stage: str,
        started: float,
        peak_gb: float,
        on_event: Callable[[Event], None],
    ) -> None:
        seconds = time.perf_counter() - started
        rec.timings_s[stage] = round(seconds, 2)
        rec.peak_mem_gb[stage] = round(peak_gb, 2)
        on_event(StageFinished(stage, seconds, peak_gb))
        logs.event(
            logging.INFO,
            stage,
            "stage_finished",
            "done",
            seconds=round(seconds, 2),
            peak_gb=round(peak_gb, 2),
            swap_gb=rec.swap_gb.get(stage),
        )
