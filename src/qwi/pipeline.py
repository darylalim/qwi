"""Orchestrates the stages (PLAN.md §1, §4, §5.6, §5.11): `Pipeline().run(request, on_event)`.

One run at a time (`RunLock`). Each stage: pre-flight check → load → run → evict in `finally`.
`run()` always returns the `RunRecord`; its `status` and `error` say how it ended, and the
traceback of any error is in `run.log`. Only `PipelineBusy` is raised, because no run exists then.

Stages: guard (input) → rewrite → guard (rewrite) → encode → generate (§1, §5.1, §5.2);
`--no-rewrite` skips the rewrite and the second pass. M3 adds rerank after generate.
"""

import logging
import random
import threading
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
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
    RewriteReady,
    RunStarted,
    StageFinished,
    StageStarted,
)
from qwi.lock import RunLock
from qwi.record import RunRecord
from qwi.stages import guard
from qwi.stages.encode import PromptEmbeds
from qwi.stages.generate import Candidate, Generate
from qwi.stages.rewrite import Rewrite, Rewritten

ENCODER = "A-q4"  # the q4 pack's own encoder through mflux's recipe (§3, §8)
HISTORY_KEYS = ("width", "height", "n", "encoder")  # what pre-flight history must match (§5.11)
ROLES = ["guard", "rewrite", "image"]  # models a run can load; rerank joins in M3


@dataclass
class GenerateRequest:
    prompt: str
    n: int = DEFAULTS.n
    width: int = DEFAULTS.width
    height: int = DEFAULTS.height
    steps: int = DEFAULTS.steps
    seed: int | None = None  # base seed; random when None, recorded either way
    guard_mode: str = DEFAULTS.guard_mode
    rewrite: bool = True  # False: --no-rewrite, generate from the prompt as given
    rewrite_style: str = DEFAULTS.rewrite_style
    allow_swap: bool = False
    verbose: bool = False
    cancel: threading.Event = field(default_factory=threading.Event)


class GuardStage(Protocol):
    def load(self) -> None: ...

    def __call__(self, text: str) -> guard.Verdict: ...

    def unload(self) -> None: ...


class RewriteStage(Protocol):
    def load(self) -> None: ...

    def __call__(self, prompt: str) -> Rewritten: ...

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
RewriteFactory = Callable[[Path, str], RewriteStage]


def default_generate_stage(
    model_path: Path, callbacks: Sequence[object], watchdog: memory.SwapWatchdog
) -> GenerateStage:
    return Generate(model_path, callbacks=callbacks, watchdog=watchdog)


def default_guard_stage(model_path: Path) -> GuardStage:
    return guard.Guard(model_path)


def default_rewrite_stage(model_path: Path, style: str) -> RewriteStage:
    return Rewrite(model_path, style)


def roles(request: GenerateRequest) -> list[str]:
    """The models this request loads: what startup checks and `run.json` records."""
    return [r for r in ROLES if r != "rewrite" or request.rewrite]


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
        make_rewrite: RewriteFactory = default_rewrite_stage,
    ) -> None:
        self._make_generate = make_generate
        self._make_guard = make_guard
        self._make_rewrite = make_rewrite
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
            model_revisions=record.model_revisions(roles(request)),
            config={
                "width": request.width,
                "height": request.height,
                "n": request.n,
                "steps": request.steps,
                "encoder": ENCODER,
                "rewrite": request.rewrite_style if request.rewrite else "none",
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
                self._startup(roles(request))  # disk, $QWI_HOME writable, models cached (§5.11)
                prompt = self._prompt_stages(request, rec, run, on_event)
                self._generate(request, prompt, rec, run, on_event)
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

    def _prompt_stages(
        self,
        request: GenerateRequest,
        rec: RunRecord,
        run: Path,
        on_event: Callable[[Event], None],
    ) -> str:
        """Stages 1-2: guard the input, rewrite, guard the rewrite (§5.2, §5.1).

        Returns the prompt to generate from. The guard stays loaded through the rewrite for its
        second pass, then both are evicted before encoding (§4). A block raises `GuardBlocked`.
        """
        policy = guard.load_policy()  # a broken table fails before anything loads
        self._preflight(request, rec, run, "guard", preflight.STATIC_GB["guard"])
        checker = self._make_guard(self._resolve("guard"))
        rewriter: RewriteStage | None = None
        try:
            with self._measure(rec, "guard", on_event, mode=request.guard_mode):
                checker.load()
                verdict = checker(request.prompt)
            result = guard.decide(request.guard_mode, verdict, policy=policy)
            self._verdict(rec, run, on_event, "input", verdict, result, policy)
            if not request.rewrite:
                return request.prompt

            self._preflight(request, rec, run, "rewrite", preflight.STATIC_GB["rewrite"])
            rewriter = self._make_rewrite(self._resolve("rewrite"), request.rewrite_style)
            with self._measure(rec, "rewrite", on_event, style=request.rewrite_style):
                rewriter.load()
                out = rewriter(request.prompt)
                # A fallback generates from the input, which the first pass already checked.
                second = checker(out.prompt) if out.fallback_reason is None else None
            rec.rewritten_prompt = out.text
            rec.rewrite_fallback_reason = out.fallback_reason
            rec.prompt_source = "user" if out.fallback_reason else "rewrite"
            record.write(rec, run)
            on_event(RewriteReady(out.text, out.fallback_reason, out.style))
            logs.event(
                logging.WARNING if out.fallback_reason else logging.INFO,
                "rewrite",
                "rewrite_ready",
                f"fallback: {out.fallback_reason}" if out.fallback_reason else "rewrite used",
                style=out.style,
                words=len(out.text.split()),
            )
            if second is not None:
                result = guard.decide(request.guard_mode, verdict, second, policy=policy)
                self._verdict(rec, run, on_event, "rewrite", second, result, policy)
            return out.prompt
        finally:
            checker.unload()
            if rewriter is not None:
                rewriter.unload()

    def _verdict(
        self,
        rec: RunRecord,
        run: Path,
        on_event: Callable[[Event], None],
        check: str,
        verdict: guard.Verdict,
        result: guard.GuardResult,
        policy: guard.Policy,
    ) -> None:
        """Record the decision so far, report this pass, and raise on a block."""
        rec.guard = result.as_record()
        record.write(rec, run)
        action, _ = guard.judge(verdict, result.mode, policy)
        summary = guard.describe_pass(check, verdict, result.mode)
        on_event(GuardVerdict(check, verdict.level, verdict.categories, action, summary))
        logs.event(
            logging.WARNING if action == "warn" else logging.INFO,
            self._stage,
            "guard_verdict",
            action,
            check=check,
            level=verdict.level,
            categories=list(verdict.categories),
            reason=result.reason,
        )
        if result.decision == "block":
            raise GuardBlocked(result.summary())

    def _preflight(
        self, request: GenerateRequest, rec: RunRecord, run: Path, stage: str, est: float
    ) -> None:
        """The stage's memory check before anything loads, then a last chance to cancel."""
        self._stage = stage
        rec.preflight[stage] = preflight.check(
            stage, est, allow_swap=request.allow_swap
        ).as_record()
        record.write(rec, run)
        if request.cancel.is_set():
            raise GenerationCancelled()

    @contextmanager
    def _measure(
        self, rec: RunRecord, stage: str, on_event: Callable[[Event], None], **context: Any
    ) -> Generator[None]:
        """Time, peak memory and swap for a stage whose models load and run inside the block."""
        self._stage = stage
        watchdog = memory.SwapWatchdog(stage)
        watchdog.start()  # baseline before the model loads (§8 watchdog method)
        on_event(StageStarted(stage))
        logs.event(logging.INFO, stage, "stage_started", "loading", **context)
        started = time.perf_counter()
        try:
            with memory.track_peak() as peak:
                yield
        finally:
            rec.swap_gb[stage] = round(watchdog.delta_bytes / memory.GB, 2)
        self._finish_stage(rec, stage, started, peak.gb, on_event)

    def _generate(
        self,
        request: GenerateRequest,
        prompt: str,
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
        self._preflight(request, rec, run, stage, est)

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
                (embeds,) = generator.encode([prompt], ENCODER)
            self._finish_stage(rec, "encode", encode_started, encode_peak.gb, on_event)
            logs.event(logging.INFO, "encode", "encoded", "encoder evicted", tokens=embeds.tokens)
            record.write(rec, run)
            if request.cancel.is_set():
                raise GenerationCancelled()

            self._stage = stage
            on_event(StageStarted(stage))
            started = time.perf_counter()
            candidates = generator(
                prompt,
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
