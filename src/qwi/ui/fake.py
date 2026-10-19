"""A scripted stand-in for `Pipeline` (PLAN.md §5.6 "Tests"): `QWI_PIPELINE=fake`.

No models: it emits the real event sequence, writes a real run folder (tiny PNGs, `best.png`,
`run.json`, `run.log`) and takes the real `$QWI_HOME/.lock`, so the UI and `JobManager` tests see
what a run looks like without a GPU. Cancel is checked on every step, like the real step hook.
"""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from qwi import logs, paths, record
from qwi.errors import GenerationCancelled, GuardBlocked
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
from qwi.lock import RunLock
from qwi.pipeline import GenerateRequest
from qwi.record import RunRecord

COLORS = [(200, 40, 40), (40, 160, 60), (40, 70, 200), (220, 180, 30)]
SIZE = 8  # pixels: the images only need to exist
GUARD_VERDICTS = {
    "allow": GuardVerdict("input", "Safe", (), "allow", "input: Safe"),
    "warn": GuardVerdict(
        "input",
        "Controversial",
        ("Politically Sensitive Topics",),
        "warn",
        "input: Controversial (Politically Sensitive Topics), default mode",
    ),
    "block": GuardVerdict(
        "input", "Unsafe", ("Violent",), "block", "input: Unsafe (Violent), default mode"
    ),
}


@dataclass
class Hold:
    """Pauses a fake run at a denoising step until `release()`, for tests that act mid-run."""

    candidate: int = 0
    step: int = 1

    def __post_init__(self) -> None:
        self.reached = threading.Event()
        self._go = threading.Event()

    def release(self) -> None:
        self._go.set()

    def wait_here(self) -> None:
        self.reached.set()
        if not self._go.wait(10):
            raise TimeoutError("Hold was never released")


class FakePipeline:
    """`run()` acts like `Pipeline.run`: it returns the record, or raises `PipelineBusy`.

    - `raises` is raised out of `run()` at the first denoising step (a pipeline bug), with the
      lock held, so callers can check that it's released.
    - `fails` is raised there too, but ends the run like the real pipeline does: an `error`
      record with the type, and the traceback in run.log.
    - `swap_risk` sends a `PreflightWarning` before generate.
    - `margins` are the rerank scores per candidate and `prob` their probability (below
      `min_score`: the low-confidence warning).
    - `guard` is the input pass's action: allow, warn (the run continues) or block (it ends
      `blocked`).
    """

    def __init__(
        self,
        *,
        raises: BaseException | None = None,
        fails: Exception | None = None,
        swap_risk: bool = False,
        hold: Hold | None = None,
        margins: tuple[float, ...] = (0.5, 2.0, -1.0, 1.0),
        prob: float = 0.9,
        guard: str = "allow",
    ) -> None:
        self.raises = raises
        self.fails = fails
        self.swap_risk = swap_risk
        self.hold = hold
        self.margins = margins
        self.prob = prob
        self.guard = guard
        self.requests: list[GenerateRequest] = []
        self.evictions = 0  # stages unloaded, as the real pipeline does in `finally`

    @classmethod
    def scripted(cls, name: str) -> "FakePipeline":
        """A named run for `QWI_FAKE_SCRIPT`: ok, warn (guard), block, low (score), swap (risk)."""
        scripts: dict[str, dict] = {
            "ok": {},
            "warn": {"guard": "warn"},
            "block": {"guard": "block"},
        }
        scripts["low"] = {"prob": 0.2}
        scripts["swap"] = {"swap_risk": True}
        if name not in scripts:
            raise ValueError(f"unknown fake script {name!r}; one of {', '.join(scripts)}")
        return cls(**scripts[name])

    def run(self, request: GenerateRequest, on_event: Callable[[Event], None]) -> RunRecord:
        with RunLock():
            self.requests.append(request)
            return self._run(request, on_event)

    def _run(self, request: GenerateRequest, on_event: Callable[[Event], None]) -> RunRecord:
        base = request.seed if request.seed is not None else 1000
        seeds = [base + i for i in range(request.n)]
        run = paths.new_run_dir()
        rec = RunRecord(request.prompt, seeds=seeds, config={"n": request.n, "fake": True})
        with logs.run_log(run, request.verbose):
            record.write(rec, run)
            logs.event(logging.INFO, None, "run_started", "run started", run=run.name)
            on_event(RunStarted(run, seeds))
            try:
                self._prompt(request, rec, on_event)
                self._generate(request, rec, run, on_event)
                self._rerank(request, rec, run, on_event)
            except GenerationCancelled as exc:
                record.finish(rec, GenerationCancelled(len(rec.candidates), request.n))
                logs.event(logging.WARNING, None, "cancelled", str(exc))
            except GuardBlocked as exc:
                record.finish(rec, exc, "guard")
                logs.event(logging.WARNING, "guard", "blocked", str(exc))
            except Exception as exc:
                if exc is self.raises:
                    raise
                record.finish(rec, exc, "generate")
                logs.exception("generate", "error", str(exc), type=type(exc).__name__)
            else:
                record.finish(rec)
            finally:
                record.write(rec, run)
        return rec

    def _stage(self, stage: str, on_event: Callable[[Event], None]) -> None:
        on_event(StageStarted(stage))
        on_event(StageFinished(stage, 0.01, 1.0))
        logs.event(logging.INFO, stage, "stage_finished", "done")
        self.evictions += 1

    def _prompt(
        self, request: GenerateRequest, rec: RunRecord, on_event: Callable[[Event], None]
    ) -> None:
        self._stage("guard", on_event)
        on_event(GUARD_VERDICTS[self.guard])
        if self.guard == "block":
            raise GuardBlocked(GUARD_VERDICTS["block"].summary)
        if request.rewrite:
            self._stage("rewrite", on_event)
            rec.rewritten_prompt = f"A detailed photograph of {request.prompt}."
            rec.prompt_source = "rewrite"
            on_event(RewriteReady(rec.rewritten_prompt, None, request.rewrite_style))

    def _generate(
        self,
        request: GenerateRequest,
        rec: RunRecord,
        run: Path,
        on_event: Callable[[Event], None],
    ) -> None:
        self._stage("encode", on_event)
        if self.swap_risk:
            on_event(PreflightWarning("generate", 14.97, 9.87))
        on_event(StageStarted("generate"))
        try:
            for index, seed in enumerate(rec.seeds):
                for step in range(1, request.steps + 1):
                    on_event(DenoiseStep(index, step, request.steps))
                    if self.hold is not None and (index, step) == (
                        self.hold.candidate,
                        self.hold.step,
                    ):
                        self.hold.wait_here()
                    if self.raises is not None:
                        raise self.raises
                    if self.fails is not None:
                        raise self.fails
                    if request.cancel.is_set():
                        raise GenerationCancelled()
                path = run / paths.candidate_name(index, seed)
                Image.new("RGB", (SIZE, SIZE), COLORS[index % len(COLORS)]).save(path)
                rec.candidates.append({"index": index, "seed": seed, "path": path.name})
                record.write(rec, run)
                on_event(CandidateReady(index, seed, path))
        finally:
            self.evictions += 1
        on_event(StageFinished("generate", 0.01, 1.0))

    def _rerank(
        self,
        request: GenerateRequest,
        rec: RunRecord,
        run: Path,
        on_event: Callable[[Event], None],
    ) -> None:
        self._stage("rerank", on_event)
        margins = [self.margins[i % len(self.margins)] for i in range(len(rec.candidates))]
        probs = [self.prob for _ in margins]
        order = sorted(range(len(margins)), key=lambda i: -margins[i])
        ranks = [order.index(i) + 1 for i in range(len(margins))]
        best = order[0]
        for candidate, margin, prob, rank in zip(
            rec.candidates, margins, probs, ranks, strict=True
        ):
            candidate.update(margin=margin, prob=prob, rank=rank)
        rec.best_index = best
        low = probs[best] < request.min_score
        rec.rerank = {"min_score": request.min_score, "low_confidence": low}
        best_path = paths.link_best(run / rec.candidates[best]["path"])
        on_event(
            RerankDone(
                best,
                best_path,
                list(rec.seeds),
                [run / c["path"] for c in rec.candidates],
                margins,
                probs,
                ranks,
                low,
                request.min_score,
            )
        )
