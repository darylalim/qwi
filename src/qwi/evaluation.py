"""`qwi eval`: generate the fixed eval set for one variant (PLAN.md §5.4 "Protocol").

Stage-batched, not one pipeline per prompt: the runner holds the lock once and runs guard (all
prompts) → rewrite (all) → guard on the rewrites (all) → encode (all) → generate (prompts x
seeds), loading each model once (§5.4 stage order). Each seed is its own sample (no best-of-N).
A prompt the guard blocks gets no images; the result lists it. Images go to
`$QWI_HOME/eval/<date>_<variant>/`; `results.jsonl` (one row per image, appended as each image is
saved) and `review.csv` go to the committed `eval/results/<date>_<variant>/`.

M3 fills the reranker and OCR columns.
"""

import csv
import json
import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from PIL.Image import Image

from qwi import errors, logs, memory, models, paths, preflight, storage
from qwi.config import DEFAULTS
from qwi.errors import EvalExists, EvalSetError, GenerationCancelled
from qwi.events import (
    CandidateReady,
    Event,
    GuardVerdict,
    RunStarted,
    StageFinished,
    StageStarted,
)
from qwi.lock import RunLock
from qwi.pipeline import (
    ENCODER,
    GuardFactory,
    RewriteFactory,
    RewriteStage,
    StageFactory,
    default_generate_stage,
    default_guard_stage,
    default_rewrite_stage,
    ignore_event,
)
from qwi.stages import guard

ISOLATED_WHITE = 235  # all channels ≥ this counts as near-white (§5.4 automated checks)
ISOLATED_BORDER = 0.10  # outer 10% of each side
ISOLATED_MIN_SHARE = 0.9
REVIEW_COLUMNS = ["item", "prompt_id", "seed", "image", "check", "result", "notes"]


@dataclass(frozen=True)
class EvalPrompt:
    id: str
    prompt: str
    tags: tuple[str, ...]
    checks: tuple[str, ...]


def load_prompts(path: Path | None = None) -> list[EvalPrompt]:
    path = path or paths.EVAL_SET / "prompts.jsonl"
    prompts: list[EvalPrompt] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            prompts.append(
                EvalPrompt(row["id"], row["prompt"], tuple(row["tags"]), tuple(row["checks"]))
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise EvalSetError(f"{path.name} line {n}: {exc}") from exc
    ids = [p.id for p in prompts]
    if len(set(ids)) != len(ids):
        raise EvalSetError(f"{path.name}: duplicate ids")
    return prompts


def white_border_share(image: Image) -> float:
    """Share of near-white pixels in the outer border: the `isolated` automated check."""
    import numpy as np

    pixels = np.asarray(image.convert("RGB"))
    h, w = pixels.shape[:2]
    bh, bw = max(1, int(h * ISOLATED_BORDER)), max(1, int(w * ISOLATED_BORDER))
    white = (pixels >= ISOLATED_WHITE).all(axis=2)
    border = np.ones((h, w), dtype=bool)
    border[bh : h - bh, bw : w - bw] = False
    return float(white[border].mean())


def image_name(prompt_id: str, seed: int) -> str:
    return f"{prompt_id}_{seed}.png"


@dataclass
class EvalRequest:
    variant: str
    seeds: Sequence[int] = DEFAULTS.eval_seeds
    only: Sequence[str] | None = None  # a subset of prompt ids (smoke runs)
    width: int = DEFAULTS.width
    height: int = DEFAULTS.height
    steps: int = DEFAULTS.steps
    guard_mode: str = DEFAULTS.guard_mode
    rewrite: bool = True  # False: --no-rewrite, generate from the eval prompts as written
    rewrite_style: str = DEFAULTS.rewrite_style
    allow_swap: bool = False
    verbose: bool = False
    day: date = field(default_factory=date.today)
    cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def name(self) -> str:
        return f"{self.day.isoformat()}_{self.variant}"

    @property
    def roles(self) -> list[str]:
        return ["guard", *(["rewrite"] if self.rewrite else []), "image"]


@dataclass
class EvalResult:
    name: str
    results_dir: Path
    images_dir: Path
    rows: list[dict[str, Any]]
    status: str = "ok"  # ok | cancelled | error
    error: dict[str, Any] | None = None
    blocked: list[str] = field(default_factory=list)  # prompt ids the guard blocked: no images


@dataclass
class Prepared:
    """One eval prompt after the guard and rewrite stages: what generation uses and records."""

    prompt: str  # the text generated from: the rewrite, or the eval prompt
    rewrite: str | None  # the cleaned rewrite (kept on a fallback), None with --no-rewrite
    fallback_reason: str | None
    guard: guard.GuardResult
    seconds: dict[str, float]  # per-prompt call time per stage (model loads excluded)
    peak_gb: dict[str, float]  # stage peaks, shared by every prompt


class EvalRunner:
    def __init__(
        self,
        make_generate: StageFactory = default_generate_stage,
        resolve: Callable[[str], Path] = models.resolve,
        startup: Callable[[list[str]], None] = storage.startup,
        make_guard: GuardFactory = default_guard_stage,
        make_rewrite: RewriteFactory = default_rewrite_stage,
    ) -> None:
        self._make_generate = make_generate
        self._resolve = resolve
        self._startup = startup
        self._make_guard = make_guard
        self._make_rewrite = make_rewrite
        self._stage = "startup"

    def run(
        self, request: EvalRequest, on_event: Callable[[Event], None] = ignore_event
    ) -> EvalResult:
        prompts = load_prompts()
        if request.only:
            unknown = set(request.only) - {p.id for p in prompts}
            if unknown:
                raise EvalSetError(f"unknown prompt ids: {', '.join(sorted(unknown))}")
            prompts = [p for p in prompts if p.id in request.only]
        results_dir = paths.eval_results_dir() / request.name
        images_dir = paths.eval_dir() / request.name
        # Only saved images make a variant: a run that failed before its first image (e.g. a
        # SwapRisk at generate) can be retried under the same name; run.log keeps both attempts.
        saved = results_dir / "results.jsonl"
        if saved.exists() and saved.stat().st_size:
            raise EvalExists(f"{results_dir} has results; pick another --variant name")
        with RunLock():
            results_dir.mkdir(parents=True, exist_ok=True)
            images_dir.mkdir(parents=True, exist_ok=True)
            result = EvalResult(request.name, results_dir, images_dir, [])
            with logs.run_log(images_dir, request.verbose):
                on_event(RunStarted(images_dir, list(request.seeds)))
                logs.event(logging.INFO, None, "eval_started", request.name, prompts=len(prompts))
                self._stage = "startup"
                try:
                    self._startup(request.roles)
                    prepared = self._prompt_stages(request, prompts, on_event)
                    result.blocked = [
                        p.id for p in prompts if prepared[p.id].guard.decision == "block"
                    ]
                    allowed = [p for p in prompts if p.id not in result.blocked]
                    self._stage = "generate"
                    self._generate(request, allowed, prepared, result, on_event)
                except (Exception, KeyboardInterrupt) as exc:
                    failure = errors.classify(exc, self._stage)
                    result.status = errors.status(failure) or "error"
                    if result.status == "error":
                        result.error = {"type": type(failure).__name__, "message": str(failure)}
                        logs.exception(
                            self._stage, "error", str(failure), type=type(failure).__name__
                        )
                    else:
                        logs.event(logging.WARNING, self._stage, result.status, str(failure))
                finally:
                    write_review(results_dir / "review.csv", prompts, result.rows)
                logs.event(
                    logging.INFO, None, "eval_finished", result.status, rows=len(result.rows)
                )
        return result

    def _prompt_stages(
        self, request: EvalRequest, prompts: list[EvalPrompt], on_event: Callable[[Event], None]
    ) -> dict[str, Prepared]:
        """Guard every prompt, rewrite every allowed one, guard the rewrites (§5.4 stage order).

        Each model loads once; the guard stays loaded through the rewrite for its second pass,
        then both are evicted before the image model loads (§4).
        """
        policy = guard.load_policy()
        mode = request.guard_mode
        self._check("guard", request)
        checker = self._make_guard(self._resolve("guard"))
        rewriter: RewriteStage | None = None
        try:
            on_event(StageStarted(self._stage))
            started = time.perf_counter()
            verdicts: dict[str, guard.Verdict] = {}
            guard_s: dict[str, float] = {}
            with memory.track_peak() as guard_peak:
                checker.load()
                for p in prompts:
                    t = time.perf_counter()
                    verdicts[p.id] = checker(p.prompt)
                    guard_s[p.id] = time.perf_counter() - t
            on_event(StageFinished("guard", time.perf_counter() - started, guard_peak.gb))
            peaks = {"guard": round(guard_peak.gb, 2)}
            prepared: dict[str, Prepared] = {}
            for p in prompts:
                decided = guard.decide(mode, verdicts[p.id], policy=policy)
                seconds = {"guard": round(guard_s[p.id], 2)}
                prepared[p.id] = Prepared(p.prompt, None, None, decided, seconds, peaks)
            self._report(prompts, prepared, "input", on_event)
            if not request.rewrite:
                return prepared

            self._check("rewrite", request)
            rewriter = self._make_rewrite(self._resolve("rewrite"), request.rewrite_style)
            on_event(StageStarted(self._stage))
            started = time.perf_counter()
            with memory.track_peak() as rewrite_peak:
                rewriter.load()
                for p in prompts:
                    if prepared[p.id].guard.decision == "block":
                        continue  # never rewrite a blocked prompt
                    if request.cancel.is_set():
                        raise GenerationCancelled()
                    t = time.perf_counter()
                    out = rewriter(p.prompt)
                    seconds = prepared[p.id].seconds | {"rewrite": time.perf_counter() - t}
                    second = checker(out.prompt) if out.fallback_reason is None else None
                    decided = guard.decide(mode, verdicts[p.id], second, policy=policy)
                    prepared[p.id] = Prepared(
                        out.prompt, out.text, out.fallback_reason, decided, seconds, peaks
                    )
                    level = logging.WARNING if out.fallback_reason else logging.INFO
                    logs.event(
                        level,
                        "rewrite",
                        "rewrite_ready",
                        out.fallback_reason or "rewrite used",
                        id=p.id,
                        words=len(out.text.split()),
                    )
            on_event(StageFinished("rewrite", time.perf_counter() - started, rewrite_peak.gb))
            peaks["rewrite"] = round(rewrite_peak.gb, 2)  # shared dict: every row sees it
            self._report(prompts, prepared, "rewrite", on_event)
            return prepared
        finally:
            checker.unload()
            if rewriter is not None:
                rewriter.unload()

    def _check(self, stage: str, request: EvalRequest) -> None:
        self._stage = stage
        preflight.check(stage, preflight.STATIC_GB[stage], allow_swap=request.allow_swap)
        if request.cancel.is_set():
            raise GenerationCancelled()

    @staticmethod
    def _report(
        prompts: list[EvalPrompt],
        prepared: dict[str, Prepared],
        check: str,
        on_event: Callable[[Event], None],
    ) -> None:
        """A `GuardVerdict` per prompt whose pass warned or blocked; allows stay quiet."""
        for p in prompts:
            result = prepared[p.id].guard
            if result.decision == "allow" or result.trigger != check:
                continue
            summary = f"{p.id}: {result.summary()}"
            verdict = result.rewrite if check == "rewrite" else result.input
            assert verdict is not None
            on_event(
                GuardVerdict(check, verdict.level, verdict.categories, result.decision, summary)
            )
            level = logging.WARNING if result.decision == "warn" else logging.INFO
            logs.event(
                level, "guard", "guard_verdict", result.decision, id=p.id, reason=result.reason
            )

    def _generate(
        self,
        request: EvalRequest,
        prompts: list[EvalPrompt],
        prepared: dict[str, Prepared],
        result: EvalResult,
        on_event: Callable[[Event], None],
    ) -> None:
        if not prompts:
            return  # every prompt was blocked
        est = preflight.generate_gb(request.width, request.height, encoder_resident=False)
        check = preflight.check("generate", est, allow_swap=request.allow_swap)
        watchdog = memory.SwapWatchdog("generate", preflight.swap_limit_bytes(request.allow_swap))
        watchdog.start()
        generator = self._make_generate(self._resolve("image"), [], watchdog)
        try:
            self._stage = "encode"
            on_event(StageStarted("encode"))
            started = time.perf_counter()
            with memory.track_peak() as peak:
                generator.load()
                generator.encode([prepared[p.id].prompt for p in prompts], ENCODER)
            encode_s = time.perf_counter() - started
            on_event(StageFinished("encode", encode_s, peak.gb))
            self._stage = "generate"
            on_event(StageStarted("generate"))
            started = time.perf_counter()
            peaks: list[float] = []
            per_image = 1 / len(request.seeds)  # a prompt's rewrite serves all its seeds
            with (result.results_dir / "results.jsonl").open("a", encoding="utf-8") as out:
                for prompt in prompts:
                    ready = prepared[prompt.id]
                    swap_before = watchdog.delta_bytes
                    for candidate in generator(
                        ready.prompt,
                        request.seeds,
                        width=request.width,
                        height=request.height,
                        steps=request.steps,
                    ):
                        name = image_name(prompt.id, candidate.seed)
                        candidate.image.save(result.images_dir / name)
                        swap_after = watchdog.delta_bytes
                        row = self._row(
                            request, prompt, ready, candidate.seed, name, candidate.image
                        )
                        shared = {k: round(v * per_image, 2) for k, v in ready.seconds.items()}
                        row |= {
                            "seconds": shared | {"generate": round(candidate.seconds, 2)},
                            "peak_gb": ready.peak_gb | {"generate": round(candidate.peak_gb, 2)},
                            "swap_gb": round((swap_after - swap_before) / memory.GB, 2),
                            "preflight": check.as_record(),
                        }
                        swap_before = swap_after
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out.flush()
                        result.rows.append(row)
                        peaks.append(candidate.peak_gb)
                        on_event(
                            CandidateReady(
                                len(result.rows) - 1, candidate.seed, result.images_dir / name
                            )
                        )
                        logs.event(
                            logging.INFO, "generate", "image_ready", name, seconds=row["seconds"]
                        )
                        if request.cancel.is_set():
                            raise GenerationCancelled(
                                len(result.rows), len(prompts) * len(request.seeds)
                            )
            on_event(
                StageFinished("generate", time.perf_counter() - started, max(peaks, default=0.0))
            )
        finally:
            generator.unload()

    @staticmethod
    def _row(
        request: EvalRequest,
        prompt: EvalPrompt,
        ready: Prepared,
        seed: int,
        name: str,
        image: Image,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": prompt.id,
            "seed": seed,
            "variant": request.variant,
            "prompt": prompt.prompt,
            "tags": list(prompt.tags),
            "rewrite": ready.rewrite,
            "rewrite_fallback_reason": ready.fallback_reason,
            "generated_prompt": ready.prompt,
            "guard": ready.guard.as_record(),
            "image": name,  # relative to $QWI_HOME/eval/<date>_<variant>/
            "config": {
                "width": request.width,
                "height": request.height,
                "steps": request.steps,
                "encoder": ENCODER,
                "rewrite": request.rewrite_style if request.rewrite else "none",
                "guard_mode": request.guard_mode,
            },
            "rerank_margin": None,  # M3
            "rerank_prob": None,  # M3
            "ocr": None,  # M3, text-* prompts
            "ocr_exact": None,  # M3
        }
        if prompt.id == "isolated":
            share = white_border_share(image)
            row["white_border_share"] = round(share, 4)
            row["white_border_pass"] = share >= ISOLATED_MIN_SHARE
        return row


def write_review(path: Path, prompts: list[EvalPrompt], rows: list[dict[str, Any]]) -> None:
    """Single-variant review sheet: one row per (image, check), `result` left for the reviewer."""
    checks = {p.id: p.checks for p in prompts}
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")  # LF, like the rest of the repo
        writer.writerow(REVIEW_COLUMNS)
        for row in rows:
            for check in checks[row["id"]]:
                item = f"{row['id']}_{row['seed']}"
                writer.writerow([item, row["id"], row["seed"], row["image"], check, "", ""])


# --- Guard false-positive set (§5.4 `eval/guard_allow.jsonl`) ------------------------------


@dataclass(frozen=True)
class GuardAllowRow:
    id: str
    prompt: str
    might_trip: str  # the category a weak classifier might wrongly flag
    result: guard.GuardResult

    @property
    def passed(self) -> bool:
        """Pass = not blocked; a warning still passes (§5.4)."""
        return self.result.decision != "block"


def run_guard_allow(
    mode: str = DEFAULTS.guard_mode,
    path: Path | None = None,
    make_guard: GuardFactory = default_guard_stage,
    resolve: Callable[[str], Path] = models.resolve,
    startup: Callable[[list[str]], None] = storage.startup,
) -> list[GuardAllowRow]:
    """Judge every prompt in the set with one guard load, under the run lock. Text only: seconds."""
    path = path or paths.EVAL_SET / "guard_allow.jsonl"
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        cases = [(r["id"], r["prompt"], r["might_trip"]) for r in rows]
    except (ValueError, KeyError, TypeError) as exc:
        raise EvalSetError(f"{path.name}: {exc}") from exc
    policy = guard.load_policy()
    with RunLock():
        startup(["guard"])
        preflight.check("guard", preflight.STATIC_GB["guard"])
        stage = make_guard(resolve("guard"))
        try:
            stage.load()
            verdicts = [stage(prompt) for _, prompt, _ in cases]
        finally:
            stage.unload()
    results = [
        GuardAllowRow(id_, prompt, trip, guard.decide(mode, verdict, policy=policy))
        for (id_, prompt, trip), verdict in zip(cases, verdicts, strict=True)
    ]
    for row in results:
        logs.event(
            logging.INFO if row.passed else logging.WARNING,
            "guard",
            "guard_allow",
            row.result.decision,
            id=row.id,
            level=row.result.input.level,
            reason=row.result.reason,
        )
    return results
