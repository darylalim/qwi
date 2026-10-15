"""`qwi eval`: generate the fixed eval set for one variant (PLAN.md §5.4 "Protocol").

Stage-batched, not one pipeline per prompt: the runner holds the lock once, loads the image model
once, encodes every prompt, evicts the encoder, then generates prompts x seeds. Each seed is its
own sample (no best-of-N). Images go to `$QWI_HOME/eval/<date>_<variant>/`; `results.jsonl` (one
row per image, appended as each image is saved) and `review.csv` go to the committed
`eval/results/<date>_<variant>/`.

M2 adds the guard and rewrite stages before encoding; M3 fills the reranker and OCR columns.
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
from qwi.events import CandidateReady, Event, RunStarted, StageFinished, StageStarted
from qwi.lock import RunLock
from qwi.pipeline import (
    ENCODER,
    GuardFactory,
    StageFactory,
    default_generate_stage,
    default_guard_stage,
    ignore_event,
)
from qwi.stages import guard

ROLES = ["image"]  # models this runner loads; guard + rewrite join it in M2 (§5.4 stage order)

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
    allow_swap: bool = False
    verbose: bool = False
    day: date = field(default_factory=date.today)
    cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def name(self) -> str:
        return f"{self.day.isoformat()}_{self.variant}"


@dataclass
class EvalResult:
    name: str
    results_dir: Path
    images_dir: Path
    rows: list[dict[str, Any]]
    status: str = "ok"  # ok | cancelled | error
    error: dict[str, Any] | None = None


class EvalRunner:
    def __init__(
        self,
        make_generate: StageFactory = default_generate_stage,
        resolve: Callable[[str], Path] = models.resolve,
        startup: Callable[[list[str]], None] = storage.startup,
    ) -> None:
        self._make_generate = make_generate
        self._resolve = resolve
        self._startup = startup

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
        if results_dir.exists():
            raise EvalExists(f"{results_dir} exists; pick another --variant name")
        with RunLock():
            results_dir.mkdir(parents=True)
            images_dir.mkdir(parents=True, exist_ok=True)
            result = EvalResult(request.name, results_dir, images_dir, [])
            with logs.run_log(images_dir, request.verbose):
                on_event(RunStarted(images_dir, list(request.seeds)))
                logs.event(logging.INFO, None, "eval_started", request.name, prompts=len(prompts))
                stage = "startup"
                try:
                    self._startup(ROLES)
                    stage = "generate"
                    self._generate(request, prompts, result, on_event)
                except (Exception, KeyboardInterrupt) as exc:
                    failure = errors.classify(exc, stage)
                    result.status = errors.status(failure) or "error"
                    if result.status == "error":
                        result.error = {"type": type(failure).__name__, "message": str(failure)}
                        logs.exception(stage, "error", str(failure), type=type(failure).__name__)
                    else:
                        logs.event(logging.WARNING, stage, result.status, str(failure))
                finally:
                    write_review(results_dir / "review.csv", prompts, result.rows)
                logs.event(
                    logging.INFO, None, "eval_finished", result.status, rows=len(result.rows)
                )
        return result

    def _generate(
        self,
        request: EvalRequest,
        prompts: list[EvalPrompt],
        result: EvalResult,
        on_event: Callable[[Event], None],
    ) -> None:
        est = preflight.generate_gb(request.width, request.height, encoder_resident=False)
        check = preflight.check("generate", est, allow_swap=request.allow_swap)
        watchdog = memory.SwapWatchdog("generate", preflight.swap_limit_bytes(request.allow_swap))
        watchdog.start()
        generator = self._make_generate(self._resolve("image"), [], watchdog)
        try:
            on_event(StageStarted("encode"))
            started = time.perf_counter()
            with memory.track_peak() as peak:
                generator.load()
                generator.encode([p.prompt for p in prompts], ENCODER)
            encode_s = time.perf_counter() - started
            on_event(StageFinished("encode", encode_s, peak.gb))
            on_event(StageStarted("generate"))
            started = time.perf_counter()
            peaks: list[float] = []
            with (result.results_dir / "results.jsonl").open("a", encoding="utf-8") as out:
                for prompt in prompts:
                    swap_before = watchdog.delta_bytes
                    for candidate in generator(
                        prompt.prompt,
                        request.seeds,
                        width=request.width,
                        height=request.height,
                        steps=request.steps,
                    ):
                        name = image_name(prompt.id, candidate.seed)
                        candidate.image.save(result.images_dir / name)
                        swap_after = watchdog.delta_bytes
                        row = self._row(request, prompt, candidate.seed, name, candidate.image)
                        row |= {
                            "seconds": {"generate": round(candidate.seconds, 2)},
                            "peak_gb": {"generate": round(candidate.peak_gb, 2)},
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
        request: EvalRequest, prompt: EvalPrompt, seed: int, name: str, image: Image
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "id": prompt.id,
            "seed": seed,
            "variant": request.variant,
            "prompt": prompt.prompt,
            "tags": list(prompt.tags),
            "rewrite": None,  # M2
            "image": name,  # relative to $QWI_HOME/eval/<date>_<variant>/
            "config": {
                "width": request.width,
                "height": request.height,
                "steps": request.steps,
                "encoder": ENCODER,
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
