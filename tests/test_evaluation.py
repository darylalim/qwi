"""`qwi eval`: the eval set, the automated check, and the stage-batched runner (fake stage)."""

import csv
import json
import re
import threading
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from typer.testing import CliRunner

from qwi import cli, evaluation, lock, memory, paths, preflight
from qwi.errors import EvalExists, EvalSetError
from qwi.evaluation import EvalRequest, EvalRunner
from qwi.events import CandidateReady, Event

from .test_pipeline import LIMIT, FakeStage, vm

DAY = date(2026, 10, 15)


@pytest.fixture(autouse=True)
def machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight, "wired_limit_bytes", lambda: LIMIT)
    monkeypatch.setattr(memory, "vm_stat", lambda: vm(30 * 10**9))


class Runner:
    def __init__(self) -> None:
        self.stages: list[FakeStage] = []
        self.fail_at: tuple[int, BaseException] | None = None
        self.runner = EvalRunner(
            self._make, resolve=lambda role: Path("/w"), startup=lambda r: None
        )

    def _make(self, model_path: Path, callbacks: Sequence[object], _: object) -> FakeStage:
        stage = FakeStage(callbacks)
        stage.fail_at = self.fail_at
        self.stages.append(stage)
        return stage

    def run(self, on_event: Any = None, **overrides: Any) -> evaluation.EvalResult:
        fields: dict[str, Any] = {"width": 64, "height": 64, "steps": 2, "day": DAY}
        request = EvalRequest("baseline", **(fields | overrides))
        return self.runner.run(request, on_event or (lambda e: None))


@pytest.fixture
def r() -> Runner:
    return Runner()


def test_the_eval_set_matches_plan_section_5_4() -> None:
    prompts = evaluation.load_prompts()
    assert [p.id for p in prompts] == [
        "text-en", "text-multi", "text-zh", "count", "spatial", "binding",
        "vague", "detailed", "style", "photo-person", "isolated", "exclusion",
    ]  # fmt: skip
    assert all(p.prompt and p.tags and p.checks for p in prompts)
    zh = next(p for p in prompts if p.id == "text-zh")
    assert "清风茶舍" in zh.prompt


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ('{"id": "a", "prompt": "p", "tags": []}\n', "line 1"),
        ("not json\n", "line 1"),
        ('{"id": "a", "prompt": "p", "tags": [], "checks": []}\n' * 2, "duplicate"),
    ],
)
def test_a_malformed_eval_set_is_refused(tmp_path: Path, text: str, match: str) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text(text)
    with pytest.raises(EvalSetError, match=match):
        evaluation.load_prompts(path)


def test_white_border_share() -> None:
    white = Image.new("RGB", (100, 100), "white")
    assert evaluation.white_border_share(white) == 1.0
    white.paste((20, 120, 40), (20, 20, 80, 80))  # a subject in the middle: border stays white
    assert evaluation.white_border_share(white) == 1.0
    assert evaluation.white_border_share(Image.new("RGB", (100, 100), (234, 255, 255))) == 0.0


def test_a_full_run_is_stage_batched_and_writes_results(r: Runner) -> None:
    result = r.run()
    assert result.status == "ok"
    assert result.name == "2026-10-15_baseline"
    assert result.results_dir == paths.eval_results_dir() / "2026-10-15_baseline"
    assert result.images_dir == paths.eval_dir() / "2026-10-15_baseline"
    assert len(result.rows) == 24

    (stage,) = r.stages  # the image model loads once for the whole run
    assert stage.loaded
    assert stage.unloaded
    assert stage.encoded is not None
    assert len(stage.encoded[0]) == 12  # every prompt encoded in one batch, before generating
    assert not lock.is_held()

    lines = (result.results_dir / "results.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert rows == result.rows
    assert [(row["id"], row["seed"]) for row in rows[:3]] == [
        ("text-en", 1000),
        ("text-en", 1001),
        ("text-multi", 1000),
    ]
    first = rows[0]
    assert first["image"] == "text-en_1000.png"
    assert (result.images_dir / first["image"]).is_file()
    assert first["config"] == {"width": 64, "height": 64, "steps": 2, "encoder": "A-q4"}
    assert (first["rerank_margin"], first["ocr"], first["rewrite"]) == (None, None, None)
    assert first["swap_gb"] == 0.0
    assert first["preflight"]["decision"] == "ok"
    isolated = next(row for row in rows if row["id"] == "isolated")
    assert isolated["white_border_share"] == 0.0  # the fake stage draws black images
    assert isolated["white_border_pass"] is False


def test_review_csv_has_one_row_per_image_and_check(r: Runner) -> None:
    result = r.run()
    assert b"\r" not in (result.results_dir / "review.csv").read_bytes()  # LF like the repo
    with (result.results_dir / "review.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    checks = sum(len(p.checks) for p in evaluation.load_prompts())
    assert len(rows) == checks * 2
    assert rows[0] == {
        "item": "text-en_1000",
        "prompt_id": "text-en",
        "seed": "1000",
        "image": "text-en_1000.png",
        "check": "sign text exactly `THE DAILY GRIND`",
        "result": "",
        "notes": "",
    }


def test_only_runs_a_subset(r: Runner) -> None:
    result = r.run(only=["count", "isolated"], seeds=[7])
    assert [(row["id"], row["seed"]) for row in result.rows] == [("count", 7), ("isolated", 7)]


def test_unknown_ids_are_refused(r: Runner) -> None:
    with pytest.raises(EvalSetError, match="nope"):
        r.run(only=["count", "nope"])


def test_results_are_never_overwritten(r: Runner) -> None:
    r.run(only=["count"])
    with pytest.raises(EvalExists):
        r.run(only=["count"])


def test_cancel_keeps_finished_rows(r: Runner) -> None:
    cancel = threading.Event()

    def on_event(event: Event) -> None:
        if isinstance(event, CandidateReady):
            cancel.set()

    result = r.run(on_event, cancel=cancel)
    assert (result.status, len(result.rows)) == ("cancelled", 1)
    assert len((result.results_dir / "results.jsonl").read_text().splitlines()) == 1
    assert r.stages[0].unloaded


def test_a_failure_mid_run_keeps_rows_and_logs_the_error(r: Runner) -> None:
    r.fail_at = (1, RuntimeError("[metal::malloc] Resource limit (499000) exceeded."))
    result = r.run(only=["count"])  # seed 1000 done, then seed 1001 fails
    assert result.status == "error"
    assert result.error is not None
    assert result.error["type"] == "DeviceOutOfMemory"
    assert len(result.rows) == 1
    assert r.stages[0].unloaded
    assert not lock.is_held()


runner = CliRunner()


def plain(text: str) -> str:
    """Drop ANSI styling: CI advertises color, so Rich splits option names with escape codes."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_cli_requires_a_variant() -> None:
    result = runner.invoke(cli.app, ["eval"])
    assert result.exit_code == 2
    assert "--variant is required" in plain(result.output)


def test_cli_runs_the_eval_and_reports_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[EvalRequest] = []

    class FakeRunner:
        def run(self, request: EvalRequest, on_event: Any) -> evaluation.EvalResult:
            seen.append(request)
            return evaluation.EvalResult("d_v", Path("/results/d_v"), Path("/images/d_v"), [{}])

    monkeypatch.setattr(cli, "EvalRunner", FakeRunner)
    result = runner.invoke(cli.app, ["eval", "--variant", "baseline", "--only", "count,isolated"])
    assert result.exit_code == 0
    assert "ok: 1 images" in result.output
    (request,) = seen
    assert (request.variant, list(request.seeds), request.only) == (
        "baseline",
        [1000, 1001],
        ["count", "isolated"],
    )
