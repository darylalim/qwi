import logging
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from qwi import cli, logs, paths, record, storage
from qwi.errors import InsufficientDisk
from qwi.events import CandidateReady, DenoiseStep, RunStarted, StageFinished, StageStarted
from qwi.record import RunRecord

runner = CliRunner()


def test_doctor_prints_a_row_per_check_and_exits_0(monkeypatch: pytest.MonkeyPatch) -> None:
    checks = [
        storage.Check("disk", "pass", "46.8 GB free"),
        storage.Check("runs size", "warn", "2.1 GB"),
    ]
    monkeypatch.setattr(storage, "doctor", lambda: checks)
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0
    assert result.output.splitlines() == [
        "ok    disk       46.8 GB free",
        "WARN  runs size  2.1 GB",
    ]


def test_doctor_exits_1_on_any_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "doctor", lambda: [storage.Check("disk", "fail", "5 GB free")])
    assert runner.invoke(cli.app, ["doctor"]).exit_code == 1


def test_models_fetch_reports_insufficient_disk_with_exit_6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(role: str, *, dry_run: bool) -> None:
        raise InsufficientDisk(role, 10 * 10**9, 29 * 10**9, 20 * 10**9)

    monkeypatch.setattr(storage, "fetch", refuse)
    result = runner.invoke(cli.app, ["models", "fetch", "rerank"])
    assert result.exit_code == 6
    assert "Not enough disk for rerank" in result.output


def test_models_fetch_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []

    def fake(role: str, *, dry_run: bool) -> None:
        seen.append(dry_run)

    monkeypatch.setattr(storage, "fetch", fake)
    result = runner.invoke(cli.app, ["models", "fetch", "image", "--dry-run"])
    assert (result.exit_code, seen) == (0, [True])
    assert "nothing downloaded" in result.output


class FakePipeline:
    def __init__(self, status: str, error: dict | None = None) -> None:
        self.status, self.error = status, error
        self.requests: list = []

    def __call__(self) -> "FakePipeline":
        return self

    def run(self, request, on_event) -> RunRecord:
        self.requests.append(request)
        on_event(RunStarted(Path("/runs/20261015-093005-abcdef"), [7, 8]))
        on_event(StageStarted("generate"))
        on_event(DenoiseStep(0, 1, 2))
        on_event(CandidateReady(0, 7, Path("/runs/x/cand_0_seed7.png")))
        on_event(StageFinished("generate", 251.6, 13.8))
        rec = RunRecord(request.prompt, status=self.status, error=self.error)
        rec.candidates = [{"index": 0}]
        return rec


def test_generate_renders_events_and_exits_0(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakePipeline("ok")
    monkeypatch.setattr(cli, "Pipeline", fake)
    result = runner.invoke(cli.app, ["generate", "a red apple", "-n", "2", "--seed", "7"])
    assert result.exit_code == 0
    assert "run 20261015-093005-abcdef (seeds 7…8)" in result.output
    assert "candidate 1 (seed 7): /runs/x/cand_0_seed7.png" in result.output
    assert "generate: 251.6 s, peak 13.8 GB" in result.output
    (request,) = fake.requests
    assert (request.prompt, request.n, request.seed, request.width) == ("a red apple", 2, 7, 768)


def test_generate_exit_code_follows_the_record(monkeypatch: pytest.MonkeyPatch) -> None:
    error = {"type": "SwapRisk", "message": "Not enough free memory for generate"}
    monkeypatch.setattr(cli, "Pipeline", FakePipeline("error", error))
    result = runner.invoke(cli.app, ["generate", "p"])
    assert result.exit_code == 4
    assert "error: Not enough free memory for generate" in result.output


def test_generate_reports_busy_with_exit_3(monkeypatch: pytest.MonkeyPatch) -> None:
    from qwi.errors import PipelineBusy

    class Busy:
        def run(self, request, on_event) -> None:
            raise PipelineBusy("2026-10-15T09:30:05")

    monkeypatch.setattr(cli, "Pipeline", Busy)
    result = runner.invoke(cli.app, ["generate", "p"])
    assert (result.exit_code, "Another run is in progress" in result.output) == (3, True)


def _saved_run(name: str, **fields: Any) -> Path:
    run = paths.runs_dir() / name
    run.mkdir(parents=True)
    rec = RunRecord(
        "a red apple",
        seeds=[7, 8],
        config={"width": 768, "height": 768, "n": 2, "steps": 20, "encoder": "A-q4"},
        **fields,
    )
    record.write(rec, run)
    return run


def test_runs_show_prints_the_summary() -> None:
    _saved_run(
        "20261015-093005-abcdef",
        status="ok",
        candidates=[
            {"index": 0, "seed": 7, "path": "cand_0_seed7.png", "seconds": 125.8, "peak_gb": 13.8}
        ],
        timings_s={"generate": 251.6},
        peak_mem_gb={"generate": 13.8},
        swap_gb={"generate": 0.0},
    )
    result = runner.invoke(cli.app, ["runs", "show", "20261015-0930"])  # a unique prefix
    assert result.exit_code == 0
    out = result.output
    assert "run      20261015-093005-abcdef  [ok]" in out
    assert "768x768, 20 steps, n=2, encoder A-q4, seeds [7, 8]" in out
    assert "cand 0  seed 7  cand_0_seed7.png  125.8 s  peak 13.8 GB" in out
    assert "swap     generate 0 GB" in out


def test_runs_show_repairs_an_interrupted_run() -> None:
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    _saved_run("20261015-093005-abcdef", pid=child.pid)
    result = runner.invoke(cli.app, ["runs", "show", "20261015-093005-abcdef"])
    assert "[interrupted]" in result.output


def test_runs_show_log_filters_by_level_and_prints_tracebacks() -> None:
    run = _saved_run("20261015-093005-abcdef", status="error")
    with logs.run_log(run):
        logs.event(logging.INFO, "generate", "candidate_ready", "cand_0_seed7.png", seed=7)
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            logs.exception("generate", "error", "Unexpected error", type="RuntimeError")
    result = runner.invoke(cli.app, ["runs", "show", run.name, "--log", "--level", "warning"])
    assert result.exit_code == 0
    assert "candidate_ready" not in result.output
    assert "ERROR   generate  error: Unexpected error  type=RuntimeError" in result.output
    assert "    RuntimeError: boom" in result.output


@pytest.mark.parametrize("run_id", ["nope", "20261015"])
def test_runs_show_unknown_or_ambiguous_id_exits_1(run_id: str) -> None:
    _saved_run("20261015-093005-abcdef")
    _saved_run("20261015-093010-123456")
    result = runner.invoke(cli.app, ["runs", "show", run_id])
    assert result.exit_code == 1
    assert f"Run {run_id!r}" in result.output
