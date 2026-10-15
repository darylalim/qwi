from pathlib import Path

import pytest
from typer.testing import CliRunner

from qwi import cli, storage
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
