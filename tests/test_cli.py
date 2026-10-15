import pytest
from typer.testing import CliRunner

from qwi import cli, storage
from qwi.errors import InsufficientDisk

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
