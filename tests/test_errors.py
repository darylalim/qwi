import errno
import json

import pytest

from qwi import errors

from .conftest import ROOT

FIXTURE = json.loads((ROOT / "tests/fixtures/mlx_errors.json").read_text())


@pytest.mark.parametrize("case", FIXTURE["observed"], ids=lambda c: c["probe"])
def test_classify_maps_observed_mlx_failures(case: dict) -> None:
    exc = RuntimeError(case["message"])
    classified = errors.classify(exc, "generate")
    assert type(classified).__name__ == case["classify_as"]


@pytest.mark.parametrize("fragment", FIXTURE["binary_only"])
def test_classify_maps_mlx_format_strings_from_the_binary(fragment: str) -> None:
    message = fragment.replace("{}", "Insufficient Memory") + "16 GB)"
    classified = errors.classify(RuntimeError(message), "generate")
    assert isinstance(classified, errors.DeviceOutOfMemory)
    assert classified.stage == "generate"


def test_every_pattern_is_backed_by_the_fixture() -> None:
    known = [c["message"] for c in FIXTURE["observed"]] + FIXTURE["binary_only"]
    for pattern in errors.OOM_PATTERNS:
        assert any(pattern in message for message in known), pattern


def test_unknown_runtime_errors_stay_unexpected() -> None:
    exc = RuntimeError("[gather] Index out of bounds")
    assert errors.classify(exc, "generate") is exc
    assert (errors.status(exc), errors.exit_code(exc)) == ("error", 1)


def test_enospc_is_disk_full_and_other_os_errors_are_not() -> None:
    full = errors.classify(OSError(errno.ENOSPC, "No space left on device"), "generate")
    assert isinstance(full, errors.DiskFull)
    assert "qwi runs prune" in str(full)
    denied = OSError(errno.EACCES, "Permission denied")
    assert errors.classify(denied, "generate") is denied


def _named(name: str, module: str) -> Exception:
    return type(name, (Exception,), {"__module__": module})()


def test_mflux_stop_is_cancelled_but_a_lookalike_is_not() -> None:
    stop = _named("StopImageGenerationException", "mflux.utils.exceptions")
    assert isinstance(errors.classify(stop, "generate"), errors.GenerationCancelled)
    lookalike = _named("StopImageGenerationException", "elsewhere")
    assert errors.classify(lookalike, "generate") is lookalike


def test_qwi_errors_pass_through_classify() -> None:
    exc = errors.ModelUnavailable("image", "missing")
    assert errors.classify(exc, "generate") is exc


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (errors.GuardBlocked("unsafe: violent"), "blocked", 2),
        (errors.PipelineBusy("2026-10-15T09:30:05"), None, 3),
        (errors.InsufficientMemory("generate", 30.0, 22.9), "error", 4),
        (errors.SwapRisk("generate", 18.0, 12.0), "error", 4),
        (errors.DeviceOutOfMemory("generate", "swapped"), "error", 4),
        (errors.ModelUnavailable("image", "missing"), "error", 5),
        (errors.DiskFull(10**9), "error", 6),
        (errors.GenerationCancelled(2, 4), "cancelled", 130),
        (ValueError("boom"), "error", 1),
    ],
    ids=lambda v: type(v).__name__ if isinstance(v, BaseException) else str(v),
)
def test_status_and_exit_code_follow_the_plan_table(
    exc: BaseException, status: str | None, code: int
) -> None:
    assert errors.status(exc) == status
    assert errors.exit_code(exc) == code


def test_messages_carry_the_plan_actions() -> None:
    assert str(errors.GenerationCancelled(2, 4)) == "Cancelled after 2 of 4 candidates"
    assert str(errors.GenerationCancelled()) == "Cancelled"
    assert "--allow-swap" in str(errors.SwapRisk("generate", 18.0, 12.0))
    assert "512²" in str(errors.InsufficientMemory("generate", 30.0, 22.9))
    assert "qwi models fetch image" in str(errors.ModelUnavailable("image", "missing"))
    assert "started 2026-10-15T09:30:05" in str(errors.PipelineBusy("2026-10-15T09:30:05"))


def test_keyboard_interrupt_is_cancelled() -> None:
    assert isinstance(errors.classify(KeyboardInterrupt(), "generate"), errors.GenerationCancelled)


@pytest.mark.parametrize(
    ("status", "error_type", "code"),
    [
        ("ok", None, 0),
        ("cancelled", None, 130),
        ("blocked", None, 2),
        ("error", "SwapRisk", 4),
        ("error", "ModelUnavailable", 5),
        ("error", "ValueError", 1),
        ("interrupted", None, 1),
    ],
)
def test_exit_code_for_a_finished_record(status: str, error_type: str | None, code: int) -> None:
    assert errors.exit_code_for(status, error_type) == code
