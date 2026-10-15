import json

import pytest

from qwi import paths, preflight
from qwi.config import DEFAULTS, GiB
from qwi.errors import InsufficientMemory, SwapRisk

LIMIT = 26800603136  # max_recommended_working_set_size on the M2 Max (24.96 GiB)
HEADROOM_GB = DEFAULTS.preflight_headroom_bytes / 1e9


def test_generate_estimate_reproduces_the_measured_peaks() -> None:
    assert preflight.generate_gb(256, 256) == pytest.approx(12.15, abs=0.01)
    assert preflight.generate_gb(768, 768) == pytest.approx(17.78, abs=0.01)
    assert preflight.generate_gb(2048, 2048) == pytest.approx(37.92, abs=0.01)
    # The M1 exit runs' max pipeline peak at 768² with the encoder evicted.
    assert preflight.generate_gb(768, 768, encoder_resident=False) == pytest.approx(14.95, abs=0.01)


def test_generate_estimate_interpolates_and_extends() -> None:
    sizes = [128, 256, 512, 768, 1024, 2048, 2560]
    estimates = [preflight.generate_gb(s, s) for s in sizes]
    assert estimates == sorted(estimates)  # bigger images never estimate less
    assert 12.15 < preflight.generate_gb(512, 512) < 17.78
    # 128²: below the first measurement, the first segment is extended
    assert preflight.generate_gb(128, 128) == pytest.approx(11.62, abs=0.01)


def test_ok_when_the_estimate_fits() -> None:
    check = preflight.check("generate", 17.8, limit_bytes=LIMIT, available_bytes=25 * GiB)
    assert check.decision == "ok"
    assert check.limit_gb == pytest.approx(LIMIT / 1e9 - HEADROOM_GB)


def test_refuses_over_the_wired_limit_even_with_allow_swap() -> None:
    with pytest.raises(InsufficientMemory) as excinfo:
        preflight.check(
            "generate", 37.9, allow_swap=True, limit_bytes=LIMIT, available_bytes=30 * GiB
        )
    assert not isinstance(excinfo.value, SwapRisk)


def test_swap_risk_is_refused_without_allow_swap() -> None:
    with pytest.raises(SwapRisk):
        preflight.check("generate", 17.8, limit_bytes=LIMIT, available_bytes=16 * GiB)


def test_swap_risk_continues_with_allow_swap() -> None:
    check = preflight.check(
        "generate", 17.8, allow_swap=True, limit_bytes=LIMIT, available_bytes=16 * GiB
    )
    assert check.decision == "swap_risk_allowed"


def test_headroom_is_subtracted_from_available_memory() -> None:
    # 19 GB available fits 17.8 GB only without the 2 GiB headroom.
    with pytest.raises(SwapRisk):
        preflight.check("generate", 17.8, limit_bytes=LIMIT, available_bytes=19 * 10**9)


def test_reads_the_machine_when_not_given(monkeypatch: pytest.MonkeyPatch) -> None:
    import mlx.core as mx

    monkeypatch.setattr(mx, "device_info", lambda: {"max_recommended_working_set_size": LIMIT})
    assert preflight.wired_limit_bytes() == LIMIT
    assert preflight.check("guard", 0.1).decision == "ok"


def test_as_record_rounds_for_run_json() -> None:
    check = preflight.Check(17.7812, 21.3456, 24.8059, "ok")
    assert check.as_record() == {
        "est_gb": 17.78,
        "avail_gb": 21.35,
        "limit_gb": 24.81,
        "decision": "ok",
    }


def _run(name: str, config: dict, peaks: dict) -> None:
    run = paths.runs_dir() / name
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"config": config, "peak_mem_gb": peaks}))


def test_history_uses_the_largest_matching_peak() -> None:
    size = {"width": 768, "height": 768, "n": 4, "encoder": "A-q4"}
    _run("20261015-090000-aaaaaa", size, {"generate": 17.9})
    _run("20261015-090100-bbbbbb", size, {"generate": 18.4})
    _run("20261015-090200-cccccc", {**size, "width": 1024}, {"generate": 25.0})
    _run("20261015-090300-dddddd", size, {"guard": 1.0})
    assert preflight.history_gb("generate", size) == 18.4


def test_history_only_looks_at_the_last_matching_runs() -> None:
    size = {"width": 768, "height": 768}
    _run("20261015-080000-000000", size, {"generate": 99.0})  # oldest, outside the window
    for i in range(preflight.HISTORY_RUNS):
        _run(f"20261015-09{i:02d}00-{i:06d}", size, {"generate": 18.0})
    assert preflight.history_gb("generate", size) == 18.0


def test_history_skips_unreadable_records_and_handles_no_runs() -> None:
    assert preflight.history_gb("generate", {}) == 0.0
    bad = paths.runs_dir() / "20261015-090000-badbad"
    bad.mkdir(parents=True)
    (bad / "run.json").write_text("{half a record")
    assert preflight.history_gb("generate", {}) == 0.0


def test_swap_limit_rises_with_allow_swap() -> None:
    assert preflight.swap_limit_bytes(False) == DEFAULTS.swap_limit_bytes
    assert preflight.swap_limit_bytes(True) == DEFAULTS.allow_swap_limit_bytes
