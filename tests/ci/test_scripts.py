"""`.github/scripts`: release gate, wheel check, eval gate (PLAN.md §5.4, §5.9)."""

import zipfile
from pathlib import Path

import pytest

from ..conftest import ROOT, load_module

SCRIPTS = ROOT / ".github" / "scripts"
release_gate = load_module(SCRIPTS / "release_gate.py")
check_wheel = load_module(SCRIPTS / "check_wheel.py")
eval_gate = load_module(SCRIPTS / "eval_gate.py")


# release_gate


@pytest.mark.parametrize(
    ("version", "tags", "expected"),
    [
        ("0.0.0", [], (False, False)),  # scaffold version: never released
        ("0.1.0", [], (True, False)),  # no tags yet
        ("0.1.0", ["v0.1.0"], (False, False)),  # same version: nothing to release
        ("0.2.0", ["v0.1.0"], (True, False)),  # bump
        ("0.3.0rc1", ["v0.2.0"], (True, True)),  # pre-release
        ("0.3.0", ["v0.3.0rc1", "v0.2.0"], (True, False)),  # rc → final
        ("0.10.0", ["v0.9.0", "v0.3.0"], (True, False)),  # numeric, not lexical
        ("0.2.0", ["v0.1.0", "nightly", "vbad"], (True, False)),  # non-version tags ignored
        ("0.1", ["v0.1.0"], (False, False)),  # PEP 440 equality
    ],
)
def test_release_gate_decides(version: str, tags: list[str], expected: tuple[bool, bool]) -> None:
    assert release_gate.decide(version, tags) == expected


@pytest.mark.parametrize(
    ("version", "tags"),
    [("0.1.0", ["v0.2.0"]), ("0.3.0rc1", ["v0.3.0"]), ("0.9.0", ["v0.10.0"])],
)
def test_release_gate_refuses_downgrades(version: str, tags: list[str]) -> None:
    with pytest.raises(release_gate.GateError):
        release_gate.decide(version, tags)


# check_wheel


def _wheel(tmp_path: Path, files: dict[str, bytes]) -> str:
    path = tmp_path / "qwi-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        for name, data in files.items():
            wheel.writestr(name, data)
    return str(path)


GOOD_WHEEL = {
    "qwi/__init__.py": b"",
    "qwi-0.0.0.dist-info/licenses/LICENSE": b"apache",
    "qwi-0.0.0.dist-info/licenses/NOTICE": b"notice",
}


def test_check_wheel_accepts_good_wheel(tmp_path: Path) -> None:
    assert check_wheel.wheel_problems(_wheel(tmp_path, GOOD_WHEEL)) == []


def test_check_wheel_flags_missing_and_oversized(tmp_path: Path) -> None:
    files = {
        "other/__init__.py": b"",
        "qwi-0.0.0.dist-info/licenses/LICENSE": b"apache",
        "other/weights.bin": b"\0" * (1024 * 1024 + 1),
    }
    problems = check_wheel.wheel_problems(_wheel(tmp_path, files))
    assert "no qwi/ package" in problems
    assert "missing NOTICE" in problems
    assert any(p.startswith("other/weights.bin") for p in problems)


# eval_gate


@pytest.mark.parametrize(
    "path",
    [
        "models.toml",
        "prompts/rewrite_system.md",
        "prompts/rewrite_system_long.md",
        "src/qwi/stages/guard_policy.toml",
        "src/qwi/stages/encode.py",
        "src/qwi/stages/rerank.py",
    ],
)
def test_eval_gate_triggers(path: str) -> None:
    assert eval_gate.triggered([path], mflux_pin_changed=False) == [path]


def test_eval_gate_ignores_other_files() -> None:
    paths = ["src/qwi/cli.py", "src/qwi/stages/generate.py", "pyproject.toml", "README.md"]
    assert eval_gate.triggered(paths, mflux_pin_changed=False) == []


def test_eval_gate_mflux_pin() -> None:
    diff = (
        '--- a/pyproject.toml\n+++ b/pyproject.toml\n-    "mflux==0.22.0",\n+    "mflux==0.23.0",\n'
    )
    assert eval_gate.mflux_line_changed(diff)
    assert not eval_gate.mflux_line_changed('-    "typer>=0.27",\n+    "typer>=0.28",\n')
    assert eval_gate.triggered([], mflux_pin_changed=True) == ["pyproject.toml (mflux pin)"]


@pytest.mark.parametrize(
    ("body", "ok"),
    [
        ("Eval: eval/sessions/2026-11-01_a_vs_b/report.md", True),
        ("- **Eval:** eval/sessions/x", True),
        ("Eval: n/a — refactor, no behavior change", True),
        ("Eval: n/a - docs only", True),
        ("Eval: n/a", False),  # a reason is required
        ("Eval: <session path, or n/a — reason>", False),  # template placeholder
        ("Eval:", False),
        ("", False),
        ("Evaluation: done", False),
    ],
)
def test_eval_gate_eval_line(body: str, ok: bool) -> None:
    assert eval_gate.has_eval_line(body) is ok
