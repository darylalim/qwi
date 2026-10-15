"""Decision logic and no-toolchain paths of hooks H1-H4 (PLAN.md §5.8)."""

import ast
from pathlib import Path

import pytest

from .conftest import HOOKS_DIR, load_hook, run_hook, stage

py_format = load_hook("py_format")
pre_commit = load_hook("pre_commit")
stop_gates = load_hook("stop_gates")
session_context = load_hook("session_context")


@pytest.mark.parametrize("script", sorted(HOOKS_DIR.glob("*.py")), ids=lambda p: p.name)
def test_hooks_parse_as_python_39(script: Path) -> None:
    # `python3` on the hook PATH may be Apple's /usr/bin/python3.
    ast.parse(script.read_text(), feature_version=(3, 9))


# H1: py_format


@pytest.mark.parametrize(
    ("file_path", "expected"),
    [
        ("src/qwi/paths.py", "src/qwi/paths.py"),
        ("README.md", None),
        ("/elsewhere/outside.py", None),
        (".venv/lib/site.py", None),
        (None, None),
    ],
)
def test_py_format_targets_repo_python_files(
    tmp_path: Path, file_path: str | None, expected: str | None
) -> None:
    payload = {"tool_name": "Edit", "tool_input": {"file_path": file_path}}
    target = py_format.target_file(payload, tmp_path)
    assert target == (tmp_path / expected).resolve() if expected else target is None


def test_py_format_absolute_path_inside_repo(tmp_path: Path) -> None:
    payload = {"tool_input": {"file_path": str(tmp_path / "a.py")}}
    assert py_format.target_file(payload, tmp_path) == (tmp_path / "a.py").resolve()


def test_py_format_skips_without_toolchain(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x=1\n")
    result = run_hook("py_format", {"tool_input": {"file_path": "a.py"}}, tmp_path)
    assert result.returncode == 0
    assert "no pyproject.toml yet" in result.stdout
    assert (tmp_path / "a.py").read_text() == "x=1\n"


# H2: pre_commit


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ('git commit -m "x"', True),
        ("git add -A && git commit -q -F -", True),
        ("git -C repo commit --amend --no-edit", True),
        ("git status", False),
        ("git log --grep commit", True),  # over-matches; running the gates is harmless
        ("echo commit", False),
    ],
)
def test_pre_commit_detects_commits(command: str, expected: bool) -> None:
    assert pre_commit.is_commit(command) is expected


def test_pre_commit_file_rules() -> None:
    files = [
        ("src/qwi/cli.py", 100),
        ("tests/fixtures/ok.png", 100),
        ("big.bin", 6 * 1024 * 1024),
        ("weights/model.safetensors", 10),
        ("q.GGUF", 10),
        ("emb.npz", 10),
        ("var/runs/run.json", 10),
        ("docs/shot.png", 10),
    ]
    flagged = {p.split(":")[0] for p in pre_commit.file_problems(files)}
    assert flagged == {
        "big.bin",
        "weights/model.safetensors",
        "q.GGUF",
        "emb.npz",
        "var/runs/run.json",
        "docs/shot.png",
    }


def test_pre_commit_blocks_staged_weights(git_repo: Path) -> None:
    stage(git_repo, "model.safetensors")
    result = run_hook("pre_commit", {"tool_input": {"command": "git commit -m x"}}, git_repo)
    assert result.returncode == 2
    assert "model.safetensors" in result.stderr


def test_pre_commit_blocks_oversized_staged_file(git_repo: Path) -> None:
    stage(git_repo, "data.bin", b"\0" * (5 * 1024 * 1024 + 1))
    result = run_hook("pre_commit", {"tool_input": {"command": "git commit -m x"}}, git_repo)
    assert result.returncode == 2
    assert "5 MB" in result.stderr


def test_pre_commit_skips_gates_without_toolchain(git_repo: Path) -> None:
    stage(git_repo, "notes.txt")
    result = run_hook("pre_commit", {"tool_input": {"command": "git commit -m x"}}, git_repo)
    assert result.returncode == 0
    assert "no pyproject.toml yet" in result.stdout


def test_pre_commit_ignores_other_commands(git_repo: Path) -> None:
    stage(git_repo, "model.safetensors")
    result = run_hook("pre_commit", {"tool_input": {"command": "git status"}}, git_repo)
    assert result.returncode == 0


# H3: stop_gates


def test_stop_gates_parses_porcelain() -> None:
    porcelain = ' M src/qwi/cli.py\nR  old.txt -> .claude/settings.json\n?? "a b.md"\n'
    assert stop_gates.changed_paths(porcelain) == [
        "src/qwi/cli.py",
        ".claude/settings.json",
        "a b.md",
    ]


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        (["src/qwi/cli.py"], True),
        (["pyproject.toml"], True),
        ([".claude/settings.json"], True),
        (["PLAN.md", "README.md"], False),
        ([], False),
    ],
)
def test_stop_gates_only_for_code_changes(paths: list[str], expected: bool) -> None:
    assert stop_gates.needs_gates(paths) is expected


def test_stop_gates_respects_loop_guard(git_repo: Path) -> None:
    (git_repo / "pyproject.toml").write_text("[project]\n")
    (git_repo / "a.py").write_text("x = 1\n")
    result = run_hook("stop_gates", {"stop_hook_active": True}, git_repo)
    assert result.returncode == 0
    assert result.stdout == ""


def test_stop_gates_skips_without_toolchain(git_repo: Path) -> None:
    (git_repo / "pkg").mkdir()
    (git_repo / "pkg" / "a.py").write_text("x = 1\n")  # untracked file in an untracked dir
    result = run_hook("stop_gates", {"stop_hook_active": False}, git_repo)
    assert result.returncode == 0
    assert "no pyproject.toml yet" in result.stdout


# H4: session_context

PLAN = """\
## 6. Milestones
### M0a — Repo foundation (do first)
- [x] Scaffold.
- [ ] Guardrails (§5.8).
### M0b — Platform spike
- [ ] Storage first.
## 7. Risks
- [ ] Not a task.
"""


def test_session_context_finds_first_unchecked_task() -> None:
    assert session_context.next_task(PLAN) == "M0a: Guardrails (§5.8)."


def test_session_context_none_when_all_done() -> None:
    assert (
        session_context.next_task(PLAN.replace("- [ ] G", "- [x] G").replace("- [ ] S", "- [x] S"))
        is None
    )


def test_session_context_never_fails_outside_git(tmp_path: Path) -> None:
    result = run_hook("session_context", {}, tmp_path)
    assert result.returncode == 0
    assert "next task (PLAN.md §6): none unchecked" in result.stdout
    assert "disk:" in result.stdout
