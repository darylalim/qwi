"""Load and run the `.claude/hooks` scripts (PLAN.md §5.8 "Verification")."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ..conftest import ROOT, load_module

HOOKS_DIR = ROOT / ".claude" / "hooks"


def load_hook(name: str) -> ModuleType:
    return load_module(HOOKS_DIR / f"{name}.py")


def run_hook(name: str, payload: dict, project: Path) -> subprocess.CompletedProcess[str]:
    """Run a hook the way Claude Code does: JSON on stdin, `CLAUDE_PROJECT_DIR` set."""
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(project)}
    return subprocess.run(
        [sys.executable, "-I", str(HOOKS_DIR / f"{name}.py")],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=60,
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """An empty git repo with no toolchain (no pyproject.toml)."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    return tmp_path


def stage(repo: Path, name: str, data: bytes = b"x") -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    subprocess.run(["git", "add", name], cwd=repo, check=True)


def commit(repo: Path, message: str = "c") -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", message],
        cwd=repo,
        check=True,
    )


def run_range(repo: Path, spec: str) -> subprocess.CompletedProcess[str]:
    """`pre_commit.py --range <spec>` as CI's hygiene job runs it (§5.9)."""
    return subprocess.run(
        [sys.executable, "-I", str(HOOKS_DIR / "pre_commit.py"), "--range", spec],
        capture_output=True,
        text=True,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(repo)},
        check=False,
        timeout=60,
    )
