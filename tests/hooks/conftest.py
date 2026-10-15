"""Load and run the `.claude/hooks` scripts (PLAN.md §5.8 "Verification")."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

HOOKS_DIR = Path(__file__).resolve().parents[2] / ".claude" / "hooks"


def load_hook(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"hook_{name}", HOOKS_DIR / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
