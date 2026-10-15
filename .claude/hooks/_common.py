"""Shared helpers for the Claude Code hooks (PLAN.md §5.8). Standard library only, Python 3.9+."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# The "All gates" command (§5.5), one step at a time so a failure names its step.
GATES = (
    ("uv", "run", "ruff", "format", "--check"),
    ("uv", "run", "ruff", "check"),
    ("uv", "run", "ty", "check"),
    ("uv", "run", "pytest", "-q"),
)
MAX_FEEDBACK_CHARS = 4000


def project_dir() -> Path:
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[2]


def read_payload() -> dict[str, Any]:
    try:
        data = json.load(sys.stdin)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def toolchain_missing(root: Path) -> str | None:
    """Why the hooks can't run the toolchain yet, or None if they can (§5.8 "No toolchain yet")."""
    if not (root / "pyproject.toml").is_file():
        return "no pyproject.toml yet"
    if shutil.which("uv") is None:
        return "uv is not on PATH"
    return None


def run(
    cmd: tuple[str, ...] | list[str], root: Path, stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=root, input=stdin, capture_output=True, text=True, check=False)


def tail(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_FEEDBACK_CHARS:
        return text
    return "…(truncated)\n" + text[-MAX_FEEDBACK_CHARS:]


def run_gates(root: Path) -> tuple[bool, str]:
    """Run every gate in order; on the first failure return its command and output."""
    for cmd in GATES:
        result = run(cmd, root)
        if result.returncode != 0:
            return False, f"$ {' '.join(cmd)}\n" + tail(result.stdout + result.stderr)
    return True, ""
