"""H2 (PreToolUse, `git commit`): staged-file rules, then all gates; exit 2 blocks the commit."""

from __future__ import annotations

import re
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import project_dir, read_payload, run, run_gates, toolchain_missing

MAX_BYTES = 5 * 1024 * 1024
WEIGHT_SUFFIXES = {".safetensors", ".gguf", ".npz"}
COMMIT_RE = re.compile(r"\bgit\b[^;&|\n]*\bcommit\b")


def is_commit(command: str) -> bool:
    """Fallback for the settings `if` filter: does this shell command run `git commit`?"""
    return bool(COMMIT_RE.search(command))


def file_problems(files: list[tuple[str, int]]) -> list[str]:
    """Rule violations for (repo-relative path, size in bytes) pairs (§5.8 H2)."""
    problems = []
    for name, size in files:
        path = PurePosixPath(name)
        if size > MAX_BYTES:
            problems.append(f"{name}: {size / 1e6:.1f} MB is over the 5 MB limit")
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            problems.append(f"{name}: model weights are never committed (§5.7)")
        if path.parts and path.parts[0] == "var":
            problems.append(f"{name}: var/ is runtime data ($QWI_HOME)")
        if path.suffix.lower() == ".png" and not name.startswith("tests/fixtures/"):
            problems.append(f"{name}: images are only allowed under tests/fixtures/")
    return problems


def staged_files(root: Path) -> list[tuple[str, int]]:
    names = run(("git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"), root)
    paths = [p for p in names.stdout.split("\0") if p]
    if not paths:
        return []
    sizes = run(
        ("git", "cat-file", "--batch-check=%(objectsize)"),
        root,
        stdin="".join(f":{p}\n" for p in paths),
    ).stdout.split()
    return [(p, int(s)) for p, s in zip(paths, sizes) if s.isdigit()]  # noqa: B905 - Python 3.9


def main() -> int:
    payload = read_payload()
    command = (payload.get("tool_input") or {}).get("command", "")
    if not isinstance(command, str) or not is_commit(command):
        return 0
    root = project_dir()
    problems = file_problems(staged_files(root))
    if problems:
        print("Commit blocked by staged-file rules:\n- " + "\n- ".join(problems), file=sys.stderr)
        return 2
    missing = toolchain_missing(root)
    if missing:
        print(f"pre_commit: gates skipped ({missing})")
        return 0
    ok, output = run_gates(root)
    if not ok:
        print("Commit blocked: All gates failed.\n" + output, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
