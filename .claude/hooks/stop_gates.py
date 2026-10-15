"""H3 (Stop): if code or tooling changed, run all gates; exit 2 (once per stop) keeps working."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import project_dir, read_payload, run, run_gates, toolchain_missing


def changed_paths(porcelain: str) -> list[str]:
    """Paths from `git status --porcelain` output (the new name for renames)."""
    paths = []
    for line in porcelain.splitlines():
        if len(line) > 3:
            paths.append(line[3:].split(" -> ")[-1].strip('"'))
    return paths


def needs_gates(paths: list[str]) -> bool:
    return any(
        p.endswith(".py") or p == "pyproject.toml" or p.startswith(".claude/") for p in paths
    )


def main() -> int:
    if read_payload().get("stop_hook_active"):
        return 0
    root = project_dir()
    status = run(("git", "status", "--porcelain", "--untracked-files=all"), root)
    if not needs_gates(changed_paths(status.stdout)):
        return 0
    missing = toolchain_missing(root)
    if missing:
        print(f"stop_gates: skipped ({missing})")
        return 0
    ok, output = run_gates(root)
    if not ok:
        print("All gates fail; fix this before stopping.\n" + output, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
