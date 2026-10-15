"""H1 (PostToolUse, Write|Edit): format and lint an edited .py file; exit 2 on leftover errors."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import project_dir, read_payload, run, tail, toolchain_missing


def target_file(payload: dict, root: Path) -> Path | None:
    """The edited file if it's a *.py file inside the repo, else None."""
    raw = (payload.get("tool_input") or {}).get("file_path")
    if not isinstance(raw, str) or not raw.endswith(".py"):
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        relative = path.relative_to(root.resolve())
    except ValueError:
        return None
    if relative.parts and relative.parts[0] == ".venv":
        return None
    return path


def main() -> int:
    root = project_dir()
    path = target_file(read_payload(), root)
    if path is None or not path.is_file():
        return 0
    missing = toolchain_missing(root)
    if missing:
        print(f"py_format: skipped ({missing})")
        return 0
    run(("uv", "run", "ruff", "format", str(path)), root)
    check = run(("uv", "run", "ruff", "check", "--fix", str(path)), root)
    if check.returncode != 0:
        print(
            f"ruff check still fails on {path.name}:\n" + tail(check.stdout + check.stderr),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
