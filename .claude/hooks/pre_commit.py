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
IMAGE_DIRS = ("tests/fixtures/", "docs/screenshots/")


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
        if path.suffix.lower() == ".png" and not name.startswith(IMAGE_DIRS):
            problems.append(f"{name}: images are only allowed under {' and '.join(IMAGE_DIRS)}")
    return problems


def changed_files(root: Path, diff_args: tuple[str, ...], rev: str) -> list[tuple[str, int]]:
    """(path, size) of files added or modified by `git diff <diff_args>`, sized at `rev`.

    `rev` is "" for the index (staged blobs), or a commit for a range.
    """
    names = run(("git", "diff", *diff_args, "--name-only", "--diff-filter=ACMR", "-z"), root)
    if names.returncode != 0:
        raise RuntimeError(names.stderr.strip() or "git diff failed")
    paths = [p for p in names.stdout.split("\0") if p]
    if not paths:
        return []
    sizes = run(
        ("git", "cat-file", "--batch-check=%(objectsize)"),
        root,
        stdin="".join(f"{rev}:{p}\n" for p in paths),
    ).stdout.split()
    return [(p, int(s)) for p, s in zip(paths, sizes) if s.isdigit()]  # noqa: B905 - Python 3.9


def range_end(spec: str) -> str:
    """The newer side of `A..B` / `A...B` (HEAD when omitted, or for a bare revision)."""
    for sep in ("...", ".."):
        if sep in spec:
            return spec.split(sep, 1)[1] or "HEAD"
    return "HEAD"


def check_range(spec: str) -> int:
    """CI hygiene (§5.9): apply the staged-file rules to every file a commit range changes."""
    diff_spec = spec if ".." in spec else f"{spec}..HEAD"
    problems = file_problems(changed_files(project_dir(), (diff_spec,), range_end(spec)))
    if problems:
        print(f"File rules failed for {spec}:\n- " + "\n- ".join(problems), file=sys.stderr)
        return 1
    print(f"File rules passed for {spec}")
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--range":
        return check_range(sys.argv[2])
    payload = read_payload()
    command = (payload.get("tool_input") or {}).get("command", "")
    if not isinstance(command, str) or not is_commit(command):
        return 0
    root = project_dir()
    problems = file_problems(changed_files(root, ("--cached",), ""))
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
