"""CI hygiene (PLAN.md §5.4, §5.9): a PR touching an eval re-run trigger needs an `Eval:` line.

Usage: PR_BODY="..." python3 -I eval_gate.py --range <base>...<head>
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from fnmatch import fnmatch

TRIGGER_PATTERNS = (
    "models.toml",
    "prompts/rewrite_system*.md",
    "*guard_policy.toml",
    "src/qwi/stages/encode.py",
    "src/qwi/stages/rerank.py",
)
# `Eval: eval/sessions/<...>` or `Eval: n/a — <reason>`; the template's `<...>` placeholder fails.
EVAL_LINE = re.compile(r"^[\s*-]*\**Eval:\**\s*(?P<value>.*?)\s*$", re.MULTILINE)
NA_WITH_REASON = re.compile(r"n/a\s*[\u2014\u2013:-]+\s*\S", re.IGNORECASE)  # em/en dash, :, -


def triggered(paths: list[str], mflux_pin_changed: bool) -> list[str]:
    hits = [p for p in paths if any(fnmatch(p, pattern) for pattern in TRIGGER_PATTERNS)]
    if mflux_pin_changed:
        hits.append("pyproject.toml (mflux pin)")
    return hits


def has_eval_line(body: str) -> bool:
    for match in EVAL_LINE.finditer(body):
        value = match.group("value")
        if not value or value.startswith("<"):
            continue
        if value.lower().startswith("n/a") and not NA_WITH_REASON.match(value):
            continue
        return True
    return False


def mflux_line_changed(diff: str) -> bool:
    return any(
        line[:1] in "+-" and not line.startswith(("+++", "---")) and "mflux" in line
        for line in diff.splitlines()
    )


def git(*args: str) -> str:
    return subprocess.run(("git", *args), capture_output=True, text=True, check=True).stdout


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "--range":
        print(__doc__, file=sys.stderr)
        return 2
    spec = argv[1]
    paths = [p for p in git("diff", "--name-only", spec).splitlines() if p]
    hits = triggered(paths, mflux_line_changed(git("diff", spec, "--", "pyproject.toml")))
    if not hits:
        print("eval gate: no re-run triggers touched")
        return 0
    if has_eval_line(os.environ.get("PR_BODY", "")):
        print("eval gate: triggers touched, Eval: line present → " + ", ".join(hits))
        return 0
    print(
        "eval gate: this PR touches eval re-run triggers (§5.4):\n- "
        + "\n- ".join(hits)
        + "\nAdd an `Eval:` line to the PR body: a session path, or `n/a — <reason>`.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
