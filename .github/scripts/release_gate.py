"""Decide whether a CI-verified commit on main gets a release (PLAN.md §5.9).

Usage: uv run --no-project --with packaging .github/scripts/release_gate.py <version>
Writes `release=true|false` and `prerelease=true|false` to $GITHUB_OUTPUT (or stdout).
"""

from __future__ import annotations

import os
import subprocess
import sys

from packaging.version import InvalidVersion, Version

# The scaffold version: nothing is released before M1 bumps to 0.1.0 (PLAN.md §5.9, §8).
UNRELEASED = Version("0.0.0")


class GateError(Exception):
    """The version can't be released (e.g. a downgrade)."""


def decide(version: str, tags: list[str]) -> tuple[bool, bool]:
    """(release, prerelease) for `version`, given the existing git tags."""
    new = Version(version)
    if new == UNRELEASED:
        return False, False
    released = []
    for tag in tags:
        if not tag.startswith("v"):
            continue
        try:
            released.append(Version(tag[1:]))
        except InvalidVersion:
            continue
    if new in released:
        return False, False  # no version bump in this commit, or already released
    if released and new < max(released):
        raise GateError(f"version {new} is lower than the latest release v{max(released)}")
    return True, new.is_prerelease


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    tags = subprocess.run(
        ("git", "tag", "--list", "v*"), capture_output=True, text=True, check=True
    ).stdout.split()
    try:
        release, prerelease = decide(argv[0], tags)
    except (GateError, InvalidVersion) as exc:
        print(f"release gate: {exc}", file=sys.stderr)
        return 1
    lines = f"release={str(release).lower()}\nprerelease={str(prerelease).lower()}\n"
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as fh:  # noqa: PTH123 - stdlib-only script
            fh.write(lines)
    print(lines, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
