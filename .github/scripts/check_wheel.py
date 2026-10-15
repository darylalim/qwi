"""Release sanity check (PLAN.md §5.9): the wheel has qwi/, LICENSE and NOTICE, no file over 1 MB.

Usage: python3 -I .github/scripts/check_wheel.py dist/*.whl
"""

from __future__ import annotations

import sys
import zipfile

MAX_BYTES = 1024 * 1024
LICENSE_FILES = ("LICENSE", "NOTICE")


def wheel_problems(path: str) -> list[str]:
    with zipfile.ZipFile(path) as wheel:
        infos = wheel.infolist()
    names = [info.filename for info in infos]
    problems = []
    if not any(name.startswith("qwi/") for name in names):
        problems.append("no qwi/ package")
    for license_file in LICENSE_FILES:
        if not any(name.endswith(f".dist-info/licenses/{license_file}") for name in names):
            problems.append(f"missing {license_file}")
    problems.extend(
        f"{info.filename}: {info.file_size / 1e6:.1f} MB is over the 1 MB limit"
        for info in infos
        if info.file_size > MAX_BYTES
    )
    return problems


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    failed = False
    for path in argv:
        problems = wheel_problems(path)
        if problems:
            failed = True
            print(f"{path}:\n- " + "\n- ".join(problems), file=sys.stderr)
        else:
            print(f"{path}: ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
