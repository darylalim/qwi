"""H4 (SessionStart): print branch, status, next plan task, disk, and offline mode. Never fails."""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import project_dir, run

MIN_FREE_GB = 20  # §5.7
MAX_TASK_CHARS = 240


def next_task(plan: str) -> str | None:
    """First unchecked task in §6 Milestones, prefixed with its milestone heading."""
    plan = "\n" + plan
    section = plan.split("\n## 6.", 1)[-1].split("\n## 7.", 1)[0] if "\n## 6." in plan else ""
    milestone = None
    for line in section.splitlines():
        if line.startswith("### "):
            milestone = re.split(r"\s+[—(]", line[4:], maxsplit=1)[0].strip()
        elif line.lstrip().startswith("- [ ]"):
            task = line.split("- [ ]", 1)[1].strip()
            if len(task) > MAX_TASK_CHARS:
                task = task[:MAX_TASK_CHARS].rstrip() + "…"
            return f"{milestone}: {task}" if milestone else task
    return None


def context_lines(root: Path) -> list[str]:
    lines = []
    branch = run(("git", "branch", "--show-current"), root).stdout.strip() or "(detached)"
    status = [s for s in run(("git", "status", "-s"), root).stdout.splitlines() if s]
    summary = f"{len(status)} changed" if status else "clean"
    lines.append(f"git: {branch}, {summary}")
    lines.extend(f"  {s}" for s in status[:8])
    if len(status) > 8:
        lines.append(f"  … and {len(status) - 8} more")

    plan = root / "PLAN.md"
    task = next_task(plan.read_text(encoding="utf-8")) if plan.is_file() else None
    lines.append(f"next task (PLAN.md §6): {task or 'none unchecked'}")

    free_gb = shutil.disk_usage(root).free / 1e9
    verdict = "ok" if free_gb >= MIN_FREE_GB else "BELOW the 20 GB rule (§5.7)"
    lines.append(f"disk: {free_gb:.0f} GB free, {verdict}")

    offline = os.environ.get("HF_HUB_OFFLINE")
    lines.append(f"HF_HUB_OFFLINE: {offline if offline else 'not set in this shell'}")
    return lines


def main() -> int:
    try:
        print("qwi session context:\n" + "\n".join(context_lines(project_dir())))
    except Exception as exc:
        print(f"qwi session context unavailable: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
