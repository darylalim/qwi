"""The sidebar's `qwi doctor` status line (PLAN.md §5.6, §5.7), without Streamlit.

The line runs only the quick checks. The folder sizes and the run-history scan walk every file
under `$QWI_HOME`, too slow for a line redrawn on each rerun; `qwi doctor` prints the full table.
"""

from dataclasses import dataclass
from typing import Literal

from qwi import storage
from qwi.storage import Check, Status

BadgeColor = Literal["green", "orange", "red"]

COLORS: dict[Status, BadgeColor] = {"pass": "green", "warn": "orange", "fail": "red"}
ICONS: dict[Status, str] = {
    "pass": ":material/check_circle:",
    "warn": ":material/warning:",
    "fail": ":material/error:",
}


@dataclass(frozen=True)
class StatusLine:
    status: Status
    label: str
    problems: list[str]  # "name: detail" for each check that didn't pass, failures first

    @property
    def color(self) -> BadgeColor:
        return COLORS[self.status]

    @property
    def icon(self) -> str:
        return ICONS[self.status]


def quick_checks() -> list[Check]:
    """Disk, models, `$QWI_HOME`, offline mode and torch: the rows that can stop a run."""
    return [
        storage.check_disk(),
        *storage.check_models(),
        storage.check_home(),
        storage.check_offline(),
        storage.check_torch(),
    ]


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def summarize(checks: list[Check]) -> StatusLine:
    failed = [c for c in checks if c.status == "fail"]
    warned = [c for c in checks if c.status == "warn"]
    problems = [f"{c.name}: {c.detail}" for c in failed + warned]
    if failed:
        return StatusLine("fail", f"Doctor: {_count(len(failed), 'failure')}", problems)
    if warned:
        return StatusLine("warn", f"Doctor: {_count(len(warned), 'warning')}", problems)
    return StatusLine("pass", "Doctor: ok", problems)
