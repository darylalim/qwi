"""What the History page shows, from `runs/*/run.json` (PLAN.md §5.6, §5.7, §5.11). No `st` here.

Listing repairs `interrupted` runs the way `qwi runs show` does: a `running` record whose run
can't still be going (lock free or pid gone) is rewritten as `interrupted` the first time it's
listed (§5.11 "Crash-safe run records").
"""

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from qwi import lock, record, storage
from qwi.record import RunRecord
from qwi.ui.view import STAGE_LABELS, CandidateView

# `st.badge` colors per status: semantic names only, the theme picks the shades (§5.6).
BadgeColor = Literal["green", "blue", "orange", "gray", "red"]
STATUS_COLORS: dict[str, BadgeColor] = {
    "ok": "green",
    "running": "blue",
    "interrupted": "orange",
    "cancelled": "gray",
    "blocked": "red",
    "error": "red",
}


@dataclass(frozen=True)
class RunEntry:
    run_dir: Path
    record: RunRecord

    @property
    def run_id(self) -> str:
        return self.run_dir.name

    @property
    def started(self) -> str:
        return self.record.started_at[:19].replace("T", " ")

    @property
    def best_prob(self) -> float | None:
        rec = self.record
        if rec.best_index is None or rec.best_index >= len(rec.candidates):
            return None
        return rec.candidates[rec.best_index].get("prob")

    @property
    def total_seconds(self) -> float | None:
        return sum(self.record.timings_s.values()) if self.record.timings_s else None

    @property
    def status_color(self) -> BadgeColor:
        return STATUS_COLORS.get(self.record.status, "gray")

    def row(self) -> dict[str, Any]:
        """One row of the runs table."""
        return {
            "Started": self.started,
            "Prompt": self.record.user_prompt,
            "Status": self.record.status,
            "Best prob": self.best_prob,
            "Seconds": self.total_seconds,
            "Starred": self.record.starred,
        }

    def candidates(self) -> tuple[CandidateView, ...]:
        return tuple(
            CandidateView(
                c["index"],
                c["seed"],
                self.run_dir / c["path"],
                c.get("prob"),
                c.get("margin"),
                c.get("rank"),
            )
            for c in self.record.candidates
        )

    def config_line(self) -> str | None:
        cfg = self.record.config
        if not cfg:
            return None
        size = f"{cfg.get('width')}x{cfg.get('height')}"
        seeds = ", ".join(map(str, self.record.seeds)) or "-"
        return (
            f"{size}, {cfg.get('steps')} steps, N={cfg.get('n')}, "
            f"encoder {cfg.get('encoder')}, seeds {seeds}"
        )

    def guard_lines(self) -> list[str]:
        """Each guard pass's level and categories, then the decision and its mode (§5.2)."""
        guard = self.record.guard
        if not guard:
            return []
        lines = []
        for check in ("input", "rewrite"):
            if verdict := guard.get(check):
                level = verdict.get("level") or "unparseable"
                categories = ", ".join(verdict.get("categories") or [])
                lines.append(
                    f"{check.capitalize()}: {level}" + (f" ({categories})" if categories else "")
                )
        decision = f"Decision: {guard.get('decision')} ({guard.get('mode')} mode)"
        if reason := guard.get("reason"):
            decision += f": {reason}"
        return [*lines, decision]

    def stage_rows(self) -> list[dict[str, Any]]:
        """Time, peak memory and swap per stage, in the order the stages ran."""
        rec = self.record
        stages = list(dict.fromkeys([*rec.timings_s, *rec.peak_mem_gb, *rec.swap_gb]))
        return [
            {
                "Stage": STAGE_LABELS.get(stage, stage.capitalize()),
                "Seconds": rec.timings_s.get(stage),
                "Peak GB": rec.peak_mem_gb.get(stage),
                "Swap GB": rec.swap_gb.get(stage),
            }
            for stage in stages
        ]


def selected_run(rows: list[int], ids: list[str]) -> str | None:
    """The run id of the table's selected row, by the ids that table was drawn with: the row
    index alone can point at another run once a newer one is listed above it."""
    return ids[rows[0]] if rows and rows[0] < len(ids) else None


def load(runs: Path | None = None) -> list[RunEntry]:
    """Every run, newest first, with dead `running` records repaired as `interrupted`."""
    held = lock.is_held()
    entries = []
    for run, rec in storage.run_records(runs):
        if rec.status == "running":
            # Unreadable or unwritable now: show the record as listed.
            with contextlib.suppress(OSError, ValueError, TypeError):
                rec = record.repair_if_interrupted(run, lock_held=held)
        entries.append(RunEntry(run, rec))
    return entries
