"""What the Eval review page can open (PLAN.md §5.6), without Streamlit, so plain pytest covers it.

A source is a session in `eval/sessions/` (paired or best-of-N, by its manifest's `kind`) or a
results folder in `eval/results/` with a `review.csv` (single-variant mode). Its `key`
(`session/<name>`, `results/<name>`) is both the picker's option and the URL value (a bound
selectbox puts the shown text in the URL); the prefix matters because a session and a results
folder can share a name (`2026-10-17_bo4` is both).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from qwi import paths, sessions

Kind = Literal["paired", "bestof", "variant"]
LABELS: dict[Kind, str] = {"paired": "paired", "bestof": "best-of-N", "variant": "single variant"}


@dataclass(frozen=True)
class Source:
    kind: Kind
    folder: Path

    @property
    def key(self) -> str:
        return f"{'results' if self.kind == 'variant' else 'session'}/{self.folder.name}"

    @property
    def label(self) -> str:
        return LABELS[self.kind]


def session_source(folder: Path) -> Source:
    """A session folder as a source; the manifest's `kind` picks the mode. Unreadable → paired,
    so opening it shows the paired page's error rather than hiding the session."""
    try:
        kind = json.loads((folder / "manifest.json").read_text(encoding="utf-8")).get("kind")
    except (OSError, ValueError):
        kind = None
    return Source("bestof" if kind == sessions.BESTOF_KIND else "paired", folder)


def sources(extra: Path | None = None) -> list[Source]:
    """Sessions, then results folders, each newest first (names start with the date).

    `extra` is the session `qwi eval review` was given, listed even when it isn't in
    `eval/sessions/`."""
    session_root = paths.EVAL_SET / "sessions"
    found = [
        session_source(d)
        for d in sorted(session_root.glob("*/"), reverse=True)
        if (d / "manifest.json").is_file()
    ]
    if extra is not None and extra.resolve() not in {s.folder.resolve() for s in found}:
        found.insert(0, session_source(extra))
    found += [
        Source("variant", d)
        for d in sorted(paths.eval_results_dir().glob("*/"), reverse=True)
        if (d / "review.csv").is_file()
    ]
    return found
