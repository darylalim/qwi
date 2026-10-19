"""Single-variant review: one results folder's `review.csv`, one image at a time (PLAN.md §5.6).

`qwi eval` writes the sheet (`evaluation.write_review`): one row per (image, check), `result`
blank. The Eval review page fills it in with the automated hints in view, since nothing here is
blind: it's a look at one variant, not a comparison. Notes are per image and live on the image's
first row. All file logic is here; the page only renders it.
"""

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwi import paths
from qwi.errors import SessionError
from qwi.evaluation import REVIEW_COLUMNS
from qwi.sessions import HINT_KEYS, RESULTS, load_results


@dataclass(frozen=True)
class VariantItem:
    item: str  # `<prompt id>_<seed>`, as `write_review` names it
    prompt_id: str
    seed: int
    checks: tuple[str, ...]
    image: Path
    prompt: str
    hints: dict[str, Any]


@dataclass(frozen=True)
class VariantJudgment:
    results: dict[str, str]  # check → "pass" | "fail" | "" (not judged yet)
    notes: str = ""

    def complete(self, checks: tuple[str, ...]) -> bool:
        return all(self.results.get(c) in RESULTS for c in checks)


@dataclass
class VariantReview:
    folder: Path
    items: list[VariantItem]

    @property
    def name(self) -> str:
        return self.folder.name


def _read(folder: Path) -> list[dict[str, str]]:
    path = folder / "review.csv"
    if not path.is_file():
        raise SessionError(f"{folder.name} has no review.csv")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def open_results(folder: Path) -> VariantReview:
    """Items in `review.csv` order (prompt, then seed), each with its image and hints."""
    _, rows = load_results([folder])
    images = paths.eval_dir() / folder.name  # §5.7: images live under $QWI_HOME
    checks: dict[str, list[str]] = defaultdict(list)
    keys: dict[str, tuple[str, int]] = {}
    for row in _read(folder):
        checks[row["item"]].append(row["check"])
        keys[row["item"]] = (row["prompt_id"], int(row["seed"]))
    items = []
    for item, key in keys.items():
        if key not in rows:
            raise SessionError(f"{folder.name}: {item} is in review.csv but not results.jsonl")
        row = rows[key]
        hints = {k: row[k] for k in HINT_KEYS if row.get(k) is not None}
        hints["seconds"] = round(sum(row.get("seconds", {}).values()), 1)
        items.append(
            VariantItem(
                item,
                key[0],
                key[1],
                tuple(checks[item]),
                images / row["image"],
                row["prompt"],
                hints,
            )
        )
    return VariantReview(folder, items)


def read_judgments(folder: Path) -> dict[str, VariantJudgment]:
    results: dict[str, dict[str, str]] = defaultdict(dict)
    notes: dict[str, str] = {}
    for row in _read(folder):
        results[row["item"]][row["check"]] = row["result"].lower()
        if row["notes"] and row["item"] not in notes:
            notes[row["item"]] = row["notes"]
    return {i: VariantJudgment(r, notes.get(i, "")) for i, r in results.items()}


def save_judgment(folder: Path, item: VariantItem, judgment: VariantJudgment) -> None:
    """Rewrite `review.csv` with this image's rows replaced; atomic, so a crash loses nothing."""
    if set(judgment.results.values()) - {*RESULTS, ""}:
        raise SessionError(f"{item.item}: results must be pass/fail")
    rows = _read(folder)
    mine = [r for r in rows if r["item"] == item.item]
    if not mine:
        raise SessionError(f"{item.item} is not in {folder.name}/review.csv")
    for n, row in enumerate(mine):
        row["result"] = judgment.results.get(row["check"], "")
        row["notes"] = judgment.notes if n == 0 else ""
    target = folder / "review.csv"
    tmp = target.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, REVIEW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(target)
