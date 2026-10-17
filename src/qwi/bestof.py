"""Best-of-N review sessions for the M3 calibration: `qwi eval session --bo4` (PLAN.md §5.4).

One item per prompt: its N candidates (one per seed) in a shuffled order, shown as candidates
1..N. The reviewer marks each candidate acceptable or not and picks one best; no reranker score
or seed is shown, so the human's picks stay independent of the scores they're compared with.
Item order and each item's candidate order come from a stored seed. `review.csv` has one row per
candidate (`acceptable`) plus one per item (`best`, `notes`); the candidate → seed mapping lives
in `manifest.json`. `qwi eval calibrate` (the next step) reads both.
"""

import csv
import json
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from qwi import paths
from qwi.errors import SessionError
from qwi.sessions import BESTOF_KIND, load_manifest, load_results, results_dir, stored

REVIEW_COLUMNS = ["item", "candidate", "acceptable", "best", "notes"]
ACCEPTABLE = ("yes", "no")
MIN_CANDIDATES = 2


@dataclass(frozen=True)
class BestOfItem:
    item: str
    prompt_id: str
    seeds: tuple[int, ...]  # in the order shown: candidate n is seeds[n - 1]

    @property
    def candidates(self) -> range:
        return range(1, len(self.seeds) + 1)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BestOfItem":
        return cls(str(d["item"]), str(d["prompt_id"]), tuple(int(s) for s in d["seeds"]))

    def to_dict(self) -> dict[str, Any]:
        return {"item": self.item, "prompt_id": self.prompt_id, "seeds": list(self.seeds)}


@dataclass(frozen=True)
class BestOfJudgment:
    acceptable: dict[int, str]  # candidate → "yes" | "no" | "" (not judged yet)
    best: int | None = None
    notes: str = ""

    def problem(self, item: BestOfItem) -> str | None:
        """Why this judgment can't be saved yet, or None when it's complete."""
        if any(self.acceptable.get(n) not in ACCEPTABLE for n in item.candidates):
            return "Mark every candidate acceptable or not."
        if self.best not in item.candidates:
            return "Pick the best candidate."
        if any(self.acceptable[n] == "yes" for n in item.candidates) and (
            self.acceptable[self.best] != "yes"
        ):
            return "The best candidate must be an acceptable one, when any is."
        return None


def create(results: Path, *, seed: int | None = None, day: date | None = None) -> Path:
    """Write `eval/sessions/<date>_<variant>_bo4/`: `manifest.json` and an empty `review.csv`."""
    name, rows = load_results([results])
    seeds: dict[str, list[int]] = {}
    for prompt_id, s in sorted(rows):
        seeds.setdefault(prompt_id, []).append(s)
    counts = {len(v) for v in seeds.values()}
    if len(counts) != 1 or min(counts) < MIN_CANDIDATES:
        found = ", ".join(f"{k}: {len(v)}" for k, v in seeds.items())
        raise SessionError(
            f"best-of-N needs the same number (>= {MIN_CANDIDATES}) of seeds for every prompt; "
            f"got {found}"
        )
    seed = random.SystemRandom().randrange(2**31) if seed is None else seed
    rng = random.Random(seed)
    prompt_ids = sorted(seeds)
    rng.shuffle(prompt_ids)
    items = []
    for n, prompt_id in enumerate(prompt_ids, 1):
        order = list(seeds[prompt_id])
        rng.shuffle(order)
        items.append(BestOfItem(f"i{n:03d}", prompt_id, tuple(order)))

    day = day or date.today()
    suffix = "" if name.endswith(BESTOF_KIND) else f"_{BESTOF_KIND}"  # no `bo4_bo4`
    folder = paths.EVAL_SET / "sessions" / f"{day.isoformat()}_{name}{suffix}"
    if folder.exists():
        raise SessionError(f"{folder} exists")
    folder.mkdir(parents=True)
    manifest = {
        "version": 1,
        "kind": BESTOF_KIND,
        "created": day.isoformat(),
        "shuffle_seed": seed,
        "variant": name,
        "results": stored(results),
        "items": [i.to_dict() for i in items],
    }
    (folder / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write(folder / "review.csv", [row for i in items for row in _blank_rows(i)])
    return folder


def _blank_rows(item: BestOfItem) -> list[dict[str, str]]:
    rows = [{"item": item.item, "candidate": "", "acceptable": "", "best": "", "notes": ""}]
    rows += [
        {"item": item.item, "candidate": str(n), "acceptable": "", "best": "", "notes": ""}
        for n in item.candidates
    ]
    return rows


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    """Atomic: a crash mid-save never leaves half a `review.csv`."""
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, REVIEW_COLUMNS, lineterminator="\n")  # LF, like the repo
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _read(folder: Path) -> list[dict[str, str]]:
    with (folder / "review.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bestof_manifest(folder: Path) -> dict[str, Any]:
    manifest = load_manifest(folder)
    if manifest.get("kind") != BESTOF_KIND:
        raise SessionError(f"{folder.name} is not a best-of-N session")
    return manifest


@dataclass
class BestOfSession:
    folder: Path
    items: list[BestOfItem]
    prompts: dict[str, str]  # prompt id → the eval prompt
    images: dict[tuple[str, int], Path]  # (item id, candidate) → image

    @property
    def name(self) -> str:
        return self.folder.name


def open_session(folder: Path) -> BestOfSession:
    """Items in their shuffled order with each candidate's image. Seeds and scores stay out."""
    manifest = bestof_manifest(folder)
    items = [BestOfItem.from_dict(i) for i in manifest["items"]]
    results = manifest["results"]
    images_dir = paths.eval_dir() / Path(results).name  # §5.7: images live under $QWI_HOME
    _, rows = load_results([results_dir(results)])
    prompts: dict[str, str] = {}
    images: dict[tuple[str, int], Path] = {}
    for item in items:
        for n, s in zip(item.candidates, item.seeds, strict=True):
            row = rows[(item.prompt_id, s)]
            prompts[item.prompt_id] = row["prompt"]
            images[(item.item, n)] = images_dir / row["image"]
    return BestOfSession(folder, items, prompts, images)


def read_judgments(folder: Path) -> dict[str, BestOfJudgment]:
    acceptable: dict[str, dict[int, str]] = {}
    extra: dict[str, tuple[int | None, str]] = {}
    for row in _read(folder):
        if row["candidate"]:
            acceptable.setdefault(row["item"], {})[int(row["candidate"])] = row[
                "acceptable"
            ].lower()
        else:
            extra[row["item"]] = (int(row["best"]) if row["best"] else None, row["notes"])
    ids = acceptable.keys() | extra.keys()
    return {i: BestOfJudgment(acceptable.get(i, {}), *extra.get(i, (None, ""))) for i in ids}


def save_judgment(folder: Path, item: BestOfItem, judgment: BestOfJudgment) -> None:
    """Rewrite `review.csv` with this item's rows replaced; only a complete judgment saves."""
    problem = judgment.problem(item)
    if problem:
        raise SessionError(f"{item.item}: {problem}")
    rows = _read(folder)
    if not any(r["item"] == item.item for r in rows):
        raise SessionError(f"{item.item} is not in {folder.name}/review.csv")
    for row in rows:
        if row["item"] != item.item:
            continue
        if row["candidate"]:
            row["acceptable"] = judgment.acceptable[int(row["candidate"])]
        else:
            row["best"], row["notes"] = str(judgment.best), judgment.notes
    _write(folder / "review.csv", rows)
