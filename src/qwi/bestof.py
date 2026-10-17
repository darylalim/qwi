"""Best-of-N review sessions for the M3 calibration: `qwi eval session --bo4` (PLAN.md §5.4).

One item per prompt: its N candidates (one per seed) in a shuffled order, shown as candidates
1..N. The reviewer marks each candidate acceptable or not and picks one best; no reranker score
or seed is shown, so the human's picks stay independent of the scores they're compared with.
Item order and each item's candidate order come from a stored seed. `review.csv` has one row per
candidate (`acceptable`) plus one per item (`best`, `notes`); the candidate → seed mapping lives
in `manifest.json`.

`calibrate()` (`qwi eval calibrate`) compares the reranker's top pick per prompt with the human's,
for the run's instruction and every `rerank_<label>.jsonl` re-score, and checks whether the best
candidate's prob separates prompts with no acceptable candidate (the `--min-score` choice).
"""

import csv
import json
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from qwi import paths
from qwi.config import DEFAULTS
from qwi.errors import SessionError
from qwi.evaluation import RERANK_PRESETS, read_rows
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


# --- Calibration (`qwi eval calibrate`, §5.4 "Best-of-4 calibration" steps 3-4) -------------

RUN_LABEL = "run"  # the run's own scores, when its instruction matches no preset


@dataclass(frozen=True)
class Pick:
    """One prompt under one instruction: the reranker's top candidate and the human's labels."""

    prompt_id: str
    top: int  # candidate number the reranker ranks first
    best: int  # the human's best
    acceptable: frozenset[int]
    top_prob: float

    @property
    def agrees(self) -> bool:
        return self.top == self.best

    @property
    def top_acceptable(self) -> bool:
        return self.top in self.acceptable


@dataclass
class InstructionResult:
    label: str
    instruction: str
    picks: list[Pick]

    @property
    def agreement(self) -> int:
        return sum(p.agrees for p in self.picks)

    @property
    def top_acceptable(self) -> int:
        return sum(p.top_acceptable for p in self.picks if p.acceptable)


@dataclass
class Calibration:
    session: str
    results: str
    candidates: int
    items: int
    with_acceptable: int  # prompts with at least one acceptable candidate
    instructions: list[InstructionResult]
    base_seed_agreement: int  # "no reranker": always take the base (lowest) seed
    base_seed_acceptable: int
    chance_agreement: float  # a random pick, in expectation
    chance_acceptable: float
    winner: str
    min_score: float
    min_score_reason: str


def _scores(path: Path, rows: list[dict[str, Any]]) -> tuple[str, dict[tuple[str, int], tuple]]:
    """(instruction, {(prompt id, seed): (margin, prob)}) from a results or re-score file."""
    found = {
        (r["id"], int(r["seed"])): (r.get("rerank_margin"), r.get("rerank_prob")) for r in rows
    }
    if any(m is None or pr is None for m, pr in found.values()):
        raise SessionError(f"{path.name} has unscored rows: run `qwi eval rescore` first")
    instructions = {r.get("rerank_instruction") or r.get("instruction") for r in rows}
    if len(instructions) != 1:
        raise SessionError(f"{path.name}: expected one reranker instruction, got {instructions}")
    return str(instructions.pop()), found


def _label(instruction: str) -> str:
    return next((k for k, v in RERANK_PRESETS.items() if v == instruction), RUN_LABEL)


def calibrate(folder: Path) -> Calibration:
    """Score a fully judged best-of-N session against each instruction's top picks."""
    manifest = bestof_manifest(folder)
    items = [BestOfItem.from_dict(i) for i in manifest["items"]]
    judgments = read_judgments(folder)
    missing = [i.item for i in items if judgments[i.item].problem(i) is not None]
    if missing:
        raise SessionError(
            f"{len(missing)} prompts are not judged yet (e.g. {', '.join(missing[:5])}); "
            "calibrate needs every prompt marked"
        )
    results = results_dir(manifest["results"])
    sources = [(None, results / "results.jsonl")]
    sources += [(f.stem.removeprefix("rerank_"), f) for f in sorted(results.glob("rerank_*.jsonl"))]
    instructions: list[InstructionResult] = []
    for label, path in sources:
        instruction, scores = _scores(path, read_rows(path))
        picks = []
        for item in items:
            judged = judgments[item.item]
            # Highest margin; on a tie the lowest seed, as the pipeline's lower index wins.
            top_seed = max(item.seeds, key=lambda s: (scores[(item.prompt_id, s)][0], -s))
            assert judged.best is not None  # complete judgments have one
            picks.append(
                Pick(
                    item.prompt_id,
                    item.seeds.index(top_seed) + 1,
                    judged.best,
                    frozenset(n for n in item.candidates if judged.acceptable[n] == "yes"),
                    float(scores[(item.prompt_id, top_seed)][1]),
                )
            )
        instructions.append(InstructionResult(label or _label(instruction), instruction, picks))

    base = [
        (
            item.seeds.index(min(item.seeds)) + 1,
            judgments[item.item],
        )
        for item in items
    ]
    n = len(items[0].seeds)
    with_acceptable = [j for _, j in base if "yes" in j.acceptable.values()]
    accept_share = [sum(v == "yes" for v in j.acceptable.values()) / n for j in with_acceptable]
    default = _label(DEFAULTS.rerank_instruction)
    # Most agreement, then most acceptable top picks; a full tie keeps the current default.
    ranked = sorted(
        instructions,
        key=lambda r: (r.agreement, r.top_acceptable, r.label == default),
        reverse=True,
    )
    min_score, why = _min_score(ranked[0].picks)
    return Calibration(
        folder.name,
        manifest["results"],
        n,
        len(items),
        len(with_acceptable),
        instructions,
        sum(top == j.best for top, j in base),
        sum(j.acceptable[top] == "yes" for top, j in base if "yes" in j.acceptable.values()),
        len(items) / n,
        sum(accept_share),
        ranked[0].label,
        min_score,
        why,
    )


def _min_score(picks: list[Pick]) -> tuple[float, str]:
    """A threshold on the top candidate's prob that splits prompts with no acceptable candidate
    from the rest, if one exists; otherwise keep the default and call the warning advisory."""
    keep = DEFAULTS.rerank_min_score
    none_ok = [p.top_prob for p in picks if not p.acceptable]
    some_ok = [p.top_prob for p in picks if p.acceptable]
    if not none_ok:
        return keep, (
            f"every prompt had an acceptable candidate, so nothing to separate: keep {keep}; "
            "the warning is advisory"
        )
    if not some_ok:
        return keep, f"no prompt had an acceptable candidate: keep {keep}; the warning is advisory"
    low, high = max(none_ok), min(some_ok)
    if low < high:
        threshold = round((low + high) / 2, 2)
        return threshold, (
            f"prompts with no acceptable candidate topped out at prob {low:.2f}, the others "
            f"started at {high:.2f}: the midpoint {threshold:.2f} separates them"
        )
    return keep, (
        f"overlap: a prompt with no acceptable candidate reached prob {low:.2f}, one with an "
        f"acceptable candidate went as low as {high:.2f}; keep {keep}, the warning is advisory"
    )


def render_calibration(c: Calibration) -> str:
    def rate(k: int | float, n: int) -> str:
        return f"{k:g}/{n} ({100 * k / n:.0f}%)" if n else "n/a"

    lines = [
        f"# Calibration `{c.session}`",
        "",
        f"Results `{c.results}`: {c.items} prompts x {c.candidates} candidates; "
        f"{c.with_acceptable} prompts have at least one acceptable candidate.",
        "Top-1 = the candidate the reranker ranks first (ties go to the lowest seed). "
        "Top-1 acceptable counts only prompts with an acceptable candidate (PLAN.md §5.4).",
        "",
        f"## Instruction: **{c.winner}**",
        "",
        "| Instruction | Top-1 = human best | Top-1 acceptable |",
        "|---|---|---|",
        *[
            f"| `{r.label}` | {rate(r.agreement, c.items)} | "
            f"{rate(r.top_acceptable, c.with_acceptable)} |"
            for r in c.instructions
        ],
        f"| no reranker (base seed) | {rate(c.base_seed_agreement, c.items)} | "
        f"{rate(c.base_seed_acceptable, c.with_acceptable)} |",
        f"| chance (random pick) | {rate(round(c.chance_agreement, 1), c.items)} | "
        f"{rate(round(c.chance_acceptable, 1), c.with_acceptable)} |",
        "",
        *[f"- `{r.label}`: {r.instruction}" for r in c.instructions],
        "",
        f"## `--min-score`: **{c.min_score:g}**",
        "",
        c.min_score_reason[0].upper() + c.min_score_reason[1:] + ".",
        "",
        f"## Per prompt (`{c.winner}`)",
        "",
        "| Prompt | Top-1 | Human best | Acceptable | Top-1 prob |",
        "|---|---|---|---|---|",
    ]
    winner = next(r for r in c.instructions if r.label == c.winner)
    for p in sorted(winner.picks, key=lambda p: p.prompt_id):
        accepted = ", ".join(map(str, sorted(p.acceptable))) or "none"
        lines.append(f"| `{p.prompt_id}` | {p.top} | {p.best} | {accepted} | {p.top_prob:.2f} |")
    lines.append("")
    return "\n".join(lines)


def write_calibration(folder: Path) -> tuple[Calibration, Path]:
    report = calibrate(folder)
    path = folder / "report.md"
    path.write_text(render_calibration(report), encoding="utf-8")
    return report, path
