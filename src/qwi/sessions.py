"""Blind paired review sessions: `qwi eval session` and `qwi eval compare` (PLAN.md §5.4).

A session pairs two variants' images of the same (prompt, seed). Item ids are anonymous, item
order is shuffled and left/right sides are randomized per item, all from a stored seed. Reviewers
mark each check pass/fail per side in `review.csv` (the review app does this in M2); `compare`
refuses to reveal the variants until every check is judged, then scores (prompt, seed, check)
triples as win (only B passes), loss (only A passes) or tie, and applies the decision rule.
"""

import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from qwi import paths
from qwi.errors import SessionError, SessionIncomplete
from qwi.evaluation import load_prompts

REVIEW_COLUMNS = ["item", "check", "left", "right", "prefer", "notes"]
RESULTS = ("pass", "fail")
PREFER = ("left", "right", "same")
QUALITY_SLACK = 2  # rule 1: losses - wins <= max(2, noise floor)
BETTER_MARGIN = 3  # "B is better": wins - losses >= 3 (or a preference majority)
INCONCLUSIVE_PAIRS = 8  # a rule failure on >= 8 discordant pairs ...
INCONCLUSIVE_P = 0.05  # ... that the sign test can't tell from chance: extend the seeds
TIME_RATIO = 1.15
PEAK_LIMIT_GB = 24.0
EXTEND_SEEDS = (1002, 1003)


def results_dir(arg: str) -> Path:
    """A results folder given as a path or as a name under `eval/results/`."""
    path = Path(arg)
    if not (path / "results.jsonl").is_file():
        path = paths.eval_results_dir() / arg
    if not (path / "results.jsonl").is_file():
        raise SessionError(f"no results.jsonl for {arg!r} (a path, or a name in eval/results/)")
    return path


def load_results(dirs: list[Path]) -> tuple[str, dict[tuple[str, int], dict[str, Any]]]:
    """One variant's rows keyed by (prompt id, seed), across one or more results folders."""
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    variants: set[str] = set()
    for d in dirs:
        for line in (d / "results.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[(row["id"], int(row["seed"]))] = row
                variants.add(row["variant"])
    if len(variants) != 1:
        raise SessionError(f"one variant per side, got {sorted(variants)}")
    return variants.pop(), rows


@dataclass(frozen=True)
class Item:
    item: str
    prompt_id: str
    seed: int
    tags: tuple[str, ...]
    checks: tuple[str, ...]
    left: str  # "a" | "b"; the other side is the other variant

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Item":
        return cls(
            str(d["item"]),
            str(d["prompt_id"]),
            int(d["seed"]),
            tuple(d["tags"]),
            tuple(d["checks"]),
            str(d["left"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "tags": list(self.tags), "checks": list(self.checks)}

    def variant_of(self, side: str) -> str:
        return self.left if side == "left" else ("b" if self.left == "a" else "a")


def _items(keys: list[tuple[str, int]], rows: dict, rng: random.Random, start: int) -> list[Item]:
    checks = {p.id: p.checks for p in load_prompts()}
    keys = sorted(keys)
    rng.shuffle(keys)
    items = []
    for n, (prompt_id, seed) in enumerate(keys, start):
        tags = tuple(rows[(prompt_id, seed)]["tags"])
        items.append(
            Item(f"i{n:03d}", prompt_id, seed, tags, checks[prompt_id], rng.choice(("a", "b")))
        )
    return items


def _write_review(path: Path, items: list[Item], judged: dict[tuple[str, str], list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")  # LF, like the rest of the repo
        writer.writerow(REVIEW_COLUMNS)
        for item in items:
            # The item's own row (empty check) holds "prefer" and "notes".
            writer.writerow(judged.get((item.item, ""), [item.item, "", "", "", "", ""]))
            for check in item.checks:
                writer.writerow(judged.get((item.item, check), [item.item, check, "", "", "", ""]))


def create(
    a: list[Path],
    b: list[Path],
    *,
    seed: int | None = None,
    day: date | None = None,
    extend: Path | None = None,
) -> Path:
    """Write `eval/sessions/<date>_<a>_vs_<b>/` with `manifest.json` and an empty `review.csv`.

    With `extend`, the old session's items and judgments are kept and only the new pairs (for
    example seeds 1002/1003) are added for review.
    """
    name_a, rows_a = load_results(a)
    name_b, rows_b = load_results(b)
    if name_a == name_b:
        raise SessionError(f"both sides are {name_a!r}: compare two different variants")
    old_items: list[Item] = []
    judged: dict[tuple[str, str], list[str]] = {}
    if extend is not None:
        old = _load_manifest(extend)
        if (old["a"]["name"], old["b"]["name"]) != (name_a, name_b):
            raise SessionError("--extend needs the same two variants, on the same sides")
        old_items = [Item.from_dict(i) for i in old["items"]]
        for row in _read_review(extend):
            judged[(row["item"], row["check"])] = [row[c] for c in REVIEW_COLUMNS]
        a = [Path(p) for p in old["a"]["results"]] + a
        b = [Path(p) for p in old["b"]["results"]] + b
    seen = {(i.prompt_id, i.seed) for i in old_items}
    new_keys = [k for k in rows_a.keys() & rows_b.keys() if k not in seen]
    if not new_keys:
        raise SessionError("no (prompt, seed) pairs in both variants' results")
    seed = random.SystemRandom().randrange(2**31) if seed is None else seed
    items = old_items + _items(new_keys, rows_a | rows_b, random.Random(seed), len(old_items) + 1)

    day = day or date.today()
    folder = paths.EVAL_SET / "sessions" / f"{day.isoformat()}_{name_a}_vs_{name_b}"
    if extend is not None:
        folder = folder.with_name(f"{extend.name}_ext")
    if folder.exists():
        raise SessionError(f"{folder} exists")
    folder.mkdir(parents=True)
    manifest = {
        "version": 1,
        "created": day.isoformat(),
        "shuffle_seed": seed,
        "extends": extend.name if extend else None,
        "a": {"name": name_a, "results": [str(p) for p in a]},
        "b": {"name": name_b, "results": [str(p) for p in b]},
        "items": [i.to_dict() for i in items],
    }
    (folder / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_review(folder / "review.csv", items, judged)
    return folder


def _load_manifest(session: Path) -> dict[str, Any]:
    path = session / "manifest.json"
    if not path.is_file():
        raise SessionError(f"no manifest.json in {session}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_review(session: Path) -> list[dict[str, str]]:
    with (session / "review.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided sign test p-value on the discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


@dataclass
class Tally:
    wins: int = 0
    losses: int = 0
    ties: int = 0

    def add(self, a_pass: bool, b_pass: bool) -> None:
        if b_pass and not a_pass:
            self.wins += 1
        elif a_pass and not b_pass:
            self.losses += 1
        else:
            self.ties += 1

    @property
    def discordant(self) -> int:
        return self.wins + self.losses


@dataclass
class Resources:
    median_s: float
    peak_gb: float
    swapped_images: int


def _resources(rows: list[dict[str, Any]]) -> Resources:
    seconds = [sum(r["seconds"].values()) for r in rows]
    peaks = [max(r["peak_gb"].values(), default=0.0) for r in rows]
    return Resources(
        statistics.median(seconds) if seconds else 0.0,
        max(peaks, default=0.0),
        sum(1 for r in rows if r.get("swap_gb", 0) > 0),
    )


@dataclass
class Report:
    session: str
    a: str
    b: str
    items: int
    overall: Tally
    per_tag: dict[str, Tally]
    per_check: dict[tuple[str, str], Tally]
    text: Tally
    prefer: Counter[str]
    res_a: Resources
    res_b: Resources
    noise_floor: int
    rules: dict[str, bool] = field(default_factory=dict)
    verdict: str = ""
    reason: str = ""


def compare(session: Path, noise_floor: int = 0) -> Report:
    """Score a fully judged session and apply the §5.4 decision rule."""
    manifest = _load_manifest(session)
    items = {i["item"]: Item.from_dict(i) for i in manifest["items"]}
    rows = _read_review(session)
    missing = [
        f"{r['item']}/{r['check']}"
        for r in rows
        if r["check"] and (r["left"].lower() not in RESULTS or r["right"].lower() not in RESULTS)
    ]
    if missing:
        raise SessionIncomplete(len(missing), missing[:5])
    bad_prefer = [
        r["item"]
        for r in rows
        if not r["check"] and r["prefer"] and r["prefer"].lower() not in PREFER
    ]
    if bad_prefer:
        raise SessionError(f"prefer must be left/right/same or empty: {', '.join(bad_prefer[:5])}")

    overall, text = Tally(), Tally()
    per_tag: dict[str, Tally] = defaultdict(Tally)
    per_check: dict[tuple[str, str], Tally] = defaultdict(Tally)
    prefer: Counter[str] = Counter()
    for r in rows:
        item = items[r["item"]]
        if not r["check"]:
            if r["prefer"]:
                side = r["prefer"].lower()
                prefer[side if side == "same" else item.variant_of(side)] += 1
            continue
        result = {
            item.variant_of("left"): r["left"].lower(),
            item.variant_of("right"): r["right"].lower(),
        }
        a_pass, b_pass = result["a"] == "pass", result["b"] == "pass"
        overall.add(a_pass, b_pass)
        per_check[(item.prompt_id, r["check"])].add(a_pass, b_pass)
        for tag in item.tags:
            per_tag[tag].add(a_pass, b_pass)
        if item.prompt_id.startswith("text-"):
            text.add(a_pass, b_pass)

    keys = {(i.prompt_id, i.seed) for i in items.values()}
    _, rows_a = load_results([Path(p) for p in manifest["a"]["results"]])
    _, rows_b = load_results([Path(p) for p in manifest["b"]["results"]])
    report = Report(
        session.name,
        manifest["a"]["name"],
        manifest["b"]["name"],
        len(items),
        overall,
        dict(per_tag),
        dict(per_check),
        text,
        prefer,
        _resources([rows_a[k] for k in keys if k in rows_a]),
        _resources([rows_b[k] for k in keys if k in rows_b]),
        noise_floor,
    )
    _decide(report)
    return report


def _decide(report: Report) -> None:
    o, ra, rb = report.overall, report.res_a, report.res_b
    report.rules = {
        "quality": o.losses - o.wins <= max(QUALITY_SLACK, report.noise_floor),
        "text": report.text.losses <= report.text.wins,
        "time": rb.median_s <= TIME_RATIO * ra.median_s,
        "peak": rb.peak_gb <= PEAK_LIMIT_GB,
        "swap": rb.swapped_images == 0,
    }
    voted = report.prefer["a"] + report.prefer["b"] + report.prefer["same"]
    majority = voted > 0 and report.prefer["b"] > voted / 2
    p = sign_test(o.wins, o.losses)
    if not all(report.rules[r] for r in ("time", "peak", "swap")):
        failed = [r for r in ("time", "peak", "swap") if not report.rules[r]]
        report.verdict, report.reason = "worse", f"B fails the resource rule ({', '.join(failed)})"
    elif report.rules["quality"] and report.rules["text"]:
        if o.wins - o.losses >= BETTER_MARGIN or majority:
            why = (
                f"wins - losses = {o.wins - o.losses}"
                if o.wins - o.losses >= BETTER_MARGIN
                else "preference majority"
            )
            report.verdict, report.reason = "better", f"not worse on every rule, and {why}"
        else:
            report.verdict, report.reason = "not worse", "passes rules 1-3"
    elif o.discordant >= INCONCLUSIVE_PAIRS and p >= INCONCLUSIVE_P:
        report.verdict = "inconclusive"
        seeds = ",".join(map(str, EXTEND_SEEDS))
        report.reason = (
            f"a quality rule fails on {o.discordant} discordant pairs, but the sign test "
            f"can't separate it from chance (p = {p:.3f}): extend both variants to seeds {seeds} "
            f"(`qwi eval --variant <name>-ext --seeds {seeds}`), then "
            f"`qwi eval session --extend {report.session} --a <A-ext> --b <B-ext>`"
        )
    else:
        failed = [r for r in ("quality", "text") if not report.rules[r]]
        report.verdict, report.reason = "worse", f"B fails rule {' and '.join(failed)}"


def _row(name: str, t: Tally) -> str:
    return f"| {name} | {t.wins} | {t.losses} | {t.ties} | {sign_test(t.wins, t.losses):.3f} |"


def _rule_rows(report: Report) -> list[tuple[str, str, bool]]:
    o, ra, rb, r = report.overall, report.res_a, report.res_b, report.rules
    ratio = rb.median_s / ra.median_s if ra.median_s else float("inf")
    return [
        (
            f"1 quality: losses - wins <= max({QUALITY_SLACK}, noise floor)",
            str(o.losses - o.wins),
            r["quality"],
        ),
        (
            "2 text: losses <= wins on `text-*`",
            f"{report.text.losses} vs {report.text.wins}",
            r["text"],
        ),
        (
            f"3a median time per image <= {TIME_RATIO}x A",
            f"{rb.median_s:.1f} s vs {ra.median_s:.1f} s ({ratio:.2f}x)",
            r["time"],
        ),
        (
            f"3b peak memory <= {PEAK_LIMIT_GB:g} GB",
            f"{rb.peak_gb:.1f} GB (A {ra.peak_gb:.1f})",
            r["peak"],
        ),
        (
            "3c zero swap on every image",
            f"{rb.swapped_images} swapped (A {ra.swapped_images})",
            r["swap"],
        ),
    ]


def render(report: Report) -> str:
    o = report.overall
    mark = {True: "pass", False: "**FAIL**"}
    lines = [
        f"# Session `{report.session}`",
        "",
        f"A = `{report.a}`, B = `{report.b}`; {report.items} items; "
        f"noise floor {report.noise_floor}.",
        "Win = only B passes a check; loss = only A passes; tie = same result (PLAN.md §5.4).",
        "",
        f"## Verdict: B is **{report.verdict}**",
        "",
        report.reason + ".",
        "",
        "| Rule | Measured | Result |",
        "|---|---|---|",
        *[f"| {rule} | {measured} | {mark[ok]} |" for rule, measured, ok in _rule_rows(report)],
        "",
        "## Overall",
        "",
        "| | Wins | Losses | Ties | Sign test p |",
        "|---|---|---|---|---|",
        _row("all checks", o),
        "",
        "## Per tag",
        "",
        "| Tag | Wins | Losses | Ties | Sign test p |",
        "|---|---|---|---|---|",
        *[_row(tag, t) for tag, t in sorted(report.per_tag.items())],
        "",
        "## Per check",
        "",
        "| Check | Wins | Losses | Ties | Sign test p |",
        "|---|---|---|---|---|",
        *[_row(f"`{pid}`: {check}", t) for (pid, check), t in sorted(report.per_check.items())],
        "",
        "## Preference",
        "",
        f"B {report.prefer['b']}, A {report.prefer['a']}, same {report.prefer['same']}.",
        "",
    ]
    return "\n".join(lines)


def write_report(session: Path, noise_floor: int = 0) -> tuple[Report, Path]:
    report = compare(session, noise_floor)
    path = session / "report.md"
    path.write_text(render(report), encoding="utf-8")
    return report, path


# --- The review app's data layer (`qwi eval review`, §5.4 "Eval review page, session mode") -

SIDES = ("left", "right")
REVIEW_SESSION_ENV = "QWI_REVIEW_SESSION"  # how `qwi eval review` tells the app which session
HINT_KEYS = ("rewrite", "ocr", "ocr_exact", "rerank_prob", "white_border_share")


@dataclass(frozen=True)
class Judgment:
    """One item's review: pass/fail per check and side, plus the optional preference and notes."""

    left: dict[str, str]  # check → "pass" | "fail" | "" (not judged yet)
    right: dict[str, str]
    prefer: str = ""  # "left" | "right" | "same" | ""
    notes: str = ""

    def complete(self, checks: tuple[str, ...]) -> bool:
        return all(self.left.get(c) in RESULTS and self.right.get(c) in RESULTS for c in checks)


@dataclass(frozen=True)
class Side:
    """What the app shows for one side of an item. `hints` stay hidden until it's saved."""

    image: Path
    prompt: str
    hints: dict[str, Any]


@dataclass
class ReviewSession:
    folder: Path
    items: list[Item]
    sides: dict[tuple[str, str], Side]  # (item id, "left" | "right") → what to show

    @property
    def name(self) -> str:
        return self.folder.name


def open_session(folder: Path) -> ReviewSession:
    """Items in their shuffled order, each side's image and hints. Variant names stay out."""
    manifest = _load_manifest(folder)
    items = [Item.from_dict(i) for i in manifest["items"]]
    by_variant: dict[str, dict[tuple[str, int], tuple[Path, dict[str, Any]]]] = {}
    for variant in ("a", "b"):
        found: dict[tuple[str, int], tuple[Path, dict[str, Any]]] = {}
        for results in manifest[variant]["results"]:
            images = paths.eval_dir() / Path(results).name  # §5.7: images live under $QWI_HOME
            _, rows = load_results([Path(results)])
            found |= {key: (images / row["image"], row) for key, row in rows.items()}
        by_variant[variant] = found
    sides: dict[tuple[str, str], Side] = {}
    for item in items:
        for side in SIDES:
            image, row = by_variant[item.variant_of(side)][(item.prompt_id, item.seed)]
            hints = {k: row[k] for k in HINT_KEYS if row.get(k) is not None}
            hints["seconds"] = round(sum(row.get("seconds", {}).values()), 1)
            sides[(item.item, side)] = Side(image, row["prompt"], hints)
    return ReviewSession(folder, items, sides)


def read_judgments(folder: Path) -> dict[str, Judgment]:
    left: dict[str, dict[str, str]] = defaultdict(dict)
    right: dict[str, dict[str, str]] = defaultdict(dict)
    extra: dict[str, tuple[str, str]] = {}
    for row in _read_review(folder):
        if row["check"]:
            left[row["item"]][row["check"]] = row["left"].lower()
            right[row["item"]][row["check"]] = row["right"].lower()
        else:
            extra[row["item"]] = (row["prefer"].lower(), row["notes"])
    ids = left.keys() | extra.keys()
    return {i: Judgment(left[i], right[i], *extra.get(i, ("", ""))) for i in ids}


def save_judgment(folder: Path, item: Item, judgment: Judgment) -> None:
    """Rewrite `review.csv` with this item's rows replaced; atomic, so a crash loses nothing."""
    bad = {*judgment.left.values(), *judgment.right.values()} - {*RESULTS, ""}
    if bad or judgment.prefer not in (*PREFER, ""):
        raise SessionError(f"{item.item}: results must be pass/fail, prefer left/right/same")
    rows = _read_review(folder)
    if not any(r["item"] == item.item for r in rows):
        raise SessionError(f"{item.item} is not in {folder.name}/review.csv")
    for row in rows:
        if row["item"] != item.item:
            continue
        if row["check"]:
            row["left"] = judgment.left.get(row["check"], "")
            row["right"] = judgment.right.get(row["check"], "")
        else:
            row["prefer"], row["notes"] = judgment.prefer, judgment.notes
    target = folder / "review.csv"
    tmp = target.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, REVIEW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(target)


def session_dir(arg: str) -> Path:
    """A session given as a path, or as a name under `eval/sessions/`."""
    path = Path(arg)
    return path if (path / "manifest.json").is_file() else paths.EVAL_SET / "sessions" / arg
