"""Blind sessions and the §5.4 decision rule, on synthetic results.jsonl + review.csv fixtures."""

import csv
import json
import math
import re
import shutil
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from qwi import cli, paths, sessions
from qwi.errors import SessionError, SessionIncomplete
from qwi.evaluation import load_prompts

DAY = date(2026, 10, 15)
Outcome = Callable[[str, int, str], tuple[str, str]]  # (prompt id, seed, check) -> (A, B)
Prefer = Callable[[str, int], str]  # (prompt id, seed) -> "a" | "b" | "same" | ""


def results(
    variant: str,
    seeds: tuple[int, ...] = (1000, 1001),
    seconds: float = 125.0,
    peak: float = 13.8,
    swap: float = 0.0,
) -> Path:
    """A variant's results.jsonl in the tmp eval set, every prompt x `seeds`."""
    folder = paths.eval_results_dir() / f"2026-10-15_{variant}"
    folder.mkdir(parents=True)
    with (folder / "results.jsonl").open("w") as f:
        for p in load_prompts():
            for seed in seeds:
                row = {
                    "id": p.id,
                    "seed": seed,
                    "variant": variant,
                    "tags": list(p.tags),
                    "seconds": {"generate": seconds},
                    "peak_gb": {"generate": peak},
                    "swap_gb": swap,
                }
                f.write(json.dumps(row) + "\n")
    return folder


def ties(prompt_id: str, seed: int, check: str) -> tuple[str, str]:
    return "pass", "pass"


def judge(session: Path, outcome: Outcome = ties, prefer: str | Prefer = "") -> None:
    """Fill review.csv from intended A/B results, through the manifest's random sides."""
    manifest = json.loads((session / "manifest.json").read_text())
    items = {i["item"]: i for i in manifest["items"]}
    with (session / "review.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        item = items[row["item"]]
        a_side = "left" if item["left"] == "a" else "right"
        b_side = "right" if a_side == "left" else "left"
        if row["check"]:
            a, b = outcome(item["prompt_id"], item["seed"], row["check"])
            row[a_side], row[b_side] = a, b
        else:
            choice = prefer(item["prompt_id"], item["seed"]) if callable(prefer) else prefer
            if choice:
                row["prefer"] = {"a": a_side, "b": b_side}.get(choice, choice)
    with (session / "review.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sessions.REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def first_checks(n: int, a: str, b: str, *, text: bool | None = None) -> Outcome:
    """`n` (prompt, seed, check) triples get (a, b); the rest tie. `text` filters text-* prompts."""
    chosen: set[tuple[str, int, str]] = set()
    for p in load_prompts():
        if text is not None and p.id.startswith("text-") != text:
            continue
        for seed in (1000, 1001):
            for check in p.checks:
                if len(chosen) < n:
                    chosen.add((p.id, seed, check))

    def outcome(prompt_id: str, seed: int, check: str) -> tuple[str, str]:
        return (a, b) if (prompt_id, seed, check) in chosen else ("pass", "pass")

    return outcome


@pytest.fixture
def ab() -> tuple[Path, Path]:
    return results("baseline"), results("short")


def new(ab: tuple[Path, Path], seed: int = 7) -> Path:
    return sessions.create([ab[0]], [ab[1]], seed=seed, day=DAY)


# --- creating a session --------------------------------------------------------------------


def test_create_writes_an_anonymous_shuffled_manifest(ab: tuple[Path, Path]) -> None:
    session = new(ab)
    assert session == paths.EVAL_SET / "sessions" / "2026-10-15_baseline_vs_short"
    manifest = json.loads((session / "manifest.json").read_text())
    assert (manifest["a"]["name"], manifest["b"]["name"], manifest["shuffle_seed"]) == (
        "baseline",
        "short",
        7,
    )
    items = manifest["items"]
    assert len(items) == 24
    assert [i["item"] for i in items] == [f"i{n:03d}" for n in range(1, 25)]  # ids reveal nothing
    keys = [(i["prompt_id"], i["seed"]) for i in items]
    assert keys != sorted(keys)  # order shuffled
    assert {i["left"] for i in items} == {"a", "b"}  # sides randomized per item


def test_the_same_seed_reproduces_the_session(ab: tuple[Path, Path]) -> None:
    first = json.loads((new(ab, seed=7) / "manifest.json").read_text())["items"]
    for item in paths.EVAL_SET.joinpath("sessions").iterdir():
        for f in item.iterdir():
            f.unlink()
        item.rmdir()
    assert json.loads((new(ab, seed=7) / "manifest.json").read_text())["items"] == first


def test_review_csv_starts_empty_with_one_row_per_item_and_check(ab: tuple[Path, Path]) -> None:
    session = new(ab)
    assert b"\r" not in (session / "review.csv").read_bytes()  # LF like the repo
    with (session / "review.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    checks = sum(len(p.checks) for p in load_prompts()) * 2
    assert len(rows) == 24 + checks  # one item row (prefer, notes) + one row per check
    assert all(not (r["left"] or r["right"] or r["prefer"]) for r in rows)


def test_create_refuses_bad_inputs(ab: tuple[Path, Path]) -> None:
    with pytest.raises(SessionError, match="different variants"):
        sessions.create([ab[0]], [ab[0]], seed=1, day=DAY)
    other = results("other", seeds=(5,))
    with pytest.raises(SessionError, match="no \\(prompt, seed\\) pairs"):
        sessions.create([ab[0]], [other], seed=1, day=DAY)
    new(ab)
    with pytest.raises(SessionError, match="exists"):
        new(ab)
    with pytest.raises(SessionError, match=r"no results\.jsonl"):
        sessions.results_dir("missing")


def test_results_dir_accepts_a_name_or_a_path(ab: tuple[Path, Path]) -> None:
    assert sessions.results_dir("2026-10-15_baseline") == ab[0]
    assert sessions.results_dir(str(ab[0])) == ab[0]


# --- blinding ------------------------------------------------------------------------------


def test_compare_refuses_until_every_check_is_judged(ab: tuple[Path, Path]) -> None:
    session = new(ab)
    with pytest.raises(SessionIncomplete) as excinfo:
        sessions.compare(session)
    assert "baseline" not in str(excinfo.value)  # nothing revealed
    assert not (session / "report.md").exists()

    judge(session)
    with (session / "review.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows[1]["left"] = ""  # one check left unjudged
    with (session / "review.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sessions.REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SessionIncomplete, match=r"^1 checks"):
        sessions.compare(session)


def test_a_bad_preference_value_is_refused(ab: tuple[Path, Path]) -> None:
    session = new(ab)
    judge(session, prefer="maybe")
    with pytest.raises(SessionError, match="prefer"):
        sessions.compare(session)


def test_scores_follow_the_variants_not_the_sides(ab: tuple[Path, Path]) -> None:
    session = new(ab)
    judge(session, first_checks(3, "fail", "pass"))  # B passes 3 checks A fails
    report = sessions.compare(session)
    o = report.overall
    assert (o.wins, o.losses) == (3, 0)
    assert o.ties == sum(len(p.checks) for p in load_prompts()) * 2 - 3


# --- the decision rule ---------------------------------------------------------------------


def test_all_ties_is_not_worse(ab: tuple[Path, Path]) -> None:
    session = new(ab)
    judge(session)
    report = sessions.compare(session)
    assert report.verdict == "not worse"
    assert all(report.rules.values())


def test_rule_1_allows_two_net_losses_and_fails_on_three(ab: tuple[Path, Path]) -> None:
    session = new(ab)
    judge(session, first_checks(2, "pass", "fail", text=False))
    assert sessions.compare(session).verdict == "not worse"
    judge(session, first_checks(3, "pass", "fail", text=False))
    report = sessions.compare(session)
    assert (report.verdict, report.rules["quality"]) == ("worse", False)


def test_noise_floor_widens_rule_1(ab: tuple[Path, Path]) -> None:
    session = new(ab)
    judge(session, first_checks(3, "pass", "fail", text=False))
    assert sessions.compare(session, noise_floor=3).rules["quality"] is True


def test_rule_2_fails_on_a_single_net_text_loss(ab: tuple[Path, Path]) -> None:
    session = new(ab)
    judge(session, first_checks(1, "pass", "fail", text=True))
    report = sessions.compare(session)
    assert report.rules["quality"] is True
    assert (report.verdict, report.rules["text"]) == ("worse", False)


@pytest.mark.parametrize(
    ("b_kwargs", "rule"),
    [
        ({"seconds": 125.0 * 1.16}, "time"),
        ({"peak": 24.1}, "peak"),
        ({"swap": 0.05}, "swap"),
    ],
)
def test_rule_3_resources_fail_the_variant_outright(b_kwargs: dict[str, Any], rule: str) -> None:
    a = results("baseline")
    b = results("short", **b_kwargs)
    session = sessions.create([a], [b], seed=7, day=DAY)
    judge(session, first_checks(5, "fail", "pass"))  # even with a clear quality win
    report = sessions.compare(session)
    assert report.verdict == "worse"
    assert report.rules[rule] is False
    assert rule in report.reason


def test_time_exactly_at_the_limit_passes() -> None:
    session = sessions.create(
        [results("baseline")], [results("short", seconds=125.0 * 1.15)], seed=7, day=DAY
    )
    judge(session)
    assert sessions.compare(session).rules["time"] is True


def test_better_needs_three_net_wins(ab: tuple[Path, Path]) -> None:
    session = new(ab)
    judge(session, first_checks(2, "fail", "pass"))
    assert sessions.compare(session).verdict == "not worse"
    judge(session, first_checks(3, "fail", "pass"))
    report = sessions.compare(session)
    assert report.verdict == "better"
    assert "wins - losses = 3" in report.reason


def test_better_by_preference_majority(ab: tuple[Path, Path]) -> None:
    session = new(ab)
    judge(session, prefer="b")
    report = sessions.compare(session)
    assert (report.verdict, report.prefer["b"]) == ("better", 24)
    assert "preference" in report.reason


def a_preferred(n: int) -> Prefer:
    """A preferred on the first `n` (prompt, seed) items in eval-set order, B on the rest."""
    order = [(p.id, seed) for p in load_prompts() for seed in (1000, 1001, 1002, 1003)]
    first = set(order[:n])
    return lambda prompt_id, seed: "a" if (prompt_id, seed) in first else "b"


def test_rule_4_a_preference_majority_against_b_extends_the_seeds(ab: tuple[Path, Path]) -> None:
    # The Turbo session (§8 2026-10-18): rules 1-3 pass, A preferred 16 to 8 (p = 0.152).
    session = new(ab)
    judge(session, prefer=a_preferred(32))  # 2 seeds per prompt here: 16 items for A, 8 for B
    report = sessions.compare(session)
    assert (report.prefer["a"], report.prefer["b"]) == (16, 8)
    assert report.rules["preference"] is False
    assert report.verdict == "inconclusive"
    assert "A is preferred in 16 of 24 items (p = 0.152)" in report.reason
    assert f"--extend {session.name}" in report.reason


def test_rule_4_also_blocks_better_by_wins(ab: tuple[Path, Path]) -> None:
    session = new(ab)
    judge(session, first_checks(3, "fail", "pass"), prefer="a")
    assert sessions.compare(session).verdict == "inconclusive"


def test_rule_4_a_tie_or_same_majority_does_not_trigger(ab: tuple[Path, Path]) -> None:
    session = new(ab)
    judge(session, prefer=lambda pid, seed: "a" if seed == 1000 else "b")  # 12 vs 12
    assert sessions.compare(session).verdict == "not worse"
    judge(session, prefer=lambda pid, seed: "a" if pid == "count" else "same")  # 2 of 24
    report = sessions.compare(session)
    assert (report.verdict, report.reason) == ("not worse", "passes rules 1-4")


@pytest.mark.parametrize(
    ("a_items", "verdict", "reason"),
    [
        (32, "worse", "B fails rule 4: after extending, A is preferred in 32 of 48 items"),
        (27, "not worse", "is not significant after extending"),
    ],
)
def test_rule_4_after_extending_decides_on_the_preference_sign_test(
    ab: tuple[Path, Path], a_items: int, verdict: str, reason: str
) -> None:
    old = new(ab)
    judge(old)
    a_ext = results("baseline-ext", seeds=(1002, 1003))
    b_ext = results("short-ext", seeds=(1002, 1003))
    session = sessions.create([a_ext], [b_ext], seed=8, day=DAY, extend=old)
    judge(session, prefer=a_preferred(a_items))  # all 48 items, old and new
    report = sessions.compare(session)
    assert report.extended
    assert report.prefer["a"] == a_items  # 32 vs 16: p = 0.029; 27 vs 21: p = 0.471
    assert report.verdict == verdict
    assert reason in report.reason


def test_a_noisy_failure_is_inconclusive_and_says_how_to_extend(ab: tuple[Path, Path]) -> None:
    # 6 losses, 2 wins: rule 1 fails (net 4 > 2) on 8 discordant pairs, sign test p = 0.289.
    losses = first_checks(6, "pass", "fail", text=False)
    wins = first_checks(8, "fail", "pass", text=False)

    def outcome(prompt_id: str, seed: int, check: str) -> tuple[str, str]:
        if losses(prompt_id, seed, check) != ("pass", "pass"):
            return "pass", "fail"
        return wins(prompt_id, seed, check)

    session = new(ab)
    judge(session, outcome)
    report = sessions.compare(session)
    assert (report.overall.wins, report.overall.losses) == (2, 6)
    assert report.verdict == "inconclusive"
    assert "--seeds 1002,1003" in report.reason
    assert f"--extend {session.name}" in report.reason


def test_a_clear_failure_on_few_pairs_is_worse_not_inconclusive(ab: tuple[Path, Path]) -> None:
    session = new(ab)
    judge(session, first_checks(5, "pass", "fail", text=False))  # 5 discordant < 8
    assert sessions.compare(session).verdict == "worse"


def test_the_manifest_names_results_so_another_checkout_can_compare(
    ab: tuple[Path, Path], eval_set: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = new(ab)
    judge(old)
    manifest = json.loads((old / "manifest.json").read_text())
    assert (manifest["a"]["results"], manifest["b"]["results"]) == (
        ["2026-10-15_baseline"],
        ["2026-10-15_short"],
    )
    # Another checkout: the same committed eval set somewhere else, the original gone.
    moved = tmp_path / "elsewhere" / "eval"
    shutil.move(eval_set, moved)
    monkeypatch.setattr(paths, "EVAL_SET", moved)
    assert sessions.compare(moved / "sessions" / old.name).overall.ties > 0
    ext = sessions.create(
        [results("baseline-ext", seeds=(1002, 1003))],
        [results("short-ext", seeds=(1002, 1003))],
        seed=8,
        day=DAY,
        extend=moved / "sessions" / old.name,
    )
    manifest = json.loads((ext / "manifest.json").read_text())
    assert manifest["a"]["results"] == ["2026-10-15_baseline", "2026-10-15_baseline-ext"]


def test_extending_keeps_judgments_and_adds_only_new_pairs(ab: tuple[Path, Path]) -> None:
    old = new(ab)
    judge(old, first_checks(3, "pass", "fail", text=False))
    # Named as the inconclusive verdict says: `<name>-ext` counts as `<name>`.
    a_ext = results("baseline-ext", seeds=(1002, 1003))
    b_ext = results("short-ext", seeds=(1002, 1003))
    with pytest.raises(SessionError, match="same two variants"):
        sessions.create([b_ext], [a_ext], seed=8, day=DAY, extend=old)  # sides swapped
    session = sessions.create([a_ext], [b_ext], seed=8, day=DAY, extend=old)
    assert session.name == f"{old.name}_ext"
    manifest = json.loads((session / "manifest.json").read_text())
    assert len(manifest["items"]) == 48
    assert manifest["extends"] == old.name
    with (session / "review.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    judged_rows = [r for r in rows if r["check"] and r["left"]]
    assert len(judged_rows) == sum(len(p.checks) for p in load_prompts()) * 2  # old ones kept
    with pytest.raises(SessionIncomplete):
        sessions.compare(session)  # the new pairs still need judging
    judge(session)
    assert sessions.compare(session).overall.ties > 0  # both folders load as one variant per side


def test_an_extension_with_a_different_config_is_refused(ab: tuple[Path, Path]) -> None:
    a, _ = ab
    other = results("baseline-ext", seeds=(1002, 1003))
    for folder, steps in ((a, 20), (other, 8)):
        path = folder / "results.jsonl"
        lines = path.read_text().splitlines()
        rows = [json.loads(line) | {"config": {"steps": steps}} for line in lines]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(SessionError, match="one run config per side: steps"):
        sessions.load_results([a, other])


def test_an_extension_may_record_config_keys_an_older_run_lacks(ab: tuple[Path, Path]) -> None:
    a, _ = ab  # no config at all, like an M1 row without `rewrite` / `guard_mode`
    ext = results("baseline-ext", seeds=(1002, 1003))
    path = ext / "results.jsonl"
    rows = [
        json.loads(line) | {"config": {"rewrite": "none", "guard_mode": "default"}}
        for line in path.read_text().splitlines()
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    name, loaded = sessions.load_results([a, ext])
    assert (name, len({seed for _, seed in loaded})) == ("baseline", 4)


def test_sign_test_exact_values() -> None:
    assert sessions.sign_test(0, 0) == 1.0
    assert sessions.sign_test(5, 0) == pytest.approx(2 / 32)
    assert sessions.sign_test(2, 8) == pytest.approx(2 * (1 + 10 + 45) / 1024)
    assert sessions.sign_test(4, 4) == 1.0
    assert sessions.sign_test(0, 20) == pytest.approx(2 / 2**20)
    assert sessions.sign_test(30, 30) == min(
        1.0, 2 * sum(math.comb(60, i) for i in range(31)) / 2**60
    )


# --- report and CLI ------------------------------------------------------------------------


def test_report_md_is_written_with_verdict_rules_and_tables(ab: tuple[Path, Path]) -> None:
    session = new(ab)
    judge(session, first_checks(3, "fail", "pass"))
    _, path = sessions.write_report(session)
    text = path.read_text()
    assert path == session / "report.md"
    assert "## Verdict: B is **better**" in text
    assert "| 3c zero swap on every image | 0 swapped (A 0) | pass |" in text
    assert "## Per tag" in text
    assert "| text |" in text
    assert "## Per check" in text


runner = CliRunner()


def plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_cli_session_then_compare(ab: tuple[Path, Path]) -> None:
    result = runner.invoke(
        cli.app,
        ["eval", "session", "--a", "2026-10-15_baseline", "--b", "2026-10-15_short", "--seed", "7"],
    )
    assert result.exit_code == 0, result.output
    name = plain(result.output).split()[1].rstrip(":")
    session = paths.EVAL_SET / "sessions" / name

    result = runner.invoke(cli.app, ["eval", "compare", name])
    assert result.exit_code == 1
    assert "not judged yet" in plain(result.output)

    judge(session)
    result = runner.invoke(cli.app, ["eval", "compare", name])
    assert result.exit_code == 0
    assert "B is not worse" in result.output
    assert (session / "report.md").is_file()
