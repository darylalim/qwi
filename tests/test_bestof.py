"""Best-of-N calibration sessions (`qwi eval session --bo4`, PLAN.md §5.4): data, page, CLI."""

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest
from typer.testing import CliRunner

from qwi import bestof, cli, evaluation, paths, sessions
from qwi.bestof import BestOfJudgment
from qwi.config import DEFAULTS
from qwi.errors import SessionError
from qwi.evaluation import load_prompts

from .test_review import variant as paired_variant

APP = str(Path(cli.__file__).parent / "ui" / "bestof_app.py")
DAY = date(2026, 10, 17)
SEEDS = (2000, 2001, 2002, 2003)
COLORS = {2000: (255, 0, 0), 2001: (0, 255, 0), 2002: (0, 0, 255), 2003: (255, 255, 0)}


def bo4(name: str = "bo4", seeds: dict[str, tuple[int, ...]] | None = None) -> Path:
    """Results + tiny images like `qwi eval --variant bo4 --seeds 2000,2001,2002,2003` writes."""
    folder = paths.eval_results_dir() / f"2026-10-17_{name}"
    images = paths.eval_dir() / folder.name
    folder.mkdir(parents=True)
    images.mkdir(parents=True)
    with (folder / "results.jsonl").open("w") as f:
        for p in load_prompts():
            for seed in (seeds or {}).get(p.id, SEEDS):
                image = f"{p.id}_{seed}.png"
                Image.new("RGB", (8, 8), COLORS.get(seed, (0, 0, 0))).save(images / image)
                row = {
                    "id": p.id,
                    "seed": seed,
                    "variant": name,
                    "prompt": p.prompt,
                    "tags": list(p.tags),
                    "image": image,
                    "rerank_margin": seed / 1000,  # the task instruction always picks 2003
                    "rerank_prob": 0.9,
                    "rerank_instruction": DEFAULTS.rerank_instruction,
                    "seconds": {"generate": 120.0},
                    "peak_gb": {"generate": 14.0},
                    "swap_gb": 0.0,
                }
                f.write(json.dumps(row) + "\n")
    return folder


@pytest.fixture
def session() -> Path:
    return bestof.create(bo4(), seed=7, day=DAY)


def complete(item: bestof.BestOfItem, best: int = 1, ok: str = "yes") -> BestOfJudgment:
    return BestOfJudgment(dict.fromkeys(item.candidates, ok), best, "")


# --- Creating a session --------------------------------------------------------------------


def test_create_shuffles_prompts_and_candidates_reproducibly(session: Path) -> None:
    assert session == paths.EVAL_SET / "sessions" / "2026-10-17_bo4"  # no `bo4_bo4`
    manifest = json.loads((session / "manifest.json").read_text())
    assert (manifest["kind"], manifest["variant"], manifest["results"]) == (
        "bo4",
        "bo4",
        "2026-10-17_bo4",  # a name in eval/results/, not a home-directory path
    )
    items = manifest["items"]
    assert [i["item"] for i in items] == [f"i{n:03d}" for n in range(1, 13)]
    assert sorted(i["prompt_id"] for i in items) == sorted(p.id for p in load_prompts())
    assert all(sorted(i["seeds"]) == list(SEEDS) for i in items)
    assert any(i["seeds"] != list(SEEDS) for i in items)  # candidate order really is shuffled
    again = bestof.create(bo4("bo4b"), seed=7, day=DAY)
    replay = json.loads((again / "manifest.json").read_text())["items"]
    assert [(i["prompt_id"], i["seeds"]) for i in replay] == [
        (i["prompt_id"], i["seeds"]) for i in items
    ]


def test_review_csv_is_blind(session: Path) -> None:
    with (session / "review.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == bestof.REVIEW_COLUMNS
    assert len(rows) == 12 * (1 + 4)  # an item row (best, notes) + one per candidate
    assert rows[:5] == [
        {"item": "i001", "candidate": c, "acceptable": "", "best": "", "notes": ""}
        for c in ("", "1", "2", "3", "4")
    ]
    text = (session / "review.csv").read_text()
    assert not any(str(s) in text for s in SEEDS)  # seeds stay in manifest.json


@pytest.mark.parametrize(
    "seeds",
    [{"count": (2000, 2001, 2002)}, dict.fromkeys((p.id for p in load_prompts()), (2000,))],
    ids=["uneven", "one-seed"],
)
def test_create_needs_the_same_two_or_more_seeds_per_prompt(
    seeds: dict[str, tuple[int, ...]],
) -> None:
    with pytest.raises(SessionError, match="same number"):
        bestof.create(bo4(seeds=seeds), seed=7, day=DAY)


def test_create_never_overwrites(session: Path) -> None:
    with pytest.raises(SessionError, match="exists"):
        bestof.create(paths.eval_results_dir() / "2026-10-17_bo4", seed=7, day=DAY)


# --- Opening, judging, saving --------------------------------------------------------------


def test_open_session_maps_each_candidate_to_its_seeds_image(session: Path) -> None:
    review = bestof.open_session(session)
    prompts = {p.id: p.prompt for p in load_prompts()}
    for item in review.items:
        assert review.prompts[item.prompt_id] == prompts[item.prompt_id]
        for n, seed in zip(item.candidates, item.seeds, strict=True):
            assert Image.open(review.images[(item.item, n)]).getpixel((0, 0)) == COLORS[seed]


@pytest.mark.parametrize(
    ("judgment", "problem"),
    [
        (BestOfJudgment({1: "yes", 2: "no", 3: "yes"}, 1), "Mark every candidate"),
        (BestOfJudgment({1: "yes", 2: "no", 3: "yes", 4: "maybe"}, 1), "Mark every candidate"),
        (BestOfJudgment({1: "yes", 2: "no", 3: "yes", 4: "no"}, None), "Pick the best"),
        (BestOfJudgment({1: "yes", 2: "no", 3: "yes", 4: "no"}, 5), "Pick the best"),
        (BestOfJudgment({1: "yes", 2: "no", 3: "yes", 4: "no"}, 2), "must be an acceptable"),
    ],
    ids=["missing", "bad-value", "no-best", "best-out-of-range", "best-unacceptable"],
)
def test_incomplete_or_inconsistent_judgments_have_a_problem(
    session: Path, judgment: BestOfJudgment, problem: str
) -> None:
    item = bestof.open_session(session).items[0]
    found = judgment.problem(item)
    assert found is not None
    assert problem in found


def test_with_nothing_acceptable_the_least_bad_is_still_picked(session: Path) -> None:
    item = bestof.open_session(session).items[0]
    assert complete(item, best=3, ok="no").problem(item) is None


def test_save_round_trips_and_touches_only_that_item(session: Path) -> None:
    first, second = bestof.open_session(session).items[:2]
    judgment = BestOfJudgment({1: "no", 2: "yes", 3: "yes", 4: "no"}, 3, "sharp sign")
    bestof.save_judgment(session, first, judgment)
    saved = bestof.read_judgments(session)
    assert saved[first.item] == judgment
    assert saved[second.item] == BestOfJudgment({1: "", 2: "", 3: "", 4: ""}, None, "")
    assert b"\r" not in (session / "review.csv").read_bytes()
    assert not (session / "review.csv.tmp").exists()


def test_save_refuses_an_incomplete_judgment_or_an_unknown_item(session: Path) -> None:
    item = bestof.open_session(session).items[0]
    before = (session / "review.csv").read_text()
    with pytest.raises(SessionError, match="Pick the best"):
        bestof.save_judgment(session, item, BestOfJudgment(dict.fromkeys(item.candidates, "no")))
    stranger = bestof.BestOfItem("i999", "count", SEEDS)
    with pytest.raises(SessionError, match="i999 is not in"):
        bestof.save_judgment(session, stranger, complete(stranger))
    assert (session / "review.csv").read_text() == before


def test_paired_and_best_of_sessions_refuse_each_other(session: Path) -> None:
    with pytest.raises(SessionError, match="best-of-N session, not an A vs B pair"):
        sessions.compare(session)
    with pytest.raises(SessionError, match="best-of-N session"):
        sessions.open_session(session)
    paired = sessions.create(
        [paired_variant("alpha")], [paired_variant("beta")], seed=7, day=date(2026, 10, 15)
    )
    with pytest.raises(SessionError, match="not a best-of-N session"):
        bestof.open_session(paired)


# --- The page (AppTest) --------------------------------------------------------------------


def app(session: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setenv(sessions.REVIEW_SESSION_ENV, str(session))
    at = AppTest.from_file(APP, default_timeout=10).run()
    assert not at.exception
    return at


def texts(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    return "\n".join(str(p) for p in parts)


def fill(at: AppTest, item: bestof.BestOfItem, marks: dict[int, str], best: int | None) -> None:
    for n, mark in marks.items():
        at.segmented_control(key=f"{item.item}-{n}-acceptable").set_value(mark)
    at.segmented_control(key=f"{item.item}-best").set_value(best)


def test_the_first_prompt_shows_its_candidates_and_nothing_else(
    session: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = app(session, monkeypatch)
    review = bestof.open_session(session)
    item = review.items[0]
    prompt = review.prompts[item.prompt_id]
    assert "Prompt 1 of 12" in texts(at)
    assert f"**Prompt:** {prompt}" in texts(at)
    assert [img.proto.imgs[0].alt for img in at.image] == [
        f"Candidate {n}: {prompt}" for n in item.candidates
    ]
    assert [h.value for h in at.subheader] == [f"Candidate {n}" for n in item.candidates]
    assert len(at.segmented_control) == 4 + 1  # acceptable per candidate, then best
    shown = texts(at)
    assert not any(str(s) in shown for s in SEEDS)  # no seeds ...
    assert "0.9" not in shown  # ... and no reranker scores
    assert at.progress[0].value == 0


def test_an_incomplete_save_is_refused(session: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = app(session, monkeypatch)
    item = bestof.open_session(session).items[0]
    before = (session / "review.csv").read_text()
    fill(at, item, {1: "yes", 2: "no"}, 1)
    at.form_submit_button[0].click().run()
    assert at.warning[0].value == "Mark every candidate acceptable or not."
    assert (session / "review.csv").read_text() == before


def test_the_best_must_be_acceptable_when_any_is(
    session: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = app(session, monkeypatch)
    item = bestof.open_session(session).items[0]
    fill(at, item, {1: "yes", 2: "no", 3: "no", 4: "no"}, 2)
    at.form_submit_button[0].click().run()
    assert at.warning[0].value == "The best candidate must be an acceptable one, when any is."


def test_save_writes_review_csv_and_marks_the_prompt_saved(
    session: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = app(session, monkeypatch)
    item = bestof.open_session(session).items[0]
    fill(at, item, {1: "no", 2: "yes", 3: "yes", 4: "no"}, 3)
    at.text_input(key=f"{item.item}-notes").set_value("crisp apples")
    at.form_submit_button[0].click().run()
    assert not at.exception
    assert bestof.read_judgments(session)[item.item] == BestOfJudgment(
        {1: "no", 2: "yes", 3: "yes", 4: "no"}, 3, "crisp apples"
    )
    assert ":green-badge[:material/check: Saved]" in texts(at)
    assert at.progress[0].value == round(100 / 12)


def test_reopening_starts_at_the_first_unjudged_prompt(
    session: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = bestof.open_session(session).items[0]
    bestof.save_judgment(session, first, complete(first))
    at = app(session, monkeypatch)
    assert "Prompt 2 of 12" in texts(at)


def test_paging_stays_within_bounds(session: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at = app(session, monkeypatch)
    for _ in range(11):
        at.button(key="next").click().run()
    assert "Prompt 12 of 12" in texts(at)
    assert at.button(key="next").disabled
    at.button(key="previous").click().run()
    assert "Prompt 11 of 12" in texts(at)


def test_a_paired_session_shows_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    paired = sessions.create(
        [paired_variant("alpha")], [paired_variant("beta")], seed=7, day=date(2026, 10, 15)
    )
    monkeypatch.setenv(sessions.REVIEW_SESSION_ENV, str(paired))
    at = AppTest.from_file(APP, default_timeout=10).run()
    assert not at.exception
    assert "not a best-of-N session" in at.error[0].value


# --- CLI -----------------------------------------------------------------------------------

runner = CliRunner()


def test_cli_creates_a_best_of_session() -> None:
    bo4()
    result = runner.invoke(cli.app, ["eval", "session", "--bo4", "2026-10-17_bo4", "--seed", "7"])
    assert result.exit_code == 0, result.output
    assert "qwi eval review 2026-10-17_bo4" in result.output
    manifest = json.loads((paths.EVAL_SET / "sessions/2026-10-17_bo4/manifest.json").read_text())
    assert manifest["kind"] == "bo4"


@pytest.mark.parametrize(
    "args",
    [["--bo4", "x", "--a", "y"], ["--bo4", "x", "--extend", "z"], [], ["--a", "y"]],
    ids=["bo4-and-a", "bo4-and-extend", "nothing", "a-only"],
)
def test_cli_session_flags_must_fit_one_kind(args: list[str]) -> None:
    result = runner.invoke(cli.app, ["eval", "session", *args])
    assert result.exit_code == 2


def test_cli_review_launches_the_best_of_page(
    session: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def call(command: list[str], **_: Any) -> int:
        calls.append(command)
        return 0

    monkeypatch.setattr(cli.subprocess, "call", call)
    result = runner.invoke(cli.app, ["eval", "review", session.name])
    assert result.exit_code == 0, result.output
    assert calls[0][4] == APP


# --- Calibration (`qwi eval calibrate`) ----------------------------------------------------


def judge_all(
    session: Path,
    best_seed: int = 2000,
    ok_seeds: tuple[int, ...] = (2000, 2001),
    none_ok: tuple[str, ...] = (),
) -> None:
    """Judge every prompt: `best_seed` is best, `ok_seeds` acceptable (none for `none_ok`)."""
    for item in bestof.open_session(session).items:
        oks = () if item.prompt_id in none_ok else ok_seeds
        marks = {
            n: "yes" if s in oks else "no" for n, s in zip(item.candidates, item.seeds, strict=True)
        }
        best = item.seeds.index(best_seed) + 1
        bestof.save_judgment(session, item, BestOfJudgment(marks, best, ""))


def rescored(label: str, margin: Any, prob: Any = None, name: str = "2026-10-17_bo4") -> None:
    """A `rerank_<label>.jsonl` with margin(prompt_id, seed) and prob(prompt_id, seed)."""
    rows = evaluation.read_rows(paths.eval_results_dir() / name / "results.jsonl")
    evaluation.write_rows(
        paths.eval_results_dir() / name / f"rerank_{label}.jsonl",
        [
            {
                "id": r["id"],
                "seed": r["seed"],
                "image": r["image"],
                "label": label,
                "instruction": evaluation.RERANK_PRESETS.get(label, f"{label} text"),
                "rerank_margin": margin(r["id"], r["seed"]),
                "rerank_prob": (prob or (lambda i, s: 0.8))(r["id"], r["seed"]),
                "rerank_resized": False,
            }
            for r in rows
        ],
    )


def test_calibrate_refuses_until_every_prompt_is_judged(session: Path) -> None:
    first = bestof.open_session(session).items[0]
    bestof.save_judgment(session, first, complete(first))
    with pytest.raises(SessionError, match="11 prompts are not judged yet"):
        bestof.calibrate(session)


def test_calibrate_scores_each_instruction_against_the_human(session: Path) -> None:
    judge_all(session, best_seed=2000, ok_seeds=(2000, 2001))
    rescored("generic", lambda i, s: -s)  # generic prefers the lowest seed: 2000
    c = bestof.calibrate(session)
    task, generic = c.instructions
    assert (task.label, task.instruction) == ("task", DEFAULTS.rerank_instruction)
    assert (generic.label, generic.instruction) == ("generic", DEFAULTS.rerank_generic_instruction)
    # task always picks 2003 (never best, never acceptable); generic always picks 2000.
    assert (task.agreement, task.top_acceptable) == (0, 0)
    assert (generic.agreement, generic.top_acceptable) == (12, 12)
    assert c.winner == "generic"
    assert (c.items, c.candidates, c.with_acceptable) == (12, 4, 12)
    assert (c.base_seed_agreement, c.base_seed_acceptable) == (12, 12)  # 2000 is the base seed
    assert (c.chance_agreement, c.chance_acceptable) == (3.0, 6.0)  # 12/4; 12 x 2/4


def test_a_margin_tie_goes_to_the_lowest_seed(session: Path) -> None:
    judge_all(session, best_seed=2000)
    rescored("flat", lambda i, s: 1.0)
    flat = next(r for r in bestof.calibrate(session).instructions if r.label == "flat")
    assert flat.agreement == 12


def test_a_full_tie_keeps_the_default_instruction(session: Path) -> None:
    judge_all(session, best_seed=2003, ok_seeds=(2003,))
    rescored("generic", lambda i, s: s)  # the same picks as task
    assert bestof.calibrate(session).winner == "task"


def test_min_score_separates_prompts_with_no_acceptable_candidate(session: Path) -> None:
    judge_all(session, best_seed=2003, ok_seeds=(2003,), none_ok=("count", "spatial"))
    low = {"count": 0.3, "spatial": 0.42}
    rescored("generic", lambda i, s: -s, lambda i, s: low.get(i, 0.7))  # worse picks: task wins
    c = bestof.calibrate(session)
    assert c.winner == "task"  # task agrees on the 10 prompts with an acceptable candidate
    # Rewrite task's probs in results.jsonl the same way, so its picks carry them.
    results = paths.eval_results_dir() / "2026-10-17_bo4" / "results.jsonl"
    rows = evaluation.read_rows(results)
    for r in rows:
        r["rerank_prob"] = low.get(r["id"], 0.7)
    evaluation.write_rows(results, rows)
    c = bestof.calibrate(session)
    assert c.min_score == 0.56  # midpoint of 0.42 and 0.7
    assert "separates them" in c.min_score_reason


def test_overlapping_probs_keep_the_default_min_score(session: Path) -> None:
    judge_all(session, best_seed=2003, ok_seeds=(2003,), none_ok=("count",))
    results = paths.eval_results_dir() / "2026-10-17_bo4" / "results.jsonl"
    rows = evaluation.read_rows(results)
    for r in rows:
        r["rerank_prob"] = 0.95 if r["id"] == "count" else (0.4 if r["id"] == "style" else 0.8)
    evaluation.write_rows(results, rows)
    c = bestof.calibrate(session)
    assert c.min_score == 0.5
    assert c.min_score_reason.startswith("overlap: ")


@pytest.mark.parametrize(
    ("ok_seeds", "phrase"),
    [((2000,), "every prompt had an acceptable candidate"), ((), "no prompt had an acceptable")],
    ids=["all-acceptable", "none-acceptable"],
)
def test_nothing_to_separate_keeps_the_default_min_score(
    session: Path, ok_seeds: tuple[int, ...], phrase: str
) -> None:
    judge_all(session, best_seed=2000, ok_seeds=ok_seeds)
    c = bestof.calibrate(session)
    assert c.min_score == 0.5
    assert phrase in c.min_score_reason


def test_calibrate_needs_scored_results(session: Path) -> None:
    judge_all(session)
    results = paths.eval_results_dir() / "2026-10-17_bo4" / "results.jsonl"
    rows = evaluation.read_rows(results)
    rows[0]["rerank_margin"] = None
    evaluation.write_rows(results, rows)
    with pytest.raises(SessionError, match="unscored rows"):
        bestof.calibrate(session)


def test_calibrate_refuses_a_paired_session() -> None:
    paired = sessions.create(
        [paired_variant("alpha")], [paired_variant("beta")], seed=7, day=date(2026, 10, 15)
    )
    with pytest.raises(SessionError, match="not a best-of-N session"):
        bestof.calibrate(paired)


def test_the_report_and_the_cli(session: Path) -> None:
    judge_all(session, best_seed=2000, ok_seeds=(2000, 2001))
    rescored("generic", lambda i, s: -s)
    result = runner.invoke(cli.app, ["eval", "calibrate", session.name])
    assert result.exit_code == 0, result.output
    assert "task: top-1 = best 0/12, top-1 acceptable 0/12" in result.output
    assert "generic: top-1 = best 12/12, top-1 acceptable 12/12" in result.output
    assert "no reranker (base seed): top-1 = best 12/12" in result.output
    assert "instruction: generic; --min-score 0.5" in result.output
    report = (session / "report.md").read_text()
    assert "## Instruction: **generic**" in report
    assert "| `generic` | 12/12 (100%) | 12/12 (100%) |" in report
    assert "| no reranker (base seed) | 12/12 (100%) | 12/12 (100%) |" in report
    assert "| chance (random pick) | 3/12 (25%) | 6/12 (50%) |" in report
    assert "## `--min-score`: **0.5**" in report
    assert report.count("\n| `") >= 12 + 2  # per-prompt rows plus the instruction rows


def test_cli_calibrate_reports_an_incomplete_session(session: Path) -> None:
    result = runner.invoke(cli.app, ["eval", "calibrate", session.name])
    assert result.exit_code == 1
    assert "12 prompts are not judged yet" in result.output
