"""Blind paired review (`qwi eval review`, PLAN.md §5.4/§5.6): data layer and launcher.

The page itself is tested in `test_eval_review_page.py`.
"""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from typer.testing import CliRunner

from qwi import cli, paths, sessions
from qwi.errors import SessionError
from qwi.evaluation import load_prompts
from qwi.sessions import Judgment

APP = str(Path(cli.__file__).parent / "ui" / "streamlit_app.py")
DAY = date(2026, 10, 15)
COLORS = {"alpha": "red", "beta": "blue", "gamma": "green", "delta": "white"}


def variant(name: str) -> Path:
    """Results + tiny images for every eval prompt at seed 1000, like `qwi eval` writes them."""
    folder = paths.eval_results_dir() / f"2026-10-15_{name}"
    images = paths.eval_dir() / folder.name
    folder.mkdir(parents=True)
    images.mkdir(parents=True)
    with (folder / "results.jsonl").open("w") as f:
        for p in load_prompts():
            image = f"{p.id}_1000.png"
            Image.new("RGB", (8, 8), COLORS[name]).save(images / image)
            row = {
                "id": p.id,
                "seed": 1000,
                "variant": name,
                "prompt": p.prompt,
                "tags": list(p.tags),
                "rewrite": f"{name} rewrite of {p.id}",
                "image": image,
                "rerank_prob": None,
                "seconds": {"generate": 120.0, "encode": 4.04},
                "peak_gb": {"generate": 14.0},
                "swap_gb": 0.0,
            }
            f.write(json.dumps(row) + "\n")
    return folder


@pytest.fixture
def session() -> Path:
    return sessions.create([variant("alpha")], [variant("beta")], seed=7, day=DAY)


def color(path: Path) -> str:
    rgb = Image.open(path).getpixel((0, 0))
    assert isinstance(rgb, tuple)
    return {"(255, 0, 0)": "red", "(0, 0, 255)": "blue"}[str(rgb)]


def full(item: sessions.Item, left: str = "pass", right: str = "fail") -> Judgment:
    return Judgment(dict.fromkeys(item.checks, left), dict.fromkeys(item.checks, right))


# --- Data layer ----------------------------------------------------------------------------


def test_open_session_maps_each_side_to_its_variant(session: Path) -> None:
    review = sessions.open_session(session)
    manifest = json.loads((session / "manifest.json").read_text())
    assert [i.item for i in review.items] == [i["item"] for i in manifest["items"]]
    for item in review.items:
        for side in sessions.SIDES:
            shown = review.sides[(item.item, side)]
            name = "alpha" if item.variant_of(side) == "a" else "beta"
            assert color(shown.image) == COLORS[name]
            assert shown.hints["rewrite"] == f"{name} rewrite of {item.prompt_id}"
            assert shown.hints["seconds"] == 124.0
            assert "rerank_prob" not in shown.hints  # None values are left out
    assert {item.left for item in review.items} == {"a", "b"}  # sides really are randomized


def test_a_new_session_has_nothing_judged(session: Path) -> None:
    review = sessions.open_session(session)
    judgments = sessions.read_judgments(session)
    assert not any(judgments[i.item].complete(i.checks) for i in review.items)


def test_save_judgment_round_trips_and_touches_only_that_item(session: Path) -> None:
    first, second = sessions.open_session(session).items[:2]
    before = sessions.read_judgments(session)[second.item]
    saved = Judgment(
        dict.fromkeys(first.checks, "pass"),
        dict.fromkeys(first.checks, "fail"),
        "left",
        "the left one, clearly",
    )
    sessions.save_judgment(session, first, saved)
    after = sessions.read_judgments(session)
    assert after[first.item] == saved
    assert after[first.item].complete(first.checks)
    assert after[second.item] == before
    assert not (session / "review.csv.tmp").exists()


def test_a_fully_saved_session_can_be_compared(session: Path) -> None:
    for item in sessions.open_session(session).items:
        sessions.save_judgment(session, item, full(item, "pass", "pass"))
    assert sessions.compare(session).overall.ties > 0


@pytest.mark.parametrize(
    "bad",
    [
        Judgment({"x": "maybe"}, {}),
        Judgment({}, {}, prefer="both"),
    ],
)
def test_save_judgment_rejects_bad_values(session: Path, bad: Judgment) -> None:
    item = sessions.open_session(session).items[0]
    with pytest.raises(SessionError, match="pass/fail"):
        sessions.save_judgment(session, item, bad)


def test_save_judgment_rejects_an_unknown_item(session: Path) -> None:
    item = sessions.open_session(session).items[0]
    stranger = sessions.Item("i999", item.prompt_id, 1000, item.tags, item.checks, "a")
    with pytest.raises(SessionError, match="i999"):
        sessions.save_judgment(session, stranger, full(stranger))


def test_session_dir_accepts_a_name_or_a_path(session: Path) -> None:
    assert sessions.session_dir(session.name) == session
    assert sessions.session_dir(str(session)) == session


# --- Launcher ------------------------------------------------------------------------------


def test_cli_launches_streamlit_on_the_review_port(
    session: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def call(command: list[str], *, env: dict[str, str], cwd: Path) -> int:
        calls.append({"command": command, "env": env, "cwd": cwd})
        return 0

    monkeypatch.setattr(cli.subprocess, "call", call)
    result = CliRunner().invoke(cli.app, ["eval", "review", session.name])
    assert result.exit_code == 0
    (launched,) = calls
    assert launched["command"][1:4] == ["-m", "streamlit", "run"]
    assert launched["command"][4] == APP
    assert launched["command"][-2:] == ["--server.port", "8502"]
    assert launched["env"][sessions.REVIEW_SESSION_ENV] == str(session.resolve())
    assert launched["cwd"] == paths.REPO  # where .streamlit/config.toml lives
    assert "http://localhost:8502" in result.output


def test_cli_refuses_a_missing_session_before_launching(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.subprocess, "call", lambda *a, **k: pytest.fail("launched"))
    result = CliRunner().invoke(cli.app, ["eval", "review", "no-such-session"])
    assert result.exit_code == 1
    assert "no manifest.json" in result.output
