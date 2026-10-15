"""The package metadata matches the README and the repo page (PLAN.md §5.9 "Repo metadata")."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def _readme_intro() -> str:
    paragraphs = (ROOT / "README.md").read_text().split("\n\n")
    body = [p.strip() for p in paragraphs if p.strip() and not p.lstrip().startswith("#")]
    return " ".join(body[0].split())


def test_description_matches_readme_intro() -> None:
    assert _project()["description"] == _readme_intro()


def test_keywords_are_valid_github_topics() -> None:
    keywords = _project()["keywords"]
    assert 0 < len(keywords) <= 20
    assert len(set(keywords)) == len(keywords)
    for keyword in keywords:
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", keyword), keyword
