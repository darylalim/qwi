import re
from datetime import datetime
from pathlib import Path

import pytest

from qwi import paths


def test_paths_honor_qwi_home(qwi_home: Path) -> None:
    assert paths.home() == qwi_home
    assert paths.runs_dir() == qwi_home / "runs"
    assert paths.lock_path() == qwi_home / ".lock"
    assert paths.models_dir() == qwi_home / "models"
    assert paths.eval_dir() == qwi_home / "eval"
    assert paths.logs_dir() == qwi_home / "logs"


def test_home_defaults_to_repo_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QWI_HOME")
    assert paths.home() == paths.REPO / "var"
    assert (paths.REPO / "pyproject.toml").is_file()


def test_new_run_dir_is_timestamped_and_unique() -> None:
    now = datetime(2026, 10, 15, 9, 30, 5)
    first, second = paths.new_run_dir(now), paths.new_run_dir(now)
    assert first.parent == paths.runs_dir()
    assert re.fullmatch(r"20261015-093005-[0-9a-f]{6}", first.name)
    assert first != second
    assert first.is_dir()


def test_candidate_name() -> None:
    assert paths.candidate_name(2, 1001) == "cand_2_seed1001.png"


def test_best_is_a_hard_link(tmp_path: Path) -> None:
    candidate = tmp_path / paths.candidate_name(0, 7)
    candidate.write_bytes(b"png")
    best = paths.link_best(candidate)
    assert best == tmp_path / "best.png"
    assert best.samefile(candidate)


def test_best_replaces_a_previous_best(tmp_path: Path) -> None:
    old, new = tmp_path / "cand_0_seed0.png", tmp_path / "cand_1_seed1.png"
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    paths.link_best(old)
    assert paths.link_best(new).read_bytes() == b"new"


def test_best_falls_back_to_a_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_links(self: Path, target: Path) -> None:
        raise OSError("links not supported")

    monkeypatch.setattr(Path, "hardlink_to", no_links)
    candidate = tmp_path / "cand_0_seed0.png"
    candidate.write_bytes(b"png")
    best = paths.link_best(candidate)
    assert best.read_bytes() == b"png"
    assert not best.samefile(candidate)


def test_run_json_paths_are_relative(tmp_path: Path) -> None:
    run = tmp_path / "20261015-093005-abcdef"
    assert paths.relative(run / "cand_0_seed0.png", run) == "cand_0_seed0.png"
    with pytest.raises(ValueError, match="not in the subpath"):
        paths.relative(tmp_path / "elsewhere.png", run)


def test_free_bytes_works_before_qwi_home_exists(qwi_home: Path) -> None:
    assert not qwi_home.exists()
    assert paths.free_bytes() > 0
