"""Doctor checks, startup checks and `models fetch`, with the disk, cache and Hub mocked."""

import json
import shutil
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from qwi import models, paths, record, storage
from qwi.errors import InsufficientDisk, ModelUnavailable, PipelineBusy
from qwi.lock import RunLock
from qwi.record import RunRecord

GB = 10**9


def disk(free: int) -> Callable[[Any], Any]:
    return lambda _: SimpleNamespace(total=500 * GB, used=500 * GB - free, free=free)


@pytest.fixture
def cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> set[str]:
    """Roles in this set resolve to a fake snapshot; others are missing from the cache."""
    roles = set(models.load())

    def resolve(role: str) -> Path:
        if role not in roles:
            raise ModelUnavailable(role, "not in the local HF cache")
        return tmp_path / role

    monkeypatch.setattr(models, "resolve", resolve)
    return roles


def by_name(checks: list[storage.Check]) -> dict[str, storage.Check]:
    return {c.name: c for c in checks}


@pytest.mark.parametrize(("free", "status"), [(20 * GB, "pass"), (20 * GB - 1, "fail")])
def test_disk_rule(monkeypatch: pytest.MonkeyPatch, free: int, status: str) -> None:
    monkeypatch.setattr(shutil, "disk_usage", disk(free))
    assert storage.check_disk().status == status


def test_models_report_each_role(cached: set[str]) -> None:
    cached.discard("rerank")
    checks = by_name(storage.check_models())
    assert checks["model rerank"].status == "fail"
    assert "qwi models fetch rerank" in checks["model rerank"].detail
    assert checks["model image"].status == "pass"
    assert checks["model image"].detail == models.spec("image").pinned


def test_home_writable_before_it_exists(qwi_home: Path) -> None:
    check = storage.check_home()
    assert check.status == "pass"
    assert "will be created" in check.detail


def test_home_not_writable(qwi_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage.os, "access", lambda *_: False)
    assert storage.check_home().status == "fail"


def test_tree_bytes_counts_hard_links_once(tmp_path: Path) -> None:
    tmp_path = tmp_path / "run"  # its own folder: tmp_path also holds the eval_set fixture
    tmp_path.mkdir()
    candidate = tmp_path / "cand_0_seed0.png"
    candidate.write_bytes(b"x" * 1000)
    paths.link_best(candidate)
    (tmp_path / "run.json").write_bytes(b"y" * 10)
    assert storage.tree_bytes(tmp_path) == 1010
    assert storage.tree_bytes(tmp_path / "missing") == 0


def test_runs_size_warns_over_2_gb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "tree_bytes", lambda _: 2 * GB + 1)
    assert storage.check_runs_size().status == "warn"
    monkeypatch.setattr(storage, "tree_bytes", lambda _: 2 * GB)
    assert storage.check_runs_size().status == "pass"


def _write_runs(count: int) -> None:
    for i in range(count):
        run = paths.runs_dir() / f"20261015-{i:06d}-abcdef"
        run.mkdir(parents=True)
        (run / "run.json").write_text(json.dumps({"user_prompt": f"p{i}", "status": "ok"}))


def test_list_runs_is_newest_first_and_skips_bad_records() -> None:
    _write_runs(3)
    bad = paths.runs_dir() / "20261015-999999-badbad"
    bad.mkdir()
    (bad / "run.json").write_text("{half")
    records = storage.list_runs()
    assert [r.user_prompt for r in records] == ["p2", "p1", "p0"]
    assert all(isinstance(r, RunRecord) for r in records)


def test_run_history_warns_past_2000_runs() -> None:
    _write_runs(storage.INDEX_TRIGGER_RUNS + 1)
    check = storage.check_run_index()
    assert check.status == "warn"
    assert "SQLite index" in check.detail


def test_run_history_warns_on_a_slow_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter([0.0, 1.5])
    monkeypatch.setattr(storage.time, "perf_counter", lambda: next(ticks, 1.5))
    assert storage.check_run_index().status == "warn"


def test_offline_and_torch_checks_on_this_install() -> None:
    assert storage.check_offline().status == "pass"
    assert storage.check_torch().status == "pass"


def test_offline_check_fails_when_offline_mode_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    assert storage.check_offline().status == "fail"


def test_doctor_runs_every_check(cached: set[str]) -> None:
    names = [c.name for c in storage.doctor()]
    assert names[0] == "disk"
    assert {"QWI_HOME writable", "runs size", "eval images size", "run history"} <= set(names)
    assert {"HF_HUB_OFFLINE", "torch"} <= set(names)
    assert sum(name.startswith("model ") for name in names) == len(models.load())


def test_startup_refuses_below_the_disk_rule(
    monkeypatch: pytest.MonkeyPatch, cached: set[str]
) -> None:
    monkeypatch.setattr(shutil, "disk_usage", disk(19 * GB))
    with pytest.raises(InsufficientDisk):
        storage.startup(["image"])


def test_startup_raises_model_unavailable(
    monkeypatch: pytest.MonkeyPatch, cached: set[str]
) -> None:
    monkeypatch.setattr(shutil, "disk_usage", disk(40 * GB))
    cached.discard("image")
    with pytest.raises(ModelUnavailable):
        storage.startup(["image"])
    storage.startup(["guard"])


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """A fake Hub: `remote_bytes` returns `size`; downloads are recorded with the offline flag."""
    import huggingface_hub
    from huggingface_hub import constants

    state: dict[str, Any] = {"size": 10 * GB, "downloads": [], "offline_seen": None}
    monkeypatch.setattr(models, "remote_bytes", lambda _: state["size"])

    def download(repo_id: str, *, revision: str, allow_patterns: list[str] | None) -> str:
        state["offline_seen"] = constants.HF_HUB_OFFLINE
        state["downloads"].append((repo_id, revision, allow_patterns))
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    return state


def test_fetch_returns_a_cached_model_without_the_network(
    cached: set[str], hub: dict[str, Any]
) -> None:
    path = storage.fetch("image")
    assert path is not None
    assert path.name == "image"
    assert hub["downloads"] == []


def test_fetch_downloads_online_at_the_pinned_revision(
    monkeypatch: pytest.MonkeyPatch, cached: set[str], hub: dict[str, Any]
) -> None:
    from huggingface_hub import constants

    cached.discard("rerank")
    monkeypatch.setattr(shutil, "disk_usage", disk(40 * GB))
    spec = models.spec("rerank")
    assert storage.fetch("rerank") is not None
    assert hub["downloads"] == [(spec.repo, spec.revision, None)]
    assert hub["offline_seen"] is False
    assert constants.HF_HUB_OFFLINE is True  # restored after the download


def test_fetch_refuses_below_the_threshold(
    monkeypatch: pytest.MonkeyPatch, cached: set[str], hub: dict[str, Any]
) -> None:
    from huggingface_hub import constants

    cached.discard("rerank")
    monkeypatch.setattr(shutil, "disk_usage", disk(29 * GB))  # 29 - 10 < 20
    with pytest.raises(InsufficientDisk, match="20 GB must stay free"):
        storage.fetch("rerank")
    assert hub["downloads"] == []
    assert constants.HF_HUB_OFFLINE is True


def test_fetch_dry_run_checks_but_downloads_nothing(
    monkeypatch: pytest.MonkeyPatch, cached: set[str], hub: dict[str, Any]
) -> None:
    cached.discard("rerank")
    monkeypatch.setattr(shutil, "disk_usage", disk(40 * GB))
    assert storage.fetch("rerank", dry_run=True) is None
    assert hub["downloads"] == []


def test_fetch_passes_allow_patterns(monkeypatch: pytest.MonkeyPatch, hub: dict[str, Any]) -> None:
    spec = models.ModelSpec(
        "encoder-bf16", "Qwen/Qwen-Image-2.1", "d" * 40, "qwen-research", ("text_encoder/*",)
    )

    def missing(role: str) -> Path:
        raise ModelUnavailable(role, "not cached")

    monkeypatch.setattr(models, "spec", lambda _: spec)
    monkeypatch.setattr(models, "resolve", missing)
    monkeypatch.setattr(shutil, "disk_usage", disk(60 * GB))
    storage.fetch("encoder-bf16")
    assert hub["downloads"] == [(spec.repo, spec.revision, ["text_encoder/*"])]


def test_allow_patterns_load_from_models_toml(tmp_path: Path) -> None:
    toml = tmp_path / "models.toml"
    toml.write_text(
        '[roles.encoder-bf16]\nrepo = "Qwen/Qwen-Image-2.1"\nrevision = "' + "d" * 40 + '"\n'
        'license = "qwen-research"\nallow_patterns = ["text_encoder/*"]\n'
    )
    assert models.load(toml)["encoder-bf16"].allow_patterns == ("text_encoder/*",)
    assert models.load()["image"].allow_patterns is None


NOW = datetime(2026, 10, 19, 12, 0).astimezone()


def _run(name: str, days_ago: float, *, starred: bool = False, status: str = "ok") -> Path:
    run = paths.runs_dir() / name
    run.mkdir(parents=True)
    (run / "cand_0_seed7.png").write_bytes(b"x" * 100)
    started = (NOW - timedelta(days=days_ago)).isoformat(timespec="seconds")
    rec = RunRecord("a red apple", status=status, starred=starred, started_at=started)
    record.write(rec, run)
    return run


def _names() -> list[str]:
    return sorted(p.name for p in paths.runs_dir().iterdir())


def test_prune_keep_days_spares_recent_and_starred_runs() -> None:
    _run("20261001-090000-aaaaaa", 18)
    _run("20261005-090000-bbbbbb", 14, starred=True)
    _run("20261018-090000-cccccc", 1)
    result = storage.prune(keep_days=7, now=NOW)
    [(name, size)] = result.removed
    assert (name, size > 100) == ("20261001-090000-aaaaaa", True)  # the image and run.json
    assert result.kept_starred == ["20261005-090000-bbbbbb"]
    assert _names() == ["20261005-090000-bbbbbb", "20261018-090000-cccccc"]


def test_prune_keep_last_counts_starred_runs_toward_n() -> None:
    _run("20261001-090000-aaaaaa", 18, starred=True)
    _run("20261005-090000-bbbbbb", 14)
    _run("20261018-090000-cccccc", 1, starred=True)
    result = storage.prune(keep_last=1, now=NOW)
    assert [name for name, _ in result.removed] == ["20261005-090000-bbbbbb"]
    assert result.kept_starred == ["20261001-090000-aaaaaa"]
    assert _names() == ["20261001-090000-aaaaaa", "20261018-090000-cccccc"]


def test_prune_keep_last_0_removes_everything_unstarred() -> None:
    _run("20261001-090000-aaaaaa", 18)
    _run("20261018-090000-cccccc", 1, starred=True)
    storage.prune(keep_last=0, now=NOW)
    assert _names() == ["20261018-090000-cccccc"]


def test_prune_dry_run_deletes_nothing() -> None:
    _run("20261001-090000-aaaaaa", 18)
    result = storage.prune(keep_days=7, dry_run=True, now=NOW)
    assert [name for name, _ in result.removed] == ["20261001-090000-aaaaaa"]
    assert _names() == ["20261001-090000-aaaaaa"]


def test_prune_never_touches_unreadable_run_folders() -> None:
    broken = paths.runs_dir() / "20261001-090000-broken"
    broken.mkdir(parents=True)
    (broken / "run.json").write_text("{not json", encoding="utf-8")
    storage.prune(keep_last=0, now=NOW)
    assert broken.exists()


def test_prune_refuses_while_a_run_holds_the_lock() -> None:
    _run("20261001-090000-aaaaaa", 18)
    with RunLock(), pytest.raises(PipelineBusy):
        storage.prune(keep_last=0, now=NOW)
    assert _names() == ["20261001-090000-aaaaaa"]


def test_prune_removes_a_dead_running_run_by_age() -> None:
    _run("20261001-090000-aaaaaa", 18, status="running")
    storage.prune(keep_days=7, now=NOW)
    assert _names() == []


def test_prune_eval_removes_old_image_folders_only() -> None:
    for name in ["2026-10-01_baseline", "2026-10-18_bo4", "notes"]:
        (paths.eval_dir() / name).mkdir(parents=True)
        (paths.eval_dir() / name / "p01_1000.png").write_bytes(b"x")
    _run("20261001-090000-aaaaaa", 18)
    result = storage.prune(keep_days=7, eval_images=True, now=NOW)
    assert [name for name, _ in result.removed] == ["2026-10-01_baseline"]
    assert sorted(p.name for p in paths.eval_dir().iterdir()) == ["2026-10-18_bo4", "notes"]
    assert _names() == ["20261001-090000-aaaaaa"]  # runs untouched


def test_prune_eval_keep_last_orders_by_date() -> None:
    for name in ["2026-10-18_a", "2026-10-01_z", "2026-10-15_m"]:
        (paths.eval_dir() / name).mkdir(parents=True)
    result = storage.prune(keep_last=2, eval_images=True, now=NOW)
    assert [name for name, _ in result.removed] == ["2026-10-01_z"]


def test_prune_targets_needs_exactly_one_rule() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        storage.prune_targets([], keep_days=1, keep_last=1, now=NOW)
    with pytest.raises(ValueError, match="exactly one"):
        storage.prune_targets([], keep_days=None, keep_last=None, now=NOW)
