"""Storage checks and the one download path (PLAN.md §5.7): `qwi doctor`, `qwi models fetch`.

`doctor()` is read-only and returns pass/warn/fail rows. `startup()` is the cheap subset that
`Pipeline.run` calls before anything loads, raising instead of reporting (§5.11).
"""

import importlib.util
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from qwi import logs, models, paths
from qwi.config import DEFAULTS
from qwi.errors import InsufficientDisk, ModelUnavailable
from qwi.record import RUN_JSON, RunRecord

RUNS_WARN_BYTES = 2 * 10**9  # §5.7 "Retention"
INDEX_TRIGGER_RUNS = 2000  # §5.7 "Database": past either, add the SQLite index
INDEX_TRIGGER_SECONDS = 1.0

Status = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def tree_bytes(root: Path) -> int:
    """Disk used under `root`, counting hard-linked files once (`best.png`, §5.7)."""
    seen: set[tuple[int, int]] = set()
    total = 0
    for path in root.rglob("*") if root.exists() else ():
        st = path.lstat()
        if path.is_file() and (st.st_dev, st.st_ino) not in seen:
            seen.add((st.st_dev, st.st_ino))
            total += st.st_size
    return total


def list_runs(runs: Path | None = None) -> list[RunRecord]:
    """Every readable `run.json`, newest first; logs the scan time (§5.7 "Measure, don't guess")."""
    runs = runs or paths.runs_dir()
    started = time.perf_counter()
    records: list[RunRecord] = []
    for path in sorted(runs.glob(f"*/{RUN_JSON}"), reverse=True):
        try:
            records.append(RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, ValueError, TypeError):
            continue
    seconds = time.perf_counter() - started
    logs.event(logging.DEBUG, None, "runs_scan", "scanned runs", runs=len(records), seconds=seconds)
    return records


def check_disk() -> Check:
    free = paths.free_bytes()
    ok = free >= DEFAULTS.min_free_disk_bytes
    rule = f"{_gb(free)} free (needs ≥ {_gb(DEFAULTS.min_free_disk_bytes)})"
    return Check("disk", "pass" if ok else "fail", rule)


def check_models() -> list[Check]:
    checks = []
    for role, spec in models.load().items():
        try:
            models.resolve(role)
        except ModelUnavailable:
            checks.append(Check(f"model {role}", "fail", f"missing: run `qwi models fetch {role}`"))
        else:
            checks.append(Check(f"model {role}", "pass", spec.pinned))
    return checks


def check_home() -> Check:
    home = paths.home()
    where = paths.existing_ancestor(home)
    ok = os.access(where, os.W_OK)
    detail = str(home) if where == home else f"{home} (will be created in {where})"
    return Check("QWI_HOME writable", "pass" if ok else "fail", detail)


def check_runs_size() -> Check:
    size = tree_bytes(paths.runs_dir())
    detail = _gb(size) + (" (over 2 GB: `qwi runs prune`)" if size > RUNS_WARN_BYTES else "")
    return Check("runs size", "warn" if size > RUNS_WARN_BYTES else "pass", detail)


def check_eval_size() -> Check:
    return Check("eval images size", "pass", _gb(tree_bytes(paths.eval_dir())))


def check_run_index() -> Check:
    started = time.perf_counter()
    count = len(list_runs())
    seconds = time.perf_counter() - started
    slow = count > INDEX_TRIGGER_RUNS or seconds > INDEX_TRIGGER_SECONDS
    detail = f"{count} runs, scanned in {seconds:.2f} s"
    if slow:
        detail += " (past the index trigger: add the SQLite index, §5.7)"
    return Check("run history", "warn" if slow else "pass", detail)


def check_offline() -> Check:
    from huggingface_hub import constants

    if constants.HF_HUB_OFFLINE:
        return Check("HF_HUB_OFFLINE", "pass", "on: runs never download")
    return Check("HF_HUB_OFFLINE", "fail", "off: huggingface_hub was imported before qwi")


def check_torch() -> Check:
    if "torch" in sys.modules:
        return Check("torch", "fail", "imported: something on the runtime path pulls it in")
    if importlib.util.find_spec("torch") is not None:
        return Check("torch", "warn", "installed (transformers will import it): `uv sync`")
    return Check("torch", "pass", "not installed, not imported")


def doctor() -> list[Check]:
    return [
        check_disk(),
        *check_models(),
        check_home(),
        check_runs_size(),
        check_eval_size(),
        check_run_index(),
        check_offline(),
        check_torch(),
    ]


def startup(roles: list[str]) -> None:
    """Fail fast before a run loads anything: disk rule, `$QWI_HOME` writable, models cached."""
    free = paths.free_bytes()
    if free < DEFAULTS.min_free_disk_bytes:
        raise InsufficientDisk("a run", 0, free, DEFAULTS.min_free_disk_bytes)
    if check_home().status == "fail":
        raise PermissionError(f"$QWI_HOME is not writable: {paths.home()}")
    for role in roles:
        models.resolve(role)


def fetch(role: str, *, dry_run: bool = False) -> Path | None:
    """Download `role` at its pinned revision if the 20 GB rule allows; None for a dry run.

    Already-cached models return at once, without touching the network.
    """
    from huggingface_hub import constants, snapshot_download

    spec = models.spec(role)
    try:
        return models.resolve(role)
    except ModelUnavailable:
        pass
    with models.online():
        need = models.remote_bytes(spec)
        free = paths.free_bytes(Path(constants.HF_HUB_CACHE))
        logs.event(logging.INFO, None, "fetch", "checked", role=role, need=need, free=free)
        if free - need < DEFAULTS.min_free_disk_bytes:
            raise InsufficientDisk(spec.pinned, need, free, DEFAULTS.min_free_disk_bytes)
        if dry_run:
            return None
        path = snapshot_download(
            spec.repo,
            revision=spec.revision,
            allow_patterns=list(spec.allow_patterns) if spec.allow_patterns else None,
        )
    logs.event(logging.INFO, None, "fetch", "downloaded", role=role, path=path)
    return Path(path)
