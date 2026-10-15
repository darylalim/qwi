"""Every on-disk location goes through this module (PLAN.md §5.7 "Where things live").

`$QWI_HOME` is the storage root (default `<repo>/var/`). It's read on every call, so tests can
set `QWI_HOME=<tmp_path>` and the CLI and UI always share the same lock.
"""

import os
import secrets
import shutil
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS_TOML = REPO / "models.toml"
GUARD_POLICY = REPO / "guard_policy.toml"  # §5.2: Controversial actions per --guard-mode
PROMPTS = REPO / "prompts"  # §5.1: rewriter system prompts, one file per --rewrite-style
EVAL_SET = REPO / "eval"  # committed: prompts.jsonl, results/, sessions/ (§5.4, §5.7)


def home() -> Path:
    return Path(os.environ.get("QWI_HOME") or REPO / "var")


def runs_dir() -> Path:
    return home() / "runs"


def lock_path() -> Path:
    return home() / ".lock"


def models_dir() -> Path:
    return home() / "models"


def eval_dir() -> Path:
    return home() / "eval"


def logs_dir() -> Path:
    return home() / "logs"


def eval_results_dir() -> Path:
    """Committed per-variant `results.jsonl` + `review.csv`; the images live in `eval_dir()`."""
    return EVAL_SET / "results"


def new_run_dir(now: datetime | None = None) -> Path:
    """Create `runs/<YYYYmmdd-HHMMSS>-<short-id>/`; the id keeps same-second runs apart."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    run = runs_dir() / f"{stamp}-{secrets.token_hex(3)}"
    run.mkdir(parents=True)
    return run


def candidate_name(index: int, seed: int) -> str:
    return f"cand_{index}_seed{seed}.png"


def link_best(candidate: Path) -> Path:
    """Point `best.png` at the winning candidate: a hard link, or a copy if linking fails."""
    best = candidate.with_name("best.png")
    best.unlink(missing_ok=True)
    try:
        best.hardlink_to(candidate)
    except OSError:
        shutil.copyfile(candidate, best)
    return best


def relative(path: Path, run: Path) -> str:
    """The form `run.json` stores, so `$QWI_HOME` can be moved (§5.7 "Run files")."""
    return path.relative_to(run).as_posix()


def free_bytes(target: Path | None = None) -> int:
    """Free space on the volume holding `target` (default `$QWI_HOME`), even before it exists."""
    return shutil.disk_usage(existing_ancestor(target or home())).free


def existing_ancestor(path: Path) -> Path:
    """`path` itself, or its nearest parent that exists (`$QWI_HOME` is created lazily)."""
    while not path.exists():
        path = path.parent
    return path
