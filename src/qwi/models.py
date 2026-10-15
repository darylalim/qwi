"""Model roles from `models.toml`, pinned to exact revisions (PLAN.md §5.7).

`resolve()` only reads the local HF cache; the runtime never downloads (`HF_HUB_OFFLINE=1`).
`online()` lifts offline mode for `qwi models fetch` alone, the only sanctioned download path.
"""

import tomllib
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from qwi import paths
from qwi.errors import ModelUnavailable


@dataclass(frozen=True)
class ModelSpec:
    role: str
    repo: str
    revision: str
    license: str
    allow_patterns: tuple[str, ...] | None = None  # e.g. ("text_encoder/*",); None = whole repo

    @property
    def pinned(self) -> str:
        """`repo@revision`, the form `run.json` records (§5.7)."""
        return f"{self.repo}@{self.revision}"


def load(path: Path = paths.MODELS_TOML) -> dict[str, ModelSpec]:
    roles = tomllib.loads(path.read_text())["roles"]
    return {
        role: ModelSpec(
            role,
            entry["repo"],
            entry["revision"],
            entry["license"],
            tuple(entry["allow_patterns"]) if "allow_patterns" in entry else None,
        )
        for role, entry in roles.items()
    }


def spec(role: str) -> ModelSpec:
    specs = load()
    if role not in specs:
        raise ModelUnavailable(role, f"no [roles.{role}] entry in models.toml")
    return specs[role]


def resolve(role: str) -> Path:
    """The local snapshot for `role` at its pinned revision, or `ModelUnavailable`."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    model = spec(role)
    try:
        path = snapshot_download(
            model.repo,
            revision=model.revision,
            allow_patterns=list(model.allow_patterns) if model.allow_patterns else None,
            local_files_only=True,
        )
        return Path(path)
    except LocalEntryNotFoundError as exc:
        raise ModelUnavailable(role, f"{model.pinned} is not in the local HF cache") from exc


@contextmanager
def online() -> Generator[None]:
    """Allow Hub requests inside the block; offline mode comes back even if it raises.

    huggingface_hub reads `HF_HUB_OFFLINE` once at import (`qwi/__init__.py` forces it on), but
    every request checks `constants.is_offline_mode()`, which reads this module global.
    """
    from huggingface_hub import constants

    before = constants.HF_HUB_OFFLINE
    constants.HF_HUB_OFFLINE = False
    try:
        yield
    finally:
        constants.HF_HUB_OFFLINE = before


def remote_bytes(model: ModelSpec) -> int:
    """Download size of `model` at its pinned revision (call inside `online()`)."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import filter_repo_objects

    info = HfApi().model_info(model.repo, revision=model.revision, files_metadata=True)
    files = filter_repo_objects(
        info.siblings or [],
        allow_patterns=list(model.allow_patterns) if model.allow_patterns else None,
        key=lambda f: f.rfilename,
    )
    return sum(f.size or 0 for f in files)
