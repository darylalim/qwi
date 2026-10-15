"""Model roles from `models.toml`, pinned to exact revisions (PLAN.md §5.7).

`resolve()` only reads the local HF cache; the runtime never downloads (`HF_HUB_OFFLINE=1`).
"""

import tomllib
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

    @property
    def pinned(self) -> str:
        """`repo@revision`, the form `run.json` records (§5.7)."""
        return f"{self.repo}@{self.revision}"


def load(path: Path = paths.MODELS_TOML) -> dict[str, ModelSpec]:
    roles = tomllib.loads(path.read_text())["roles"]
    return {
        role: ModelSpec(role, entry["repo"], entry["revision"], entry["license"])
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
        return Path(snapshot_download(model.repo, revision=model.revision, local_files_only=True))
    except LocalEntryNotFoundError as exc:
        raise ModelUnavailable(role, f"{model.pinned} is not in the local HF cache") from exc
