import re
from pathlib import Path

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from qwi import models
from qwi.errors import ModelUnavailable


def test_every_role_is_pinned_to_a_full_commit() -> None:
    specs = models.load()
    assert {"guard", "rewrite", "rerank", "image"} <= specs.keys()
    for role, spec in specs.items():
        assert spec.role == role
        assert re.fullmatch(r"[0-9a-f]{40}", spec.revision), role


def test_pinned_is_repo_at_revision() -> None:
    image = models.spec("image")
    assert image.pinned == f"mlx-community/Qwen-Image-2.1-mflux-q4@{image.revision}"


def test_unknown_role_is_unavailable() -> None:
    with pytest.raises(ModelUnavailable, match="qwi models fetch encoder-bf16"):
        models.spec("encoder-bf16")


def test_resolve_reads_the_cache_offline_at_the_pinned_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import huggingface_hub

    calls: list[tuple[str, str, list[str] | None, bool]] = []

    def fake(
        repo_id: str, *, revision: str, allow_patterns: list[str] | None, local_files_only: bool
    ) -> str:
        calls.append((repo_id, revision, allow_patterns, local_files_only))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    guard = models.spec("guard")
    assert models.resolve("guard") == tmp_path
    assert calls == [(guard.repo, guard.revision, None, True)]


def test_resolve_raises_model_unavailable_when_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    def missing(*_: object, **__: object) -> str:
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    with pytest.raises(ModelUnavailable, match="not in the local HF cache") as excinfo:
        models.resolve("rerank")
    assert excinfo.value.role == "rerank"
