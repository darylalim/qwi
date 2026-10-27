"""`.gitignore` keeps runtime data, weights and images out, committed files in (PLAN.md §5.7)."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

MUST_IGNORE = [
    "var/runs/x/run.json",
    "var/.lock",
    "var/models/x.safetensors",
    "eval/results/d_v/text-en_1000.png",
    "var/eval/d_v/text-en_1000.png",
    "model.safetensors",
    "weights.gguf",
    "image.png",
    "docs/shot.png",
    "out.webp",
    ".claude/settings.local.json",
    ".streamlit/secrets.toml",
    ".env",
    ".venv/x",
    "__pycache__/x.pyc",
    ".ruff_cache/x",
    "dist/x.whl",
    ".DS_Store",
]

MUST_TRACK = [
    "src/qwi/paths.py",
    "models.toml",
    "PLAN.md",
    "eval/results/d_v/review.csv",
    "eval/results/d_v/results.jsonl",
    "eval/sessions/s/manifest.json",
    "eval/sessions/s/report.md",
    "tests/fixtures/rerank/red_cube.png",
    "docs/screenshots/generate.png",
    ".claude/settings.json",
    ".streamlit/config.toml",
    "uv.lock",
    ".python-version",
    "LICENSE",
    "NOTICE",
    "CLAUDE.md",
]


def _ignored(path: str) -> bool:
    # --no-index: judge the rules alone, independent of what's tracked.
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path], cwd=ROOT, check=False
    )
    assert result.returncode in (0, 1), f"git check-ignore failed for {path}"
    return result.returncode == 0


@pytest.mark.parametrize("path", MUST_IGNORE)
def test_ignored(path: str) -> None:
    assert _ignored(path)


@pytest.mark.parametrize("path", MUST_TRACK)
def test_tracked(path: str) -> None:
    assert not _ignored(path)
