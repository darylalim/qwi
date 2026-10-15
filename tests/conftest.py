"""Shared test setup."""

import importlib.util
import os
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path) -> ModuleType:
    """Import a standalone script (hooks, CI scripts) by file path."""
    spec = importlib.util.spec_from_file_location(f"script_{path.stem}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pytest_configure() -> None:
    # Hosted macOS runners have no Metal GPU (PLAN.md §5.9): keep any incidental MLX op on CPU.
    if os.environ.get("CI") == "true":
        try:
            import mlx.core as mx
        except ImportError:
            return
        mx.set_default_device(mx.cpu)
