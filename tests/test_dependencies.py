"""Dependency invariants the runtime relies on (PLAN.md §5.5, §8)."""

import importlib.util
import tomllib

from .conftest import ROOT


def test_torch_is_not_installed() -> None:
    # Installed, transformers imports it at runtime; the override in pyproject.toml keeps it out.
    assert importlib.util.find_spec("torch") is None
    assert importlib.util.find_spec("torchvision") is None


def test_mflux_is_pinned_exactly() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert "mflux==0.23.0" in project["dependencies"]
