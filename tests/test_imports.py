"""Heavy libraries load only inside `load()` / the UI entrypoint (CLAUDE.md "Imports")."""

import json
import pkgutil
import subprocess
import sys

HEAVY = ["mlx", "mflux", "mlx_lm", "mlx_vlm", "streamlit", "torch"]
UI = "qwi.ui"


STREAMLIT_HELPERS = {"components"}  # shared `st` code the pages import (§5.6)


def is_page(module: str) -> bool:
    """Streamlit scripts (`*_app`, `app_pages/`) and helpers may import streamlit; nothing else."""
    name = module.removeprefix(f"{UI}.")
    return module.startswith(f"{UI}.") and (
        name.endswith("_app") or name.startswith("app_pages") or name in STREAMLIT_HELPERS
    )


def qwi_modules() -> list[str]:
    import qwi

    found = ["qwi"]
    found += [m.name for m in pkgutil.walk_packages(qwi.__path__, "qwi.")]
    return [m for m in found if not is_page(m)]


def test_finds_every_module() -> None:
    found = set(qwi_modules())
    assert {
        "qwi.memory",
        "qwi.record",
        "qwi.stages.generate",
        "qwi.ui.jobs",
        "qwi.ui.view",
    } <= found
    assert "qwi.ui.review" in found
    assert "qwi.ui.app_pages.eval_review" not in found


def test_qwi_modules_import_no_heavy_libraries() -> None:
    # A fresh interpreter: this test process may already have imported mlx.
    code = (
        "import importlib, json, sys\n"
        f"for m in {qwi_modules()!r}: importlib.import_module(m)\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    loaded = set(json.loads(out.stdout))
    assert loaded.isdisjoint(HEAVY), loaded & set(HEAVY)
