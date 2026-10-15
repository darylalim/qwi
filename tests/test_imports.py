"""Heavy libraries load only inside `load()` / the UI entrypoint (CLAUDE.md "Imports")."""

import json
import subprocess
import sys

HEAVY = ["mlx", "mflux", "mlx_lm", "mlx_vlm", "streamlit", "torch"]
MODULES = [
    "qwi",
    "qwi.cli",
    "qwi.config",
    "qwi.errors",
    "qwi.memory",
    "qwi.models",
    "qwi.paths",
    "qwi.stages.generate",
]


def test_qwi_modules_import_no_heavy_libraries() -> None:
    # A fresh interpreter: this test process may already have imported mlx.
    code = (
        "import importlib, json, sys\n"
        f"for m in {MODULES!r}: importlib.import_module(m)\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    loaded = set(json.loads(out.stdout))
    assert loaded.isdisjoint(HEAVY), loaded & set(HEAVY)
