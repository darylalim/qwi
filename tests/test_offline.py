"""The runtime is offline: importing qwi forces HF_HUB_OFFLINE=1 (PLAN.md §5.7)."""

import os
import subprocess
import sys


def test_importing_qwi_forces_hub_offline() -> None:
    code = (
        "import qwi, huggingface_hub.constants as c, os; "
        "print(os.environ['HF_HUB_OFFLINE'], c.HF_HUB_OFFLINE)"
    )
    env = {**os.environ, "HF_HUB_OFFLINE": "0"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["1", "True"]
