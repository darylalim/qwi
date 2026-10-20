"""Console noise (PLAN.md §9): the two transformers warnings and mflux's tqdm bar stay quiet."""

import logging
import os
import subprocess
import sys
from types import ModuleType

import pytest

from qwi import quiet

TORCH_ADVICE = "PyTorch was not found"


def import_transformers_stderr(code: str) -> str:
    # mlx-vlm / mlx-lm set these in os.environ when another test imports them; a run has neither.
    hushed = ("TRANSFORMERS_NO_ADVISORY_WARNINGS", "TRANSFORMERS_VERBOSITY")
    env = {k: v for k, v in os.environ.items() if k not in hushed}
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env
    )
    return done.stderr


def test_transformers_torch_advice_is_dropped_once_qwi_is_imported() -> None:
    # The bare import proves the message still exists, so the second assert isn't vacuous.
    assert TORCH_ADVICE in import_transformers_stderr("import transformers")
    assert import_transformers_stderr("import qwi, transformers") == ""


def test_processor_kwargs_warning_is_dropped_and_others_pass(
    capsys: pytest.CaptureFixture[str],
) -> None:
    quiet.install()
    logger = logging.getLogger("transformers.processing_utils")
    handler = logging.StreamHandler(sys.stderr)
    logger.addHandler(handler)
    try:
        logger.warning(
            "Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, "
            "not in `**kwargs`"
        )
        logger.warning("something else")
    finally:
        logger.removeHandler(handler)
    assert capsys.readouterr().err == "something else\n"


def test_install_is_idempotent() -> None:
    quiet.install()
    quiet.install()
    for name in quiet.DROPPED:
        filters = logging.getLogger(name).filters
        assert sum(isinstance(f, quiet.DropMessages) for f in filters) == 1


def test_silence_tqdm_prints_nothing_but_keeps_elapsed(capsys: pytest.CaptureFixture[str]) -> None:
    from tqdm import tqdm

    module = ModuleType("fake")
    vars(module)["tqdm"] = tqdm  # a stand-in for mflux's module global
    quiet.silence_tqdm(module)
    quiet.silence_tqdm(module)  # a second load doesn't wrap twice
    bar = vars(module)["tqdm"](range(1000))
    assert sum(bar) == sum(range(1000))
    assert capsys.readouterr().err == ""
    assert bar.format_dict["elapsed"] > 0  # mflux's generation_time


def test_mflux_config_bar_is_silenced(capsys: pytest.CaptureFixture[str]) -> None:
    """The real module Generate.load() patches: its `time_steps` reads the `tqdm` global."""
    from mflux.models.common.config import config as mflux_config

    assert "tqdm" in vars(mflux_config)
    quiet.silence_tqdm(mflux_config)
    bar = vars(mflux_config)["tqdm"](range(3))
    assert list(bar) == [0, 1, 2]
    assert capsys.readouterr().err == ""
