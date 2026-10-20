"""Quiets third-party console noise during runs (PLAN.md §9 "Console noise"). Cosmetic only.

Run records and qwi's own logs are unchanged: this drops two known transformers warnings and
mflux's per-image `tqdm` bar, which interleaved with `qwi generate`'s step counter.
"""

import functools
import io
import logging
from types import ModuleType

# Messages dropped per logger. A logger's filters see only records logged at that exact logger,
# so each entry names the logger the warning is emitted from.
DROPPED: dict[str, tuple[str, ...]] = {
    # Once per process at `import transformers`: expected, the runtime is torch-free (§8).
    "transformers": ("PyTorch was not found",),
    # Once per rewrite: mlx-vlm's `apply_chat_template` passes `enable_thinking=False`, which
    # the Qwen3-VL template doesn't reference, so transformers reads it as a processor kwarg.
    "transformers.processing_utils": ("Kwargs passed to `processor.__call__`",),
}


class DropMessages(logging.Filter):
    def __init__(self, prefixes: tuple[str, ...]) -> None:
        super().__init__()
        self.prefixes = prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(self.prefixes)


def install() -> None:
    """Attach the filters. Stdlib only, so `qwi/__init__.py` runs it before transformers loads."""
    for name, prefixes in DROPPED.items():
        logger = logging.getLogger(name)  # the same object transformers' `get_logger` returns
        if not any(isinstance(f, DropMessages) for f in logger.filters):
            logger.addFilter(DropMessages(prefixes))


class _Discard(io.TextIOBase):
    def write(self, s: str) -> int:
        return len(s)


def silence_tqdm(module: ModuleType) -> None:
    """Make `module`'s bars print nothing. mflux 0.23.0 has no progress setting (§8).

    Not `disable=True`: a disabled bar reports `elapsed` 0, and mflux reads
    `format_dict["elapsed"]` as the generation time. Not `TQDM_DISABLE` either: tqdm reads it
    once at import and for every bar, which would also hide `qwi models fetch`'s downloads.
    """
    from tqdm import tqdm

    namespace = vars(module)  # the module's globals, which its bar factory reads
    if not isinstance(namespace["tqdm"], functools.partial):  # once per process
        namespace["tqdm"] = functools.partial(tqdm, file=_Discard())
