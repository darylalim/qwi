"""Memory discipline for one-stage-at-a-time loading (PLAN.md §4).

- `configure()` caps MLX's buffer cache; call it once before any model loads.
- `evict()` frees a stage's models after the caller drops its references.
- `track_peak()` measures a phase's peak MLX memory.
- `SwapWatchdog` is an mflux step hook that raises `DeviceOutOfMemory` once system swap grows
  past a limit. `mx.set_memory_limit` doesn't stop swapping on MLX 0.32 (§8 2026-10-15).

`mlx` is imported inside the functions, never at module level (CLAUDE.md "Imports").
"""

import gc
import re
import subprocess
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from qwi.config import DEFAULTS
from qwi.errors import DeviceOutOfMemory

GB = 1e9  # run records report decimal GB, like mflux and Activity Monitor (§8)

_PAGE_SIZE = re.compile(r"page size of (\d+) bytes")
_COUNTER = re.compile(r'^"?([^":\n]+)"?:\s+(\d+)\.?$', re.MULTILINE)  # some keys are quoted


def configure(cache_limit_bytes: int = DEFAULTS.cache_limit_bytes) -> None:
    import mlx.core as mx

    mx.set_cache_limit(cache_limit_bytes)


def evict() -> None:
    """Free whatever the caller just dropped: run `del model` (or set it to `None`) first."""
    import mlx.core as mx

    gc.collect()
    mx.clear_cache()


@dataclass
class Peak:
    """Peak MLX memory for one phase, in GB; filled in when the `track_peak` block exits."""

    gb: float = 0.0


@contextmanager
def track_peak() -> Generator[Peak]:
    import mlx.core as mx

    peak = Peak()
    mx.reset_peak_memory()
    try:
        yield peak
    finally:
        peak.gb = mx.get_peak_memory() / GB


@dataclass(frozen=True)
class VmStat:
    """`/usr/bin/vm_stat` counters (pages) and the page size from its header."""

    page_size: int
    pages: dict[str, int]

    def bytes(self, *names: str) -> int:
        missing = [name for name in names if name not in self.pages]
        if missing:
            raise ValueError(f"vm_stat output has no {', '.join(missing)} line")
        return sum(self.pages[name] for name in names) * self.page_size

    @property
    def swapout_bytes(self) -> int:
        """Swapped out since boot, system-wide (§8 watchdog method): only deltas mean anything."""
        return self.bytes("Swapouts")

    @property
    def available_bytes(self) -> int:
        """Free + inactive + purgeable: what pre-flight treats as available (§5.11)."""
        return self.bytes("Pages free", "Pages inactive", "Pages purgeable")


def parse_vm_stat(text: str) -> VmStat:
    page = _PAGE_SIZE.search(text)
    if page is None:
        raise ValueError("vm_stat output has no page size header")
    return VmStat(int(page.group(1)), {k: int(v) for k, v in _COUNTER.findall(text)})


def vm_stat() -> VmStat:
    out = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True, check=True).stdout
    return parse_vm_stat(out)


def swapouts() -> int:
    return vm_stat().swapout_bytes


class SwapWatchdog:
    """Raises `DeviceOutOfMemory` when swap-outs grow more than `limit_bytes` past `start()`.

    The counter is system-wide, so other processes count too; only the delta is meaningful.
    Start it before the stage loads its model. mflux calls `call_in_loop` after every denoising
    step; nothing runs during the VAE decode, so callers also `check()` after each image.
    """

    def __init__(
        self,
        stage: str,
        limit_bytes: int = DEFAULTS.swap_limit_bytes,
        read: Callable[[], int] = swapouts,
    ) -> None:
        self.stage = stage
        self.limit_bytes = limit_bytes
        self._read = read
        self._baseline: int | None = None

    def start(self) -> None:
        self._baseline = self._read()

    @property
    def delta_bytes(self) -> int:
        if self._baseline is None:
            raise RuntimeError("SwapWatchdog.start() was not called")
        return self._read() - self._baseline

    def check(self) -> None:
        delta = self.delta_bytes
        if delta > self.limit_bytes:
            raise DeviceOutOfMemory(
                self.stage,
                f"the system swapped {delta / GB:.1f} GB (limit {self.limit_bytes / GB:.1f} GB)",
            )

    def call_in_loop(self, **_: Any) -> None:
        self.check()
