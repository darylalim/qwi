"""The single source of defaults (PLAN.md §5): the CLI, UI and eval runner all read `DEFAULTS`."""

from dataclasses import dataclass

GiB = 2**30


@dataclass(frozen=True)
class Defaults:
    width: int = 768
    height: int = 768
    steps: int = 20
    n: int = 4
    eval_seeds: tuple[int, ...] = (1000, 1001)  # §5.4: fixed for every prompt and variant
    # Caps MLX's buffer cache; without it the VAE decode leaves ≈ 15 GB of freed buffers cached
    # and macOS swaps (PLAN.md §4, §8 2026-10-15 "Swap cause found").
    cache_limit_bytes: int = 2 * GiB
    # System-wide swap-outs a stage may add before the step-hook watchdog raises
    # DeviceOutOfMemory (PLAN.md §8 2026-10-15 memory-limit result, watchdog threshold).
    swap_limit_bytes: int = 2 * GiB
    # The watchdog limit with --allow-swap: some swap is accepted, thrashing isn't. 6 GiB is the
    # probes' kill line, already ≈ 10x slower per step (PLAN.md §8 2026-10-15 --allow-swap).
    allow_swap_limit_bytes: int = 6 * GiB
    # Pre-flight budget = min(wired limit, available memory) minus this (PLAN.md §5.11).
    preflight_headroom_bytes: int = 2 * GiB
    # Free disk that must remain after any download, and before a run starts (PLAN.md §5.7).
    min_free_disk_bytes: int = 20 * 10**9
    # run.log / app log level; `-v` or QWI_LOG_LEVEL=DEBUG overrides it (§5.11 "Logs").
    log_level: str = "INFO"


DEFAULTS = Defaults()
