"""The single source of defaults (PLAN.md §5): the CLI, UI and eval runner all read `DEFAULTS`."""

from dataclasses import dataclass, field

GiB = 2**30


@dataclass(frozen=True)
class Defaults:
    width: int = 768
    height: int = 768
    steps: int = 20
    n: int = 4
    eval_seeds: tuple[int, ...] = (1000, 1001)  # §5.4: fixed for every prompt and variant
    # --guard-mode: how Controversial verdicts are handled (PLAN.md §5.2, guard_policy.toml).
    guard_mode: str = "default"
    # The guard replies with two short lines; greedy decoding makes the verdict deterministic.
    guard_max_tokens: int = 64
    # --rewrite-style (PLAN.md §5.1): "long" won the M2 three-way comparison (§8 2026-10-16).
    rewrite_style: str = "long"
    # Low temperature keeps rewrites faithful; one fixed seed makes them reproducible (§5.1).
    rewrite_temperature: float = 0.3
    rewrite_top_p: float = 0.9
    rewrite_seed: int = 0
    # Per style: generation budget, and the word count above which the original prompt is kept.
    rewrite_max_tokens: dict[str, int] = field(default_factory=lambda: {"short": 256, "long": 1024})
    rewrite_max_words: dict[str, int] = field(default_factory=lambda: {"short": 200, "long": 700})
    # Reranker (PLAN.md §5.3): upstream's generic instruction (the model card's default) won M3's
    # best-of-4 calibration over the task-specific one (§8 2026-10-17 "Reranker instruction").
    rerank_instruction: str = (
        "Given a search query, retrieve relevant candidates that answer the query."
    )
    # The task-specific instruction it beat, kept as the `task` preset for re-scoring (§5.4).
    rerank_task_instruction: str = (
        "Given an image-generation request, judge whether the image faithfully depicts "
        "everything the request asks for."
    )
    # --min-score: below this best prob (the model leans "no"), warn that nothing clearly matched.
    rerank_min_score: float = 0.5
    # Eval OCR hint for text-* prompts (PLAN.md §5.4 "Automated checks"): Qwen3-VL-2B, greedy.
    ocr_prompt: str = "Transcribe all text visible in this image exactly. Output only the text."
    ocr_max_tokens: int = 128  # the eval's longest text is two short lines
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
    # Streamlit ports: the main UI and the eval review app run side by side (PLAN.md §5.6).
    ui_port: int = 8501
    review_port: int = 8502
    # run.log / app log level; `-v` or QWI_LOG_LEVEL=DEBUG overrides it (§5.11 "Logs").
    log_level: str = "INFO"


DEFAULTS = Defaults()
