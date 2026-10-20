from qwi.config import DEFAULTS, GiB


def test_defaults_match_plan_section_1() -> None:
    # PLAN.md §1: 768x768, 20 steps, N=4.
    assert (DEFAULTS.width, DEFAULTS.height, DEFAULTS.steps, DEFAULTS.n) == (768, 768, 20, 4)


def test_memory_defaults_match_plan_section_4() -> None:
    assert DEFAULTS.cache_limit_bytes == 2 * GiB
    assert DEFAULTS.swap_limit_bytes == 2 * GiB
    assert DEFAULTS.allow_swap_limit_bytes == 6 * GiB
    assert DEFAULTS.preflight_headroom_bytes == 2 * GiB


def test_disk_rule_matches_plan_section_5_7() -> None:
    assert DEFAULTS.min_free_disk_bytes == 20 * 10**9


def test_eval_seeds_match_plan_section_5_4() -> None:
    assert DEFAULTS.eval_seeds == (1000, 1001)


def test_thumbnail_sizes_match_plan_section_5_6() -> None:
    assert DEFAULTS.thumb_sizes == {"S": 256, "M": 384, "L": 512}
    assert DEFAULTS.thumb_size == "M"
    # M is pixel-exact on Retina (2 device px per CSS px) only at the native image width.
    assert DEFAULTS.thumb_sizes[DEFAULTS.thumb_size] * 2 == DEFAULTS.width
    assert DEFAULTS.reading_width == 760


def test_full_view_fits_a_laptop_window() -> None:
    # 1.5x M; with the dialog's 146 px of chrome and 48 px margin it fits 779 px (§8 2026-10-19).
    assert DEFAULTS.full_view_width == DEFAULTS.thumb_sizes["M"] * 3 // 2 == 576
    assert DEFAULTS.full_view_width + 146 + 48 <= 779
