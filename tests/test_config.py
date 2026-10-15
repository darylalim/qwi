from qwi.config import DEFAULTS, GiB


def test_defaults_match_plan_section_1() -> None:
    # PLAN.md §1: 768x768, 20 steps, N=4.
    assert (DEFAULTS.width, DEFAULTS.height, DEFAULTS.steps, DEFAULTS.n) == (768, 768, 20, 4)


def test_memory_defaults_match_plan_section_4() -> None:
    assert DEFAULTS.cache_limit_bytes == 2 * GiB
    assert DEFAULTS.swap_limit_bytes == 2 * GiB
    assert DEFAULTS.allow_swap_limit_bytes == 6 * GiB
    assert DEFAULTS.preflight_headroom_bytes == 2 * GiB
