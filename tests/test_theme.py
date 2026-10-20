"""The "Neutral gallery" theme in `.streamlit/config.toml` (PLAN.md §5.6 "Theme").

Contrast uses the WCAG 2.x relative-luminance formula. The targets are the §5.6 table's.
"""

import colorsys
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / ".streamlit" / "config.toml"
UI = REPO / "src" / "qwi" / "ui"

MODES = ["light", "dark"]
SEMANTIC = ["redColor", "orangeColor", "yellowColor", "greenColor", "blueColor", "violetColor"]
PAIRED_HUES = ["primaryColor", "linkColor", *SEMANTIC]
MAX_HUE_GAP = 12  # degrees; §5.6 measured ≤ 10
WHITE = "#FFFFFF"


def theme() -> dict[str, Any]:
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))["theme"]


def section(mode: str, sidebar: bool = False) -> dict[str, Any]:
    colors = theme()[mode]
    return colors["sidebar"] if sidebar else colors


def page_colors(mode: str) -> dict[str, Any]:
    """A mode's own keys, without the nested sidebar table."""
    return {k: v for k, v in section(mode).items() if k != "sidebar"}


def rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return r, g, b


def luminance(hex_color: str) -> float:
    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in rgb(hex_color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def hue(hex_color: str) -> float:
    return colorsys.rgb_to_hls(*rgb(hex_color))[0] * 360


def test_light_and_dark_define_the_same_keys() -> None:
    assert page_colors("light").keys() == page_colors("dark").keys()
    assert section("light", sidebar=True).keys() == section("dark", sidebar=True).keys()


def test_every_key_is_a_streamlit_option() -> None:
    """A misspelled theme key is silently ignored, so check them against Streamlit's options."""
    from streamlit import config

    known = config._config_options_template  # private, but the only full list of options
    names = [f"theme.{k}" for k, v in theme().items() if not isinstance(v, dict)]
    for mode in MODES:
        names += [f"theme.{mode}.{k}" for k in page_colors(mode)]
        names += [f"theme.{mode}.sidebar.{k}" for k in section(mode, sidebar=True)]
    assert [name for name in names if name not in known] == []


def test_shared_keys_stay_in_theme_and_sections_nest_mode_first() -> None:
    shared = {k for k, v in theme().items() if not isinstance(v, dict)}
    for mode in MODES:
        assert shared.isdisjoint(page_colors(mode))
        assert shared.isdisjoint(section(mode, sidebar=True))
    assert "sidebar" not in theme()  # [theme.sidebar.light] is the wrong order


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("sidebar", [False, True], ids=["page", "sidebar"])
def test_text_link_and_primary_meet_their_contrast_targets(mode: str, sidebar: bool) -> None:
    colors = section(mode, sidebar)
    bg = colors["backgroundColor"]
    assert contrast(colors["textColor"], bg) >= 7
    assert contrast(colors["linkColor"], bg) >= 4.5
    assert contrast(colors["primaryColor"], bg) >= 3
    assert contrast(WHITE, colors["primaryColor"]) >= 4.5  # button text


@pytest.mark.parametrize("mode", MODES)
def test_gray_semantic_and_chart_colors_meet_their_contrast_targets(mode: str) -> None:
    colors = section(mode)
    bg = colors["backgroundColor"]
    assert contrast(colors["grayColor"], bg) >= 4.5
    assert {
        k: round(contrast(colors[k], bg), 2) for k in SEMANTIC if contrast(colors[k], bg) < 4.5
    } == {}
    assert [c for c in colors["chartCategoricalColors"] if contrast(c, bg) < 3] == []


def test_layer_steps_match_across_modes() -> None:
    steps = [
        contrast(section(m)["secondaryBackgroundColor"], section(m)["backgroundColor"])
        for m in MODES
    ]
    assert steps[0] == pytest.approx(steps[1], abs=0.05)


@pytest.mark.parametrize("key", PAIRED_HUES)
def test_light_and_dark_colors_share_a_hue(key: str) -> None:
    gap = abs(hue(section("light")[key]) - hue(section("dark")[key])) % 360
    assert min(gap, 360 - gap) <= MAX_HUE_GAP


@pytest.mark.parametrize("mode", MODES)
def test_chart_scales_have_ten_colors_and_sequential_gains_contrast(mode: str) -> None:
    colors = section(mode)
    assert len(colors["chartSequentialColors"]) == 10
    assert len(colors["chartDivergingColors"]) == 10
    steps = [contrast(c, colors["backgroundColor"]) for c in colors["chartSequentialColors"]]
    assert steps == sorted(steps)
    assert len(set(steps)) == 10


@pytest.mark.parametrize("mode", MODES)
def test_sidebar_is_darker_than_the_page(mode: str) -> None:
    page = section(mode)["backgroundColor"]
    sidebar = section(mode, sidebar=True)["backgroundColor"]
    assert luminance(sidebar) < luminance(page)


def test_ui_code_has_no_css_and_no_hex_colors() -> None:
    """Theming lives only in config.toml; pages use semantic color names (CLAUDE.md "UI")."""
    offenders = []
    for path in sorted(UI.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if "unsafe_allow_html=True" in text or "<style" in text:
            offenders.append(f"{path.name}: CSS")
        if re.search(r"#[0-9A-Fa-f]{6}\b", text):
            offenders.append(f"{path.name}: hex color")
    assert offenders == []


def test_server_never_watches_files() -> None:
    """The watcher's `sys.modules` scan prints a torch traceback per transformers alias (§8)."""
    from streamlit import config

    server = tomllib.loads(CONFIG.read_text(encoding="utf-8"))["server"]
    assert server["fileWatcherType"] == "none"
    assert "server.fileWatcherType" in config._config_options_template
