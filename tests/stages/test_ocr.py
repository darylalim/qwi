"""The OCR hint's exact-match rule (PLAN.md §5.4 "Automated checks"); no model needed."""

import pytest

from qwi.stages.ocr import exact, match


@pytest.mark.parametrize(
    ("transcription", "found"),
    [
        ("THE DAILY GRIND", True),
        ("Welcome to\nTHE DAILY\n  GRIND\nOpen daily", True),  # line breaks are not text
        ("The Daily Grind", False),  # case counts
        ("THE DAILY GRIN", False),
        ("", False),
    ],
    ids=["same", "wrapped", "case", "typo", "empty"],
)
def test_a_quoted_string_must_appear_verbatim(transcription: str, found: bool) -> None:
    assert exact(["THE DAILY GRIND"], transcription) == {"THE DAILY GRIND": found}


def test_each_string_is_checked_on_its_own_with_punctuation_and_script() -> None:
    read = "NORTHERN LIGHTS\nLive at Harbor Hall - March 14\n清风茶舍"
    assert exact(["NORTHERN LIGHTS", "Live at Harbor Hall · March 14", "清风茶舍"], read) == {
        "NORTHERN LIGHTS": True,
        "Live at Harbor Hall · March 14": False,  # "-" is not "·"
        "清风茶舍": True,
    }


def test_no_quoted_strings_means_nothing_to_check() -> None:
    assert exact([], "anything") == {}


def test_match_ignores_only_case() -> None:
    # What the model returns for an all-caps sign (§8 "OCR hint"): title case, or a lost digit.
    assert match(["OPEN 24 HOURS"], "Open\n24 Hours") == {"OPEN 24 HOURS": True}
    assert exact(["OPEN 24 HOURS"], "Open\n24 Hours") == {"OPEN 24 HOURS": False}
    assert match(["OPEN 24 HOURS"], "Open Hours") == {"OPEN 24 HOURS": False}
    assert match(["清风茶舍"], "清风茶舍") == {"清风茶舍": True}
