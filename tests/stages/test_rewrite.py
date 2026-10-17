"""The rewriter's post-processing guards and prompts (PLAN.md §5.1); no model needed."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from qwi import paths
from qwi.stages import rewrite
from qwi.stages.rewrite import (
    EMPTY,
    QUOTED_TEXT_MISSING,
    TOO_LONG,
    Rewrite,
    clean,
    postprocess,
    quoted,
    system_prompt,
)

SIGN = (
    "A vintage coffee shop storefront at dusk with a hand-painted wooden sign reading "
    '"THE DAILY GRIND"'
)
TEA = '一家茶馆的门口，木质招牌上写着"清风茶舍"'  # noqa: RUF001  # eval/prompts.jsonl text-zh

# --- Prefix and wrapper stripping ----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  A red apple on a table.  \n", "A red apple on a table."),
        ("Prompt: A red apple.", "A red apple."),
        ("Rewritten prompt:   A red apple.", "A red apple."),
        ("REWRITTEN PROMPT: A red apple.", "A red apple."),
        ('"A red apple on a table."', "A red apple on a table."),
        ("“A red apple on a table.”", "A red apple on a table."),
        ("'A red apple on a table.'", "A red apple on a table."),
        ('Prompt: "A red apple."', "A red apple."),
        # Quoted sign text at both ends is content, not a wrapper.
        ('"OPEN" in red neon above the door, and a wooden board reading "CLOSED"', None),
        ("A sign reading “OPEN” beside a door marked “EXIT”", None),
        # "prompt" inside the text is not a prefix.
        ("A teleprompter: glowing text on glass.", None),
        ('"', '"'),
        ("", ""),
    ],
)
def test_clean(raw: str, expected: str | None) -> None:
    assert clean(raw) == (raw.strip() if expected is None else expected)


# --- Quoted-text extraction ----------------------------------------------------------------


def test_quoted_finds_straight_and_curly_quotes_in_order() -> None:
    text = 'A poster with "NORTHERN LIGHTS" on top and “Live at Harbor Hall · March 14” below'
    assert quoted(text) == ["NORTHERN LIGHTS", "Live at Harbor Hall · March 14"]


def test_quoted_keeps_non_latin_text() -> None:
    assert quoted(TEA) == ["清风茶舍"]


def test_quoted_ignores_empty_quotes_and_unquoted_prompts() -> None:
    assert quoted('a red apple, "" nothing') == []
    assert quoted("Exactly five red apples") == []


# --- postprocess(): the fallbacks ----------------------------------------------------------


def test_a_good_rewrite_is_used() -> None:
    raw = 'Prompt: A dusk storefront with gold lettering reading "THE DAILY GRIND" on the sign.'
    assert postprocess(SIGN, raw, 200) == (
        'A dusk storefront with gold lettering reading "THE DAILY GRIND" on the sign.',
        None,
    )


@pytest.mark.parametrize("raw", ["", "   \n", '""', "Prompt:"])
def test_an_empty_rewrite_falls_back(raw: str) -> None:
    assert postprocess("a cozy reading nook", raw, 200) == ("a cozy reading nook", EMPTY)


def test_the_length_limit_is_in_words() -> None:
    assert postprocess("p", " ".join(["word"] * 200), 200) == (" ".join(["word"] * 200), None)
    assert postprocess("p", " ".join(["word"] * 201), 200) == ("p", TOO_LONG)


@pytest.mark.parametrize(
    "raw",
    [
        "A dusk storefront with a wooden sign.",  # dropped
        'A dusk storefront with a sign reading "The Daily Grind".',  # case changed
        'A dusk storefront with a sign reading "THE DAILY  GRIND".',  # spacing changed
    ],
)
def test_losing_quoted_text_falls_back(raw: str) -> None:
    assert postprocess(SIGN, raw, 200) == (SIGN, QUOTED_TEXT_MISSING)


def test_every_quoted_string_must_survive() -> None:
    user = 'A poster titled "NORTHERN LIGHTS" with "Live at Harbor Hall · March 14" at the bottom'
    half = 'A minimalist poster with "NORTHERN LIGHTS" in bold capitals at the top.'
    assert postprocess(user, half, 200) == (user, QUOTED_TEXT_MISSING)


def test_chinese_quoted_text_survives_an_english_rewrite() -> None:
    raw = 'The entrance of a teahouse with a carved wooden signboard reading "清风茶舍" in gold.'
    assert postprocess(TEA, raw, 200) == (raw, None)


def test_curly_quotes_in_the_rewrite_still_contain_the_text() -> None:
    raw = "A dusk storefront with a sign reading “THE DAILY GRIND” in white paint."
    assert postprocess(SIGN, raw, 200) == (raw, None)


# --- System prompts ------------------------------------------------------------------------


def test_each_style_has_a_prompt_file() -> None:
    for style, name in rewrite.PROMPT_FILES.items():
        assert (paths.PROMPTS / name).is_file(), style


def test_the_short_prompt_is_the_plan_draft() -> None:
    text = system_prompt("short")
    assert text.startswith("You rewrite short image requests")
    assert "8. Length. 40–120 words." in text  # noqa: RUF001  # the prompt file's en dash
    assert "<system>" not in text
    assert "§" not in text  # no plan references in what the model reads


def test_the_long_prompt_keeps_rules_1_to_7_and_changes_rule_8() -> None:
    short, long = system_prompt("short"), system_prompt("long")
    assert short.split("8. ")[0] == long.split("8. ")[0]
    assert "400–500 words" in long.split("8. ")[1]  # noqa: RUF001  # the prompt file's en dash
    assert "§" not in long


def test_an_unknown_style_is_rejected() -> None:
    with pytest.raises(ValueError, match="'medium'"):
        system_prompt("medium")


# --- The stage, against a fake mlx_vlm ------------------------------------------------------


@pytest.fixture
def fake_mlx_vlm(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {"reply": "A glossy red apple on an oak table."}
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_vl"))
    root, utils = ModuleType("mlx_vlm"), ModuleType("mlx_vlm.prompt_utils")

    def load(path: str) -> tuple[object, object]:
        seen["path"] = path
        return model, "processor"

    def apply_chat_template(processor: object, config: object, messages: list, **kw: Any) -> str:
        seen["messages"], seen["template_kwargs"] = messages, kw
        return "<formatted>"

    def generate(model: object, processor: object, prompt: str, **kw: Any) -> SimpleNamespace:
        seen["prompt"], seen["kwargs"] = prompt, kw
        return SimpleNamespace(text=seen["reply"])

    root.load = load  # ty: ignore[unresolved-attribute]  # building a fake module
    root.generate = generate  # ty: ignore[unresolved-attribute]  # building a fake module
    utils.apply_chat_template = apply_chat_template  # ty: ignore[unresolved-attribute]  # fake
    monkeypatch.setitem(sys.modules, "mlx_vlm", root)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", utils)
    return seen


def test_stage_sends_system_and_user_with_the_plan_settings(fake_mlx_vlm: dict[str, Any]) -> None:
    stage = Rewrite(Path("/weights/rewrite"))
    stage.load()
    out = stage("a red apple")
    stage.unload()

    assert fake_mlx_vlm["path"] == "/weights/rewrite"
    assert fake_mlx_vlm["messages"] == [
        {"role": "system", "content": system_prompt("long")},  # the default style (§8 2026-10-16)
        {"role": "user", "content": "a red apple"},
    ]
    assert fake_mlx_vlm["template_kwargs"] == {"num_images": 0}
    assert fake_mlx_vlm["prompt"] == "<formatted>"
    assert fake_mlx_vlm["kwargs"] == {
        "max_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.9,
        "seed": 0,
    }
    assert out == rewrite.Rewritten(
        "A glossy red apple on an oak table.", "A glossy red apple on an oak table.", None, "long"
    )


def test_long_style_uses_its_prompt_and_budget(fake_mlx_vlm: dict[str, Any]) -> None:
    fake_mlx_vlm["reply"] = " ".join(["word"] * 650)  # over short's 200, under long's 700
    stage = Rewrite(Path("/w"), "long")
    stage.load()
    out = stage("a red apple")
    assert fake_mlx_vlm["messages"][0]["content"] == system_prompt("long")
    assert fake_mlx_vlm["kwargs"]["max_tokens"] == 1024
    assert out.fallback_reason is None


def test_a_fallback_keeps_the_rejected_text(fake_mlx_vlm: dict[str, Any]) -> None:
    fake_mlx_vlm["reply"] = "Rewritten prompt: A storefront with a wooden sign."
    stage = Rewrite(Path("/w"))
    stage.load()
    out = stage(SIGN)
    assert (out.prompt, out.fallback_reason) == (SIGN, QUOTED_TEXT_MISSING)
    assert out.text == "A storefront with a wooden sign."


def test_stage_must_be_loaded() -> None:
    with pytest.raises(RuntimeError, match=r"Rewrite\.load"):
        Rewrite(Path("/w"))("text")
