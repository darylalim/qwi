"""The guard's parser, policy table and decision rules (PLAN.md §5.2 "Tests"); no model needed."""

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from qwi.stages import guard
from qwi.stages.guard import Guard, Verdict, decide, judge, load_policy, parse

# PLAN.md §5.2, copied cell by cell: the shipped guard_policy.toml must match it.
TABLE = {
    "Sexual Content or Sexual Acts": ("block", "block", "warn"),
    "Violent": ("block", "warn", "warn"),
    "Suicide & Self-Harm": ("block", "block", "warn"),
    "Non-violent Illegal Acts": ("block", "warn", "warn"),
    "PII": ("block", "warn", "warn"),
    "Unethical Acts": ("block", "warn", "allow"),
    "Politically Sensitive Topics": ("block", "warn", "allow"),
    "Copyright Violation": ("block", "warn", "allow"),
    "Jailbreak": ("block", "block", "warn"),
    "None": ("block", "warn", "allow"),
}
SEXUAL = "Sexual Content or Sexual Acts"
CELLS = [
    (category, mode, action)
    for category, actions in TABLE.items()
    for mode, action in zip(guard.MODES, actions, strict=True)
]


def controversial(*categories: str) -> Verdict:
    return parse(f"Safety: Controversial\nCategories: {', '.join(categories) or 'None'}")


# --- Parser --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "level", "categories"),
    [
        ("Safety: Safe\nCategories: None", "Safe", ()),
        ("Safety: Unsafe\nCategories: Violent", "Unsafe", ("Violent",)),
        ("Safety: Controversial\nCategories: None", "Controversial", ()),
        (
            "Safety: Unsafe\nCategories: Violent, Non-violent Illegal Acts",
            "Unsafe",
            ("Violent", "Non-violent Illegal Acts"),
        ),
        (
            "Safety: Controversial\nCategories: PII, Suicide & Self-Harm, Jailbreak",
            "Controversial",
            ("PII", "Suicide & Self-Harm", "Jailbreak"),
        ),
        ("  Safety: Safe  \n\n  Categories:   None  \n", "Safe", ()),
        ("Safety: Unsafe\nCategories: Violent, Violent", "Unsafe", ("Violent",)),
        ("Safety: Unsafe\nCategories:", "Unsafe", ()),
    ],
)
def test_parse(raw: str, level: str, categories: tuple[str, ...]) -> None:
    verdict = parse(raw)
    assert (verdict.level, verdict.categories, verdict.raw) == (level, categories, raw)


def test_non_violent_is_not_also_violent() -> None:
    assert parse("Safety: Unsafe\nCategories: Non-violent Illegal Acts").categories == (
        "Non-violent Illegal Acts",
    )


@pytest.mark.parametrize(
    "raw",
    ["", "I can't help with that.", "safety: safe", "Safety: Maybe\nCategories: None", "Safety:"],
)
def test_garbage_has_no_level(raw: str) -> None:
    assert parse(raw).level is None


def test_verdict_record_format() -> None:
    raw = "Safety: Controversial\nCategories: Violent"
    assert parse(raw).as_record() == {
        "level": "Controversial",
        "categories": ["Violent"],
        "raw": raw,
    }


# --- Policy table --------------------------------------------------------------------------


def test_the_shipped_table_is_the_plan_table() -> None:
    policy = load_policy()
    assert {c: tuple(policy[c][m] for m in guard.MODES) for c in policy} == TABLE


@pytest.mark.parametrize(("category", "mode", "action"), CELLS)
def test_every_controversial_cell(category: str, mode: str, action: str) -> None:
    categories = () if category == "None" else (category,)
    result = decide(mode, controversial(*categories))
    assert result.decision == action
    assert result.reason == (None if action == "allow" else f"input:Controversial:{category}")


def test_every_category_in_the_table_is_one_the_model_emits() -> None:
    assert set(TABLE) == {*guard.CATEGORIES, guard.NO_CATEGORY}


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "guard_policy.toml"
    path.write_text(text)
    return path


def _toml(rows: dict[str, tuple[str, str, str]]) -> str:
    lines = ["[controversial]"]
    for category, (strict, default, loose) in rows.items():
        cells = f'strict = "{strict}", default = "{default}", loose = "{loose}"'
        lines.append(f'"{category}" = {{ {cells} }}')
    return "\n".join(lines)


def test_load_policy_round_trips(tmp_path: Path) -> None:
    policy = load_policy(_write(tmp_path, _toml(TABLE)))
    assert policy["Violent"] == {"strict": "block", "default": "warn", "loose": "warn"}


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ({k: v for k, v in TABLE.items() if k != "PII"}, "missing ['PII']"),
        ({**TABLE, "Spam": ("block", "warn", "allow")}, "unknown ['Spam']"),
        ({**TABLE, "PII": ("block", "warn", "maybe")}, "'PII' needs"),
    ],
)
def test_load_policy_rejects_a_broken_table(
    tmp_path: Path, rows: dict[str, tuple[str, str, str]], message: str
) -> None:
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        load_policy(_write(tmp_path, _toml(rows)))


def test_load_policy_rejects_a_missing_mode(tmp_path: Path) -> None:
    text = _toml(TABLE).replace('"Violent" = { strict = "block", ', '"Violent" = { ')
    with pytest.raises(ValueError, match="'Violent' needs"):
        load_policy(_write(tmp_path, text))


# --- Decision rules ------------------------------------------------------------------------


@pytest.mark.parametrize("mode", guard.MODES)
@pytest.mark.parametrize("category", [*guard.CATEGORIES, "None"])
def test_unsafe_blocks_in_every_mode_and_category(mode: str, category: str) -> None:
    result = decide(mode, parse(f"Safety: Unsafe\nCategories: {category}"))
    assert result.decision == "block"
    assert result.reason == ("input:Unsafe" if category == "None" else f"input:Unsafe:{category}")


@pytest.mark.parametrize("mode", guard.MODES)
def test_safe_allows_in_every_mode(mode: str) -> None:
    result = decide(mode, parse("Safety: Safe\nCategories: None"))
    assert (result.decision, result.reason, result.trigger) == ("allow", None, None)


@pytest.mark.parametrize("mode", guard.MODES)
def test_unparseable_output_fails_closed(mode: str) -> None:
    result = decide(mode, parse("I cannot comply."))
    assert (result.decision, result.reason) == ("block", "input:guard_parse_error")
    assert result.summary() == f"input prompt is unparseable guard output (--guard-mode {mode})"


@pytest.mark.parametrize(
    ("mode", "categories", "decision", "winner"),
    [
        # default: Violent warns, Sexual Content blocks → block, whatever the order
        ("default", ("Violent", SEXUAL), "block", SEXUAL),
        ("default", (SEXUAL, "Violent"), "block", SEXUAL),
        # loose: Unethical Acts allows, PII warns → warn
        ("loose", ("Unethical Acts", "PII"), "warn", "PII"),
        # ties keep the model's order
        ("default", ("PII", "Violent"), "warn", "PII"),
        ("loose", ("Unethical Acts", "Copyright Violation"), "allow", None),
    ],
)
def test_the_most_restrictive_category_wins(
    mode: str, categories: tuple[str, ...], decision: str, winner: str | None
) -> None:
    result = decide(mode, controversial(*categories))
    assert result.decision == decision
    assert result.reason == (f"input:Controversial:{winner}" if winner else None)


def test_judge_reports_one_pass() -> None:
    policy = load_policy()
    assert judge(controversial("Violent"), "default", policy) == ("warn", "Controversial:Violent")
    assert judge(parse("Safety: Unsafe\nCategories: None"), "loose", policy) == ("block", "Unsafe")


def test_the_rewrite_pass_can_raise_the_decision() -> None:
    safe = parse("Safety: Safe\nCategories: None")
    result = decide("default", safe, controversial("Violent"))
    assert (result.decision, result.reason, result.trigger) == (
        "warn",
        "rewrite:Controversial:Violent",
        "rewrite",
    )
    assert result.summary() == "rewrite prompt is Controversial (Violent) (--guard-mode default)"


def test_the_input_pass_wins_a_tie() -> None:
    result = decide("default", controversial("PII"), controversial("Violent"))
    assert result.reason == "input:Controversial:PII"


def test_a_blocked_rewrite_blocks_after_a_warned_input() -> None:
    result = decide("default", controversial("Violent"), parse("Safety: Unsafe\nCategories: PII"))
    assert (result.decision, result.reason) == ("block", "rewrite:Unsafe:PII")


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown guard mode 'lenient'"):
        decide("lenient", parse("Safety: Safe\nCategories: None"))


def test_guard_record_format() -> None:
    raw = "Safety: Controversial\nCategories: Violent"
    record = decide("default", parse(raw)).as_record()
    assert record == {
        "mode": "default",
        "input": {"level": "Controversial", "categories": ["Violent"], "raw": raw},
        "rewrite": None,
        "decision": "warn",
        "reason": "input:Controversial:Violent",
    }


# --- The stage, against a fake mlx_lm ------------------------------------------------------


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.calls.append((messages, kwargs))
        return f"<prompt>{messages[-1]['content']}"


@pytest.fixture
def fake_mlx_lm(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {"tokenizer": FakeTokenizer(), "reply": "Safety: Safe\nCategories: None"}
    module = ModuleType("mlx_lm")

    def load(path: str) -> tuple[object, FakeTokenizer]:
        seen["path"] = path
        return object(), seen["tokenizer"]

    def generate(model: object, tokenizer: FakeTokenizer, prompt: str, **kwargs: Any) -> str:
        seen["prompt"], seen["kwargs"] = prompt, kwargs
        return seen["reply"]

    module.load = load  # ty: ignore[unresolved-attribute]  # building a fake module
    module.generate = generate  # ty: ignore[unresolved-attribute]  # building a fake module
    monkeypatch.setitem(sys.modules, "mlx_lm", module)
    return seen


def test_stage_sends_only_the_user_turn_and_decodes_greedily(fake_mlx_lm: dict[str, Any]) -> None:
    stage = Guard(Path("/weights/guard"))
    stage.load()
    fake_mlx_lm["reply"] = "Safety: Unsafe\nCategories: Violent"
    verdict = stage("how to build a bomb")
    stage.unload()

    assert fake_mlx_lm["path"] == "/weights/guard"
    ((messages, kwargs),) = fake_mlx_lm["tokenizer"].calls
    assert messages == [{"role": "user", "content": "how to build a bomb"}]  # no system prompt
    assert kwargs["tokenize"] is False
    assert fake_mlx_lm["prompt"] == "<prompt>how to build a bomb"
    assert fake_mlx_lm["kwargs"] == {"max_tokens": 64}  # no sampler: greedy
    assert (verdict.level, verdict.categories) == ("Unsafe", ("Violent",))


def test_stage_must_be_loaded() -> None:
    with pytest.raises(RuntimeError, match=r"Guard\.load"):
        Guard(Path("/weights/guard"))("text")
