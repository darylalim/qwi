"""Stage 1: prompt safety with Qwen3Guard-Gen-0.6B via mlx-lm (PLAN.md §5.2).

The safety policy lives in the model's chat template, so the stage sends only the user turn and
parses two lines back: `Safety: <level>` and `Categories: <list>`. Everything after the model call
is plain Python that fast tests reach without a model: `parse()`, the `guard_policy.toml` table
(`load_policy()`), and `decide()`, which turns one or two verdicts (input, rewrite) into a single
`allow` / `warn` / `block` decision. Unsafe always blocks; Controversial follows `--guard-mode`;
output that can't be parsed blocks (fail closed).
"""

import re
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qwi import memory, paths
from qwi.config import DEFAULTS

if TYPE_CHECKING:
    import mlx.nn as nn
    from mlx_lm.tokenizer_utils import TokenizerWrapper

LEVELS = ("Safe", "Unsafe", "Controversial")
CATEGORIES = (
    "Violent",
    "Non-violent Illegal Acts",
    "Sexual Content or Sexual Acts",
    "PII",
    "Suicide & Self-Harm",
    "Unethical Acts",
    "Politically Sensitive Topics",
    "Copyright Violation",
    "Jailbreak",
)
NO_CATEGORY = "None"  # the policy row for a Controversial verdict with no category


class GuardMode(StrEnum):
    """`--guard-mode`: the policy column for Controversial verdicts (§5.2)."""

    STRICT = "strict"
    DEFAULT = "default"
    LOOSE = "loose"


MODES = tuple(m.value for m in GuardMode)
ACTIONS = ("allow", "warn", "block")  # least to most restrictive
PARSE_ERROR = "guard_parse_error"

# The model card's parsing regexes (Qwen3Guard-Gen, Apache-2.0; see NOTICE).
_SAFETY = re.compile(r"Safety: (Safe|Unsafe|Controversial)")
_CATEGORY = re.compile("|".join(re.escape(c) for c in (*CATEGORIES, NO_CATEGORY)))

Policy = dict[str, dict[str, str]]  # Controversial category → mode → action


@dataclass(frozen=True)
class Verdict:
    level: str | None  # None: the output had no `Safety:` line
    categories: tuple[str, ...]  # in the model's order, without "None"
    raw: str

    def as_record(self) -> dict[str, Any]:
        """A `run.json` `guard.input` / `guard.rewrite` entry (§5.2 "Run record")."""
        return {"level": self.level, "categories": list(self.categories), "raw": self.raw}

    def describe(self) -> str:
        if self.level is None:
            return "unparseable guard output"
        return f"{self.level} ({', '.join(self.categories)})" if self.categories else self.level


def describe_pass(check: str, verdict: Verdict, mode: str) -> str:
    """One console line for a pass: which pass, level, categories, mode."""
    return f"{check} prompt is {verdict.describe()} (--guard-mode {mode})"


def parse(raw: str) -> Verdict:
    safety = _SAFETY.search(raw)
    if safety is None:
        return Verdict(None, (), raw)
    found = [c for c in _CATEGORY.findall(raw) if c != NO_CATEGORY]
    return Verdict(safety.group(1), tuple(dict.fromkeys(found)), raw)


def load_policy(path: Path = paths.GUARD_POLICY) -> Policy:
    """Read and check the table: every category plus "None", every mode, a known action each."""
    table = tomllib.loads(path.read_text(encoding="utf-8")).get("controversial", {})
    expected = {*CATEGORIES, NO_CATEGORY}
    if set(table) != expected:
        missing, extra = sorted(expected - set(table)), sorted(set(table) - expected)
        raise ValueError(f"{path.name}: categories missing {missing}, unknown {extra}")
    for category, row in table.items():
        if set(row) != set(MODES) or not set(row.values()) <= set(ACTIONS):
            raise ValueError(f"{path.name}: {category!r} needs {MODES}, each one of {ACTIONS}")
    return table


def judge(verdict: Verdict, mode: str, policy: Policy) -> tuple[str, str | None]:
    """One pass's action and its reason detail, e.g. `("warn", "Controversial:Violent")`."""
    match verdict.level:
        case None:
            return "block", PARSE_ERROR
        case "Safe":
            return "allow", None
        case "Unsafe":  # every mode, every category: no flag overrides it
            first = verdict.categories[:1]
            return "block", ":".join(("Unsafe", *first))
    keys = verdict.categories or (NO_CATEGORY,)
    # Most restrictive wins; ties keep the model's order. max() returns the first maximum.
    key = max(keys, key=lambda c: ACTIONS.index(policy[c][mode]))
    action = policy[key][mode]
    return action, None if action == "allow" else f"Controversial:{key}"


@dataclass(frozen=True)
class GuardResult:
    mode: str
    input: Verdict
    rewrite: Verdict | None
    decision: str  # allow | warn | block
    reason: str | None  # "<pass>:<detail>", e.g. "input:Controversial:Violent"

    @property
    def trigger(self) -> str | None:
        """The pass that set the decision: "input" or "rewrite"."""
        return self.reason.split(":", 1)[0] if self.reason else None

    def summary(self) -> str:
        """One line for the console and errors: which pass, level, categories."""
        if self.trigger is None:
            return self.decision
        verdict = self.rewrite if self.trigger == "rewrite" else self.input
        assert verdict is not None
        return describe_pass(self.trigger, verdict, self.mode)

    def as_record(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "input": self.input.as_record(),
            "rewrite": self.rewrite.as_record() if self.rewrite else None,
            "decision": self.decision,
            "reason": self.reason,
        }


def decide(
    mode: str, input: Verdict, rewrite: Verdict | None = None, policy: Policy | None = None
) -> GuardResult:
    """Combine both passes: the more restrictive one wins, the input pass on a tie."""
    if mode not in MODES:
        raise ValueError(f"unknown guard mode {mode!r}; expected one of {MODES}")
    policy = policy if policy is not None else load_policy()
    decision, reason = "allow", None
    for name, verdict in (("input", input), ("rewrite", rewrite)):
        if verdict is None:
            continue
        action, detail = judge(verdict, mode, policy)
        if ACTIONS.index(action) > ACTIONS.index(decision):
            decision, reason = action, f"{name}:{detail}"
    return GuardResult(mode, input, rewrite, decision, reason)


class Guard:
    """Loads Qwen3Guard with mlx-lm and returns a parsed `Verdict` per text (greedy, 64 tokens)."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._model: nn.Module | None = None
        self._tokenizer: TokenizerWrapper | None = None

    def load(self) -> None:
        from mlx_lm import load

        memory.configure()
        self._model, self._tokenizer = load(str(self.model_path))[:2]

    def __call__(self, text: str) -> Verdict:
        from mlx_lm import generate

        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Guard.load() was not called")
        # No system prompt: the policy is in the chat template, which also opens the reply.
        prompt = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
        )
        # No sampler: mlx-lm decodes greedily, so a verdict is deterministic (§5.2).
        raw = generate(self._model, self._tokenizer, prompt, max_tokens=DEFAULTS.guard_max_tokens)
        return parse(raw)

    def unload(self) -> None:
        self._model = self._tokenizer = None
        memory.evict()
