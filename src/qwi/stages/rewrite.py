"""Stage 2: prompt rewriting with Qwen3-VL-8B via mlx-vlm, text only (PLAN.md §5.1).

The system prompt for each `--rewrite-style` lives in `prompts/`. The model's output is never
trusted as is: `postprocess()` strips wrappers and prefixes, then falls back to the user's own
prompt if the rewrite is empty, too long, or lost any quoted text. Those checks are plain Python,
so fast tests reach them without a model.
"""

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qwi import memory, paths
from qwi.config import DEFAULTS

if TYPE_CHECKING:
    import mlx.nn as nn


class RewriteStyle(StrEnum):
    """`--rewrite-style`: which system prompt in `prompts/` the rewriter uses (§5.1)."""

    SHORT = "short"
    LONG = "long"


PROMPT_FILES = {
    RewriteStyle.SHORT: "rewrite_system.md",
    RewriteStyle.LONG: "rewrite_system_long.md",
}

# Fallback reasons: `RunRecord.rewrite_fallback_reason` (None = the rewrite is used).
EMPTY = "empty"
TOO_LONG = "too_long"
QUOTED_TEXT_MISSING = "quoted_text_missing"

_PREFIX = re.compile(r"^(?:rewritten prompt|prompt)\s*:\s*", re.IGNORECASE)
_QUOTED = re.compile(r'"([^"]+)"|“([^”]+)”')
_WRAPPERS = (('"', '"'), ("“", "”"), ("'", "'"))


def system_prompt(style: str) -> str:
    return (paths.PROMPTS / PROMPT_FILES[RewriteStyle(style)]).read_text(encoding="utf-8").strip()


def quoted(text: str) -> list[str]:
    """Every `"..."` and `“...”` string in `text`, in order: the text the image must show."""
    return [a or b for a, b in _QUOTED.findall(text)]


def clean(raw: str) -> str:
    """Strip whitespace, a `Prompt:` / `Rewritten prompt:` prefix, and quotes around the whole.

    Quotes come off only when they wrap everything: a rewrite that starts and ends with quoted
    sign text (`"OPEN" ... "CLOSED"`) has other quotes of the same kind inside and is kept.
    """
    text = _PREFIX.sub("", raw.strip()).strip()
    for left, right in _WRAPPERS:
        wrapped = len(text) > 1 and text.startswith(left) and text.endswith(right)
        inner = text[len(left) : -len(right)]
        if wrapped and left not in inner and right not in inner:
            return inner.strip()
    return text


def postprocess(user_prompt: str, raw: str, max_words: int) -> tuple[str, str | None]:
    """The prompt to generate from, and why the user's own prompt was kept (None: rewrite used)."""
    text = clean(raw)
    if not text:
        return user_prompt, EMPTY
    if len(text.split()) > max_words:
        return user_prompt, TOO_LONG
    if any(q not in text for q in quoted(user_prompt)):
        return user_prompt, QUOTED_TEXT_MISSING
    return text, None


@dataclass(frozen=True)
class Rewritten:
    prompt: str  # what generation uses: the rewrite, or the user's prompt on a fallback
    text: str  # the cleaned model output, kept even on a fallback so the record shows why
    fallback_reason: str | None
    style: str


class Rewrite:
    """Loads Qwen3-VL-8B with mlx-vlm and rewrites one prompt per call (§5.1 settings)."""

    def __init__(self, model_path: Path, style: str = DEFAULTS.rewrite_style) -> None:
        self.model_path = model_path
        self.style = RewriteStyle(style).value
        self._system = system_prompt(self.style)  # read now: a missing file fails before loading
        self._model: nn.Module | None = None
        self._processor: Any = None

    def load(self) -> None:
        from mlx_vlm import load

        memory.configure()
        self._model, self._processor = load(str(self.model_path))

    def __call__(self, prompt: str) -> Rewritten:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        if self._model is None:
            raise RuntimeError("Rewrite.load() was not called")
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": prompt},
        ]
        formatted = apply_chat_template(self._processor, self._model.config, messages, num_images=0)
        if not isinstance(formatted, str):  # text-only messages render to one string
            raise TypeError(f"unexpected chat template output: {type(formatted).__name__}")
        result = generate(
            self._model,
            self._processor,
            formatted,
            max_tokens=DEFAULTS.rewrite_max_tokens[self.style],
            temperature=DEFAULTS.rewrite_temperature,
            top_p=DEFAULTS.rewrite_top_p,
            seed=DEFAULTS.rewrite_seed,  # with temperature > 0: a position-keyed sampler
        )
        text = clean(result.text)
        final, reason = postprocess(prompt, text, DEFAULTS.rewrite_max_words[self.style])
        return Rewritten(final, text, reason, self.style)

    def unload(self) -> None:
        self._model = self._processor = None
        memory.evict()
