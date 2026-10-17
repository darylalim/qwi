"""OCR hint for the `text-*` eval prompts: Qwen3-VL-2B transcribes each image (PLAN.md §5.4).

The 8B rewrites prompts, so a separate small model grades the text the images render. Each quoted
string from the eval prompt is then checked against the transcription (`exact`, and `match`
ignoring case). It's a hint, not a verdict: OCR makes mistakes, and the human check is final.

Measured on 2026-10-17 (§8 "OCR hint"): the model writes all-caps text in title case ("OPEN" →
"Open") whatever the prompt asks, and drops small detail (a 72 px "24" on a 768² image), so
`exact` misses most all-caps renders; `match` is the more useful hint for them.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qwi import memory
from qwi.config import DEFAULTS

if TYPE_CHECKING:
    import mlx.nn as nn


def _squash(text: str) -> str:
    return " ".join(text.split())


def exact(expected: Sequence[str], transcription: str) -> dict[str, bool]:
    """Each expected string → whether it appears verbatim in the transcription.

    Case, punctuation and script count; whitespace runs don't, since OCR breaks lines where a
    sign or poster wraps them ("THE DAILY\\nGRIND" matches "THE DAILY GRIND").
    """
    seen = _squash(transcription)
    return {s: _squash(s) in seen for s in expected}


def match(expected: Sequence[str], transcription: str) -> dict[str, bool]:
    """Like `exact`, ignoring case: the model title-cases all-caps text."""
    seen = _squash(transcription).casefold()
    return {s: _squash(s).casefold() in seen for s in expected}


class Ocr:
    """Loads Qwen3-VL-2B with mlx-vlm and transcribes one image file per call (greedy)."""

    def __init__(self, model_path: Path, prompt: str = DEFAULTS.ocr_prompt) -> None:
        self.model_path = model_path
        self.prompt = prompt
        self._model: nn.Module | None = None
        self._processor: Any = None

    def load(self) -> None:
        from mlx_vlm import load

        memory.configure()
        self._model, self._processor = load(str(self.model_path))

    def __call__(self, image: Path) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        if self._model is None:
            raise RuntimeError("Ocr.load() was not called")
        formatted = apply_chat_template(
            self._processor, self._model.config, self.prompt, num_images=1
        )
        if not isinstance(formatted, str):  # one text message renders to one string
            raise TypeError(f"unexpected chat template output: {type(formatted).__name__}")
        result = generate(
            self._model,
            self._processor,
            formatted,
            image=[str(image)],
            max_tokens=DEFAULTS.ocr_max_tokens,
            temperature=0.0,  # greedy: the same image always reads the same
        )
        return result.text.strip()

    def unload(self) -> None:
        self._model = self._processor = None
        memory.evict()
