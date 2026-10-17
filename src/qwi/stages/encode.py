"""Stage 3: prompt embeddings, the typed handoff into generation (PLAN.md §3, §4).

Every encoder option produces a `PromptEmbeds`; generation then reads it from mflux's
`model.prompt_cache` with no text encoder loaded. Option A (M1, "A-q4") runs mflux's own recipe on
the weight pack's encoder: `Generate.encode()`. Option B (M4, `EncodeB`) runs the same recipe on
the shared Qwen3-VL-8B through mlx-vlm, and `Generate.inject()` hands its `PromptEmbeds` over.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qwi import memory

if TYPE_CHECKING:
    import mlx.core as mx
    import mlx.nn as nn

# mflux's tokenizer truncates the templated prompt here (`Qwen21WeightDefinition`, 0.22.0).
MAX_TOKENS = 2048


@dataclass(frozen=True)
class PromptEmbeds:
    prompt: str  # the exact string `generate_image` receives: the prompt_cache key
    embeds: "mx.array"  # [1, tokens, hidden], final hidden states after the system prefix
    mask: "mx.array"  # [1, tokens]
    encoder: str  # "A-q4" | "A-bf16" | "B" (run.json `config.encoder`, §3)

    @property
    def tokens(self) -> int:
        return int(self.embeds.shape[1])


def token_ids(tokenizer: Any, prompt: str, template: str, prefix: str) -> tuple[list[int], int]:
    """The templated prompt's token ids and the system-prefix length to drop (§2 recipe).

    Mirrors mflux's `LanguageTokenizer` + `Qwen21PromptEncoder`: an empty prompt becomes " ",
    special tokens are added as the tokenizer defines them, and the ids are truncated at
    `MAX_TOKENS`.
    """
    text = template.format(prompt if prompt.strip() else " ")
    ids = tokenizer(text, truncation=True, max_length=MAX_TOKENS)["input_ids"]
    drop = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    return list(ids), drop


class EncodeB:
    """Option B: mflux's encoder recipe on Qwen3-VL-8B loaded with mlx-vlm (§3).

    The template and system prefix come from mflux's own `Qwen21PromptEncoder`, so only the
    forward pass differs from option A: the final-norm hidden states of mlx-vlm's text stack, with
    one position ladder replicated across the three mrope axes (text only, no vision tower).
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._model: nn.Module | None = None
        self._tokenizer: Any = None
        self._template = self._prefix = ""

    def load(self) -> None:
        from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import (
            Qwen21PromptEncoder,
        )
        from mlx_vlm import load

        memory.configure()
        self._template = Qwen21PromptEncoder.PROMPT_TEMPLATE_T2I
        self._prefix = Qwen21PromptEncoder.SYSTEM_PREFIX
        self._model, processor = load(str(self.model_path))
        self._tokenizer = processor.tokenizer

    def __call__(self, prompts: Sequence[str]) -> list[PromptEmbeds]:
        """Encode each distinct prompt once, in order, materialized so unloading frees the model."""
        import mlx.core as mx

        if self._model is None:
            raise RuntimeError("EncodeB.load() was not called")
        text = self._model.language_model.model  # mlx-vlm Qwen3-VL's text stack
        out: list[PromptEmbeds] = []
        for prompt in dict.fromkeys(prompts):
            ids, drop = token_ids(self._tokenizer, prompt, self._template, self._prefix)
            positions = mx.broadcast_to(mx.arange(len(ids))[None, None, :], (3, 1, len(ids)))
            hidden = text(mx.array([ids]), position_ids=positions)
            embeds = hidden[:, drop:, :]
            # One prompt per call, so no padding: mflux's mask is all ones (int64 from NumPy).
            mask = mx.ones((1, len(ids) - drop), dtype=mx.int64)
            mx.eval(embeds, mask)
            out.append(PromptEmbeds(prompt, embeds, mask, "B"))
        return out

    def unload(self) -> None:
        self._model = self._tokenizer = None
        memory.evict()
