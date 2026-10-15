"""Stage 3: prompt embeddings, the typed handoff into generation (PLAN.md §3, §4).

Every encoder option produces a `PromptEmbeds`; generation then reads it from mflux's
`model.prompt_cache` with no text encoder loaded. Option A (M1, "A-q4") runs mflux's own recipe on
the weight pack's encoder: `Generate.encode()`. Option B (M4) will compute the same recipe on the
shared Qwen3-VL-8B and hand its `PromptEmbeds` to the generate stage.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx


@dataclass(frozen=True)
class PromptEmbeds:
    prompt: str  # the exact string `generate_image` receives: the prompt_cache key
    embeds: "mx.array"  # [1, tokens, hidden], final hidden states after the system prefix
    mask: "mx.array"  # [1, tokens]
    encoder: str  # "A-q4" | "A-bf16" | "B" (run.json `config.encoder`, §3)

    @property
    def tokens(self) -> int:
        return int(self.embeds.shape[1])
