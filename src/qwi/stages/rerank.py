"""Stage 5: score candidates against the user's prompt with Qwen3-VL-Reranker-2B (PLAN.md §5.3).

The model has no scoring head: the score is the `yes` - `no` logit margin at the last position of
the upstream message format, and `prob = sigmoid(margin)`. Images are scored one at a time (no
padding). Everything around the forward pass is plain Python that fast tests reach without a
model: `messages()`, `to_rgb()`, `token_ids()`, `resized()` and `rank()`.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from qwi import memory
from qwi.config import DEFAULTS

if TYPE_CHECKING:
    import mlx.nn as nn

# The official scripts/qwen3_vl_reranker.py system prompt, word for word (§5.3).
SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".'
)


def messages(instruction: str, query: str, image: Image.Image) -> list[dict[str, Any]]:
    """The upstream message structure; the chat template joins the parts with no separators."""
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<Instruct>: " + instruction},
                {"type": "text", "text": "<Query>:"},
                {"type": "text", "text": query},
                {"type": "text", "text": "\n<Document>:"},
                {"type": "image", "image": image},
            ],
        },
    ]


def to_rgb(image: Image.Image) -> Image.Image:
    """Candidates are RGB (§1); anything else is composited onto white rather than converted.

    A plain `convert("RGB")` (what mlx-vlm does) drops alpha and shows whatever color the
    transparent pixels hold, often black.
    """
    if image.mode == "RGB":
        return image
    rgba = image.convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, rgba).convert("RGB")


def token_ids(tokenizer: Any) -> tuple[int, int]:
    """The `yes` and `no` ids; each must be a single token, or the margin means nothing."""
    vocab = tokenizer.get_vocab()
    ids = []
    for word in ("yes", "no"):
        encoded = tokenizer.encode(word, add_special_tokens=False)
        if word not in vocab or encoded != [vocab[word]]:
            raise ValueError(f"reranker tokenizer: {word!r} is not a single token ({encoded})")
        ids.append(vocab[word])
    return ids[0], ids[1]


def resized(grid_thw: Sequence[int], patch: int, image: Image.Image) -> bool:
    """Whether the processor changed the image size: grid (t, h, w) patches vs. its pixels.

    768² gives a 48 x 48 grid of 16 px patches: unchanged. Upstream caps an image at 1,280
    visual tokens and rounds each side to a multiple of 32, so larger or odd sizes change.
    """
    _, rows, cols = grid_thw
    return (rows * patch, cols * patch) != (image.height, image.width)


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    z = math.exp(x)  # no overflow for very negative margins
    return z / (1 + z)


@dataclass(frozen=True)
class Score:
    margin: float  # logit("yes") - logit("no") at the last position
    prob: float  # sigmoid(margin)
    resized: bool  # the processor resized the image (§5.3 "Image size")


@dataclass(frozen=True)
class Ranking:
    ranks: list[int]  # aligned with the candidates; 1 = best
    best_index: int
    low_confidence: bool  # the best prob is below --min-score: nothing clearly matched


def rank(scores: Sequence[Score], min_score: float = DEFAULTS.rerank_min_score) -> Ranking:
    """Highest margin first; on a tie the lower index (the earlier seed) wins (§5.3)."""
    if not scores:
        raise ValueError("nothing to rank")
    order = sorted(range(len(scores)), key=lambda i: (-scores[i].margin, i))
    ranks = [0] * len(scores)
    for position, index in enumerate(order, start=1):
        ranks[index] = position
    best = order[0]
    return Ranking(ranks, best, scores[best].prob < min_score)


class Rerank:
    """Loads the reranker with mlx-vlm and scores one (query, image) pair per call."""

    def __init__(self, model_path: Path, instruction: str = DEFAULTS.rerank_instruction) -> None:
        self.model_path = model_path
        self.instruction = instruction
        self._model: nn.Module | None = None
        self._processor: Any = None
        self._yes = self._no = -1

    def load(self) -> None:
        from mlx_vlm import load

        memory.configure()
        self._model, self._processor = load(str(self.model_path))
        self._yes, self._no = token_ids(self._processor.tokenizer)

    def __call__(self, query: str, image: Image.Image) -> Score:
        import mlx.core as mx
        from mlx_vlm.utils import prepare_inputs

        if self._model is None:
            raise RuntimeError("Rerank.load() was not called")
        image = to_rgb(image)
        text = self._processor.apply_chat_template(
            messages(self.instruction, query, image), tokenize=False, add_generation_prompt=True
        )
        inputs = prepare_inputs(self._processor, images=[image], prompts=text)
        out = self._model(
            inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            mask=inputs["attention_mask"],
            image_grid_thw=inputs["image_grid_thw"],
        )
        last = out.logits[0, -1].astype(mx.float32)
        margin = (last[self._yes] - last[self._no]).item()
        if not isinstance(margin, float) or not math.isfinite(margin):
            raise ValueError(f"reranker margin is not a finite number: {margin!r}")
        grid = inputs["image_grid_thw"].tolist()[0]
        patch = self._processor.image_processor.patch_size
        return Score(margin, sigmoid(margin), resized(grid, patch, image))

    def unload(self) -> None:
        self._model = self._processor = None
        memory.evict()
