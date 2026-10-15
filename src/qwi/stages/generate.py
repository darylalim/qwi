"""Stages 3-4: Qwen-Image-2.1 via mflux `QwenImage21`, one image per seed (PLAN.md §1, §3, §4).

`encode()` computes the prompt's embeddings once with the pack's own encoder (option A), caches
them in `model.prompt_cache`, and evicts the encoder before denoising: stock mflux would keep it
resident (≈ 4 GB at q4, §8). Seeds then run in sequence on one loaded model, all from the cache.
"""

import time
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL.Image import Image

from qwi import memory
from qwi.config import DEFAULTS
from qwi.stages.encode import PromptEmbeds

if TYPE_CHECKING:
    from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21


@dataclass
class Candidate:
    seed: int
    image: Image
    seconds: float
    peak_gb: float


class Generate:
    """Loads `QwenImage21` from a local weight snapshot and generates one candidate per seed.

    `callbacks` are mflux hooks (`call_in_loop`, ...) registered on load, such as a
    `memory.SwapWatchdog`. A watchdog is also checked before each seed: the VAE decode runs after
    the last step hook, so swap there stops the next seed while the decoded image is still kept.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        callbacks: Sequence[object] = (),
        watchdog: memory.SwapWatchdog | None = None,
    ) -> None:
        self.model_path = model_path
        self.callbacks = [*callbacks, *([watchdog] if watchdog else [])]
        self.watchdog = watchdog
        self._model: QwenImage21 | None = None

    def load(self) -> None:
        from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

        memory.configure()
        model = QwenImage21(model_path=str(self.model_path))
        for callback in self.callbacks:
            model.callbacks.register(callback)
        self._model = model

    def encode(self, prompts: Sequence[str], encoder: str = "A-q4") -> list[PromptEmbeds]:
        """Encode every prompt with mflux's recipe, cache them, then evict the encoder once (§4).

        Batch runs (the eval runner) pass all their prompts, so the encoder loads once.
        """
        import mlx.core as mx
        from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import (
            Qwen21PromptEncoder,
        )

        model = self._require()
        out: list[PromptEmbeds] = []
        for prompt in dict.fromkeys(prompts):  # each distinct prompt once, in order
            embeds, mask = Qwen21PromptEncoder.encode_prompt(
                prompt=prompt,
                prompt_cache={},
                tokenizer=model.tokenizers["qwen21"],
                text_encoder=model.text_encoder,
            )
            mx.eval(embeds, mask)  # materialize first, so evicting frees the weights now (§4)
            # Keyed by the exact prompt: mflux encodes an empty prompt as " " but looks up the
            # original string, which would miss and call the evicted encoder.
            model.prompt_cache[prompt] = (embeds, mask)
            out.append(PromptEmbeds(prompt, embeds, mask, encoder))
        model.text_encoder = None  # ty: ignore[invalid-assignment]  # evict (mflux types it non-None)
        memory.evict()
        return out

    def _require(self) -> "QwenImage21":
        if self._model is None:
            raise RuntimeError("Generate.load() was not called")
        return self._model

    def __call__(
        self,
        prompt: str,
        seeds: Sequence[int],
        *,
        width: int = DEFAULTS.width,
        height: int = DEFAULTS.height,
        steps: int = DEFAULTS.steps,
    ) -> Generator[Candidate]:
        """Yield each candidate as soon as it's decoded, so callers can save it and stop early."""
        model = self._require()
        for seed in seeds:
            if self.watchdog is not None:
                self.watchdog.check()  # swap since the last step hook (load, or the last decode)
            started = time.perf_counter()
            with memory.track_peak() as peak:
                image = model.generate_image(
                    seed=seed,
                    prompt=prompt,
                    num_inference_steps=steps,
                    width=width,
                    height=height,
                ).image
            yield Candidate(seed, image, time.perf_counter() - started, peak.gb)

    def unload(self) -> None:
        self._model = None
        memory.evict()
