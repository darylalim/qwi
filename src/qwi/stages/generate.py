"""Stage 4: Qwen-Image-2.1 via mflux `QwenImage21`, one image per seed (PLAN.md §1, §4).

Seeds run in sequence on one loaded model. mflux caches the prompt's embeddings in
`model.prompt_cache` on the first seed, so later seeds skip the encoder.
"""

import time
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL.Image import Image

from qwi import memory
from qwi.config import DEFAULTS

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
        if self._model is None:
            raise RuntimeError("Generate.load() was not called")
        for seed in seeds:
            if self.watchdog is not None:
                self.watchdog.check()  # swap since the last step hook (load, or the last decode)
            started = time.perf_counter()
            with memory.track_peak() as peak:
                image = self._model.generate_image(
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
