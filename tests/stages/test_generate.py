"""The generate stage against a fake `QwenImage21`: no weights, no GPU."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import pytest
from PIL import Image

from qwi import memory
from qwi.errors import DeviceOutOfMemory
from qwi.stages.encode import PromptEmbeds
from qwi.stages.generate import Generate

MFLUX_MODULE = "mflux.models.qwen21.variants.txt2img.qwen_image_21"
ENCODER_MODULE = "mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder"


class FakeRegistry:
    def __init__(self) -> None:
        self.registered: list[Any] = []

    def register(self, callback: Any) -> None:
        self.registered.append(callback)


class FakeQwenImage21:
    instances: ClassVar[list["FakeQwenImage21"]] = []

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.callbacks = FakeRegistry()
        self.tokenizers = {"qwen21": "tokenizer"}
        self.text_encoder: object | None = "encoder"
        self.prompt_cache: dict[str, tuple[object, object]] = {}
        self.calls: list[dict[str, Any]] = []
        FakeQwenImage21.instances.append(self)

    def generate_image(
        self, *, seed: int, prompt: str, num_inference_steps: int, width: int, height: int
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "seed": seed,
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "width": width,
                "height": height,
            }
        )
        for callback in self.callbacks.registered:
            callback.call_in_loop(t=0)
        return SimpleNamespace(image=Image.new("RGB", (width, height)))


class FakePromptEncoder:
    """mflux's recipe as far as caching goes: an empty prompt is encoded as " "."""

    @staticmethod
    def encode_prompt(
        prompt: str, prompt_cache: dict, tokenizer: object, text_encoder: object
    ) -> tuple[str, str]:
        assert text_encoder == "encoder"
        key = prompt if prompt.strip() else " "
        prompt_cache[key] = (f"embeds:{key}", f"mask:{key}")
        return prompt_cache[key]


@pytest.fixture
def mlx_calls() -> list[str]:
    return []


@pytest.fixture(autouse=True)
def fake_mflux(monkeypatch: pytest.MonkeyPatch, mlx_calls: list[str]) -> None:
    import mlx.core as mx

    module = ModuleType(MFLUX_MODULE)
    module.QwenImage21 = FakeQwenImage21  # ty: ignore[unresolved-attribute]  # test fake
    monkeypatch.setitem(sys.modules, MFLUX_MODULE, module)
    encoder_module = ModuleType(ENCODER_MODULE)
    encoder_module.Qwen21PromptEncoder = FakePromptEncoder  # ty: ignore[unresolved-attribute]  # test fake
    monkeypatch.setitem(sys.modules, ENCODER_MODULE, encoder_module)
    FakeQwenImage21.instances = []
    for name in ("set_cache_limit", "reset_peak_memory"):
        monkeypatch.setattr(mx, name, lambda *_: None)
    monkeypatch.setattr(mx, "clear_cache", lambda: mlx_calls.append("clear_cache"))
    monkeypatch.setattr(mx, "eval", lambda *arrays: mlx_calls.append(f"eval{arrays}"))
    monkeypatch.setattr(mx, "get_peak_memory", lambda: 13_000_000_000)


def test_generates_one_candidate_per_seed_in_order() -> None:
    stage = Generate(Path("/weights/q4"))
    stage.load()
    candidates = list(stage("a red apple", [7, 8, 9], width=256, height=128, steps=2))

    model = FakeQwenImage21.instances[0]
    assert model.model_path == "/weights/q4"
    assert [c.seed for c in candidates] == [7, 8, 9]
    assert all(c.image.size == (256, 128) for c in candidates)
    assert all(c.peak_gb == pytest.approx(13.0) for c in candidates)
    assert model.calls[0] == {
        "seed": 7,
        "prompt": "a red apple",
        "num_inference_steps": 2,
        "width": 256,
        "height": 128,
    }


def test_defaults_come_from_config() -> None:
    stage = Generate(Path("/weights/q4"))
    stage.load()
    list(stage("p", [0]))
    call = FakeQwenImage21.instances[0].calls[0]
    assert (call["width"], call["height"], call["num_inference_steps"]) == (768, 768, 20)


def test_registers_callbacks_and_the_watchdog() -> None:
    hook = SimpleNamespace(call_in_loop=lambda **_: None)
    watchdog = memory.SwapWatchdog("generate", read=lambda: 0)
    stage = Generate(Path("/w"), callbacks=[hook], watchdog=watchdog)
    stage.load()
    assert FakeQwenImage21.instances[0].callbacks.registered == [hook, watchdog]


def test_watchdog_stops_the_remaining_seeds_after_a_swapping_decode() -> None:
    # Swap that happens after the last step hook (the VAE decode) is caught before the next seed.
    swap = iter([0, 0, 0, 10**10])  # start, pre-seed 0 check, seed 0 step hook, pre-seed 1
    watchdog = memory.SwapWatchdog("generate", limit_bytes=1, read=lambda: next(swap))
    watchdog.start()
    stage = Generate(Path("/w"), watchdog=watchdog)
    stage.load()
    candidates = stage("p", [0, 1, 2])
    assert next(candidates).seed == 0  # the swap is found after seed 0 is decoded and yielded
    with pytest.raises(DeviceOutOfMemory):
        next(candidates)
    assert len(FakeQwenImage21.instances[0].calls) == 1


def test_call_before_load_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="load"):
        list(Generate(Path("/w"))("p", [0]))


def test_unload_drops_the_model() -> None:
    stage = Generate(Path("/w"))
    stage.load()
    stage.unload()
    with pytest.raises(RuntimeError, match="load"):
        list(stage("p", [0]))


def test_encode_caches_the_embeddings_then_evicts_the_encoder(mlx_calls: list[str]) -> None:
    stage = Generate(Path("/w"))
    stage.load()
    (embeds,) = stage.encode(["a red apple"])
    model = FakeQwenImage21.instances[0]
    assert model.text_encoder is None
    assert model.prompt_cache == {"a red apple": ("embeds:a red apple", "mask:a red apple")}
    assert (embeds.prompt, embeds.encoder) == ("a red apple", "A-q4")
    # evaluated before the eviction frees the weights (§4)
    assert mlx_calls == ["eval('embeds:a red apple', 'mask:a red apple')", "clear_cache"]


@pytest.mark.parametrize("prompt", ["", "   "])
def test_encode_keys_an_empty_prompt_by_the_exact_string(prompt: str) -> None:
    # mflux encodes "" as " " but looks up "": without the exact key the lookup would miss and
    # call the evicted encoder.
    stage = Generate(Path("/w"))
    stage.load()
    stage.encode([prompt])
    assert prompt in FakeQwenImage21.instances[0].prompt_cache


def test_inject_caches_option_b_embeddings_without_running_the_encoder(
    mlx_calls: list[str],
) -> None:
    # FakePromptEncoder asserts the encoder is loaded: inject() must never call it.
    import mlx.core as mx

    empty, other = (PromptEmbeds(p, mx.zeros((1, 2, 4)), mx.ones((1, 2)), "B") for p in ("", "x"))
    stage = Generate(Path("/w"))
    stage.load()
    stage.inject([empty, other])
    model = FakeQwenImage21.instances[0]
    assert model.prompt_cache == {"": (empty.embeds, empty.mask), "x": (other.embeds, other.mask)}
    assert model.text_encoder is None
    assert mlx_calls == ["clear_cache"]


def test_inject_before_load_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="load"):
        Generate(Path("/w")).inject([])


def test_encode_before_load_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="load"):
        Generate(Path("/w")).encode(["p"])


def test_encode_batches_prompts_and_evicts_once(mlx_calls: list[str]) -> None:
    stage = Generate(Path("/w"))
    stage.load()
    embeds = stage.encode(["a", "b", "a"])
    assert [e.prompt for e in embeds] == ["a", "b"]  # duplicates encoded once
    assert set(FakeQwenImage21.instances[0].prompt_cache) == {"a", "b"}
    assert mlx_calls.count("clear_cache") == 1
