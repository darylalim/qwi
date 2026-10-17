"""Option B's encoder recipe (PLAN.md §2, §3) against fakes; the parity test needs real models."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qwi import models
from qwi.stages.encode import MAX_TOKENS, EncodeB, token_ids

ENCODER_MODULE = "mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder"
TEMPLATE = "<s>{}<e>"
PREFIX = "<s>"


class FakeTokenizer:
    """One id per character, so slicing is easy to read; records every call's options."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        self.calls.append((text, kwargs))
        return {"input_ids": [ord(c) for c in text]}


def test_token_ids_template_the_prompt_and_measure_the_prefix() -> None:
    tokenizer = FakeTokenizer()
    ids, drop = token_ids(tokenizer, "ab", TEMPLATE, PREFIX)
    assert ids == [ord(c) for c in "<s>ab<e>"]
    assert drop == 3
    # mflux's tokenizer: special tokens on, truncated at 2048; the prefix is measured without them.
    assert tokenizer.calls == [
        ("<s>ab<e>", {"truncation": True, "max_length": MAX_TOKENS}),
        (PREFIX, {"add_special_tokens": False}),
    ]
    assert MAX_TOKENS == 2048


@pytest.mark.parametrize("prompt", ["", "   "])
def test_token_ids_encode_an_empty_prompt_as_a_space(prompt: str) -> None:
    ids, _ = token_ids(FakeTokenizer(), prompt, TEMPLATE, PREFIX)
    assert ids == [ord(c) for c in "<s> <e>"]


class FakeTextStack:
    """Hidden state = the token id in both channels: the sliced result shows which ids survive."""

    def __init__(self) -> None:
        self.positions: list[np.ndarray] = []

    def __call__(self, ids: Any, position_ids: Any) -> Any:
        import mlx.core as mx

        self.positions.append(np.array(position_ids))
        return mx.repeat(ids[..., None].astype(mx.float32), 2, axis=-1)


@pytest.fixture
def fake_libraries(monkeypatch: pytest.MonkeyPatch) -> FakeTextStack:
    import mlx.core as mx

    stack = FakeTextStack()
    model = SimpleNamespace(language_model=SimpleNamespace(model=stack))
    processor = SimpleNamespace(tokenizer=FakeTokenizer())
    mlx_vlm = ModuleType("mlx_vlm")
    mlx_vlm.load = lambda path: (model, processor)  # ty: ignore[unresolved-attribute]  # test fake
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    encoder = ModuleType(ENCODER_MODULE)
    encoder.Qwen21PromptEncoder = SimpleNamespace(  # ty: ignore[unresolved-attribute]  # test fake
        PROMPT_TEMPLATE_T2I=TEMPLATE, SYSTEM_PREFIX=PREFIX
    )
    monkeypatch.setitem(sys.modules, ENCODER_MODULE, encoder)
    monkeypatch.setattr(mx, "set_cache_limit", lambda *_: None)
    return stack


def test_encode_b_drops_the_system_prefix_and_masks_every_token(
    fake_libraries: FakeTextStack,
) -> None:
    stage = EncodeB(Path("/weights/qwen3-vl-8b"))
    stage.load()
    (embeds,) = stage(["ab"])
    assert (embeds.prompt, embeds.encoder, embeds.tokens) == ("ab", "B", 5)
    assert np.array(embeds.embeds)[0, :, 0].tolist() == [ord(c) for c in "ab<e>"]
    mask = np.array(embeds.mask)
    assert (mask.shape, mask.dtype, mask.all()) == ((1, 5), np.int64, True)


def test_encode_b_replicates_one_position_ladder_across_the_mrope_axes(
    fake_libraries: FakeTextStack,
) -> None:
    stage = EncodeB(Path("/w"))
    stage.load()
    stage(["ab"])
    (positions,) = fake_libraries.positions
    assert positions.shape == (3, 1, 8)  # all of "<s>ab<e>": the prefix is dropped afterwards
    assert all(axis[0].tolist() == list(range(8)) for axis in positions)


def test_encode_b_encodes_each_distinct_prompt_once(fake_libraries: FakeTextStack) -> None:
    stage = EncodeB(Path("/w"))
    stage.load()
    assert [e.prompt for e in stage(["a", "b", "a"])] == ["a", "b"]
    assert len(fake_libraries.positions) == 2


def test_encode_b_before_load_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="load"):
        EncodeB(Path("/w"))(["p"])


def test_encode_b_unload_drops_the_model(fake_libraries: FakeTextStack) -> None:
    stage = EncodeB(Path("/w"))
    stage.load()
    stage.unload()
    with pytest.raises(RuntimeError, match="load"):
        stage(["p"])


# --- Parity with option A (PLAN.md §3) ------------------------------------------------------

PARITY_PROMPTS = [
    "a red apple on a wooden table",
    'A coffee shop sign reading "THE DAILY GRIND", warm morning light',
    "一只橘猫坐在窗台上",  # text-zh: a CJK prompt tokenizes differently
    "",
]


def cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-token cosine of two [tokens, hidden] arrays."""
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


@pytest.mark.slow
def test_option_b_matches_option_a_q4() -> None:
    # B's hand-built recipe vs. mflux's own on the q4 pack's encoder: same tokens, mask and dtype,
    # and mean per-token cosine ≥ 0.98, far above any deliberately broken recipe (the controls:
    # prefix one token off, no final norm, one layer short). No per-token floor: the user-turn
    # <|im_start|> varies most under the two quantizations. Figures (-s) in §8 2026-10-17.
    import gc

    import mlx.core as mx
    from mflux.models.qwen21.model.qwen21_text_encoder.qwen21_prompt_encoder import (
        Qwen21PromptEncoder,
    )

    from qwi.stages.generate import Generate

    def f32(x: mx.array) -> np.ndarray:
        return np.array(x.astype(mx.float32))[0]

    image = Generate(models.resolve("image"))
    image.load()
    try:
        model = image._model
        assert model is not None
        a_ids = [model.tokenizers["qwen21"].tokenize(p or " ").input_ids for p in PARITY_PROMPTS]
        a = image.encode(PARITY_PROMPTS)
    finally:
        image.unload()
    gc.collect()

    template, prefix = Qwen21PromptEncoder.PROMPT_TEMPLATE_T2I, Qwen21PromptEncoder.SYSTEM_PREFIX
    stage = EncodeB(models.resolve("rewrite"))
    stage.load()
    try:
        b = stage(PARITY_PROMPTS)
        assert stage._model is not None
        text = stage._model.language_model.model
        tokenizer = stage._tokenizer
        b_ids, controls = [], []
        for prompt in PARITY_PROMPTS:
            ids, drop = token_ids(tokenizer, prompt, template, prefix)
            b_ids.append(ids)
            x = mx.array([ids])
            pos = mx.broadcast_to(mx.arange(len(ids))[None, None, :], (3, 1, len(ids)))
            no_norm = text(x, position_ids=pos, apply_final_norm=False)
            short = text.norm(
                text(x, position_ids=pos, stop_after_layer=35, apply_final_norm=False)
            )
            controls.append((f32(no_norm[:, drop:]), f32(short[:, drop:])))
    finally:
        stage.unload()

    rows = []
    for prompt, ids_a, ids_b, ea, eb, (no_norm, short) in zip(
        PARITY_PROMPTS, a_ids, b_ids, a, b, controls, strict=True
    ):
        assert np.array(ids_a)[0].tolist() == ids_b, prompt  # same tokenizer and template
        assert (eb.embeds.shape, eb.embeds.dtype) == (ea.embeds.shape, ea.embeds.dtype), prompt
        assert eb.mask.dtype == ea.mask.dtype, prompt
        assert np.array(eb.mask).tolist() == np.array(ea.mask).tolist(), prompt
        xa, xb = f32(ea.embeds), f32(eb.embeds)
        cos = cosines(xa, xb)
        shifted = cosines(xa[1:], xb[:-1]).mean()  # the prefix dropped one token off
        rows.append((prompt, cos, shifted, cosines(xa, no_norm).mean(), cosines(xa, short).mean()))
        rel = np.linalg.norm(xa - xb) / np.linalg.norm(xa)
        worst = int(cos.argmin())
        print(
            f"{prompt[:24]!r:28} tokens {len(cos):3}  cosine min {cos.min():.5f} (token {worst}) "
            f"mean {cos.mean():.5f}  rel diff {rel:.4f} | controls: shifted {rows[-1][2]:.4f} "
            f"no-norm {rows[-1][3]:.4f} 35-layer {rows[-1][4]:.4f}"
        )
    best_control = max(max(r[2:]) for r in rows)
    assert best_control < 0.9, best_control  # the controls measure a broken recipe
    assert all(r[1].mean() >= 0.98 for r in rows), [(r[0], r[1].mean()) for r in rows]
