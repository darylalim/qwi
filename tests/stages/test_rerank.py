"""The reranker's message format, scoring and selection (PLAN.md §5.3); no model needed."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from qwi import models
from qwi.config import DEFAULTS
from qwi.stages.rerank import (
    SYSTEM,
    Rerank,
    Score,
    messages,
    rank,
    resized,
    sigmoid,
    to_rgb,
    token_ids,
)

# --- Messages -------------------------------------------------------------------------------


def test_messages_follow_the_upstream_structure() -> None:
    image = Image.new("RGB", (8, 8))
    assert messages("INSTR", "a red cube", image) == [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<Instruct>: INSTR"},
                {"type": "text", "text": "<Query>:"},
                {"type": "text", "text": "a red cube"},
                {"type": "text", "text": "\n<Document>:"},
                {"type": "image", "image": image},
            ],
        },
    ]
    assert SYSTEM == (
        "Judge whether the Document meets the requirements based on the Query and the Instruct "
        'provided. Note that the answer can only be "yes" or "no".'
    )


# --- Chat template snapshot (§5.3 Tests: catches template drift) ---------------------------

FIXTURES = Path(__file__).parents[1] / "fixtures" / "rerank"
# The pinned reranker's chat_template.jinja (Apache-2.0, see NOTICE): the processor renders
# with exactly this file (checked 2026-10-17), so CI can render it without the model cache.
TEMPLATE = FIXTURES / "chat_template.jinja"
RENDERED = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
    "<Instruct>: Given an image-generation request, judge whether the image faithfully depicts "
    "everything the request asks for.<Query>:a red cube on a white table\n"
    "<Document>:<|vision_start|><|image_pad|><|vision_end|><|im_end|>\n"
    "<|im_start|>assistant\n"
)


def render(msgs: list[dict[str, Any]], template: str) -> str:
    from transformers.utils.chat_template_utils import render_jinja_template

    (text,), _ = render_jinja_template([msgs], chat_template=template, add_generation_prompt=True)
    return text


def test_messages_render_to_the_upstream_sequence() -> None:
    image = Image.new("RGB", (768, 768))
    msgs = messages(DEFAULTS.rerank_instruction, "a red cube on a white table", image)
    # Parts joined with no separators; ends with the assistant turn whose first logit is scored.
    assert render(msgs, TEMPLATE.read_text(encoding="utf-8")) == RENDERED


@pytest.mark.slow
def test_the_fixture_is_the_pinned_models_template() -> None:
    # Needs the local HF cache (no weights load): a revision bump that changes the template
    # fails here, so the snapshot above is updated on purpose, not silently.
    import mlx_vlm.models.qwen3_vl.processing_qwen3_vl  # noqa: F401  # mlx-vlm's processor patch
    from mlx_vlm.utils import load_processor

    snapshot = models.resolve("rerank")
    assert (snapshot / "chat_template.jinja").read_bytes() == TEMPLATE.read_bytes()
    processor = load_processor(snapshot)
    msgs = messages(
        DEFAULTS.rerank_instruction, "a red cube on a white table", Image.new("RGB", (768, 768))
    )
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert text == RENDERED


# The §5.3 sanity test's fixed image: Qwen-Image-2.1 output (Qwen Research License, see
# NOTICE) from `qwi generate "a red cube on a white table" -n 1 --seed 0 --no-rewrite`.
RED_CUBE = FIXTURES / "red_cube.png"


@pytest.mark.slow
def test_the_correct_caption_scores_highest() -> None:
    image = Image.open(RED_CUBE)
    image.load()
    stage = Rerank(models.resolve("rerank"))
    stage.load()
    try:
        correct = stage("a red cube on a white table", image)
        wrong = [stage(q, image) for q in ("a blue sphere on a wooden floor", "a bowl of soup")]
    finally:
        stage.unload()
    assert correct.prob > 0.5
    assert all(correct.margin > w.margin for w in wrong), (correct, wrong)
    assert not correct.resized


# --- Image mode safeguard -------------------------------------------------------------------


def test_rgb_images_pass_through_unchanged() -> None:
    image = Image.new("RGB", (4, 4), "red")
    assert to_rgb(image) is image


@pytest.mark.parametrize(
    "image",
    [
        Image.new("RGBA", (4, 4), (0, 0, 0, 0)),  # transparent black: convert() would give black
        Image.new("LA", (4, 4), (0, 0)),
    ],
    ids=["RGBA", "LA"],
)
def test_transparent_pixels_become_white(image: Image.Image) -> None:
    out = to_rgb(image)
    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == (255, 255, 255)


def test_opaque_and_half_transparent_pixels_are_composited() -> None:
    image = Image.new("RGBA", (2, 1), (255, 0, 0, 255))
    image.putpixel((1, 0), (0, 0, 255, 128))
    out = to_rgb(image)
    assert out.getpixel((0, 0)) == (255, 0, 0)
    assert out.getpixel((1, 0)) == (127, 127, 255)


def test_grayscale_is_converted() -> None:
    assert to_rgb(Image.new("L", (2, 2), 200)).getpixel((0, 0)) == (200, 200, 200)


# --- Token ids ------------------------------------------------------------------------------


class Tokenizer:
    def __init__(self, vocab: dict[str, int], pieces: dict[str, list[int]] | None = None) -> None:
        self.vocab = vocab
        self.pieces = pieces or {}

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert not add_special_tokens
        return self.pieces.get(text, [self.vocab[text]] if text in self.vocab else [0, 1])


def test_yes_and_no_ids_come_from_the_vocab() -> None:
    # The cached Reranker-2B's ids (checked 2026-10-17).
    assert token_ids(Tokenizer({"yes": 9693, "no": 2152})) == (9693, 2152)


@pytest.mark.parametrize(
    "tokenizer",
    [
        Tokenizer({"no": 2152}),  # "yes" missing
        Tokenizer({"yes": 9693, "no": 2152}, {"yes": [88, 89]}),  # "yes" splits in two
        Tokenizer({"yes": 9693, "no": 2152}, {"no": [7]}),  # encodes to another id
    ],
    ids=["missing", "two-tokens", "other-id"],
)
def test_yes_and_no_must_be_single_tokens(tokenizer: Tokenizer) -> None:
    with pytest.raises(ValueError, match="not a single token"):
        token_ids(tokenizer)


# --- Resize detection -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "grid", "expected"),
    [
        ((768, 768), (1, 48, 48), False),  # 576 visual tokens, unchanged
        ((1024, 1024), (1, 64, 64), False),
        ((1536, 1536), (1, 70, 70), True),  # over the 1,280-token cap
        ((1000, 700), (1, 44, 62), True),  # (w, h); sides rounded to multiples of 32
    ],
)
def test_resize_is_read_from_the_patch_grid(
    size: tuple[int, int], grid: tuple[int, int, int], expected: bool
) -> None:
    # Grids measured with the cached processor (patch 16, merge 2), 2026-10-17.
    assert resized(grid, 16, Image.new("RGB", size)) is expected


# --- Scores and selection -------------------------------------------------------------------


def test_sigmoid() -> None:
    assert sigmoid(0.0) == 0.5
    assert sigmoid(2.11) == pytest.approx(0.89, abs=0.01)  # M0b's matching caption
    assert sigmoid(-2.89) == pytest.approx(0.05, abs=0.01)  # and its unrelated one
    assert sigmoid(-1000.0) == 0.0  # no overflow
    assert sigmoid(1000.0) == 1.0


def score(margin: float) -> Score:
    return Score(margin, sigmoid(margin), False)


def test_highest_margin_wins_and_ranks_follow() -> None:
    ranking = rank([score(0.5), score(2.0), score(-1.0), score(1.0)])
    assert (ranking.best_index, ranking.ranks) == (1, [3, 1, 4, 2])
    assert not ranking.low_confidence


def test_a_tie_goes_to_the_lower_index() -> None:
    ranking = rank([score(-1.0), score(1.5), score(1.5), score(1.5)])
    assert (ranking.best_index, ranking.ranks) == (1, [4, 1, 2, 3])


def test_low_confidence_when_the_best_prob_is_below_min_score() -> None:
    assert rank([score(-0.2), score(-1.0)]).low_confidence  # best prob 0.45 < 0.5
    assert not rank([score(0.0)]).low_confidence  # exactly 0.5 is not below it
    assert rank([score(1.0)], min_score=0.9).low_confidence  # 0.73 < 0.9


def test_nothing_to_rank() -> None:
    with pytest.raises(ValueError, match="nothing to rank"):
        rank([])


# --- The stage, against a fake mlx_vlm ------------------------------------------------------

YES, NO = 7, 3


@pytest.fixture
def fake_mlx_vlm(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import mlx.core as mx

    seen: dict[str, Any] = {"yes": 2.5, "no": 0.5, "grid": [[1, 48, 48]]}
    root, utils = ModuleType("mlx_vlm"), ModuleType("mlx_vlm.utils")

    class Processor:
        tokenizer = Tokenizer({"yes": YES, "no": NO})
        image_processor = SimpleNamespace(patch_size=16)

        def apply_chat_template(self, msgs: list, **kw: Any) -> str:
            seen["messages"], seen["template_kwargs"] = msgs, kw
            return "<formatted>"

    def model(input_ids: Any, **kw: Any) -> SimpleNamespace:
        seen["model_kwargs"] = kw
        logits = mx.zeros((1, 5, 10))
        last = mx.zeros((10,)).at[YES].add(seen["yes"]).at[NO].add(seen["no"])
        logits[0, -1] = last
        logits[0, 0] = mx.full((10,), 9.0)  # an earlier position the score must ignore
        return SimpleNamespace(logits=logits.astype(mx.bfloat16))

    def load(path: str) -> tuple[object, object]:
        seen["path"] = path
        return model, Processor()

    def prepare_inputs(processor: object, images: list, prompts: str) -> dict[str, Any]:
        seen["images"], seen["prompts"] = images, prompts
        return {
            "input_ids": mx.zeros((1, 5), dtype=mx.int32),
            "pixel_values": mx.zeros((4, 4)),
            "attention_mask": mx.ones((1, 5), dtype=mx.int32),
            "image_grid_thw": mx.array(seen["grid"]),
        }

    root.load = load  # ty: ignore[unresolved-attribute]  # building a fake module
    utils.prepare_inputs = prepare_inputs  # ty: ignore[unresolved-attribute]  # fake
    monkeypatch.setitem(sys.modules, "mlx_vlm", root)
    monkeypatch.setitem(sys.modules, "mlx_vlm.utils", utils)
    return seen


def test_stage_scores_the_yes_no_margin_at_the_last_position(
    fake_mlx_vlm: dict[str, Any],
) -> None:
    stage = Rerank(Path("/weights/rerank"))
    stage.load()
    image = Image.new("RGB", (768, 768))
    out = stage("a red cube", image)
    stage.unload()

    assert fake_mlx_vlm["path"] == "/weights/rerank"
    assert fake_mlx_vlm["messages"] == messages(DEFAULTS.rerank_instruction, "a red cube", image)
    assert fake_mlx_vlm["template_kwargs"] == {"tokenize": False, "add_generation_prompt": True}
    assert (fake_mlx_vlm["prompts"], fake_mlx_vlm["images"]) == ("<formatted>", [image])
    assert set(fake_mlx_vlm["model_kwargs"]) == {"pixel_values", "mask", "image_grid_thw"}
    assert out == Score(2.0, sigmoid(2.0), False)


def test_stage_reports_a_resize(fake_mlx_vlm: dict[str, Any]) -> None:
    fake_mlx_vlm["grid"] = [[1, 70, 70]]
    stage = Rerank(Path("/w"))
    stage.load()
    assert stage("q", Image.new("RGB", (1536, 1536))).resized


def test_stage_composites_non_rgb_input_before_scoring(fake_mlx_vlm: dict[str, Any]) -> None:
    stage = Rerank(Path("/w"), "Is it a cube?")
    stage.load()
    stage("q", Image.new("RGBA", (4, 4), (0, 0, 0, 0)))
    (sent,) = fake_mlx_vlm["images"]
    assert (sent.mode, sent.getpixel((0, 0))) == ("RGB", (255, 255, 255))
    assert fake_mlx_vlm["messages"][1]["content"][0]["text"] == "<Instruct>: Is it a cube?"


def test_stage_rejects_a_non_finite_margin(fake_mlx_vlm: dict[str, Any]) -> None:
    fake_mlx_vlm["yes"] = float("nan")
    stage = Rerank(Path("/w"))
    stage.load()
    with pytest.raises(ValueError, match="not a finite number"):
        stage("q", Image.new("RGB", (8, 8)))


def test_stage_must_be_loaded() -> None:
    with pytest.raises(RuntimeError, match=r"Rerank\.load"):
        Rerank(Path("/w"))("q", Image.new("RGB", (8, 8)))
