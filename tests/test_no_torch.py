"""The runtime never imports torch (PLAN.md §8): run each stage's real code, then check sys.modules.

Slow: loads real models from the local HF cache at the pinned `models.toml` revisions, offline.
M1, M2 and M3 each add a test here for their new stage (PLAN.md §6).
"""

import gc
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

import qwi  # noqa: F401  # forces HF_HUB_OFFLINE=1 before huggingface_hub is imported

from .conftest import ROOT

pytestmark = pytest.mark.slow

CACHE_LIMIT = 2 * 2**30  # PLAN.md §4; the runtime takes it from config.py once M1 adds it


def snapshot(role: str) -> Path:
    from huggingface_hub import snapshot_download

    entry = tomllib.loads((ROOT / "models.toml").read_text())["roles"][role]
    return Path(snapshot_download(entry["repo"], revision=entry["revision"], local_files_only=True))


def assert_torch_free() -> None:
    assert "torch" not in sys.modules
    assert "torchvision" not in sys.modules


@pytest.fixture(autouse=True)
def mlx_memory() -> Iterator[None]:
    import mlx.core as mx

    mx.set_cache_limit(CACHE_LIMIT)
    yield
    gc.collect()
    mx.clear_cache()


def test_model_libraries_import_without_torch() -> None:
    import mflux  # noqa: F401
    import mlx_lm  # noqa: F401
    import mlx_vlm  # noqa: F401

    assert_torch_free()


@pytest.mark.parametrize("role", ["rewrite", "rerank"])
def test_qwen3_vl_processor_preprocesses_an_image(role: str) -> None:
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs
    from PIL import Image

    model, processor = load(str(snapshot(role)), lazy=True)
    prompt = apply_chat_template(processor, model.config, "Describe this image.", num_images=1)
    inputs = prepare_inputs(
        processor, images=[Image.new("RGB", (768, 768), "red")], prompts=[prompt]
    )
    # 768² in 16 px patches: no resize (PLAN.md §5.3).
    assert inputs["image_grid_thw"].tolist() == [[1, 48, 48]]
    assert_torch_free()


def test_qwen_image_generates() -> None:
    from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21

    model = QwenImage21(model_path=str(snapshot("image")))
    image = model.generate_image(
        seed=0, prompt="a red apple", num_inference_steps=2, width=256, height=256
    ).image
    assert (image.size, image.mode) == ((256, 256), "RGB")
    assert_torch_free()
