# qwi

Local text-to-image pipeline for Apple Silicon: Qwen-Image-2.1 on MLX with a Qwen3 safety guard, prompt rewriting and best-of-N reranking. CLI + Streamlit UI.

## Model licenses

This repository contains no model weights. Models are downloaded separately and keep their own licenses. The table is generated from `models.toml` (`tests/test_licenses.py` keeps it in sync):

| Role | Model | Revision | License |
|---|---|---|---|
| guard | [`mlx-community/Qwen3Guard-Gen-0.6B-MLX`](https://huggingface.co/mlx-community/Qwen3Guard-Gen-0.6B-MLX) | `919a4577` | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| rewrite | [`mlx-community/Qwen3-VL-8B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen3-VL-8B-Instruct-4bit) | `defcdea7` | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| rerank | [`mlx-community/Qwen3-VL-Reranker-2B-8bit`](https://huggingface.co/mlx-community/Qwen3-VL-Reranker-2B-8bit) | `e9cd8bbe` | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) |

**Qwen-Image-2.1** is under the **Qwen Research License** (research and non-commercial use), and that applies to images generated with it. Its row is added once its weight source is pinned. Repacks of it, such as `mlx-community/Qwen-Image-2.1-mflux-q4`, are treated as Qwen Research License too, whatever their card says.

## License

The code is licensed under the [Apache License 2.0](LICENSE); see [NOTICE](NOTICE). It does not cover model weights or images generated with them.
