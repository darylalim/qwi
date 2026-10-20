# qwi

Local text-to-image pipeline for Apple Silicon: Qwen-Image-2.1 on MLX with a Qwen3 safety guard, prompt rewriting and best-of-N reranking. CLI + Streamlit UI.

Each run goes through four stages, one model in memory at a time:

1. **Guard:** Qwen3Guard checks the prompt (and later the rewrite). Unsafe prompts are blocked; how *Controversial* ones are handled depends on `--guard-mode`.
2. **Rewrite:** Qwen3-VL-8B expands a short prompt into a detailed description of the image. Turn it off with `--no-rewrite`, or rewrite first and edit the result.
3. **Generate:** Qwen-Image-2.1 (4-bit, through [mflux](https://github.com/filipstrand/mflux)) draws N candidates, one seed each.
4. **Rerank:** Qwen3-VL-Reranker scores each candidate against *your* prompt and marks the best. It warns when no candidate clearly matches.

Everything runs offline once the models are downloaded. Nothing is sent anywhere.

## Requirements

- **A Mac with Apple Silicon** and about **32 GB of memory**. qwi is developed on an M2 Max (32 GB). The largest stage peaks at about 14 GB, and a check before each stage stops a run that wouldn't fit.
- **About 20 GB of disk for the models**, plus **20 GB that must stay free** after any download (qwi refuses to download below that). Each default run adds 3–4 MB.
- **[uv](https://docs.astral.sh/uv/)**, which installs the right Python (3.12) for you.

A default run (4 candidates, 768², 20 steps) takes about **8½ minutes** on the M2 Max: about 2 minutes per image, and seconds for the other stages.

## Install

```bash
git clone https://github.com/darylalim/qwi.git
cd qwi
uv sync --all-extras    # the CLI, plus Streamlit for the UI
uv run qwi doctor       # checks disk, models, storage, offline mode
```

`doctor` reports the missing models until you download them (next section).

## Models

`qwi models fetch` is the only command that downloads anything. It fetches a role's weights from Hugging Face at the exact revision pinned in `models.toml`, after checking that 20 GB stays free:

```bash
uv run qwi models fetch image --dry-run   # size and free disk only
uv run qwi models fetch guard
uv run qwi models fetch rewrite
uv run qwi models fetch rerank
uv run qwi models fetch image
uv run qwi models fetch ocr               # optional: only the eval's text check uses it
```

| Role | Size on disk |
|---|---|
| guard | 1.1 GiB |
| rewrite | 5.4 GiB |
| rerank | 2.5 GiB |
| image | 8.9 GiB |
| ocr (eval only) | 2.5 GiB |

Weights go to the standard Hugging Face cache (`~/.cache/huggingface`). Runs always load with `HF_HUB_OFFLINE=1` and never download.

### Model licenses

This repository contains no model weights. Models are downloaded separately and keep their own licenses. The table is generated from `models.toml` (`tests/test_licenses.py` keeps it in sync):

| Role | Model | Revision | License |
|---|---|---|---|
| guard | [`mlx-community/Qwen3Guard-Gen-0.6B-MLX`](https://huggingface.co/mlx-community/Qwen3Guard-Gen-0.6B-MLX) | `919a4577` | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| rewrite | [`mlx-community/Qwen3-VL-8B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen3-VL-8B-Instruct-4bit) | `defcdea7` | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| rerank | [`mlx-community/Qwen3-VL-Reranker-2B-8bit`](https://huggingface.co/mlx-community/Qwen3-VL-Reranker-2B-8bit) | `e9cd8bbe` | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| ocr | [`mlx-community/Qwen3-VL-2B-Instruct-8bit`](https://huggingface.co/mlx-community/Qwen3-VL-2B-Instruct-8bit) | `b0338e0e` | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| image | [`mlx-community/Qwen-Image-2.1-mflux-q4`](https://huggingface.co/mlx-community/Qwen-Image-2.1-mflux-q4) | `746a5855` | [Qwen Research License (research and non-commercial use)](https://huggingface.co/Qwen/Qwen-Image-2.1) |

**Qwen-Image-2.1** is under the **Qwen Research License** (research and non-commercial use), and that applies to images generated with it. Repacks of it, such as `mlx-community/Qwen-Image-2.1-mflux-q4` (the `image` row above), are treated as Qwen Research License too, whatever their card says.

## Usage

### The UI

```bash
uv run qwi ui           # http://localhost:8501
```

- **Generate:** the prompt and settings on the left, candidates on the right as they finish. Cancel stops after the current step and keeps the finished images. **Rewrite only** shows the rewrite in an editable box first; **Generate from this prompt** uses your edit and still ranks against the original prompt. Each card has **View** (larger) and **Download**.
- **History:** every run, newest first, with its images, prompts, guard verdicts, timings and log. Star a run to keep it from `qwi runs prune`. Each run has its own link (`?run=<id>`).
- **Eval review:** blind side-by-side judging for the eval workflow (see Development).

The sidebar's **Thumbnail size** (S / M / L) and **Stacked layout** (one column, for half-screen windows) are kept in the URL, so a bookmark keeps them. Light and dark themes follow the system; switch them in the ⋮ menu. Only one run happens at a time, across the UI and the CLI.

### The CLI

```bash
uv run qwi generate "a vintage coffee shop sign reading \"THE DAILY GRIND\"" -n 4
uv run qwi generate "a red apple" --no-rewrite --seed 42 --steps 30
uv run qwi rewrite "a cozy reading nook"      # see the rewrite, then:
uv run qwi generate "<edited rewrite>" --no-rewrite
uv run qwi generate --continue-seeds <run-id> -n 4   # 4 more seeds for the same prompt
```

Images and a `run.json` go to `$QWI_HOME/runs/<run-id>/` (default `$QWI_HOME` is `var/` in the repo). `best.png` is a link to the top-ranked candidate.

| Command | What it does | Key options (default) |
|---|---|---|
| `qwi generate "<prompt>"` | The full run: guard, rewrite, generate, rerank | `-n` (4), `--steps` (20), `--width` / `--height` (768), `--seed` (random), `--guard-mode` strict \| default \| loose (default), `--no-rewrite`, `--rewrite-style` short \| long (long), `--show-rewrite`, `--min-score` (0.5), `--continue-seeds <run-id>`, `--allow-swap`, `-v` |
| `qwi rewrite "<prompt>"` | Guard and rewrite only; prints the result | `--guard-mode`, `--rewrite-style` |
| `qwi ui` | The Streamlit UI | `--port` (8501) |
| `qwi doctor` | Read-only health check | — |
| `qwi models fetch <role>` | Download a role's pinned weights | `--dry-run` |
| `qwi runs list` | Runs, newest first: id, date, status, star, size, prompt | — |
| `qwi runs show <id>` | One run's summary (an id prefix works) | `--log`, `--level` (info) |
| `qwi runs star <id>` | Toggle a run's star; prune never deletes starred runs | — |
| `qwi runs prune` | Delete old runs | `--keep-days N` or `--keep-last N`, `--eval` (eval images instead), `--dry-run` |
| `qwi eval …` | The eval workflow (see Development) | `--variant`, `--seeds` (1000,1001) |
| `qwi --version` | The installed version | — |

Every command has `--help`.

## Troubleshooting

Start with:

```bash
uv run qwi doctor                  # disk, models, $QWI_HOME, offline mode, torch
uv run qwi runs show <id> --log    # what a run did, step by step, with any traceback
```

A failed run keeps the candidates it finished, and every failure frees the models' memory and the run lock. qwi never retries on its own.

| Message | Exit code | What to do |
|---|---|---|
| Blocked by guard | 2 | The prompt (or its rewrite) was judged unsafe. Rephrase it; `--guard-mode loose` relaxes only *Controversial* verdicts, never *Unsafe*. |
| Another run is in progress | 3 | Wait for it, or cancel it in the UI. Only one run at a time. |
| Not enough free memory for *stage* | 4 | Close other apps, or use 512² or fewer candidates (the UI offers one-click retries). |
| Ran out of GPU memory during *stage* | 4 | Same as above. The finished candidates are kept, unranked. |
| *role* model not available | 5 | `uv run qwi models fetch <role>`. |
| Disk full / Not enough disk | 6 | Free space: `uv run qwi runs prune --keep-last 20` (try `--dry-run` first), or remove unused models with `hf cache`. |
| Cancelled after *n* of *N* candidates | 130 | Nothing; the finished candidates are kept. |
| Unexpected error | 1 | The traceback is in the run's log: `qwi runs show <id> --log`. |

If macOS starts swapping, a watchdog stops the stage after 2 GiB of swap. `--allow-swap` raises that to 6 GiB.

## Development

```bash
uv sync --all-extras
uv run ruff format && uv run ruff check --fix
# All gates (CI runs the same):
uv run ruff format --check && uv run ruff check && uv run ty check && uv run pytest
```

- `uv run pytest` runs the fast tests. They use a fake pipeline and need no models.
- `uv run pytest -m slow` loads the real models (they must be downloaded) and takes several minutes.
- Add dependencies with `uv add` / `uv remove` only. `mflux` is pinned exactly.
- The runtime must not import torch (mflux declares it; a `[tool.uv]` override keeps it out). The tests check this.
- To try the UI without models: `QWI_PIPELINE=fake QWI_HOME=/tmp/qwi uv run qwi ui`.

[`PLAN.md`](PLAN.md) holds the design, the decisions and the measurements behind them; [`CLAUDE.md`](CLAUDE.md) has the working rules.

### Eval workflow

Changes that can affect image quality (the mflux version, a model revision, the rewriter prompt, the reranker, the guard policy) are judged on a fixed set of 12 prompts (`eval/prompts.jsonl`), blind, before they merge:

```bash
uv run qwi eval --variant baseline       # on main → eval/results/<date>_baseline
uv run qwi eval --variant my-change      # on the branch → eval/results/<date>_my-change
uv run qwi eval session --a <date>_baseline --b <date>_my-change   # blind pairs, sides shuffled
uv run qwi eval review <session>         # judge in the browser (port 8502)
uv run qwi eval compare <session>        # unblind and score; writes report.md
```

`qwi eval session` prints the session's name (it lives in `eval/sessions/`). Results and sessions are committed. The images stay in `$QWI_HOME/eval`. CI fails a PR that touches one of these files without an `Eval:` line in its description.

## License

The code is licensed under the [Apache License 2.0](LICENSE); see [NOTICE](NOTICE). It does not cover model weights or images generated with them.
