# qwi

Local text-to-image pipeline for Apple Silicon (M2 Max, 32 GB): Qwen3Guard → Qwen3-VL-8B rewrite → Qwen-Image-2.1 (mflux, MLX) × N seeds → Qwen3-VL-Reranker picks the best. CLI + Streamlit UI.

## PLAN.md is the source of truth

- **Before any task, read the relevant `PLAN.md` sections.** §6 lists milestones and tasks in order. §5.x has the spec each task points to (e.g. "§5.2" = guard policy). §2 has verified facts. §8 has decisions already made.
- **Work the next unchecked `- [ ]` task in the earliest unfinished milestone** (M0a → M0b → M1 → M2 → M3 → M4 → M5a → M5b → M5c; dependency map in §6). Don't skip ahead. A milestone is done only when its **Exit** line holds and the §6 "Definition of done" is met.
- **Don't reopen a §8 decision** without new evidence. If you have evidence (a measurement, a source-code finding), add a new dated §8 row explaining the change; don't silently diverge.
- **Keep the plan current as you go:**
  - tick `- [x]` when a task is done;
  - record measurements and findings in §8 (and §2 if they correct a fact);
  - put new questions in §9;
  - if code and plan disagree, fix one of them in the same change.
- Specs in the plan are written against **mflux 0.23.0, Streamlit 1.66.0, uv 0.12, ruff 0.17, ty 0.0.x**. If an installed version behaves differently, trust what you observe, note it in §8, and adapt.

## Commands

```bash
uv sync --all-extras                       # set up / sync
uv run ruff format && uv run ruff check --fix
uv run ty check
uv run pytest                              # fast tests (default: -m 'not slow')
uv run pytest -m slow                      # loads real models; local M2 Max only, ask first
# All gates (must pass before stopping or committing; hooks enforce this):
uv run ruff format --check && uv run ruff check && uv run ty check && uv run pytest
uv run qwi doctor                          # disk, models, offline, torch checks
uv run qwi runs show <id> --log            # a run's summary + JSON-lines log (§5.11)
uv run qwi ui                              # Streamlit UI
```

## Hard rules

- **Dependencies only via `uv add` / `uv remove`.** Never use `pip` or `uv pip install`, and never hand-edit `uv.lock`. Bump versions only with `uv version --bump …`. `mflux` stays pinned exactly (`==0.23.0`).
- **Never download weights or run real models without asking.** That includes `hf download`, `qwi models fetch`, `mflux-generate*`, `mflux-save`, and `pytest -m slow`. Free disk is tight (about 48 GB); keep **≥ 20 GB free** after any download (§5.7). Never start a bare `--model qwen-image-2.1` run (33 GB bf16 download).
- **Never commit weights, generated images, or `var/`.** The only images in git are `tests/fixtures/**` and the README screenshots in `docs/screenshots/`. Never copy third-party prompts, code, or docs unless their license is known and compatible (§5.10). Qwen's official rewrite prompt is reference-only.
- **The runtime is torch-free and offline.** mflux declares torch, but a `[tool.uv]` override keeps it uninstalled, because transformers imports it whenever it's present. Nothing may import it (`test_dependencies.py`, `test_no_torch.py`). The runtime sets `HF_HUB_OFFLINE=1`. Load models with the pinned revisions in `models.toml`.
- **Memory:** one stage's models at a time. Load → run → `del` + `gc.collect()` + `mx.clear_cache()` (§4). Never cache models in Streamlit. Only one pipeline run at a time (`$QWI_HOME/.lock`).
- **Failures:** raise the typed errors in `qwi.errors` (§5.11); every path evicts models and releases the lock in `finally`; write `run.json` atomically; never add automatic retries.
- **Imports:** `mlx`, `mflux`, `mlx_lm`, `mlx_vlm`, and `streamlit` are imported inside `load()` / the UI entrypoint, never at module level (`test_imports.py`). **Defaults** live only in `qwi/config.py`.
- **Output is RGB only:** mflux's text-to-image variant has no transparency (§1); don't add alpha handling to the core pipeline.
- **Paths:** always through `qwi.paths` (`$QWI_HOME`, default `<repo>/var/`). Tests set `QWI_HOME=<tmp_path>`.
- **UI:** a thin client of `Pipeline.run(request, on_event)`. The worker thread never calls `st.*`. Theming lives only in `.streamlit/config.toml`: no CSS, no hex colors in page code; use semantic names (`st.badge(color="violet")`, `st.warning`). Follow Streamlit's current API (`width="stretch"`, `st.navigation`/`st.Page`, `alt=` on images).
- **Releases come from CI only.** Never create tags or GitHub releases locally. Finishing a milestone = a minor version bump in a PR (§5.9).

## Code conventions

- Python 3.12, `src/qwi/` layout. Each stage is a small class with `load()` / `__call__()` / `unload()`. Typed boundaries (`RunRecord`, events, `PromptEmbeds`) so ty stays clean.
- Wrap untyped libraries (mflux, mlx-vlm) inside the stage modules. Fix type errors; the only allowed suppressions are rule-specific `# ty: ignore[rule]  # reason` at those boundaries.
- Put logic where fast tests can reach it without models (parsers, policy tables, guards, templates, path and storage helpers). Mark model-loading tests `@pytest.mark.slow`.
- Match existing style. ruff decides formatting and import order.

## Guardrails already in place (after M0a)

`.claude/settings.json` denies `pip`, edits to `uv.lock`/`var/`, and asks before downloads, model runs, `sudo`, tags, and pushes. Hooks:
- format and lint on every Python edit;
- all gates plus a staged-file check before `git commit`;
- all gates before stopping;
- session context at start.

If a hook blocks you, fix the cause; never bypass it (no `--no-verify`).

- `main` is protected by the "main: required CI" ruleset (lint, hygiene, test; no bypass, even for the owner). Always work on a branch and open a PR; a direct push to `main` is rejected.
- Run `git commit` in its own Bash call, separate from edits. H2 checks the whole command line before any of it runs, so `fix && git add && git commit` is checked against the unfixed tree and blocked.
