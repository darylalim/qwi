## Summary

<!-- What changed and why. Link the PLAN.md milestone task (§6). -->

## Checklist

- [ ] "All gates" passed locally: `uv run ruff format --check && uv run ruff check && uv run ty check && uv run pytest`
- [ ] `uv run pytest -m slow` result, if this touches `src/qwi/stages/`, `pipeline.py` or `models.toml`: <!-- passed / n/a -->
- [ ] PLAN.md §8 decision row added, if this changes a decision
- [ ] No weights or generated images committed

Eval: <session path, or n/a — reason>

<!-- The Eval: line is required (CI's eval gate) when this PR touches an eval re-run trigger (PLAN.md §5.4):
     the mflux pin, models.toml, prompts/rewrite_system*.md, guard_policy.toml, stages/encode.py or stages/rerank.py. -->
