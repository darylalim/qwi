# Calibration `2026-10-17_bo4`

Results `2026-10-17_bo4`: 12 prompts x 4 candidates; 12 prompts have at least one acceptable candidate.
Top-1 = the candidate the reranker ranks first (ties go to the lowest seed). Top-1 acceptable counts only prompts with an acceptable candidate (PLAN.md §5.4).

## Instruction: **generic**

| Instruction | Top-1 = human best | Top-1 acceptable |
|---|---|---|
| `task` | 3/12 (25%) | 7/12 (58%) |
| `generic` | 5/12 (42%) | 8/12 (67%) |
| no reranker (base seed) | 4/12 (33%) | 8/12 (67%) |
| chance (random pick) | 3/12 (25%) | 7/12 (58%) |

- `task`: Given an image-generation request, judge whether the image faithfully depicts everything the request asks for.
- `generic`: Given a search query, retrieve relevant candidates that answer the query.

## `--min-score`: **0.5**

Every prompt had an acceptable candidate, so nothing to separate: keep 0.5; the warning is advisory.

## Per prompt (`generic`)

| Prompt | Top-1 | Human best | Acceptable | Top-1 prob |
|---|---|---|---|---|
| `binding` | 4 | 3 | 1, 3 | 0.85 |
| `count` | 2 | 1 | 1, 3 | 0.85 |
| `detailed` | 1 | 3 | 1, 3 | 0.87 |
| `exclusion` | 3 | 2 | 2, 4 | 0.68 |
| `isolated` | 1 | 1 | 1, 2 | 0.82 |
| `photo-person` | 2 | 4 | 2, 3, 4 | 0.83 |
| `spatial` | 4 | 4 | 1, 3, 4 | 0.91 |
| `style` | 2 | 2 | 2, 3 | 0.81 |
| `text-en` | 2 | 2 | 2, 3 | 0.87 |
| `text-multi` | 2 | 1 | 1, 4 | 0.87 |
| `text-zh` | 1 | 4 | 1, 2, 3, 4 | 0.81 |
| `vague` | 1 | 1 | 1, 3 | 0.53 |
