# Session `2026-10-18_mflux-0.23_vs_turbo`

A = `mflux-0.23`, B = `turbo`; 24 items; noise floor 0.
Win = only B passes a check; loss = only A passes; tie = same result (PLAN.md §5.4).

## Verdict: B is **not worse**

passes rules 1-3.

| Rule | Measured | Result |
|---|---|---|
| 1 quality: losses - wins <= max(2, noise floor) | 0 | pass |
| 2 text: losses <= wins on `text-*` | 0 vs 0 | pass |
| 3a median time per image <= 1.15x A | 56.5 s vs 130.3 s (0.43x) | pass |
| 3b peak memory <= 24 GB | 15.0 GB (A 14.9) | pass |
| 3c zero swap on every image | 0 swapped (A 0) | pass |

## Overall

| | Wins | Losses | Ties | Sign test p |
|---|---|---|---|---|
| all checks | 2 | 2 | 58 | 1.000 |

## Per tag

| Tag | Wins | Losses | Ties | Sign test p |
|---|---|---|---|---|
| anatomy | 2 | 0 | 4 | 0.500 |
| binding | 0 | 0 | 10 | 1.000 |
| counting | 0 | 0 | 6 | 1.000 |
| isolation | 0 | 0 | 6 | 1.000 |
| layout | 0 | 0 | 4 | 1.000 |
| multilingual | 0 | 0 | 4 | 1.000 |
| negation | 0 | 0 | 6 | 1.000 |
| photoreal | 2 | 0 | 4 | 0.500 |
| rewrite-heavy | 0 | 0 | 6 | 1.000 |
| rewrite-light | 0 | 0 | 4 | 1.000 |
| spatial | 0 | 0 | 6 | 1.000 |
| style | 0 | 2 | 2 | 0.500 |
| text | 0 | 0 | 14 | 1.000 |

## Per check

| Check | Wins | Losses | Ties | Sign test p |
|---|---|---|---|---|
| `binding`: no swaps or leaks | 0 | 0 | 2 | 1.000 |
| `binding`: the cat has the red scarf, the dog has the blue hat | 0 | 0 | 2 | 1.000 |
| `count`: all red | 0 | 0 | 2 | 1.000 |
| `count`: exactly 5 apples | 0 | 0 | 2 | 1.000 |
| `count`: in a row | 0 | 0 | 2 | 1.000 |
| `detailed`: every stated element present | 0 | 0 | 2 | 1.000 |
| `detailed`: the rewrite is nearly identical to the original | 0 | 0 | 2 | 1.000 |
| `exclusion`: no cars | 0 | 0 | 2 | 1.000 |
| `exclusion`: no people | 0 | 0 | 2 | 1.000 |
| `exclusion`: the rewrite rephrases this positively ("deserted", "empty") | 0 | 0 | 2 | 1.000 |
| `isolated`: clean plain white background with no scene or shadow | 0 | 0 | 2 | 1.000 |
| `isolated`: single subject | 0 | 0 | 2 | 1.000 |
| `isolated`: the rewrite keeps "plain white background" | 0 | 0 | 2 | 1.000 |
| `photo-person`: golden-hour light | 0 | 0 | 2 | 1.000 |
| `photo-person`: natural face | 1 | 0 | 1 | 1.000 |
| `photo-person`: plausible hands and fingers | 1 | 0 | 1 | 1.000 |
| `spatial`: colors not swapped | 0 | 0 | 2 | 1.000 |
| `spatial`: cone on the left | 0 | 0 | 2 | 1.000 |
| `spatial`: cube on sphere | 0 | 0 | 2 | 1.000 |
| `style`: not photographic | 0 | 0 | 2 | 1.000 |
| `style`: reads as ukiyo-e (flat color, outlines, woodgrain texture) | 0 | 2 | 0 | 0.500 |
| `text-en`: legible | 0 | 0 | 2 | 1.000 |
| `text-en`: on the sign | 0 | 0 | 2 | 1.000 |
| `text-en`: sign text exactly `THE DAILY GRIND` | 0 | 0 | 2 | 1.000 |
| `text-multi`: both strings exact, including `·` | 0 | 0 | 2 | 1.000 |
| `text-multi`: title at top, subtitle at bottom | 0 | 0 | 2 | 1.000 |
| `text-zh`: sign reads exactly `清风茶舍` | 0 | 0 | 2 | 1.000 |
| `text-zh`: the rewrite is in English with the Chinese kept verbatim | 0 | 0 | 2 | 1.000 |
| `vague`: coherent scene | 0 | 0 | 2 | 1.000 |
| `vague`: reads as "cozy" | 0 | 0 | 2 | 1.000 |
| `vague`: the rewrite adds setting and lighting | 0 | 0 | 2 | 1.000 |

## Preference

B 8, A 16, same 0.
