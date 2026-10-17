# Session `2026-10-16_baseline_vs_long_ext`

A = `baseline`, B = `long`; 48 items; noise floor 0.
Win = only B passes a check; loss = only A passes; tie = same result (PLAN.md §5.4).

## Verdict: B is **better**

not worse on every rule, and wins - losses = 24.

| Rule | Measured | Result |
|---|---|---|
| 1 quality: losses - wins <= max(2, noise floor) | -24 | pass |
| 2 text: losses <= wins on `text-*` | 4 vs 4 | pass |
| 3a median time per image <= 1.15x A | 129.8 s vs 123.5 s (1.05x) | pass |
| 3b peak memory <= 24 GB | 15.0 GB (A 14.9) | pass |
| 3c zero swap on every image | 0 swapped (A 0) | pass |

## Overall

| | Wins | Losses | Ties | Sign test p |
|---|---|---|---|---|
| all checks | 40 | 16 | 68 | 0.002 |

## Per tag

| Tag | Wins | Losses | Ties | Sign test p |
|---|---|---|---|---|
| anatomy | 9 | 0 | 3 | 0.004 |
| binding | 5 | 0 | 15 | 0.062 |
| counting | 1 | 0 | 11 | 1.000 |
| isolation | 0 | 3 | 9 | 0.250 |
| layout | 1 | 3 | 4 | 0.625 |
| multilingual | 2 | 0 | 6 | 0.500 |
| negation | 8 | 0 | 4 | 0.008 |
| photoreal | 9 | 0 | 3 | 0.004 |
| rewrite-heavy | 10 | 0 | 2 | 0.002 |
| rewrite-light | 3 | 5 | 0 | 0.727 |
| spatial | 1 | 0 | 11 | 1.000 |
| style | 0 | 4 | 4 | 0.125 |
| text | 4 | 4 | 20 | 1.000 |

## Per check

| Check | Wins | Losses | Ties | Sign test p |
|---|---|---|---|---|
| `binding`: no swaps or leaks | 4 | 0 | 0 | 0.125 |
| `binding`: the cat has the red scarf, the dog has the blue hat | 0 | 0 | 4 | 1.000 |
| `count`: all red | 0 | 0 | 4 | 1.000 |
| `count`: exactly 5 apples | 0 | 0 | 4 | 1.000 |
| `count`: in a row | 1 | 0 | 3 | 1.000 |
| `detailed`: every stated element present | 1 | 3 | 0 | 0.625 |
| `detailed`: the rewrite is nearly identical to the original | 2 | 2 | 0 | 1.000 |
| `exclusion`: no cars | 3 | 0 | 1 | 0.250 |
| `exclusion`: no people | 2 | 0 | 2 | 0.500 |
| `exclusion`: the rewrite rephrases this positively ("deserted", "empty") | 3 | 0 | 1 | 0.250 |
| `isolated`: clean plain white background with no scene or shadow | 0 | 0 | 4 | 1.000 |
| `isolated`: single subject | 0 | 3 | 1 | 0.250 |
| `isolated`: the rewrite keeps "plain white background" | 0 | 0 | 4 | 1.000 |
| `photo-person`: golden-hour light | 4 | 0 | 0 | 0.125 |
| `photo-person`: natural face | 3 | 0 | 1 | 0.250 |
| `photo-person`: plausible hands and fingers | 2 | 0 | 2 | 0.500 |
| `spatial`: colors not swapped | 0 | 0 | 4 | 1.000 |
| `spatial`: cone on the left | 0 | 0 | 4 | 1.000 |
| `spatial`: cube on sphere | 1 | 0 | 3 | 1.000 |
| `style`: not photographic | 0 | 0 | 4 | 1.000 |
| `style`: reads as ukiyo-e (flat color, outlines, woodgrain texture) | 0 | 4 | 0 | 0.125 |
| `text-en`: legible | 1 | 1 | 2 | 1.000 |
| `text-en`: on the sign | 0 | 0 | 4 | 1.000 |
| `text-en`: sign text exactly `THE DAILY GRIND` | 0 | 0 | 4 | 1.000 |
| `text-multi`: both strings exact, including `·` | 1 | 0 | 3 | 1.000 |
| `text-multi`: title at top, subtitle at bottom | 0 | 3 | 1 | 0.250 |
| `text-zh`: sign reads exactly `清风茶舍` | 1 | 0 | 3 | 1.000 |
| `text-zh`: the rewrite is in English with the Chinese kept verbatim | 1 | 0 | 3 | 1.000 |
| `vague`: coherent scene | 3 | 0 | 1 | 0.250 |
| `vague`: reads as "cozy" | 3 | 0 | 1 | 0.250 |
| `vague`: the rewrite adds setting and lighting | 4 | 0 | 0 | 0.125 |

## Preference

B 30, A 16, same 2.
