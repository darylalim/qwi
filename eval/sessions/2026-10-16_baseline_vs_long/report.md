# Session `2026-10-16_baseline_vs_long`

A = `baseline`, B = `long`; 24 items; noise floor 0.
Win = only B passes a check; loss = only A passes; tie = same result (PLAN.md §5.4).

## Verdict: B is **inconclusive**

a quality rule fails on 27 discordant pairs, but the sign test can't separate it from chance (p = 0.052): extend both variants to seeds 1002,1003 (`qwi eval --variant <name>-ext --seeds 1002,1003`), then `qwi eval session --extend 2026-10-16_baseline_vs_long --a <A-ext> --b <B-ext>`.

| Rule | Measured | Result |
|---|---|---|
| 1 quality: losses - wins <= max(2, noise floor) | -11 | pass |
| 2 text: losses <= wins on `text-*` | 3 vs 1 | **FAIL** |
| 3a median time per image <= 1.15x A | 130.2 s vs 123.6 s (1.05x) | pass |
| 3b peak memory <= 24 GB | 15.0 GB (A 14.9) | pass |
| 3c zero swap on every image | 0 swapped (A 0) | pass |

## Overall

| | Wins | Losses | Ties | Sign test p |
|---|---|---|---|---|
| all checks | 19 | 8 | 35 | 0.052 |

## Per tag

| Tag | Wins | Losses | Ties | Sign test p |
|---|---|---|---|---|
| anatomy | 4 | 0 | 2 | 0.125 |
| binding | 3 | 0 | 7 | 0.250 |
| counting | 0 | 0 | 6 | 1.000 |
| isolation | 0 | 1 | 5 | 1.000 |
| layout | 0 | 2 | 2 | 0.500 |
| multilingual | 0 | 0 | 4 | 1.000 |
| negation | 5 | 0 | 1 | 0.062 |
| photoreal | 4 | 0 | 2 | 0.125 |
| rewrite-heavy | 4 | 0 | 2 | 0.125 |
| rewrite-light | 2 | 2 | 0 | 1.000 |
| spatial | 1 | 0 | 5 | 1.000 |
| style | 0 | 2 | 2 | 0.500 |
| text | 1 | 3 | 10 | 0.625 |

## Per check

| Check | Wins | Losses | Ties | Sign test p |
|---|---|---|---|---|
| `binding`: no swaps or leaks | 2 | 0 | 0 | 0.500 |
| `binding`: the cat has the red scarf, the dog has the blue hat | 0 | 0 | 2 | 1.000 |
| `count`: all red | 0 | 0 | 2 | 1.000 |
| `count`: exactly 5 apples | 0 | 0 | 2 | 1.000 |
| `count`: in a row | 0 | 0 | 2 | 1.000 |
| `detailed`: every stated element present | 1 | 1 | 0 | 1.000 |
| `detailed`: the rewrite is nearly identical to the original | 1 | 1 | 0 | 1.000 |
| `exclusion`: no cars | 2 | 0 | 0 | 0.500 |
| `exclusion`: no people | 1 | 0 | 1 | 1.000 |
| `exclusion`: the rewrite rephrases this positively ("deserted", "empty") | 2 | 0 | 0 | 0.500 |
| `isolated`: clean plain white background with no scene or shadow | 0 | 0 | 2 | 1.000 |
| `isolated`: single subject | 0 | 1 | 1 | 1.000 |
| `isolated`: the rewrite keeps "plain white background" | 0 | 0 | 2 | 1.000 |
| `photo-person`: golden-hour light | 2 | 0 | 0 | 0.500 |
| `photo-person`: natural face | 1 | 0 | 1 | 1.000 |
| `photo-person`: plausible hands and fingers | 1 | 0 | 1 | 1.000 |
| `spatial`: colors not swapped | 0 | 0 | 2 | 1.000 |
| `spatial`: cone on the left | 0 | 0 | 2 | 1.000 |
| `spatial`: cube on sphere | 1 | 0 | 1 | 1.000 |
| `style`: not photographic | 0 | 0 | 2 | 1.000 |
| `style`: reads as ukiyo-e (flat color, outlines, woodgrain texture) | 0 | 2 | 0 | 0.500 |
| `text-en`: legible | 1 | 1 | 0 | 1.000 |
| `text-en`: on the sign | 0 | 0 | 2 | 1.000 |
| `text-en`: sign text exactly `THE DAILY GRIND` | 0 | 0 | 2 | 1.000 |
| `text-multi`: both strings exact, including `·` | 0 | 0 | 2 | 1.000 |
| `text-multi`: title at top, subtitle at bottom | 0 | 2 | 0 | 0.500 |
| `text-zh`: sign reads exactly `清风茶舍` | 0 | 0 | 2 | 1.000 |
| `text-zh`: the rewrite is in English with the Chinese kept verbatim | 0 | 0 | 2 | 1.000 |
| `vague`: coherent scene | 1 | 0 | 1 | 1.000 |
| `vague`: reads as "cozy" | 1 | 0 | 1 | 1.000 |
| `vague`: the rewrite adds setting and lighting | 2 | 0 | 0 | 0.500 |

## Preference

B 14, A 9, same 1.
