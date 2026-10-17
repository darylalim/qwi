# Session `2026-10-16_short_vs_long`

A = `short`, B = `long`; 24 items; noise floor 0.
Win = only B passes a check; loss = only A passes; tie = same result (PLAN.md §5.4).

## Verdict: B is **inconclusive**

a quality rule fails on 18 discordant pairs, but the sign test can't separate it from chance (p = 0.481): extend both variants to seeds 1002,1003 (`qwi eval --variant <name>-ext --seeds 1002,1003`), then `qwi eval session --extend 2026-10-16_short_vs_long --a <A-ext> --b <B-ext>`.

| Rule | Measured | Result |
|---|---|---|
| 1 quality: losses - wins <= max(2, noise floor) | 4 | **FAIL** |
| 2 text: losses <= wins on `text-*` | 0 vs 2 | pass |
| 3a median time per image <= 1.15x A | 130.2 s vs 125.4 s (1.04x) | pass |
| 3b peak memory <= 24 GB | 15.0 GB (A 15.0) | pass |
| 3c zero swap on every image | 0 swapped (A 0) | pass |

## Overall

| | Wins | Losses | Ties | Sign test p |
|---|---|---|---|---|
| all checks | 7 | 11 | 44 | 0.481 |

## Per tag

| Tag | Wins | Losses | Ties | Sign test p |
|---|---|---|---|---|
| anatomy | 2 | 1 | 3 | 1.000 |
| binding | 1 | 0 | 9 | 1.000 |
| counting | 0 | 0 | 6 | 1.000 |
| isolation | 0 | 6 | 0 | 0.031 |
| layout | 0 | 0 | 4 | 1.000 |
| multilingual | 0 | 0 | 4 | 1.000 |
| negation | 1 | 0 | 5 | 1.000 |
| photoreal | 2 | 1 | 3 | 1.000 |
| rewrite-heavy | 1 | 0 | 5 | 1.000 |
| rewrite-light | 0 | 2 | 2 | 0.500 |
| spatial | 1 | 0 | 5 | 1.000 |
| style | 0 | 2 | 2 | 0.500 |
| text | 2 | 0 | 12 | 0.500 |

## Per check

| Check | Wins | Losses | Ties | Sign test p |
|---|---|---|---|---|
| `binding`: no swaps or leaks | 0 | 0 | 2 | 1.000 |
| `binding`: the cat has the red scarf, the dog has the blue hat | 0 | 0 | 2 | 1.000 |
| `count`: all red | 0 | 0 | 2 | 1.000 |
| `count`: exactly 5 apples | 0 | 0 | 2 | 1.000 |
| `count`: in a row | 0 | 0 | 2 | 1.000 |
| `detailed`: every stated element present | 0 | 1 | 1 | 1.000 |
| `detailed`: the rewrite is nearly identical to the original | 0 | 1 | 1 | 1.000 |
| `exclusion`: no cars | 0 | 0 | 2 | 1.000 |
| `exclusion`: no people | 0 | 0 | 2 | 1.000 |
| `exclusion`: the rewrite rephrases this positively ("deserted", "empty") | 1 | 0 | 1 | 1.000 |
| `isolated`: clean plain white background with no scene or shadow | 0 | 2 | 0 | 0.500 |
| `isolated`: single subject | 0 | 2 | 0 | 0.500 |
| `isolated`: the rewrite keeps "plain white background" | 0 | 2 | 0 | 0.500 |
| `photo-person`: golden-hour light | 0 | 0 | 2 | 1.000 |
| `photo-person`: natural face | 1 | 0 | 1 | 1.000 |
| `photo-person`: plausible hands and fingers | 1 | 1 | 0 | 1.000 |
| `spatial`: colors not swapped | 0 | 0 | 2 | 1.000 |
| `spatial`: cone on the left | 0 | 0 | 2 | 1.000 |
| `spatial`: cube on sphere | 1 | 0 | 1 | 1.000 |
| `style`: not photographic | 0 | 0 | 2 | 1.000 |
| `style`: reads as ukiyo-e (flat color, outlines, woodgrain texture) | 0 | 2 | 0 | 0.500 |
| `text-en`: legible | 1 | 0 | 1 | 1.000 |
| `text-en`: on the sign | 0 | 0 | 2 | 1.000 |
| `text-en`: sign text exactly `THE DAILY GRIND` | 1 | 0 | 1 | 1.000 |
| `text-multi`: both strings exact, including `·` | 0 | 0 | 2 | 1.000 |
| `text-multi`: title at top, subtitle at bottom | 0 | 0 | 2 | 1.000 |
| `text-zh`: sign reads exactly `清风茶舍` | 0 | 0 | 2 | 1.000 |
| `text-zh`: the rewrite is in English with the Chinese kept verbatim | 0 | 0 | 2 | 1.000 |
| `vague`: coherent scene | 1 | 0 | 1 | 1.000 |
| `vague`: reads as "cozy" | 0 | 0 | 2 | 1.000 |
| `vague`: the rewrite adds setting and lighting | 0 | 0 | 2 | 1.000 |

## Preference

B 15, A 9, same 0.
