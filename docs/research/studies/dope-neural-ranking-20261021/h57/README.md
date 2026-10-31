# H57 direct-utility antithetic ES from r882

## Hypothesis

Antithetic evolution strategies (ES) that optimize direct family-grouped DL-Train utility could improve the 29,211-parameter adapter, starting from an r882 copy, on the untouched seed-4900 fold B.

## Setup

Seeds 15/16/17 × sigma {0.001, 0.003}. Each run used 8 fixed generations, 16 antithetic pairs (population 32), and Adam lr 0.01. Only the final mean theta is used, with no generation selection. Fitness used fold A (164 families); fold B (41 families, 218 contexts) was scored once per candidate (6 reads), paired with the r882 copy. There is also a drift-only null arm (seed 15, sigma 0.001, permuted fitness), which is descriptive only. DOPE `c51a56d0` was byte-identical before and after every cell. No holdout, sealed, or DL-Val payloads were opened. Outcome run: `~/jope-data/h57/h57-wave-retry-20261031T0338Z`, prereg `2100c5d5`, source `19c57c29`, total GPU cell wall 4,029 s of the 5,400 s cap, lock wait 2,443 s (excluded from the cap).

## Outcome

| Arm | Pooled mean gain | Mean LCB95 (1-sided, family bootstrap) | Pooled family median [95% CI] | Families < −0.01 | Elo vs r882 by seed (15/16/17) | Admission |
|---|---:|---:|---:|---:|---|---|
| sigma 0.001 | +0.01558 | −0.02721 | 0.0000 [0.0000, +0.0041] | 22.0% | 1499.3 / 1537.7 / 1507.9 (median 1507.9) | No |
| sigma 0.003 | +0.01397 | −0.02735 | 0.0000 [−0.0048, +0.0014] | 31.7% | 1500.6 / 1502.0 / 1495.9 (median 1500.6) | No |

Win/tie/loss over the 218 fold-B contexts per cell: s001 35/148/35, 50/142/26, 40/141/37; s003 40/142/36, 40/139/39, 42/134/42.

## Verdict

No admission. The LCB95 is negative and harm exceeds 10% for both arms. ES is the only H46–H61 family with a positive pooled mean gain over r882, but the gain sits on a zero median: most contexts tie, a few improve, and about a quarter of families get worse. One seed (s001/16, +37.7 Elo) drives the arm, which is consistent with seed noise. No terminal DL-Val read was allowed.

Red-team: no leakage found (fold A/B family-disjoint, B read once per candidate, no best-generation pick). DOPE is frozen. Metric direction is higher-is-better utility (`utility = -fitness`). Ties fail. The r882 control saw fold B in its original training, so this comparison is biased against candidates (conservative). `build_metrics.py` had a repo-root bug (`HERE.parents[3]`), fixed to `parents[2]` in this PR. It is a closeout-only script and does not affect outcomes.

## Next

Do not promote. The positive mean with zero median motivates one longer ES test (H77), preregistered with more generations, a larger population, fresh seeds, and a family-balanced fitness. It queues FIFO behind H76.
