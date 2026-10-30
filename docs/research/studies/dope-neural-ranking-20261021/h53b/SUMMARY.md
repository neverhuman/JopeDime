# H53b summary

**Result: no admission.** All six fresh equal-budget cells completed on CUDA in preregistered order. The timing-only pilot and CPU infra-void attempt are excluded.

| Scale / seed | Rounds / updates | Final family utility | Mean gain / one-sided LCB95 | Median gain [95% CI] | Elo delta | W/T/L |
|---|---:|---:|---:|---:|---:|---:|
| 1 / 12 | 121 / 242 | 1.098315 | -0.0553 / -0.0931 | -0.0098 [-0.0944, 0.0000] | -74.25 | 18/35/38 |
| 1 / 13 | 185 / 370 | 1.045452 | -0.1082 / -0.2228 | -0.0093 [-0.0970, 0.0032] | -36.71 | 28/21/42 |
| 1 / 14 | 153 / 306 | 1.083345 | -0.0703 / -0.1375 | -0.0145 [-0.0747, 0.0176] | -36.71 | 29/19/43 |
| 0.1 / 12 | 211 / 422 | 1.124595 | -0.0290 / -0.0674 | 0.0000 [-0.0199, 0.0132] | -21.97 | 20/45/26 |
| 0.1 / 13 | 300 / 600 | 1.095493 | -0.0581 / -0.1074 | -0.0098 [-0.0622, 0.0089] | -62.85 | 19/36/36 |
| 0.1 / 14 | 300 / 600 | 1.096619 | -0.0570 / -0.1090 | 0.0000 [-0.0115, 0.0072] | -21.97 | 24/37/30 |

The r882 harness comparator is 1.1536153; higher family utility is better and ties fail. Every per-cell mean LCB95 is below zero. Elo is a pairwise diagnostic, not AUC.

The matched-horizon diagnostic is round 100 (R*=121, the minimum completed rounds); all six values and 25-round trajectories are in [README.md](README.md) and [metrics.json](metrics.json). Best-observed DL-Val reads are inadmissible. No sealed/holdout payload or outer row was opened/delivered.

See [README.md](README.md) for the full protocol, resource accounting, trajectory and overfit interpretation.
