# H79 — W4 cached-pool screen: prior-top4 versus two-plus-two reveal

Status: **descriptive retrospective screening only; no W4 primary result or advancement verdict.**

## Why this shard

W4 ([PROPOSALS.md](../rd_breakthrough/PROPOSALS.md#w4--active-reveal-within-four-total-evaluations)) advances only over **nonadaptive prior-top4** at four total observations. [H67](../h67_w4_active_reveal_2plus2/README.md) froze the 50-context fit-only roster. The [H75 W4 shard](../h75_w4_cached_baseline_shard/README.md) scored random4 and hash_ordered4 only. Neither computed the prior-top4 comparator, a two-plus-two arm, family Elo or a bootstrap. W1's verified-U4 already sees four historical anchors (eight observations), so it is not the W4 zero-history four-fit endpoint. This shard runs the missing cached-pool screen with a non-neural prior. It spends zero native fits and zero GPU hours.

## Protocol (fixed before the run)

Contexts, seeds and utility are those of the H75 W4 shard: the frozen H67 roster (50 DL-Train regression contexts, 10 per H67 fold), seeds 701–703, the admitted H50 pools, and utility `-fitness` on gate pass, else `-2`. Each arm reveals exactly four distinct candidates through an oracle that returns only what it reveals. Retained-best is the best of those four.

| Arm | Selection |
| --- | --- |
| `random4` | The H75 W4 selector, imported unchanged; the run refuses unless its per-seed means equal the published ones exactly. |
| `prior10_top4` | Four lowest `prior10_cost` (the frozen r882 base program prior; it carries inherited training exposure). |
| `gbm_top4` | **Comparator and Elo anchor.** Four highest `mu = -oof_cost_prog412_ctx`, the W1 out-of-fold context-aware GBM prior. |
| `gbm_2plus2` | Reveal the GBM top two. Their residuals `u - mu` update an RBF Gaussian-process posterior mean over the pool. Reveal the top two of the remaining candidates under that mean. |
| `gbm_2plus2_swap` | Placebo: same, but the two residuals are swapped between the revealed programs. |

GP settings are fixed a priori and not tuned. The kernel is RBF on the W1 `proj16` program embedding. The lengthscale is the median pairwise distance over the context's full pool (outcome-blind), and the noise-to-signal ratio is 1. All rankings break ties by program SHA, so the prior arms are identical across seeds. If both revealed residuals are zero, `gbm_2plus2` equals `gbm_top4` exactly. The difference between the two arms therefore isolates what the two revealed outcomes change.

**Primary contrast:** `gbm_2plus2 - gbm_top4`. Seeds are averaged within a context and contexts within a family; each family has equal weight.

**Reported for every contrast:**
- family mean and median;
- fraction of families losing more than `.01`;
- family W/T/L at tie `1e-4`;
- a paired family-cluster bootstrap: 10,000 draws, seed `20261031`, one index matrix shared by all contrasts.

**Secondary (descriptive):** `gbm_top4 - random4`, `gbm_top4 - prior10_top4`, and `gbm_2plus2 - gbm_2plus2_swap`.

**Family Elo:** the canonical `python/src/jope/north_star/elo_tracker.py` (SHA `9ece9832…`, hash-checked) with `gbm_top4` anchored at 1500, tie `1e-4`, prior `.01` and 400 iterations. r882 is not in this cached panel, so the ratings are not r882-anchored.

## Leakage boundary

- The GBM prior is the S3-custodied W1 artifact `w1_train_only_priors.npz` (SHA `5213faa8…`). W1 built it per outer fold from fit families only.
- H67 and W1 number their seed-4900 folds differently, so exclusion is checked per family rather than by fold label. The runner refuses unless every acquisition family sits wholly inside one W1 fold. The out-of-fold model that scored a context therefore never saw any context of that context's family.
- No DL-Val, outer-test, sealed, holdout or sibling rows are read. No new S3 input prefix is read. DOPE stays at `c51a56d0`.
- Fit-only cached contexts are development inputs, not held-family evaluation. A cached screen cannot pass the W4 gate: the prospective stratum and matched elapsed time remain required.

## Reproduce

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 python3 -B -m unittest research.flywheel_selfplay.h79_w4_prior_2plus2_cached.test_run_shard -v

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 15 python3 -B research/flywheel_selfplay/h79_w4_prior_2plus2_cached/run_shard.py \
  --out ~/jope-data/h79_w4_prior_2plus2_cached/<run_id> \
  --s3-prefix s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W4/<run_id>
```

The defaults read `~/jope-data/h50/input`, the H67 roster, and W1's `cache/w1_train_only_priors.npz` and `fold_manifest.json`.
