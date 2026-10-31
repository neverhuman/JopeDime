# H81 — W4 prospective stratum: prior-top4 versus two-plus-two on new programs

Status: **BLOCKED before scoring.** The preregistered comparator-strength guard failed, so no arm was scored and
**zero native fits were spent**. There is no W4 effect estimate or advancement verdict.

## Why this shard

[H79](../h79_w4_prior_2plus2_cached/README.md) and the [H75 W4 shard](../h75_w4_cached_baseline_shard/README.md)
screened only programs already in the cached pools. Both state that a cached screen cannot pass the W4 gate,
because the **prospective new-program stratum** and matched elapsed time are still required. H81 builds that
stratum: same frozen [H67](../h67_w4_active_reveal_2plus2/README.md) roster, same arms and same update rule, but
every revealed program is new and evaluated by native training CV.

## Protocol (fixed before any outcome)

- **Scope.** 50 fit-only DL-Train contexts (44 families) × seeds 701–703. For each context and seed, the pinned
  live proposer generates 16 new proposals that are not in the pool.
- **Arms.** Each arm reveals exactly four programs with zero history:
  - `gbm_top4` is the comparator and the Elo anchor.
  - `gbm_2plus2` uses H79's rule. A test checks that the `ranked`, `lengthscale` and `posterior_mean` functions
    are numerically identical to H79's merged code.
  - `gbm_2plus2_swap` is a placebo.
  - `random4` picks uniformly.
  - r882 is refused, because its API needs four revealed anchors.
- **Endpoint.** Retained-best original-cut utility, with seeds averaged within context and contexts within family.
  The primary contrast is `gbm_2plus2 − gbm_top4`, tested with the common gate on a 10,000-draw paired family
  bootstrap (seed 20261031). Family Elo uses the canonical `elo_tracker`. Matched elapsed time is reported per arm.
- **Comparator.** The W1 context-aware GBM recipe is refit per W1 outer fold: features prog412 + lm48_z + dope629,
  192 slots per context drawn by `default_rng(700)`, `random_state 800+k`, with W1's calibration. Each context is
  scored only by the model that excluded its whole family.
- **Budget.** At most 1,800 native fits, on xbabe2 only (the evaluator is x86-64), nice 15, load-gated, no GPU.
  Execution errors stay errors and make an arm INCOMPLETE.

## What happened

1. W1's prep did not retain its fitted per-fold models. Its manifest shows every shard with
   `cached_shards_reused: true` and 0 s elapsed. Only W1's out-of-fold predictions on existing pool slots survive,
   so **new programs cannot be scored by the exact W1 comparator**. A W1-recipe refit is not byte-identical to them:
   correlation with W1's stored predictions is 0.952 on average, 0.850 at minimum.
2. The guard therefore asked whether the refit is at least as strong a prior on the roster's admitted cached pools.
   Its cached `gbm_top4` family-mean retained-best was required to be within 0.02 of W1's stored predictions. It
   scored **1.8620 against 1.9159 (−0.054), so the guard failed** and the runner refused to score proposals. The
   refit's overall ranking quality is equivalent: held rank correlation with true utility is 0.700 for the refit and
   0.703 for W1.
3. A descriptive paired family bootstrap of the gap gives a mean of −0.054, **95% CI [−0.169, +0.047]**, median
   −0.003, and W/T/L 13/5/26 across 44 families. Because the interval includes zero, "weaker" is not established.
   The guard was fixed before the result, so it was not relaxed and the model was not tuned to pass it.

## How to unblock (either is enough)

- The W1 owner persists the fitted outer-fold context-aware models, as pickles with sha256, to S3. H81 then
  scores proposals with the exact W1 comparator and needs no refit.
- Or ASTRA or the owner preregisters, before any outcome, acceptance of this refit (`models.pkl` sha256
  `c8bfbe94f74d6ffe3f40b9848e43e22e832ada8672828d31f8ada647ee3096a1`) as the comparator, recording the descriptive
  gap above.

## Reproduce

```sh
python3 -m pytest -q research/flywheel_selfplay/h81_w4_prospective_2plus2/test_run_prospective.py   # 9 tests, no data
cd research/flywheel_selfplay/h81_w4_prospective_2plus2
python3 -I run_prospective.py models  --out ~/jope-data/h81/<run_id> --run-id <run_id>  # fit + guard
python3 -I run_prospective.py check   --out ~/jope-data/h81/<run_id> --run-id <run_id>  # guard only, fitted models
python3 -I run_prospective.py run     --out ~/jope-data/h81/<run_id> --run-id <run_id>  # refuses unless guard passed
python3 -I run_prospective.py analyze --out ~/jope-data/h81/<run_id> --run-id <run_id>
```

Inputs, all hash-pinned in the runner:

- the H50 packet's `selected.json` and pools;
- W1's fold manifest, prior manifest, train-only context features and stored priors;
- the pinned proposer and evaluator;
- the canonical `elo_tracker`.

No DL-Val, outer-test, sealed or holdout row is read, and DOPE is not involved.

## Evidence

[metrics.json](metrics.json) records the verdict, the guard, the gap bootstrap and all S3 objects. Those objects
were written create-only under `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W4/opus-h81-20261031a/`,
each with a version-pinned readback hash: the comparator check, the gap bootstrap, the fit log, `models.pkl` and
both sources. No HWM candidate was produced.
