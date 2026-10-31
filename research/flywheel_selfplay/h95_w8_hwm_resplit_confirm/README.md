# H95 — W8 resplit confirmation of H94's 45 apparent high-water marks

Status: **complete.** Of the 45 apparent HWMs, **30 are confirmed** by the H72 common endpoint, and **27 are
material** (mean confirmation-resplit gain > .001). The material ones lie in 13 contexts and 12 families. Fifteen do
not survive resplitting, and 3 of the confirmed ones are complexity-only. These are confirmed candidates; none is
admitted or promoted.

## Hypothesis

H94 revealed 45 new programs, in 20 DL-Train contexts, whose single original-cut fitness beat the admitted pool's
best. [H63](../h63_training_resplits/README.md) showed that such original-cut wins often lose their sign when the same
training rows are re-cut. **Most H94 apparent HWMs should still beat the incumbent on fresh resplits, which no H94
decision used.**

## Setup (fixed in the runner before any H95 fit)

- **Challengers.** All 45 records in H94's `apparent_hwm.json`, pinned by sha256 to the published S3 object.
- **Incumbent.** The context champion in `champions.json`. In 4 of the 20 contexts, the pool's best program (whose DSL
  is not stored) beats the champion by ≤ 0.0007 fitness. Those rows are flagged (`champion_is_pool_best = false`),
  not dropped.
- **Cuts.** Each program is fit on the original cut (identity check), the W8 decision seeds 701–703 and the H72
  confirmation seeds 801–803. A resplit is a balanced permutation of the original fold bytes, with rows, features and
  targets unchanged. The evaluator and envelope are the merged
  [H72](../h72_w8_control_prereg/PREREG.md) helpers, pinned by full sha256.
- **Primary.** H72's common endpoint: CONFIRMED iff the mean over 801–803 of (u_challenger − u_incumbent) > 0.
- **Secondary.**
  - MATERIAL iff that mean is > .001, the common gate's LCB threshold.
  - The same rule on 701–703.
  - COMPLEXITY_ONLY iff the original-cut log ratio equals the incumbent's, so the gain is only the parenthesis
    penalty.
  - Family Elo over contexts: INCUMBENT = 1500 against BEST_CHALLENGER, the context's best challenger by original
    cut, scored on 801–803.
- **Execution errors.** These make a row INCOMPLETE and are never mapped to −2. There were none.

## Outcome

| | Count |
|---|---:|
| Apparent HWMs tested | 45 (20 contexts, 19 families) |
| Identity checks: challenger reproduces its H94 fitness / champion reproduces its pool utility | 45 / 45 |
| **Confirmed** (mean 801–803 gain > 0) | **30 (66.7%)** |
| **Material** (mean 801–803 gain > .001) | **27**, in 13 contexts and 12 families |
| Confirmed but complexity-only (exactly +0.0005 on every cut) | 3 |
| Confirmed on both 701–703 and 801–803 | 24 |
| Confirmed on 701–703 | 26 |
| Not confirmed (sign lost on fresh resplits) | 15 |

- **Size of the material gains.** The median mean 801–803 gain is +0.022. One context with a large utility scale
  gains +1.742; its challenger beats the champion on all seven cuts. Excluding it, the mean of the other 26 material
  gains is +0.023.
- **Family Elo.** The best challenger per context, by original cut, scores **+18.3 Elo** over the incumbent on
  801–803, with 11 wins, 0 ties and 9 losses across the 20 contexts. Picking each context's best by its single
  original cut is a poor rule. In 9 contexts the original-cut leader loses on fresh resplits; in 3 of those 9,
  another challenger is confirmed.
- **By the H94 arm that found them.** PRIOR_top4: 18 of 29 confirmed, 15 material. CTX_top4: 12 of 16 confirmed,
  12 material.

## Verdict and reading

- **Two-thirds of the apparent HWMs survive.** That is close to H63's 57.6% sign retention, and it confirms the W8
  lesson: an original-cut "improvement" is a candidate, not an improvement.
- **27 programs are material, resplit-confirmed improvements over the current champion in 13 contexts.** They come
  from prospective, never-seen proposals scored by a CPU-only pipeline. They are the most credible HWM candidates the
  R&D waves have produced so far. They are still unadmitted: promotion through the live gate remains a separate,
  owner-gated step.
- **Choosing by the original cut is the weak link.** The original-cut-best challenger loses in 9 of 20 contexts (in 3
  of them another challenger is confirmed), so any adoption should use the resplit-confirmed ranking
  (`confirmed_hwm.json`), not H94's original-cut ordering.

## Red-team checks

- **The confirmation seeds are independent of every decision.** H94 selected programs by model scores and the
  original cut. Seeds 801–803 were never used by H94, H63 or this run's flags, and the primary rule was copied from
  H72's merged preregistration.
- **Exact reproduction.** All 45 challengers reproduce their H94 original-cut fitness, and all champions reproduce
  their stored pool utility, to 1e-9.
- **Incumbent choice.** In 4 contexts the champion is not the pool best (gap ≤ 0.0007). A challenger confirmed against
  the champion could therefore fall short of the unstored pool best by up to that margin. Every row carries
  `champion_is_pool_best`.
- **Complexity-only wins are reported separately.** All three have an identical model and differ only in the
  parenthesis penalty.
- **Data scope.**
  - DL-Train only: train-only JTT1 tables, `champions.json` and the admitted pools.
  - Zero DL-Val, outer-test, sealed or holdout reads.
  - DOPE c51a56d0 is not loaded, and no GPU is used.
- **Elo, not AUC.** Ties are scored as 0.5.
- **Host.** One host. The evaluator is x86-64, so xbabe4 (aarch64) is ineligible, and xbabe1 hosts the live loop.
  At 455 fits taking about 1 s each, sharding across hosts would cost more than it saves.

## Next step

1. **Owner-gated adoption.** The 27 material programs, in `confirmed_hwm.json` on S3 and pointed in
   `HWM_CANDIDATES.jsonl`, are ready for the live admission path, if the owner chooses to promote R&D-found programs.
   This needs a GOV decision and is not done here.
2. **Rule lesson for the search loop.** Resplit confirmation removes a third of apparent wins at about 6 extra fits
   per candidate. That supports H72's reliability policy for promotions.

## Reproduce

```sh
python3 -m pytest -q research/flywheel_selfplay/h95_w8_hwm_resplit_confirm/test_run_resplit_confirm.py   # 4 synthetic tests
cd research/flywheel_selfplay/h95_w8_hwm_resplit_confirm
nice -n 15 python3 -I run_resplit_confirm.py --hwm <H94 apparent_hwm.json, sha256 8452d19f...> \
  --out ~/jope-data/h95/<run_id> --run-id <run_id> --workers 12
```

## Evidence

Run `opus-h95-20261031a` used xbabe2 at nice 15 with 12 workers: 0:42 wall, 455 native fits (492 s summed wall),
0 failures and zero GPU. [metrics.json](metrics.json) holds every count above, the confirmed program list and the S3
object versions. The objects live under
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W8/opus-h95-20261031a/`: manifest, metrics,
per-challenger rows, `confirmed_hwm.json`, plan, raw fit results, logs and sources. Each was written create-only with
a version-pinned readback hash. The `HWM_CANDIDATES.jsonl` pointer is version `Ea5HoyzrsytmNKz8NJ3pTMJwClFEojgz`,
written as a conditional append with the prefix preserved.
