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

## Result receipt

Run `luna-tel-h79-20261031T0601Z-01` used source commit `4120ee31e`, the protocol commit before this branch was rebased. After the rebase onto main it is `2fdee26ea`; every H79 file is identical, and `run_shard.py` SHA-256 `1bd88cec…` equals the manifest's `source_file_sha256`. It ran on xbabe2 at nice 15 with one BLAS thread: 37.6 s wall, 137 MB peak RSS. It covered 50 contexts / 44 family groups, three seeds and five arms: 3,000 cached reveals, zero native fits, zero GPU hours.

Both refusal checks passed:
- random4 reproduced the published H75 W4 per-seed means exactly (delta `0.0` on all three seeds).
- All 44 acquisition families sit inside one W1 fold. So do all 205 families, although 40 of the 50 contexts have a different fold label in H67 and W1.

| Arm | Family-mean retained-best utility | Family Elo |
| --- | ---: | ---: |
| `gbm_2plus2` | 1.9228 | 1523.04 |
| `gbm_top4` (anchor) | 1.9159 | 1500.00 |
| `gbm_2plus2_swap` | 1.8980 | 1514.79 |
| `random4` | 1.4130 | 1308.73 |
| `prior10_top4` | 1.3329 | 1261.32 |

| Contrast (44 families) | Mean | Median | Losing > .01 | Family W/T/L | Bootstrap 95% |
| --- | ---: | ---: | ---: | ---: | --- |
| **`gbm_2plus2 - gbm_top4` (primary)** | +0.0069 | 0.0000 | 6.8% | 14/18/12 | [−0.0169, +0.0341] |
| `gbm_top4 - random4` | +0.5029 | +0.0546 | 11.4% | 32/0/12 | [+0.1418, +1.0663] |
| `gbm_top4 - prior10_top4` | +0.5830 | +0.0718 | 15.9% | 34/1/9 | [+0.2345, +1.0346] |
| `gbm_2plus2 - gbm_2plus2_swap` | +0.0247 | 0.0000 | 0.0% | 2/42/0 | [0.0000, +0.0642] |

**Reading (descriptive, cached, fit-only):**
- **No screening signal for the two-plus-two update.** The two revealed outcomes change the second pair on 47 of 50 contexts, but retained-best barely moves: the median family delta is 0, and the interval includes zero (its 5th percentile is −0.0132, far below the common gate's +.001). The swap placebo is only slightly worse, so where the residual sits carries little usable information here.
- **The comparator matters more than the update.** The W1 out-of-fold GBM prior's top four beats random4 by +191 Elo, and only 1 of its 200 reveals failed the gate. The frozen r882 base prior10 ranks below random4 at zero history (−47 Elo against random4). This matches W1 track 2, where `RIDGE_PRIOR10_ONLY` trails `RANDOM4`. Any W4 active gain measured against prior10-top4 would mostly reflect a weak comparator. The prospective W4 comparator should be the strongest prior.
- **Construction cost is negligible either way:** selection CPU totals 0.07 s for `gbm_top4` and 0.16 s for `gbm_2plus2` over 150 context-seeds, against native fits that cost seconds each.

**Limits:**
- These are fit-only development contexts in a retrospective cached screen. There is no r882 history arm, no prospective new-program stratum, no `val_reg_inner`, and no matched elapsed-time native run.
- The prior arms are deterministic, so the three seeds replicate rather than resample them.
- One non-neural update rule is tested; it says nothing about a learned history policy.
- `scientific_verdict` and `advance_gate_evaluated` stay null/false.

Create-only S3 outputs under `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W4/luna-tel-h79-20261031T0601Z-01/` were downloaded at their version IDs, and their SHA-256 matched:

| Object | Version ID | SHA-256 |
| --- | --- | --- |
| `metrics.json` | `r9cYx.GPvvAW8cpYF_dTG7ORIZ3GcxQ7` | `3ec9f1210cc3525aed2ffa7cdb0e8bf947fb92a38b0c6d1acc5f1abe49954f3c` |
| `manifest.json` | `bMYnUZ0KC_llgJJpmlvFYQBLdXhRWVo8` | `929efdbace835faaef4fedc769956133b96a7c763d45a366c68ac0281cda4882` |
| `selection_trace.jsonl` | `twjEqZDe_Qq_YbR.IKsd0ta.ien3wCyR` | `1ccc2f9051ac61bd475ba159f78ff277cfca9887e1c3d9bf1e417d62b00feb7c` |

`metrics.json` and `run_manifest.json` in this folder are byte-identical copies. The selection trace stays in S3. The W1 prior input is S3 version `3JA4fEgIcT2kFF52jndixa.IapGN_1hl` (185,695,712 bytes; its local SHA equals W1's recorded custody hash). No HWM candidate was produced: the shard selects among existing cached programs only.
