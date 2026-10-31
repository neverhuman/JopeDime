# H83 — W6 cross-family transfer: cached held-family screen

Status: **cached screen complete — NO_ADVANCE.** Both primaries fail the common gate's harm-tail rule. This is not
a prospective W6 result, and no native fits were spent.

## Why this shard

W6 ([PROPOSALS.md](../rd_breakthrough/PROPOSALS.md), [H66 design](../h66_w6_cross_family_transfer/README.md),
[H72 roster](../h72_w6_train_roster_prep/README.md)) asks whether conditional preferences across programs and
structural neighbours improve new-family selection. The baselines are the generic program-only prior and current
raw-lm12 retrieval. [DELIVERY](../rd_breakthrough/DELIVERY.md) lists "W1 model outputs and admitted outcome
history" as prerequisites. W1's stored out-of-fold priors supply both, and they allow **leakage-free stacking**. For
outer fold k, `cal_cost_*_by_fold[k]` holds inner-pair out-of-fold costs on every fit-family slot and outer-model
costs on fold k's held slots. A fold-k residual model is therefore fit only on honest fit-family residuals and scored
on held families. Families never cross folds, and the W1 manifest records zero cross-family table aliases.

## Arms (fixed before any held-family read)

Every arm uses the same four hash-ordered anchors and the same unrevealed slots, with program-SHA tie-breaks.
The endpoint is verified **U4**: the best utility among the anchors and the arm's top-4 unrevealed slots.
U0 (anchors only) is 1.4582.

| Arm | Definition | Family-mean U4 | Family Elo |
|---|---|---:|---:|
| `PRIOR` (control, anchor) | W1 program-only prior, `mu = −cal_cost_prog412` | 1.6498 | 1500.0 |
| `TREAT` (registered) | PRIOR + rank-8 ridge bilinear residual on standardized lm48_z × proj16; ridge 10; residuals centred within context; fit on fit-family residuals | 1.6583 | 1523.3 |
| `SHUF` (placebo) | TREAT fitted with context features permuted across fit contexts | 1.6548 | 1509.0 |
| `KNN` | PRIOR + mean centred residual of the 32 nearest fit-family slots in proj16 | 1.6556 | 1490.9 |
| `RETR` (current retrieval) | Proximity to the champions of the 4 nearest fit-family contexts by lm48_z[:12] | 1.5591 | 1457.4 |
| `CTX` (reference) | W1 context-aware GBM prior, `−cal_cost_prog412_ctx` | 1.6804 | 1566.4 |
| `RANDOM4` (floor) | Four unrevealed slots in sha256-keyed order | 1.5773 | 1489.8 |

Elo uses the canonical `elo_tracker` (sha `9ece9832…`) with PRIOR anchored at 1500, tie 1e-4, prior 0.01 and
400 iterations, over 1,219 contexts in 205 families.

## Results (205 families; 10,000-draw paired family bootstrap, seed 20261031, shared draws)

| Contrast | Mean | Adjusted LCB | Median | Losing > .01 | W/T/L | Gate |
|---|---:|---:|---:|---:|---|---|
| **TREAT − PRIOR** (primary, Bonferroni 2) | +0.0085 | +0.0029 | 0.0000 | **13.2%** | 90/59/56 | fail (harm tail) |
| **TREAT − RETR** (primary, Bonferroni 2) | +0.0991 | +0.0404 | +0.0061 | **17.6%** | 138/5/62 | fail (harm tail) |
| TREAT − SHUF | +0.0035 | −0.0021 | 0.0000 | 18.0% | 90/38/77 | fail |
| KNN − PRIOR | +0.0058 | −0.0056 | 0.0000 | 19.0% | 70/55/80 | fail |
| CTX − PRIOR | +0.0306 | +0.0096 | +0.0040 | 16.1% | 134/8/63 | fail (harm tail) |
| PRIOR − RANDOM4 | +0.0725 | +0.0386 | 0.0000 | 31.7% | 101/7/97 | fail |

Primary LCBs are one-sided at α = 0.05/2; secondary LCBs are one-sided at α = 0.05.

**Reading (descriptive, cached, development folds):**

- **The registered treatment adds a small mean gain, but no demonstrable context-specific signal.** TREAT beats
  PRIOR on average (+0.0085, +23 Elo), but TREAT − SHUF is +0.0035 [−0.003, +0.010]. A context-shuffled residual of
  identical capacity does almost as well. On these data, the low-rank dataset × program interaction is not
  established as cross-family transfer.
- **W1's nonlinear context model is the strongest context-conditioned scorer**, at +66 Elo over PRIOR and
  +0.031 [+0.006, +0.059] U4. It still fails the 10% harm tail, with 16% of families losing more than .01.
- **Current raw-lm12 champion retrieval is worse than random** (−43 Elo against PRIOR, against −10 for RANDOM4).
- **Exact-program transfer has little to work with:** only 7.0% of held unrevealed programs (98,091 of 1,399,411)
  appear in any fit-family pool.
- **The harm tail binds every arm**, because per-family U4 is heavy-tailed. Any W6 learner must reduce downside
  losses, not only raise the mean, to clear the common gate.

**Limits:**

- This is a retrospective cached screen on development folds. There is no prospective new-program stratum, no
  r882 continuity arm (it needs the neural legacy path and has in-sample exposure), and no AST-template novelty split.
- W1's stored priors come from cached shards of an earlier W1 prep run (see H81, [PR #895](https://github.com/neverhuman/JopeDime/pull/895));
  this screen uses them as stored, pinned by sha256.

## Reproduce

```sh
python3 -m pytest -q research/flywheel_selfplay/h83_w6_cross_family_cached/test_run_w6_cached.py   # 6 tests, synthetic
cd research/flywheel_selfplay/h83_w6_cross_family_cached
OMP_NUM_THREADS=8 nice -n 15 python3 -I run_w6_cached.py --out ~/jope-data/h83/<run_id> --run-id <run_id>
```

Pinned inputs (refused on mismatch):

- `selected.json`
- the W1 fold manifest, stored priors and train-only context features
- the H58 `champions_featurized.json`, used for retrieval only
- the canonical `elo_tracker`

The admitted H50 training pools are also read. No DL-Val, outer-test, sealed or holdout rows are read, and DOPE is
not involved.

## Evidence

Run `opus-h83-20261031a` used xbabe2, nice 15 and 8 threads: 22:57 wall, 2.5 GB peak, zero native fits and zero GPU.
[metrics.json](metrics.json) records every number above, the fit receipts and the coverage census. The S3 objects
live under `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W6/opus-h83-20261031a/` (metrics,
gzipped per-context results, logs and both sources). Each was written create-only with a version-pinned readback
hash. No HWM candidate was produced.
