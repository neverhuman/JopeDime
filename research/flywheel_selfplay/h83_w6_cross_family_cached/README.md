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

## Red-team checks (closeout t0105)

Each check below was verified against the inputs, not asserted.

- **Whole-family out-of-fold scoring.** Every held context in fold k is scored only by fold-k objects: the outer W1
  prior for fold k, a residual, placebo and kNN fit only on fit-family slots of folds ≠ k, and a retrieval bank
  built only from champions of folds ≠ k. The stacking prior on fit slots is W1's inner-pair out-of-fold prediction,
  which excludes both fold k and the slot's own fold. The W1 fold manifest assigns all 1,219 contexts. **0 of 205
  families span more than one fold**, and the manifest records 0 cross-family table aliases. There is therefore no
  same-family or same-fold leakage into any score.
- **Zero DL-Val, sealed or holdout reads.**
  - No DL-Val pool, label, feature or table was opened.
  - The W1 priors and the W1 context-feature file contain exactly the 1,219 DL-Train contexts, checked in
    `selected.json` train order, and the H58 `champions_featurized.json` holds 1,219 rows, all `DL-Train`.
  - Disclosure: `selected.json` is parsed whole. Its `dev` list (91 DL-Val **metadata** rows: identifiers and hashes,
    no outcomes) is loaded but never used.
  - No sealed, holdout or outer-test object exists in any input.
- **DOPE c51a56d0 untouched.** No neural model is loaded. The screen uses only stored W1 predictions, pools and
  linear algebra.
- **Family bootstrap.** Ten thousand paired family-cluster draws (seed 20261031) share one index matrix across all
  contrasts. Both the mean and the **median** have 95% intervals:

| Contrast | Family median [95% CI] | Family mean [95% CI] | Exact-tie families |
|---|---|---|---:|
| TREAT − PRIOR | 0.0000 [0.0000, +0.0002] | +0.0085 [+0.0029, +0.0143] | 28.8% |
| TREAT − RETR | +0.0061 [+0.0027, +0.0099] | +0.0991 [+0.0404, +0.1794] | 2.4% |
| TREAT − SHUF | 0.0000 [0.0000, +0.0001] | +0.0035 [−0.0030, +0.0100] | 18.5% |
| KNN − PRIOR | 0.0000 [0.0000, 0.0000] | +0.0058 [−0.0069, +0.0245] | 26.8% |
| CTX − PRIOR | +0.0040 [+0.0021, +0.0099] | +0.0306 [+0.0060, +0.0587] | 3.9% |
| PRIOR − RANDOM4 | 0.0000 [−0.0017, +0.0012] | +0.0725 [+0.0332, +0.1179] | 3.4% |

  TREAT's mean gain over PRIOR comes from a minority of families; the median family is unchanged. Only CTX − PRIOR
  and TREAT − RETR have a clearly positive median.
- **Elo, not AUC.** Every rating is a family-grouped Bradley–Terry Elo from the canonical `elo_tracker`. No AUC is
  computed or reported.
- **Ties do not count as wins.** A difference with |Δ| ≤ 1e-4 is a tie. Ties are reported as their own W/T/L column,
  and the Elo helper scores a tie as 0.5 to each side (`elo_tracker.py:205-216`), never as a win.

## Next step

- **The next free wave number is H85** (H80 and H82 are running on GPU; H84 is merged).
- **Recommended H85: a downside-controlled W6 learner, tested on this exact cached harness.** Every arm here fails the
  10% harm tail, and TREAT's gain sits in a minority of families. H85 should preregister a rule that keeps the prior
  unless the residual's adjustment is reliable. Examples: apply the adjustment only where its fit-fold bootstrap LCB
  is above zero, or rank by a fit-calibrated lower quantile. Score it on the same folds and anchors, with this
  placebo and these gates, so the result is comparable.
- **Prospective follow-up.** W1's context GBM (CTX) is the strongest scorer (+66 Elo, median +0.004). It is the
  natural comparator for a prospective W4/W6 new-program stratum. That stratum needs W1's fitted fold models
  persisted (H81 / PR #895).
- **Live-loop flag.** Current raw-lm12 champion retrieval ranks below random here (−43 Elo against PRIOR, −32 against
  RANDOM4). `propose_flywheel_candidates` uses that retrieval, so it merits a cheap ablation against
  context-model-ranked retrieval before any live change. That change would need separate owner approval.

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

Closeout addendum (t0105): `addendum-median_ci.json` in the same S3 prefix (create-only, version `9Pt490EfpfgnAS0hI.SDB9vchkaZaNv2`) holds the family-bootstrap median and mean intervals in the red-team table.
