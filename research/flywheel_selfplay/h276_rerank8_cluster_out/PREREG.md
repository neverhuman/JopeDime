# H276: leave-family-cluster-out test of the confirmed RERANK8 re-ranker, xbabe2 CPU

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t2030_opus_h276.md` (watcher / Grok Bot for Jepson,
2026-11-03 20:30Z, decided). **Claim:** `~/jope_plan/ns/receipts/H276_claim.txt`.

This is a diagnostic only. **Zero updates.** The r882 sha256 and its DOPE identity `c51a56d0` are checked before and
after. Every policy picks one candidate with **0 test-time reveals**. Sealed, holdout and meta-test data and the live
flywheel are never touched.

**Publication.** Under MASTER_MERGE_HOLD and the CI freeze, this prereg is pushed to `origin/v1/next` (non-force) before
any outcome is read. The launch is bound to the published commit, its verified ancestry and the sha256 of a durable
same-agent review receipt.

## Hypothesis

**What is known.** RERANK8 has replicated twice: H272 CONFIRMED +0.0633, and H273 (clean-room, rd_argon) REPLICATED
+0.0571. The open adoption risk is generalisation to *dissimilar* families. In H272's leave-cluster-out panel the lift
was +0.0173 [−0.0314, +0.0836] (not significant), RERANK8 was below RERANK4, and it made more invalid picks.

**H276 asks:** when **whole family clusters are held out**, does RERANK8 still lift single-pick utility over r882 (PASS
needs > +0.010)? Does the H273 invalid-gate veto keep that lift without extra invalid picks?

## Source (H272 science unchanged)

- [topk.py](topk.py) and [h265_core.py](h265_core.py) are **byte-identical** to the H272/H270 execution source (pinned
  and tested). That covers K = 8, the features, labels, LGBMRanker grid, strict selector and margin rule.
- **Regression first.** The smoke reproduces the H272 cell `F272140-S272101-K0` exactly (RERANK8 receipt, invalid,
  changed and deep-pick counts). It also checks Native91 (91/91 bit-exact) and TOP4_PICK@r882. `test_h276.py` passes
  before compute.
- **Only additions:** [veto.py](veto.py) (RERANK8_VETO) and the cluster-out fold construction.

## Arms

- **CONTROL** = r882.
- **RERANK4** = the H265 reference.
- **RERANK8** = H272 frozen.
- **RERANK8_VETO** = RERANK8 plus the **H273-specified invalid gate**:
  - an `LGBMClassifier(objective="binary", n_estimators=70, learning_rate=0.05, num_leaves=15, min_child_samples=25,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0, n_jobs=1)` predicts `y_invalid` (gate fails);
  - it is fit on TRAIN candidate rows only, using the H272 K = 8 per-candidate feature rows;
  - inner classifiers are fit on the inner-train folds only;
  - any slot k ≥ 1 with p_invalid ≥ **τ = 0.10** has its ranker score masked to −∞ before the unchanged margin
    selection, in inner FIT selection and at scoring alike;
  - the ranker random streams equal RERANK8's, so the two arms differ only by the mask.
- **RERANK8_SHUF_UNC** = the within-family label-shuffle placebo with the **fallback disabled**: its ASSESS picks use
  the unconstrained inner-best (lr, n_est, margin).

## Folds: leave-family-CLUSTER-out (pre-declared, outcome-blind)

- **Family descriptor.** The family mean of standardised per-context [lm48_z, dope629[:25], r882 cost mean, std,
  top-1/top-2 gap and top-8 mean, base-prior mean and std, proj16 mean]. No fitness, gate or utility enters.
- **Clusterings (k = 5).**
  - **(A) k-means** (n_init 10) with seeds 276140, 276141 and 276142;
  - **(B) Ward agglomerative**, which is deterministic.

  That makes **4 partitions × 5 held-out clusters**. The partitions were computed before any outcome and are pinned in
  [fold_manifest.json](fold_manifest.json) (sizes: KM276140 59/37/36/50/23; KM276141 73/49/59/23/**1**; KM276142
  54/30/46/55/20; WARD 53/39/50/55/8). The KM276141 singleton cluster (one outlier family) is kept as declared.
- **Cells.** Model seeds 276101–276103 give 4 × 5 × 3 = **60 cells**. In each cell the held-out cluster is ASSESS.
  Inner model selection uses the unchanged H265 selector over 4 **random family-grouped** folds of the training
  clusters only.
- **Lock.** All 60 cells' selections are fsync-locked before any DL-Val91 read.

## Statistics and decision

- **PRIMARY (confirmatory).** RERANK8 − CONTROL out-of-cluster single-pick, equal-family utility over all held-out
  clusters. Each partition yields one full 205-family OOF panel.
  - Per model seed, the gain is the median over the 4 partitions; the statistic is the median over seeds.
  - The CI comes from a **10,000-draw two-level cluster bootstrap** (seed 276150). Each draw resamples, within every
    partition independently, its 5 clusters with replacement, with families weighted by draw count.
  - **Holm m = 2** over {RERANK8, RERANK8_VETO}; the Holm-level LCB is the bootstrap 1.25% quantile.
- **PASS** (for RERANK8; RERANK8_VETO is evaluated with the same gates) requires all six of:
  - median > **+0.010**;
  - Holm-level LCB > 0;
  - **3/3** seeds positive;
  - **≥ 4/5 clusters with non-negative gain in ≥ 2/3 seeds, in every one of the 4 partitions**;
  - DL-Train −2.0 picks ≤ CONTROL's (all cells);
  - RERANK8_SHUF_UNC median ≤ **+0.005**.

  Ties keep r882. Fallback identity means NO_ADVANCE.
- **SECONDARY (descriptive):**
  - per-cluster gains, including the worst cluster;
  - RERANK8 − RERANK4 and RERANK8_VETO − RERANK8;
  - DL-Val91, supporting only because it is heavily reused, from the KM276140 cells' 5-model mixture per seed;
  - MM Bradley–Terry Elo with r882 = 1500;
  - harmed families and `fam-769243ee…`.
- **Not used.** No AUC, and no speed verdicts.

## Resources

- xbabe2, CPU only: `taskset` 44–47, nice 19, LightGBM n_jobs 1, ≤ 4 threads, RSS ≤ 4 GiB.
- Admission at load5 ≤ 100; pause at > 108.
- **Memory guard:** MemAvailable < 12 GiB sustained or < 6 GiB is a hard stop; < 20 GiB triggers a cooperative pause.
- **Swap:** swap is full but idle, so the guard watches the **swap-in rate** (pause above 2,000 pages/s, resume below
  200 for 30 s), **not swap_free**.
- **Deadlines.** Prereg by **21:05Z**; compute stop **23:00Z**; closeout by **23:45Z**. A single run with no resume.
  Statistics are packaging only: errors are recorded, not raised.

## Custody

- Full outputs and the cluster manifest go to S3 `…/rd_breakthrough_20261030/H276/<run_id>/` with VersionId first.
- Git gets small text only: the README, a metrics JSON under 50 KB and the index rows.
- No main, PR #2071, v1/fastlane or CI-config change. Registrations are deferred under the freeze.
