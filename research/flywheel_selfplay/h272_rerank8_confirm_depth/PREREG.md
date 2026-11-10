# H272: fresh-split confirmation of the H270 RERANK8 re-ranker, with descriptive depth arms, xbabe2 CPU

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t1910_opus_h272.md` (watcher / Grok Bot for Jepson,
2026-11-03 19:10Z, decided). **Claim:** `~/jope_plan/ns/receipts/H272_claim.txt`.

This is a diagnostic only. **No parameter of r882 or DOPE is updated.** The r882 sha256 and its DOPE identity
(`c51a56d0`) are checked before and after. Every policy picks one candidate from r882's own pool with **0 test-time
reveals**. The live flywheel and the H269/H271 files are never touched.

**Publication.** Under the PR #2071 split (MASTER_MERGE_HOLD active) this prereg is pushed to `origin/v1/next`
(non-force) before any outcome is read. The launch is bound to the published commit, its verified ancestry and the
sha256 of a durable same-agent review receipt.

## Hypothesis

**What is known.** H270 (`aee5dd042`) ADVANCED: RERANK8 − r882 = +0.0523 [+0.0171, +0.1060] on DL-Train OOF (Holm LCB
+0.0125, 3/3 seeds, placebo 0) and +0.0346 on DL-Val91. But it is one preregistered run. DL-Val91 has been reused by
H265–H270, RERANK8 made 2 invalid DL-Val91 picks against 0, and 64–68 of 205 families lose per seed.

**H272 asks:** does RERANK8's lift **replicate on fresh, never-used family splits and seeds**? And, descriptively, how
does the lift change with depth (K = 12, 16)?

## Frozen (byte-identical) from the H270 execution source

[topk.py](topk.py) and [h265_core.py](h265_core.py) are byte-identical to the H270 execution source (`v1/next`
`fb33cb6a1`, unchanged at `aee5dd042`; sha256 pinned in [manifest.json](manifest.json) and asserted by `test_h272.py`).
That covers the feature builder, labels, LGBMRanker grid, strict inner-FIT selector and margin rule. RERANK4 keeps H270's
salt-free random streams; the K ≥ 8 arms mix the fold seed into their streams, as in H270. **Only seeds change** for the
confirmatory arm. The depth arms change only K.

**Regression before compute.** The pre-publication smoke reproduces one real H270 cell (`F270140-S270101-K0`, RERANK8
at H270 seeds) exactly: the receipt and all 247 ASSESS picks must be equal. That reads only H270-published TRAIN
outcomes, not H272 outcomes. The smoke also checks Native91 1.1377941898164525 (91/91 bit-exact) and TOP4_PICK@r882.
The H265, H270 and H272 test files pass first.

## Arms

- **CONTROL** = r882 top-1.
- **RERANK8** = the confirmatory arm, at K = 8.
- **RERANK8_SHUF** = the within-family label-shuffle placebo.
- **RERANK4** = the H265/H270 reference at K = 4.
- **RERANK12, RERANK16** = descriptive depth arms. These are **not gated**; only K changes, and the slices are always
  full, since pools hold ≥ 1,151 candidates. I report where −2.0 picks and harmed families start rising.

## Design

- **PRIMARY: DL-Train repeated nested family CV, full 45-cell cross.** **Fresh** fold seeds 272140, 272141 and 272142
  (distinct from H265–H271) × 5 folds × model seeds 272101–272103. Per cell, FIT is 164 families and ASSESS is 41.
  - Per model seed, the gain is the **median over the 3 fold seeds** of the equal-family OOF gain. The statistic is the
    median over seeds, with a family bootstrap (10,000 draws, boot seed 272150).
- **Depth arms.** They also run on the full 45-cell cross, in a second phase after the confirmatory arms are locked
  (estimated runtime is well inside the window). No new depth cell starts after **21:15Z**. If that limit is hit, the
  depth panels are reported as truncated, and the confirmatory result is unaffected.
- **SECONDARY: DL-Val91**, each seed's mixture over its 5 fold-seed-272140 cells. DL-Val91 is heavily reused, so the
  DL-Train OOF on fresh folds is the decision endpoint.
- **TERTIARY (descriptive), leave-family-cluster-out.** The 41 families (20%) least similar to the rest form an outer
  panel.
  - **Similarity:** the Euclidean distance of each family's centroid of standardised `lm48_z` context descriptors (TRAIN
    only, outcome-free) from the mean family centroid. The list was computed before any outcome and is pinned in
    [fold_manifest.json](fold_manifest.json).
  - **Folds:** the other 164 families form 4 random folds (seed 272140) for the unchanged inner selector.
  - **Scope:** RERANK4, RERANK8 and RERANK8_SHUF, per model seed.
- **Locks.** Phase A (confirmatory), phase B (depth) and the tertiary selections are fsync-locked, in that order,
  **before any DL-Val91 feature, pool or label is read**. Each policy is scored once.

## Statistics and decision

- **Holm m = 2** over {PRIMARY, SECONDARY} for RERANK8 − r882. The Holm-level interval is the bootstrap [1.25%, 98.75%].
- **CONFIRMED** requires all six of:
  - PRIMARY median > **+0.002**;
  - Holm-level LCB > 0;
  - **3/3** seeds positive;
  - RERANK8 beats RERANK8_SHUF;
  - SECONDARY > 0 on **3/3** seeds;
  - RERANK8's DL-Train −2.0 picks (all 45 ASSESS folds) ≤ CONTROL's.

  Otherwise NOT_CONFIRMED / NO_ADVANCE. Fallback identity means NO_ADVANCE.
- **Also reported:**
  - MM Bradley–Terry Elo with r882 = 1500;
  - family W/T/L and harmed families, including those recurring from H265–H270;
  - RERANK8 − RERANK4;
  - the depth curve (RERANK12/16 − r882 and − RERANK8, with −2.0 counts);
  - the tertiary panel, the oracle headroom and context #27.
- **Not used.** No AUC, and no speed verdicts.

## Resources

- xbabe2, CPU only: `taskset` 44–47, nice 19, LightGBM n_jobs 1, ≤ 4 threads, RSS < 4 GiB, disk free ≥ 300 GB, under
  the H239 A1 guard (pause rather than abort; swap is full).
- **Deadlines.** Prereg by **19:55Z**; compute stop **21:40Z**; closeout by **22:25Z**. A single run with no resume
  path.
- **Durability.** Cell files and panels are written before the statistics. The statistics are packaging only: errors
  are recorded, not raised. Full outputs go to S3 before the ≤ 50 KB git `metrics.json` is derived.

## Custody

- Registrations are deferred under the freeze (no CI-config, records-pin, schema or root-README edits).
- S3: `…/rd_breakthrough_20261030/H272/<run_id>/`, create-only, with VersionId readback.
- Git gets small text only. The fold manifest is ≤ 50 KB. No model or `.npz`.
