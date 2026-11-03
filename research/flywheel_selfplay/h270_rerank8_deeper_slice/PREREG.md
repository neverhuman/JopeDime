# H270: re-ranking a deeper slice (top-8) of r882's own pool (zero-update, single-pick), xbabe2 CPU

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t1830_opus_h270.md` (watcher / Grok Bot for Jepson,
2026-11-03 18:30Z, decided). **Claim:** `~/jope_plan/ns/receipts/H270_claim.txt`.

This is a diagnostic only. **No parameter of r882 or DOPE is updated.** The r882 sha256 and its recorded DOPE identity
(`c51a56d0`) are checked before and after. Every policy picks one candidate from r882's own pool with **0 test-time
reveals**. The live flywheel and the H268/H269 files are never touched.

**Publication.** Under the PR #2071 split this prereg is pushed to `origin/v1/next` (non-force) before any outcome is
read; nothing goes to main or v1/fastlane. The pre-publication smoke runs only the Native91 and TOP4_PICK@r882
regressions plus a synthetic K = 8 exercise, reading 0 TRAIN contexts. The launch is bound to the published commit,
its verified ancestry and the sha256 of a durable same-agent review receipt.

## Hypothesis

**What is known.** H265–H267 show the r882 top-4 re-ranker captures only about +0.013–0.019 of the ORACLE_TOP4
headroom (+0.10 on DL-Val91, +0.28 on DL-Train). H266 shows ORACLE_POOL far above ORACLE_TOP4, and H227/H229 showed
that wider *planner* pools do not help.

**H270 asks:** does re-ranking a **deeper slice of r882's existing pool (its top-8)** with the same zero-update ranker
lift DL-Train out-of-fold single-pick utility over r882?

## Frozen from H265; only K changes

- [h265_core.py](h265_core.py) is a byte-identical copy of the `v1/next` H265 runner. [topk.py](topk.py) is its
  K-generic form, and **at K = 4 it reproduces h265_core exactly**. `test_h270.py` asserts equality of the features,
  labels, LightGBM predictions, `run_cell` receipts and ASSESS picks.
- **What changes with K:**
  - the slice is r882's K lowest-cost candidates;
  - per-context z-features (slice z, slice softmax, slice spread (c[K−1] − c[0]) / sd) are computed **within the
    K-slice**;
  - the per-pair raw and z cost differences cover the other K − 1 slots, so the feature width is 568 at K = 8 versus
    560 at K = 4;
  - the slot position is j / (K − 1);
  - labels are K − 1 … 0 for valid candidates by ascending fitness, with gate-failed candidates at 0 (worst);
  - the LGBMRanker groups and the lambdarank truncation level are K.
- **What does not change:** the fixed context descriptors (the top-8 sem24 centroid and pool rows 0–3), the
  LGBMRanker recipe and its grid (lr {0.03, 0.05, 0.10} × n_est {50, 100, 200} × margin {0, …, 1, ∞}), the 4-fold
  inner-CV strict selector (gain > .001 and LCB95 > 0, else keep r882) and the margin rule.
- **Slice depth.** Every pool holds 1,151–1,152 candidates, so every context has a full K = 8 slice; the runner asserts
  ≥ 8. Gate-failed candidates inside the slice stay selectable, as in H265, but are labelled worst. Their counts are
  reported, not hidden.

## Arms

- **CONTROL** = r882 top-1.
- **RERANK4** = H265 frozen (K = 4; reference), run with H265's own random streams.
- **RERANK8** (primary) = the same pipeline over r882 slots 0–7.
- **RERANK8_SHUF** = RERANK8 with labels permuted within family (the H265 shuffle).

RERANK8 and RERANK8_SHUF mix the fold seed into every random stream. Descriptive only: ORACLE_TOP4, ORACLE_TOP8 and
ORACLE_POOL headroom, and the share of RERANK8 picks from slots 4–7.

## Design

- **Primary: DL-Train repeated nested family CV, full 45-cell cross.** 3 fresh fold seeds (270140, 270141, 270142) × 5
  folds × 3 model seeds (270101–270103). Per cell, FIT is 4 folds (164 families) and ASSESS is the held-out fold.
  - Every TRAIN context is scored out of fold once per (fold seed, model seed).
  - Per model seed, the gain is the **median over the 3 fold seeds** of the equal-family OOF gain. The statistic is the
    median over the 3 model seeds, with a family bootstrap (10,000 draws, boot seed 270150) resampling families jointly.
- **Secondary: DL-Val91** single-pick, each model seed's mixture over its 5 fold-seed-270140 cells.
  - **Disclosure:** DL-Val91 has been read by many waves (H241–H269), so it is heavily reused. **DL-Train OOF on fresh
    folds is the decision endpoint.**
- **Lock.** All 45 cells' selections are fsync-locked in one `selection.json` **before any DL-Val91 feature, pool or
  label is read**. Each policy is scored once.
- **Regression first.** Native91 1.1377941898164525 (91/91 bit-exact) and TOP4_PICK@r882 (1.2403601682250796 /
  1.2519997168684853), before any TRAIN packet. The H265 tests and `test_h270.py` pass before compute.

## Statistics and decision

- **Holm m = 2** over {PRIMARY, SECONDARY} p-values for RERANK8 − r882. The Holm-level interval is the bootstrap
  [1.25%, 98.75%].
- **ADVANCE** requires all six of:
  - PRIMARY median family gain > **+0.002**;
  - PRIMARY Holm-level LCB > 0;
  - PRIMARY **3/3 model seeds positive**;
  - **RERANK8 beats RERANK8_SHUF** (PRIMARY median > 0);
  - SECONDARY > 0 on **3/3 seeds**;
  - RERANK8's DL-Train −2.0 picks (summed over all 45 cells' ASSESS) **≤ CONTROL's**.

  Otherwise NO_ADVANCE. Fallback identity or ties are NO_ADVANCE.
- **Also reported:**
  - RERANK8 − RERANK4 (descriptive);
  - per-seed and per-fold-seed gains, W/T/L and harmed families;
  - MM Bradley–Terry Elo with r882 = 1500;
  - −2.0 counts per arm on DL-Train and DL-Val91;
  - oracle headroom;
  - context #27, reported separately.
- **Not used.** No AUC, and no speed verdicts.

## Resources and the A1 memory guard (H239 A1)

- xbabe2, CPU only: `taskset` 44–47, nice 19, LightGBM n_jobs 1, ≤ 4 threads, RSS < 4 GiB, disk free ≥ 300 GB.
- No GPU, xbabe0, xbabe1 writes or xbabe3. Admission at load5 ≤ 100 and MemAvailable ≥ 30 GiB. Cooperative pause at
  load5 > 108 or MemAvailable < 20 GiB (swap is full, so pause rather than abort). Resume at load5 ≤ 100 and ≥ 25 GiB
  sustained for 30 s. Hard stops as in H239 A1.
- **Deadlines.** Prereg by **19:15Z**; compute stop **20:50Z**; closeout by **21:35Z**. A single run with no resume
  path: an incomplete run closes as CLOSED INCOMPLETE.
- **Durability.** Cell files, both panels and `val_detail.json` are written before the statistics. The statistics
  step records errors instead of raising, and the r882 and DOPE after-checks are recorded, not asserted. Full outputs
  go to S3 first; the ≤ 50 KB git `metrics.json` is derived from them as a packaging step.

## Custody

- **Repo.** `test_h270.py` and `test_h265.py`, Ruff, JSON parse and `git diff --check` precede the push. The private
  `jope-records` gates are recorded; registrations are deferred under the freeze.
- **S3.** Full outputs go to create-only S3 `…/rd_breakthrough_20261030/H270/<run_id>/` with VersionId readback.
- **Closeout.** The README, a metrics JSON of 50 KB or less and both index rows go to v1/next. No model or `.npz` goes
  in Git.
