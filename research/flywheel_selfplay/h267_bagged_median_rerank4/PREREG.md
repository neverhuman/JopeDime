# H267: bagged median-rank top-4 re-ranker (zero-update, single-pick), xbabe2 CPU

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t1805_opus_h267.md` (watcher / Grok Bot for Jepson,
2026-11-03 18:05Z, decided). **Claim:** `~/jope_plan/ns/receipts/H267_claim.txt`.

This is a diagnostic only. It cannot advance a model, a serving release or a live flywheel checkpoint. **No parameter
of r882 or DOPE is updated.** The r882 checkpoint sha256 and its recorded DOPE identity (`c51a56d0`) are checked before
and after. Every policy picks one candidate from r882's top-4 with **0 test-time reveals**. The live flywheel and H266
files are never touched.

**Publication.** Under the PR #2071 split this prereg is pushed to `origin/v1/next` (non-force) before any outcome is
read. The pre-publication smoke runs only the Native91 and TOP4_PICK@r882 regressions plus a synthetic bagging
exercise; it reads 0 TRAIN contexts. The launch is bound to the published commit, its verified ancestry in
`origin/v1/next` and the sha256 of a durable same-agent review receipt.

## Hypothesis

**What is known.** H265 RERANK4 (`da5a08afb`) advanced on DL-Val91: +0.0187 family gain, Holm LCB +0.0051. The red-team
caveats are seed instability (seed 3 roughly halved), 3–5 harmed families per seed, and a non-significant 205-family
DL-Train OOF result (p 0.062).

**H267 asks:** does family-bootstrap **bagging** with **median-rank** aggregation make the re-ranker's lift stable and
positive on the 205-family DL-Train out-of-fold panel, and does it still lift DL-Val91?

## Frozen from H265 (imported unchanged)

[h265_core.py](h265_core.py) is a byte-identical copy of `v1/next` `h265_top4_listwise_reranker/runner.py` (sha256
pinned in [manifest.json](manifest.json)). H267 imports from it, unchanged:
- the 560-feature top-4 builder;
- the relevance labels (gate-failed = worst);
- the `LGBMRanker(lambdarank)` recipe (depth 3, 7 leaves, 200 trees, n_jobs 1, deterministic) and its grid: lr ∈
  {0.03, 0.05, 0.10}, n_est ∈ {50, 100, 200}, margin ∈ {0, …, 1, ∞};
- the 4-fold inner-CV strict selector (`run_cell_fit_and_select`: gain > .001 and LCB95 > 0, else keep r882);
- the margin rule, and the within-family label shuffle.

Its own inputs (r882 cached costs, `common_h252.py`) are read-only sha-verified copies under `~/jope-data/h267/inputs/`.

## Arms

- **CONTROL** = r882's slot 0.
- **RERANK4** = the H265 single model, re-run per cell with H267 seeds (reference).
- **BAG_MED (primary):**
  - **Hyperparameters:** (lr*, n_est*) chosen ONCE per cell by the frozen H265 inner selector on the full FIT set (the
    RERANK4 run of that cell).
  - **Bags:** B = 25 LGBMRankers at (lr*, 200 trees), each fit on a **family-level bootstrap resample** of the 164 FIT
    families (with replacement; every context of a drawn family is included). Bag b uses bootstrap seed 267160 + b,
    streamed with the model seed and the assess fold.
  - **Score:** each bag predicts at n_est* and ranks the context's 4 slots (rank 0 = best; ties go to the smaller slot).
    BAG_MED picks the slot with the smallest **median rank** over bags. Ties keep r882 order, so slot 0 is kept unless
    another slot's median rank is strictly smaller.
- **BAG_SHUF (placebo):** the same hyperparameters and the same bootstrap family draws, with labels permuted within
  family (the H265 shuffle function, seeded per cell).
- **Strict selector for the bagged arms.** The inner evaluation is **out-of-bag**: each FIT context is scored by the
  median rank over the bags that did not draw its family, at no extra fit. An arm is admitted only if its OOB
  equal-family gain over r882 on the 164 FIT families is > .001 **and** the family-bootstrap LCB95 (2,000 draws) is
  > 0. Otherwise the arm is r882 for that cell.

## Design (interpretation frozen here)

- **Primary: DL-Train repeated nested family CV.** Each model seed has its own fresh fold seed: **267101 ↔ 267140,
  267102 ↔ 267141, 267103 ↔ 267142**. That gives 3 independent family partitions × 5 folds = **15 cells**.
  - Per cell, FIT is 4 folds (164 families) and ASSESS is the held-out fold.
  - Each TRAIN context is scored exactly once per seed, giving a 1,219-context out-of-fold panel per seed.
  - **Interpretation note:** the brief's "3 fold seeds × 5 folds × model seeds 267101–267103" is read as paired
    repetitions. A full 3 × 5 × 3 = 45-cell cross is infeasible before the 20:30Z stop: one fit measured 3.0 s, so a
    cell (≈ 64 fits) takes ≈ 190 s, and 45 cells would take ≈ 2.4 h. Pairing gives every seed an independent
    partition, which directly tests H265's seed instability.
- **Secondary: DL-Val91 single-pick.** Each seed's mixture over its 5 cells' policies, i.e. the same 15 cells. For seeds
  267102 and 267103 the folds come from 267141 and 267142 rather than 267140; this is part of the same pairing.
- **Lock.** All 15 cells' selections and admissions (RERANK4 strict margin; BAG_MED and BAG_SHUF OOB admission) are
  written to one fsync-locked `selection.json` **before any DL-Val91 feature, pool or label is read**. Each policy is
  scored once.
- **Regression first.** Native91 single-pick 1.1377941898164525 (91/91 bit-exact) and TOP4_PICK@r882 1.2403601682250796
  / 1.2519997168684853, before any TRAIN packet is built. The H265 test file and `test_h267.py` run before compute.

## Statistics and decision

- **Statistic.** Per-seed equal-family gain; the median over the 3 seeds; family-bootstrap CI95 (10,000 draws, boot seed
  267150); and the Holm-level (m = 2) interval [1.25%, 98.75%].
- **Holm m = 2** is applied over {PRIMARY, SECONDARY} for BAG_MED − r882.
- **ADVANCE** requires all five of:
  - PRIMARY median family gain > **+0.002**;
  - PRIMARY Holm-level LCB > 0;
  - PRIMARY **3/3 seeds positive**;
  - **BAG_MED beats BAG_SHUF** on PRIMARY (median gain > 0);
  - SECONDARY gain > 0 on **3/3 seeds**.

  Otherwise NO_ADVANCE. A fallback identity (no cell admitted) is NO_ADVANCE, not equivalence.
- **Descriptive (not gated):**
  - BAG_MED − RERANK4, per-seed spread, harmed families per seed (< −1e-4) and W/T/L;
  - −2.0 invalid picks and changed choices;
  - context #27, reported separately and never excluded;
  - MM Bradley–Terry Elo with r882 = 1500.
- **Not used.** No AUC, and no speed verdicts.

## Resources and the A1 memory guard (H239 A1)

- xbabe2, CPU only: `taskset` 44–47, nice 19, LightGBM n_jobs 1, ≤ 4 threads, tree RSS < 4 GiB (bags fit sequentially
  and are freed; models are kept only as text), disk free ≥ 300 GB.
- No GPU, xbabe0, xbabe1 writes, xbabe3 or live flywheel. No foreign process control. PR #2071, main and v1/fastlane
  are not touched.
- **Admission** at load5 ≤ 100 and MemAvailable ≥ 30 GiB. **Cooperative pause** at load5 > 108 or MemAvailable <
  20 GiB; swap is full, so the guard pauses rather than aborts. **Resume** at load5 ≤ 100 and MemAvailable ≥ 25 GiB,
  sustained for 30 s. **Hard stop** at MemAvailable < 12 GiB sustained for 30 s or < 6 GiB on any sample, plus the RSS,
  disk, thread and deadline stops.
- **Deadlines.** Prereg on v1/next by **18:45Z**; compute hard stop **20:30Z**; closeout by **21:15Z**. A single run
  with no resume path: an incomplete run closes as CLOSED INCOMPLETE.

## Custody

- **Repo.** pytest (`test_h267.py` and the H265 test file), Ruff, JSON parse and `git diff --check` precede the push.
  The private `jope-records` gates are recorded. Registrations are deferred under the freeze (no CI-config, records-pin,
  schema or root-README edits). A durable same-agent review receipt names the exact head.
- **S3.** Full outputs, including per-context detail, go to create-only S3 `…/rd_breakthrough_20261030/H267/<run_id>/`
  with VersionId readback.
- **Closeout.** It publishes only the README, a metrics JSON of 50 KB or less and the index rows to v1/next. No model or
  `.npz` goes in Git.
