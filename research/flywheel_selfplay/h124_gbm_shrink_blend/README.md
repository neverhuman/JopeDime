# H124 — fixed-lambda GBM shrink blend toward r882

**Status: scored attempt finished all 45 fits and its readout, but is formally INCOMPLETE (nonzero runner exit; see
below). Computed verdict: NO_ADVANCE.** Shrinking the GBM context prior halfway toward r882 keeps a clear U4 gain:
median seed gain **+0.032** (95% CI [+0.015, +0.054], one-sided LCB95 +0.017). It also cuts the harm tail from 36
families (pure GBM) to 22. But **22 of 205 families (10.7%) are harmed**, above the 10% cap, so the preregistered
admission fails. Of the three gates, only the harm cap fails, by two families.

**Run disposition.** The runner exited 1 at its final metrics-size guard. All 45 fits, the readout and the verdict
had completed before the guard. Amendment 2 (#1088, merged during this run) treats any nonzero-exit attempt as
incomplete: it "cannot establish an admitted outcome". This attempt reports a non-admission, so the outcome does not
change. It is recorded as **NO_ADVANCE from an incomplete attempt**. Amendment 2 and the overseer both direct no
duplicate run, and none was started.

## Hypothesis

[H119](../h119_gbm_ctx_confirm/README.md) found a positive U4 gain for `GBM_PROG412_CTX` but harmed 40/205 families
(19.5%). H124 fixes lambda at 0.5 in advance and asks one question: does blending the GBM utility score halfway
toward the matching r882 score bring harm under 10% while keeping a positive equal-family gain? H119 values are
motivation only. None of its folds, seeds, predictions or outcomes enter H124.

## Setup

- **Preregistration.** The [PREREG.md](PREREG.md) protocol is unchanged.
- **Data.** Real-S3 H50 DL-Train packet: 1,219 contexts in 205 families.
- **Split and seeds.**
  - Fresh family fivefold split with `default_rng(12400)`.
  - Model seeds 64–66.
  - 192 sampled programs per context, sampling RNG 12400.
- **Models.**
  - H119/H85 HistGBM recipe.
  - Fit-only positive-slope calibration from inner pairs.
  - 45 fits: 15 per seed.
- **Comparator.** A fresh CPU r882 reference: frozen DOPE with the H40b fused adapter.
- **Blend.** `blend = r882 + 0.5·(GBM − r882)` in utility space.
- **Endpoint.** H119 Track-2 U4.
- **Admission.** All three must hold: median of seed-mean family gains > 0, family-bootstrap LCB95 > +0.001 (10,000
  draws, RNG 124000), and harm ≤ 10% (families with three-seed mean delta < −0.01). There is a single primary, so no
  multiplicity (Holm) adjustment applies.
- **Amendment 1** (operational, committed and pushed before scoring as `1b5d3b478`, pinned by `2901636d1`; squash-merged on main via #1088). It
  responds to the stopped first attempt (7.05 GB RSS) and to [AUDIT_t0855.md](AUDIT_t0855.md) findings A1–A3:
  - The runner reads only train rows of the stored H58 caches, by byte range. **0 DL-Val rows decoded**: 1,219 of
    1,310 rows per feature member, and proj16 exactly the 1,404,287 train-pool rows.
  - Pools load lazily, and the r882 network is released before fitting.
  - Each fit builds a single float64 design matrix, which gives bit-identical estimator input.
  - A continuous RSS watchdog enforces the 6 GiB ceiling.
  - The headline Elo is the registered median-of-seed Elo.
- **Source identity.** This attempt ran Amendment 1 source `2901636d1`, not the later main runner.
  - Amendment 2 (`sol_v1_obj02`, #1088) closed two gaps in that source:
    - buffered reads, which can prefetch up to one buffer of bytes next to a train row into process memory without
      decoding them;
    - watchdog failure paths.
  - Amendment 2 applies to future executions. It explicitly leaves this attempt's source and disposition unchanged.
  - The attempt's receipts record 0 DL-Val rows decoded into arrays. Whole-file identity hashing streams the opaque
    archive bytes, as both amendments disclose.

## Outcome (205 families, 1,219 contexts, 3 seeds; complete panel)

| Arm vs r882 | Seed-mean gains (64/65/66) | Median gain [95% CI] | LCB95 | Harmed families | Gate |
|---|---|---|---:|---:|---|
| **Blend λ = 0.5** (primary) | +0.044 / +0.032 / +0.022 | **+0.032** [+0.015, +0.054] | **+0.017** | **22 / 205 (10.7%)** | **fail (harm)** |
| Pure GBM λ = 1.0 (report only) | +0.068 / +0.064 / +0.050 | +0.064 [+0.030, +0.103] | +0.034 | 36 / 205 (17.6%) | not assessed |

| Arm | Median-of-seed Elo (r882 = 1500) [95% CI] | Per-seed Elo 64 / 65 / 66 | Pooled-score Elo (additional) |
|---|---|---|---|
| Blend λ = 0.5 | **1543.9** [1528.8, 1559.3] | 1548.3 / 1538.0 / 1543.9 | 1555.5 [1538.7, 1572.5] |
| Pure GBM λ = 1.0 | 1537.9 [1515.5, 1558.5] | 1543.4 / 1537.9 / 1525.6 | 1557.3 [1534.6, 1581.2] |

Elo uses family-grouped Bradley–Terry (tie margin 1e-4, prior 0.01). The interval comes from 1,000 family resamples
(RNG 124001) drawn jointly across seeds, and it is report-only. The pooled-score Elo averages each context's U4 over
seeds first; the audit asked that it be labelled as an additional estimate.

## Reading

- **Shrinkage does what it was meant to, but not enough.**
  - Halving the GBM's departure from r882 removes 14 of the 36 harmed families.
  - It keeps half of the gain: +0.032 against +0.064.
  - It also tightens the per-seed Elo spread.
- **The harm cap needs ≤ 20 families.** The blend has 22. Gain and LCB pass comfortably, and both arms rate
  materially above r882.
- **This is conditional development evidence.** r882 had earlier exposure to all H50 DL-Train families, so this is
  not an unseen-family or serving result. Nothing here supports a default change.

## Next step

Do not tune lambda on these folds. A smaller fixed lambda or a per-family fallback could plausibly clear the cap, but
choosing either from this result would be selection on outcomes.

If pursued, preregister one alternative on a fresh family split and fresh seeds. Two candidates:

1. A fixed lambda chosen a priori, for example 0.25.
2. A harm-aware rule that falls back to r882 when the inner-fold GBM calibration slope is weak.

Owner call. There is no live-loop or flywheel change.

## Evidence and custody

- **Run.** `run-20261031T1517Z` from source `2901636d1`, runner sha256 `2ff72805…`:
  - xbabe2, nice 15, four threads;
  - 2:15:15 wall, 24,064 CPU s;
  - **peak RSS 4,455,500 KiB < 6 GiB** (the watchdog saw 4,504,836 KiB over 32,258 samples);
  - 97.5 MB scratch;
  - DOPE `c51a56d0…` before and after, 0 optimizer updates, no GPU, CUDA not initialized.
- **Runner exit.** The run exited 1 at its final size guard: the full `metrics.json` (108,246 bytes, sha256
  `976af71c…`) exceeds 100 KB. Every fit, the readout and the verdict had completed before the guard, and nothing
  was rerun.
- **[metrics.json](metrics.json)** is the compact copy (22 KB). It drops only the per-family gain arrays and
  family-id list (their sha256 is kept) and fit-receipt boilerplate. Every gate value is unchanged.
- **Read boundary.**
  - 1,219 DL-Train rows.
  - 0 DL-Val, holdout, sealed or meta-test payloads.
  - 0 DL-Val cache rows decoded.
  - No live-loop path.
- **Custody.**
  - S3 `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H124/opus-h124-20261031a/` holds 102
    objects, each written create-only with a version-pinned readback.
    - The PREREG said no S3 write; the later overseer nudge `rd_opus_t1100` directs S3 custody, and the upload
      happened after scoring.
    - `r882-copy.pt`, a byte-identical copy of the pinned checkpoint, is not uploaded.
  - Receipts: `/home/ubuntu/jope-rescue-20260921/h124-rd_opus-20261031/`.
  - First stopped attempt: `/home/ubuntu/jope_plan/evidence/H124/partial-run-blocker.json`.

## Audit corrections for future execution

The [dated audit](AUDIT_t0855.md) covers the original source. OPUS Amendment 1
adds train-row cache reads, lazy/streamed inputs, median seed Elo and an RSS
watchdog. Owner t0955 preserves those commits and closes the remaining prefetch
and guard-failure paths in Amendment 2; existing running jobs are untouched.

Feature reads now use unbuffered archive/header/row access, validate row layout
before payload reads, and refuse DL-Val spans. Context IDs and projection offsets
are metadata. Whole-file hashing still reads opaque archive bytes, as explicitly
disclosed in the amendment; validation features are not decoded into model inputs.
`registered_estimator=median_seed_elo` identifies the headline and its paired
family-bootstrap interval. The pooled-score estimate is labelled additional.

The watchdog samples current and peak RSS at entry, exit and nominal 0.25-second
intervals. RSS breaches and monitor/receipt failures terminate the worker with
code 86. It is a sampled abort guard, not a kernel allocation limit. Partial
outputs from a breached/nonzero-exit run cannot establish completion. A new source
commit cannot retroactively qualify an old run: retain each run's actual source,
preregistration and terminal resource evidence. No scoring or retry is started by
these corrections. The operator must avoid duplicating the existing attempt.
