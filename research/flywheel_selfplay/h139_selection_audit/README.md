# H139 — winner's-curse / selection-rule audit of the r882 plateau

**Status: complete. Verdict: SELECTION_ARTEFACT** (rule preregistered in [PREREG.md](PREREG.md), PR #1241; CPU, zero optimizer
updates).

- **No distinct adapter is significantly worse than r882 on DL-Train.** Each of R782, R859 and REJ has a 95% CI that
  reaches 0 or above.
- **The val ranking does not predict the DL-Train ranking.** The Spearman correlation is -0.95, with
  CI [-0.9487, +0.9487], which includes 0.
- **The val loser leads on DL-Train.** R782, the *lowest*-scoring arm on the flywheel's 91-context val, is the *best*
  arm on DL-Train: +0.0133 [+0.0023, +0.0273], and Holm rejects in its favour.
- **One context decides the val gap.** R782's whole val deficit against r882 (−0.030) is a single context where its
  first-argmin pick is invalid (−2). Its DL-Train edge is likewise carried by about 10 families with invalid-pick
  flips (13 vs 17 invalid picks).

The 91-context argmax ranking therefore turns on single cliff events, not on a stable quality difference. r882's
plateau value is a selection outcome, not evidence that r882 is better.

## Hypothesis

r882's 1.1377942 is the single-argmax winner of about 1,700 admission evaluations on 91 val contexts in 19 families.
The hypothesis is that, on data independent of that selection, r882 is not better than the other flywheel adapters
by more than noise.

Pre-outcome metadata from the admission log: only 4 of 1,709 rounds were accepted, and **r882 tied round 859**
exactly.

## Setup

- **Arms.** Byte-pinned copies of the live run's checkpoints, taken read-only at claim time: R882 (reference), R782,
  R859, REJ (`rejected/round-2482`), R1703 and R2481.
  - R1703 and R2481 are the restored r882 state, with bitwise-identical adapters. The runner verified that their
    outputs equal r882's exactly, and they are excluded from Holm and from the decision.
  - The brief's R2479 had rotated out before claim time, so R2481 (latest retained) takes its place.
  - All arms carry DOPE `c51a56d0` and identical non-adapter tensors.
- **Forward pass.** The pinned H124 CPU forward pass (`9ebce965`), with no parameter updates.
- **Endpoint.** The native flywheel endpoint (first argmin over the full pool; −2 if the pick is invalid). U4 is
  descriptive only.
- **Primary.** Equal-family gains against R882 on the 1,219 DL-Train contexts in 205 families. The 95% CI comes from
  10,000 family-bootstrap draws, with Holm over the three distinct arms and paired Elo with R882 = 1500. AUC is not
  used.
- **Fold seeds.** Seeds 139191–139193 give identical point estimates for fixed arms, as disclosed in the
  preregistration. They drive the cross-fit argmax audit.
- **Run.** xbabe1, nice 19, 4 threads, 32 min wall, peak RSS 968,344 KiB. No GPU and no
  qualification lock.

## Outcome

**Regression: PASS.** The R882 copy reproduces the live `best_dev_checkpoint.json` on **91/91** contexts, with max
error 0.0 and mean 1.1377941898. The recomputed val values also equal the admission log's recorded
values for R782, R859 and REJ, to within 2e-16.

| Arm vs R882 | DL-Train mean gain [95% CI] | p | Holm | Families W/T/L | Harm | U4 gain (desc.) | Paired Elo [95% CI] | Flywheel 91-ctx val |
|---|---|---:|---|---|---:|---:|---|---:|
| R782 | +0.0133 [+0.0023, +0.0273] | 0.008 | reject | 37/135/33 | 6.8% | -0.0010 | 1501.7 [1496.9, 1506.7] | 1.1074 |
| R859 | -0.0001 [-0.0006, +0.0002] | 0.642 | no | 3/200/2 | 0.5% | +0.0001 | 1500.1 [1498.7, 1501.4] | 1.1378 |
| REJ | +0.0020 [-0.0106, +0.0170] | 0.792 | no | 33/149/23 | 5.9% | +0.0001 | 1501.9 [1496.2, 1507.3] | 1.1354 |
| R1703 | +0.0000 [+0.0000, +0.0000] | 1.000 | — | 0/205/0 | 0.0% | +0.0000 | 1500.0 (= R882) | 1.1378 |
| R2481 | +0.0000 [+0.0000, +0.0000] | 1.000 | — | 0/205/0 | 0.0% | +0.0000 | 1500.0 (= R882) | 1.1378 |

R882's own val score is 1.1378.

**Spearman (val vs DL-Train, R882 / R782 / R859 / REJ):** -0.949, 95% CI [-0.9487, +0.9487].
There are only n = 4 adapters, and R882 ties R859 on val.

**Cross-fit argmax audit.** On all 15 folds (3 fold seeds × 5) the four-fold argmax picks R782. Its held-out gain
(+0.0133) equals its in-sample gain, so the DL-Train selection shows no optimism.

## Reading

- **The plateau is a selection artefact of the native metric's cliffs.** On both val and DL-Train, arm differences
  are dominated by rare contexts where the first-argmin pick fails its gate (utility −2).
  - On 91 contexts, one such event is worth about 0.03, which is the whole R782–r882 val gap.
  - The single-argmax admission rule cannot tell those events from real improvement.
  - Ranking by U4 (the best of four revealed and four proposed) removes most of the difference: |gain| ≤ 0.001 for
    every arm.
- **Caveat 1: DL-Train is training data for every arm.** It is independent of the selection, not of training, so
  R782's DL-Train edge is a better training-set fit. It is not shown to generalize.
- **Caveat 2: R782's edge is fragile.** It is concentrated in about 10 families: the top-10 share is 113%, and the
  trimmed mean is +0.0005.
- **Caveat 3: the Spearman test has almost no power** with four adapters.
- **The live flywheel was not changed.**

## Next step: proposed admission rule (not deployed)

Replace the single 91-context argmax with the following rule:
- Accept a candidate only if the **bootstrap median of its family-cluster gain over the incumbent exceeds +0.001**.
  Use 19 val families, at least 2,000 draws and a fixed RNG.
- **Ties are rejected.**
- Report the invalid-pick count (−2 events) separately, so a single cliff cannot decide admission.

Owner call: whether to trial this rule on a shadow copy of the loop's admission log before any live change.

## Data, red-team and custody

- **Reads.** 1,219 DL-Train contexts, plus the flywheel's own 91-context val contract, used only for the regression
  and the descriptive val values. Zero sealed, holdout or meta-test payloads. 0 DL-Val cache rows decoded beyond the
  contract. `source_manifest.json` and `cache_manifest.json` were not opened.
- **Live directory.** Read-only copies only, with 0 writes.
- **DOPE.** `c51a56d0` before and after for every arm, with 0 optimizer updates.
- **Metric direction.** Utility is higher-is-better, and gain = arm − R882. The degenerate arms give exactly 0 on
  205/205 families.
- **No selection on outcomes.** The arms were fixed by round number and the decision rule was as preregistered.
- **Custody.**
  - S3 `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H139/opus-h139-20261101a/` holds 23
    objects: metrics, per-context utilities, per-arm costs and receipts, and the log. Arm inputs are under
    `H139/opus-h139-inputs-20261031/`. All are create-only with a version-pinned readback.
  - Rescue copy: `/home/ubuntu/jope-rescue-20260921/h139-rd_opus-20261101/`.
- **[metrics.json](metrics.json)** is 10 KB.
