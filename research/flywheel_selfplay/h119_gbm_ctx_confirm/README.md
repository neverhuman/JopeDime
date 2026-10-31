# H119 — prospective GBM_PROG412_CTX confirmation vs r882

**Verdict: `NO_ADVANCE`.** This is the complete owner-assigned CPU
confirmation in `h119_gbm_ctx_confirm_t0705.md`, not the separate W4 screen
renumbered H120 by the overseer. All 48 fixed GBM fits completed. No GPU,
qualification lock, live-loop action, HWM append or serving promotion occurred.

## Hypothesis and fixed setup

H108 chose GBM_PROG412_CTX post hoc among 10 contenders on family-fold seed4900.
Its U4 lead motivated this preregistered recheck; it was not prospective evidence.
[PREREG.md](PREREG.md) was first committed as `da81b26d3653b8f5ef6e25104f214cbceca40eeb` before any H119
scoring; the executed source was `93e5d0e7480bae1890f05cc418ddd0f33526aa91`. Both original commits
are recoverable from the S3 replay bundle, with prerequisite Git base `a1b08cbfabe77410a7c323762146c85a50d7f16f`.

Use all 1,219 real-S3 H50 DL-Train contexts/205 families. Fresh whole-family
fivefold seed 11900 yields 41 families per fold and held context counts
[246, 234, 212, 235, 292]. GBM seeds 51/52/53 share the split.
The exact H85 recipe uses 412 program features + 48 LM48 + 629 frozen DOPE features,
192 fixed unrevealed training candidates/context (sampling seed 700), centered
utility targets, ascending fit-fold concatenation and HistGradientBoosting
max_iter300/lr.08/leaves63/L2=1 with unchanged sklearn defaults. Five outer and
ten inner-pair models per seed give 15 outer + 30 fit-only calibration fits; three
full-training fits serve the report-only secondary panel. Actual iteration range
was 300–300. Versions are recorded in [metrics.json](metrics.json).

R882 is a freshly staged offline round-882 COPY, restored with 0 missing/unexpected
keys. Its full-pool costs were recomputed on CPU using H108's zero context/prior/
SOS, scale 0.4208 and four revealed-anchor recipe. Frozen DOPE composite SHA
`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` matched before/after both reference phases;
no neural parameter was trained. Input/source/cache/pool/table identities and
train-only JTT1 headers were checked. No sealed/holdout/meta-test/outer-test/live
payload was read; the audit found 0 forbidden data attempts and 0 CUDA initialization.
H108's full-pool slate centering and its known slate-size dependence are retained;
these results make no claim for a different live evaluation slate.

These205 families and the fixed features were previously exposed during
development, and r882 trained on these contexts. A fresh partition is a conditional
recheck on a reused corpus, not independent new-family or serving qualification.

## Primary outcomes and verdict

Track2 U4 selects the best utility among four revealed anchors and four lowest-cost
unrevealed candidates. Track1 legacy uses whole-pool argmin, including revealed
slots, matching H108. The primary statistic is the median of three seed-level
equal-family paired mean gains. All 205 family clusters are resampled together,
10,000 common draws (seed 119000), preserving paired seed vectors. Holm controls
the two preregistered primaries; this is the stated percentile-bootstrap approximation.
No subset, seed selection or rounded number determines a gate.

| Track | GBM family utility (median seed) | r882 | Paired median gain [95% CI] | LCB95 / Holm LCB | Harmed families | GBM Elo [95% CI] |
|---|---:|---:|---|---|---:|---|
| Track2 U4 | 1.660687 | 1.609212 | +0.051474 [+0.023571, +0.085002] | +0.027521 / +0.023571 | 40/205 | 1549.87 [1525.42, 1575.12] |
| Track1 legacy | 1.297052 | 1.385355 | -0.088303 [-0.203001, +0.011202] | -0.184654 / -0.184654 | 107/205 | 1449.18 [1423.61, 1476.28] |

- U4 median across seed means >0: **PASS**.
- U4 Holm LCB >+.001: **PASS**.
- Track1 Holm LCB >-.01: **FAIL**.
- U4 harmed families <=10%: **FAIL**.

Harm means seed-mean family delta<-.01; the <=10% harm gate applies to U4.
Legacy harm counts are descriptive. Holm bounds must strictly exceed their
respective margins; ties are not wins. Full original-precision statistics and
per-seed effects are in metrics.json; identical seed utility groups are reported
without treating them as independent confirmations.

Bradley-Terry uses the canonical family-weighted two-contender panel, r882=1500,
prior=0.01/tie_margin=1e-4/400 iterations, and 1,000 family bootstrap draws per track.
It compares seed-mean context utilities; per-seed ratings are reported separately.
Canonical-helper bootstrap parity is tested. This two-contender Elo is not directly
comparable with H108's ten-contender ratings. Context W/T/L and strict family W/T/L
are retained separately, including explicit ties. No AUC is reported.

## Secondary DL-Val — report only

Primary vectors/metrics were sealed before opening any of the 91 DL-Val outcome
pools (19 families). Three full-DL-Train GBMs were fitted with the same recipe and
training-only OOF calibration; no secondary value affected the primary protocol,
retuning, selection or admission. These historically used development families
are not a new external test.

| Track | GBM family utility (median seed) | r882 | Paired median gain [95% CI] | LCB95 / Holm LCB | Harmed families | GBM Elo [95% CI] |
|---|---:|---:|---|---|---:|---|
| Track2 U4 | 1.285669 | 1.277656 | +0.008013 [-0.014710, +0.047837] | -0.010468 / -0.014710 | 4/19 | 1525.62 [1440.98, 1621.65] |
| Track1 legacy | 0.988356 | 1.153615 | -0.165260 [-0.344030, -0.055611] | -0.313654 / -0.313654 | 12/19 | 1489.04 [1406.27, 1570.55] |

Secondary bounds/harm/Elo are descriptive only. There is no secondary admission.

## Verification, resources and custody

All 48 fit receipts, context/family rosters, raw predictions, calibration, primary
and secondary bootstrap/Elo outputs replay exactly. CPU-reference parity against
the pinned H108 vectors is {"legacy": {"changed_float_value_count": 183, "context_mean_difference": 1.2864561298466232e-18, "max_abs_context_difference": 1.7763568394002505e-15, "utility_difference_count_above_1e_minus_12": 0}, "u4": {"changed_float_value_count": 187, "context_mean_difference": -1.775992533239586e-18, "max_abs_context_difference": 1.7763568394002505e-15, "utility_difference_count_above_1e_minus_12": 0}}.
This compares H108's seed-mean utilities with a single CPU reference; tiny floating
reduction differences count as changed values, not proven candidate-choice changes.
Targeted synthetic CPU controls, Ruff/format, syntax, links and candidate records
checks are reported in the exact-head review/status; no hosted CI was requested.

Measured worker wall 4085.57s and process CPU
31547.62s; peak RSS 17,843,920KiB.
Three fit workers, OMP/MKL 4 each, BLAS 1; the nested nice 15 launcher inherited
niceness and ran at safer effective nice 19. Maximum observed host CPU
47.7%, load1 71.95; own-process guard
pauses 0. Raw scratch 101,319,496B stayed below 2 GB.
Scratch reclamation follows the verified S3 receipt; no shared cache was removed.

Raw replay archive: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H119/confirm-20261031T131405Z/raw-replay.tar.gz`;
VersionId `rhp6IsNPg.UzNSY7UPwXtiHdV7TaYg8R`;
SHA256 `72170fb1190c525b222c5825cfb2ac164b47424b04ce21c9a609f52d91465a30`.
Manifest: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H119/confirm-20261031T131405Z/manifest.json`;
VersionId `BenkJ6Jm13sCCT.NVCbrDcq2OKfS6OmM`;
SHA256 `c97df26558c102e9c94dc7df79ce92a042fd53aa17335b3e06a4410db725e14e`.
Both uploads were create-only and read back by exact VersionId/size/SHA before
reclamation. The complete archive restores all raw members by hash. No pickled
GBM model is included in Git; metrics.json is 67,310B.

## Next

Do not advance or promote GBM_PROG412_CTX from this confirmation. Diagnose the failed preregistered endpoint(s) before proposing another fixed protocol; do not choose a new seed, fold, calibration or threshold from these outcomes.
