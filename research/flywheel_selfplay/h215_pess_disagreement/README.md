# H215 — NO_ADVANCE

The single attempt finished at 2026-11-02T10:37:41.910055+00:00. Status: **COMPLETE**;
scientific verdict: **NO_ADVANCE**. It executed preregistration
PR #1801 at `79c02dbb233e6c88afe584357c6a072ff5919966`.

H215 tested whether penalizing ensemble disagreement improves full-pool native
r882 selections: first argmin of `cost_r882 + kappa * sigma_ensemble`.
Sigma is the population standard deviation of ten independent fresh adapter
members' within-context standardized costs. Kappa zero preserves native first
argmin exactly. [PREREG.md](PREREG.md) fixes the protocol and literal coefficient
grid `0, .25, .5, 1, 2`; coefficients are not quantile probabilities.

The historical real-S3 TRAIN cohort contains 205 families and 1,219 contexts.
Three outer seeds each use five family folds (123 FIT, 41 B, 41 A).
Three inner family folds select kappa on FIT alone, followed by a full-FIT
refit. Ten independent members per phase give 600 planned fits, each with
40 FIT-only optimizer updates. Only fresh adapters train; r882 and DOPE remain
frozen. The exact H178-calibrated four-arm B selector precedes A outcome reads.

- Native91 exact: PASS; mean 1.1377941898164525.
- Cached native-cost parity: PASS on all 1,219 TRAIN contexts.
- Actual optimizer updates: 24,000 / 24,000.
- Durable completed fits: 600 / 600.
- Worker elapsed: 2,232.574 s; supervisor elapsed: 2,239.918 s.
- Global selection lock verified before A; A evaluation completed.
- Pinned assets and inputs rehashed after completion: all unchanged.

PESS_NEST is the sole primary. Gains are medians of three seed equal-family
means; CIs use 10,000 paired family bootstrap draws. Admission requires the
calibrated B gate, positive primary gain and lower CI bound, Holm-adjusted
p < .05 across all four arms, and at most 10% harmed families. A family is
harmed when its seed-average gain is below −.01. Ties reject; diagnostic arms
cannot win.

All four policies failed B admission, so their B-selected fallback remains
r882. The A table reports the preregistered raw-policy diagnostics, evaluated
regardless of B rejection; it does not claim admitted or promoted policies.

Primary B admitted: **No**.

| Arm | Median gain | 95% CI | Holm p | Harmed families | Elo (r882=1500) |
|---|---:|---|---:|---:|---:|
| PESS_NEST | -0.001330 | [-0.006886, +0.008135] | 1.000000 | 13/205 | 1496.68 |
| PESS_FIX | -0.002497 | [-0.018004, +0.009967] | 1.000000 | 16/205 | 1498.98 |
| PESS_SHUF | +0.000481 | [-0.012145, +0.002224] | 1.000000 | 23/205 | 1497.69 |
| OPT | +0.001557 | [-0.005327, +0.009068] | 1.000000 | 15/205 | 1498.82 |

Adoption counts per seed (of 1,219 contexts each):

- PESS_NEST: [33, 74, 62].
- PESS_FIX: [66, 69, 81].
- PESS_SHUF: [69, 143, 104].
- OPT: [66, 60, 72].

Selected kappa counts across 15 outer cells: 0.0: 2, 0.25: 2, 0.5: 3, 1.0: 7, 2.0: 1.

All 600 independently recovered fit proofs show 40 updates, nonzero gradients,
changed parameter hashes and changed future forwards on cuda:0. Saved family
lists and member seeds match all 600 preregistered fit partitions. All FIT choices
and B panels were durably locked and exact-version S3 verified before A.
[metrics.json](metrics.json) records per-seed gains, adjusted p values, admission
decisions, adoption, utilities and Elo intervals; full fit proofs and outcome
panels remain in custody.

The result does not advance: the calibrated B selector rejected the primary,
its median gain and CI lower bound were nonpositive, and its adjusted p value
did not pass .05.

Next step: keep r882 and use the fixed pessimistic, shuffled and optimistic
diagnostics to motivate a separately preregistered hypothesis. No retry,
threshold retuning or serving promotion follows this attempt.

The attempt had a fixed 90-minute cap and used GPU0, qualification.lock FIFO, nice 19 and one
numeric thread. Maximum load1 was 78.523, load5 45.713,
disk 93.6465%, and worker-tree RSS 2076696576 B.
Minimum swap-free was 0 B (receipt only; MemAvailable
is the memory guard). The longest load1 overload was 0.000 s.

U4 is unmeasured: no verified multi-proposal budget was spent. These are nested
TRAIN development scores on a reused historical cohort, not sealed qualification.
No sealed, holdout or outer-test rows enter the learning input closure.

Custody: 1312 worker objects recovered by exact VersionId and SHA256;
1389 runtime archive members recovered and individually rehashed.
Closeout performs no new inference or scientific replay.

`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H215/astra-t0230/closeout-t0230/manifest.json`

VersionId `AoXxY9lgOKe7mkrZJPqSMBnSAGlzGs9W`; SHA256
`50d1a67449857f1e35470be6cc4b3025a806aadbbbb056297dfea24ca5c5aaf1`. [manifest.json](manifest.json) binds the receipt.
