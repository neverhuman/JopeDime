# H213 — NO_ADVANCE

The single attempt finished at 2026-11-02T09:14:27.787060+00:00. Status: **COMPLETE**;
scientific verdict: **NO_ADVANCE**. Source
`ae683dabb2a810b416cb6734bce216a4aa6d1016`; preregistration PR #1773.

The hypothesis was that a small CUDA head could predict improvement over frozen
r882, with a B-only harm-constrained gate separating useful predictions from
unsafe adoption. [PREREG.md](PREREG.md) fixes the H211 science, fresh H213 seeds,
205-family grouped folds, 30 fits × 128 updates and the sustained-load guard.
Only the separate head trains; native r882/DOPE remain frozen. No sealed or
outer-test rows enter this historical real-S3 TRAIN study.

- Native91 bit-exact: True; mean 1.1377941898164525.
- Actual optimizer updates: 3,840 / 3,840.
- Worker elapsed: 1,752.357 s; supervisor elapsed: 1,758.559 s.
- Global selection lock: True; A evaluated: True.
- Pinned assets/inputs rehashed after completion: all unchanged.

The sole primary is DELTA_SAFE. Gains are the median of three seed equal-family
means; CIs use 10,000 paired family bootstrap draws. Holm multiplicity 1; strict
LCB>0 and p<.05, harm<=10% are all required. Diagnostic arms cannot win.

| Arm | Median gain |95% CI|p|Harmed families|Elo (r882=1500)|
|---|---:|---|---:|---:|---:|
|DELTA_SAFE|+0.007008|[-0.001146, +0.018764]|0.076892|6/205|1501.68|
|DELTA_RAW|-0.005293|[-0.086507, +0.081535]|0.573843|87/205|1549.90|
|SHUF_SAFE|+0.000000|[+0.000000, +0.000000]|1.000000|2/205|1500.00|

Adoption counts per seed (of 1,219 contexts each):

- DELTA_RAW: [1170, 1181, 1188].
- DELTA_SAFE: [25, 21, 18].
- SHUF_SAFE: [4, 0, 0].

All 30 durable per-fit proofs show 128 updates, nonzero gradients,
changed state hashes and changed future forwards on cuda:0. All 30 B choices
were locked and exact-version S3 verified before A. Full losses and gate grids
remain in S3 custody; [metrics.json](metrics.json) contains compact summaries,
Elo intervals and per-seed utilities.

Next step: keep r882; use the fixed raw/gated/placebo diagnostics to inform a
separately preregistered hypothesis. No retry, threshold retuning or promotion.

The raw head adopted 3,539/3,657 choices but harmed 87/205 families and had negative
median utility gain despite its higher Elo. The safe gate reduced adoption to
64/3,657 and harm to 6/205, but its gain CI crossed zero and p exceeded .05.
Elo is diagnostic and does not override the failed primary utility gate.

The load guard survived a 34.870 s overload
(max load1 101.297); load5 max 62.687.
Disk max 93.5995%, RSS max 1,960,546,304 B.

U4 is unmeasured: this protocol spends no verified multi-proposal budget.
These are nested TRAIN development scores on a reused historical cohort,
not sealed qualification. Serving promotion is false.

Custody: 96 worker objects recovered by exact VersionId and SHA256;
139 runtime archive members recovered and individually rehashed.
No new inference or scientific replay was performed for closeout.

`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261101/H213/astra-t0125/closeout-t0125/manifest.json`

VersionId `5nkZa3TX.96X92seBIxbFUqRilRWaYSj`; SHA256
`e22a191c02d81dadc115c95c1e663a6b4e544108825d3309fe567180c1e3fbb7`. [manifest.json](manifest.json) binds the receipt.
