# H176 — data scaling versus adapter representation limits

**COMPLETE — DATA_SATURATED** (corrected from initial `INCONCLUSIVE` after fixing the `analysis.py:396` `axis=1` $\to$ `axis=0` bootstrap reduction; see below). H176 measured the r882 zero-update TRAIN-to-DL-Val gap and a nested FIT-family learning curve. It is diagnostic only; no checkpoint, product, or serving state was advanced.

## Hypothesis and setup

The study asked whether the adapter is data-limited (more real TRAIN families improve held-out utility) or capacity/representation-limited (more families do not help). It used the pinned 1,219-context / 205-family DL-Train cohort, 91-context / 19-family DL-Val cohort, cached frozen-DOPE features, five family-grouped outer folds, and three independent seeds. Each fold trained fresh native-random adapters on nested FIT subsets of 31, 62, or 123 families, with an r882-initialized 123-family continuation anchor. Every cell received the same 40 updates; all 60 trajectories completed.

The exact native r882 DL-Val baseline passed 91/91 bit-for-bit (mean `1.1377941898164525`). The runner held DOPE and all non-adapter state fixed and assessed learning-curve arms only on their held-out A families. DL-Val was descriptive and did not affect training or selection.

## Results

The zero-update r882 equal-family utility was `1.385355` on TRAIN and `1.153615` on DL-Val, a descriptive gap of `+0.231740`. The TRAIN oracle-headroom-normalized gap was `0.496033`; the exact `-2.0` utility count was 17 TRAIN contexts and 0 DL-Val contexts.

| Arm | Median equal-family utility | 95% family-bootstrap CI | Elo vs r882 | Elo 95% CI |
|---|---:|---:|---:|---:|
| Fresh, 25% FIT families | 1.390206 | [1.141763, 1.677076] | 1509.24 | [1495.26, 1523.63] |
| Fresh, 50% FIT families | 1.389931 | [1.145126, 1.678068] | 1510.25 | [1496.21, 1524.06] |
| Fresh, 100% FIT families | 1.388255 | [1.142293, 1.675383] | 1509.22 | [1495.06, 1522.91] |
| r882 continuation, 100% FIT families | 1.355150 | [1.114262, 1.639407] | 1485.08 | [1475.19, 1494.50] |

The paired 25%→50% utility gain was `+0.000936` [`−0.004266`, `+0.006808`] (per-seed gains `[+0.000936, −0.003109, +0.008024]`). The 50%→100% gain was `−0.001676` [`−0.006674`, `+0.001683`] (per-seed gains `[−0.001670, −0.001676, −0.004552]`). The 25%→100% gain was `−0.000734` [`−0.006641`, `+0.004873`]. The fitted slope per log2 doubling was `−0.000367` [`−0.003320`, `+0.002437`]. Intervals use 10,000 paired family resamples with seed 176141 (`axis=0` median over the three seeds per bootstrap draw); displayed utility is median over three seeds.

> **Red-team audit note (`rd_argon`, `t0510`)**: In `analysis.py:396`, `paired_differences` computed `np.median(delta[:, draw].mean(axis=2), axis=1)` over `axis=1` (the 10,000 bootstrap draws) instead of `axis=0` (the 3 seeds), taking `[0.025, 0.975]` quantiles of the 3 per-seed bootstrap medians (`[−0.004140, −0.001623]` for 50%→100% and `[−0.002689, +0.007189]` for 25%→50%) and discarding family-resample variance. Recomputing `paired_differences` from `learning_curve_family_panels.npz` under the preregistered `axis=0` statistic (`np.median(..., axis=0)` across the 3 seeds for each of the 10,000 paired family resamples, matching `arms`, `slope_per_log2_doubling`, and `elo_vs_r882`) yields `[−0.006674, +0.001683]` for 50%→100% and `[−0.004266, +0.006808]` for 25%→50%.

Under the corrected 10,000-draw paired family-bootstrap intervals, the preregistered rule returns **DATA_SATURATED**: both the 50%→100% interval (`[−0.006674, +0.001683]`) and the 25%→50% interval (`[−0.004266, +0.006808]`) include zero. Adding more FIT families from 25% (`31` families) to 100% (`123` families) does not improve held-out A-family utility for the 29,211-parameter adapter, whereas unvalidated 40-step continuation from `r882` degrades held-out A-family utility (`1.355150`, Elo `1485.08`). The next step is to test variance reduction across independent fresh adapter initializations (`H179` deep ensemble) and representation/capacity changes rather than expanding TRAIN families.

## Integrity and custody

All 60 trajectories and 2,400 optimizer updates completed. The resource guard passed (CPU-only xbabe2, nice 19, thread/RSS/disk/memory ceilings held). The first coordinator invocation was voided before opening inputs because its launch JSON misspelled `prereg_sha256`; it performed zero updates and read no model or outcomes. The corrected launch used the same merged/reviewed source and produced the sole scientific run. Both records are retained in external custody.

Raw panels, per-cell traces, source closure, launch, audit and guard receipts are in the [versioned S3 custody archive](custody.json); archive, manifest and supplemental attempt claim were fetched by exact VersionId and byte-verified. Source commit `6f489fd4b20adcbd58685d9349d0243d97c6eacd` was preregistered in [PR #1493](https://github.com/neverhuman/JopeDime/pull/1493), reviewed at `b42f8868695ee5180ee7e341ddd4530fad0a3d41` (review 5727836812), and merged without source-tree changes.
