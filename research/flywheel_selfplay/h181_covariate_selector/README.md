# H181: covariate-adjusted family-paired selector

**NO_GAIN**. CUPED1 and CUPED_K fail calibration and do not achieve the required family MDE reduction; CUPED1 also exceeds the bias limit. Zero optimizer updates/backward calls; every network weight frozen. This is an exploratory estimator-variance result on the motivating H175 bank, not model lift, a closed-wave re-verdict or serving promotion.

## Hypothesis and setup

Baseline difficulty may explain family gain variance. CUPED1/CUPED_K fit baseline-only control variates on FIT123, excluding evaluated B41 and unused A41. The target is the same policy's finite 205-family TRAIN mean with known pre-treatment cohort moments; POSTSTRAT is descriptive. The strict +.001 margin/LCB95>0/tie rejection remain fixed. Both source 175 slates and policy draws are retained; new 181 bootstrap/target activation streams do not create independent datasets. Full pools repeat the same X across seeds.

Source: prereg #1537, merged 6acd818d6897b2eff929a7bc1c3f2a15decd38e8; rd_argon exact-source review 5728353643. Fresh native91 baseline mean **1.1377941898164525**, utility and candidate cost bytes exact before TRAIN outcome decoding. DOPE **c51a56d0...**, full model **6eb0e998...** before=after, trainable parameters 0, protected reads 0. Real-S3 catalog 276 IDs; 1219 TRAIN contexts/205 families, disjoint DL-Val91/19 baseline-only. 56 original bank cells  + 12 target-power cells × 3 seeds × 200 policy draws; 10k paired bootstrap; CP uses n=200 PER fold, never pooled folds. Read [PREREG](PREREG.md) for estimand, exclusions, calibration and conditional-CI scope.

## Outcome

Values below are medians across three source MC/analysis streams, with 10k shared-family bootstrap 95% intervals (seed reduction axis=0). Intervals condition on fixed folds, policies and fitted nuisance coefficients. MDE intervals recompute grid crossings; >grid denotes right-censoring. Higher power is descriptive unless all three preregistered gates pass.

| Slate / activation | STRICT MDE80 | CUPED1 MDE80 | CUPED_K MDE80 | POSTSTRAT MDE80 |
|---|---:|---:|---:|---:|
| 24 / context | 0.015834 [0.012802, 0.019641] | 0.015834 [0.012802, 0.019656] | 0.064411 [0.051547, 0.079499] | 0.015834 [0.012802, 0.019641] |
| 24 / family | 0.063957 [0.050620, 0.079213] | 0.063957 [0.050620, 0.079213] | >0.063957 (right-censored) | 0.063957 [0.050620, 0.079213] |
| full / context | 0.022802 [0.017698, 0.029217] | 0.022802 [0.017950, 0.054935] | 0.093858 [0.050298, 0.119548] | 0.022802 [0.017698, 0.029217] |
| full / family | 0.094678 [0.072472, 0.121377] | >0.094678 (right-censored) | >0.094678 (right-censored) | 0.094678 [0.072472, 0.121377] |

| Estimator | Family MDE reduction 24/full | Max null/mirror CP95 upper | Max absolute seed/cell bias | Qualifies |
|---|---:|---:|---:|---|
| STRICT | 0.0%/0.0% | 0.031143 | 0.000000 | False |
| CUPED1 | 0.0%/censored | 0.095401 | 0.001054 | False |
| CUPED_K | censored/censored | 0.197911 | 0.000952 | False |
| POSTSTRAT | 0.0%/0.0% | 0.031143 | 0.004245 | False |

Every mirror cell is conservatively gated, even when random_nonoracle improves a poor baseline. CP bounds are marginal one-sided 95% per-cell bounds, not simultaneous coverage or exact H177 pooled/Holm qualification. Bias is the worst absolute per-seed/cell mean, not cancellation across cells. Raw conditional nonpositive-gain admission denominators are retained in S3.

### Power at requested expected true gains

Oracle activation probability is target / frozen-bank headroom; utility is never rescaled. Each policy draw's realized gain varies. Full-pool family activation is shown; compact metrics retain all slates/activation units.

| Expected gain | Realized gain | STRICT power | CUPED1 power | CUPED_K power | POSTSTRAT power |
|---|---:|---:|---:|---:|---:|
| 0.016 | +0.016496 [+0.012256, +0.021962] | +0.049000 [+0.048390, +0.049707] | +0.191000 [+0.167878, +0.215587] | +0.208000 [+0.193512, +0.220439] | +0.049000 [+0.048390, +0.049707] |
| 0.028 | +0.028810 [+0.021663, +0.038206] | +0.228000 [+0.222317, +0.233586] | +0.359000 [+0.337488, +0.384122] | +0.326000 [+0.306512, +0.345490] | +0.228000 [+0.222317, +0.233586] |
| 0.042 | +0.042561 [+0.032206, +0.054998] | +0.526000 [+0.519488, +0.532440] | +0.539000 [+0.519756, +0.559366] | +0.440000 [+0.418853, +0.462805] | +0.526000 [+0.519488, +0.532440] |

### Estimator variance and selected-policy outcomes

At full-pool family activation with expected gain +0.028:

| Estimator | Error variance / STRICT | Bias vs STRICT | Selected actual utility | Selected context-pairwise Elo |
|---|---:|---:|---:|---:|
| STRICT | +1.000000 [+1.000000, +1.000000] | +0.000000 [+0.000000, +0.000000] | +1.397947 [+1.153251, +1.700311] | +1508.832684 [+1508.379894, +1509.351600] |
| CUPED1 | +1.025588 [+1.014261, +1.037820] | +0.000282 [-0.003650, +0.003387] | +1.401831 [+1.156648, +1.704682] | +1510.683850 [+1510.163974, +1511.498956] |
| CUPED_K | +1.056485 [+1.042951, +1.071142] | -0.000034 [-0.004420, +0.003478] | +1.399620 [+1.154913, +1.702249] | +1509.121813 [+1508.513893, +1509.826525] |
| POSTSTRAT | +0.995423 [+0.957431, +1.039571] | -0.000061 [-0.001499, +0.001462] | +1.397947 [+1.153251, +1.700311] | +1508.832684 [+1508.379894, +1509.351600] |

Variance ratio is empirical B-estimator error variance about each same-policy TRUE finite 205-family gain; coefficients and folds are held fixed in the conditional CI. Actual selected-policy utility/Elo use unadjusted outcomes. Context wins/ties are aggregated within family before the two-policy BT prior .01 with r882=1500; Elo is not derived from the sign of a family mean.

## Closed-wave descriptive diagnostics

**POSTHOC ONLY / NO RE-VERDICT.** These reuse correlated TRAIN cohorts, not independent confirmation. H170/H171 use raw inner-B panels; H171 CTRL_EX4 logical 10 = actual 40. H177 uses raw pooled_b candidate columns at actual 40, not selected fallback panels.

| Wave / arm (actual 40) | STRICT family gain | CUPED_K family gain |
|---|---:|---:|
| H170 / CTRL | -0.030752 [-0.056976, -0.004032] | -0.032809 [-0.059076, -0.005542] |
| H170 / CTRL_MEANMATCH | -0.042895 [-0.073734, -0.014701] | -0.043294 [-0.074741, -0.015518] |
| H170 / OC15 | -0.019880 [-0.046446, +0.002670] | -0.021604 [-0.048736, +0.001193] |
| H170 / OC3 | -0.035727 [-0.065338, -0.009236] | -0.036678 [-0.066704, -0.010223] |
| H171 / ACC16 | -0.041592 [-0.072181, -0.011441] | -0.043500 [-0.073940, -0.013866] |
| H171 / ACC4 | -0.031138 [-0.059434, -0.006937] | -0.032093 [-0.061599, -0.008031] |
| H171 / CTRL | -0.033338 [-0.060837, -0.008334] | -0.035496 [-0.062403, -0.009468] |
| H171 / CTRL_EX4 | -0.035555 [-0.063573, -0.011447] | -0.037578 [-0.064906, -0.012956] |
| H177 / ACC16 | -0.034108 [-0.062539, -0.006937] | -0.035180 [-0.064617, -0.009755] |
| H177 / ACC4 | -0.035227 [-0.062668, -0.010197] | -0.036704 [-0.063996, -0.011884] |
| H177 / CTRL | -0.030760 [-0.058332, -0.005955] | -0.033029 [-0.060815, -0.007140] |
| H177 / OC15 | -0.018773 [-0.040326, +0.001438] | -0.020982 [-0.042907, +0.000211] |

## Verification and custody

18/18 data-free tests, Ruff lint/format, all five jope-records gates, rd_argon prereg cross-review and independent raw-array recount PASS. Recount: 68 cases, 112 original delta/win arrays, 2040 independent FIT-only least-squares coefficient panels, 40800 independently sampled paired intervals, 4080 CP bounds, and 816 complete seed-axis CI summaries. All admissions/actual-outcome deployment/gate verdicts recounted. This is targeted local proof; full hosted/workspace CI is not claimed.

Guard: 455 samples; peak RSS 0.973GB, max aggregate threads 2, max CPU 30.8%, min available 76.52GiB, max disk 92.928% (limit 93.8%). No resource violations. CPU xbabe2 nice 19/math threads 1, zero GPU, network optimizer/backward forbidden.

[Compact metrics](metrics.json): 91,587 bytes (<100KB). Raw policies/theta/covariates/decisions/closed-wave panels, source and input closure are in [versioned S3 custody](custody.json): `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H181/codex-t0600/725fa13b44551d6f/H181-evidence.zip`, version `8PIdu5hnthVSZ5BSHKsnyZNtvu4kMGRX`; actual exact-version restore verified all 248 member hashes and decoded 3709 arrays from 130 NPZ files. Source175 banks remain available to other owners; no large object enters Git.

## Verdict and next step

**NO_GAIN** under the preregistered conjunction. Keep the live selector unchanged. Any further estimator work needs a separate preregistration and fresh-fold prospective validation; these motivating-bank or posthoc panels do not qualify a model or revise previous null studies.
