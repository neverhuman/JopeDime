# H170: one-cycle LR and inverse momentum

**NO_ADVANCE — all 15 cross-fit cells / 60 trajectories / 2,400 optimizer updates completed.** [Metrics](metrics.json), [preregistration](PREREG.md), [applied schedules](schedules.json), [S3 custody](custody.json). No serving promotion.

Hypothesis: a one-cycle LR/momentum policy could regularize r882 continuation at a fixed 40-update budget. CTRL uses AdamW lr=1e-4, beta1=.9. OC3/OC15 warm from 1e-5 to peaks 3e-4/1.5e-4 at update 12, then cosine-anneal to 1e-6 at 40; beta1 cycles .95 -> .85 -> .95 and beta2 stays .999. CTRL_MEANMATCH uses constant lr=0.0001481125, beta1=.9, matching OC3's discrete LR sum. This compares the whole LR/momentum policy at matched mean LR, not LR shape alone or identical optimizer trajectories.

Frozen DOPE c51a56d0 and every non-adapter byte are checked after each update/evaluation. Only 47 adapter tensors / 29,211 parameters may train on a read-only r882 copy. Exact native 91-context baseline utilities pass (mean 1.1377941898164525), and all step-zero cached/native cost bits match before opening TRAIN outcome pools. All 276 dataset IDs join origin=real in the actual pinned S3 catalog GET; 1,319 numeric input members were rehashed against versioned recovery custody. No protected/refused reads or live-loop writes.

Seeds 170101–170103, fold 170140, bootstrap 170141; 1,219 TRAIN contexts / 205 families in 5 folds. Each cell fits 123 families, selects on 41 and assesses 41; 91-context/19-family DL-Val is secondary. H156's paired 8-context/24-candidate batches and loss are inherited. Within each cell, A/B/Val never fit weights or set optimizer schedules. Strict inner choice uses the earliest best step among 0/10/20/30/40, gain > .001 with 1e-12 tie rejection and 95% bootstrap LCB > 0, else r882. Decisions are fsynced before any trained-policy A/Val assessment.

## Primary: family out-of-fold utility

| Policy | Median family gain vs r882 | 95% family CI | Median Elo |
| --- | ---: | --- | ---: |
| CTRL | +0.000000000 | [+0.000000000, +0.000000000] | 1500.000 |
| OC3 | +0.000000000 | [+0.000000000, +0.000000000] | 1500.000 |
| OC15 | +0.000000000 | [+0.000000000, +0.000000000] | 1500.000 |
| CTRL_MEANMATCH | +0.000000000 | [+0.000000000, +0.000000000] | 1500.000 |

Points are medians of the three per-seed equal-family mean gains. Intervals use 10,000 shared family draws, with the median seed effect calculated in every draw. Holm adjusts the three treatments versus r882. Advance requires median > .001, LCB > 0 and Holm p < .05. Canonical H148 400-iteration Bradley–Terry MM uses family-weighted context-pairwise wins, tie margin 1e-4, prior .01, r882=1500 and 1,000 family-bootstrap Elo draws. Full per-seed effects, adjusted p-values and Elo intervals are in metrics.json; medians of paired contrasts need not equal differences between arm medians.

The registered median and CI are exactly zero because seeds 170101 and 170102 retain r882 throughout. In seed 170103, the final cell selects step 10 for CTRL, OC15 and CTRL_MEANMATCH; their OOF family gains are -0.002990847, -0.000178169 and -0.007327091, respectively. OC3 retains r882 in all 15 cells. Every conditional bootstrap draw still has two zero seed effects, so its median remains zero. These point-mass intervals describe that registered functional; they do not establish that updated policies are equivalent to r882 or harmless.

| Policy | Seed 170101 gain | Seed 170102 gain | Seed 170103 gain |
| --- | ---: | ---: | ---: |
| CTRL | 0 | 0 | -0.002990847 |
| OC3 | 0 | 0 | 0 |
| OC15 | 0 | 0 | -0.000178169 |
| CTRL_MEANMATCH | 0 | 0 | -0.007327091 |

## Secondary: historical DL-Val

| Policy | Median family gain vs r882 | 95% family CI | Median Elo |
| --- | ---: | --- | ---: |
| CTRL | +0.000000000 | [+0.000000000, +0.000000000] | 1500.000 |
| OC3 | +0.000000000 | [+0.000000000, +0.000000000] | 1500.000 |
| OC15 | +0.000000000 | [+0.000000000, +0.000000000] | 1500.000 |
| CTRL_MEANMATCH | +0.000000000 | [+0.000000000, +0.000000000] | 1500.000 |

Each seed averages the context utilities of its five selected policies before 19-family inference. This is a uniform-policy-mixture utility expectation, not averaging weights or choosing from Val, and 455 policy-context observations are not treated as independent.

## Descriptive unconstrained best

| Policy | Median family gain vs r882 | 95% family CI | Median Elo |
| --- | ---: | --- | ---: |
| CTRL | -0.004320776 | [-0.013056780, +0.002994171] | 1497.703 |
| OC3 | -0.003655517 | [-0.009512215, +0.000692516] | 1499.452 |
| OC15 | -0.011603045 | [-0.032841785, +0.001530500] | 1496.166 |
| CTRL_MEANMATCH | -0.006459436 | [-0.018204915, +0.004030427] | 1496.712 |

The same B-best step including 0 is used without the eligibility gates, locked before A/Val; it is not an A/Val oracle and cannot advance. Nonzero choices out of 15 cells per arm are shown below.

| Policy | Strict nonzero picks | Unconstrained nonzero picks |
| --- | ---: | ---: |
| CTRL | 1 | 6 |
| OC3 | 0 | 5 |
| OC15 | 1 | 6 |
| CTRL_MEANMATCH | 1 | 6 |

## Parameter movement and replication

| Arm | Median step-40 L2 | Range over 15 trajectories |
| --- | ---: | --- |
| CTRL | 0.377898787 | [0.357789558, 0.393039018] |
| OC3 | 0.526898461 | [0.512174641, 0.572610879] |
| OC15 | 0.262291151 | [0.251313948, 0.281573353] |
| CTRL_MEANMATCH | 0.565915816 | [0.534173110, 0.590813803] |

Drift is float64 L2 over all 29,211 coordinates relative to private r882, including inactive tensors. None contributes zero to gradient diagnostics while native AdamW skips those inactive gradients; observed None tensor counts across steps: [6]. These are trainable-inventory counts, not a claim every coordinate received a data gradient.

Exact-identical utility group sizes (matched ordered contexts) are strict OOF [12], strict Val [12, 2], unconstrained OOF [3], unconstrained Val [3]. Labels, per-outer/per-snapshot selector groups and adapter SHA groups are in metrics.json. All arms selector-identical to CTRL: False. All 60 fitted step-40 adapters have distinct hashes. The selected-adapter duplicate group contains the 57 r882 fallbacks. Utility-group labels include the three baseline-reference seed panels. Step-0/fallback duplication is expected and is not independent prediction replication. Distinct adapter hashes also do not imply independent predictions.

## Verification, custody and limits

Independent recount passes 104,630 saved choice/label/gate checks, all 240 snapshot hashes, 2,400 actual LR/beta traces, 60 step-40 drifts, all selector decisions, 28 utility contrasts, canonical Elo points/CIs and duplicate-group disclosures. Fresh native replay matches 91 baseline cost vectors bit-for-bit and 36 trained-checkpoint choices/utilities. Source/input/asset/parent/package pins and r882 read-only state remain unchanged. Twelve data-free controls, constructor-only smoke and all five static records gates pass; the missing four non-test source-classification entries found in initial review were repaired before merge/launch without changing scientific bytes. Full hosted/native workspace qualification is not claimed.

Scientific source fcf6861eb2cca547f91b7c231d79cbcbb74e0bbb was merged in prereg PR #1451 after REDTEAM review 5727276813 and exact-head author review 5727279381. CPU xbabe2, nice 19, one math thread: measured supervisor/worker peak 3 threads, 3.453 GB RSS, maximum host CPU 62.1%, raw load 55.554; minimum available RAM 66.12 GiB and free disk 629.00 GB. Hard limits remained satisfied. No GPU, speed metrics or serving change.

This is conditional continuation on historically exposed TRAIN/r882 and repeatedly exposed Val. Family bootstrap conditions on fitted/selected policies and does not retrain during resampling; it does not establish full calibrated uncertainty for overlapping fitted policies. A fallback panel does not establish every nonzero trajectory harms or calibrated selector power; the planted +.005 fixture tests mechanical reachability. Original H50/H58 derivative S3 versions are not established; the exact numeric closure is recovered from the versioned H144 archive and runtime sources are separately pinned.

Evidence manifest: s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H170/codex-t0120/580a1829356ad171/manifest.json (VersionId bgkT_BVVsiXikfKsZzmk56Cjhc2Phcu8, SHA256 e3c71664ec18be3002d5e9e8e7aad39b12c90fdbadfbe7bc3830439ede70a620). Create-only uploads and actual exact-version byte-readback verify the archive and every member; large objects and source bundles stay outside Git. Next step: Retain r882 and record this fixed screen; any follow-up requires its own prospective assignment and registration.
