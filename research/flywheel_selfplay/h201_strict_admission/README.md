# H201: strict admission and r3175 audit

**INCONCLUSIVE. The sole scientific attempt is INCOMPLETE because native/archive parity failed for r3175.** The zero-update audit and its descriptive reports are closed; no checkpoint or admission policy was adopted.

The pinned r3175 checkpoint replays at context-mean utility **1.1366829009431036**, below r882’s **1.1377941898164525**, while its live journal reports **1.1389572403748425**. One of 91 contexts selects a different program. r882 and r3100 reproduce their archived vectors exactly. This failed gate is retained; the statistics below are explicitly post-failure descriptive analysis of the saved outputs, without another model forward or a primary qualification claim.

## Hypothesis and setup

H1 asks whether the r3175 development gain is compatible with selection noise after repeated admissions. H2 asks whether a strict paired-family lower-bound admission rule would reject all four ties and at least three of four recorded improvements. The original merged [preregistration](PREREG.md) and executed source remain unchanged.

Three immutable checkpoint copies (r882, r3100, r3175), the H197 native Engine/runtime and the H199/H139 first-argmin endpoint are pinned. The native r882 regression passes before candidate forwards or history inference: all 91 utilities and all candidate-cost bytes are exact. DOPE c51a56d0 and all non-adapter tensors match across checkpoints. All inference weights are frozen; optimizer and backward calls are forbidden. Actual optimizer/backward calls: **0 / 0**.

The approved cohort contains **91 contexts from 19 families**, correcting the assignment’s 20 before the run. Family splits are 9/10, with no padded cluster. Inputs join 19 real S3-catalog datasets; no synthetic, sealed, holdout or meta-test payloads are used, and no TRAIN outcome pools are opened. The shared approved feature cache and roster metadata are read-only.

The history snapshot contains **2,418 admissions, rounds 775–3192**, with all per-context vectors. The last journal was initially in flight; both the initial snapshot and completion of that same frontier are retained. Primary history-selection analyses stop at r3175 (**N=2,401**). SMA uses the full frozen snapshot, starting from its actual r774 prehistory vector. No later rounds enter this audit.

Bootstrap seeds 201101–201103 each produce 10,000 paired family draws; reported intervals are the median seedwise endpoints. These seeds repeat Monte Carlo calculations on the same policies and families, rather than supplying three independent models. Fold seeds are 201141–201143, with derived streams rooted at 201150. [Compact metrics](metrics.json) retain every seed’s intervals.

## Native parity gate

| Checkpoint | CPU-native context mean | Archived context mean | Different contexts |
|---|---:|---:|---:|
| r882 | 1.1377941898164525 | 1.1377941898164525 | 0/91 |
| r3100 | 1.1383668110784328 | 1.1383668110784328 | 0/91 |
| r3175 | 1.1366829009431036 | 1.1389572403748425 | 1/91 |

The affected context is `25aff0bcd9fb3ea4dd28c6a71463b1e02d8293401f40a2e48e56c2b5a3ab54d8`. CPU-native argmin is pool index 68, utility 0.285862976884233; the archived selected program is index 521, utility 0.492827865172446. Their saved CPU costs differ by 0.00003719329833984375. The selection changes realized utility by 0.20696488828821297.

The required checkpoint SHA `1d6ea900ebed8d2a05c3b3f37576adc15aadd2b53c585bfd5cb71a103e9ff97b` is exact. Its 29,211-parameter adapter hash `9d9814193d64171bbb9caa40e7ebcb2ea12cf6ab967cf8bd3509c8ea6730e37b` matches the journal, and every non-adapter tensor matches r882. Projection metadata says disabled. A wrong checkpoint or DOPE drift is therefore not the demonstrated cause. The live utility routine evaluates one context per forward, whereas the inherited CPU Engine groups up to eight; execution device and batch shape are possible contributors, **not an established diagnosis**. No additional inference was run.

## Descriptive comparisons

Native comparisons use the saved CPU costs and real pool labels. Archived comparisons use the immutable live per-context vectors. They are reported separately. Family means weight all 19 families equally; the context means above weight all 91 contexts equally.

| Contrast | Saved CPU-native family gain [CI95] | Archived-live family gain [CI95] |
|---|---:|---:|
| R3175_minus_R882 | -0.00106450 [-0.00653573, +0.00269232] | +0.00111408 [-0.00046416, +0.00324083] |
| R3175_minus_R3100 | -0.00161301 [-0.00653573, +0.00169671] | +0.00056557 [+0.00000000, +0.00169671] |
| R3100_minus_R882 | +0.00054851 [-0.00069623, +0.00234177] | +0.00054851 [-0.00069623, +0.00234177] |

| Policy | CPU-native Elo [CI95] | Archived-live Elo [CI95] | Native family harm | Native context harm | Native exact -2 count /91 |
|---|---:|---:|---:|---:|---:|
| R882 | 1500.000 [1500.000, 1500.000] | 1500.000 [1500.000, 1500.000] | 0.000000 | 0.000000 | 0 |
| R3100 | 1500.000 [1489.036, 1510.963] | 1500.000 [1489.036, 1510.964] | 0.052632 | 0.010989 | 0 |
| R3175 | 1496.346 [1485.378, 1507.309] | 1500.000 [1489.036, 1510.964] | 0.105263 | 0.032967 | 0 |

Elo uses context-pair wins/ties before equal-family weighting, a 1e-4 tie band, Bradley–Terry MM400 with symmetric prior .01, and r882=1500. Harm is the fraction with strictly lower utility than r882.

## Descriptive selection-noise checks

The empirical null centers each historical candidate’s equal-family gain to zero, then resamples shared families across all candidates, preserving their dependence. It is conditional on the adaptive history and does not identify the true distribution of independent no-gain candidates.

For N=2,401, observed maximum family gain is +0.00111408; expected null maximum is **+0.01460916**, null-max CI95 [0.00020779085934881247, 0.03867908420122163], and max-test p **0.87131287**. The requested numerical WINNERS_CURSE criterion is satisfied descriptively by both the saved-native and archived-live r3175 gain intervals and this max test. The formal preregistered qualification remains INCONCLUSIVE because the native/archive parity gate failed.

| Fold seed | Selected archived candidate round on 9 families | Gain on 10 remaining families [CI95] |
|---|---:|---:|
| 201141 | 782 | -0.06225346 [-0.18753540, +0.00094024] |
| 201142 | 860 | -0.00567444 [-0.01702333, +0.00000000] |
| 201143 | 1382 | -0.01543379 [-0.03101503, -0.00132284] |

All three held-half gains are negative. These are retrospective selections from archived vectors; both halves already influenced the live adaptive history. They are not independent prospective validation.

## Descriptive SMA replay

SMA admits only when the candidate’s equal-family mean gain and paired-family lower 95% endpoint are both strictly positive. Exact ties reject; rejected proposals leave the SMA incumbent unchanged. The observed-incumbent screen is retained separately and does not replace the counterfactual chain.

| Original live action | Rows | Sequential SMA accepts | Sequential SMA rejects | Observed-incumbent screen accepts |
|---|---:|---:|---:|---:|
| accept_improvement | 4 | 0 | 4 | 0 |
| accept_tie | 4 | 0 | 4 | 0 |
| restore_best_model_and_optimizer | 2410 | 0 | 2410 | 0 |

**Descriptive H2 holds:** SMA rejects all four ties and all four improvements; total 0 accepts /2,418 rejects, retaining the initial r774 vector. The current rule records eight accepts. This is evidence about this replay, not proof of useful future power or sequential false-positive control. The [SMA proposal](SMA_PROPOSAL.md) requests a separate reviewed implementation and calibration; no live change was made.

## Verification, custody and verdict

Prereg PR #1670 merged `da5a4dd13a83e0827df1ff1b39583569c0e2ec4c`, exact-head same-agent review 5730331796. Thirteen data-free controls, Ruff/format, source pins and all five jope-records gates pass. The single native attempt exited at the preregistered parity gate, with no resource violation and no inference retry. This is not a scientific COMPLETE or parity PASS.

Independent verification recomputed all 273 native choices from 91 hashed pools, every native and archived contrast/seed CI, dependent null maxima, all three family selections, all 2,418 sequential/observed SMA decisions, BT Elo and harm. It confirms the failed primary gate and INCONCLUSIVE verdict. Separate checkpoint checks confirm exact adapter/journal identity and non-adapter equality. Post-failure work constructed no model and performed no inference.

[S3 custody](custody.json): `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H201/codex-t1356/4f0f35f65f1bc5df/H201-evidence.zip` VersionId `lO63G9i7srsHuGl3xuq8Q1Y0DfJK9uvz`. Actual versioned restoration verifies all 230 member payloads, 91 DL-Val pools, 1,195 arrays and all three full checkpoint dictionaries. Original failed output/traceback/resource summary, initial/completed snapshots, descriptive matrices and every SMA decision are retained. No checkpoint or npz enters Git.

Compact metrics: 18,390 bytes (<100 KB). Full hosted/workspace CI is not claimed; the owner authorized targeted local checks and self-squash without remote CI wait. Both the scientific failure and descriptive completion are explicit.

**Formal verdict: INCONCLUSIVE; primary attempt INCOMPLETE.** Keep the live flywheel read-only. Resolve native/archive execution-contract parity in a separately registered qualification before judging the saved r3175 checkpoint for adoption. Review the SMA proposal separately, including power and repeated-testing calibration.
