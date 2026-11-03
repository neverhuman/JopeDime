# H250 — per-context standardized loss and balanced sampling

**COMPLETE / NO_ADVANCE.** CTXZ_FB primary: **NO_ADVANCE**. CTXZ secondary: **NO_ADVANCE**.

## Hypothesis and setup

Per-context standardized loss may reduce drift toward high-spread contexts. CTXZ uses z-space smooth-L1 and ListNet KL (CTRL weight0.1/T0.2); CTXZ_FB adds eight uniformly sampled distinct families and one uniform context per family. CONTROL retains its exact raw-scale loss and data order; CTXZ retains that order. All arms use the same AdamW, 40 updates, batch shape, global clip and r882 initialization.

[Preregistration](https://github.com/neverhuman/JopeDime/pull/2103) merged before Native91, smoke or primary updates. Execution `0f8461b7040fc2c51f0a422af0b50b4eb523cdb8`. [Frozen protocol](PREREG.md), [epsilon and sampler hashes](normalization.json), [folds](fold_manifest.json), [schedules](schedules.json).

This tests adapter continuation on fixed candidate pools. Targets and predictions are standardized separately within each sampled 24-candidate training slate; epsilon comes from full-pool FIT label spreads and evaluation uses raw full-pool argmin. Positive-scale invariance is conditional on the epsilon floor being inactive and on already corrected targets; the fixed +1 invalid-cost penalty also retains a native cost scale. CTXZ tests the prescribed normalized-loss recipe, including its failure correction and normalized KL operands; it does not isolate those components individually. The combined primary also changes sampling, so its comparison with CONTROL does not alone identify each component’s contribution or establish a live candidate-generation bottleneck.

Epsilon is 0.1× the median corrected-label full-pool population standard deviation over that outer cell’s optimization FIT subset. Calibration made zero model forwards, updates or utility scores. Invalid-gate utility −2 is cost +2; both normalized arms replace that target by the full context’s worst valid cost +1 before label statistics. Valid labels and evaluation utilities stay intact. Both normalized loss terms use z-pred/z-true, with gradients through prediction mean/std; no extra family loss weights. All 45 per-arm sampler hashes were frozen before prereg.

| Outer fold | Frozen epsilon | FIT contexts / families | Invalid FIT-pool labels replaced |
|---|---:|---:|---:|
| 0 | 0.043985157095 | 782 / 123 | 39309 of 900863 |
| 1 | 0.041376050828 | 726 / 123 | 35175 of 836351 |
| 2 | 0.043957714671 | 685 / 123 | 31204 of 789119 |
| 3 | 0.038650237722 | 703 / 123 | 33795 of 809856 |
| 4 | 0.039707248594 | 761 / 123 | 39974 of 876672 |

Fresh partition seed 250140, optimizer seeds 250101–250103. Each cell has 123 FIT / 41 SELECT / 41 ASSESS families. All 15 cell locks preceded trained-policy assessment. Completed **1,800 primary optimizer updates plus 3 smoke updates** over 45 arm-cells. Applied failure-label replacement occurrences across sampled primary updates: `{"CONTROL": 0, "CTXZ": 4262, "CTXZ_FB": 4431}`; repeated sampled labels are counted repeatedly. These are training targets only.

## DL-Val91 primary outcome

| Arm vs r882 | Per-seed family gains | Median [CI95] | Positive seeds | Holm p | Median Elo [CI95] | Harm families per seed |
|---|---|---|---|---|---|---|
| CONTROL | -0.00115979 / -0.00094410 / +0.00000000 | -0.00094410 [-0.0024292285771845834, 0.0001561630849492865] | 0/3 | control | 1496.346 [1473.4450708123275, 1508.2214201345669] | [4, 3, 0] of 19 |
| CTXZ_FB | -0.00048761 / +0.00000000 / +0.00000000 | +0.00000000 [0.0, 0.0] | 0/3 | 1.0 | 1500.000 [1500.0, 1500.0] | [2, 0, 0] of 19 |
| CTXZ | +0.00000000 / +0.00000000 / +0.00000000 | +0.00000000 [0.0, 0.0] | 0/3 | 1.0 | 1500.000 [1500.0, 1500.0] | [0, 0, 0] of 19 |

Selected step counts across 15 cells per arm: `{"CONTROL": {"0": 13, "10": 2}, "CTXZ": {"0": 15}, "CTXZ_FB": {"0": 14, "20": 1}}`. Step 0 is r882 fallback. Per-seed DL-Val utilities average five selected cell-policy utilities; these are not independent replications or an extra prediction ensemble. Family-bootstrap intervals condition on the fixed policies and this exposed development panel. Holm m=2 corrects the two registered contrasts within this wave; it does not correct the campaign-wide adaptive search. Baseline-only seeds can pin median intervals at zero; that is not equivalence evidence.

| Contrast against exact CONTROL | Per-seed family gains | Median [CI95] |
|---|---|---|
| CTXZ_FB − CONTROL | +0.00067218 / +0.00094410 / +0.00000000 | +0.00067218 [-4.226251606158287e-05, 0.0016367298963206557] |
| CTXZ − CONTROL | +0.00115979 / +0.00094410 / +0.00000000 | +0.00094410 [-0.00015616308494928712, 0.0024292285771845834] |

## Strict TRAIN OOF (1,219 contexts / 205 families)

| Arm vs r882 | Per-seed family gains | Median [CI95] | Median Elo [CI95] | Harm families per seed |
|---|---|---|---|---|
| CONTROL | +0.00259799 / -0.00231434 / +0.00000000 | +0.00000000 [-0.0005272318498181213, 0.00034386929375087514] | 1499.661 [1497.62747063822, 1500.3474058332667] | [8, 6, 0] |
| CTXZ_FB | +0.00029085 / +0.00000000 / +0.00000000 | +0.00000000 [0.0, 0.0] | 1500.000 [1500.0, 1500.0] | [5, 0, 0] |
| CTXZ | +0.00000000 / +0.00000000 / +0.00000000 | +0.00000000 [0.0, 0.0] | 1500.000 [1500.0, 1500.0] | [0, 0, 0] |

## Unconstrained step-40 DL-Val91

| Arm vs r882 | Per-seed family gains | Median [CI95] | Median Elo [CI95] | Harm families per seed |
|---|---|---|---|---|
| CONTROL | -0.03830846 / -0.03661748 / -0.03383254 | -0.03661748 [-0.06744362750161147, -0.008289452538163562] | 1493.600 [1448.6073058278243, 1550.64767394403] | [9, 10, 11] |
| CTXZ_FB | +0.00153738 / +0.00129345 / +0.00315323 | +0.00153738 [-0.0008762034308802845, 0.005340801915667673] | 1520.121 [1503.6314068559975, 1537.6335060280503] | [5, 6, 5] |
| CTXZ | +0.00146851 / +0.00101995 / +0.00079369 | +0.00101995 [-0.0016259946215683044, 0.0046330249984107945] | 1497.257 [1474.3394346889186, 1529.3000917580125] | [6, 6, 7] |

Both normalized arms have positive step-40 gains in all three seeds, but their median gains remain below +0.002 and their family confidence intervals cross zero. CTXZ_FB minus CONTROL is +0.03791093 [CI95 +0.01009522, +0.06940929], a descriptive contrast against a degrading control. These diagnostics do not replace the strict DL-Val admission rule.

## Unconstrained step-40 TRAIN OOF

| Arm vs r882 | Per-seed family gains | Median [CI95] | Median Elo [CI95] | Harm families per seed |
|---|---|---|---|---|
| CONTROL | -0.03101062 / -0.02822451 / -0.03201181 | -0.03101062 [-0.05652320511391656, -0.006826123460449176] | 1485.906 [1474.1120463289558, 1494.4803246547656] | [85, 87, 89] |
| CTXZ_FB | +0.00591348 / +0.01117643 / +0.00353606 | +0.00591348 [-0.008733859195568811, 0.02323451885894383] | 1495.478 [1490.3682747822706, 1501.2856158557727] | [49, 54, 51] |
| CTXZ | +0.00025718 / +0.00597385 / +0.00560565 | +0.00560565 [-0.013855385577644772, 0.023801800551363064] | 1495.647 [1488.8890614520799, 1502.0827146415386] | [54, 57, 46] |

These diagnostics do not replace the strict DL-Val admission rule.

## Unconstrained SELECT-best DL-Val91

| Arm vs r882 | Per-seed family gains | Median [CI95] | Median Elo [CI95] | Harm families per seed |
|---|---|---|---|---|
| CONTROL | -0.00550747 / -0.00158609 / -0.00238336 | -0.00238336 [-0.006326579817301918, 0.0004683531323308425] | 1495.430 [1476.168768019609, 1519.350140603801] | [4, 4, 5] |
| CTXZ_FB | +0.00106591 / +0.00124491 / -0.00586256 | +0.00106591 [-0.0015433198300208015, 0.0034413889352842113] | 1501.827 [1487.1970621198057, 1524.7042769866505] | [5, 5, 6] |
| CTXZ | +0.00173867 / -0.00044498 / +0.00046270 | +0.00046270 [-0.0008364805341821219, 0.001856845393821315] | 1505.482 [1477.1262709797993, 1536.6247950131606] | [5, 7, 5] |

These diagnostics do not replace the strict DL-Val admission rule.

## Unconstrained SELECT-best TRAIN OOF

| Arm vs r882 | Per-seed family gains | Median [CI95] | Median Elo [CI95] | Harm families per seed |
|---|---|---|---|---|
| CONTROL | -0.00425977 / -0.00326517 / +0.00097524 | -0.00326517 [-0.012591932067376303, 0.004575730588887263] | 1500.593 [1496.5258706961629, 1503.5588606790236] | [25, 11, 20] |
| CTXZ_FB | +0.00607374 / -0.00323170 / -0.00228011 | -0.00228011 [-0.013614266269918934, 0.007831137614810732] | 1498.600 [1494.660340748679, 1502.654011350001] | [45, 23, 27] |
| CTXZ | -0.00951751 / +0.00029531 / +0.00163801 | +0.00029531 [-0.022597518076530664, 0.019018541552482643] | 1497.731 [1493.0239066183553, 1502.5762659355235] | [33, 47, 19] |

These diagnostics do not replace the strict DL-Val admission rule.

## Drift, context #27 and historical plateau

| Arm | Step-40 adapter L2 median / min / max |
|---|---|
| CONTROL | 0.37161112 / 0.35560524 / 0.38102777 |
| CTXZ_FB | 0.22377043 / 0.18400081 / 0.28385846 |
| CTXZ | 0.23010878 / 0.19753714 / 0.25932500 |

Zero-based context #27 `43022ae6a8bc83e9e25ea4a2882389152856226f5b2bdd4668db07f2f510ae91`, family `fam-9f67f1449b77152cbf20a6c82926f90496dacd7f05a222376edf0fdfb45f667f`, is retained in every primary panel. Baseline utility 1.122694955279. Values below are each seed’s five-cell utility average; per-cell choices and invalid gates remain in the archived cell receipts.

| Arm | Strict #27 utilities by seed | Step-40 #27 utilities by seed |
|---|---|---|
| r882 | +1.12269496 / +1.12269496 / +1.12269496 | +1.12269496 / +1.12269496 / +1.12269496 |
| CONTROL | +1.12269496 / +1.12269496 / +1.12269496 | +1.12269496 / +0.49815596 / +0.49815596 |
| CTXZ_FB | +1.12269496 / +1.12269496 / +1.12269496 | +1.10835877 / +1.10835877 / +1.10835877 |
| CTXZ | +1.12269496 / +1.12269496 / +1.12269496 | +1.10835877 / +1.12269496 / +1.12269496 |

| Arm | Strict DL-Val context means by seed | Difference from plateau 1.1389572 |
|---|---|---|
| r882 | +1.13779419 / +1.13779419 / +1.13779419 | -0.00116301 / -0.00116301 / -0.00116301 |
| CONTROL | +1.13658342 / +1.13680859 / +1.13779419 | -0.00237378 / -0.00214861 / -0.00116301 |
| CTXZ_FB | +1.13728515 / +1.13779419 / +1.13779419 | -0.00167205 / -0.00116301 / -0.00116301 |
| CTXZ | +1.13779419 / +1.13779419 / +1.13779419 | -0.00116301 / -0.00116301 / -0.00116301 |

Step-40 CTXZ_FB context means are 1.139399149773 / 1.139144495072 / 1.141063358470, above the archived plateau in all three seeds. These are five-cell utility mixtures, not a single checkpoint or an admitted HWM candidate. H250 therefore records this signal in its diagnostic metrics without adding a misleading HWM_CANDIDATES registry entry. Context #27 still declines for CTXZ_FB in all three step-40 seed mixtures.

The plateau utility comparison has context weighting, while the admission gain has equal-family weighting. Archived-only r3175 pairwise Elo below uses its matched historical vector as anchor 1500; it is separate from the source-run r882-anchored Elo. H226 CPU replay remains failed and excluded. No new r3175 model evaluation or CPU qualification occurred, and historical winner-selection bias remains. The reference was pinned before H250 prereg/outcomes.

| Arm | Strict Elo lift vs archived r3175 [CI95] | Step-40 Elo lift [CI95] |
|---|---|---|
| r882 | +0.00000 [-10.96375087323986, 10.96375087323986] | +0.00000 [-10.96375087323986, 10.96375087323986] |
| CONTROL | +0.00000 [-14.622111004910721, 10.96375087323986] | -14.62211 [-55.261921155005666, 36.66926237746179] |
| CTXZ_FB | +0.00000 [-10.96375087323986, 10.96375087323986] | +21.94938 [-3.653505696515367, 51.52155855087426] |
| CTXZ | +0.00000 [-10.96375087323986, 10.96375087323986] | +7.30782 [-18.283714600102257, 40.36775524137147] |

## Verdict and next step

**NO_ADVANCE**, with primary **NO_ADVANCE** and secondary **NO_ADVANCE**, under the registered median gain >0.002, family LCB95 >0, Holm m=2 p<0.05 and ≥2/3 positive-seed gates. The CONTROL contrasts and unconstrained diagnostics are descriptive. No serving promotion. Retain the registered comparison; any follow-up requires a separate preregistered hypothesis, not post-outcome tuning of this run.

## Verification, resources and custody

Native91 matched all 91 frozen r882 utilities exactly (mean 1.1377941898164525). The smoke made three actual updates, verified changed parameters/frozen DOPE, and restored r882 exactly before primary training. Terminal replay passed 82,315 choice/label checks across 251 cost files, 180 adapter snapshots, 45 arm-cell update traces, all 45 arm selectors and 15 cell locks, and every strict/unconstrained/step40 panel. Statistical replay uses the same frozen implementation; it is not an independent estimator or independent-agent review.

CPU48–51/nice19/numeric1; A1 guard unchanged. Elapsed 2496.2s, peak sampled aggregate RSS 3,343,429,632 bytes, 3 threads, 0 pause/resume transitions. No process restart, deadline reset or speed verdict. DOPE/non-adapter state stayed frozen; zero protected reads. Calibration and primary receipts distinguish their costs.

Runtime: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H250/astra-t1131/closeout/runtime.tar.gz`; VersionId `_uh1d9vZLHaxDi_vb3.K_O_mpVnjdg5z`; SHA256 `25e1b74bf6f37b090cb6db114f949379bcafcbdb7c20bdf49586bb46751feb28`.

Manifest: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H250/astra-t1131/closeout/manifest.json`; VersionId `iQz5qaNRy0SA43EI9H9D3InyTSSmQrFE`; SHA256 `778db5b8f1530319141b588e04041379fa818bcff5646524c6d5baf99092b71a`. Exact-version archive and all member hashes recovered. [Compact metrics](metrics.json) retain every requested panel; full harm-family identities, optimizer traces, costs, checkpoints, input/source and access receipts are in the runtime archive. No large objects entered Git.

14 targeted tests, Ruff, schema/JSON, diff-check, five records checks and source/index pins accompany exact-head same-agent review. Full remote CI NOT_RUN under explicit hw_t1131 owner exception. No serving promotion, other-job control, live-loop access, GPU, xbabe0/xbabe3 or PR2071 mutation. H250 exclusions do not retrospectively qualify r882’s historical training/selection.
