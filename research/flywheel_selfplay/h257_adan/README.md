# H257 — Adan adapter continuation

**COMPLETE / NO_ADVANCE.** ADAN primary: **NO_ADVANCE**. ADAN_HALF secondary: **NO_ADVANCE**.

## Hypothesis and setup

Adan uses smoothed gradient differences and three bias-corrected moments. ADAN uses lr 1e-4, ADAN_HALF lr 5e-5; both use betas (0.98, 0.92, 0.99), weight decay 0.02, epsilon 1e-8 and official proximal decay. CONTROL is exact H170 AdamW lr 1e-4, betas (0.9, 0.999), weight decay 1e-4. All arms retain raw CONTROL loss, identical batch order, global clip 1 and 40 updates. This compares the prescribed optimizer recipes, including different weight decay; it does not isolate gradient-difference smoothing causally.

[Preregistration](https://github.com/neverhuman/JopeDime/commit/2fe50b3131b2ff610e6e2ff9bc1de60a1134f80a) published to v1/next at 2026-11-03T13:39:42.967632+00:00, before Native91, smoke or training. Execution `2fe50b3131b2ff610e6e2ff9bc1de60a1134f80a`. [Frozen protocol](PREREG.md), [folds](fold_manifest.json), [schedules](schedules.json).

Fresh partition 257140; optimizer seeds 257101–257103. Each cell has 123 FIT / 41 SELECT / 41 ASSESS families. All 15 cell locks preceded trained-policy assessment within their cells. Completed **1,800 primary updates plus 3 smoke updates** over 45 arm-cells. Fixed candidate-pool continuation does not establish a live GP-generation mechanism.

## DL-Val91 primary outcome

| Arm vs r882 | Per-seed family gains | Median [CI95] | Positive seeds | Holm p | Median Elo [CI95] | Harm families per seed |
|---|---|---|---|---|---|---|
| CONTROL | +0.00000000 / +0.00000000 / +0.00000000 | +0.00000000 [0.0, 0.0] | 0/3 | control | 1500.000 [1500.0, 1500.0] | [0, 0, 0] of 19 |
| ADAN | +0.00000000 / +0.00000000 / +0.00000000 | +0.00000000 [0.0, 0.0] | 0/3 | 1.0 | 1500.000 [1500.0, 1500.0] | [0, 0, 0] of 19 |
| ADAN_HALF | +0.00000000 / +0.00000000 / +0.00000000 | +0.00000000 [0.0, 0.0] | 0/3 | 1.0 | 1500.000 [1500.0, 1500.0] | [0, 0, 0] of 19 |

Selected step counts across 15 cells per arm: `{"ADAN": {"0": 15}, "ADAN_HALF": {"0": 15}, "CONTROL": {"0": 15}}`. Step 0 is r882 fallback. Per-seed DL-Val utilities average five selected cell-policy utilities; these are not independent replications or an extra prediction ensemble. Family-bootstrap intervals condition on the fixed policies and this exposed development panel. Holm m=2 corrects the two registered contrasts within this wave; it does not correct the campaign-wide adaptive search. All 45 strict arm-cell selectors fell back to r882. The zero gains and zero-width intervals follow from identical baseline policies; they are not evidence that the trained optimizers are equivalent.

| Contrast against exact CONTROL | Per-seed family gains | Median [CI95] |
|---|---|---|
| ADAN − CONTROL | +0.00000000 / +0.00000000 / +0.00000000 | +0.00000000 [0.0, 0.0] |
| ADAN_HALF − CONTROL | +0.00000000 / +0.00000000 / +0.00000000 | +0.00000000 [0.0, 0.0] |

## Strict TRAIN OOF (1,219 contexts / 205 families)

| Arm vs r882 | Per-seed family gains | Median [CI95] | Median Elo [CI95] | Harm families per seed |
|---|---|---|---|---|
| CONTROL | +0.00000000 / +0.00000000 / +0.00000000 | +0.00000000 [0.0, 0.0] | 1500.000 [1500.0, 1500.0] | [0, 0, 0] |
| ADAN | +0.00000000 / +0.00000000 / +0.00000000 | +0.00000000 [0.0, 0.0] | 1500.000 [1500.0, 1500.0] | [0, 0, 0] |
| ADAN_HALF | +0.00000000 / +0.00000000 / +0.00000000 | +0.00000000 [0.0, 0.0] | 1500.000 [1500.0, 1500.0] | [0, 0, 0] |

## Unconstrained step-40 DL-Val91

| Arm vs r882 | Per-seed family gains | Median [CI95] | Median Elo [CI95] | Harm families per seed |
|---|---|---|---|---|
| CONTROL | -0.04199293 / -0.03760994 / -0.03813039 | -0.03813039 [-0.09445254437719248, -0.0032652601375440862] | 1483.548 [1445.6910030785307, 1538.6187154100876] | [9, 8, 9] |
| ADAN | -0.01294048 / -0.01218951 / -0.01121478 | -0.01218951 [-0.03164108461801514, 0.0012536342163219807] | 1488.123 [1464.2836092997277, 1513.7091469160298] | [7, 7, 7] |
| ADAN_HALF | -0.00601758 / -0.00690402 / -0.00489760 | -0.00601758 [-0.015342036581486218, 0.0011686294427933424] | 1495.430 [1476.182053257112, 1520.1112676160399] | [4, 4, 5] |

These diagnostics do not replace the strict DL-Val admission rule.

Both Adan arms lost utility against r882 in all three seeds at step 40. Their descriptive gains against the degrading CONTROL were ADAN +0.02691561 (CI95 [+0.00319563, +0.06853866]) and ADAN_HALF +0.03323278 (CI95 [+0.00324275, +0.07942273]). These paired contrasts do not establish improvement over r882.

## Unconstrained step-40 TRAIN OOF

| Arm vs r882 | Per-seed family gains | Median [CI95] | Median Elo [CI95] | Harm families per seed |
|---|---|---|---|---|
| CONTROL | -0.02955890 / -0.03336225 / -0.02759549 | -0.02955890 [-0.056033681188255566, -0.005545161717333876] | 1484.818 [1475.118675692976, 1495.1211333294773] | [86, 85, 92] |
| ADAN | -0.01921453 / -0.02555339 / -0.02107173 | -0.02107173 [-0.04624107262491264, 0.0006754790345622635] | 1490.885 [1482.8233578566899, 1500.1463589475322] | [67, 75, 74] |
| ADAN_HALF | -0.00580231 / -0.00631478 / -0.01126763 | -0.00631478 [-0.0219331794544386, 0.011074775886195565] | 1497.551 [1492.101998928557, 1503.597262582811] | [46, 43, 47] |

These diagnostics do not replace the strict DL-Val admission rule.

## Unconstrained SELECT-best DL-Val91

| Arm vs r882 | Per-seed family gains | Median [CI95] | Median Elo [CI95] | Harm families per seed |
|---|---|---|---|---|
| CONTROL | -0.00140212 / -0.00905635 / -0.00258500 | -0.00258500 [-0.006857542547367601, 0.0001840538817858839] | 1493.606 [1471.6457970830386, 1520.111811447138] | [6, 8, 4] |
| ADAN | -0.00179713 / -0.00240536 / -0.00145607 | -0.00179713 [-0.004500153322438121, 0.00044443695141181694] | 1500.000 [1481.7088534231407, 1520.1771841917914] | [4, 4, 4] |
| ADAN_HALF | -0.00233699 / -0.00190128 / -0.00251798 | -0.00233699 [-0.005890716764812259, 0.0006416971155745583] | 1497.260 [1478.0347071241533, 1516.470414824626] | [4, 4, 4] |

These diagnostics do not replace the strict DL-Val admission rule.

## Unconstrained SELECT-best TRAIN OOF

| Arm vs r882 | Per-seed family gains | Median [CI95] | Median Elo [CI95] | Harm families per seed |
|---|---|---|---|---|
| CONTROL | +0.00190729 / -0.00523573 / -0.00728471 | -0.00523573 [-0.012651974408346313, 0.004030154914765563] | 1497.755 [1491.2700613181441, 1502.4149256431579] | [10, 16, 20] |
| ADAN | -0.00111584 / -0.00437478 / -0.00423378 | -0.00423378 [-0.012576660649487802, 0.0012386031681189717] | 1497.875 [1493.3618870793996, 1501.6805411054847] | [17, 20, 18] |
| ADAN_HALF | -0.00033232 / -0.00495585 / -0.00285378 | -0.00285378 [-0.0070413860721499745, 0.001660139587477812] | 1498.847 [1494.4211560665076, 1502.337418367635] | [15, 14, 29] |

These diagnostics do not replace the strict DL-Val admission rule.

## Drift, context #27 and historical plateau

| Arm | Step-40 adapter L2 median / min / max |
|---|---|
| CONTROL | 0.37448771 / 0.35851555 / 0.39412046 |
| ADAN | 0.26325109 / 0.23591365 / 0.28853234 |
| ADAN_HALF | 0.13166524 / 0.11783065 / 0.14412468 |

Adan produced smaller adapter drift than CONTROL under these recipes. Both Adan arms retained baseline utility on context #27 in all three step-40 seed mixtures; CONTROL declined on the third. Neither observation establishes the registered utility lift or isolates a causal effect of gradient-difference smoothing.

Zero-based context #27 `43022ae6a8bc83e9e25ea4a2882389152856226f5b2bdd4668db07f2f510ae91`, family `fam-9f67f1449b77152cbf20a6c82926f90496dacd7f05a222376edf0fdfb45f667f`, is retained in every primary panel. Baseline utility 1.122694955279. Values below are each seed’s five-cell utility average; per-cell choices and invalid gates remain in the archived cell receipts.

| Arm | Strict #27 utilities by seed | Step-40 #27 utilities by seed |
|---|---|---|
| r882 | +1.12269496 / +1.12269496 / +1.12269496 | +1.12269496 / +1.12269496 / +1.12269496 |
| CONTROL | +1.12269496 / +1.12269496 / +1.12269496 | +1.12269496 / +1.12269496 / +0.49815596 |
| ADAN | +1.12269496 / +1.12269496 / +1.12269496 | +1.12269496 / +1.12269496 / +1.12269496 |
| ADAN_HALF | +1.12269496 / +1.12269496 / +1.12269496 | +1.12269496 / +1.12269496 / +1.12269496 |

| Arm | Strict DL-Val context means by seed | Difference from plateau 1.1389572 |
|---|---|---|
| r882 | +1.13779419 / +1.13779419 / +1.13779419 | -0.00116301 / -0.00116301 / -0.00116301 |
| CONTROL | +1.13779419 / +1.13779419 / +1.13779419 | -0.00116301 / -0.00116301 / -0.00116301 |
| ADAN | +1.13779419 / +1.13779419 / +1.13779419 | -0.00116301 / -0.00116301 / -0.00116301 |
| ADAN_HALF | +1.13779419 / +1.13779419 / +1.13779419 | -0.00116301 / -0.00116301 / -0.00116301 |

The plateau utility comparison has context weighting, while the admission gain has equal-family weighting. Archived-only r3175 pairwise Elo below uses its matched historical vector as anchor 1500; it is separate from the source-run r882-anchored Elo. H226 CPU replay remains failed and excluded. No new r3175 model evaluation or CPU qualification occurred, and historical winner-selection bias remains. The reference was pinned before H257 prereg/outcomes.

| Arm | Strict Elo lift vs archived r3175 [CI95] | Step-40 Elo lift [CI95] |
|---|---|---|
| r882 | +0.00000 [-10.96375087323986, 10.96375087323986] | +0.00000 [-10.96375087323986, 10.96375087323986] |
| CONTROL | +0.00000 [-10.96375087323986, 10.96375087323986] | -10.96375 [-47.79301473354781, 40.36775524137125] |
| ADAN | +0.00000 [-10.96375087323986, 10.96375087323986] | -18.28371 [-44.07537881909776, 3.653505696515367] |
| ADAN_HALF | +0.00000 [-10.96375087323986, 10.96375087323986] | -3.65351 [-25.619933469944726, 25.619933469944954] |

## Verdict and next step

**NO_ADVANCE**, with primary **NO_ADVANCE** and secondary **NO_ADVANCE**, under the registered median gain >0.002, family LCB95 >0, Holm m=2 p<0.05 and 3/3 positive-seed gates. The CONTROL contrasts and unconstrained diagnostics are descriptive. No serving promotion. Retain the registered comparison; any follow-up requires a separate preregistered hypothesis, not post-outcome tuning of this run.

## Verification, resources and custody

Native91 matched all 91 frozen r882 utilities exactly (mean 1.1377941898164525). The smoke made three actual updates, verified changed parameters/frozen DOPE, and restored r882 exactly before primary training. Terminal replay passed 79,217 choice/label checks across 242 cost files, 180 adapter snapshots, 45 arm-cell update traces, all 45 arm selectors and 15 cell locks, and every strict/unconstrained/step40 panel. Statistical replay uses the same frozen implementation; it is not an independent estimator or independent-agent review.

CPU 52–55 / nice 19 / one numeric thread; A1 guard unchanged. Elapsed 3710.1s, peak sampled aggregate RSS 3,353,071,616 bytes, 3 threads, 2 pause/resume transitions. No process restart, deadline reset or speed verdict. DOPE/non-adapter state stayed frozen; zero protected reads. Smoke and primary counts are reported separately.

Runtime: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H257/astra-t1335/closeout-v2/runtime.tar.gz`; VersionId `uwk4UHTAfuPGXZpk5e7Ri1GoliBjDcRh`; SHA256 `c9cf4aa81fcc8a89ee04dc089c1bca7fd022765c81327057b2c9363f05f3a68d`.

Manifest: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H257/astra-t1335/closeout-v2/manifest.json`; VersionId `_4jKy179e52ROROPMRcIlYRJsHCOV_aj`; SHA256 `c3dac7eb11831ccaf3ee3dc684a660607a32e3c78d9f43944fd6eabc401f1683`. Exact-version archive and all member hashes recovered. [Compact metrics](metrics.json) retain every requested panel; full harm-family identities, optimizer traces, costs, checkpoints, input/source and access receipts are in the runtime archive. No large objects entered Git.

Archive custody required one retry: the first archive’s member manifest disagreed with a verification helper edited during packing. Its failure receipt and manifest are preserved. The replacement `closeout-v2` archive hashes the same bytes it stores; all 552 members passed recovery verification. The scientific run was not repeated.

13 targeted tests, 640 bit-exact official optimizer comparisons, Ruff, JSON, diff-check and source/index pins accompany exact-head same-agent review. Full remote CI NOT_RUN under the explicit PR2071 split; no global records/schema/inventory registration was modified. No serving promotion, other-job control, live-loop access, GPU, xbabe0/xbabe1/xbabe3 or PR2071 mutation. H257 exclusions do not retrospectively qualify r882’s historical training/selection.
