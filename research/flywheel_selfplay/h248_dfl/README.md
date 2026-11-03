# H248 — decision-focused expected-regret continuation

**COMPLETE / NO_ADVANCE.** DFL_MIX primary: **NO_ADVANCE**. DFL secondary: **NO_ADVANCE**.

## Hypothesis and setup

Directly differentiating the expected regret of a softmin choice may generate better adapters than the inherited surrogate. DFL_MIX adds 0.5×DFL to exact H170 CTRL; DFL uses expected regret plus the unchanged CTRL ListNet KL term. All three arms use the same AdamW, 40 updates, candidate/context order, global clip and r882 initialization.

[Preregistration](https://github.com/neverhuman/JopeDime/pull/2100) merged before Native91, smoke or primary updates. Execution `e93e607a54be801171a72a01896de1c984e7a99f`. [Frozen protocol](PREREG.md), [temperatures](temperatures.json), [folds](fold_manifest.json), [schedules](schedules.json).

This is adapter continuation on fixed candidate pools. The differentiable objective is a softmin relaxation over the sampled 24-candidate training slate; temperature comes from full-pool scores, and utility uses discrete full-pool argmin. The experiment therefore tests this particular loss/scale/slate regime. Its verdict alone cannot establish that live candidate generation is the bottleneck or that every decision-focused objective fails.

Temperature is the median frozen-r882 full-pool score standard deviation over that outer cell’s optimization FIT subset, frozen before prereg with zero updates and no assessment scoring. Invalid-gate utility −2 is native cost +2; DFL alone replaces that cost by the full context’s worst valid cost +1. Valid costs and evaluation labels remain intact. Fixed inverse-FIT-family context-frequency weights give the DFL term equal-family weighting under the inherited uniform-context sampler; CONTROL and its KL retain their original reductions.

| Outer fold | Frozen T | FIT contexts / families | Invalid FIT-pool labels replaced |
|---|---:|---:|---:|
| 0 | 0.956125533983 | 718 / 123 | 35132 of 827135 |
| 1 | 0.962811650156 | 701 / 123 | 34000 of 807551 |
| 2 | 0.990905250052 | 716 / 123 | 34481 of 824832 |
| 3 | 0.969651794949 | 782 / 123 | 38369 of 900864 |
| 4 | 0.952762145358 | 740 / 123 | 37475 of 852479 |

Fresh partition seed 248140, optimizer seeds 248101–248103. Each cell has 123 FIT / 41 SELECT / 41 ASSESS families. All 15 cell locks preceded trained-policy assessment. Completed **1,800 primary optimizer updates plus 3 smoke updates** over 45 arm-cells. Applied failure-label replacement occurrences across sampled primary updates: `{"CONTROL": 0, "DFL": 3895, "DFL_MIX": 3895}`; repeated sampled labels are counted repeatedly. These are training targets only.

## DL-Val91 primary outcome

| Arm vs r882 | Per-seed family gains | Median [CI95] | Positive seeds | Holm p | Median Elo [CI95] | Harm families per seed |
|---|---|---|---|---|---|---|
| CONTROL | -0.00038089 / +0.00000000 / +0.00000000 | +0.00000000 [0.0, 0.0] | 0/3 | control | 1500.000 [1500.0, 1500.0] | [3, 0, 0] of 19 |
| DFL_MIX | -0.00076178 / +0.00000000 / -0.00093966 | -0.00076178 [-0.002356671110696413, 0.00034469945685266774] | 0/3 | 1.0 | 1500.000 [1480.7557875471384, 1510.9637508732399] | [3, 0, 3] of 19 |
| DFL | -0.00083937 / +0.00000000 / -0.00067193 | -0.00067193 [-0.0016124892909238763, 0.00015611771077694554] | 0/3 | 1.0 | 1496.346 [1484.4642767039882, 1505.5042402160896] | [4, 0, 3] of 19 |

Selected step counts across 15 cells per arm: `{"CONTROL": {"0": 14, "10": 1}, "DFL": {"0": 13, "20": 2}, "DFL_MIX": {"0": 12, "10": 3}}`. Step 0 is r882 fallback. Per-seed DL-Val utilities average five selected cell-policy utilities; these are not independent replications or an extra prediction ensemble. Family-bootstrap intervals condition on the fixed policies and this exposed development panel. Baseline-only seeds can pin median intervals at zero; that is not equivalence evidence.

| Contrast against exact CONTROL | Per-seed family gains | Median [CI95] |
|---|---|---|
| DFL_MIX − CONTROL | -0.00038089 / +0.00000000 / -0.00093966 | -0.00038089 [-0.0014872573473729337, 0.0003446994568526473] |
| DFL − CONTROL | -0.00045848 / +0.00000000 / -0.00067193 | -0.00045848 [-0.0009169523173570413, 0.0] |

## Strict TRAIN OOF (1,219 contexts / 205 families)

| Arm vs r882 | Per-seed family gains | Median [CI95] | Median Elo [CI95] | Harm families per seed |
|---|---|---|---|---|
| CONTROL | +0.00116734 / +0.00000000 / +0.00000000 | +0.00000000 [0.0, 0.0] | 1500.000 [1500.0, 1500.0] | [7, 0, 0] |
| DFL_MIX | +0.00079649 / +0.00000000 / -0.00549852 | +0.00000000 [-0.0012925683986716755, 0.0002060547649660412] | 1499.901 [1496.7232270057425, 1500.1129761769685] | [15, 0, 10] |
| DFL | -0.00149935 / +0.00000000 / -0.00221101 | -0.00149935 [-0.005041246358648705, 0.0004992565922443385] | 1499.294 [1496.5823211675013, 1500.1129761726938] | [11, 0, 11] |

## Unconstrained step-40 DL-Val91

| Arm vs r882 | Per-seed family gains | Median [CI95] | Median Elo [CI95] | Harm families per seed |
|---|---|---|---|---|
| CONTROL | -0.02968709 / -0.02568014 / -0.02639127 | -0.02639127 [-0.058714422024076104, -0.0031865438820820892] | 1495.432 [1441.0783752795833, 1560.8549655487568] | [9, 8, 9] |
| DFL_MIX | -0.02623628 / -0.01394709 / -0.01678486 | -0.01678486 [-0.039248379016393976, -0.0006516404788255408] | 1499.086 [1451.2281491772205, 1564.5832041404853] | [9, 9, 9] |
| DFL | -0.00820466 / -0.00666076 / -0.00732528 | -0.00732528 [-0.018646217889818253, 0.0011341833428795485] | 1509.137 [1480.4637276406597, 1538.725041394465] | [7, 7, 7] |

These diagnostics do not replace the strict DL-Val admission rule.

## Unconstrained step-40 TRAIN OOF

| Arm vs r882 | Per-seed family gains | Median [CI95] | Median Elo [CI95] | Harm families per seed |
|---|---|---|---|---|
| CONTROL | -0.03276783 / -0.03029350 / -0.02909406 | -0.03029350 [-0.05886753264314093, -0.00450328932896307] | 1484.488 [1474.4618647190412, 1495.7278607602063] | [90, 94, 93] |
| DFL_MIX | -0.03231121 / -0.02929110 / -0.02681973 | -0.02929110 [-0.05684453208645508, -0.005920235133525938] | 1487.255 [1476.5138369660453, 1496.7831430495378] | [87, 87, 86] |
| DFL | -0.00872321 / -0.01058317 / -0.01296012 | -0.01058317 [-0.028739144636258346, 0.007230856327584001] | 1493.715 [1486.9789079591264, 1501.7849732548386] | [62, 64, 60] |

These diagnostics do not replace the strict DL-Val admission rule.

## Unconstrained SELECT-best DL-Val91

| Arm vs r882 | Per-seed family gains | Median [CI95] | Median Elo [CI95] | Harm families per seed |
|---|---|---|---|---|
| CONTROL | -0.00263178 / -0.00315119 / -0.00306818 | -0.00306818 [-0.007173078118309716, 0.0005251297284027566] | 1490.865 [1467.9706955145928, 1512.7952321230146] | [4, 4, 4] |
| DFL_MIX | -0.00076178 / -0.00085297 / -0.00260971 | -0.00085297 [-0.0029575882771797433, 0.000516339605940339] | 1498.172 [1489.031121194164, 1509.1474623000086] | [3, 3, 4] |
| DFL | -0.00110709 / -0.00233095 / -0.00217647 | -0.00217647 [-0.006156717509282918, 0.000966070536096971] | 1494.517 [1475.3015038689048, 1516.4515747842947] | [3, 5, 4] |

These diagnostics do not replace the strict DL-Val admission rule.

## Unconstrained SELECT-best TRAIN OOF

| Arm vs r882 | Per-seed family gains | Median [CI95] | Median Elo [CI95] | Harm families per seed |
|---|---|---|---|---|
| CONTROL | +0.00021727 / +0.00087276 / -0.00352396 | +0.00021727 [-0.006767777542752377, 0.0032145462668270123] | 1499.972 [1495.6388782851764, 1503.248129505926] | [20, 21, 21] |
| DFL_MIX | +0.00079649 / -0.00549807 / -0.00686307 | -0.00549807 [-0.014114273163307033, 0.0007797283047200334] | 1499.576 [1496.0312252126398, 1502.4856066804894] | [15, 11, 20] |
| DFL | -0.00287478 / +0.00448426 / -0.00128737 | -0.00128737 [-0.00518086884648921, 0.0025904149418074706] | 1499.153 [1494.762795104601, 1503.050409418734] | [36, 25, 25] |

These diagnostics do not replace the strict DL-Val admission rule.

## Drift, context #27 and historical plateau

| Arm | Step-40 adapter L2 median / min / max |
|---|---|
| CONTROL | 0.37447296 / 0.35369577 / 0.38787110 |
| DFL_MIX | 0.34246535 / 0.32094288 / 0.35757314 |
| DFL | 0.26535597 / 0.23514534 / 0.29518092 |

Zero-based context #27 `43022ae6a8bc83e9e25ea4a2882389152856226f5b2bdd4668db07f2f510ae91`, family `fam-9f67f1449b77152cbf20a6c82926f90496dacd7f05a222376edf0fdfb45f667f`, is retained in every primary panel. Baseline utility 1.122694955279. Values below are each seed’s five-cell utility average; per-cell choices and invalid gates remain in the archived cell receipts.

| Arm | Strict #27 utilities by seed | Step-40 #27 utilities by seed |
|---|---|---|
| r882 | +1.12269496 / +1.12269496 / +1.12269496 | +1.12269496 / +1.12269496 / +1.12269496 |
| CONTROL | +1.12269496 / +1.12269496 / +1.12269496 | +1.12269496 / +1.12372142 / +1.12269496 |
| DFL_MIX | +1.12269496 / +1.12269496 / +1.12269496 | +1.12269496 / +1.12269496 / +1.12269496 |
| DFL | +1.12269496 / +1.12269496 / +1.12269496 | +1.12269496 / +1.12384880 / +1.12269496 |

| Arm | Strict DL-Val context means by seed | Difference from plateau 1.1389572 |
|---|---|---|
| r882 | +1.13779419 / +1.13779419 / +1.13779419 | -0.00116301 / -0.00116301 / -0.00116301 |
| CONTROL | +1.13739656 / +1.13779419 / +1.13779419 | -0.00156064 / -0.00116301 / -0.00116301 |
| DFL_MIX | +1.13699892 / +1.13779419 / +1.13681322 | -0.00195828 / -0.00116301 / -0.00214398 |
| DFL | +1.13691793 / +1.13779419 / +1.13709272 | -0.00203927 / -0.00116301 / -0.00186448 |

The plateau utility comparison has context weighting, while the admission gain has equal-family weighting. Archived-only r3175 pairwise Elo below uses its matched historical vector as anchor 1500; it is separate from the source-run r882-anchored Elo. H226 CPU replay remains failed and excluded. No new r3175 model evaluation or CPU qualification occurred, and historical winner-selection bias remains. The reference was pinned before H248 prereg/outcomes.

| Arm | Strict Elo lift vs archived r3175 [CI95] | Step-40 Elo lift [CI95] |
|---|---|---|
| r882 | +0.00000 [-10.96375087323986, 10.96375087323986] | +0.00000 [-10.96375087323986, 10.96375087323986] |
| CONTROL | +0.00000 [-10.96375087323986, 10.96375087323986] | -7.30782 [-51.52155855087426, 44.07537881909798] |
| DFL_MIX | +0.00000 [-14.622111004910721, 14.622111004910721] | -7.30782 [-47.79301473354781, 44.07537881909798] |
| DFL | +0.00000 [-14.622111004910721, 10.96375087323986] | +10.96375 [-32.97903122275261, 62.87636796863738] |

## Verdict and next step

**NO_ADVANCE**, with primary **NO_ADVANCE** and secondary **NO_ADVANCE**, under the registered median gain >0.002, family LCB95 >0, Holm m=2 p<0.05 and ≥2/3 positive-seed gates. The CONTROL contrasts and unconstrained diagnostics are descriptive. No serving promotion. The strict selectors retained r882 in 12/15 DFL_MIX cells and 13/15 DFL cells. At step 40, DFL had a smaller negative DL-Val median gain (−0.00732528) than CONTROL (−0.02639127), but remained below r882; this diagnostic does not rescue the failed admission. Retain the registered comparison; any follow-up requires a separate preregistered hypothesis, not post-outcome tuning of this run.

## Verification, resources and custody

Native91 matched all 91 frozen r882 utilities exactly (mean 1.1377941898164525). The smoke made three actual updates, verified changed parameters/frozen DOPE, and restored r882 exactly before primary training. Terminal replay passed 82,045 choice/label checks across 251 cost files, 180 adapter snapshots, 45 arm-cell update traces, all 45 arm selectors and 15 cell locks, and every strict/unconstrained/step40 panel. Statistical replay uses the same frozen implementation; it is not an independent estimator or independent-agent review.

CPU48–51/nice19/numeric1; A1 guard unchanged. Elapsed 2179.2s, peak sampled aggregate RSS 3,396,210,688 bytes, 3 threads, 0 pause/resume transitions. No process restart, deadline reset or speed verdict. DOPE/non-adapter state stayed frozen; zero protected reads. Calibration and primary receipts distinguish their costs.

Runtime: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H248/astra-t1035/closeout/runtime.tar.gz`; VersionId `daPmSlgYM.WUBpJqzQul08hYiWsQI8uC`; SHA256 `128b5547cc9cfa202d00631617183c4c83e759e4ee175670ab5f3fd0c804c9a1`.

Manifest: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H248/astra-t1035/closeout/manifest.json`; VersionId `Ukai2cVzINs6WVK1EN.NrzANOsXsDgzi`; SHA256 `3f5dc927a89e31a80305f8bf7445783a79b4de8fe2bf13cad91b2befc4a431e1`. Exact-version archive and all member hashes recovered. [Compact metrics](metrics.json) retain every requested panel; full harm-family identities, optimizer traces, costs, checkpoints, input/source and access receipts are in the runtime archive. No large objects entered Git.

13 targeted tests, Ruff, schema/JSON, diff-check, five records checks and source/index pins accompany exact-head same-agent review. Full remote CI NOT_RUN under explicit hw_t1035 owner exception. No serving promotion, other-job control, live-loop access, GPU, xbabe0/xbabe3 or PR2071 mutation. H248 exclusions do not retrospectively qualify r882’s historical training/selection.
