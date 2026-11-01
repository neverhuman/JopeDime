# H175: selector power and oracle headroom

**COMPLETE — MIXED. Zero optimizer updates, zero backward calls, all model tensors frozen.** [Compact metrics](metrics.json) (83,526 bytes), [preregistration](PREREG.md), [S3 custody](custody.json). The single CPU audit completed all 56 cases and 168,000 candidate/fold/draw decisions per rule; no model advancement or serving change.

## Hypothesis and setup

The audit asks whether prior r882 continuation nulls reflect little available candidate utility or a strict selector with limited power. It uses the fixed real-S3 regression cohort: 1,219 TRAIN contexts / 205 families, three seeds 175101–175103, five family folds (123 unused FIT / 41 B select / 41 A assess), fold 175140 and bootstrap 175141. Each f=0,0.5,1,2,5,10,20% has 200 counterfactual draws per seed/fold, with pointwise oracle or random-nonoracle replacements and context or whole-family activation. These are hypothetical policies on real labels; no synthetic dataset or learned policy is created. Masks/slates use IDs only, and role-local B decisions are fsynced before A. Each global policy draw is revisited across five folds; 3×5×200 decisions are not 3,000 independent policies/datasets.

H170/H165 evaluation uses full pools. The requested 24 slate keeps the same four storage-order native reveals plus 20 prespecified random remaining candidates per seed/context, selecting from the corresponding cached r882 costs. Its baseline differs from the full-pool baseline. Exact native 91-context utilities (mean 1.1377941898164525), cached/native costs and prior native cost bytes pass before TRAIN outcome decoding. DOPE c51a56d0 and the entire model state remain byte-identical; no parameter can train, optimizer construction/backward raise, and the private r882 copy remains read-only. Original H50/H58 derivative S3 versions are not established; exact numeric closure has versioned H144 recovery custody and current input/pool/asset hashes are verified.

STRICT requires B equal-family gain>.001+1e-12 and a paired 1,000-family-resample two-sided 95% LCB>0; ties reject. H168_NEW removes only the LCB condition. **H175 tests one prespecified counterfactual candidate against baseline, not a best-of-four snapshot search.** This is an optimistic sparse-oracle sensitivity audit, not calibrated power for every learned, mixed-sign alternative. Reported points are medians of three seed statistics; 10k shared-family CIs condition on fixed folds/policies/decisions. Admission also has 10k five-fold-block intervals in metrics.json; fixed-family intervals do not imply independent folds or new-fold calibration. Elo uses family-weighted context-pairwise wins, tie margin 1e-4 and the exact two-policy H148 BT solution with .01 pseudo-wins, r882=1500. It describes randomized counterfactual-policy wins, not mean-utility comparison or deployable model quality.

## Outcome: available oracle headroom

| Slate / cohort | Family headroom median [95% CI] | Oracle-optimal contexts by seed | -2 picks / avoidable by oracle | Oracle Elo [95% CI] |
| --- | ---: | --- | --- | ---: |
| 24 / TRAIN | +0.320167 [0.256873, 0.393324] | [142, 167, 166] | [18, 22, 23] / [18, 22, 23] | 1958.53 [1933.43, 1988.13] |
| 24 / DLVal | +0.221494 [0.118495, 0.352190] | [16, 13, 18] | [3, 1, 1] / [3, 1, 1] | 1913.59 [1835.34, 2000.48] |
| full / TRAIN | +0.467187 [0.360408, 0.596798] | [4, 4, 4] | [17, 17, 17] / [17, 17, 17] | 2601.89 [2471.75, 2913.41] |
| full / DLVal | +0.250985 [0.125016, 0.414955] | [1, 1, 1] | [0, 0, 0] / [0, 0, 0] | 2394.12 [2212.28, 2811.59] |

These are equal-family gains. Oracle-optimal includes tied true utility. Full-pool baseline/oracle panels are identical across seeds, not independent model replications. Context and family gap quantiles, per-seed utilities and intervals are retained in metrics.json. DL-Val is descriptive headroom only; it never calibrates power or selection. Corpus-level headroom does not establish that the frozen representation/adapter can learn the oracle.

All 17 full-pool TRAIN -2 picks are actual gate failures and all 17 have better valid alternatives. The same 17 IDs, native choices and plain costs match H159's saved native records. H159's **0/17** was avoidability by its tested fail-logit **veto grid**, not by an unrestricted true-utility oracle: its own records contain valid alternatives for 17/17, and a distinct any-threshold check avoids 16/17. This scope clarification is descriptive and does not alter the prespecified H175 verdict.

## Grid MDE80 and selector power

| Slate | Activation | Rule | Grid MDE80 utility [95% CI] | Headroom % [95% CI] | First qualifying f |
| --- | --- | --- | ---: | ---: | ---: |
| 24 | context | STRICT | +0.015834 [0.012813, 0.019535] | 4.946 [4.720, 5.290] | 5% |
| 24 | context | H168_NEW | +0.015834 [0.012813, 0.019535] | 4.946 [4.720, 5.290] | 5% |
| 24 | family | STRICT | +0.063957 [0.050670, 0.078891] | 19.976 [19.154, 20.607] | 20% |
| 24 | family | H168_NEW | +0.016364 [0.012822, 0.020966] | 5.111 [4.754, 5.527] | 5% |
| full | context | STRICT | +0.022802 [0.017640, 0.029257] | 4.881 [4.742, 5.062] | 5% |
| full | context | H168_NEW | +0.009355 [0.007143, 0.011880] | 2.002 [1.904, 2.075] | 2% |
| full | family | STRICT | +0.094678 [0.072169, 0.121619] | 20.266 [19.702, 20.726] | 20% |
| full | family | H168_NEW | +0.024480 [0.018267, 0.032459] | 5.240 [4.856, 5.637] | 5% |

MDE80 is the smallest tested f with median admission>=.8; it is a discrete-grid endpoint, not a continuously estimated threshold. For full/context STRICT, power is .657 at 2% (gain +.009355) and .996 at 5% (gain +.022802). The family-CI uncertainty at the detected grid point is separate from this grid resolution. No grid crossings were right censored in this run; the implementation and controls handle censoring explicitly.

Full-pool context activation:

| Oracle activation f | Injected gain median [95% CI] | STRICT admission [95% CI] | NEW admission [95% CI] |
| --- | ---: | ---: | ---: |
| 0% | +0.000000 [0.000000, 0.000000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 0.5% | +0.002201 [0.001723, 0.002837] | 0.023 [0.022, 0.024] | 0.323 [0.318, 0.328] |
| 1% | +0.004461 [0.003420, 0.005856] | 0.176 [0.173, 0.178] | 0.581 [0.573, 0.589] |
| 2% | +0.009355 [0.007143, 0.011880] | 0.657 [0.650, 0.660] | 0.874 [0.871, 0.877] |
| 5% | +0.022802 [0.017640, 0.029257] | 0.996 [0.995, 0.997] | 0.999 [0.999, 0.999] |
| 10% | +0.045948 [0.035469, 0.058722] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| 20% | +0.093858 [0.071959, 0.119998] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |

Full-pool family activation:

| Oracle activation f | Injected gain median [95% CI] | STRICT admission [95% CI] | NEW admission [95% CI] |
| --- | ---: | ---: | ---: |
| 0% | +0.000000 [0.000000, 0.000000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 0.5% | +0.002228 [0.001643, 0.002907] | 0.000 [0.000, 0.000] | 0.155 [0.154, 0.156] |
| 1% | +0.005057 [0.003566, 0.006522] | 0.000 [0.000, 0.000] | 0.309 [0.308, 0.310] |
| 2% | +0.009826 [0.007108, 0.012742] | 0.008 [0.007, 0.009] | 0.513 [0.511, 0.515] |
| 5% | +0.024480 [0.018267, 0.032459] | 0.150 [0.146, 0.154] | 0.833 [0.831, 0.835] |
| 10% | +0.048789 [0.036865, 0.063317] | 0.609 [0.606, 0.612] | 0.975 [0.974, 0.976] |
| 20% | +0.094678 [0.072169, 0.121619] | 0.976 [0.975, 0.977] | 1.000 [1.000, 1.000] |

All fraction curves, family panels and locks are in S3. Compact Git metrics include f=0/2/5/20 for both slates, activation variants, replacement types and selectors, plus all eight MDEs. Oracle injections are nonnegative and cannot stand in for the signed errors and correlations of learned policies. Family-clustered gains require substantially larger effects in this specified experiment; a blanket claim that all prior learned alternatives have this exact power is unsupported.

## Harmful mirror and false admission

Random non-oracle replacement can improve a poor r882 choice. Separate total admission of the mirror bank from false admission conditional on its known finite-TRAIN equal-family gain<=0. For full/context f=2%, H168_NEW total admission is **0.094 [0.092, 0.096]**, with fold-block CI [0.084, 0.109]; STRICT is 0.002. Nonpositive-population draws by seed are [194, 198, 193] out of 200. Conditional false admission is **8.394%** NEW versus **0.202%** STRICT; the full matched conditional CIs/denominators are in metrics.json. NEW selected OOF utility is -0.001161 [-0.001504, -0.000751], with counterfactual Elo 1499.766 [1499.695, 1499.812]. The 0.094 total rate and 0.08394 conditional false rate have different denominators; do not conflate them. The f=0 control has 0 admission for both selectors by construction, which does not establish calibrated noisy false-positive control.

## Historical comparison and verdict

| Wave | Recorded inner-B maximum | Records / type | Eligibility for historical threshold |
| --- | ---: | --- | --- |
| H160 | unavailable | 0 / NO_REPLAYABLE_INNER_B_GAINS | excluded; PREREGISTERED_NOT_EXECUTED |
| H161 | +0.028484444 | 4 / arm_max_summary | included; COMPLETE |
| H162 | +0.030191000 | 60 / cell_best | included; COMPLETE |
| H163 | unavailable | 0 / NO_REPLAYABLE_INNER_B_GAINS | excluded; COMPLETE |
| H164 | unavailable | 0 / NO_REPLAYABLE_INNER_B_GAINS | excluded; PREREGISTERED_NOT_EXECUTED |
| H165 | +0.035818420 | 60 / cell_best | included; COMPLETE |
| H166 | +0.025117978 | 60 / cell_best | included; COMPLETE |
| H167 | +0.017286333 | 60 / cell_best | included; COMPLETE |
| H168 | unavailable | 0 / NO_REPLAYABLE_INNER_B_GAINS | excluded; COMPLETE |
| H169 | unavailable | 0 / NO_REPLAYABLE_INNER_B_GAINS | excluded; COMPLETE_DIAGNOSTIC_LEAK_BIASED |
| H170 | +0.027813314 | 60 / cell_best | included; COMPLETE |
| H171 | unavailable | 0 / UNAVAILABLE_AT_PREREG | excluded; UNAVAILABLE_AT_PREREG |
| H172 | unavailable | 0 / UNAVAILABLE_AT_PREREG | excluded; UNAVAILABLE_AT_PREREG |
| H173 | unavailable | 0 / UNAVAILABLE_AT_PREREG | excluded; UNAVAILABLE_AT_PREREG |
| H174 | unavailable | 0 / UNAVAILABLE_AT_PREREG | excluded; UNAVAILABLE_AT_PREREG |

H171/H172/H173/H174 metrics were unavailable at the frozen initial source-base snapshot and are not imputed zero. H163/H168 retained no usable B gains in the pinned publication schemas; their outer OOF/Val results are not substituted. H160/H164 were not executed in those snapshots. Known-leaky H169 and explicit incomplete records are excluded. H161 contributes arm-max summaries, distinguished from individual cell records. Independent replay verifies 304 explicit B JSON paths. The median of six eligible per-study maxima is **+.028148879**; these in-split maxima are optimistic descriptive comparators, not validated effects.

The prespecified full/context STRICT grid MDE +.022802 is below that historical median, so the registered verdict is **MIXED**, with substantial oracle headroom. Full/family STRICT MDE +.094678 is much larger. H175 supports a family-sparsity/power concern while separating it from finite-candidate headroom; it does not identify a trainable oracle, universal representation ceiling, or a safe reason to remove the LCB.

## Next step: H177 plus the H178 calibration gate

Per owner ruling DECISIONS_20261029.md 11:28Z and closeout order t0440, retain margin .001, LCB95>0 and rejected ties. H168_NEW is rejected for adoption; the live flywheel rule remains unchanged. **H177** tests pooled cross-fit selection across 205 families (each selection-fold once), one prespecified candidate step 40 per arm, seeds averaged within family, 10k paired family bootstrap/Holm and A-family confirmation. Pin this merged closeout as its anchor.

Use FAMILY-clustered MDEs in planning: 24/full STRICT +.063957/+.094678, whose divided-by-sqrt5 values are **+.028602/+.042341**. Those are planning heuristics, not measured pooled-rule power. **H178**, zero updates on the same counterfactual bank, must replay the exact H177 rule and report context/family MDE80 and Clopper-Pearson upper bounds; admission for f=0 and every harmful-mirror cell must be <= .05. Any H177 admit is lift only if that calibration gate passes; otherwise it is diagnostic. If pooled family MDE remains > .03, use more families/data or a prospectively registered group-sequential alpha-spending design, without lowering the margin. H175 does not execute either next wave.

## Verification and custody

Independent replay verifies 91 baseline cost vectors, 1,509,119 true candidate labels, 56 counterfactual cases, 840 B locks, all 168,000 scalar bootstrap CIs/decisions per rule, 10k family summaries/Elo/MDEs, harmful denominators and historical criterion. Whole model hash before/after equals private r882 **6eb0e998896bd87553bbea815be06b534d5f3f1ae2a63cbf7ff4ec22e89c4f8a**; DOPE c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814; frozen non-adapter 8d1070faf574ee1cd2a4e630887e1662b43189b3f89b04982143ef2704818aae. Underlying 276 dataset IDs join the actual pinned versioned S3 origin=real catalog. No protected/refused reads, optimizer/backward, GPU, serving or live-loop writes.

Scientific source 9cd28fc563d92e9696178957e387f8cb8d31bb0e (PR #1490), REDTEAM PASS 5727712619 at 1cb289d545912561f09ac332a0e8344948ac3e67, exact-tree carry to 704076e20dfca22c10e0000e0b9a796c422bec93 and author review 5727738272 precede the sole attempt. Initial historical extractor BLOCK 5727655362 was repaired prospectively with explicit H162/H166 schemas/poisoned outer-panel tests; numerical verdict thresholds and frozen history/input bytes did not change. 18 data-free controls, guarded 0-model/0-forward smoke, Ruff/format/diff and all five records gates pass. CPU xbabe2/nice19/math1 measured peak 2 aggregate threads, 3.264 GB RSS, host CPU 32.2%, raw load 30.43; minimum MemAvailable 68.48 GiB/free disk 563.80 GB. No full hosted/native workspace qualification or speed metric is claimed.

S3 manifest: s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H175/codex-t0330/005b086b35ac1d4c/manifest.json (VersionId cGaJmNsjCXYikZkjK7r4m3MCd04WE.Kt, SHA256 b182577b8a6c480c469407c55b312b3aa401c6e2b082c65efc6bedd8baf41c01). Full raw metrics, simulation metrics, NPZ panels/locks, source/inputs/assets and audits stay in S3 custody. Actual versioned download, ZIP decode and all **1,018** member hashes pass, with disk restore probes. Small Git summary stays < 100 KB; no NPZ or source bundle is committed.
