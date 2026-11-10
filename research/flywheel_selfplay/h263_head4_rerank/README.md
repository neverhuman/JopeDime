# H263 — trust-region head-4 reranking

**COMPLETE / NO_ADVANCE.** HEAD4 primary: **NO_ADVANCE**. HEAD4_NOTR secondary: **NO_ADVANCE**.

All 60/60 strict selections retained r882. Both head-loss arms have selected median gain 0, 95% CI[0,0], Holm LCB 0, Holm p 1 and 0/3 positive seeds: **fallback identity, not equivalence**. Unselected step 40 HEAD4 gain is -0.00257490; HEAD4_NOTR is -0.02151701. The anchor sharply limits drift, but neither head-loss continuation improves the primary comparison.

## Hypothesis and setup

FIT-only pairwise ordering inside frozen r882’s predicted top4 might improve single-pick utility, while H142 anchoring protects against drift. CONTROL uses exact H170 AdamW/loss/batch; L2SP adds the exact beta 50 quadratic anchor and gamma 0.25 post-step retraction; HEAD4 adds 0.5 pairwise logistic to L2SP; HEAD4_NOTR adds the same head loss to CONTROL. All start at identical r882, adapter only, frozen DOPE fp32. The anchor is soft and supplies no hard drift guarantee. As in H142, its all-parameter penalty creates gradients even for task-inactive tensors; AdamW may then decay tensors that CONTROL leaves with grad=None. The anchor comparison includes that optimizer interaction.

[Preregistration commit](https://github.com/neverhuman/JopeDime/commit/c10815a2d8ef214ebb86ef3422282cb67f1041a0) published 2026-11-03T16:41:21.383232+00:00 before all study forwards, pool reads and smoke. [Frozen protocol](PREREG.md), [manifest](manifest.json), [family split](fold_manifest.json), [schedule](schedules.json), [metrics](metrics.json), [local schema](metrics.schema.json). Real DL-Train: 1,219 contexts / 205 families, five folds, seeds 263101–263103, 40 updates per arm/cell: **2,400 primary + 4 smoke updates**. Exact H257 strict SELECT rule,123 FIT / 41 SELECT / 41 ASSESS families; all 15 cell selections fsync-locked globally before trained ASSESS/DL-Val.

Head4 IDs are fixed once from native r882 scores. Six pairs/context; gaps<1e-4 and invalid/invalid pairs dropped; invalid sentinel utility−2 ranks worst. Each head-loss forward retains the full native pool and fixed first-four support. Primary makes one model argmin and adds no oracle reveal/reselection; inherited Native91 support labels remain inputs. This is not a zero-label-conditioning protocol.

Five independently selected cell **utilities** are averaged per seed before equal-family comparison. This is not a prediction ensemble or one deployable checkpoint. Three seeds share one fold assignment. Development-panel intervals and within-wave Holm correction do not correct campaign-wide adaptive search.

## DL-Val91 single-pick outcome

| Contrast | Per-seed family gains | Median [95% CI] | Holm lower 95% bound | Holm p | Positive seeds | Verdict |
|---|---|---|---|---|---|---|
| HEAD4_minus_r882 | +0.00000000 / +0.00000000 / +0.00000000 | +0.00000000 [0.0, 0.0] | 0.0 | 1.0 | 0/3 | NO_ADVANCE |
| HEAD4_NOTR_minus_r882 | +0.00000000 / +0.00000000 / +0.00000000 | +0.00000000 [0.0, 0.0] | 0.0 | 1.0 | 0/3 | NO_ADVANCE |
| CONTROL_minus_r882 | +0.00000000 / +0.00000000 / +0.00000000 | +0.00000000 [0.0, 0.0] | descriptive | descriptive | 0/3 | descriptive |
| L2SP_minus_r882 | +0.00000000 / +0.00000000 / +0.00000000 | +0.00000000 [0.0, 0.0] | descriptive | descriptive | 0/3 | descriptive |
| HEAD4_minus_L2SP | +0.00000000 / +0.00000000 / +0.00000000 | +0.00000000 [0.0, 0.0] | descriptive | descriptive | 0/3 | descriptive |

ADVANCE requires median family gain>+.002, Holm LCB95>0, Holm p<.05 and 3/3 positive seeds for the two head-loss contrasts. CONTROL/L2SP and HEAD4−L2SP are descriptive. Step 0 fallback is **fallback identity, not equivalence**; identical policy outputs are not independent replications.

| Arm | Context utility by seed | Family utility by seed | Median MM-BT Elo [95% CI] |
|---|---|---|---|
| r882 | +1.13779419 / +1.13779419 / +1.13779419 | +1.15361528 / +1.15361528 / +1.15361528 | 1500.000 [1500.0, 1500.0] |
| CONTROL | +1.13779419 / +1.13779419 / +1.13779419 | +1.15361528 / +1.15361528 / +1.15361528 | 1500.000 [1500.0, 1500.0] |
| L2SP | +1.13779419 / +1.13779419 / +1.13779419 | +1.15361528 / +1.15361528 / +1.15361528 | 1500.000 [1500.0, 1500.0] |
| HEAD4 | +1.13779419 / +1.13779419 / +1.13779419 | +1.15361528 / +1.15361528 / +1.15361528 | 1500.000 [1500.0, 1500.0] |
| HEAD4_NOTR | +1.13779419 / +1.13779419 / +1.13779419 | +1.15361528 / +1.15361528 / +1.15361528 | 1500.000 [1500.0, 1500.0] |

Single-pick r882 is anchored at 1500. Exact Native91: 91/91, context mean 1.1377941898164525.

## Selection, unselected step 40 and drift

| Arm | Nonzero selected cells /15 | Step 40 family gain by seed | Step 40 median [95% CI] | Step 40 L2 drift min / median / max |
|---|---|---|---|---|
| CONTROL | 0 | -0.03175896 / -0.02768578 / -0.03397467 | -0.03175896 [-0.0644373925999305, -0.004512597379911988] | 0.34563006 / 0.37729733 / 0.39362024 |
| L2SP | 0 | -0.00321863 / -0.00321863 / -0.00202401 | -0.00321863 [-0.009423801626617115, 0.0] | 0.00225168 / 0.00271342 / 0.00334890 |
| HEAD4 | 0 | -0.00257490 / -0.00321863 / -0.00202401 | -0.00257490 [-0.007539041301293692, 0.0] | 0.00242060 / 0.00284683 / 0.00347067 |
| HEAD4_NOTR | 0 | -0.02151701 / -0.02000742 / -0.02314762 | -0.02151701 [-0.049737571689683936, -0.0016035742691184972] | 0.32044800 / 0.35595549 / 0.37173508 |

All 60 selection decisions, all 240 candidate snapshots, sampler traces and raw SELECT/ASSESS/DL-Val costs are in versioned custody. Metrics retain every decision, update 40 drift, TRAIN OOF summaries and DL-Val harmed-family identities; full TRAIN harm identities remain in the archived runtime metrics. Step 40 is unselected and diagnostic only.

## Head ordering, harms and TOP4_PICK

Frozen TRAIN head eligibility histogram (0–6 usable pairs/context): `{'0': 8, '1': 0, '2': 0, '3': 37, '4': 6, '5': 222, '6': 946}`. Per-cell FIT exposure counts are in metrics. These verify eligible labels, not a separately measured auxiliary-gradient norm.

| Arm | Top1-within-frozen-head accuracy, context by seed | Equal-family accuracy by seed | Full-pool pick in frozen head by seed |
|---|---|---|---|
| r882 | +0.23076923 / +0.23076923 / +0.23076923 | +0.22105263 / +0.22105263 / +0.22105263 | +1.00000000 / +1.00000000 / +1.00000000 |
| CONTROL | +0.23076923 / +0.23076923 / +0.23076923 | +0.22105263 / +0.22105263 / +0.22105263 | +1.00000000 / +1.00000000 / +1.00000000 |
| L2SP | +0.23076923 / +0.23076923 / +0.23076923 | +0.22105263 / +0.22105263 / +0.22105263 | +1.00000000 / +1.00000000 / +1.00000000 |
| HEAD4 | +0.23076923 / +0.23076923 / +0.23076923 | +0.22105263 / +0.22105263 / +0.22105263 | +1.00000000 / +1.00000000 / +1.00000000 |
| HEAD4_NOTR | +0.23076923 / +0.23076923 / +0.23076923 | +0.22105263 / +0.22105263 / +0.22105263 | +1.00000000 / +1.00000000 / +1.00000000 |

| Arm | Step 40 head accuracy, context by seed | Step 40 equal-family head accuracy | Step 40 full-pool pick membership |
|---|---|---|---|
| r882 | +0.23076923 / +0.23076923 / +0.23076923 | +0.22105263 / +0.22105263 / +0.22105263 | +1.00000000 / +1.00000000 / +1.00000000 |
| CONTROL | +0.20879121 / +0.20659341 / +0.21098901 | +0.20000000 / +0.19789474 / +0.20210526 | +0.95824176 / +0.96483516 / +0.95384615 |
| L2SP | +0.20879121 / +0.20879121 / +0.21318681 | +0.20000000 / +0.20000000 / +0.20421053 | +1.00000000 / +1.00000000 / +1.00000000 |
| HEAD4 | +0.21318681 / +0.20879121 / +0.21318681 | +0.20421053 / +0.20000000 / +0.20421053 | +1.00000000 / +1.00000000 / +1.00000000 |
| HEAD4_NOTR | +0.21098901 / +0.20659341 / +0.20439560 | +0.20210526 / +0.19789474 / +0.19578947 | +0.96703297 / +0.96703297 / +0.95604396 |

Step 40 head metrics above are reconstructed from the already-scored raw assessment records. They add no forward or decision gate. Step 40 TOP4 context/family utilities are also retained in the derived diagnostic metrics.

Accuracy selects the current lowest prediction within the frozen r882 four, compares with the best valid raw cost within 1e-4, and accepts all-invalid ties. All-invalid frozen heads: **0/91**. Membership distinguishes ordering from moving the full-pool pick outside the frozen head.

| Contrast | Harmed-family counts by seed (delta<−1e-4) | Family W/T/L by seed |
|---|---|---|
| HEAD4_minus_r882 | [0, 0, 0] | [[0, 19, 0], [0, 19, 0], [0, 19, 0]] |
| HEAD4_NOTR_minus_r882 | [0, 0, 0] | [[0, 19, 0], [0, 19, 0], [0, 19, 0]] |
| CONTROL_minus_r882 | [0, 0, 0] | [[0, 19, 0], [0, 19, 0], [0, 19, 0]] |
| L2SP_minus_r882 | [0, 0, 0] | [[0, 19, 0], [0, 19, 0], [0, 19, 0]] |
| HEAD4_minus_L2SP | [0, 0, 0] | [[0, 19, 0], [0, 19, 0], [0, 19, 0]] |

Selected policies have zero utility−2 events in every seed and fold. Context 27 scores 1.1226949552792034 for every selected and step-40 policy; it remains included in all statistics. Exact counts and values remain in metrics.

| Arm | Exact H258 TOP4_PICK context utility by seed | Family utility by seed |
|---|---|---|
| r882 | +1.24036017 / +1.24036017 / +1.24036017 | +1.25199972 / +1.25199972 / +1.25199972 |
| CONTROL | +1.24036017 / +1.24036017 / +1.24036017 | +1.25199972 / +1.25199972 / +1.25199972 |
| L2SP | +1.24036017 / +1.24036017 / +1.24036017 | +1.25199972 / +1.25199972 / +1.25199972 |
| HEAD4 | +1.24036017 / +1.24036017 / +1.24036017 | +1.25199972 / +1.25199972 / +1.25199972 |
| HEAD4_NOTR | +1.24036017 / +1.24036017 / +1.24036017 | +1.25199972 / +1.25199972 / +1.25199972 |

TOP4_PICK uses each final’s float32 cost gap/std<.25 gate, stable top4 and minimum revealed gated raw cost (invalid 2); otherwise top1. Native baseline reproduces H258’s rounded 1.240360 context mean. This extra-label test-time search is descriptive only, supplies no ADVANCE gate, and is not checkpoint lift. Any reported TOP4 Elo anchors TOP4_PICK@r882, separate from the primary single-pick anchor.

At step 40, HEAD4 and L2SP keep their full-pool picks inside the frozen head for every scored context, yet their within-head accuracy remains below r882. Thus the anchored treatment did not convert the available head ordering opportunity into improved DL-Val choices. HEAD4−L2SP has median gain 0 and CI[0,0] with only 1/3 positive seeds; this degenerate median interval is not evidence that every seed or checkpoint is identical. The exact per-seed gains remain in metrics.

## Cross-critique and limits

H260’s [closeout](https://github.com/neverhuman/JopeDime/blob/6bebb53fca23b9b0355cb1928a6de479e8019cd6/research/flywheel_selfplay/h260_top4_recall/README.md) found no full-pool recall/TOP4 improvement from its minibatch recall objectives. Lower drift and less single-pick damage do not by themselves isolate a braking mechanism or establish absence of ranking signal; H263 includes an anchor-only mechanism control. TOP4 oracle gain demonstrates available selection headroom, not that most errors are ordering errors or that the ordering is learnable. Low exact-oracle recall and useful head utility can coexist. Different seeds/folds/targets prevent a paired superiority claim against H260. This critique was recorded after H263 publication, with no protocol change.

## Verdict and next step

**NO_ADVANCE** under the frozen gate. Retain the measured contrast and its scope; no serving promotion, post-outcome tuning or retry. Any follow-up needs a new preregistration. A selected baseline does not establish equivalence of trained models; the update 40 and mechanism-control results distinguish those behaviors.

## Validation and custody

11 targeted tests/Ruff/JSON/schema/link/source/index checks and 16 inherited-definition checks PASS; same-agent exact-head review. Native91 exact and four changed-parameter smoke updates with exact reset; DOPE c51a56d0 unchanged. Terminal saved-evidence replay verified 94,538 choices/labels, 317 cost files, 240 adapter snapshots, all 60 selectors/sampler traces, global locks, head-order/TOP4 diagnostics and five statistical panels; 7,352 resource samples. Statistics use the same frozen estimator; this is not an independent training replay or independent-agent review.

CPU48,50-52/nice19/numeric threads 1; peak sampled aggregate RSS 3,358,380,032B, threads 3, elapsed 4296.0s; pause/resume transitions 0. H239-A1/load/RSS/disk/deadline guards retained; no restart, deadline reset, GPU or protected reads. No speed/AUC verdict.

Runtime `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H263/astra-t1635/closeout/runtime.tar.gz`, VersionId `8_w1KijijUGAP_Pl5e8V7LwqXlAIGcW4`, SHA256 `a6701b6f9f7c4770269ee04a92b7801521dcfa12458c57d603b8fd0bd8a505d5`. Member manifest `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H263/astra-t1635/closeout/manifest.json`, VersionId `S8T6EVFZez13tcLNYAx0v1dGU.6i_A7c`. Exact-version GET and every member hash verified; no .pt/.npz in Git.

Full hosted CI **NOT_RUN** under CI_FREEZE split. Published only study files and own index rows to v1/next. No main/fastlane/PR2071/global schemas/records/rootREADME/CI/live-loop changes.
