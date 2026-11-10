# H262 — Strict acceptance replay

**COMPLETE / NO_ADVANCE.** Primary: **NO_ADVANCE**; secondary: **NO_ADVANCE**. Finding: **no qualifying acceptance-rule or checkpoint lift**.

## Hypothesis and setup

A paired family-bootstrap acceptance margin may protect the incumbent against noisy admissions. CURRENT ports the observed live adapter comparator (context mean candidate utility >= incumbent, ties accepted); STRICT_LCB requires the 2,000-resample paired equal-family CI95 lower endpoint >+0.002. Both retain admitted adapter and AdamW state and restore both on rejection. NEVER remains r882. Threshold, uncertainty treatment and weighting change together, so this is a comparison of rule packages.

[Preregistration](https://github.com/neverhuman/JopeDime/commit/533530e21c5b141be8b4e460c7e9a2bcd45867b8) published at 2026-11-03T16:06:49.929959+00:00, before Native91/smoke/training. [Frozen protocol](PREREG.md), [live-source receipt](current_rule_source.json), [family split](fold_manifest.json), [schedule](schedules.json). CURRENT source was read at deployed commit 486b709c3ee84a351a2cd7b55ef0290017e6ecd0; the live process omitted the optional strict-margin flags. The experiment moved acceptance to FIT-accept and did not modify the live loop. Program-pool promotions and adapter admissions are distinct; promotion frequency does not establish adapter learning.

886 FIT-train contexts / 143 families and 333 FIT-accept contexts / 62 families, partition seed 262140. Seeds 262101–262103 each ran ten rounds per rule, forty exact H170 AdamW updates per candidate. **2,400 primary updates + 2 smoke updates**, with one numeric CPU thread and frozen DOPE/anchor. Identical random/sampler streams were used at each round index; realized candidates can differ after incumbent trajectories diverge. This is fixed-pool offline continuation, not live GP generation.

## Final DL-Val91 single-pick outcome

| Contrast | Per-seed family gains | Median [CI95] | Holm lower bound | Holm p | Positive seeds | Verdict |
|---|---|---|---|---|---|---|
| STRICT_LCB_minus_r882 | +0.00000000 / +0.00000000 / +0.00000000 | +0.00000000 [0.0, 0.0] | 0.0 | 1.0 | 0/3 | NO_ADVANCE |
| STRICT_LCB_minus_CURRENT | +0.01740799 / +0.01912082 / +0.00000000 | +0.01740799 [0.0, 0.044587230007909345] | -0.00014680453627310017 | 0.21257874212578742 | 2/3 | NO_ADVANCE |
| CURRENT_minus_r882 | -0.01740799 / -0.01912082 / +0.00000000 | -0.01740799 [-0.044587230007909345, 0.0] | descriptive | descriptive | 0/3 | descriptive |

The primary gate requires median family gain >+0.002, the preregistered step-down Holm lower endpoint >0, Holm p<.05 and 3/3 positive seeds. A passing secondary alone is an acceptance-rule finding, not checkpoint lift. Family intervals condition on these final policies and an exposed development panel. Three seeds share the partition; within-wave Holm correction does not correct campaign-wide adaptive search.

| Arm | Context utility by seed | Family utility by seed | Median Elo [CI95] |
|---|---|---|---|
| NEVER | +1.13779419 / +1.13779419 / +1.13779419 | +1.15361528 / +1.15361528 / +1.15361528 | 1500.000 [1500.0, 1500.0] |
| CURRENT | +1.11962101 / +1.11783289 / +1.13779419 | +1.13620729 / +1.13449445 / +1.15361528 | 1489.036 [1463.3307376225382, 1507.3078195706648] |
| STRICT_LCB | +1.13779419 / +1.13779419 / +1.13779419 | +1.15361528 / +1.15361528 / +1.15361528 | 1500.000 [1500.0, 1500.0] |

STRICT_LCB rejected all 30 candidates and retained r882 exactly in every seed. Its zero gain and interval therefore reflect baseline identity, not evidence of policy equivalence or checkpoint improvement. Against CURRENT, the median gain +0.01740799 has ordinary CI95 [0, +0.04458723], Holm lower bound −0.00014680, Holm p 0.21258 and only 2/3 positive seeds: it does not qualify as an acceptance-rule finding under the frozen gates.

## Admissions, drift and harms

| Seed / rule | Accepted rounds of 10 | Final L2 drift from r882 |
|---|---|---|
| 262101-CURRENT | 1 | 0.36864666 |
| 262101-STRICT_LCB | 0 | 0.00000000 |
| 262102-CURRENT | 1 | 0.37589639 |
| 262102-STRICT_LCB | 0 | 0.00000000 |
| 262103-CURRENT | 0 | 0.00000000 |
| 262103-STRICT_LCB | 0 | 0.00000000 |

| Rule | Trajectories with accepts | Final utility worse than r882 | Accept-then-final-worse rate |
|---|---|---|---|
| CURRENT | 2 | 2 | 1.0 |
| STRICT_LCB | 0 | 0 | undefined (zero eligible trajectories) |

CURRENT accepted seed 262101 round 8 and seed 262102 round 3, with FIT-accept context-mean gains +0.0060748736 and +0.0027040807. **Both were strict improvements, not ties.** Both admitted trajectories finished worse on DL-Val, but this small descriptive observation does not establish that tie acceptance caused the harms or that the stricter package generalizes. The third CURRENT trajectory and all STRICT_LCB trajectories retained r882.

These rates use final equal-family utility and condition on trajectories with at least one acceptance. Per-round/per-accept DL-Val harm is **NOT_MEASURED**: measuring it would violate the final-only scoring rule. A null rate means zero eligible trajectories. All 60 rounds locked before any trained DL-Val evaluation; six final trained incumbents were each scored once. NEVER reuses Native91. Raw final costs also produce TOP4_PICK without another model call.

| Contrast | Harmed-family counts by seed (delta <−1e-4) | Family W/T/L by seed |
|---|---|---|
| STRICT_LCB_minus_r882 | [0, 0, 0] | [[0, 19, 0], [0, 19, 0], [0, 19, 0]] |
| STRICT_LCB_minus_CURRENT | [4, 3, 0] | [[6, 9, 4], [6, 10, 3], [0, 19, 0]] |
| CURRENT_minus_r882 | [6, 6, 0] | [[4, 9, 6], [3, 10, 6], [0, 19, 0]] |

All harmed-family identities remain in [metrics.json](metrics.json). No contexts or families were removed after outcomes. Shared r882/fallback utilities, if present, are identical policies rather than independent replications or equivalence evidence.

## TOP4_PICK diagnostic

Baseline TOP4_PICK context mean: **1.240360168225**, reproducing H258’s rounded 1.240360. Exact H258 gated definition: predicted top-gap/std<.25 enables four true evaluations, otherwise zero reveals/top1; select the minimum raw gated cost among revealed top4, stable prediction-order ties. Each final supplies its own outcome-free gate. This is test-time search, not checkpoint lift; gate incidence and therefore reveal counts may differ.

| Arm | Context utility by seed | Family utility by seed | Median Elo [CI95] |
|---|---|---|---|
| NEVER | +1.24036017 / +1.24036017 / +1.24036017 | +1.25199972 / +1.25199972 / +1.25199972 | 1500.000 [1500.0, 1500.0] |
| CURRENT | +1.22912891 / +1.22983505 / +1.24036017 | +1.24124136 / +1.24191776 / +1.25199972 | 1452.207 [1429.6396049409432, 1485.3778889950893] |
| STRICT_LCB | +1.24036017 / +1.24036017 / +1.24036017 | +1.25199972 / +1.25199972 / +1.25199972 | 1500.000 [1500.0, 1500.0] |

| TOP4 contrast | Per-seed family gains | Median [CI95] |
|---|---|---|
| STRICT_LCB_minus_r882 | +0.00000000 / +0.00000000 / +0.00000000 | +0.00000000 [0.0, 0.0] |
| STRICT_LCB_minus_CURRENT | +0.01075836 / +0.01008195 / +0.00000000 | +0.01008195 [0.0021726763227135804, 0.020448036031804943] |
| CURRENT_minus_r882 | -0.01075836 / -0.01008195 / +0.00000000 | -0.01008195 [-0.020448036031804943, -0.0021726763227135812] |

TOP4 panels are descriptive only and supply no additional ADVANCE opportunity. Their Elo anchor is TOP4_PICK@r882; single-pick Elo above is anchored at single-pick r882. Historical r3175 qualification is not changed by this experiment.

## Verdict and next step

**NO_ADVANCE: no qualifying acceptance-rule or checkpoint lift.** Retain this fixed comparison. Any follow-up requires a new preregistration; no post-outcome tuning, process retries or serving promotion occurred. Repeated FIT-accept use can itself overfit, and this replay does not qualify a production acceptance mechanism.

## Verification, resources and custody

Native91 exact 91/91 (mean 1.1377941898164525), two changed-parameter smoke updates with exact reset, frozen DOPE c51a56d0 before/after. Terminal audit replayed 20,950 choice/label checks across 68 cost files, 66 candidate/final snapshots, all 60 decisions and optimizer-step/rollback traces, paired sampler hashes, final locks and both final statistical panels. 1,745 resource samples replayed. This uses the same frozen statistical implementation; it is not an independent estimator or independent-agent review.

CPU 48,50–52 / nice 19 / one numeric thread; peak sampled aggregate RSS 3,333,926,912 B and 3 threads; elapsed 1004.0 s; pause/resume transitions 0. A1/load/RSS/disk/deadline guards retained, no process restart/deadline reset, zero protected reads. No AUC or speed verdict.

Runtime `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H262/astra-t1558/closeout/runtime.tar.gz`, VersionId `FLwhbQpUAXjTYkqUlRcgGHJ.jVx0QTit`, SHA256 `101cd2b805d93130f92f43d7839eb8c53fa202c16125a41028572b8a3ca10d01`.

Member manifest `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H262/astra-t1558/closeout/manifest.json`, VersionId `JQF0JzGi4oHnBGutT3r.1G9.gYxs8FLM`, SHA256 `fa4f19b98b29ac2094759bcaabd4299970fffacc682bbc8665ebbe399dabefac`. Exact-version GET and every member hash verified before bulk reclamation. Source/data/code closure, raw costs, candidate/final adapter snapshots, decisions and resource receipts are preserved there; no large objects entered Git.

13 targeted tests, Ruff, JSON/schema, source/index pins, 13 inherited-definition checks and same-agent exact-head review passed. Full hosted CI **NOT_RUN** under the owner’s PR2071 split; no global schemas/records pins/root README/CI configuration changes, main merge, fastlane push or live-loop modification.
