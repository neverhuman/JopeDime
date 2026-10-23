# Research audit and next wave — 2026-10-23

No established Dope-state or GP-landmark ranking lift has been demonstrated by the current H27 study. The next wave should test a strong tree selector and more qualified task breadth before making a longer neural run the default. This summary preserves earlier positive evidence, nulls, failures and exposure limits; it authorizes no execution or promotion.

## What earlier research supports

The audit covers all 18 records in `docs/experiments/experiments.jsonl` plus nine related research groups. The inspected research files match the pinned public-main source tree; this is file-level source verification, not independent numerical replay of every historical experiment.

| Prior evidence | Observation | Scope |
|---|---|---|
| LV4 / LightGBM | On the corrected 7,120-view comparison, equal-lineage R² was 0.784889 for LightGBM and 0.690235 for DL; DL−LightGBM 95% interval[−0.132907,−0.056273]. | Historical development GP-HWM prediction; different inputs/selection procedures and unresolved encoder exposure. This is not candidate-slate ranking utility. |
| EXP-018 contextual surrogate | Historical 93-feature LOPO strength-of-schedule-debiased surrogate: shadow-v2 R4 top-1 gain over program prior +0.043596[0.010236,0.082032], over ridge_program +0.066919[0.030060,0.105549]. | Post-iteration shadow evidence. Seven development iterations disclosed; the joint flywheel criterion remained false. Active/passive pairwise and B8 gains were inconclusive. |
| LV4 routing | Saved CV-only table: skipping~10% of contexts skipped 10.1% of compute proxy, missing 0.41% of wins and 0.58% of gain. | Compute proxy is training-row count; target is selected GP-HWM and has winner-selection caveats. No deployed saving is established. |
| H08 full-pool A0 | A0 versus frozen-current selected CV-ratio reduction +7.475670%[1.293690%,14.039185%],1,594 contexts / 379 families. | Exploratory reused development; adaptive 1.836M pool overlaps the initial pool. No neural advantage established. |

R², AUC, shadow gain, geometric CV-ratio reduction, native L6 and raw constrained fitness have different meanings. They cannot be pooled into a single “best result.” Current contextual-surrogate source is v3 with stricter nested/provenance rules; the positive EXP-018 receipts measured v2. Source changes are not new numerical validation.

J8/J21/J22/self-play also need careful labels. J8 separates restart, replay, volume and feedback, but its historical closed scratch endpoints were worse than frozen v1. J21's six displayed arms were inconclusive; the larger L6-1148 panel gave regression alerts for three continuations. J22 has 18 synthetic controls and no actual GP/GPU/optimizer work; its multiseed/teacher-assessment contracts are not a completed learning study. SPQ-1 source/tests exist, while its inspected preregistration remains unpinned and ungranted. ADR0005's lineage-B step 60,100 was unchanged-corpus hardware continuation with zero fresh programs.

## H27 completed result

H27 trained 36 matched models for 14,400 updates on 850 historical TRAIN contexts / 164 families / 162 components. All180 fixed TRAIN/internal-monitor snapshots were retained, and independent saved-model QA replayed 15,300 out-of-fold choices. The two primary contrasts use 100,000 paired whole-component bootstrap draws: each central 98.75% interval contributes to the 97.5% familywise two-contrast coverage.

| Primary raw-fitness contrast; lower is better | Difference | Interval |
|---|---:|---:|
| Real state, five-fold minus zero fingerprints | −0.1055719651 | [−0.7890362249,+0.2869843098] |
| Real state, five-fold minus one-fold fingerprints | +0.1482177800 | [−0.0230595524,+0.4012829990] |

Both intervals include harm. The favorable one-fold point is exploratory; it is not a replacement primary or proof that one-fold landmarks work. State effects are secondary points with no established uncertainty here. Zero-state arms retain common input/model descriptors, so they do not remove every Dope-derived feature. This is component-crossfit development on an exposed TRAIN population, not fresh-family confirmation. QA reuses the pinned model kernel and does not independently retrain or replay native Dope fitting.

## More training versus more independent tasks

Longer static-corpus training was already attempted: the historical v1 run totaled 17,773 updates, and all 31 closed validation epochs 10–40 stayed inside [0.8968201,0.8978393] after early improvement. That establishes a plateau for that recipe, not a universal limit.

In H27, all 36 final TRAIN and internal-monitor objectives were below initialization. However, 31/36 monitor objectives worsened from step 100 to 400, versus 0/36 TRAIN objectives. The fixed 400 result remains unchanged. Different fold scales/panels mean these curves must not be averaged into one validation-loss curve, and step 100 cannot be selected retrospectively as a clean result.

Historical whole-family subsampling supported more task breadth, but the reported 2× family forecast was not an actual new-family experiment; no 10×/100× gain is established. A family-acquisition tournament was null. More candidates from existing families do not create independent validation units.

The finite first data opportunity is the original 1,604 eligible TRAIN contexts / 382 recorded families with source-bound teacher TRAIN-prefix and OOF-target integrity receipts. The old 1,594/379 restriction followed 10 unavailable Dope encodings. A tree with 88 A0 and 412 candidate features need not inherit a Dope-complete-case exclusion. Current roles, source aliases, reservations, exact overlap and evaluator/label semantics must be requalified before claiming additional admitted breadth. Headline subtraction is invalid. These remain historical TRAIN families, not a new lockbox.

## H28 status at this publication snapshot

The H28 producer completed its zero-fit saved-choice decomposition over 18 real-state files,7,650 fixed choices and all 850 contexts/162 components. Independent saved-choice arithmetic QA now passed: all three contrasts, gate-transition cells, seed/component grouping and concentration were reproduced, with the original two H27 primary estimates and intervals unchanged. Its scope is arithmetic of saved selected values; it does not repeat the 18-array extraction or model/native forward.

The descriptive gate-transition cells account for roughly 97–98% of the net contrast points. The both-passing contribution is only −0.002036 for five−zero, +0.004485 for five−one and−0.006521 for one−zero, versus total differences−0.105572, +0.148218 and−0.253790. One component contributes about 81% of the favorable one−zero net point. All components, failed choices and original population weights remain included. This concentration is neither a causal explanation nor new significance; it argues against overselling the one-fold point or tuning a new loss on these exposed outcomes.

## Next wave

[NEXT_WAVE_PLAN.md](NEXT_WAVE_PLAN.md) gives an ordered, worker-ready proposal. Root prospectively selects a new **no-Dope** contract:412 candidate features,88 A0 features and8 padded fingerprints, with zero/one/five and a label-blind per-context permuted-one-fold control. On the currently qualified850 contexts, two folds ×three seeds ×four arms means **24 final tree fits** and no new GP probe fits. The primary contrasts are one−zero and one−permuted; five-fold is secondary. Existing pinned LightGBM continuous-cost source must first qualify the proposed query-centered uncapped raw-fitness objective, TRAIN-only global scale and component/context/slate weighting.

This choice is informed by H27/H28 development observations, so the new study is **DEVELOPMENT_ONLY**. Different inputs prevent a causal architecture comparison with H27. In parallel, H29 can requalify the original1,604 metadata population before a common64-slate breadth×duration test. No broader cohort is admitted by this summary. Failure-aware calibration/tail risk stays conditional on separate TRAIN-only evidence; no gate observation is an inference filter.

Existing LambdaRank is already pairwise; H07 compressed bilinear Ridge already failed, and H09's full-main-effects residual proposal is already queued. Those are not new experiments. All failed/unknown work, negative controls and nulls must survive publication. Proposed files and commands are not an execution grant.

Primary research informs the design without guaranteeing success: [Grinsztajn et al., tree versus tabular DL](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html), [Cawley and Talbot, selection bias](https://www.jmlr.org/beta/papers/v11/cawley10a.html), and [Sagawa et al., regularized group robustness](https://openreview.net/pdf?id=ryxGuJrFvS). Literature hypotheses do not override Jope's source, role or evaluation contracts.

Machine-readable [findings](FINDINGS.json) and [source pins](SOURCE_PINS.json) retain the exact scopes and receipt identities.
