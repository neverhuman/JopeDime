# H158 — R782-to-r882 listwise distillation with a gate-cliff veto

**NO_ADVANCE.** One preregistered run completed 12 cells/480 actual optimizer updates. R882 is an unchanged reference, not another trained arm. [metrics.json](metrics.json) contains the full registered readout; no serving promotion.

Hypothesis: transfer R782's ranking distribution into r882 at low learning rate, dropping the teacher term on fit contexts where its full-pool argmin fails the native gate. [PREREG.md](PREREG.md) fixed CTRL/KD03/KD07/KD07_NOVETO, seeds 158101–158103, fold 158140 and bootstrap 158141 before scored inference. Each arm starts from identical r882 and trains 40 AdamW updates, lr 1e-5, wd 1e-4, batch 16, slate 192, clip 1. Only 47 adapter tensors/29,211 parameters train. Teacher R782 is fully frozen, forward-only and fit-only; its complete state hash remained `c638885e77c5bb9b194e2c219faf509933d71a85e8e417a294ba58f151c0082a`. DOPE c51a56d0 and every non-adapter tensor/buffer stayed byte-identical.

Loss is (1-a)*live + a*KL(teacher||student), temperature 0.2, no temperature-squared multiplier. The H151 live loss uses SmoothL1(beta 1)+0.1*target KL(tau 0.2). A veto restores full live loss. Veto uses teacher full-pool fit-context argmin/gate; KL targets are computed separately on the exact paired 192-candidate training slates with matched first 4 revealed inputs. Full-pool teacher scores are not sliced into training targets. No teacher scoring of assess or DL-Val.

Fit used 959 contexts/164 families; assess fold 0 held 260/41 families out of this continuation. Historical source-adapter TRAIN exposure remains, so both assess and full 1,219/205 TRAIN results are descriptive. DL-Val is repeatedly exposed exploratory evidence, not fresh/sealed qualification.

Primary: equal-context means within 19 families, equal-family mean within seed, median across 3 seeds, 10,000 paired family bootstrap draws. Holm family is the 3 KD arms; CTRL is diagnostic/ineligible. Rank-ordered lower quantiles .025/3,.025/2,.025 and adjusted p < .05 must pass the entire preceding prefix, plus median gain strictly > +.001. Ordinary 95%CIs are shown separately. Ties never advance; no posthoc tie band or new efficacy gate.

| Arm | Gain vs r882 | Ordinary 95% CI | Elo vs r882 | Positive seeds | Invalid picks | Holm p | Registered Holm LCB95 | Decision |
|---|---:|---|---:|---:|---|---:|---:|---|
| CTRL | -0.00243804 | [-0.00919172, +0.00187761] | 1481.72 | 0/3 | 0,0,0 | — | — | diagnostic |
| KD03 | -0.00243804 | [-0.00919172, +0.00187761] | 1481.72 | 0/3 | 0,0,0 | 1.0 | -0.00919172 | NO_ADVANCE |
| KD07 | +0.00054851 | [-0.00069623, +0.00234177] | 1500.00 | 3/3 | 0,0,0 | 1.0 | -0.00069623 | NO_ADVANCE |
| KD07_NOVETO | +0.00054851 | [-0.00069623, +0.00234177] | 1500.00 | 3/3 | 0,0,0 | 1.0 | -0.00069623 | NO_ADVANCE |

All three seeds produced identical primary outcome vectors within each arm, although saved adapter parameters differ. KD07 and KD07_NOVETO also have identical primary outcomes: gain +0.0005485109 is below the strict +0.001 margin, and their ordinary and registered adjusted lower bounds are negative. This is a repeated selected-policy result on one validation panel, not three independent datasets or evidence that the veto has no effect generally.

R882 regression passed 91/91 exactly before teacher inference/updates: context mean 1.1377941898164525, zero context/mean error; equal-family baseline 1.1536152770. Baseline invalid picks: 0/91 validation, 4/260 assess, 17/1219 full TRAIN.

| Arm | Assess gain [95% CI] | Full TRAIN gain [95% CI] | TRAIN Elo vs r882 | TRAIN invalid picks |
|---|---|---|---:|---|
| CTRL | +0.00477373 [-0.00025783, +0.01251105] | -0.00081244 [-0.00462459, +0.00215463] | 1505.08 | 18,18,19 |
| KD03 | +0.00484334 [-0.00024015, +0.01240146] | -0.00080829 [-0.00469717, +0.00199623] | 1508.47 | 18,18,18 |
| KD07 | -0.00016002 [-0.00044690, +0.00002148] | -0.00256821 [-0.00768713, +0.00005725] | 1496.61 | 17,17,18 |
| KD07_NOVETO | -0.00016002 [-0.00044690, +0.00002148] | -0.00243534 [-0.00764376, +0.00005966] | 1496.61 | 17,17,18 |

Veto support: teacher selected an invalid full-pool candidate on **10/959 fit contexts**. Each cell has 640 sampled context exposures (duplicates retained). The table records actual teacher-term drops and displacement from initialization, not weight norms. Zero or small support limits the ablation; a small KD07–NOVETO difference is not evidence against veto effectiveness on broader data. Any ADVANCE concerns the combined registered arm, not a unique causal mechanism.

| Arm | Adapter L2 drift by seed | Vetoed sampled contexts by seed (out of 640 each) |
|---|---|---|
| CTRL | 0.04333349, 0.04395314, 0.04315173 | 0, 0, 0 |
| KD03 | 0.04247997, 0.04298818, 0.04197317 | 6, 8, 9 |
| KD07 | 0.03570880, 0.03619801, 0.03316402 | 6, 8, 9 |
| KD07_NOVETO | 0.03559651, 0.03615030, 0.03446857 | 0, 0, 0 |

Execution source `25b15c794ba4c84442bdff2a753f69e9ac90e8cc`, prereg PR #1330; self-review and OPUS review 5726583662 preceded merge/run. Constructor+AdamW lazy-JIT smoke with 0 forwards and 0 updates passed; 28 data-free controls/Ruff/format/hash/link/registration/count/diff checks passed. Run wall 5331.6s, aggregate sampled peak RSS 3,332,849,664 B/max 3 OS threads, min free 634,127,572,992B, max disk used 91.9283%, max load 81.52, min MemAvailable 67,859,619,840B. Terminal guards were active throughout; no retries, GPU, qualification lock or live writes. Python audit records no protected reads/refusals but is application evidence, not an OS-wide sandbox.

Independent scalar replay reproduced all 20 primary/assess/full TRAIN contrasts, bootstrap bounds/p/Elo/invalids, Holm prefix and final verdict; all 12 saved adapter hashes and changed-parameter/norm checks passed. All 480 update-journal entries, independently regenerated batch plans/veto counts, source identities, teacher-fit inventory and cached target hashes verified. Replay also matched 21,109 saved choices to original candidate labels/gates and recomputed initial frozen-state and teacher digests directly from restored checkpoints. No model forward or training in replay. Hosted CI not run under t2300; no product qualification claim.

Custody: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H158/astra-t2300/manifest.json`, VersionId `HmAfLWjk6cxArOvoBItZEuSBJfwmDBnz`, SHA256 `e2d0c68e67cac7821fbd60baf6105b25b1efc0ff30fd8342fdb58688eef48ff9`. [custody.json](custody.json) binds create-only input/result archives with exact-version downloads and **1418 member hashes** verified before owned-bulk cleanup. Original H50/H58 derived-object versions remain unknown; exact numerical closure is archived, external implementation sources hash-pinned. Shared inputs and other owners' files retained.

Next: close without promotion; further teacher, dose, gate or schedule choices require a new preregistered wave.
