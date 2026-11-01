# H202: current-top5 hard-negative adapter fine-tuning

**NO_ADVANCE.** The single guarded CPU attempt is COMPLETE: **1,800 optimizer updates / 7,200 context backwards**, across 45 cells. R882 remains the incumbent; this study makes no live change or adoption.

The median family gain is +0.01089685 (95% CI −0.00461796 to +0.02568691), with harm on 52/205 families (25.37%). The lower-confidence-bound, p-value and harm gates fail.

## Hypothesis and protocol

H202 asks whether pairwise logistic fine-tuning on the adapter’s own current top-five ranking inversions improves utility over R882 on whole families excluded from that cell’s gradients. Prior hard-negative work exists (including H58); this experiment tests this particular current-top5 pair objective and paired family sampling, rather than claiming hard-negative mining is new.

The frozen [preregistration](PREREG.md), [manifest](manifest.json) and source were merged before the sole attempt. R882 uses an immutable offline checkpoint copy. DOPE c51a56d0 and every non-adapter tensor stay frozen; only the 47 adapter tensors (29,211 parameters, excluding its anchor) receive gradients or enter the optimizer. Each cell restores the exact same R882 adapter and uses a fresh AdamW optimizer, lr 1e-4, weight decay 1e-4, betas (0.9, 0.999), epsilon 1e-8, and global gradient clipping at 1.0, matching H197 CTRL.

HNEG40, CTRL40 and HNEG_SHUF each run 40 updates. Each update accumulates four equally weighted contexts: choose one FIT family uniformly with replacement, then a context uniformly within that family. The three arms share the exact 40-step context plan per seed/fold. CTRL uses the ordinary SmoothL1 + 0.1 KL flywheel loss. HNEG uses the mean softplus(cost_better − cost_worse) over all strict true-utility inversions within the current stable top five. Utility ties supply no pair; zero-pair loss is a differentiable zero. HNEG_SHUF uses a fixed full-pool within-context utility permutation per seed/fold/context, then mines its own current top five.

Native utility is −fitness when the gate passes and −2 otherwise. Selection uses the full-pool first argmin, preserving candidate order for cost ties. All forwards use the unchanged H197 Engine, full candidate pools, the original first four reveals and CPU batch size one. Frozen anchor features are cached once. The first scientific gate reproduced every native R882 DL-Val utility bit-exactly, mean **1.1377941898164525**, before opening TRAIN outcome pools or constructing an optimizer. Native/cached costs match exactly. Byte parity against historical H170 batch-eight costs is diagnostic only; this study fixed batch one before outcomes.

The roster is 1,219 TRAIN contexts /205 families and 91 DL-Val contexts /19 separate families, all joined to real S3-catalog entries. Three fresh independent five-fold maps (41 families/group) use fold seeds 202141–202143; model/sampling seeds are 202101–202103. In each outer cell, A is one group, B the next cyclic group and FIT the other three: 41/41/123 disjoint families. B is excluded from all gradients. No synthetic, sealed, holdout or meta-test data is used.

All 45 cells finish before selection. One common step per arm is chosen from 0,10,20,30,40 by pooled B seed-average equal-family gain; earliest exact ties select the earlier step. A fsynced, directory-fsynced, hash-bound B lock is durable before any candidate A or candidate DL-Val scoring. Saved selected state hashes are checked against that lock.

## Selection and primary pooled A outcome

| Arm | B-selected common step | B gain at selected step |
|---|---:|---:|
| HNEG40 | 40 | +0.00697257 |
| CTRL40 | 0 | +0.00000000 |
| HNEG_SHUF | 0 | +0.00000000 |

Primary pooled A assigns each TRAIN context to the model whose A fold contains its family, separately for each seed. Family means weight all 205 families equally; reported points are medians across the three seedwise means. The 10,000-draw bootstrap (seed 202150) shares family indices across seeds and arms and takes the median over seeds within each draw.

| Arm | Family gain vs R882, median [95% CI] | Positive seeds | Family harm | Elo vs R882 [95% CI] | Exact −2 picks /seed |
|---|---:|---:|---:|---:|---:|
| R882 | 0 (reference) | — | 0.000000 | 1500.000 [1500.000, 1500.000] | [17, 17, 17] |
| HNEG40 | +0.01089685 [-0.00461796, +0.02568691] | 3/3 | 0.253659 | 1502.692 [1497.707, 1505.973] | [14, 19, 16] |
| CTRL40 | +0.00000000 [+0.00000000, +0.00000000] | 0/3 | 0.000000 | 1500.000 [1500.000, 1500.000] | [17, 17, 17] |
| HNEG_SHUF | +0.00000000 [+0.00000000, +0.00000000] | 0/3 | 0.000000 | 1500.000 [1500.000, 1500.000] | [17, 17, 17] |

| Mechanism contrast | Family gain, median [95% CI] |
|---|---:|
| HNEG40_minus_CTRL40 | +0.01089685 [-0.00461796, +0.02568691] |
| HNEG40_minus_HNEG_SHUF | +0.01089685 [-0.00461796, +0.02568691] |

Primary Holm m=1 p=0.14058594. The preregistered efficacy conjunction requires median >+0.001 (strict numerical margin), lower 95% endpoint >0, 3/3 positive seed gains, seed-average family harm <=10%, and p<.05. HNEG must additionally beat CTRL and SHUF by strictly positive paired median; placebo must not meet the R882 efficacy conjunction. Ties reject. Gates: HNEG efficacy **False**, beats CTRL **True**, beats SHUF **True**, placebo efficacy **False**.

Elo uses context-pair wins/ties first, then equal-family weighting, tie tolerance 1e-4, Bradley–Terry MM400, symmetric prior 0.01 and R882=1500. Harm counts strict utility losses after seed averaging within families. The full [metrics](metrics.json) include per-seed utility/gains/Elo/CIs/harm and every B step gain.

## Report-only DL-Val

DL-Val is repeatedly selected development data, not independent confirmation. Each selected fold model scores all 91 contexts: 455 model-context pairs per seed. Reported utility averages uniformly across the five models; pairwise wins compare policies within the same fold before family aggregation. This is not score ensembling. Nineteen families receive equal weight.

| Arm | Descriptive family gain vs R882 [95% CI] | Elo [95% CI] | Exact −2 picks /455 per seed |
|---|---:|---:|---:|
| R882 | 0 (reference) | 1500.000 [1500.000, 1500.000] | [0, 0, 0] |
| HNEG40 | -0.02227137 [-0.07712459, +0.00624005] | 1503.654 [1486.842, 1516.086] | [5, 3, 4] |
| CTRL40 | +0.00000000 [+0.00000000, +0.00000000] | 1500.000 [1500.000, 1500.000] | [0, 0, 0] |
| HNEG_SHUF | +0.00000000 [+0.00000000, +0.00000000] | 1500.000 [1500.000, 1500.000] | [0, 0, 0] |

## Training and verification

| Arm | Context backwards | Inverted pairs mined | Zero-pair contexts | Nonzero-gradient updates |
|---|---:|---:|---:|---:|
| HNEG40 | 2400 | 11491 | 62 | 600 |
| CTRL40 | 2400 | 11516 | 65 | 600 |
| HNEG_SHUF | 2400 | 11862 | 26 | 600 |

Pair diagnostics are also computed for CTRL, whose actual training loss remains the ordinary objective. Zero-pair/zero-gradient telemetry distinguishes learning from possible AdamW decay; update count alone does not prove useful gradients. All training costs, B/A/Val costs, pair choices, traces, 180 adapter snapshots and the baseline costs are retained losslessly using float32 bitwise XOR against the immutable R882 reference. No output quantization or checkpoint/npz is committed to Git.

Independent read-only verification reconstructs all 1,310 pool hashes/catalog joins, all 7,200 training losses and top-five inversions/shuffles, all 1,800 paired sampling steps, 180 adapter states, every saved B/A/Val first-argmin choice and native utility, pooled B selection, all seedwise/bootstrap contrasts, BT Elo, harm and the final verdict. It constructs no model and performs zero forward/backward/optimizer calls. DOPE/non-adapter identity is asserted throughout training and at every cell; the final entire model is restored to R882. Input audit records zero refused/protected reads.

Attempt001 ran on xbabe1 at nice19 under a 3-thread aggregate/RSS<5GB/MemAvailable>=12GiB/free>=40GB/output<300MB/load1<=108.8/4h guard. Actual elapsed 2396.56s, maximum 3 threads, peak aggregate RSS 3,245,277,184B. No inference retry, GPU job, serving promotion or live-flywheel write. Eighteen data-free controls, Ruff/format/source pins and all five jope-records gates pass; full hosted/workspace CI is not claimed under the study-specific owner no-CI-wait order.

Prereg PR #1684 merged 64be8b477e23fd410e85baeaab6896bded8dd79d, exact reviewed head e0b1466132f78678366b994974f038074ed4edb1, same-agent review 5730591938. Scientific [custody](custody.json): s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H202/codex-t1505/7cda6382788e1bce/H202-evidence.zip VersionId _JDRpN3zjms92a.A4M0ca2mjtd0YtMd6. Actual versioned GET verifies all 2063 member payloads, 1,310 approved input pools, 180 adapter states and 21888 NPZ arrays. Compact metrics are 13,163B (<100KB).

## Limits and next step

A/B cyclic roles reuse the same 205 families; pooled B step selection and pooled A therefore are not independent cohorts. Independent seed/fold maps repeat training and partitioning, not acquisition of new families. Fixed epochs, current-top5 mining and the chosen common step limit this experiment’s conclusions; a B-selected step 0 may yield a baseline A policy despite all 1,800 updates having run. Repeated study selection remains a limitation of the research program.

The registered conjunction did not pass. Keep R882; close this hypothesis as NO_ADVANCE. Any new objective or schedule requires a separate registered study.
