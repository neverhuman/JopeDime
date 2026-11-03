# H239 — parameter-subset continuation of an r882 copy

Owner: **[ASTRA]**. Status: **PREREGISTERED; no H239 outcome read or training**.
Authority: `hw_t0235_astra_h239_subset.md` and pre-outcome amendment
`hw_t0237_astra_h239_A1_memguard.md` (A1, 2026-11-03 02:37 UTC).
Claim: `~/jope_plan/ns/receipts/H239_claim.txt`, under `Hwave-admission.lock`.

## Hypothesis and fixed arms

Restricting which adapter parameters move may reduce harmful cross-family drift while retaining useful
continuation signal. This complements H238's learning-rate intervention. Historical H237 results motivate
the experiment; they do not establish that its drift caused its utility loss.

| Arm | Trainable tensors | Parameters | LR | Role |
|---|---:|---:|---|---|
| CONTROL | 47 | 29,211 | 1e-4 | exact H170 CTRL AdamW |
| BIAS_ONLY | 23 | 755 | 1e-4 | primary: biases and LayerNorm gains |
| HEAD_ONLY | 2 | 25 | 1e-4 | secondary: final trajectory scoring linear layer |
| LLRD | 47 | 29,211 | 1e-4 × 0.5^d | secondary: fixed depth map |

[trainable_tensors.json](trainable_tensors.json) enumerates every tensor, shape, parameter count,
arm membership, LLRD depth and multiplier. Its explicit per-arm lists are authoritative and frozen before
any outcome. [TENSORS.md](TENSORS.md) is the readable census. The two fixed program normalization buffers
are excluded. DOPE and the anchor never enter any optimizer.

BIAS_ONLY includes all linear, attention and LayerNorm biases plus the five LayerNorm gain vectors; it
excludes scalar kernel/gating controls and embeddings. HEAD_ONLY means precisely
`adapter_10tok.traj_delta_head.2.{weight,bias}`. The earlier `mm_prior_out` branch is frozen. The final
bias is centered out of the trajectory score; zero useful gradient there is an honest capacity limitation,
not a reason to change the list. The unused parent-delta branch stays in CONTROL/LLRD and its biases/norm
in BIAS_ONLY; native AdamW skips parameters whose gradients are None, including weight decay.

The adapter is a graph with parallel branches, not a transformer stack. The fixed LLRD convention is:
d=0 final trajectory projection/output scales and unused parent final projection; d=1 trajectory hidden
projection/attention, kernel controls and unused parent hidden/norm; d=2 history projection/null history,
multimodal final projection/scale; d=3 multimodal hidden projection/attention/bilinear projections;
d=4 candidate query and modality embedding/norm/gates; d=5 program bottleneck. This is a prespecified
graph-depth convention, not a claimed unique architectural depth. No tensor is omitted or reassigned later.

## Unchanged training and boundaries

The runner is an exact-source derivative of merged H236, using H237's fixed selection/statistics and
memory-monitor lineage. Parent commit and all imported source hashes are in [manifest.json](manifest.json).
Only gradient membership and per-tensor LR change the scientific treatment. All arms keep the H170 CTRL
Huber + 0.1 KL loss, AdamW betas (0.9,0.999), epsilon 1e-8, weight decay 1e-4, global gradient clipping 1.0,
constant base LR 1e-4, H156 batches (8 FIT contexts, four top-fitness reveals plus 20 sampled candidates),
eval mode and no added dropout. Candidate steps are 0/10/20/30/40. No other optimizer or loss intervention.

Seeds 239101–239103; fresh family folds 239141–239143; bootstrap seed 239150, 10,000 utility draws and
1,000 Elo draws. Each seed partitions 205 TRAIN families into five groups of 41. Outer A is held out;
next group B selects; remaining 123 families fit. All arms share batch streams within a seed/cell.
There are 15 cells × 4 arms × 40 = **2,400 logical updates**, plus four disposable TRAIN-smoke updates
per process invocation, separately counted. Every invocation starts from the same hash-pinned r882 COPY.
A resume reuses completed cells, never rerolls them. Recomputed interrupted-cell work and smoke costs
remain in attempt-specific journals; cumulative physical updates are reported separately from 2,400
logical completed-trajectory updates. The fixed absolute compute deadline applies to every resume.

Real S3 corpus only, as documented by `~/jope_plan/master/source/S3_DATASET_CATALOG.md` and inherited
[provenance.json](provenance.json): 1,219 TRAIN contexts/205 families and 91 DL-Val contexts/19 disjoint
families. Exclude `has_meta_test=true`, sealed, holdout, meta-test and non-training-table paths.
The source audit permits only the pinned inputs/assets and declared cohort pools. Dataset labels remain
unchanged. DL-Val's prescribed features and four revealed labels are used only for prediction; its outcome
panel never chooses an update, step, arm, LR, depth map or stopping time. B alone selects; A and DL-Val
assessment happens only after each cell's selection lock is durably written. Partial output is not used
to alter science or to issue an efficacy verdict.

First, reproduce all 91 native r882 utilities bit-exactly and their mean **1.1377941898164525**. Then one
finite, changing TRAIN-only update per arm, checking the frozen subset and DOPE, followed by exact r882
restoration. Then the complete run. No scientific smoke runs before the preregistration merge.
DOPE composite `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` and non-adapter
weights are checked before/after, and after every update. Frozen subset coordinates must be bit-identical
to initial. Record all 47-coordinate step-40 L2 drift, including unchanged coordinates.

## Selection and decision

B selection is unchanged from H237: family-mean utility; earliest candidate within 1e-12 of the best
wins ties. Select it only if gain over step 0 is >0.001+1e-12 and family-bootstrap LCB95 >0; otherwise keep
r882. No second-best fallback. DL-Val averages each seed's five locked outer-cell policies; TRAIN OOF
assesses each context exactly once per seed. Higher regression utility is better. No AUC or speed verdict.

Primary is **BIAS_ONLY versus r882 on DL-Val91**. Report all three seeds; median seed family-mean gain,
95% paired family-bootstrap CI, positive-seed count, and Holm-adjusted one-sided p with **m=3** over
BIAS_ONLY/HEAD_ONLY/LLRD versus r882. ADVANCE requires all four: median >0.002+1e-12, LCB95 >0,
Holm p<0.05, and at least 2/3 positive seeds. A tie is NO_ADVANCE. Secondary-arm advances do not turn
a failed BIAS_ONLY hypothesis into a primary success. Report each treatment versus CONTROL, selected
and descriptive unconstrained panels, TRAIN OOF, and context mean against plateau best 1.1389572 (r3175).
The plateau comparison has a different historical selection identity and is descriptive.

Report canonical family-weighted Bradley–Terry Elo (r882=1500), 95% family-bootstrap intervals and paired
Elo lift versus CONTROL; utility is the decision metric. Harm families have three-seed mean selected
utility delta versus r882 below -0.01. Exact duplicate predictions and shared r882 fallback are disclosed,
not treated as independent replications. INCOMPLETE means unknown efficacy, not zero lift or NO_ADVANCE.

## Resources and A1

CPU only on xbabe2, taskset 48–51, nice 19, one math thread. No GPU, xbabe0, xbabe3, live flywheel,
foreign process control or PR #2071 mutation. RSS including supervisor/descendants <4 GiB; aggregate
thread guard 4; disk free >=300 GB. Admission requires load5<=100 and **MemAvailable>=30 GiB**.
Load5>108 requests a cooperative pause; resume only at <=100.

**A1 supersedes the initial memory rules before outcomes:** below20 GiB requests a cooperative pause
between optimizer steps (also honored between evaluation batches and pool loads); no SIGSTOP is used.
Resume after >=25 GiB sustained30s. Hard stop only if below12 GiB sustained30s, or below6 GiB on any
sample, in addition to RSS/disk/thread/deadline stops. Supervisor samples every0.5s. It signals only its
own new worker process group on a hard stop. A cooperative pause can finish the currently executing
optimizer step; it then blocks before the next. This is not an immediate scheduler freeze.

Every completed cell retains selection JSON, metrics/trace JSON and all candidate adapter NPZs, with
hash validation on recovery. Up to **two process resumes** of incomplete work are allowed within the
same window; no completed cell is recomputed. Cooperative pauses do not restart the process or reset
its budget. Recoveries use the same merged source, inputs, folds and immutable decisions and preserve
all earlier attempt evidence. Any missing/invalid completed-cell checkpoint prevents recovery.

Preregistration merge by **2026-11-03 03:45Z**; compute hard stop **07:30Z**; closeout merge **08:00Z**.
One primary attempt, at most two A1 resumes, no parameter search, no re-rolls, no deadline reset.
If incomplete, publish honest receipts and null scientific outcomes.

## Publication and custody

Targeted pytest, Ruff, JSON/schema parse, `git diff --check`, records coverage and source/test inventories
precede review and owner-authorized admin squash merge. Remote CI is not a gate; NOT_RUN is not PASS.
The execution capsule must match the merged source. Local full outcomes/adapter snapshots live outside
Git; create-only S3 prefix `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261103/H239/astra-t0235/`
holds full metrics, runtime archive, input identities, manifests and exact VersionId/SHA readback receipts.
Compact metrics and both indexes are updated in closeout. No serving promotion is authorized.
