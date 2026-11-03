# H238: lower-LR and longer-horizon r882 adapter continuation

Owner: luna_v1_res08. Authority: `hw_t0205_res08_h238_lowlr.md`, claim under
`~/jope_plan/ns/receipts/Hwave-admission.lock` in `H238_claim.txt`.
Nothing scientific runs before this preregistration merges. Offline checkpoint COPY only.

Hypothesis: smaller AdamW steps over the same or longer horizon yield a continuation that the strict
selector admits, improving DL-Val91 utility beyond r882. Prior H236/H237 outcomes motivated the dose-down;
they are historical observations, not H238 results. H133 LOWLR_LONG did not complete a treatment run.

| Arm | AdamW LR | Updates per trajectory | Selector candidates | Role |
|---|---:|---:|---|---|
| r882 | — | 0 | 0 | frozen reference, Elo 1500 |
| CONTROL | 1e-4 | 40 | 0/10/20/30/40 | exact H170 CTRL |
| LR3E5 | 3e-5 | 40 | 0/10/20/30/40 | primary |
| LR1E5_120 | 1e-5 | 120 | 0/30/60/90/120 | secondary, Holm m=2 |

The only scientific changes are learning rate and update count. ALL arms retain H170 CTRL AdamW,
betas(.9,.999), eps1e-8, weight decay1e-4, grad-norm clip1.0, identical Huber+0.1KL loss and paired
H156 sampler (8 fit contexts, 4 reveals +20 sampled candidates). The introduction's "skip clipping"
means no clipping intervention: the explicit arm specification keeps clip1.0. No SF optimizer,
warmup/schedule, extra head or architecture change. Only47 adapter tensors/29,211 parameters train.
DOPE c51a56d0 and every non-adapter byte are checked after each update and evaluation.

Reuse H236 source39fd7d58258a2aefd9aa2eadef201ae4b5069812 and H237 monitor at
4548321f0f6c2baabbbec6c33a4a143fa8242800; keep H232 None-gradient and namespace fixes. Parent runtime
sources bind immutable Git objects. No shared/foreign checkout edits or new worktree/clone; own
execution capsule binds all bytes to the merged prereg commit. Checkpoint COPY/reference under
`~/jope-data/h238/inputs/`, pinned by SHA256; never the live loop.

Fresh seeds238101–238103, family fold seeds238141–238143, bootstrap238150. Per seed: five TRAIN folds;
fit three, select next one, assess remaining one.15 cells ×(40+40+120)=3,000 primary updates.
Generate one120-batch stream per cell using the unchanged sampler; short arms use its exact first40.
Prefix hashes and arm budgets are locked before any trained outer-A/DL-Val assessment.
Arm-specific candidate grids always have five points. Select the earliest numerical tie (eps1e-12),
including r882. Accept only if selector equal-family gain>.001+eps and bootstrap LCB95>0; otherwise
select r882. The selector is unchanged; only its explicitly assigned step labels differ by arm.
Snapshot duplicate reports disclose both candidate index and actual step per arm; end drift refers
to step40 for CONTROL/LR3E5 and step120 for LR1E5_120.

ONE exclusively claimed attempt: Native91 first (all91 utilities exact, mean1.1377941898164525,
cached/native forward bit-identical) BEFORE any TRAIN pools. Then one disposable fit-C TRAIN-smoke
update per arm, finite/change/frozen checks, bit-exact r882 restoration and fresh optimizers;
then the full wave. Three smoke updates are separate from3,000 primary updates. No retry or
partial resumption, no tuning after outcomes. Unavailable/incomplete outcomes remain null.

Primary: DL-Val91 regression utility, higher=-fitness is better; failed gates have utility-2.
Per seed average the five selected policies, then equal-family aggregate. Report the three seed
gains, median gain, shared10,000 family-bootstrap CI95 and one-sided p. Holm m=2 covers LR3E5 and
LR1E5_120 versus r882. Per-arm ADVANCE requires median gain>.002+eps, family LCB95>0 and Holm p<.05;
ties reject. Primary LR3E5 verdict is separate; aggregate identifies any qualifying arm and never
changes that primary verdict. Against CONTROL is descriptive. Canonical family Bradley–Terry Elo,
r882 anchored1500, median seed lift and family CI95 (1,000 paired draws), vs r882 and CONTROL.
Secondary: strict TRAIN OOF1219/205, unconstrained diagnostics, step40/120 L2 drift per arm,
harm families (three-seed mean family delta<-.01) and duplicate policy/adapter identities.
Compare context means descriptively with r3175 plateau1.1389572. No AUC or speed verdict.

Real S3-origin data only per `~/jope_plan/master/source/S3_DATASET_CATALOG.md`; inherited276 dataset
IDs join origin=real. Pinned H50 selected/source manifest/pools and H58 cache. TRAIN1219/205 and
DL-Val91/19, disjoint families. Reject has_meta_test=True, non-regression or non-train/inner sources.
No sealed, holdout, meta-test or live-loop reads. DL-Val never enters fitting or step selection.

Host xbabe2 CPU, exact taskset40–43, nice19, one math thread, deterministic Torch, no GPU;
never xbabe0/xbabe3 or another job's controls. Admission load5<=100 AND MemAvailable>=22GiB.
Pause own worker group on load5>108 OR MemAvailable<20GiB. Resume only when load5<=100 AND
MemAvailable>=22GiB. Hard stop if MemAvailable<12GiB, own supervisor/worker RSS>=4GiB,
disk free<300GB, tree threads>4 or compute deadline2026-11-03T07:00:00Z.
Single attempt, no retries; resource refusal before admission is NOT_STARTED.

Prereg merged by03:00Z; compute STOP07:00Z; closeout merged by07:30Z, else CLOSED_INCOMPLETE.
Local gate: targeted data-free pytest, Ruff/check+format, diff-check, JSON/schema, exact fold/source/
registry checks and actual jope-records coverage validate. A configuration-only input projection is
used if the shared physical checkout is stale. Owner explicitly authorizes admin squash; remote
CI is NOT_A_GATE, record NOT_RUN honestly. Do not modify PR#2071.

Closeout: hypothesis/setup/outcome/metrics/verdict/next step in README; small metrics JSON, both
index rows and CLOSED claim line. Full output/source/receipts plus own starting COPY/reference go
create-only to versioned S3 under
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261103/H238/res08-t0205/`, with exact
VersionId byte/member replay. No large Git objects; raw roots retained.
