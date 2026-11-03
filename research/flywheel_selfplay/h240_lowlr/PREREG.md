# H240: lower-LR and longer-horizon r882 adapter continuation

Owner: luna_v1_res08. Authority: `hw_t0300_res08_h240_lowlr_rerun.md`, claim under
`~/jope_plan/ns/receipts/Hwave-admission.lock` in `H240_claim.txt`.
Nothing scientific runs before this preregistration merges. Offline checkpoint COPY only.

Hypothesis: smaller AdamW steps over the same or longer horizon yield a continuation that the strict
selector admits, improving DL-Val91 utility beyond r882. H240 is a new, fresh-seed rerun of H238, which remains CLOSED_INCOMPLETE after its assigned RAM guard. No partial H238 effect is treated as a scientific verdict. Prior H236/H237 observations motivated the original dose-down hypothesis.

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

Reuse H238 science source8c75d0d0dc54d4d89cf4b3f8153f7b74366da252 by exact copy; only fresh seed/fold bindings and process guard/recovery plumbing change. Reuse the H239 A1 monitor at1d31df4c235541c275b23ae7b9543ac3ec878ba5, adapting only its CPU mask to40–43. Native predictions, loss, sampler, optimizer rates/counts, selector and evaluation remain H238's. The explicit >=2/3 positive-seed clause is also enforced; it is already implied by a positive median over three seeds.
Parent runtime sources bind immutable Git objects. No shared/foreign checkout edits or new worktree/clone; the execution capsule binds all study bytes to the merged prereg commit. Own r882 COPY/reference are under `~/jope-data/h240/inputs/`, pinned by SHA256, never the live loop.

Fresh seeds240101–240103, family fold seeds240141–240143, bootstrap240150. Per seed: five TRAIN folds;
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
then the full wave. Three disposable smoke updates per fresh worker are separate from3,000 logical primary updates. A single primary campaign may use up to TWO source-bound infrastructure resumes after an explicit memory-guard hard stop, within the original deadline. No scientific retuning or new seed/fold draws. Unavailable/incomplete outcomes remain null.

On each completed cell, atomic JSON commits its selection/metrics/traces and all12 adapter snapshots. A resume validates the exact source, manifest, prereg and fold hashes, increments resume_number by exactly one (max2), and verifies each completed selection lock and adapter hash before copying it. Completed cells are reused byte-for-byte and NEVER re-trained or re-selected. Only not-yet-completed cells restart from the same r882 COPY with the identical deterministic seed/fold/minibatch stream; their prior partial work is retained in the previous attempt root. This is the H239 A1 no-re-roll mechanism. Every worker first re-verifies Native91 and the discarded TRAIN smoke. Logical final updates count unique completed trajectories; physical replayed work and unknown interrupted counts are reported separately, never invented as zero.


Primary: DL-Val91 regression utility, higher=-fitness is better; failed gates have utility-2.
Per seed average the five selected policies, then equal-family aggregate. Report the three seed
gains, median gain, shared10,000 family-bootstrap CI95 and one-sided p. Holm m=2 covers LR3E5 and
LR1E5_120 versus r882. Per-arm ADVANCE requires median gain>.002+eps, family LCB95>0, Holm p<.05 and >=2/3 positive seeds;
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

Host xbabe2 CPU, exact taskset40–43 (H239 owns48–51), nice19, one math thread, deterministic Torch, no GPU. Never xbabe0/xbabe3, another job's controls, or PR#2071. Admission requires load5<=100 AND MemAvailable>=30GiB. The supervisor requests a SIGSTOP-free cooperative pause when RAM<20GiB or load5>108; the worker waits at optimizer-step boundaries and safe pool/evaluation-batch boundaries, retaining its current state. Worker acknowledgements and supervisor transitions are logged. RAM resumes only after >=25GiB continuously for30s; load resumes<=100. Hard memory stop is <12GiB continuously for30s, or<6GiB on any sample. Other hard limits remain own supervisor/worker RSS>=4GiB, disk free<300GB, tree threads>4, and absolute compute stop2026-11-03T07:30:00Z. Stop events and every pause/resume are recorded. No SIGSTOP/SIGCONT is used, and no other process group is signalled.

Only memory-guard stops permit the two infrastructure resumes. Baseline/source/protocol/scientific failures, own RSS/thread or disk limits and the absolute deadline do not trigger a new worker. Admission waits and resumed workers never reset the deadline. Attempt roots and launch/guard receipts are exclusive-create and retained.

Prereg merged by04:00Z; compute STOP07:30Z; closeout merged by08:00Z, else CLOSED_INCOMPLETE.
Local gate: targeted data-free pytest, Ruff/check+format, diff-check, JSON/schema, exact fold/source/
registry checks and actual jope-records coverage validate. A configuration-only input projection is
used if the shared physical checkout is stale. Owner explicitly authorizes admin squash; remote
CI is NOT_A_GATE, record NOT_RUN honestly. Do not modify PR#2071.

Closeout: hypothesis/setup/outcome/metrics/verdict/next step in README; small metrics JSON, both
index rows and CLOSED claim line. Full output/source/receipts plus own starting COPY/reference go
create-only to versioned S3 under
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261103/H240/res08-t0300/`, with exact
VersionId byte/member replay. No large Git objects; raw roots retained.
