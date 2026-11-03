# H236: Schedule-Free AdamW for the r882 adapter continuation

Owner luna_v1_res08. Authority: `/home/ubuntu/nudges/hw_t0005_res08_h236_sfadamw.md`.
Claim `~/jope_plan/ns/receipts/H236_claim.txt`, under Hwave-admission.lock.
Preregistration precedes model construction, Native91 and training. Offline COPY only.

Hypothesis: the 40-update plateau partly reflects schedule tuning; SF-AdamW's averaged iterate improves
DL-Val91 regression utility. The only scientific intervention is the optimizer.

| Arm | Optimizer | Base LR | Role |
|---|---|---|---|
| r882 | frozen, zero updates | — | reference |
| CONTROL | exact H170 CTRL AdamW, betas .9/.999, eps 1e-8, wd 1e-4 | 1e-4 constant | paired control |
| SF | Schedule-Free AdamW, beta1 .9 interpolation, beta2 .999, eps 1e-8, wd 1e-4 | 1e-4 | primary |
| SF_LR3 | identical SF optimizer | 3e-4 | secondary; Holm m=2 |

SF uses 4-update linear warmup, r=0, weight_lr_power=2, no inner momentum. Decay is applied at y after
adaptive normalization, following the official scalar path at revision
70785b53e778d0e872c0bbb75ff4ee54ee10c291. Its two buffers (z and second moment) match AdamW's two-buffer
memory order; no third iterate is retained. The model parameter stores y for gradients and swaps to averaged
x BEFORE every trained scoring/selection/snapshot. Restored snapshots are already x. No-state update0 swaps
are exact no-ops. None gradients receive neither update nor decay; inactive tensors remain in drift inventory.
The implementation and independent explicit-x/y/z toy recurrence are tested before any study execution.
Sources: [Defazio et al. 2024](https://arxiv.org/abs/2405.15682),
[official scalar implementation](https://github.com/facebookresearch/schedule_free/blob/70785b53e778d0e872c0bbb75ff4ee54ee10c291/schedulefree/adamw_schedulefree.py).

Reuse the merged H235 runner/analysis from c927afafc5a78e1b479988b6ac6c7ba21add5272 and pinned H146
runtime at 8aa627bad4cb7855916bd476e2f60d81576fc772. Keep H232 A1 None handling and A2 namespace fix.
All arms use the unchanged H170 CTRL Huber+0.1KL loss, grad norm clip 1, and H156 paired minibatches:
8 fit-C contexts, 24 candidates (4 reveals +20 sampled). Only 47 adapter tensors /29,211 parameters train;
there is no variance head. DOPE c51a56d0 and every non-adapter byte are checked after every update/evaluation.

Fresh seeds236101–236103, family-fold seeds236141–236143, bootstrap236150. Five outer TRAIN folds per seed;
each cell fits on3 folds, selects on next1 and assesses remaining1. Exactly15 cells ×3 arms ×40updates=1800
primary updates, plus3 disposable TRAIN-smoke updates. Snapshots0/10/20/30/40. Earliest numerical tie wins,
including r882; accept a selected step only if selector equal-family gain>.001+1e-12 and CI95 LCB>0,
otherwise restore r882. Durable selection lock precedes any trained outer-A/DL-Val assessment. Every trajectory
starts fresh from r882; TRAIN smoke is discarded with bit-exact restoration and fresh optimizers.

Run order inside ONE exclusively claimed, guarded attempt: Native91 baseline first, requiring all91 utility
values exact and mean1.1377941898164525 plus cached/native forward bit equality; then one TRAIN fit-C smoke
batch for each arm, each1update with x scoring, finite/change/frozen checks and exact r882 restoration;
then the full15-cell wave. Smoke cannot tune or change arms. No scientific retry or resumed partial attempt.

Primary panel DL-Val91/19 families. Per-seed five selected policies are averaged, then equal-family utility;
report median three-seed gain and shared10,000 family-bootstrap CI95. One-sided bootstrap p is Holm-adjusted
over SF and SF_LR3 against r882. Per-arm ADVANCE requires median gain>.002+1e-12, family LCB95>0, Holm p<.05;
ties reject. Report primary SF verdict separately; aggregate ADVANCE identifies any qualifying treatment,
so a secondary result never changes the primary SF verdict. Against CONTROL is descriptive. Report canonical
family Bradley–Terry Elo vs r882 (anchor1500) and CONTROL, bootstrap CI, positive seeds, harms, drift and exact
duplicate prediction/adapter groups. Secondary TRAIN OOF1219/205 and unconstrained-best diagnostics follow H235.
Compare each seed's DL-Val context mean to plateau best1.1389572 (r3175), descriptively. No AUC/speed verdict.

Real S3-origin cohort per `~/jope_plan/master/source/S3_DATASET_CATALOG.md`; 276 dataset IDs join origin=real.
Pinned H50 selected/source manifest and pools; H58 context cache. TRAIN1219/205 plus DL-Val91; reject
has_meta_test=True, non-regression, wrong source split or family overlap. No sealed/holdout/meta-test/live-loop
reads. Read-only own r882 checkpoint COPY and reference under `~/jope-data/h236/inputs/`.

Execution binds all own source bytes to the merged prereg Git commit and parent sources to immutable Git
objects, without editing shared/foreign checkouts. The source capsule is not another clone or worktree.
The unchanged H146 runtime uses the canonical Git object database for its pinned DOPE guard only.

Host xbabe2 CPU, exact taskset40–43, nice19, one math thread, deterministic Torch, no GPU. Never xbabe0/xbabe3;
never signal another job. Own supervisor/worker RSS<4GiB, MemAvailable>=20GiB, free>=300GB, threads<=4.
Admission load5<=55; pause own worker process group when load5>60, resume<=55. Hard deadline2026-11-03 06:00Z.
Prereg merged by01:30Z; closeout merged by06:30Z, else CLOSED_INCOMPLETE with unknowns null.

Local gate: targeted data-free pytest, Ruff/check+format, diff-check, JSON/schema parsing, exact source/fold/
registry verification and jope-records coverage validate. Owner explicitly authorizes admin squash and makes
remote CI NOT_A_GATE for this H236 research publication; record hosted CI NOT_RUN honestly.

No large Git objects. Full outputs and source/receipts go create-only to versioned S3 under
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261103/H236/res08-t0005/`, with SHA256 and exact
VersionId byte/member replay. Preserve raw roots, incomplete/null outcomes and source authorship.

Status: PREREGISTERED; no H236 model/data/scientific execution yet. Metrics remain null.
