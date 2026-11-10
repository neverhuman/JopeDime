# H262 — strict acceptance replay (frozen before model reads)

Owner instruction: `hw_t1558_astra_h262.md`, 2026-11-03. This is an offline,
fixed-pool mini-flywheel on a private read-only r882 copy. It is not a live-loop
intervention or GP program-generation experiment. H255 metrics supplementation
is receipt-only, separately published, and supplies no H262 observations.

## Hypothesis and CURRENT source truth

A paired family-bootstrap margin gate may protect the incumbent against noisy
adapter admissions and improve final DL-Val utility versus r882. A result only
against CURRENT is an acceptance-rule finding, not checkpoint lift.

Read-only process/source capture at 2026-11-03T15:57:54Z found xbabe1 PID 1402163
running `research/flywheel_selfplay/run_fused_closed_loop.py` in
`/home/ubuntu/JopeDime/.agent/worktrees/flywheel-gp-fused`, commit
`486b709c3ee84a351a2cd7b55ef0290017e6ecd0`. Both read source files matched Git
bytes at that commit. No writer, screen, checkpoint or run-output mutation.
The running command omits `--strict-adapter-admission` and `--adapter-min-gain`;
the opt-in switch defaults false. `DevIncumbent.decide`,
`research/flywheel_selfplay/dev_incumbent.py:641–685`, notably line 655:

```python
accepted = value > old + self.min_gain if self.strict_admission else value >= old
```

The `_mean` method at lines 593–597 refuses nonfinite values. On acceptance,
lines 656–662 retain model and optimizer state; otherwise lines 663–664 restore
both. `run_fused_closed_loop.py:492–512` scores hard argmin, valid utility
`-fitness` and invalid utility `-2`, with a context mean; lines 625–633 wire
strict options into the incumbent and lines 1180–1187 invoke `decide`.
[Source receipt and verbatim methods](current_rule_source.json) freeze the bytes,
commit, process arguments and citations. The installed optional strict-margin
feature is not the CURRENT policy in this observed process. Program-pool
promotions and neural adapter admissions are distinct mechanisms; a program
promotion rate is not evidence of adapter improvement. H201 already examined
strict admission; this wave adds a paired sequential offline trajectory test,
not a claim to be the first acceptance-rule investigation.

## Arms and trajectory construction

- CURRENT: finite candidate context-mean FIT-accept utility >= incumbent mean;
  ties accept; worsening restores incumbent adapter and AdamW state. This ports
  the observed live adapter comparator to FIT-accept, not its original DL-Val
  selection panel or all live-loop novelty/projection mechanisms.
- STRICT_LCB: equal-family paired candidate-minus-incumbent FIT-accept utility;
  2,000 paired family resamples, percentile CI95 lower endpoint strictly >
  +0.002; ties and NaN reject. Bootstrap seed is
  `stream_seed(262150, seed, round_index, "accept")`.
- NEVER: unchanged r882 reference, no optimizer updates or acceptance decisions.

Seeds 262101–262103; 10 rounds per CURRENT/STRICT_LCB trajectory, one candidate
per round, 40 exact H170 CONTROL AdamW updates from that trajectory's current
incumbent. Exactly 2,400 primary updates and two separate one-update smoke
probes. AdamW lr1e-4, betas(0.9,0.999), wd1e-4, eps1e-8; global gradient clip1;
raw H146/H170 smooth-L1 + 0.1 ListNet KL (temperature0.2), valid training cost
clamp[-4,2], invalid cost+2. No loss, learning-rate or candidate-budget tuning.
Fresh AdamW at r882 for each trajectory, accepted optimizer state carries
forward and rejected state rolls back, identically for both rules. This is a
bounded continuation protocol, not a replay of the live optimizer's old step479
state. Adapter only: 47 tensors /29,211 coordinates; DOPE and anchor frozen fp32,
no dropout. DOPE composite c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814
verified before/after and during updates. r882 file SHA256
0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64.

Shared metadata-only partition seed262140 shuffles sorted 205 DL-Train family
IDs; floor(0.7*205)=143 FIT-train families /886 contexts; remaining62 FIT-accept
families /333 contexts. [Exact split](fold_manifest.json). All91 DL-Val contexts
/19 families remain outside training and acceptance, with family disjointness
asserted. No has_meta_test, sealed, holdout or outer-test rows. Real S3 catalog
and inherited input provenance are pinned in [provenance.json](provenance.json).
Known historical provenance/version and prior r882 selection limitations remain;
these exclusions do not retrospectively qualify r882's history.

Per round, `batch_plan(..., stream_seed(seed, round_index))` is the exact H170
8-context, top-fitness-four-reveals plus20-sampled-candidates recipe for40 steps.
Both rules have identical sampler and numeric seeds at each round index; batch
plan hashes must match. Realized candidates may differ after rule trajectories
diverge, so this is a paired policy-trajectory experiment, not identical trained
candidate bytes. The gate package changes threshold, uncertainty treatment and
context-versus-family weighting together; it does not isolate one component.

## Frozen evaluation order and reporting

After published preregistration, Native91 first: exact91/91 r882 utility vector
and mean1.1377941898164525, with native/cached-forward bit equality. Baseline
TOP4_PICK must reproduce H258 rounded mean1.240360 within5e-7. Then two smoke
updates, changed adapter/frozen DOPE checks, exact r882 restore, fresh optimizers.
No trained DL-Val scoring during smoke, rounds or acceptance decisions.

All60 candidate snapshots, acceptance cost vectors, decisions, training traces
and sampler hashes are retained. All six final adapter files are fsynced and a
global final-lock with hashes is fsynced (including its directory) before ANY
trained DL-Val scoring. Each final CURRENT/STRICT_LCB incumbent is scored once
on DL-Val (six model evaluations across3seeds); NEVER reuses the one initial
Native91 vector. No intermediate checkpoint DL-Val scoring. TOP4_PICK derives
from the same final cost vectors without additional model evaluation.

TOP4_PICK follows H258: stable ascending predicted-cost order; outcome-free
(top2 gap)/std(cost) <0.25 enables four reveals, otherwise choose top1 with zero
reveals. Among the four, select minimum raw gated cost (fitness if valid, else2),
with ties resolved by predicted-cost order. Degenerate std<=1e-12 gives gap999.
This is test-time true-evaluation search, reported separately from single-pick
checkpoint utility. Each final uses its own outcome-free gap; budget incidence
may therefore differ. Both gate/reveal counts and chosen IDs are retained.

The request for accept-then-DL-Val-worse rates cannot identify per-round harm
under the explicit final-only scoring limit. Preregistered report: per rule,
number of trajectories with >=1 acceptance and final equal-family DL-Val utility
below r882 divided by trajectories with >=1 acceptance. Zero denominator is
null. Per-accept/per-round DL-Val harm is NOT_MEASURED; no extra scoring is
introduced to fill it. Accept counts per seed/rule, final L2 drift from r882,
harmed family IDs (delta <−1e-4), family W/T/L and MM Bradley–Terry Elo anchored
at r8821500 are reported. No AUC, speed or tokens-per-second verdict.

## Statistical decision and limits

Primary: STRICT_LCB final single-pick minus r882. Secondary: STRICT_LCB final
minus CURRENT. Median across three seed-level equal-family gains. 10,000 paired
family-bootstrap draws over the same19 DL-Val families, seed262150, paired across
seeds and arms. Holm m=2 one-sided bootstrap p=(1+count(boot<=0))/(10000+1).
For Holm lower confidence bounds, order the two raw p values (primary first
on exact p ties): first uses percentile .0125 (97.5% two-sided interval);
second uses .025 only when preceding Holm p<.05, otherwise .0125. Report ordinary
CI95 as well as this step-down lower endpoint explicitly.

ADVANCE requires median family gain >+0.002, Holm LCB95>0, Holm p<.05 and3/3
positive seeds; exact ties fail. Primary determines checkpoint ADVANCE. A passing
secondary alone is an "acceptance-rule finding" with checkpoint NO_ADVANCE.
CURRENT-minus-r882 and TOP4_PICK panels are descriptive, not extra admission
opportunities. Elo:400 MM iterations, pseudo-win prior.01, context tie threshold
1e-4, family-balanced games;1,000 family bootstraps (seed262151).

Intervals condition on fixed learned policies and an exposed development panel;
three seeds share the fixed family split. Holm corrects two contrasts within
this wave, not the adaptive campaign. Repeated FIT-accept use can itself overfit;
only final evaluation is separated. Identical fallback predictions are identity,
not independent evidence or equivalence. No serving promotion follows this run.

## Resources, publication and custody

xbabe2 only, CPU48,50-52 /nice19 /numeric1 /aggregate threads<=4 /RSS<4GiB,
free disk>=300GB. Load5 admission<=100, pause>108, resume<=100. H239-A1 admission
RAM>=30GiB, cooperative pause<20GiB, resume>=25GiB sustained30s; existing hard
safety guards<6GiB immediate and<12GiB sustained30s remain. Release temporary
arrays, collect garbage and malloc_trim between rounds. Never control other jobs,
use GPU, xbabe0/xbabe3 or write to xbabe1. Process restart/post-outcome retries=0.
Prereg deadline16:40Z, absolute compute stop18:15Z, closeout19:00Z, 2026-11-03.
Resource waits consume these deadlines; incomplete runs close honestly.

Publish prereg and closeout only by non-force push to origin/v1/next. Targeted
pytest, Ruff, JSON/schema/pin checks and same-agent exact-head review. Hosted CI
NOT_RUN per CI_FREEZE split; no main/fastlane/PR2071/global schemas/records pins/
root README/CI config edits. The metrics schema is study-local only.
S3 create-only prefix `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H262/astra-t1558/`;
retain VersionIds and SHA256, verify archive member recovery before removing
local bulk. No .pt/.npz objects in Git. Closeout includes README, compact metrics,
manifest custody and both index rows; final claim and comms handoff.
