# H219 — family-balanced accumulated adapter updates

Status: preregistration, no experiment or timing result. Owner authority is
`hw_t0400_astra_h219.md`, with host correction `hw_t0410_astra_xb3.md`.
All source, tests and execution use xbabe2. Merge this preregistration before
any GPU timing smoke or full run. One attempt, no automatic retry.

## Hypothesis and H109 amendments (before outcomes)

Large families may dominate context-uniform gradient updates. H219 executes
H109's never-run family-balanced accumulation hypothesis against native r882.
[H109 PREREG](../h109_family_balanced_accum/PREREG.md) is the historical contract;
[manifest.json](manifest.json) pins its bytes and runner. Changes are explicit:

1. The owner now authorizes GPU0 execution and create-only S3 custody. The old
   H109 preparation-only/no-ticket/no-S3 restriction is superseded for H219.
2. Seeds are 219101–219103; distinct whole-family partition seeds are
   219141–219143. H109's sorted-family permutation and round-robin assignment
   are retained. Each seed rotates all five held-out folds (164 FIT/41 held),
   producing paired out-of-fold results over all 205 families and 1,219 contexts.
   This replaces H109's single shared 164/41 split; without rotation the fresh
   seed rosters would not support a paired 205-family bootstrap. Exact context
   rosters and all fold-vs-fold Jaccards against H109/H215 are recorded in
   [fold_manifest.json](fold_manifest.json) and [fold_jaccard.json](fold_jaccard.json).
3. The primary is FBAL vs r882, median of the three equal-family seed gains.
   Holm includes FBAL and FBAL_WCR only; CONTROL is diagnostic. This supersedes
   H109's three-hypothesis family and seed-averaged family-mean estimand.
   Three positive seed gains are now required by the owner. The new owner
   criteria replace H109's separate nonnegative family-median gate.
4. H109 has no actual cell trainer. H219 adds the source-bound executor using
   pinned H146 loss/model loading and H177 native adapter math. H109's sampler,
   schedule, four-loss accumulation, clipping and checkpoint-copy helper are
   called directly. Only its accepted-seed whitelist is extended; no historical
   source is edited. Its old H109-only receipt schema cannot represent this
   new wave/fold inventory; H219 emits immediate immutable fit proofs, states,
   a global lock and final panels with exact-version S3 recovery.
5. A four-update CONTROL timing smoke uses stream 219001/outer0, FIT data only,
   no held-out scoring. It is separate from all scientific cells, resets to
   r882, and counts toward the same single 90-minute attempt (1,804 total
   optimizer updates including smoke). Its time estimate is diagnostic, never
   a reason to change seeds, arms, updates, thresholds or sampling.
6. Owner resource limits are MemAvailable >=20 GiB and free disk >=200 GB;
   free bytes replace H215's percentage disk cap. Retain one numeric thread,
   nice 19, RSS <6 GB, local output <2 GB, load5 <=85% effective affinity and
   separate 300-second overload clocks for load1/host CPU >85%. Swap-free and
   disk percentage are receipts only. FIFO uses `/home/ubuntu/qualification.lock`.
7. H109's DL-Train-only boundary remains: no DL-Val pool payload or outcome,
   sealed, holdout, outer-test or meta-test rows. Native91 is NOT_RUN for this
   reason. Shared feature-container headers/CID roster are metadata; the reader
   decodes only allowed TRAIN feature-row ranges. Cached costs must match full
   native r882 costs bit-exact on every TRAIN context before any adapter update.

## Matched training and inference

Arms: CONTROL (context-uniform), FBAL (family-uniform then context-uniform),
FBAL_WCR (identical FBAL sampler with warmup/cosine restarts). Sampling is with
replacement. Each of 3 seeds ×5 folds ×3 arms performs 40 updates, each averaging
four 16-context microbatch losses (effective batch 64). Thus full training is
45 cells/1,800 updates. Cell RNG seed = outer seed ×10 + fold index, paired
across arms; H109's sampler uses its unchanged seed mixing. Candidate-subset
RNG is SHA256-derived from H219/seed/fold/update/microbatch/slot/context, not arm.
Each context uses four best-fitness FIT reveals plus 20 random remaining
candidates, 24 total, matching the pinned H146 training loss recipe.

Each cell has a new isolated read-only checkpoint copy and a fresh optimizer.
All 47 adapter tensors/29,211 parameters start at exact r882 values, independently
copied into the model; they are not random initialized. AdamW LR 1e-4,
weight_decay1e-4, betas(.9,.999), global gradient clip 1. H109 WCR has two-update
linear warmup and restart period 40/3; no LR sweep. Eval mode/dropout off.
Pinned loss: SmoothL1 against fitness clipped [-4,2], failure cost 2, plus 0.1
KL term with temperature .2. DOPE composite c51a56d0 and every nonadapter tensor
are checked before/after cells; persistent source assets are rehashed at end.
Forty optimizer steps, 41 active/6 inactive gradients, nonzero gradient norm,
changed adapter hashes and a changed future FIT forward are required per cell.

Inference is full-pool first argmin of native adapter total cost. Utility is
`-fitness` when gate passes and `-2.0` otherwise. No filter, rerank, top-k change,
checkpoint selection, B tuning or multiple-proposal oracle. All45 terminal
models and FIT proofs are saved and globally locked/fsynced/S3-recovered before
any held-out utility evaluation. Each held-out family is absent from that
model's FIT roster. Cross-fit families can train other models; this is a reused
historical TRAIN development cohort, not sealed qualification.

## Frozen readout and decision

Aggregate context utilities equally within each family, then equally over 205
families for each seed. The primary point estimate is the median of those
three paired gains against r882. Draw 10,000 paired family-bootstrap samples
(seed 219150), use the same family draw across seeds and arms, recompute the
three-seed median each draw. Report two-sided 95% CI and one-sided 95% LCB.
Holm step-down covers only FBAL/FBAL_WCR, testing gain <=.001; report adjusted
p and corresponding rank-adjusted bootstrap LCB. To reject floating-point
false ties, tails count values <=.001+1e-12 and LCBs must exceed .001+1e-12.

FBAL advances only if its LCB95 and Holm-adjusted LCB both exceed .001,
Holm p<.05, all three seed gains are positive, and no more than 10% of families
have seed-average gain below −.01. Exact ties reject. FBAL_WCR has the same
reported gates but cannot substitute for failed primary FBAL. Report paired
FBAL vs CONTROL (and WCR vs CONTROL) diagnostically; no additional winning
hypothesis. Pinned H200 family-grouped Elo, seed 219151, anchors r882 at 1500;
Elo cannot override utility gates. U4 remains null: no verified multi-proposal
budget. No serving promotion follows this study.

The supervisor consumes one create-only attempt claim after merged-source,
GPU-idle and FIFO admission. Its absolute deadline starts before smoke/model
loading and is min(start+5400s,14:00Z on 2026-11-02); latest launch 12:30Z.
The worker cannot reset this clock. Any failure preserves partial evidence and
closes INCOMPLETE; no zero-filling, fallback, extra attempt or silent amendment.
