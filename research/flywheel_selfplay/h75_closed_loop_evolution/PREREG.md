# H75 — Closed-loop evolution preregistration

Status: preregistered design, not an executed result.

**Scheduling amendment — owner t2335:** after H57, the first of W1/fixed-D2
with signed CPU pre-review PASS takes the existing qualification lock; D2 is
HOLD until repaired. This supersedes the fixed t2320 order. H75 CPU work
continues, with no new GPU placement. Scientific settings, budgets and deadline
are unchanged. Test hosts are eligible xbabe1 or local xbabe2 JMAX<=8, not
xbabe3/xbabe4 for this pass. No active job is interrupted.

**Historical scheduling amendment — owner t2320 (superseded):** current lock order is **H57 → Fused D2
→ W1**; subsequent research order is W3, W4, W6, W5 (provenance hold), W7, W9.
This supersedes t2250's H75-before-W1 placement. H75 CPU preparation continues;
t2320 gives it no GPU slot. No active job is interrupted. Scientific protocol,
metrics, budgets and deadline below are unchanged. See the dated
[delivery record](../rd_breakthrough/DELIVERY.md).

Historical authority: owner decision
`owner_lift_t2250.md` moves **H57 → H75 → W1 → H71/W5** into the GPU queue;
fused D2/E stays CPU preparation only. ASTRA owns this preregistration and
publication; OPUS owns the CPU-built runner. There is no dependency on a W1
winner. This is an offline development experiment, not a live-loop change.

The owner-assigned H75 name collides with the earlier, unrelated
[H75 W4 cached shard](../h75_w4_cached_baseline_shard/README.md). Both records
are preserved: use `H75-closed-loop` and this full directory path in run IDs,
claims and receipts. Do not import that shard's metrics as H75 results.

## Motivation and falsifiable claim

[H56](../h56_audit/SUMMARY.md) recorded 1,478 promotions across 710/1,219
training contexts and oracle utility 1.872969 → 1.905853, while its snapshot
selected an appended program on only seven contexts. These are historical
observations, not fresh measurements. The current source already appends HWM
programs and includes the best 48 in training slates; the missing claim is
**useful feedback**, not literal absence of a feedback code path.
[H58](../h58/SUMMARY.md) shows that better overall ranking correlation need
not improve selection. [H71](../h71_w5_parent_child_critic/README.md) is
preparation, not evidence that evolutionary lookahead works.

Hypothesis: after DL ranks proposals and GP evaluates them, exposing the
completed evolved outcomes to a persistent adapter and its next search cycle
improves later selection. A larger oracle, more HWM events, lower training
loss, or faster execution alone cannot establish this. We require both
feedback-to-learner improvement and learner-to-data improvement.

## Frozen scope and input closure

- Regression, existing 1,219 DL-Train contexts / 205 families; five alias-resolved
  family folds, seed 4900; seeds 701, 702, 703. Reuse the admitted W1 fold
  manifest, SHA256 `dd211134bba127960b1612687478064969dca422e35dfa889790760f316961c1`.
  Assert fold membership, physical-table alias closure and exact context IDs.
- Source manifest SHA256
  `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`, selected
  SHA256 `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`.
  Verify pool, DSL, training-table, evaluator and feature hashes before use.
  Data must be admitted S3-derived inputs or their byte-verified local mirrors.
  The execution manifest must resolve their exact existing keys and versions;
  this preregistration does not assert unverified object locations.
- Frozen DOPE SHA256
  `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.
  Start all four neural cells from offline r882 SHA256
  `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`.
  r882 and inherited assets have prior training-family exposure. The study
  estimates **conditional development adaptation**, not clean pretraining-free
  generalization. Newly acquired fit-fold labels must never cross into that
  fold's scoring-family model, calibration, retrieval bank or pilot.
- Only original context training rows, `nq=0`, unchanged features and labels.
  Fresh inner cuts change fold assignment only. No sibling-table outcomes,
  outer-test rows, DL-Val, sealed or holdout payloads. No live xbabe1 flywheel
  reads. Mixed legacy score exports with test fields are not inputs. Asset
  loaders must select admitted training-only materializations before use.
- Every new outcome records program/context/family, original row and cut hash,
  parent/operator provenance, evaluator/vocabulary/protocol, status, gate,
  unrounded fitness, seed, selection policy and source object identity.
  Native errors/timeouts remain failures, not fabricated completed labels.
  Genuine completed gate failures are valid negative feedback; unknown or
  sentinel-shaped legacy outcomes without status are masked.

## Four matched cells and two search controls

All four cells start with the same r882 weights, static memory, four
hash-ordered historical anchors and identical budget. "Frozen memory" means
new results remain in a shadow log, excluded from future training, proposal
parents and policy history. They still count toward costs and the final
retained-best search accumulator. "Fresh memory" means same-context completed
results become available to the next cycle; never import a scoring family's
outcome into fit-family training.

| ID | Adapter | Feedback into later training and search |
| --- | --- | --- |
| A | Frozen r882 | Static memory; continuity search control |
| B | Frozen r882 | Fresh memory; search-only control |
| C | Persistent updates | Static memory; update-only control |
| D | Persistent updates | Fresh memory; closed-loop treatment |

C and D use identical optimizer, loss, step count, batch size, base pool,
feature inputs and training-history construction. No rollback, reload of
r882, warm-restart noise, best-checkpoint selection or held-fold gating between
cycles. A/B stay bitwise frozen. r882 remains an immutable comparator, never
replaced as serving policy by this experiment.

Search controls: **R**, uniform four without replacement; **G**, a fixed
program412 HistGradientBoostingRegressor trained on fit families only (96
hash-selected base candidates/context, target utility minus that fit context's
mean; max_iter=100, max_leaf_nodes=15, learning_rate=.05, l2_regularization=1,
random_state=seed). No W1-selected settings or context-cache dependency.
G is a fixed non-neural comparator, not a claim to be the best possible tree.
These controls use fresh same-context history/parents like B/D. Raw-argmin
legacy scoring of G is not a primary comparison because its centered scores
are not calibrated against known anchor fitness.

## Two cycles: propose, evaluate, retain, train, continue

For each fold select **four acquisition contexts from four distinct fit
families**, without using outcome values: sort families by
SHA256(`H75-closed-loop:fit:<fold>:<family_id>`) and select the first four;
choose the smallest context SHA in each. Freeze the full roster before native
calls. The four held-family prospective contexts use the analogous `eval`
hash inside that fold. No outcome-dependent replacement for slow, failing or
uninformative contexts. If fewer eligible families exist, refuse the plan.

For each seed, fold and A/B/C/D cell, run two cycles:

1. Generate eight distinct, previously unmeasured GP proposals/context using
   the pinned proposer. DL chooses two lowest-cost proposals, then two uniform
   samples without replacement from the remainder. Log candidate IDs, ranks,
   random seed and conditional inclusion probabilities. All cells have the
   same random stream; pools may diverge because feedback is the intervention.
   No unmeasured proposal may become a known parent or anchor. If fewer than
   four distinct proposals exist, record the shortfall; do not invent outcomes.
2. Evaluate four proposals on the original full training-CV envelope. Keep
   every completed result in the audit log. B/D admit completed positives and
   negatives to same-context memory; only gate-passing improvements over the
   pre-cycle known best by >1e-6 receive an **apparent original-cut HWM** flag.
   Flagging is not independent confirmation. A/C preserve the same results in
   their shadow logs. No shared mutable pool across cells or folds.
3. C/D take **40 AdamW steps per cycle**, lr=1e-4, weight_decay=1e-2,
   betas=(.9,.999), eps=1e-8, gradient norm clip=1, batch eight contexts,
   192-candidate slates. Optimizer state persists from cycle one to two.
   Each batch has two acquisition-roster contexts and six uniformly sampled
   other fit contexts (with replacement if necessary); C/D use identical
   context sequences. D reserves up to 48 unrevealed slots for hash-ordered
   completed fresh rows, including apparent HWMs; C fills the corresponding
   slots from its static pool. Remaining slots are outcome-blind uniform
   samples. Log exposure counts separately for fresh winners, completed
   negatives and static rows. A shortage is recorded, not silently padded
   with duplicate labels claimed as new observations.
4. Training anchors are the four fixed hash-ordered base programs, never
   fitness-sorted winners. Fresh HWM targets remain unrevealed. Loss is the
   current SmoothL1 + .1 KL at temperature .2, **over unrevealed slots only**;
   target is raw fitness for gate pass and +2 for completed gate fail. No
   decomposed auxiliary loss, new prior or transition head is added. Freeze
   unused parent_delta parameters; update only tensors reached by this loss.
   Use current live slate centering for every matched cell, zero base/SoS/context
   inputs and scale .4208 for continuity; all other frozen inputs are pinned.
   Training slate192, proposal slate12 (four anchors + eight proposals), and
   full-pool legacy evaluation remain separate identities. H62 measured small
   slate dependence; this is an accepted common contract, not invariance.
5. B/D's next search history uses the best four completed same-context programs
   available from past cycles (utility descending, program SHA tie-break),
   including evolved winners. A/C retain original anchors/parents. This is
   legal observed history; never reveal labels for unexecuted proposals.
   Persist cycle/optimizer/memory hashes so resume cannot restart from r882.

Log DL-selected parents/proposals and GP parent/operator edges. This is neural
guidance of AST evolution, not a neural autoregressive program writer. Cache
invalidation must rebuild all aligned feature/ID/outcome arrays when memory
changes; an append counter alone does not prove the new program was scored.

## Terminal evaluation and memory replay

No held-family selection results are opened until all cells and replay learners
are frozen. Cycle-one learning diagnostics use fit families only. There is one
terminal held-family read, not a best-of-two-cycle gate.

**Legacy transfer panel:** score A/B/C/D and two common replay learners on the
same original static pools for all 1,219 contexts, with four original
hash-ordered anchors, full-pool raw argmin and unchanged utility:
`u=-fitness` when gate passes, otherwise `u=-2`. A/B must agree in this panel:
fit-family memory alone cannot be passed into a held context's inference.
Record context-mean `val_reg_inner` and family-mean utility separately.
This panel tests whether feedback changes the learner beyond the searched
contexts, not whether the treatment received an easier candidate set.

**Common-learner replay:** from each fold/seed's B and D acquisition logs,
reset two copies to identical r882 weights and fresh AdamW, then run 40 steps
with the same recipe. Match new row count and completed pass/fail proportions
by deterministic hash subsampling within each status stratum to their common
minimum; report discarded rows and exposure. Keep all other static rows,
batches and update opportunities equal. No held-family outcomes in replay.
No common support or zero actual updates yields UNTESTED, not a pass. Compare
replay(D) against replay(B) on the common static terminal panel. This isolates
acquired-data value from continuing the original treatment's weights.

**Prospective search panel:** on four prespecified held contexts/fold, score
all six policies A/B/C/D/G/R. Freeze model parameters. Give each policy the
same four historical anchors and **two cycles of four new evaluations**;
only legal own-context search history can change. Thus this panel is U8 after
four disclosed history observations, not U4 or a cold-start four-fit test.
A/C use static history for proposal decisions while all policies retain the
best observed result in their final accumulator. Report cycle-one U4 and
cycle-two U8 without favorable endpoint switching. These held-context labels
are never replayed into adapters. Include all four model cells, tree and
random in a single fixed Elo panel.

After policy decisions freeze, evaluate each final selected program **and the
same initial best-known anchor** on three new deterministic inner cuts
(seeds 801,802,803) of that context's original training rows. Score every
policy, including nonwinners, on this common confirmation endpoint. Resplit
results cannot change choices, rankers or targets. Pair by full identity;
shared duplicate executions may be reused with explicit accounting, never
counted as independent evidence. Report both original-cut utility and mean
resplit utility. Original-cut monotonicity is not a guarantee on fresh cuts.

## Metrics, claim gate and interpretation

Report family-grouped Bradley–Terry Elo, with r882/A=1500, tie margin1e-4,
prior .01, 400 iterations, separately for legacy and confirmed U8 panels;
use repository `elo_tracker.py` pinned by source hash. Report unrounded raw
utilities and family W/T/L beside Elo. No AUC, Spearman, oracle gain or HWM
count substitutes for selection lift. Group contexts and average seeds within
family before inference; seeds/cuts do not multiply family count.

Seven prespecified raw family-mean contrasts form one family of claims:

1. Legacy D−C (value of feedback at matched updates).
2. Legacy D−A (actual improvement over r882 continuity).
3. Legacy replay(D)−replay(B) (better acquired data teach a common learner).
4. Confirmed U8 D−B (learning beyond search-only feedback).
5. Confirmed U8 D−C (feedback beyond static-memory updates).
6. Confirmed U8 D−G (neural system versus fixed tree search).
7. Confirmed U8 D−R (neural system versus random search).

Use 10,000 paired family bootstrap resamples, seed20261031, one shared draw
schedule within each panel, and Bonferroni one-sided alpha=.05/7. A full
closed-loop development success requires every contrast's adjusted lower
bound >+.001, median family delta>=0, and at most10% of families losing>.01.
Target +5 Elo over r882 in each panel as a practical effect, not as a
replacement for the raw-utility gate. Report the 2×2 interaction
`(D−C)−(B−A)` on confirmed U8 descriptively; it alone proves neither direction.
These conditional bootstrap intervals are not a guarantee against prior
adaptive development or overlapping fitted-fold dependence.

The prospective panel has at most20 distinct families and is explicitly
small; all205 families are used for cached transfer. Before any terminal read,
report a fit-only detectable-effect/runtime curve and all roster counts. Do
not expand the roster, extend training or add seeds after seeing terminal
results. An interval crossing the gate is INCONCLUSIVE, not proof of no effect.
If only oracle/U8 improves without transfer and replay, report SEARCH_ONLY.
If only legacy transfer improves, report ADAPTER_ONLY. Missing cells or
confirmation mean INCOMPLETE. Invalid closure or changed DOPE means INVALID.
Only the full conjunction supports CLOSED_LOOP_DEV_LIFT; none authorizes
serving promotion or claims untouched-test performance.

## Resource ledger, queue and stopping

This is an early bounded realization of the existing W9 causal question,
prioritized by the owner. Reserve against W9's existing ceilings rather than
silently adding a second identical campaign: **4 GPU-hours, 4 aggregate CPU-hours,
9,600 native calls maximum**, including pilot, failures, confirmation and
replay. W9's unused remainder is not an automatic second launch.

| Work | Maximum planned native calls |
| --- | ---: |
| Acquisition:5 folds×3 seeds×4 cells×2 cycles×4 contexts×4 calls | 1,920 |
| Terminal search:5×3×6 policies×4 contexts×2 cycles×4 calls | 2,880 |
| Confirmation:5×3×6×4×2 programs×3 cuts | 2,160 |
| Fit-only timing/identity pilot (no held outcomes) | 240 |
| **Planned total** | **7,200** |

The remaining2,400 calls are unallocated, not permission for extra arms or
outcome-selected retries. C/D use2,400 optimizer updates total; common replay
uses1,200; frozen controls use0. Stop at whichever resource limit or campaign
deadline arrives first: **2026-10-31T14:00:00Z**. Resume carries used resources
and the original deadline. Pilot may refuse an infeasible plan; any schedule
change requires a dated pre-outcome amendment, never silent downsampling.

All GPU work, including model inference during acquisition, holds the existing
`/home/ubuntu/jope_plan/qualification.lock` for its duration. Do not create
`~/qualification.lock` as an independent lock. Respect active ownership and the latest t2335 scheduling amendment above;
the former H75-before-W1 position is historical, not current launch authority. Preflight MemAvailable>=20GiB, free VRAM>=20GiB,
host CPU<=85%; nice15, OMP/MKL<=8, OpenBLAS<=1. Monitor actual utilization and
child CPU seconds; load divided by cores is only a capacity proxy. Stop new
admissions when guards fail, preserve completed evidence, never kill others.

Use eligible CPU hosts under [CPU_S3](../rd_breakthrough/CPU_S3.md), accounting
for current saturation and disk. x86-64 native evaluator SHA
`e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8`
must match; Linux/aarch64 Mac Docker is ineligible for that binary without a
separate identity amendment. No xbabe0, new bulk target trees, foreign cleanup,
checkpoint/dataset/receipt deletion or live-loop changes.

## Runner admission and durable outputs

OPUS's runner must prove before acquisition: no cross-arm/fold mutable state;
no scoring labels in fit/calibration/retrieval closures; exact row identity;
new-row feature alignment and actual target exposure; losses exclude revealed
slots; state survives cycle/resume without r882 reset; errors are not labels;
cumulative fit/CPU/GPU counters enforce caps; all selectors see identical
available budgets; confirmation cannot feed back. Small structural test
fixtures verify software only; scientific outputs use real admitted data.

Before launch freeze an execution manifest with source/runner/asset hashes,
prereg hash, exact input object keys+versions, training-only materializations,
fold/context rosters, all seeds, commands, feature/loss contracts, host shards,
ledger, deadline and resource guard receipt. Missing required input identity
blocks execution, not CPU runner preparation. Log actual optimizer steps,
gradient/parameter deltas, before/after state hashes, fresh-row loss exposure,
apparent and confirmed HWMs, winner selection and both cycle costs. A checkpoint
file or configured command alone is not evidence of learning.

Write create-only objects under
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H75-closed-loop/<run_id>/`.
The manifest records Elo/val_reg_inner/U4/U8 as null with reasons when unmeasured,
all source/input/output hashes and versions, costs, exclusions and verdict.
Verify downloaded bytes by SHA256 before publication. HWM pointers go to the
existing append-only campaign index only with exact comparison/cut/evaluator
identity and status (apparent versus confirmed); never overwrite a concurrent
writer's index. No HWM pointer is warranted for this preregistration.
