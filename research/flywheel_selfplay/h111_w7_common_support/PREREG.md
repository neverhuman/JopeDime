# W7 acquisition, QD and replay: receipt preregistration

This is the CPU preparation continuation requested by `rd_astra_t0608.md`.
It binds the existing [H104 recipe](../h104_w7_acquisition_replay_prep/DESIGN.md)
and [H111 matcher](README.md) to a concrete evidence manifest. It adds no arm,
fit allocation, training execution or measured result. The recipe is a proposed
execution contract: model/initialization pins, current data admission, cumulative
budget and deadline still require a separately accepted launch manifest.
W3 remains parked. H101's negative W6 readout does not admit its memories to W7.

[pending_manifest.json](pending_manifest.json) enumerates all 15 fold/seed cells
and 45 arm slots. [manifest.schema.json](manifest.schema.json) is a closed JSON
Schema for this **pending preparation** document, not an execution validator.
Its constant false admission flags prevent mistaking schema validity for launch
permission. Nulls mean unknown or unavailable, not zero. The source snapshot SHA
identifies the inspected main commit; source byte hashes and the real S3 input
VersionId identify the precise inputs independently of the later docs commit.

## Frozen design inherited from H104

Five family/table-grouped folds, seeds 701–703 and 20 distinct fit families per
fold/seed produce 100 context-rounds per arm/seed across folds. All arms share
these contexts and four new evaluations per round: **3,600 planned native fits**.
No held-fold family or table enters its acquisition or common-learner fit stream.
The inherited H72 identities establish fold consistency, not complete current
role/alias or original-row closure. Read no sealed, holdout or outer-test bodies.
V3 imports remain disabled. Dataset targets and DOPE stay unchanged.

| Arm | Acquisition slots | Replay and archive dose |
| --- | --- | --- |
| `current_2greedy_2uniform` | Two minimum predicted costs, then two uniform draws without replacement | Same limits and learner recipe |
| `uniform4` | Four uniform draws without replacement | Same limits and learner recipe |
| `qd_2greedy_1diverse_1uniform` | Two minimum predicted costs; one diverse/disagreement choice; one uniform draw | Same limits and learner recipe |

Uniform4 is H104's explicit fresh-label comparator interpretation, still requiring
acceptance in the launch recipe freeze. QD's third slot prefers a descriptor
absent from its first two slots, then larger absolute disagreement between two
prespecified fit-fold models, then program SHA. No measured outcome participates
in acquisition. This is the implemented H104 selection rule, **not** an assertion
that acquisition consults a populated QD archive. Archive retention below is a
separate design contract. Diversity-only/uncertainty-only follow-ups stay deferred.

## Required producer artifacts

Every object reference must specify exact bucket/key, non-null VersionId, SHA-256
and byte size; read it using that VersionId and verify all four. A URI, hash-shaped
string or successful latest-key download is insufficient. Publish under a fresh
W7 run prefix using create-only writes. Do not overwrite old keys or write an HWM
pointer for this preparation. Original run identities and failed attempts survive.

| Artifact | Required contents and checks before execution acceptance |
| --- | --- |
| Admission bundle | Authoritative completed-status source joins; original rows, cuts, dataset/context/program hashes, evaluator/vocabulary identity; current family/table aliases and global role exclusion; declared training-only input closure. A train roster alone does not settle these checks. |
| Recipe freeze | Acquisition scorer and disagreement-model source/weight/normalization pins, common learner initialization and optimizer/loss/step recipe, descriptor implementation and vocabulary, candidate generator, terminal scorer/Elo code and complete contender order. Each fit-fold dependency excludes that held fold; r882 is separately labelled continuity-only. |
| Acquisition ledger | All 80 attempted selections per arm/cell, including timeouts/errors/missing completions; fold, seed, round, context/family/table, ordered candidate-set hash and retrievable candidate-set object, program/source/parent hashes, slot, predictions, descriptor, RNG state/seed and conditional selection probability; completion receipt and outcome reference when present. Four distinct candidates per round; no replacement fit outside the cumulative budget. |
| QD archive | Ordered insertion/eviction events and final contents, with completion/outcome and descriptor references. Capacity 32 per cell/arm, at most two per canonical estimator operation × AST-node-count bin (1–8, 9–16, 17–32, 33+). Within descriptor retain larger measured utility, ties by SHA; when capacity binds use the H104 descriptor round-robin rule. Preserve all completed records separately. |
| Common support | Exact H111 matcher input/output, all 240 attempts per fold/seed across arms, every matched/discarded/excluded count by family and completed-pass/completed-gate-fail status. Match the minimum count across arms without utility selection. Fewer than eight common families is UNTESTED; incomplete census stays PENDING. A candidate match is not authenticated provenance. |
| Replay schedule | The full 40 × 8 ordered record indices per learner, epoch boundaries, batch/family/status counts, record exposure histogram and initialization/recipe pins. At most one record per family in each batch, at most four acquired records per family/cell, no record repeated within an epoch. Equal matched stratum counts, updates and exposure schedules across arms must be demonstrated; support count alone does not prove feasibility. Sampler source and deterministic epoch/tail policy must be pinned before outcomes are used. |
| Cost ledger | All attempted fits, retries, inherited-anchor production and proposal costs, CPU and GPU seconds, actual updates, host/core/nice settings, start/end/timeout, cumulative spend and immutable original deadline. Equal record counts do not establish equal fit cost. Record consumed resources even when the attempt fails. |
| Terminal scoring | Complete context/seed/arm panel at original precision, context-to-family map, U0/U4 and legacy utility, all missing cases, common-learner and acquisition-learner identities, paired deltas, harm numerators/denominators and Elo inputs. Six-decimal family means alone cannot replay per-context family-weighted Elo or a strict harm boundary. |

Probabilities are conditional on the ordered eligible set and features at that
slot: selected deterministic slots have probability one, and uniform draws use
1/N over the remaining set. These are not marginal inclusion probabilities and
do not justify inverse-propensity claims on historical replay. Archive membership
never substitutes for an all-completed ledger; unknown completion is not a measured
failure. Utility is consulted for archive quality only after genuine completion.

## Common learner and decision rule

Keep two primary data-value contrasts: QD-memory common learner minus uniform4-
memory common learner, and QD-memory common learner minus current-memory common
learner. Use the same frozen learner initialization/recipe, 40 terminal updates,
matched family/status rows, batch exposures and native-fit spend. If exact matching
or replay feasibility fails, preserve the failure; do not lower the batch size,
select surviving seeds, add fits or choose another learner after scoring.
The 45 acquisition learners plus 45 common-learner replays imply 3,600 planned
updates if both panels run. There are no actual updates in this preparation.

Report both contrasts, the full contender panel and separate legacy-selection
and verified-search tracks. Average seeds within context, then contexts within
family for utility contrasts. Keep all three seed rows together in the paired
10,000-draw family bootstrap (seed 20261031), one-sided alpha .025 per contrast.
Require adjusted mean lower bound > .001, median delta >= 0, and at most 10% of
families with delta < -.01 against **both** matched controls for a W7 advance.
A +5 Elo point gain is a practical target, never a replacement for this gate.
Report full-precision harm counts and denominators, raw val_reg_inner/U4 and
family-grouped Bradley–Terry Elo: prior .01, tie margin .0001, 400 iterations,
r882 continuity anchor 1500. r882 saw DL-Train and is not a clean held-family
control. A search-primary success is labelled a search result; archive size or
HWM count alone cannot establish data value. No serving promotion follows.

## Resources and pending disposition

The existing ceilings remain **4,800 native fits, 14,400 CPU-seconds and 7,200
GPU-seconds**. The unallocated 1,200 fits are not automatic spend. Pilots, retries,
confirmation and any prospective terminal scoring share those ceilings; terminal
cached outcomes also need admitted provenance and disclosed production cost.
Remaining resources and the original deadline are unknown here; this document
starts no second campaign and grants no GPU slot.

CPU preparation uses deterministic fold shards on xbabe1/2/3 and xbabe4 Linux
Docker only, one worker each, host load/headroom below 85%, nice19 and >=150 GiB
free on xbabe1, at least nice10 elsewhere. Never access xbabe0 or alter the live
flywheel. No model, dataset row or protected outcome payload is needed to validate
this manifest. The 45 slots stay PENDING, actual fits/updates/GPU are zero, and
Elo/val_reg_inner/U4 are null. Future runtime evidence must use its own schema and
accepted launch bundle rather than changing false flags in this preparation file.
