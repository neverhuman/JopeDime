# H104 — W7 acquisition, quality-diversity and replay preparation

This is the concrete CPU preparation requested by `rd_idle_t0455.md`, extending
[H69's design](../h69_w7_informative_acquisition_qd_replay/README.md) and the
[W7 proposal](../rd_breakthrough/PROPOSALS.md#w7--informative-acquisition-quality-diversity-and-replay).
It permits metadata materialization and a dry run. It does not authorize fitting,
model loading, new outcomes, GPU work, or a scientific advance claim.

## Training identities and folds

Reuse the exact H72 five-fold manifest (seed 4900), independently reconstructed
from its version-pinned training roster by the pinned H66 builder. It contains
1,219 DL-Train regression contexts, 205 declared families and 205 family/table
components. The committed `fold_manifest.json` is byte-identical to that S3
object. It is identity metadata, not a dataset or outcome table. The runner
refuses a family or physical table spanning held and fit folds, duplicate
contexts, incomplete counts, and label fields inside assignment records.

This establishes consistency with the inherited metadata, not current global
sealed-family exclusion or original-row provenance. Those remain launch gates.
No DL-Val, outer-test, sealed, pool array, table, checkpoint or score body is read.
Do not import H101/W6 outcome rows or the H102/W5 unproved v3 additions. V3 import
is disabled. Original dataset labels, DOPE SHA `c51a56d0…814`, and the r882
continuity checksum remain frozen.

## Arms and acquisition

| Arm | Four distinct new candidates | Replay recipe |
| --- | --- | --- |
| Current | Two minimum predicted-cost candidates, then two uniform draws without replacement | Family-uniform completed outcomes |
| Uniform | Four uniform draws without replacement | Identical recipe and dose |
| QD | Two minimum predicted-cost candidates, one novel-descriptor/disagreement choice, one uniform draw | Identical recipe and dose |

The uniform arm supplies the **fresh-label uniform-acquisition comparator**
required by H69's data-value gate. H69's shorter phrase “matched all-completed
uniform replay” did not fully specify its acquisition rule. This explicit
preparation choice must be accepted in the execution recipe freeze; it adds no
arm or fit budget. Winner-only replay is not a fourth arm.

`select_four` accepts only program SHA, predicted cost, disagreement and a
structural descriptor. It refuses outcome fields and nonfinite features. Ties
use program SHA. For QD's third slot, prefer a descriptor absent from the first
two choices, then larger disagreement, then program SHA. The final uniform slot
has probability 1/N over every remaining candidate. Log the ordered eligible
set and **conditional draw probability** at every slot. Deterministic slots have
probability one for the selected candidate conditional on fixed features; other
candidates have zero at that slot. These are not marginal inclusion propensities
or a basis for inverse-propensity claims about historical data.

The function is tested with synthetic feature-only offers; no real proposal
features or outcomes are evaluated here. Future predicted cost and disagreement
need fold-excluded model/source receipts. Disagreement is the absolute prediction
difference of two prespecified fit-fold models, never a hidden true-label error.
Their recipes and pins remain unbound. r882's current-policy comparison is a
continuity comparison because r882 saw all DL-Train families; it is not clean
held-family generalization.

## Finite archive and replay contract

The proposed archive has the same capacity (32 programs, at most two per
estimator-operation × AST-size bin) in every arm. AST-size bins are 1–8, 9–16,
17–32 and 33+. Use canonical vocabulary operation IDs. Only completed measured
outcomes may update archive quality; preserve original targets and evaluator/cut
identities. Within a bin retain larger measured utility, breaking ties by program
SHA. If the global cap binds, use deterministic round-robin over descriptor IDs,
ordered by program SHA within equal-quality ties. Never keep only an unbounded
stream of “winners.” Candidate construction costs and proposal source sets must
match the arms and be recorded separately from the four new evaluations.

The archive is not the full completed-outcome log. Retain that log, status masks,
selection receipts and all execution failures separately. Replay only genuinely
completed outcomes (including measured gate failures); do not infer status from
finite values or the `fitness=100, logratio=0` signature. This follows H88/H89's
critique of inferred completion masks. There is no fitted replay sampler or
materialized archive in this preparation.

Each cell uses 20 distinct fit families, one context per family, at most four new
records per family. Batch size is eight with at most one record per family per
batch; no recent record repeats within an epoch. Repeat deterministic epochs as
needed for the fixed 40 updates and record actual record/family exposure. If
common matching leaves fewer than eight families, the batch contract is
unavailable: report UNTESTED, not a smaller favorable subset.

## Fold schedule and cost accounting

For each held fold and seed, hash-order eligible fit families with the domain
`W7-family/seed/held/family`, take the first 20, then hash-select one context per
family with a separate domain. The same context schedule applies to all three
arms. Each round has a distinct recorded arm selection seed. No held-fold
context, family or table enters that fold's acquisition/replay stream.

Five held folds × 20 rounds = **100 context-rounds per arm and seed across all
folds**, not 100 per fold. Three arms × three seeds × 100 rounds × four new fits
= **3,600 planned new fits**, within the existing **4,800 fit / four CPU-hour /
two GPU-hour** W7 ceilings. The remaining 1,200 fits are unallocated, not free
permission for another grid. Timing/power pilots, retries, anchor regeneration,
confirmation and any additional prospective scoring must share these ceilings.
Existing four-anchor history requires provenance and a disclosed cost; it is
not four free new evaluations.

The 45 terminal acquisition learners and 45 common-learner replays would each
use 40 updates: 3,600 updates in total if both panels execute. This is planning
arithmetic, not runtime feasibility. Actual campaign consumption, remaining
budget, launch deadline, model recipes and initialization pins remain unknown.
This CPU preparation does not open or reset a study window. A cumulative ledger,
fit-only timing/power pilot and explicit deadline are mandatory before execution.

Terminal held-fold evaluation is an explicitly cached, fixed-candidate U4 track
only if its completed outcomes and original-row closure pass admission. Costs of
producing those cached observations remain disclosed. This plan does not reserve
fresh prospective terminal fits outside the 4,800 ceiling; switching strata
requires a prospectively budgeted protocol. No current outcome cache is admitted
by this metadata-only work.

## Common learner and scoring gates

Apply one frozen fold-excluded initialization/recipe to each arm's acquired
memory. Match counts by acquisition family × completed status without inspecting
utility: retain the minimum available count in each stratum across the three
arms, using program-SHA ordering, and report every matched/discarded/excluded
count. This matches exposure and status mix; it does not require the different
acquisition policies to have chosen identical programs. Match native-fit spend,
optimizer updates and replay schedule. Never silently turn a shortfall into a
zero or score only surviving seeds.

The dry-run panel contains **45 pending common-learner slots**, each with an
explicit status, matching-receipt pin and unknown matched/discarded/excluded and
common-support counts. Zero common support is UNTESTED. Missing receipts are
pending. No partial panel can advance. This addresses the matching-receipt concern
raised during H102 review and the false-ADVANCE subset failure found in H96;
the current runner never computes an advance verdict at all.

Two planned primary contrasts compare the QD-memory common learner against
uniform-memory and current-memory common learners. Use family means after
context/seed averaging, 10,000 family-cluster bootstrap draws (seed 20261031),
one-sided alpha .025 per contrast, lower bound > .001, median ≥ 0, and ≤10% of
families losing more than .01. A larger archive or HWM count alone fails the
causal data-value requirement. Report all 45 planned slots and incomplete cases.

Report family-grouped Bradley–Terry **Elo**, raw `val_reg_inner`, U4/U0 and fit
costs with a fixed complete contender panel; tie margin 1e-4, prior .01, 400
iterations, r882 continuity anchor 1500. Keep legacy and verified-search tracks
separate. The evaluator/Elo source, inherited priors, normalization and training
exposure must be pinned before scoring. No AUC is used. All metric fields in this
preparation are null, not zero. No serving promotion or HWM publication follows.
