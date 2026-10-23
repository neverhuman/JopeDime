# H20 frozen execution and result contract

Status: completed 12 fresh adapters/2400 updates; six existing bases reused with
zero base fits. This is an archived plan plus actual disposition, not an order
to rerun the closed study. Independent later replay passed with limits, while
the original metadata-failure closure remains unchanged. Read results.json and
curves.json with this plan.

## Objective and fixed controls

Test one fixed wider state parameterization motivated by TRAIN-only H19 geometry.
Use B=1.0 times the original TRAIN score-scale instead of 0.1; no K/bound search.
Reuse the same six H18 bases, original fold-TRAIN scalers, score-scales, 412 program
features, 240 common context features/128 zero padding, 112 state coordinates and
three fixed seeds. Train matched trained/random adapters from identical zero
initialization, seed+1000000, 8993 learned parameters. The base, its buffers and
failure logits remain frozen/eval. No GP, baseline or native Dope refit occurs.

Loss is unchanged full-slate uncapped expected regret/original TRAIN fitness scale
plus 0.1 minimum CE and 0.1 failure BCE. The latter is constant for a frozen base.
Use AdamW 0.0003, weight decay 0.1, clip 1, FP32 CUDA, 200 updates/model, 128 queries/update,
16 query microbatches and exact archived sample order. There is no true-gate mask,
shortlist, cost clipping, calibration, early stopping or selected ensemble.

Widening changes both feasible score range and the initial derivative of score
with respect to adapter logit. At identical initial parameters that derivative
is 10 times larger; adaptive AdamW/clipping/saturation mean the later update path
is not a simple 10 times parameter step. H20 versus H18 therefore tests a wider
parameterization with changed optimization conditioning, not range alone.

## Data isolation and evidence

Each FIT worker opens only its own fold TRAIN pools with FIT/SELECT source roles.
It checks membership before pool hashing/cache/open. It loads saved scalers and
feature-only covariates, never opposite fold candidate outcomes during fitting.
TRAIN 32 snapshots 0/100/200 are fixed own fold monitors with dropout disabled.
All four three-seed completion markers and 12 model/recipe/source identities must
close before full opposite fold OOF evaluation. Six frozen-base and 12 adapter final
inferences then cover 850 contexts/162 components/979200 unique candidate rows per
arm/seed. Seeds and repeated draws do not multiply independent source volume.
CAL 118, OLDDEV 211 and known reservations remain excluded.

Primary: mean each seed's selected raw fitness/context, then equal contexts per
component, then equal 162 components. Both trained-minus-base and trained-minus-
random must have wholly favorable intervals. Use 100000 paired component bootstrap
draws, seed 20261022, quantiles 1/160 and 159/160; 98.75% each/97.5% Bonferroni familywise.
Both intervals include harm. Secondary gate/logCV/regret/top-k or descriptive
wide-minus-narrow results cannot replace the failed primary criterion.

The independent safe-weight replay checked all 12 adapters/six bases, matching
initials/samples, frozen base tensors, saved TRAIN scalers, all 200 updates, complete
source pools, final scores/choices and uncertainty. The original verifier refused
before replay because its generic source map lacked a redundant covariate key.
That same hash was already in the frozen pre-fit reuse manifest and checked by
the FIT loader. A later amendment admits that pin; 13 scientific ASTs and all fitted
models are unchanged. Keep original FAIL/CLOSED_INCOMPLETE and later PASS separate.

## Resource, reconstruction and handoff

Human compute policy removes time deadlines/RuntimeMax/forecast refusal while
retaining CPU 4 aggregate, child RAM 8 GiB, swap 0, Tasks 64, CUDA fraction 0.30, physical RAM 4 GiB/
VRAM 2 GiB reserves,temperature 85 C/host CPU 70/evidence 4 GiB guards. Costs include 2400
dataset updates, two verified CPU synthetic updates and an initial unobserved
CPU synthetic range 0 to 1; GPU synthetic updates 0. Preparation refusals and the
original failed QA remain recorded. Timings in results.json are observations,
not measured GPU-active time or an erased admission failure.

Use source/receipt digests in results.json and per-run digests in curves.json to
request the exact protocol, reuse manifest, safe checkpoints, sample schedules,
optimizer logs, fit/evaluation seals and independent replay from the root evidence
custodian. Bulk arrays, model tensors and private identities stay outside Git.
No current unknown source commit, object VersionId or remote durability is invented.

A future implementation worker reads the source contract, all archived outcomes
and exact role/source/capability evidence before preparing a distinct protocol.
It preserves the existing source and controls, verifies exact hashes and canonical
ties, independently replays the final comparison, and records all failure/missing
states. Do not relaunch H20 or choose a new bound using its heldout results.
Product integration follows the current serial owner, exact-head CI/review/guarded
merge and independent Git-custody restoration. This staging record grants no
product index/ref/CI/runner ownership and claims no serving promotion.

Highest-value next DATA work is H21 source/membership/current family-role/scaler
qualification; an archive census alone does not supply new independent families.
