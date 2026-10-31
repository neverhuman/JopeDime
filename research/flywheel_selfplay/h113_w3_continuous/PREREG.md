# H113 — W3 continuous outcomes and top-resolved loss, CPU preparation

Owner `rd_astra_t0640.md` resumes W3 preparation after the W7 handoff. This
extends [H65](../h65_w3_continuous_top_resolved/README.md) and
[H107](../h107_w3_unblock_contract/CONTRACT.md). It implements opt-in CPU loss
primitives; it is not a trainer, result, accepted W1 winner, or launch authority.
[manifest.json](manifest.json) binds source identities and all 15 fold/seed cells.
W1 prior, row admission, calibration, control models and resource receipts stay
explicitly pending. No native fits, optimizer updates or GPU work occur here.

## Registered panel and inputs

Keep the existing W3 panel: r882 continuity; a matched current SmoothL1/KL;
b matched H58-D; c top-resolved listwise; d quality/feasibility heads; and
width/compute-matched scalar and shuffled-auxiliary controls for d. The two
primaries are **c−b and d−a**, not a best-of-four selection. Seeds 701–703,
five H65/W1 family folds (seed 4900), 40 terminal updates and the original
history/prior contract stay fixed. The six trained arms imply 90 planned cells
and 3,600 updates if all run; r882 adds 15 continuity readout slots. This is
planning arithmetic, not demonstrated runtime feasibility or extra allocation.

Use the H65 fold identity, not H72's differently identified partition. The
metadata lists 1,219 DL-Train contexts / 205 families; its alias counters are
historical declarations, not current global exclusion proof. No held-family,
physical-table alias, outer-test, DL-Val, sealed or holdout row enters fitting.
DOPE stays frozen at `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.
Original dataset labels and the original evaluator/gate/utility are unchanged.

H59 is completed dependency evidence: all four arms were unadmitted, with CLIP
less harmful but still below r882. See the [published H59 result](../../../docs/research/studies/dope-neural-ranking-20261021/h59/README.md)
and [H89 disposition](../h89_w3_completion_admission/README.md). Do not inherit a
new loss clipping or normalization default from H59 or relabel it unrun.

Before importing actual numeric rows, H107 requires authoritative completion
joins, original rows/cuts/evaluator/vocabulary, a complete census, current role
and family/table/alias closure, and exact-version allowlisted objects. H86/H87
numeric signatures are not completion evidence. H113's row validation checks
shape and consistency only; it cannot authenticate producer evidence. No actual
W3 outcome export is admitted or read in this preparation.

## Loss implementation and flags

[`losses.py`](losses.py) exports `w3_loss(..., enabled=False)`. Disabled calls
return `None` without touching inputs or importing Torch. An explicitly enabled
call accepts CPU floating tensors and a rectangular list of already-admitted
training-slate rows. GPU tensors are refused. There is no optimizer, checkpoint,
network client, dataset loader, model construction or execution command.

Each row has exactly `status`, `fitness`, `logratio`, `gate`, and `revealed`.
Statuses are completed-pass, completed-gate-fail, timeout, error or unknown (the
literal underscore names are in the code). Revealed rows are masked in every
arm, preserving one matched history interface. Missing numeric components stay
masked; timeout/error/unknown require null supervision. Completed gate failures
can supply measured logratio and gate=false but never a reconstructed passing
residual. A gate flag is not inferred from a finite penalty or sentinel.

| Arm | Primitive implemented |
| --- | --- |
| a | Mean SmoothL1(beta=1) plus .1 KL with temperature .2 over each slate's eligible unrevealed targets; original passing fitness or H58 failure cost 2 |
| b | H58-D dual-temperature listwise KL plus .50 gate-margin and .25 pair-margin terms; passing standard deviation floor .05, regrets clipped [0,6], failure regret 8 |
| c | Same b objective and margins, replacing only the passing scale by max(F_(min(16,npass))−F_(1), .05) |
| d | Sum of measured-component SmoothL1 for logratio, SmoothL1 for passing fitness−.9×logratio, and gate BCE-with-logits; each component averages within slate then across slates containing it |

For b/c the target temperatures are .10/.35 with weights .7/.3; prediction
temperatures are .15/.40. The inherited margins are .50 for the best passer vs
failures, .25 for the mean top-eight passer vs failures (inner weight .5), and
.15 for strict top-four vs bottom-half passing comparisons. b is checked against
H58's extracted function for both loss and gradients on complete slates. The
new masked-cohort behavior is not a claim of historical H58 replay.

For c, fewer than 16 passing rows use the last passing order statistic; a single
passer uses the .05 floor. With no passer, all eligible failures get regret 8:
the listwise target is uniform and no pass/fail or passing-pair margin exists.
Such a slate does not select a useful winner. Entirely unlabelled slates do not
contribute; if nothing is supervised the return status is `NO_SUPERVISION` with
a differentiable zero, which a future trainer must record and refuse as an
optimizer update. Component-count receipts expose the effective supervision.

d's equal component weights and SmoothL1 beta=1 are concrete CPU-prep defaults
to freeze in the future accepted recipe, not evidence of optimal calibration.
Its heads do not yet define a selection score. Fit-family-only calibration must
map them to expected **original** utility and must preserve the unchanged gate;
no confidence-bound interpretation or near-pass promotion is permitted. Network
heads, width/compute-matched scalar and shuffled-auxiliary model construction,
calibration, batch/family weighting and optimizer integration remain unimplemented.
Keep the controls in the panel; do not claim the d−a experiment is runnable yet.

## W1 prior binding and zero-update contract

`compose_cost(prior_cost, residual_cost)` takes the externally supplied W1 cost
as a detached, fixed tensor and adds a same-shaped residual. A zero residual
preserves the prior's values and stable ordering exactly. No loss function
mutates this prior or changes inference/evaluation units. The d auxiliary loss
has no inference mapping until calibration exists; the composition test alone
is not d's zero-update model proof.

[`prior_binding.schema.json`](prior_binding.schema.json) describes the metadata
handoff once a W1 result is reviewed: exact result/review/recipe object references,
winner ID and claim scope, H65 fold identity, frozen DOPE and five held-fold prior
objects with training-component hashes. JSON shape is not acceptance. Verify all
versions/bytes, fold-excluded dependencies, matching history/normalization, source
and weights, and the result review before binding. Keep `w1_prior_binding=null`
until this exists. A search-only W1 claim must not silently become a legacy W3
winner; record the explicit compatibility disposition. No checkpoint is loaded
here and no fallback W1 arm is selected while H108 remains pending.

## Readout and admission

W3's primary is legacy-selection family raw utility; U4 is a separate reported
track. Keep all seed rows and the complete family/contender panel. Use 10,000
paired family-cluster bootstrap draws, seed 20261031, one-sided alpha .025 for
each of the two primaries. Require adjusted family-mean lower bound > .001,
median delta >= 0, at most 10% of families losing more than .01 and no excess
original gate-failure rate versus the matched control. Report per-seed utility
and Elo, family harm counts and CIs, not only pooled gains. Family-weighted
Bradley–Terry Elo uses r882=1500, prior .01, tie margin .0001 and 400 iterations
on a fixed complete panel. r882 is exposure-biased continuity, not clean OOF.

Kill auxiliary-only gains, gains matched by shuffled auxiliaries, or gains from
changed evaluation semantics. Width/compute matching and zero-update full-model
choice invariance must be demonstrated before the complete study can advance.
Use [H74's readout contract](../h74_w3_readout_gate/README.md) after upstream
admission; a readout gate does not authorize training. Preserve original-precision
per-context/seed outputs for Elo and strict harm-boundary replay, including
incomplete attempts. No partial-seed winner selection or serving promotion.

## Budget and CPU proof

The existing ceiling is two GPU-hours and **zero new native fits**. Original
cumulative spend, remaining resources and deadline are unknown, not reset.
CPU checks use one worker on each xbabe1/2/3 and xbabe4 Linux Docker, deterministic
fold/seed assignments, <=85% host load/headroom, nice19 and >=150 GiB free on
xbabe1, nice15 elsewhere. No live-loop, qualification-lock, GPU or xbabe0 action.
Synthetic gradients test algebra only. The real S3 input is a metadata-only H86
manifest retained as provenance context; its signature census does not admit
completion. Publication preserves the exact-version input/output receipts and
four-host results under a fresh `experiments/rd_breakthrough_20261030/W3/` prefix.
