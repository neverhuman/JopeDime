# H123 — continuous-regret tie-break of the H114 lattice

**Status: proposal only; no outcomes or execution.** The required read-only copy
of H114's `adapter_admissions.jsonl` was not found in local evidence custody.
H114's committed summary reports 21 candidates at the post-r882 maximum, but
does not identify their rounds or establish whether their adapter artifacts
still exist. Those facts remain unknown. No H114 live directory, checkpoint,
model, dataset payload, S3 object, GPU, or qualification lock was accessed.

## Hypothesis and frozen candidate rule

H114 reports 21 distinct candidate rounds tied at utility
`1.1377131278566008`, each below r882. If at least three of those candidates'
adapter artifacts are verified from the authorized read-only H114 log copy,
evaluate only that top-utility candidate set. On a new deterministic five-fold
whole-family split of real H50 DL-Train data (sort family IDs, permute with
NumPy `default_rng(12300)`, distribute round-robin to five folds), compute each
candidate's continuous mean regret on the four training folds:

`regret(context) = mean_i(max_j(true_utility[context,j]) - predicted_utility[context,i])`.

Average context regret within each family, then average family means equally.
Select the candidate with the lowest training-family mean regret; ties resolve
by lower H114 round number, then ascending artifact SHA-256. The selection is
fixed before examining the held-out fold. This metric follows H121's H114(a)
co-readout; it never replaces H114's official argmin utility or changes the
live flywheel admission rule.

If fewer than three artifacts exist, retain this proposal and do not score. If
artifact availability cannot be established from the required read-only copy,
as here, report feasibility UNKNOWN and do not score.

## Conditional evaluation plan

Only if the read-only copy establishes at least three usable top-tied
artifacts, freeze the candidate inventory and this rule before any scoring.
Use the same selected candidate and a fresh r882 copy as the paired reference;
use only real H50 DL-Train families, with the first four folds for the fixed
selection rule and the fifth fold for evaluation. Pair seeds `61`, `62`, `63` across the
selected candidate and r882. Primary is the median across seeds of paired,
equal-family mean utility delta (candidate minus r882). Report the family
cluster-bootstrap one-sided 95% LCB and two-sided 95% CI, and the fraction of
families with delta `< -0.01`. Advance only if median delta is strictly
positive, LCB is strictly greater than `+0.001`, and harm is at most 10% of
families; ties fail. Report family-grouped Elo and 95% CI versus r882=1500.
Report Elo, never AUC. Do not multiply choices by testing multiple candidates
on the held-out fold: the training-fold rule selects exactly one candidate.

DOPE remains frozen at SHA-256
`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`; trainable
parameters are limited to the adapter. Inputs are real S3 H50 DL-Train only.
Never read DL-Val, sealed, holdout, outer, or meta-test payloads. Keep scratch
below 2 GB and remove only H123 scratch after its evidence has a verified S3
receipt, if execution is later approved. No GPU, ticket, waiter, lock, training,
or launch is authorized by this proposal.
