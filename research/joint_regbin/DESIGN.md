# JRB-01 three-arm harness design

## Scope

This CPU orchestration/report component schedules the primary three-arm study:
`joint` (balanced 1:1 regression/binary contexts), `reg-only`, and `bin-only`.
All arms use the same three seeds, five family-held-out folds plus one full-train
job, 50 rounds, two updates per round, 32 contexts per update, and 10 conditioning
slots per context. That is 54 jobs and 100 optimizer updates per job. The prior
natural-ratio sensitivity arm is deferred by the T0218 owner scope. This source
change does not alter the learner or authorize a study run.

## Measurements and decision

Each arm emits task-specific validation curves every five rounds: log RMSE/ridge
for regression and log-loss for binary. The report computes task-separated,
family-grouped five-fold CV Elo with 1,000 paired family bootstrap replicates;
DL-Val and panel ratings are reported separately. JOINT_OK requires, on both
tasks, joint to be within 25 Elo of its matching single-task arm or the 95% CI
to include zero. SPLIT_RECOMMENDED requires the same task to trail by more than
25 Elo with its 95% CI excluding zero on each of two consecutive evaluations.
Otherwise the result is MITIGATE_AND_RERUN; an incomplete run has no verdict.

## Dry-run boundary

The two-round CPU report smoke is a synthetic protocol fixture, not a dev-data
run, learner qualification, CV estimate, or scientific result. The currently
available JRB binary packet is train-only and `PARTIAL_PANEL_ONLY`; no complete
eligible joint role manifest or qualified task-correct learner/checkpoint is
available. Do not adapt that packet into dev data or bypass the manifest checks.
A real dev-data dry-run remains required after eligible paired dev inputs and a
qualified worker are supplied. GPU launch remains a separate owner-controlled
operation under the exclusive qualification lock.

## Worker task-summary admission

The worker summary and its qualification receipt must each name exactly the two
supported tasks as a list of strings; either order is accepted. Validate the
summary shape before comparing task sets. Null/scalar/nested values, mappings,
duplicates and unsupported tasks produce the existing worker-head blocker and
cannot create a worker-bound plan. These consistency checks do not authenticate
an issuer, qualify a real worker or authorize the study.

## Two-task worker core (ENG:watcher R19, 2026-11-01)

The worker core adds a regression identity head and a linear binary-logit head
over the shared adapter output. It uses Smooth L1 for explicitly labelled
regression contexts and BCE-with-logits for binary contexts; binary calibration
and metrics load the owner-pinned `binary_clf_head_and_metrics.py` source. Both
tasks receive packed inputs through the same injected flywheel candidate-pool
packer.

The worker accepts only admitted protocol context rows. TEL's role-candidate
manifest is metadata-only, `ROLE_CANDIDATES_NOT_ADMITTED`, and has no payload
references; it is intentionally rejected by the worker until a reviewed
admission step supplies exact payload identities and the packer integration.
Synthetic tests exercise the task heads, loss, update, metrics and refusal
boundaries. They do not qualify a real worker or authorize campaign execution.

Before either update entry point calls the optimizer, at least one trainable
shared-adapter parameter must have a gradient and every present optimizer
gradient must be finite, including the binary head. Unused gradients may remain
absent (the binary head is unused for regression). Refusals occur before any
optimizer step or parameter change.

Regression metrics reject nonfinite inputs and computed RMSE. The reported RMSE
remains exact, including zero for perfect predictions. The log-ratio curve uses
an explicit relative RMSE/ridge floor of 1e-12 so perfect or smaller ratios have
a finite JSON/report value, log(1e-12). This numerical reporting convention does
not change training losses, Elo metrics, study budgets or decision predicates.
