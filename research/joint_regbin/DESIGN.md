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
