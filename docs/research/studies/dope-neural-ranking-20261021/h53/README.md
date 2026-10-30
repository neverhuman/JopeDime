# H53 learn-from-zero

Preregistered bounded pilot. See the experiment specification and current evidence in `research/flywheel_selfplay/h53/README.md`. H53 tests seeds 12/13/14 crossed with default and 0.1x fresh random outer-adapter initialization on the pinned family-grouped 1,219 DL-Train / 91 DL-Val cohort. The pilot cap is 30 minutes total. DL-Val trajectories are diagnostic-only; the sealed holdout remains unopened.

Admission from t537 requires one-sided 95% family-bootstrap mean-gain LCB > 0.001 and median family gain >= 0, with sealed holdout untouched. The former median-plus-LB95 rule is reported alongside it. No r882 adapter weights are transferred.

Status: preregistered; pilot not yet run.
