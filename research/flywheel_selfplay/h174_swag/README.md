# H174: SWAG Bayesian model averaging

**Status: INCOMPLETE. No efficacy estimate or verdict is available.** The guarded execution did not reach any selector or assessment results. This record reports only the baseline gate, execution failures, and custody.

## Hypothesis and preregistered setup

H174 tested whether a fit-only SWAG posterior over a fixed 40-update AdamW continuation from frozen r882 could improve predictions through SWA, BMA at covariance scales 0.5 and 1.0, or the strict trajectory checkpoint control. The preregistration specified 5 outer family folds × 3 seeds, 600 optimizer updates total, FIT-only trajectory construction, B-only selector decisions durably locked before A/Val assessment, and the exact 91-context r882 baseline gate before any TRAIN outcome reads. DOPE remained pinned to `c51a56d0`; only 29,211 adapter parameters were trainable.

The preregistration merged in #1479 at `a3da025d4b59a7b7ea28bbd606bff19c9a73e214`. Red-team PASS review `5727597218` required raw B costs to be retained in the pre-assessment lock. The repaired lock, data-free replay test, and missing runner import fix were merged in #1483 at `813759c5fb187d0c6139426405b2f2a1c0563d46` after exact-head PASS review `5727612515`.

## Execution and outcome

Two executions are preserved. Attempt 1 failed before completing the baseline gate because `adapter_forward` passed the wrong fused-model mask keyword; it made 0 optimizer updates and read 0 TRAIN outcome pools. The repair was merged before attempt 2.

Attempt 2 passed the exact native 91/91 r882 baseline (`1.1377941898164525`, bit-exact), then executed the first FIT-only cell through 40 of its 40 updates. The runner stopped at the preregistered non-degenerate-trajectory guard: the 16 planned adapter snapshots were byte-identical. No B selection was made and no A or candidate-Val assessment was run. The audit recorded 870 FIT pool files, 0 protected reads, and 0 refused reads.

Because the attempt is partial, no restart or cell reuse was made. The 3-seed OOF median/CI, strict-selection result, Elo, and treatment DL-Val utility were not estimated. This is not a null, NO_ADVANCE, or efficacy result.

## Resources and custody

Both runs were CPU-only on xbabe2 under `nice 19`, with CUDA disabled. The attempt-2 monitor ended `INCOMPLETE` without resource guard violations. Full logs, snapshots, launch bindings, audits, and claims are preserved in the versioned S3 archive in [metrics.json](metrics.json) and the local custody receipt. The exact-version archive and manifest readback verified all 30 members. No serving promotion occurred.

## Next step

H174 remains incomplete. Any future study would require a new authorization and preregistration that explains the unchanged FIT trajectory and validates nonzero adapter movement before assessment. The H174 partial attempt will not be resumed or reused.
