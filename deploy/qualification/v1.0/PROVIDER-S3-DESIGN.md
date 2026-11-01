# REL-07 S3 leakage provider

S3 in spec §14 requires FIT-only landmarks, rung0, rung1 and calibration,
HOLDOUT/TEST access only during finalization, and grouped/time constraints in
every CV phase. `providers/s3.py` verifies the full non-development candidate
through `jope-dist inspect/verify` and binds its embedded source SHA and artifact
SHA-256 to an independently checked clean checkout. It runs the existing product
assertions below; it does not implement another splitter or audit system.

| Existing `jope-service` tests | S3 evidence |
| --- | --- |
| Three `discover::audit::tests` selectors | Dataset-scoped row identity; FIT-only landmarks/rung0/rung1/preprocessing/calibration; forbidden readers never called; atomic refusal; finalization boundary; invalid/empty rosters refused. |
| Group, time/gap, combined group/time/ties, overlapping intervals, nested calibration, and invalid-input/cap selectors in `discover::folds::tests` | Constraints hold across rung0/rung1/calibration; nested calibration excludes outer validation; infeasible constraints fail instead of relaxing; capped rows stay within FIT. Property tests retain their existing 128-case configurations. |
| `finalization_audits_each_real_partition_and_declares_the_refit_and_score_scopes` | Actual synthetic partition reads through the product finalizer: FIT refit, HOLDOUT score, optional FIT+HOLDOUT refit, TEST score; protected rows appear only in their permitted finalization phases. |
| `completed_replay_and_recovery_do_not_read_or_score_protected_partitions_again` | Replay/recovery succeeds after protected fixture files become unavailable, without another fit or score. |
| `failed_bounded_fit_preserves_audit_and_never_retries_or_reads_holdout_or_test` | Failed fitting preserves its audit and does not advance to protected partition reads. |

All twelve exact selectors must each execute and pass once. Missing, substituted,
duplicate, ignored, measured, zero-test or failed outputs refuse PASS. Other
filtered crate tests are outside this gate's coverage. Source cleanliness and
identity, regular non-symlink artifact bytes and new external result path are
checked before verification, between cases and before exclusive publication.
Symlinks in either path's parents are refused too. Failed commands preserve their
exit status and produce no success receipt. Successful receipt counts are twelve
collected/executed/passed, with zero failed/skipped/excluded; each property test
counts as one Rust test, not as 128 separate collected tests.

Provider argv for the REL-07 controller:
`nice -n 10 python3 -B deploy/qualification/v1.0/providers/s3.py`, with the five
standard `JOPE_REL07_*` environment bindings. Run from the exact clean candidate
source; set `CARGO_INCREMENTAL=0` and
`CARGO_PROFILE_DEV_DEBUG=line-tables-only` for the entire invocation, including
candidate verification. Each test uses at most eight build jobs and one test
thread, reusing the lane's target. The controller retains stdout/stderr, commands,
execution times and receipt hashes. Evidence corresponds to the existing S3
`gate-status.md` sources: DSC-02, DSC-03, SVC-11 and DSC-09.

Acceptance controls: `python3 -B deploy/qualification/v1.0/test_provider_s3 -v`.
This follows the extensionless M1/M4/T1 harness pattern and uses only synthetic
candidate bytes and mocked child results. Those controls are **not S3 release
qualification**. The selected Rust tests use generated fixture data, never sealed
corpus payloads, GPU work or signing. They qualify these source assertions against
the verified candidate source; they do not replay arbitrary production datasets
or establish that every future execution is leak-free. A real S3 result still
requires the complete provider invocation against the supplied full candidate.
Historical status entries alone do not supply that result. Runner wiring,
28-gate receipt assembly and signing remain with their existing owners.

No common helper, product implementation, public API, test inventory, dependency,
model weight or gate-status verdict changes are included. A generic successful
command wrapper was rejected because it could accept zero tests or a different
case; a new splitter implementation would test the gate rather than the product.
