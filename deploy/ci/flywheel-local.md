# Focused local flywheel checks

The 2026-10-19 human direction selects process-local CI. `flywheel-local.sh` runs existing tests through `receipt.py`; it does not dispatch GitHub Actions, create runners, publish statuses, run a training campaign, or replace final integrated release qualification.

Run from the claimed, stopped, committed lane in its existing admitted build environment. The source must be clean and remain at the supplied full SHA. Keep its existing Cargo target. Dependencies must already be cached: an offline dependency failure is a missing prerequisite, not a passing or skipped test.

```sh
rtk proxy bash deploy/ci/flywheel-local.sh model-store FULL_SOURCE_SHA /home/ubuntu/jope-evidence/flywheel-value-20261019/model-store-UNIQUE_ID
rtk proxy bash deploy/ci/flywheel-local.sh storage FULL_SOURCE_SHA /home/ubuntu/jope-evidence/flywheel-value-20261019/storage-UNIQUE_ID
rtk proxy bash deploy/ci/flywheel-local.sh feedback FULL_SOURCE_SHA /home/ubuntu/jope-evidence/flywheel-value-20261019/feedback-UNIQUE_ID
rtk proxy bash deploy/ci/flywheel-local.sh parity FULL_SOURCE_SHA /home/ubuntu/jope-evidence/flywheel-value-20261019/parity-UNIQUE_ID
```

Choose the affected suite, rather than running all suites after every small edit. Default per-invocation wall limit is 900 seconds and two Cargo/test/BLAS workers. `JOPE_LOCAL_TIMEOUT_S` accepts 1–1800 seconds; `JOPE_LOCAL_JOBS` accepts 1–4. These limits do not supersede tighter fleet/resource admission. The deadline applies to the invocation and is never reset between its test groups.

`model-store` requires the bounded-store test module introduced by `9744ddbf7` and relocated by `bc7930213`. Running its filter on an older head collects zero tests and must fail. The other suites use existing registered integration targets. They exercise fixtures, not a qualified live feedback experiment.

Output is a fresh directory containing local event/source identity, hashes of this script and the receipt/parser/build inputs, complete command logs, original receipt counts, and `PASSED_SOURCE_SHA` only after successful checks and a final source recheck. The receipt's numeric run ID is a local execution identifier; `event=local-proof` distinguishes it from a GitHub run. It must never be linked as a GitHub Actions run.

The existing receipt tool requires a declared coverage lane. These focused commands use `rust-test` (or `emitters` for parity) as their declared routing category. Their exact command and `identity.tsv` identify the subset: a passing subset does not satisfy the entire declared lane and must not be submitted to an aggregate as full-lane coverage.

A source move, dirty tree, failure, timeout, zero executed tests, skipped cases within the selected scope, or invalid/missing receipt blocks success. Tests excluded by the explicitly printed filter (for example other `jope-datasets` library tests outside `store::bounded_version_tests`) remain recorded as excluded and do not block this focused scope; they are not passing coverage. Keep the original log and receipt. Resolve required-scope skips explicitly through the owning validation policy; do not reduce an expected case count or edit a receipt to make this helper pass. A timeout can leave a `started` receipt; that remains incomplete and no success marker is written.

Independent exact-head review, ordinary PR merge rules, effective branch protection, final declared package/feature coverage and receiving-main validation still apply. If required hosted statuses conflict with local-only CI, the existing merge owner records the enforcement issue and selects the established local-evidence publication route. This helper does not disable branch protection or manufacture a hosted status.
