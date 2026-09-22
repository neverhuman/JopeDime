# CI coverage and receipts

`coverage.json` is the explicit mapping from discovered source to execution lanes or
deferred/historical dispositions. Validate it with `python3 deploy/ci/coverage.py validate`.
`discover` prints the current static inventory without changing assignments. New packages,
features, Cargo targets, Python test-named files, workflows/jobs and benchmark sources fail
validation until their scope is assigned. Changed target requirements and feature dependencies
also require review. Git-tracked and unignored new files are included.

The three evidence stages are distinct:

* **Static discovery:** source exists and has an assigned scope. Cargo metadata and filename
  discovery do not prove that a test was collected, executed, or passed.
* **Collection:** the selected runner successfully lists concrete tests. Python collection
  errors remain errors. `test-inventory.sh --static-python` explicitly selects an AST inventory;
  it cannot substitute for failed pytest collection.
* **Execution:** a fresh receipt reports the command, outcome, counts and evidence. A configured
  workflow or deferred full-main mapping is not execution qualification.

The maintained Python selection is the explicit `python/tests/` file set in the mapping.
Research/tools test-named scripts, frozen kernel-v1 copies, maintenance fixture programs and
benchmarks have separate dispositions. PR pytest selects `not xbabe2`; exclusions must remain
visible. The PR Rust package scope and nextest profile also exclude tests. Full workspace CPU,
doctests and declared feature variants remain assigned to the upcoming bounded full-main phase.
Binary/example builds do not prove their operational entry points.

`receipt.py --lane LANE --output NEW_DIRECTORY [--counts COUNTS.json]
[--evidence FILE ...] -- COMMAND...` records one execution. It requires GitHub run ID and attempt
(local proofs may supply explicit numeric identities), binds both the PR source and actual
checkout SHA, and refuses existing output directories. `JOPE_SOURCE_SHA` names the PR head;
`GITHUB_SHA` must match the checkout, which may be a synthetic merge. Without a PR source
override, the actual checkout is the source identity.

Receipts retain command output, input and evidence SHA256 digests, start/end times, selected
nonsecret environment values, exit status, and explicit counts. Unknown counts are JSON null;
JUnit omission of skipped tests does not establish zero skips. A killed runner leaves an
incomplete `started` receipt. Missing executables/evidence produce `unavailable`, never success.
The following aggregation phase will wire these receipts into every fast lane and reject
incomplete, stale, failed, cancelled, missing, zero-test and unavailable outcomes.

Acceptance: `python3 deploy/ci/test-coverage.py`, `python3 deploy/ci/test-receipt.py`,
`bash deploy/ci/test-inventory.sh --no-rust`, and the existing exact-head CI/review lifecycle.
The first two include negative assignment, collection, identity and failure-retention cases.
Formatting and workflow validation are hygiene gates; strict Clippy and broader repairs remain
the following phase. These records do not start qualification or soak clocks.
