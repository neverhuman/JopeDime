# CI coverage and receipts

`coverage.json` is the explicit mapping from discovered source to execution lanes or
deferred/historical dispositions. Validate it with `cargo run -q --locked -p jope-records -- coverage validate`.
`discover` prints the current static inventory without changing assignments. New packages,
features, Cargo targets, Python test-named files, workflows/jobs and benchmark sources fail
validation until their scope is assigned. Changed target requirements and feature dependencies
also require review. Targets gated on non-default required features cannot be assigned to
default-feature execution. Workflow jobs use a block mapping; comments, consistent indentation
and flow-style job values are supported, while unsupported mapping forms fail closed. Git-tracked and unignored new files are included.

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

`receipt.py --lane LANE --output NEW_DIRECTORY --backend BACKEND [--junit REPORT]
[--counts COUNTS.json] [--evidence FILE ...] -- COMMAND...` records one execution. It requires GitHub run ID and attempt
(local proofs may supply explicit numeric identities), binds both the PR source and actual
checkout SHA, and refuses existing output directories. `JOPE_SOURCE_SHA` names the PR head;
`GITHUB_SHA` must match the checkout, which may be a synthetic merge. Without a PR source
override, the actual checkout is the source identity. Pull-request events require the explicit
source override. Receipts record dirty source paths before and after execution; aggregation
rejects either dirty state.

Receipts retain command output, input and evidence SHA256 digests, start/end times, selected
nonsecret environment values, exit status, and explicit counts. Unknown counts are JSON null;
JUnit omission of skipped tests does not establish zero skips. A killed runner leaves an
incomplete `started` receipt. Missing executables/evidence produce `unavailable`, never success.
Every fast lane invokes `fast-lane.sh` through this wrapper. Its existing command failures
remain failures. Evidence stays in `target/ci-evidence/RUN_ID/ATTEMPT/LANE` under the workspace,
including emitter programs and pipeline receipts, and is uploaded with 30-day artifact retention.
Cargo's persistent lane target is separate. No successful-run cleanup deletes that evidence;
checkout cleanup follows its upload. Local proofs remain under durable external custody.

The aggregate runs with `always()` after every dependency, including failed/skipped jobs. It
requires an exact current run/attempt receipt for each planned lane and checks both source and
actual checkout SHA, command, input/evidence hashes, source cleanliness, times and counts.
It recomputes the path plan from the checked-out base diff. Only a declared path exclusion may
have a skipped job with no receipt. Failed/cancelled/missing/started/unavailable outcomes, missing
reports after execution starts, ambiguous receipts, zero executed tests and unknown test counts
fail. CI receipts are read only at each artifact root; nested pipeline receipts are hashed
product evidence. Every downloaded evidence file must be declared by the CI manifest, including
hidden files. There is no `continue-on-error` success path. Retry with **Re-run all jobs**:
re-running only failed jobs leaves successful lanes without receipts for the new attempt and
therefore cannot pass this aggregate.

Nextest collection is captured separately from execution. The execution summary supplies skips
that nextest 0.9.133 omits from JUnit; serialized executed/pass/fail counts must agree. Nextest's
summary combines ignored/filter skips under `skipped`; its collection JSON retains individual
filter/ignore metadata. Pytest reports skipped and deselected counts separately. Rust harness
filtered tests appear as excluded. Counts are per invocation, not deduplicated across lanes;
adding lane totals would double-count tests that intentionally execute in multiple lanes.

The merge procedure requires the fast lanes and aggregate at the exact source head. The coming
full-main workflow remains diagnostic until the later CPU repair and qualification phase.

`python3 deploy/ci/status.py RUN_ID [--expected-sha FULL_SHA]` reads that exact run and its current
attempt through GitHub APIs; it never selects an older green run and does not need an Actions
runner. For queued jobs it reports requested labels, matching online/busy/offline registrations
and observed queue age. At ten minutes it probes owned xbabe1/xbabe2 service state and disk
availability over SSH. `--host NAME` overrides the hosts and `--no-host-probes` selects API-only
reporting. Probe failures stay unavailable, not healthy. Service properties exclude environment,
process arguments and secret-bearing registration logs. Unknown queue age stays unknown.

Acceptance: `cargo test -q --locked -p jope-records --lib -- coverage::`, `python3 deploy/ci/test-receipt.py`,
`bash deploy/ci/test-inventory.sh --no-rust`, and the existing exact-head CI/review lifecycle.
`test-results.py`, `test-aggregate.py` and `test-status.py` add negative count reconciliation,
wrong SHA/attempt, cancellation, missing results/receipts, zero/unknown execution and capability
cases; genuine docs-only path plans pass. Assignment and collection failures remain covered.
Formatting and workflow validation are hygiene gates; strict Clippy and broader repairs remain
the following phase. These records do not start qualification or soak clocks.
