# CI determinism: cancellation, projected GP and durable recovery

The changes preserve the admitted numerical and work limits. They do not qualify a live
container, GPU job, learning campaign, S3 publication or serving release.

## Cancellation and execution deadlines

The durable cancellation fixture tests logical ordering on a paused Tokio clock. The
production cancellation record and directory are still synced before notification. The
original signed smoke/run limits remain 1,000/10,000 milliseconds. The existing two-second
Tokio helpers now measure virtual time; the unchanged PR watchdog bounds real host elapsed
time at 240 seconds. A passing cancellation fixture alone does not prove deadline handling.

Two independent blocked-wait controls explicitly advance each signed limit. They require
no early kill, a kill at the limit, a durable `DeadlineExceeded` terminal failure, no successful
terminal or output publication, and no replayable/quarantined assignment after recovery.
After each advance, the fixture wakes the executor's blocked wait and awaits its poll or
kill acknowledgment. On this single-thread runtime, the executor poll returns before the
fixture observes state; `advance` alone would not establish that expired timers ran.
Both controls reject compiled early, one-millisecond-late and disabled timer mutants, and
wrong terminal classification. The production executor is restored byte-for-byte afterward.
Production executor, cancellation persistence and admission code are unchanged.

The frozen unchanged-main cancellation binary fails its original `Cancelled` assertion
when strace delays its eleventh per-thread fsync by 1,500 milliseconds. This is a durable
outbox write before the executor first yields. Other delay probes did not reproduce the
failure and are retained as nonreproductions. The corrected fixture passes the same
injected operation. The first candidate replay harness failed before executing a test;
that harness failure is retained separately from the corrected replay. Ordinary successful
runs alone do not close this regression.

## Projected GP accounting

`projected_gp_candidates_are_exactly_representable_unique_and_durable` retains the original
100-observation and 90-second budgets. It also requires `completed_evaluations = 100` and
`stop_reason = evaluation_budget`. The proposed 600-second change from inherited commit
`5346721fb025fadcfbb8fd7c034e420d9605a94c` is rejected. Existing vocabulary, unique-program,
index, progress, journal and per-candidate retained-artifact checks remain intact.

Completed observations include failures and refusals. This fixture does not establish
100 successful fits or measured improvements. Storage saturation can exhaust the legitimate
90-second request before 100 observations; that outcome fails this fixture and remains visible.

## Fault matrix and storage latency

All 18 original fault cases retain their operation order and assertions: damaged closure
refusal, complete delivery accounting, two independently fresh deterministic restores,
idempotent repeated restore, clean recovery, and fresh-host resumed publication. Four bounded
case workers overlap independent filesystem waits. Every world owns distinct local and remote
roots; no state or remote is shared across cases. Every worker is joined, any panic fails the
one enclosing test, and the 18 results are checked once each and rendered in original order.
The single original 240-second watchdog remains; cases are not split into separate deadlines.
Four case workers can use additional Tokio blocking threads, so aggregate CPU/memory limits
must be enforced by the invoking lane.

An unchanged-main reproduction at `2ba761e413f3cbe7761d42cad49d3005d61ca241` on the loaded
root ext4 device recorded only 36 GP observations at 90.991 seconds and timed out recovery
at 240.024 seconds after nine completed cases. The recovery thread was observed in
`jbd2_log_wait_commit`. The same source/binaries/profile on a separate persistent ext4 device
completed GP in 19.177 seconds and recovery in 16.347 seconds, with the same original limits.

Traces of the frozen original serial recovery binary completed all 18 cases on both devices
and retained exactly 4,358 fsync calls, 2,234 renames, 11,838 opens and 1,835 reads. The loaded
root trace spent 154.890 seconds inside fsync, versus 2.429 seconds on the other device. Traces
include overhead; futex totals aggregate waits across threads and are not additional request
wall time. This diagnoses storage wait in this reproduction. It does not reconstruct the
cause of every historical timeout or claim a product performance improvement.

The four-worker matrix passed on that same loaded root disk in 94.112 seconds under the
original watchdog. Its complete traced execution also retains all 18 cases, 4,358 fsync
calls and 2,234 renames. Its process-level open/read counts differ by one; only the durable
sync and rename counts are claimed equal. The source check separately establishes that
each case's operations and assertions are unchanged.

Durable receipts, original failures, frozen binary hashes, kernel wait samples, syscall
summaries and resource controls are retained under
`/home/ubuntu/jope-evidence/ci-determinism-20261006/`. Tests use persistent disk, not memory
or tmpfs. No sync, restoration, assertion, timeout or filter is removed. A loaded storage
failure is not waived by a passing run elsewhere.

## Acceptance

Use the lane's reused target and preserve the repository Cargo settings. Set `TMPDIR` to an
explicit owned persistent-disk fixture directory outside `/tmp`; retain its filesystem/device
identity and available capacity. Enforce the original 4-CPU/16-GiB, 15-minute batch limits.

```sh
cargo nextest run --locked --profile pr -p jope-learning -p jainnode -p jope-parallel   --test-threads 2 -E 'test(every_fault_leaves_committed_state_restorable_and_every_delivery_accounted_for) | test(concurrent_runtime_accepts_durable_cancellation_while_execution_runs) | test(projected_gp_candidates_are_exactly_representable_unique_and_durable) | test(concurrent_runtime_enforces_the_signed_)'
cargo test --locked -p jainnode
cargo clippy --locked -p jainnode -p jope-learning --all-targets -- -D warnings
just required
```

Require positive and failure-injected cancellation, deadline-branch mutants, exact complete
fault-case/sync accounting, unchanged-limit root-disk recovery and full selected PR-tier
coverage. A full selected-tier invocation exhausted its original 900-second batch after
846 passing cases and four interrupted cases; 1,559 were not run. Keep that failure and
inventory. Four fixed hash partitions may establish complete coverage in separate bounded
batches, with an independently checked disjoint union of the same 2,409 selected cases.
The existing exclusions remain 29 ignored cases and four PR-profile exclusions.
Report inherited required-gate failures separately. Source, focused proof, hosted
CI and operational qualification remain separate states.
