# Testing and audit evidence

Start with the [agent contract](../AGENTS.md), [CI replay guide](../ops/ci/README.md)
and [capability matrix](parity/capability-matrix.md). Historical counts and passing
receipts describe their recorded source; collect fresh evidence for a changed head.

## Choose the proof that covers the change

| Command or workflow | Coverage and evidence |
| --- | --- |
| `bash scripts/ci-doctor.sh` | Checks local tools and prerequisites; it does not execute the test suite. |
| `bash scripts/ci-local.sh hygiene` | Runs the declared hygiene commands, including security and CI evidence fault controls. |
| `bash scripts/ci-local.sh rust-check` | Locked workspace check, systemd path tests and the declared MSRV check. |
| `bash scripts/ci-local.sh rust-test` | Product nextest tier; locally defaults to the complete unsharded selection. |
| `bash scripts/ci-local.sh emitters` | Emitted-code and portable parity matrices, including declared sanitizer and C++ variants. |
| `bash scripts/ci-local.sh inputs` | Offline fixture verification/generation and the campaign policy/ranking checks. |
| `bash scripts/ci-local.sh python` | Locked Python lint, pytest and the declared offline learning contract fixtures. |
| `bash scripts/ci-local.sh e2e-smoke` | The existing product smoke command. |
| `just ci-pr` | A local subset plus clippy; it is not identical to every GitHub PR job. |
| `just test-main` | Whole-workspace main nextest tier; it does not execute every extended main family. |
| `just test-real-corpus /owned/read-only/custody` | Host-only real-corpus cases from the declared inventory; requires their actual retained inputs. |
| `just audit` | Pinned full-source Jankurai audit of committed tracked input, retaining a fresh source manifest and reports. |

Local replay receipts are under `target/ci-local/RUN_ID/LANE/`. Each records source,
checkout, command, exit, test counts and evidence hashes. To reproduce the hosted
Rust shards, set `JOPE_RUST_TEST_PARTITION=slice:1/2` and then `slice:2/2` in
separate invocations. Output reuse is refused. The changes job needs GitHub event
and base metadata and is unavailable through local replay. The rendered Observatory
UI job has its own workflow and test receipts; an e2e smoke pass does not replace it.

Use the commands in [coverage.json](../deploy/ci/coverage.json), the
[fast lane implementation](../deploy/ci/fast-lane.sh) and
[full main runner](../deploy/ci/full-main.py) as the executable inventories.
The [nextest configuration](../.config/nextest.toml) defines tier filters.
Ignored, host-only and unavailable cases remain explicit non-execution, including
Python tests marked `xbabe2`; static coverage mapping is not execution coverage.

## Hosted CI procedure

The [agent contract](../AGENTS.md#pull-requests) is the canonical procedure for
pull-request checks, full-head CI, review and merge. The [CI replay guide](../ops/ci/README.md)
documents the local commands. This page lists test commands and their scope; it does not
record that a hosted workflow or a particular change has passed.

## Failures and evidence custody

A failed command, missing tool, missing or invalid report, cancelled run, zero tests
where tests are required, or an unavailable capability is not passing proof.
Read the lane command log and receipt before retrying; preserve run attempts and
distinguish collection, environment and product failures. Do not cancel another
owner's jobs or widen filters to make the result green.

The Jankurai wrapper checks report consistency and captured-input integrity
separately from the audit decision. A valid report can contain findings. See the
[dated remediation record](coordination/audit-remediation-20261022.md) for its
historical baseline and status; this page does not attest to a particular audit result.

Keep `target/` evidence and frozen numerical fixtures intact. Run builds with
`CARGO_INCREMENTAL=0` and `CARGO_PROFILE_DEV_DEBUG=line-tables-only`; reuse the
claimed lane's build directory and leave global `CARGO_TARGET_DIR` unset.
Do not publish checkpoints, training claims, models or defaults from fixture tests.

## Budget, quota and stop conditions

Use the owning task's recorded resource budget and concurrency quota. Shared-host checks retain
that task's CPU, build-job, memory and disk limits; this repository's bounded lanes use no more
than eight Cargo build jobs. External paid services require the task's recorded spend cap. An
already authorized command keeps its existing authorization and limits; this guide adds no
approval step or new cloud/API allowance.
Every job that can spend money, consume shared compute, read external corpora, or retry work must
record its owner, exact command, maximum wall time, concurrency, input scope, and stop condition before
launch. The stop condition must be enforced by the command or supervising lane; reaching a bound,
losing an input pin, or receiving an unknown result stops the run without an automatic retry. Record
actual duration, attempts, resources and any external charges with the run receipt. A command that has
no declared cap is not an approved paid or shared-resource lane. The default permitted external spend
for an invocation is **$0**. A paid invocation requires a run-specific, owner-approved maximum cost in
its preflight record; unknown pricing, missing quota, or unavailable kill switch evidence means no
launch. The supervising lane must stop before the approved cost or resource ceiling, and the receipt
records the estimate, ceiling, actual charge when available, stop reason, and kill-switch result. This
policy does not invent a dollar ceiling for an unapproved operation. Keep one reused `target/` per lane,
`CARGO_TARGET_DIR` unset, `CARGO_BUILD_JOBS` at or below 16, and the BLAS/Rayon limits from the owning
proof lane. Do not infer a compute or cloud budget from a test pass.

Reaching a quota, spend cap, input-pin failure or unknown required result is a stop condition,
not permission for an automatic retry. The kill switch is the owning operator's documented
supervisor stop path for that lane's own processes. Never cancel another owner's jobs. Preserve
partial evidence, actual duration, attempts, external charges and the reason for stopping.

## Release gate evidence

A release gate is an artifact checklist bound to the exact candidate and source. This policy
requires evidence; it does not assert that any candidate has passed. Preserve the current owner,
timestamp, artifact identity and outcome for each control:

| Control | Evidence required before a release claim |
| --- | --- |
| Security | Current source-bound security results and unresolved findings. |
| Backup | Recovery artifact identity plus an actual restore verification. |
| Monitoring | Probe results, alert routing and the responsible operator. |
| Rollback | The preserved prior artifact and an executed rollback rehearsal. |
| Abuse and rate limits | Executed control/refusal cases and their limits. |
| Cost | The task budget, quota, spend cap, stop condition and owning-lane kill switch. |

Missing, stale, failed or unknown required evidence remains open. Unit tests, a signed stage,
an inventory check and merged-PR provenance do not replace release qualification. Preserve raw
failures separately from any explicitly authorized bounded disposition; do not relabel them PASS.

## Human Review Evidence

A review record must bind to the full source SHA and identify the reviewer, author relationship,
review time, scope, findings and disposition. Retain the complete diff, raw CI logs and artifacts,
and replayable commands with their results; link to those records instead of treating a summary as
proof. A native review uses the full `commit_id`; a same-agent review is labeled as such. This guide
defines the record required for a review claim and does not assert that a specific change was reviewed.
The release gate blocks promotion until the candidate-bound evidence below is current and reviewable.
A passing unit, integration, smoke, audit or main-guard job is evidence only for its stated scope. A
release-readiness claim also needs current, reviewable evidence for security controls, recoverable
backups, monitoring and alert ownership, a tested rollback path, and abuse/rate controls. Link each
artifact to the release candidate and record its owner and timestamp. Missing, stale, untested, or
unknown evidence remains open. This page makes no release-readiness claim; no gate is considered passed
without a candidate-bound artifact and an owner/time record. Release promotion must stop while any of
those artifacts is missing or unknown.

## Canonical path aliases

The repository keeps its implementation paths. Jankurai's `apps/web/` route maps to
`web/training-dashboard/` and its existing npm/Playwright lanes; `db/` maps to the dataset boundary in
`crates/jope-datasets/` and its locked Rust tests. These aliases are recorded in `agent/owner-map.json`
and `agent/test-map.json`; do not create duplicate `apps/web/` or `db/` trees to satisfy a detector.
