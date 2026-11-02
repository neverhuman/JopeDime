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

## Required PR and main checks

Follow the [exact-head PR procedure](../AGENTS.md#pull-requests).

| GitHub entrypoint | Trigger and role |
| --- | --- |
| [`ci.yml`](../.github/workflows/ci.yml) | Runs for pull requests marked ready for review and by manual dispatch. The PR run satisfies PR rules; manually dispatch it on the final branch to retain the full-head matrix evidence. |
| [`full-main.yml`](../.github/workflows/full-main.yml) | Manually dispatched trusted-main run for the extended CPU families and their aggregate. It is separate from PR checks. |
| [`main-guard.yml`](../.github/workflows/main-guard.yml) | Runs on pushes to `main` and verifies merged-PR provenance. It does not certify test coverage or source quality. |

For the manual `ci.yml` run, retain its actual full head, run ID and attempt.
Every declared manual job must pass, including both Rust shards and frontend
tests. Preserve the separate PR run and its synthetic checkout identity. A
legitimate path skip is recorded as a skip. Changed source requires fresh checks
and substantive review; same-agent review is permitted and identified.

`full-main.yml` runs the eleven extended CPU families in two non-cancelling groups.
Its aggregate verifies the entire declared partition and all evidence. The independent
trusted-main audit job executes the same immutable-input audit wrapper used locally.
Until the score reaches the 85 floor the job is report-only: a verified below-floor
audit passes with a warning and `report-only.json`, while invalid or inconsistent
evidence still fails it. A green audit job is therefore not a passing audit.
A successful `main-guard` run proves merged-PR provenance; it does not certify tests,
audit quality, scientific utility or serving readiness.

## Failures and evidence custody

A failed command, missing tool, missing or invalid report, cancelled run, zero tests
where tests are required, or an unavailable capability is not passing proof.
Read the lane command log and receipt before retrying; preserve run attempts and
distinguish collection, environment and product failures. Do not cancel another
owner's jobs or widen filters to make the result green.

The Jankurai wrapper keeps the default 85 floor and full standard policy. Its
`audit-verification.json` checks report consistency and captured-input integrity
separately from `audit_passed`. A valid report can contain a failed audit.
See the [dated remediation record](coordination/audit-remediation-20261022.md)
for the 52-point baseline and remaining findings; no threshold waiver is implied.

Keep `target/` evidence and frozen numerical fixtures intact. Run builds with
`CARGO_INCREMENTAL=0` and `CARGO_PROFILE_DEV_DEBUG=line-tables-only`; reuse the
claimed lane's build directory and leave global `CARGO_TARGET_DIR` unset.
Do not publish checkpoints, training claims, models or defaults from fixture tests.

## Cost bounds and stop conditions

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

## Release readiness evidence

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
