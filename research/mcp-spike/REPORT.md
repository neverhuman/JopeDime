# MCP-01 report — OD9 ruled: rmcp 3.8.0

OWNER selected exact-pinned **rmcp 3.8.0** for MCP-02 in the Jepson-approved t526u
ruling, `~/nudges/scope_2218.md` (2026-10-29 22:18 MDT). Its minimal server passes this
bounded two-tool wire suite for all four requested versions, supports concurrent
calls and suppresses cancelled replies. The generalized existing dispatcher passes
the three legacy versions, requires a handshake and blocks fast calls behind slow
ones. Retain `2026-07-28` as an implementation candidate; shipped support still
requires MCP-02's full product qualification.

**Status: READY_FOR_REVIEW; publication handed to ENG:watcher.** The same ruling
classifies KI-1/2/3 as known main failures to record without blocking. The later
explicit owner instruction requests a local commit and `~/nudges/mcp01.bundle`,
with no push from this worker. ENG:watcher pushes from xbabe2 and opens the PR.
[The PR draft](PR_BODY.md) begins with the required [design note](DESIGN.md);
[the commit message](COMMIT_MESSAGE.txt) includes all three trailers and `[skip ci]`.
Two COMPLEX reviewer passes remain required. No CI, push or PR merge is authorized
for this handoff.

## Experiment and identities

- Worktree: `mcp-01`; branch: `v1/mcp-01-conformance-spike-rmcp-vs-exis`.
- Upstream/branch base: `5b72f953cbdc987a12528c50e0f45ff388331908`. The spike commit
  changes only `research/mcp-spike/`; required coordination appends are outside
  the checkout in the canonical ledger. Resolve the candidate full SHA with
  `git rev-parse HEAD` after importing the bundle; the external handoff records it.
- Executed protocol receipt base: `3346ae2c00d86e7d6b52d6048ca3a87d4ad7bb20`; its source-file hashes,
  binary hashes, UTC, machine and OS are in [replay.json](evidence/replay.json).
  Product Cargo, toolchain and crate bytes are unchanged by the later base refresh.
- Dispatcher source lineage: `crates/jope-runtime/src/service.rs` at
  `5c8a16711632b4b8c588c9a16679a585abc0f135`. The plain JSON-RPC branch and runtime
  tool calls are omitted; envelope validation, lifecycle, negotiation and error
  framing are retained for the two fixture tools. The serial execution model is
  retained deliberately. Stdio framing is a small fixture wrapper.
- SDK archive: `rmcp` 3.8.0, SHA256
  `f8be15815186873ea063bf005011d16c356804973489e40c39116ae3a2878d1f`.
  Only `server` and `transport-io` SDK features are enabled; defaults are disabled.
- Rust 1.97.1, eight Cargo jobs, niceness 10, incremental 0, dev debug
  `line-tables-only`, one reused lane `target/`; build scratch is `target/tmp/`.
  No GPU, scientific datasets, DOPE, sealed/outer inputs or GitHub CI are used.

The retained run contains **120 case observations**, each with its own result.
The driver exit of zero means the comparison completed; it does not mean every
case passed. [Raw transcripts](evidence/replay.jsonl) are bound by SHA256 in the
receipt. Exact schema URLs and hashes are in [references.json](evidence/references.json).
They pin all four upstream schemas at
`84d845729393f79b3fd3d392aa17cb86f5b6f95c`.

## Per-version bounded protocol results

| Requested version | Generalized dispatcher | rmcp 3.8.0 |
| --- | --- | --- |
| 2025-11-25 | PASS (11/11) | PASS (11/11) |
| 2025-06-18 | PASS (11/11) | PASS (11/11) |
| 2024-11-05 | PASS (11/11) | PASS (11/11) |
| 2026-07-28 | FAIL (0/13) | PASS (13/13) |

Legacy checks require the exact requested version, initialize/initialized ordering,
two schema-bearing tools, both calls, structured/text agreement, tool failures,
protocol errors, string IDs, notification silence, unknown-version fallback and
exclusion of `2025-03-26`. Modern checks use fresh processes without initialize,
discovery, required complete-result fields, cache hints, per-request metadata,
unsupported-version errors/retry and cancellation. A fallback to another version
never counts as support for the requested version.

These are targeted conformance checks, **not** the upstream full conformance suite,
a named-client qualification or certification of every optional MCP feature.
All 13 modern required observations fail on the handshake-only dispatcher.

## Concurrency, cancellation and malformed input

Twenty samples per supported version: start a 200 ms `slow_echo`, wait 20 ms,
then time `echo`. p95 is the nearest-rank 95th percentile. Raw samples are retained.

| Version | Dispatcher p95 | SDK p95 |
| --- | --- | --- |
| 2025-11-25 | 180.359 ms | 0.563 ms |
| 2025-06-18 | 180.281 ms | 0.562 ms |
| 2024-11-05 | 180.308 ms | 0.554 ms |
| 2026-07-28 | unavailable: version refused | 0.702 ms |

The dispatcher serializes calls and emits a reply after cancellation. The SDK
keeps fast calls responsive and suppresses cancelled replies in all four tested
versions. This uses an asynchronous delay, **not CPU-bound model fitting**; it
cannot establish the product `run_status`/`run_control` acceptance gate.

For invalid JSON after a working call, the legacy dispatcher emits `-32700`.
The SDK emits no parse-error reply within 300 ms, then successfully handles the
next valid request. Those four SDK malformed-input diagnostics are recorded as
FAIL rather than hidden. They intentionally violate the
[stdio client's valid-message requirement](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio).
MCP-02 must decide and qualify its bounded malformed-frame policy. The dispatcher
cannot enter modern operation, so its modern malformed-input diagnostic is also
unqualified. Successful cases require clean EOF exit and no unexpected stdout.

## Dependency closures

| Server | Unique name/version packages including root | Registry packages | Tokio |
| --- | ---: | ---: | --- |
| Dispatcher | 7 | 5 | absent |
| rmcp 3.8.0 | 54 | 52 | 1.54.1 |

See [dispatcher tree](evidence/dispatcher-tree.txt),
[SDK tree](evidence/rmcp3-tree.txt) and [strict scan](evidence/dependencies.json).
The normal closure includes normal edges of proc macros; these counts are not
linked-runtime library counts. The SDK adds 47 registry packages in this fixture.
Both fixture closures and the existing 89-package appliance closure have zero
banned matches. Linux `ldd` readbacks are retained separately; platform-wide link
audit qualification remains with release work.

The card's `grep ... && exit 1 || true` command masks both banned matches and
Cargo errors. Its execution is retained as UNQUALIFIED; the strict scanner checks
Cargo status, rejects an empty graph and matches complete crate names, including
`onig_sys` and `esaxx-rs`. It passes for all three closures.

## Local gate evidence and limits

All eight exact card commands were executed; [checks.json](evidence/checks.json)
retains initial commands, exits, times, environment, output excerpts and raw-log
hashes. [recheck.json](evidence/recheck.json) records focused checks at the later
`3346ae2c` base. Historical failures are retained, including a corrected invocation
of the newer cargo-deny CLI; [spike-deny.json](evidence/spike-deny.json) is the
successful corrected SDK dependency-policy check.

- Runtime fmt and all-target Clippy: exit 0.
- Runtime nextest: **86 run, 86 passed, 0 skipped**, exit 0.
- Workspace cargo-deny: exit 0, advisories/bans/licenses/sources OK.
- Strict dependency scans: exit 0, no banned crates in either experiment or appliance.
- Spike fmt/Clippy and corrected cargo-deny: exit 0.
- Shared fixture Unicode/schema boundary: **1 test passed**, 0 failed/ignored.
- Python Ruff check/format and shell syntax: pass. No Python product code changed.
- `just lint-paths`: exit 0.
- `just hygiene`: failed; dated exact outputs are in
  [hygiene-recheck.json](evidence/hygiene-recheck.json), and a latest-base rerun is
  retained in `evidence/hygiene-latest.json`. Missing local tools were resolved
  with repository-pinned downloads under the lane target, without changing
  credentials, host-global tool configuration or product source.
- `just audit`: baseline **54**, 294 hard and 86 soft findings, exit 1 at
  `5c8a16711632b4b8c588c9a16679a585abc0f135`.
  [Verified audit receipt](evidence/baseline-audit-verification.json) binds the
  immutable source manifest and report hashes. Full artifacts remain under
  `target/jankurai/reports/local-20261030T032408Z-194952/`.
  Untracked candidate files were excluded by the owning audit exporter.
  **Candidate audit and no-regression comparison remain unknown.**

An initial hygiene run could not find npm. After lane-local tool provisioning,
the `3346ae2c` rerun found an unassigned Phase-3 Python test, shallow-history secret
scan refusal and a stale README Python count. Later upstream changes may repair
some dated findings; the latest receipt is authoritative for that newer base.
The latest `5b72f953` rerun still fails on the unassigned Phase-3 test and shallow
secret-scan history; the README count now passes after the upstream update.
No unowned inventory, README, security policy or history-depth repair was made.

MCP-01 does not close all of A4. MCP-02 still owns the real 12-tool/client adapter,
resources, legacy aliases, bounded IO, genuine fitting-load cancellation/latency
and product conformance. MCP-05 owns named-client scenarios. The
[LUNA handoff cards](FOLLOW_UPS.md) refine MCP-03 and MCP-04 without claiming them.

## Owner disposition

OD9 implementation choice: **rmcp 3.8.0 — OWNER RULED**. Authority:
`~/nudges/scope_2218.md`, Jepson-approved t526u, 2026-10-29 22:18 MDT; the latest
direct owner instruction explicitly repeats this selection.

`2026-07-28` passes the bounded spike. MCP-02 retains product conformance and
shipping qualification; this decision does not claim a qualified product release.
There remains one product implementation calling `jope-client`.

Known-main hygiene/audit failures and the unavailable candidate no-regression
comparison remain recorded above; owner direction permits the handoff without
marking them passing. The unavailable local GitHub credentials are resolved by
the explicit division of work: this worker commits and exports the bundle;
ENG:watcher imports it on xbabe2, verifies the full SHA, pushes the branch and
opens the one PR using the established writer. No alternate GitHub identity is
used. Retain this lane and local evidence through review. The worker never merges.
