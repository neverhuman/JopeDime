# Design note

Compare two disposable stdio servers outside the product workspace: an exact-pinned
`rmcp` 3.8.0 server and a minimal extraction of the JSON-RPC validation, lifecycle,
version negotiation and tool/error framing in `crates/jope-runtime/src/service.rs`
at `5c8a16711632b4b8c588c9a16679a585abc0f135`. Both expose the same two bounded
tools, `echo` and `slow_echo`; no datasets, engine, model, credentials or network
access is available to either tool. A shared tool module keeps workload differences
out of the comparison. Each implementation gets its own dependency closure.

A raw stdio client records requests, responses, process exits and individual
assertions. Fresh processes isolate lifecycle failures. Exercise exact negotiation
and initialize/initialized ordering for 2024-11-05, 2025-06-18 and 2025-11-25;
exercise discovery, per-request metadata and handshake-free dispatch for
2026-07-28. Check framing, IDs, notifications, both tools, protocol versus tool
errors, malformed input, EOF and responsiveness during a slow call. Version
fallback is recorded as unsupported, never as a pass for the requested version.
The extracted dispatcher retains its serial execution model so the spike can
measure its limitation rather than silently repair the product design.

Rejected alternatives: changing product crates before OD9, comparing different
tool workloads, using an SDK client that could hide wire incompatibilities, or
claiming all of gate A4 from a two-tool fixture. The spike supports the OD9 choice;
it does not qualify engine integration, cancellation, named clients or all of A4.

Risks: registry and upstream protocol revisions can move; pin dependencies and
retain source URLs, versions, hashes and generated receipts. SDK defaults may
enable unnecessary dependencies; use only server/stdio features and inspect the
resolved tree with cargo-deny. The 2026 protocol is stateless: a successful legacy
handshake alone supplies no evidence for it. OWNER selected rmcp 3.8.0 in the
Jepson-approved t526u scope ruling. Spec clauses: §7.1–7.4 and §16 OD9. No frozen
contract changes.

Source: main@5b72f953cbdc987a12528c50e0f45ff388331908 → v1/mcp-01-conformance-spike-rmcp-vs-exis (ENG:watcher inserts the full imported HEAD SHA when publishing this body).

Behavior changed: Add two isolated stdio fixture servers and a bounded raw-wire
comparison outside the product workspace. rmcp 3.8.0 passes all four requested
version suites; the generalized dispatcher passes the three legacy suites and
refuses modern requests. The SDK keeps fixture fast-call p95 below 1 ms and
suppresses cancelled replies; the serial dispatcher measures about 180 ms.
OWNER ruled OD9 as rmcp 3.8.0. No product behavior is changed.

Gates closed: none in full. MCP-01's two implementations, report, dependency trees
and recommendation are complete, and OWNER has recorded OD9. This does not close
the real-client/fitting-load A4 gate. All eight task commands were executed, but
hygiene/audit are not green and a committed-candidate audit comparison is unknown.
The Jepson-approved t526u ruling in `~/nudges/scope_2218.md` (2026-10-29 22:18 MDT)
selects rmcp 3.8.0 and explicitly says KI-1/2/3 known main failures are recorded,
not blocking. [The owner-ruling receipt](research/mcp-spike/evidence/owner-ruling.json)
preserves the relevant text, source-file hash and later direct delivery instruction.

Evidence: [REPORT.md](research/mcp-spike/REPORT.md),
[version/source/binary receipt](research/mcp-spike/evidence/replay.json),
[raw wire](research/mcp-spike/evidence/replay.jsonl),
[dependency closures](research/mcp-spike/evidence/dependencies.json),
[initial check receipts](research/mcp-spike/evidence/checks.json),
[focused rerun](research/mcp-spike/evidence/recheck.json),
[corrected spike cargo-deny](research/mcp-spike/evidence/spike-deny.json),
[latest hygiene](research/mcp-spike/evidence/hygiene-latest.json) and
[baseline audit](research/mcp-spike/evidence/baseline-audit-verification.json).
Raw-wire SHA256: `94b25b99526581a009b2619fdb63a1f86673d790dc80c8c8327449a98d730d42`.
Protocol experiment source base: `3346ae2c00d86e7d6b52d6048ca3a87d4ad7bb20`;
receipt hashes bind all executed fixture/harness source and binaries. Machine,
OS, UTC, sample values and source schemas are retained. Product Rust inputs are
unchanged by the later upstream refresh. The bundled commit contains the candidate;
the bundle and externally reported full HEAD identify it for review.

Spec updates: none. Implementation stays within §7.1–7.4 and OD9; no frozen
contract or product dependency changes. [LUNA handoff cards](research/mcp-spike/FOLLOW_UPS.md)
refine the existing MCP-03/MCP-04 assignments and ownership.

Known limits: A two-tool raw-wire spike is not full upstream certification. Real
engine fitting, cancellation, resources, all 12 tools and named clients remain
with MCP-02/MCP-05. SDK malformed-JSON probes intentionally fail their diagnostic
expectation (no parse-error reply within 300 ms); the next valid request succeeds.
The dispatcher blocks behind slow work and returns late replies after cancellation.
The fixture wrapper's product-size/IO limits are not qualified. Candidate
no-regression audit remains unknown and both required reviewer passes are pending.

Next owner: ENG:watcher imports `~/nudges/mcp01.bundle` on xbabe2, verifies the
reported branch and full HEAD, pushes it through the established writer and opens
ONE PR. The latest direct owner instruction explicitly forbids this worker from
pushing; no GitHub credentials or account changes are needed here. Then
super_merger_codex and super_merge_claude each review the exact frozen head.
The implementing worker never merges.

## Gate self-review

**Status: READY_FOR_REVIEW — OD9 ruled, known main failures disclosed under t526u;
bundle publication assigned to ENG:watcher; two reviewer passes pending.**

- [x] Commit changes confined to `research/mcp-spike/`; compare the bundled head
  against its recorded main base. The predecessor's untracked inventory and hashes
  were verified before staging.
- [x] Both two-tool implementations, all requested version results, trees and
  recommendation are recorded with commands, source hashes and receipts.
- [x] OWNER selected rmcp 3.8.0 in t526u. Modern-version product shipping remains
  subject to MCP-02 qualification; the spike is not a product release claim.
- [ ] Every targeted check green and no Jankurai regression proved.
  Known main failures are retained as failed/unknown and accepted for this handoff
  by owner ruling; they are not marked passed.
- [x] No banned dependency in either fixture or the existing appliance closure.
  Corrected SDK cargo-deny succeeds under the unchanged root policy.
- [x] COMPLEX design note first; existing LUNA follow-ups refined without claiming them.
- [x] Spec §15.3 handoff fields present.
- [x] Commit includes Agent-Session, Agent-Model and Agent-Task trailers.
  Exact prepared text is in `research/mcp-spike/COMMIT_MESSAGE.txt`.
- [ ] ENG:watcher opens one PR through the established writer using gh.
- [ ] Both COMPLEX reviewer passes bound to the eventual full candidate SHA.
- [x] No GitHub CI, merge, GPU, xbabe0 access or other-screen action.

Local self-review: the candidate server and harness bytes are bound in the source
receipt and were checked against the current files. Both tool workloads and
failure framing are shared; modern support is tested without initialize and
fallback never counts as a requested-version pass. Negative and diagnostic
outcomes remain visible. Security review of the isolated lock/dependency changes:
rmcp is exactly pinned and archive-hash checked, SDK defaults/HTTP/auth/client
features are disabled, all four cargo-deny categories pass for the spike, no
banned native closure is added and no secrets/datasets are exposed to either tool.
This is same-agent local review, not either required reviewer PASS or an exact
committed-head approval.

## Local commands and recorded output

All Cargo work uses nice 10, eight jobs, incremental 0, line-table dev debug and
one persistent lane target. Tools absent from the host were checksum-verified
against `deploy/docker/runner.Dockerfile` and installed under that target only.
The source and time of each earlier run remain in its receipt.

```text
cargo fmt -p jope-runtime -- --check
exit 0 (no output)

cargo clippy -p jope-runtime --all-targets --locked -- -D warnings
exit 0; Finished dev profile

cargo nextest run -P pr --locked -p jope-runtime
exit 0; Summary: 86 tests run: 86 passed, 0 skipped

cargo deny check
exit 0; advisories ok, bans ok, licenses ok, sources ok

cargo tree -p jope-appliance -e normal | grep -E "onig|esaxx|tch|torch-sys|rusqlite|libsqlite3|pyo3|openssl-sys|aws-lc-sys|syntect" && exit 1 || true
exit 0; UNQUALIFIED because the supplied pipeline masks failures
Strict counterpart: python3 research/mcp-spike/check-dependencies.py
exit 0; dispatcher 7, rmcp3 54, appliance 89 unique packages; banned=[] for all

just hygiene
exit 1; see the exact latest-base failures in hygiene-latest.json
unassigned addition: python:research/flywheel_selfplay/test_phase3_gates.py
secret scan requires full HEAD history; fetch complete history before scanning
The recorded run's Git-based checks excluded then-untracked spike files;
candidate hygiene remains unqualified and the owner-accepted main failures remain visible

just lint-paths
exit 0; 139 hits in python/src (informational until WP6 lands)

just audit
exit 1 at 5c8a16711632b4b8c588c9a16679a585abc0f135
score=54 raw=54 caps=21 findings=380
Error: audit conformance blocked in standard mode: blockers=294
artifact_valid=true, audit_passed=false; candidate comparison unknown
```

Additional fixture checks: nested-workspace fmt and all-target Clippy pass;
`cargo deny --manifest-path research/mcp-spike/Cargo.toml --locked --config deny.toml check`
passes after correcting a removed CLI flag in an earlier attempt. The shared-tool
Unicode/schema boundary test passes (1 run, 1 passed, 0 failed/ignored). Ruff and
shell syntax pass. For exact historical failures and superseding results, retain
all receipts rather than replacing failures with a summary of the passing subset.
