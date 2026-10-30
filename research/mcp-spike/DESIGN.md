# MCP-01 design note

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
handshake alone supplies no evidence for it. OWNER must record OD9; this worker
only recommends. Spec clauses: §7.1–7.4 and §16 OD9. No frozen contract changes.
