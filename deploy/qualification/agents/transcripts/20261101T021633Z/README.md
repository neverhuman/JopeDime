# MCP-05 A5: three actual named-client flows

All three admitted clients completed AG1 at runtime source
`21d65a949bbfc340981cfbcf224cacb9eb571e5c`: **48 successful MCP tool calls,
zero failed calls**, with one persisted run per isolated store after an identical
`run_start` retry. This is source-bound A5 evidence ready for review. The outer
capture verifier reports its A5 label as unverified; it checks the necessary retry
subpath and integrity, while the accompanying semantic receipts check the full flow.
Review and merge remain with ENG:watcher under `mcp01_t2020.md`.

| Actual client | Pinned version | Run | Complete flow |
| --- | --- | --- | --- |
| Claude Code | 2.1.307 | `run_01M6AEASABBCEEZGMAGS95KD58` | 16 successful calls |
| Antigravity Gemini CLI | 1.4.12 | `run_01M6AEBH3H0NFPM2K879AQD3HP` | 16 successful calls |
| Codex CLI | 0.166.0 | `run_01M6AEAT8ZHF9KR5YCSXZZDM2N` | 16 successful calls |

The watcher explicitly admitted Antigravity/Gemini as the third client in
`mcp05_t1925_jrb02.md`. Its actual protocol identity is `antigravity-client v1.0.0`;
this is not represented as a Google Gemini CLI version. Requested models were
`claude-opus-5-5`, `gemini-4-argon-high` and `gpt-6.1-sol`, respectively. Provider
model identifiers are not independently attested by an invocation argument.

Each flow includes system information, an empty complete run inventory, dataset
registration and identical retry, inspection, a dry-run CPU plan, creation, a
complete inventory **between creation and retry**, identical creation retry, a
complete unchanged inventory, terminal polling and six contiguous events, bundle
export and identical export retry, run-based model prediction and final inventory.
The strict retry verifier selects counts `0 → 1 → 1` and proves ID-set equality.
Every client, engine, MCP server and recorder exited 0; no engine was force-stopped.
Every prediction file has 25 rows and SHA256
`504063ac11b66972f872e4205b2021b1ff32e8c718f8c0b6656cc314df4d7045`.

## Design and scope

Keep the existing rmcp 3.8.0 lifecycle and bounded transport. Two actual Gemini
failures required narrow adapter repairs: permit its legacy roots-change hint
only while bootstrapping after valid modern discovery, and retain rmcp's native
JSON text fallback so a text-only client receives the same real IDs as
`structuredContent`. Resource links, API schemas, error handling, frame limits,
dependencies and backend behavior retain their existing implementations. Four
bootstrap regressions and the four-version canonical-tool check cover the fixes.
A transport that fabricated replies or a prompt that supplied hidden IDs could
not establish interoperability, so neither is used. Risks are accidentally
relaxing bootstrap validation or exposing different public data through text;
negative bootstrap tests and text/structured equality checks address those risks.
Spec: AG1, §§7.1–7.4, A5, §15.3.

The first Gemini attempt at `842753e7…` completed discovery/registration but the
client saw only “dataset_register completed” and stopped without a run. Its
partial output, the older bootstrap failure, both red regressions and intermediate
compile/toolchain failures remain in external custody. The two successful earlier
client flows are retained there; this package uses fresh runs of **all three** at
the corrected source. No waiver is used for the third-client path.

## Input, execution and limits

Input is catalog dataset `b4c8807a75fc190f`, its already-preserved 25-row real-S3
TRAIN CSV only. [source-and-input.json](source-and-input.json) pins the original
TRAIN bytes, the header-prepared CSV, executable, capture tools and resource
limits. Its internal 20/5 random split is entirely within approved TRAIN; no
external test, sealed or external holdout payload was read. Each run used one
CPU worker, one evaluation, a five-second nominal budget, heuristic guidance,
no landmarks and no DOPE/GPU. Existing client authentication was used without
new logins, purchases or shared configuration edits. No credentials, raw TRAIN,
predictions, model payloads or client reasoning are committed. Client output is
filtered to tool interactions; MCP wire/raw streams remain unmodified.

All Cargo work ran at nice 10 or lower priority on xbabe4, four build jobs, one
reused target. The installed remote helper required absent Docker, so the claimed
native Mac worktree ran 23 MCP tests and a verified Zig cross-toolchain produced
the Linux musl executable. The copied binary hash is
`9c6bdd2146f81fade386d5528543bd20598ca5ad2d85f49da1a3522f1057dc16`.
Runtime clients/engines ran on xbabe1 at nice 19, two allowed CPUs and one math
thread. Disk stayed below the 93.8% hard stop. No flywheel job was touched.

MCP tests: 23 passed, 0 failed/ignored, 125 excluded by the explicit `mcp::` filter.
Linux-target Clippy with `-D warnings`, changed-file rustfmt, coverage validation,
repository layout, Rust scope, README benchmark check and test-inventory check
pass. The first test-inventory invocation used an invalid subcommand (exit 2);
the corrected `verify-test-inventory` passed. Native Mac Clippy failed on eight
existing non-MCP platform warnings; Linux-target Clippy passed without suppressions.
`just` is absent on xbabe1: hygiene/audit recipes are **NOT_RUN**, not passing.
The exact lint-paths predicate passes. No GitHub CI was dispatched under the
latest watcher instruction binding master plan §3; full qualification CI is a
separate gate. No dependency changed.

The exported archive/model manifest and every materialized part were hash-checked;
the Rust predictor exists and export retries returned identical arguments/results.
Only final filesystem bytes are attested, with no interval inode/mtime claim.
Model parity remains `unavailable`, qualified set `not-qualified`, learned guidance
remains unqualified, and no release, signing, speed or scientific quality claim is
made by these client-flow results.

## Verify the committed capture

From the repository root, with the reviewed manifest pin in the handoff:

```sh
python3 deploy/qualification/agents/transcripts/run-prep/verify_capture.py \
  deploy/qualification/agents/transcripts/20261101T021633Z/manifest.json \
  --check-run-retry --manifest-sha256 REVIEWED_MANIFEST_SHA256
```

Per-client `*.verification.json` receipts also validate raw hashes/paired frames,
text/structured equality, TRAIN profile, bounded plan, completed run/events,
export manifests and prediction file hashes. `*.execution.json` retains the raw
runner's initial `NOT_QUALIFIED` default label; it is not overwritten into an
automatic gate decision. External replay commands, analyzer and full client
stdout remain at `~/jope_plan/evidence/MCP05-A5-sol_v1_mcp01-t1930` on xbabe1;
private stores/models remain under `~/jope-data/mcp05-a5-t1930`.
