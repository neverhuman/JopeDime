# MCP-05 named-client evidence

The [2026-11-01 three-client replay](20261101T021633Z/README.md) records complete
AG1 flows in Claude Code, owner-admitted Antigravity Gemini CLI and Codex CLI:
48 successful calls, identical run retries with complete inventories `0 → 1 → 1`,
exports and 25-row predictions. Its exact-source evidence is ready for review;
ENG:watcher owns publication and merge. Model parity and learned guidance remain
separate qualification gates.

The historical baseline below remains **A5: NOT QUALIFIED**: two clients stopped
at registration and Gemini was unavailable in that earlier host inventory.

## Design and scope

Run the installed, version-pinned clients against isolated engine storage on
xbabe2. A transparent stdio relay records the actual MCP messages separately from
each client's output. Keep source, executable, input, command, time and exit
identities with the capture. Preserve real capability failures and stop rather
than invent a dataset/run ID. Only this transcripts directory changes; backend
integration remains with its owners. Spec references: AG1, §§7.1–7.4, A5 and §15.3.

Mock responses, legacy fitting tools and batch-command substitutions were rejected
because they cannot establish the requested named-client path. The principal
risk is that merged dependency scaffolds still lack executable HTTP operations.

## Observed runs

Capture: [20261031T015221Z/manifest.json](20261031T015221Z/manifest.json).

| Client | Exact CLI version | Observed result |
| --- | --- | --- |
| Claude Code | 2.1.314 | `system_info` succeeded; two identical-key `dataset_register` calls returned `CAPABILITY_UNAVAILABLE` / 422. Client and engine exited 0. |
| Codex CLI | 0.165.0 | `system_info` succeeded; two identical-key `dataset_register` calls returned `CAPABILITY_UNAVAILABLE` / 422. Client and engine exited 0. |
| Gemini CLI | unavailable | No executable found in the recorded host inventory; no client run or version claim. |

Claude reported model `claude-opus-5-5`; Codex was invoked with `gpt-6.1-sol`.
The raw command, prompt, client JSONL, MCP JSONL, stderr and execution receipt are
retained for each available client. The manifest hashes every transcript sidecar.
Claude used `server/discover`; Codex used the initialize lifecycle. These are
observations of the captured clients, not a protocol-conformance qualification.

Both clients made exactly three tool calls: `system_info`, `dataset_register`,
and the identical registration retry. The forced retry is a diagnostic of a
rejected mutation even though the error reports `retriable:false`. **Two rejected
registrations do not prove successful mutation or run idempotency.** No dataset
ID or run ID was returned. Dry-run planning, run creation/retry, persisted run
count, polling/events, export and prediction were not executed. No run-count
measurement is available; do not interpret the absence of a run ID as such a check.

## Source and input

Engine source: `5ad6f5e05fc2161f5b62b0982169ead85e68751e`, product 0.4.1, default
engine/client/MCP features, development profile, no CUDA or embedded package.
`remote_test.sh auto` built it on xbabe3, with the source SHA embedded through a
Cargo command-line environment setting. The copied executable's hash was checked
against the remote file before and after transfer; `--version --json` reports the
pinned source. Each client used separate task-local storage and its engine was
stopped and exited 0 after capture. No shared client configuration was edited.

Input is the catalog's real-S3 tiny regression dataset `b4c8807a75fc190f`, using
only the 25-row training CSV already preserved by QLT-05. Its SHA-256 is
`782acd17eed2bfc7acfa4438be076fb63026763e6578661cfc284c2f3f9a21e1`.
The source CSV is headerless; preparation prepended exactly the column names from
its hash-verified metadata and preserved every original CSV byte. The manifest
records both input identities. No test/holdout/sealed rows were opened; no raw
rows, model fitting or new dataset payloads are included here. Registration failed
before this input could be admitted by the engine.

## Replay and remaining work

The command and prompt sidecars record the exact original invocations and local
paths. Capture machinery, build logs, artifact hashes, input preparation and
verification receipts are retained at `~/jope_plan/evidence/MCP-05/gb0145`; these
paths are host-local and the command files are not portable setup fixtures.
Codex's temporary MCP configuration followed the
[official MCP configuration documentation](https://developers.openai.com/codex/mcp)
and this installed version's help. No global MCP server entry was added.

The pinned [HTTP router](../../../../crates/jope-service/src/http/router.rs)
returns a capability error for operations other than `system_info`.
The existing external task `master/tasks/SVC-11-F01-wire-dataset-http-registration.md`
tracks dataset HTTP integration. Run lifecycle, export and prediction handlers
must also be executable through that same authenticated API before full AG1 can
be qualified. MCP-05 does not grant ownership of those paths. After those owners
land their integrations and Gemini CLI is available, rerun all three named clients
at the admitted source and verify same-key `run_start` replay against the persisted
run inventory. A successful client process exit alone does not close A5.
