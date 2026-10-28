# Search commentary

Search commentary explains recorded program edits, evaluation progress, measurements,
and selection. It is generated from those facts. Each event retains its source fact,
native evidence references, checked claims, exact differences and displayed text.

Enable capture by setting `JOPE_NARRATION_DIR` to an existing or creatable directory
before running the ordinary pipeline command. It writes a create-only
`<timestamp>-<pid>.jsonl` journal and an adjacent `.jsonl.commit` watermark. The
legacy `JOPE_LINEAGE_FACT_DIR` name is also accepted. Capture is opt-in; stdout
keeps the command's existing response format.

The initial search adapter covers the legacy pipeline's exploration, comparison,
migration and public `run`/`invent` entrypoints. `invent` reports ranked offers;
the consuming engine makes its own final selection. Fixed prediction is separately
observed through `jope_runtime::api::predict` and the shipped
`org.jopedime/math-pipeline/v1` table/cell prediction methods. Prediction events
bind actual fitted state and inputs by digest without publishing input cells.

Kernel2, neural proposal traces, advanced-engine evolution and training workers
do not acquire narration coverage from these adapters. Symbolic and portfolio
program differences retain a verified whole-program replacement; recursive
explanations remain unqualified.

## Read the retained stream

Obtain the run identity from the selected journal's `.commit` file. Start the
Rust reader against that fixed journal and identity:

```sh
cargo run --locked -p jope-observatory --bin gp-narration-stream -- \
  --journal "$journal" --run-id "$run_id" verify
cargo run --locked -p jope-observatory --bin gp-narration-stream -- \
  --journal "$journal" --run-id "$run_id" replay --after 0 --limit 64 --text
cargo run --locked -p jope-observatory --bin gp-narration-stream -- \
  --journal "$journal" --run-id "$run_id" follow --after 0 --text
cargo run --locked -p jope-observatory --bin gp-narration-stream -- \
  --journal "$journal" --run-id "$run_id" serve --listen 127.0.0.1:9877
```

The browser monitor is at `/monitor` (also `/`). It supports pause/resume,
reconnect, filtering, concise/audit views and evidence inspection. Its local
view retains at most 500 cards; the journal remains available for older events.

The service provides:

| Route | Result |
| --- | --- |
| `/api/v1/gp/reasoning?run_id=RUN&after=0` | WebSocket replay followed by committed live events |
| `/api/v1/gp/reasoning/events?run_id=RUN&after=0&limit=64` | Finite JSON page |
| `/api/v1/gp/reasoning/status` | Validated boundary, withheld tail and capture closure |

Cursors are canonical decimal **strings**, exclusive and scoped to a run.
`after=12` starts with event 13. A resumed nonzero cursor requires `run_id`.
Wrong-run, malformed and future cursors are errors. HTTP client errors are 400;
unavailable or corrupt journals are 503. Binding is restricted to loopback with
exact Host/Origin checks. Network requests cannot select a filesystem path.

For MCP, invoke the same binary with `mcp` instead of `serve`. It implements a
finite newline-delimited JSON-RPC compatibility profile for protocol 2025-11-25:
initialize, send `notifications/initialized`, then call `narration_read` or
`narration_inspect`. Stdout contains protocol frames only. This is a separate
reader process, not a search-control tool.

## Exactness and limits

Pipeline edits replay onto the retained parent and must reproduce the exact
child codec. Integer and float values retain their types and bits. Historical
printed programs preserve only the precision their producer printed. An endpoint
diff describes the net change; it does not prove which internal mutation arm or
crossover cut ran. Ordered equal parents remain two roles.

Some legacy defaults, aliases and scalar clamps have explicit implementation
notes. These describe requested versus resolved parameters, without promising
equal predictions or unchanged fitted state. Outcomes report their actual cut
and metric. A partial-cut measurement is not an all-cut selection result.

The producer queue defaults to 8,192 records and 4 MiB of owned serialized
payload, with a 1 MiB frame bound. Admission can drop on record/byte saturation or oversize; aggregate gap events disclose losses with approximate position.
No search RNG draws, evaluation budgets or controller decisions come from the
narrator. An observer can still add time, so time-budget searches have no promise
of identical decisions.

Automatic environment capture permits at most 64 session attempts per process;
subsequent attempts report an observer error and preserve the source result.
Retention across process restarts remains an operator responsibility.

Capture defaults to at most 1,000,000 events and 1 GiB of journal data. The
writer validates, renders once, writes a full record, syncs bounded batches of journal data, then
atomically publishes a synced commit watermark. All reader transports expose
only that committed prefix. Keep the journal and watermark together. Readers
withhold bytes beyond the watermark after a crash, even when they contain a
newline. Corruption inside the committed prefix fails closed.

Capture state is separate from source outcome: `complete` means accepted capture
drained with no queue loss; `incomplete` means queue losses occurred; `failed` or
`resource_limit` closes capture without a completeness claim. `capturing` does
not prove the process is alive: after a crash or unpersistable I/O error, closure
is unknown. `RunStopped` describes only the source run. `finish` closes admission
even with retained tap clones, drains accepted work and returns writer errors.
Public search/prediction entrypoints report observer startup/finish errors on
stderr and preserve the authoritative result.

Hashes detect internal inconsistency; they do not authenticate the author.
Transport heartbeats describe delivery health, not scientific progress.

## Verification

`bash ops/ci/narration.sh` runs strict Clippy and the focused core, producer,
transport and prediction contracts. Maintained Rust CI calls that same script.
Exact-head hosted CI, long-duration load, latency/overhead targets, human quality
evaluation, deployment and additional-engine coverage require separate evidence.
See the [engineering specification](coordination/search-narration-engineering-spec.md)
for the acceptance and follow-on qualification matrix.
