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

Additional program builders, proposal traces and training workers
do not acquire narration coverage from these adapters. Symbolic and portfolio
program differences retain a verified whole-program replacement; recursive
explanations remain unqualified.

## End-to-end quickstart

Run these commands from the repository root with the repository Rust toolchain,
`jq`, and ordinary POSIX utilities installed. The bundled small dataset is local;
this example uses CPU execution. Keep the lane's existing Cargo target directory.

```sh
export CARGO_INCREMENTAL=0
export CARGO_PROFILE_DEV_DEBUG=line-tables-only
mkdir -p target
demo_dir="$(mktemp -d "$PWD/target/narration-demo.XXXXXX")"

JOPE_NARRATION_DIR="$demo_dir/narration" \
  cargo run --locked -p jope-cli --bin jope -- pipeline run \
    --dataset datasets/test/pmlb-229-pwlinear \
    --budget evaluations=4,seconds=120 \
    --rng-seed 11 --seed 7 --threads 2 --islands 1 --device cpu \
    --cuts 1 --policy skip --emit rust \
    --source-commit "$(git rev-parse HEAD)" \
    --output "$demo_dir/run" --json

journal="$(find "$demo_dir/narration" -maxdepth 1 -type f -name '*.jsonl' -print -quit)"
test -n "$journal"
run_id="$(jq -er '.run_id' "$journal.commit")"
printf 'Journal: %s\nRun: %s\nReceipt: %s\n' \
  "$journal" "$run_id" "$demo_dir/run/receipt.json"

cargo run --locked -p jope-observatory --bin gp-narration-stream -- \
  --journal "$journal" --run-id "$run_id" verify --require-complete
cargo run --locked -p jope-observatory --bin gp-narration-stream -- \
  --journal "$journal" --run-id "$run_id" replay --after 0 --limit 64 --text
cargo run --locked -p jope-observatory --bin gp-narration-stream -- \
  --journal "$journal" --run-id "$run_id" serve --listen 127.0.0.1:9877
```

Open **http://127.0.0.1:9877/monitor**. The final command stays in the foreground;
Ctrl-C stops this reader. Keep the printed journal and run identity for the
commands below, or use another terminal with those variables assigned. A fresh
demo directory ensures the discovery step selects this one run. When opening an
existing capture directory, explicitly select the intended journal.

Four evaluations are a smoke example and may contain only initial candidates.
A fresh run with a larger evaluation budget can reach later generations; the
presence of a journal alone does not prove later-generation edit coverage.

To see later-generation changes, repeat the producer with a fresh directory and a
larger budget (then repeat journal discovery and the reader commands):

```sh
edit_dir="$(mktemp -d "$PWD/target/narration-edits.XXXXXX")"
JOPE_NARRATION_DIR="$edit_dir/narration" \
  cargo run --locked -p jope-cli --bin jope -- pipeline run \
    --dataset datasets/test/pmlb-229-pwlinear \
    --budget evaluations=20,seconds=120 \
    --rng-seed 11 --seed 7 --threads 2 --islands 1 --device cpu \
    --cuts 1 --policy skip --emit rust \
    --source-commit "$(git rev-parse HEAD)" --output "$edit_dir/run" --json
journal="$(find "$edit_dir/narration" -maxdepth 1 -type f -name '*.jsonl' -print -quit)"
run_id="$(jq -er '.run_id' "$journal.commit")"
```

## Read the retained stream

The quickstart assigns `$journal` and `$run_id` from the selected capture.
For later inspection, start the Rust reader against that same pair:

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

For a finite HTTP page from another terminal:

```sh
curl --fail --get http://127.0.0.1:9877/api/v1/gp/reasoning/events \
  --data-urlencode "run_id=$run_id" \
  --data-urlencode after=0 --data-urlencode limit=64
```

A WebSocket client connects to
`ws://127.0.0.1:9877/api/v1/gp/reasoning?run_id=RUN_ID&after=0`, replacing
`RUN_ID` with the printed identity. Reconnect with the same run and the last
fully handled event's decimal sequence. Heartbeat and error frames are transport
messages; do not advance the event cursor from them.

For MCP, launch the same reader as a separate stdio process:

```sh
cargo run --locked -p jope-observatory --bin gp-narration-stream -- \
  --journal "$journal" --run-id "$run_id" mcp
```

It implements a finite newline-delimited JSON-RPC compatibility profile for protocol 2025-11-25:
initialize, send `notifications/initialized`, then call `narration_read` or
`narration_inspect`. Stdout contains protocol frames only. This is a separate
reader process, not a search-control tool.

For clients accepting the common `mcpServers` JSON shape, the executable and
arguments are:

```json
{
  "mcpServers": {
    "jope_narration": {
      "command": "/absolute/repo/target/debug/gp-narration-stream",
      "args": [
        "--journal", "/absolute/repo/target/narration-demo.EXAMPLE/narration/SELECTED.jsonl",
        "--run-id", "ACTUAL_RUN_ID",
        "mcp"
      ]
    }
  }
}
```

Replace every placeholder with the printed paths/identity. Client configuration
keys vary; the native executable and argument list are the portable contract.
The preceding Cargo command builds the binary in the ordinary target location.
For an explicit custom target directory, resolve its actual executable instead.

## Stored and wire schemas

| Identifier | Meaning |
| --- | --- |
| `jope.reasoning-event/v1` | Journal envelope and event delivered by all readers |
| `jope.lineage-fact/v1` | Owned source-fact contract; envelope `fact` uses a `kind` tag |
| `controlled-search-commentary/3` | Current writer renderer; evaluation-bound and progress scope |
| `controlled-search-commentary/2` | Retained reader with earlier admission/refusal wording |
| `controlled-search-commentary/1` | Retained reader for earlier journals |
| `jope.narration-page/v1` | Finite page with `events`, `next`, `more`, `partial_tail` |
| `jope.narration-status/v1` | Validated boundary and capture state |
| `jope.narration-heartbeat/v1` | WebSocket transport health |
| `jope.narration-error/v1` | WebSocket read/limit failure |

An event retains `schema`, `renderer`, `run_id`, decimal `sequence`, run-scoped
`cursor`, `fact`, `links`, `narration`, `previous_sha256` and `sha256`.
`narration` retains text, category, checked claims, exact delta when available,
and the presentation variant. Source RNG seeds are decimal strings too. Keep
these strings intact instead of coercing them through JavaScript `Number`.
The adjacent watermark retains `run_id`, `records`, `dropped`, `sha256`,
`bytes`, `status` and `error`. Integrity hashes are hexadecimal SHA-256 values.

The reader CLI's global `--max-records` and `--max-journal-bytes` limit its
index/input; they do not change producer retention. Defaults are 1,000,000 records
and 1 GiB. Accepted reader limits are 1..=1,000,000 records and 1..=16 GiB.
Finite pages contain at most 256 records / 4 MiB. WebSocket `serve --max-clients`
defaults to 32, accepts 1..=128, and shares at most 64 concurrent read jobs.
Writes time out after 10 seconds; quiet heartbeats arrive about every 5 seconds.
MCP accepts at most 64 KiB per request and 16 MiB per response.

## Exactness and limits

Pipeline edits replay onto the retained parent and must reproduce the exact
child codec. Integer and float values retain their types and bits. Historical
printed programs preserve only the precision their producer printed. An endpoint
diff describes the net change; it does not prove which internal edit operation or
parent splice ran. Ordered equal parents remain two roles.

Some legacy defaults, aliases and scalar clamps have explicit implementation
notes. These describe requested versus resolved parameters, without promising
equal predictions or unchanged fitted state. Outcomes report their actual cut
and metric. A partial-cut measurement is not an all-cut selection result.

The producer queue defaults to 8,192 records and 4 MiB of owned serialized
payload, with a 1 MiB frame bound. Admission can drop on record/byte saturation or oversize; aggregate gap events disclose losses with approximate position.
No search RNG draws, evaluation budgets or controller decisions come from the
narrator. An observer can still add time, so time-budget searches have no promise
of identical decisions.

Automatic environment capture permits at most 64 session attempts per process.
The first rejected attempt reports an observer error; later attempts stay disabled
without repeated warnings and preserve the source result.
Retention across process restarts remains an operator responsibility.

Capture defaults to at most 1,000,000 events and 1 GiB of journal data. The
writer constructs one retained wording, validates it by rederivation, writes a full record, syncs bounded batches of journal data, then
atomically publishes a synced commit watermark. All reader transports expose
only that committed prefix. Keep the journal and watermark together. The current
writer uses renderer v3. Renderer v1 and its original two golden transcripts remain
available for retained replay, together with the separate v2 transcript. V3 has
its own create-only transcript, including every admission disposition and stop
reason. V2 wording can overstate the bound reached by multi-cut exploration;
v3 uses a phase-neutral bound and describes exhaustion as consecutive generations
with no evaluation progress. Unsupported revisions are explicit errors. Readers
withhold bytes beyond the watermark after a crash, even when they contain a
newline. Corruption inside the committed prefix fails closed.

Capture state is separate from source outcome: `complete` means accepted capture
drained with no queue loss; `incomplete` means queue losses occurred; `failed` or
`resource_limit` closes capture without a completeness claim. `capturing` does
not prove the process is alive: after a crash or unpersistable I/O error, closure
is unknown. `RunStopped` describes only the source run. `finish` closes admission
even with retained tap clones, drains accepted work and returns writer errors.
A stale `capturing` marker makes `follow` wait until interrupted; readers cannot
infer producer death from silence. Public search/prediction entrypoints report observer startup/finish errors on
stderr and preserve the authoritative result.

| Symptom | Action |
| --- | --- |
| No journal | Check capture env, stderr startup error, directory access and process session cap. The source result remains authoritative. |
| Valid prefix, incomplete capture | Inspect watermark status and gap counts; do not treat it as complete coverage. |
| `capturing` after producer disappears | Closure is unknown. Preserve both files and interrupt a waiting reader when needed. |
| Wrong-run/future cursor | Use the selected watermark identity and a sequence within the retained boundary. |
| HTTP 503 | Read `error.code`: inspect storage/integrity for `journal_unavailable`. For `request_limit` or `client_limit`, capacity is busy; reconnect with the retained cursor. |
| HTTP 403 `forbidden_origin` | Use the exact printed numeric host and port, such as `127.0.0.1:9877`; `localhost` is a different Host value. |

The native single-island observation omits its island field; a measurement can
therefore say “island unavailable” even when the draw/start coordinate is island
0. This reflects the retained observation field, without guessing a missing ref.

Hashes detect internal inconsistency; they do not authenticate the author.
Transport heartbeats describe delivery health, not scientific progress.

## Verification

`bash ops/ci/narration.sh` runs strict Clippy and the focused core, producer,
transport and prediction contracts. Maintained Rust CI calls that same script.
The focused contracts currently execute 52 tests: core 21, pipeline internal 5,
public producer 4, transport 15, prediction 6 and prediction codec 1. This count
is local contract coverage, separate from the full repository gate and hosted CI.

No representative latency/overhead benchmark is qualified by these contracts.
Provisional research targets are p95 difference/planning below 2 ms, p95 event
visibility below 250 ms and fixed-count overhead below 2%; these are unmeasured
targets. Exact-head hosted CI, long-duration load, human quality
evaluation, deployment and additional-engine coverage require separate evidence.
See the [engineering specification](coordination/search-narration-engineering-spec.md)
for the acceptance and follow-on qualification matrix.

Use `verify --require-complete` when the caller needs capture completeness as well
as prefix integrity. `follow` stops at terminal closure, returning an error for a
lossy or failed capture. Ctrl-C exits `follow` with status 0 without certifying
capture closure; inspect `verify --require-complete` separately. Prediction identities use a streaming typed encoding
with exact float bits, including non-finite payloads.
