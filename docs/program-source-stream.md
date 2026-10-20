# Complete program source stream

`gp-source-stream` maps captured program occurrences to the existing standalone training-recipe
emitters. It neither fits a model nor changes selection rules, RNG draws, reward, numerical evaluation, model
weights or serving defaults. Full source includes its runtime and documentation, not just a
DSL string, excerpt or a reference to a private library. C++17/JavaScript source exposes library
entry points; the existing emitter harness supplies CSV drivers when executing parity tests.

## Capture and delivery

Set `JOPE_GP_SOURCE_DIR` to an external directory **before the producer process starts**.
Capture is otherwise disabled; the opt-in setting is read once per process. Capture and
emission consume real wall time, so time-limited producers can complete fewer evaluations.
Keep capture settings identical in timed comparisons. Every materialized complete program at
a supported capture point is retained, including repeated programs. No sampling or unique-program
token estimator is used. Partial expressions enter only through a complete executable recipe.
Existing already-running binaries are unchanged; automatic capture requires the updated code
and setting. External producers can append through
`feat_math::source_stream::Journal::create` / `append` using an engine and coordinate.

Each process owns a create-only JSONL journal with unique occurrence IDs. Identical programs at
separate capture points retain separate IDs. Journal failures stop enabled producers visibly.
The producer writes complete lines and flushes them to the operating system; producer journals
are not fsynced per candidate, so they do not promise recovery from host power loss. The service
fsyncs its emitted event log and source files before publication completes. Keep both directories
on durable storage and preserve them after failures. Do not put bulk output in the Git tree.

The collector and compiler use separate workers. Source is published before compiler checks.
Mapping does not wait for compiler processes; both workers serialize durable event publication,
so disk I/O and validation-record publication can still limit throughput. Each scan processes
a bounded turn of eight records per producer and immediately continues a backlog. A hot or
large old journal cannot monopolize the collector. Compiler work follows source arrival in
FIFO order, including after restart; lexical IDs cannot move future work ahead of older work. Journal/event limits stop the service
visibly; producer journals remain intact. No bounded broadcast queue discards a slow client's
messages: each WebSocket reads the durable event log at its own cursor. A client that cannot
accept a write within 10 seconds disconnects and can resume. Each source hash is checked before
compilation; identical sources reuse the same successful compiler result within a service process, but
every emitted occurrence still contributes its actual token count. Failed, missing-tool and timed-out checks are retried for later occurrences rather than cached.
Restart rebuilds metrics and
pending checks from the log; replay and a re-read journal never create new token credit.

## Capture coverage

Updated producers capture materialized complete programs at the instrumented construction and
execution boundaries. Repeated programs at separate capture points remain distinct occurrences.
The counter records captures rather than independent numerical fits. External/custom producers
must append explicitly; already-running binaries and historical archives are unchanged.
The developer implementation/parser coverage record is
`docs/coordination/gp-source-stream-capture-coverage.md`.

| Language row | Existing emitter | Actual check | Failure class |
| --- | --- | --- | --- |
| `rust` | `transpile::emit` Rust source | `rustc --crate-type=lib --emit=metadata` | Emitter refusal or compiler/launch/timeout failure |
| `python` | `transpile::emit` Python source | Python built-in `compile` | Emitter refusal or parse/launch/timeout failure |
| `cpp17` | `transpile::emit_cpp17` | `g++ -std=c++17 -fsyntax-only` | Narrower certified coverage is an explicit emission error; compiler failures lose emitted tokens |
| `javascript` | `transpile::emit_javascript` | `node --check` | Narrower certified coverage is an explicit emission error; check failures lose emitted tokens |

Enabling capture in a policy/learning process also captures teacher-forcing and encode-validation
decode replay. This can create many extra labeled occurrences and reach storage limits quickly;
a journal I/O failure panics in that enabled process. Choose a dedicated external run directory.
The active `grammar/automaton.rs` source change mints teaching-store semantic-dependency and
corpus scan-policy successors. Verification refuses receipts bound to the previous listed-source
hashes; retain old receipts under their original identity. No historical evidence is relabeled.

All four rows are emitted for every captured occurrence, including failures. Source begins
pending; only successful checks qualify its tokens. No unsupported operation is replaced with
another program to improve yield. C++20 compilation is a mode over C++17 source, not another
emitted language. Use release builds for production throughput; test/debug timing is not a
production throughput claim.

## Protocol

Default endpoint: `ws://127.0.0.1:9876/api/v1/gp/source?after=0`.
Metrics: `http://127.0.0.1:9876/api/v1/gp/metrics`.
A successful WebSocket upgrade streams one JSON event per text message. `after` is the last
successfully received sequence, exclusive; reconnect from zero to replay everything. A future
cursor is refused. Save a cursor only after retaining that event locally.

Envelope fields are `schema: "jope.gp-source/v1"`, monotonic `sequence`, `at_unix_ns`, and `kind`.

- `source`: occurrence (`schema`, unique `id`, `engine`, `coordinate`, complete `program`),
  `program_sha256`, measured `emit_ns`, and exactly four `artifacts`. Each row names its
  `language`, full `source`, `sha256`, `bytes`, `code_tokens`, `doc_tokens`, initial `status`,
  and any `error`. Unsupported emission has null source/hash, zero output tokens and failed
  status. It remains an artifact error; no guessed source/token count is credited.
- `checked`: `occurrence_id`, `language`, `source_sha256`, terminal `status` (`passed` or
  `failed`), `tool` with invocation/version, retained `diagnostic` and measured `elapsed_ns`.
  A cache reuse names that reuse and records zero additional compiler elapsed time.

Browser client, from an allowed same-origin page (or a standalone JavaScript/WebSocket client
that omits Origin):

```js
let last = Number(localStorage.getItem("jope-gp-sequence") || 0);
const socket = new WebSocket(`ws://127.0.0.1:9876/api/v1/gp/source?after=${last}`);
socket.onmessage = ({data}) => {
  const event = JSON.parse(data);
  if (event.kind === "source") {
    for (const artifact of event.artifacts) {
      console.log(event.occurrence.id, artifact.language,
                  artifact.source ?? artifact.error);
    }
  } else console.log(event.occurrence_id, event.language, event.status);
  localStorage.setItem("jope-gp-sequence", String(event.sequence));
};
```

At most 32 WebSocket clients connect concurrently; cursor lookup and journal reads run outside
the async request threads. The server binds loopback only and checks Host and any supplied Origin. Use the exact
`127.0.0.1:9876` authority when tunneling. Foreign website origins are refused. This is a local
operator endpoint; it has no public authentication/deployment flow. Use an SSH tunnel for
remote access. `after` only controls replay, never generation or numerical work.

## Metrics and errors

`jope.gp-source-metrics/v1` uses the existing `feat_math::tokens` lexical rules: identifiers,
literals and punctuation/operators plus words in comments/docstrings. Whitespace is not a
token. These are source tokens, not neural BPE tokens. All four languages use the same explicit
ledger and all generated code/documentation tokens enter their language total and aggregate.

`emitted_tokens = passed_tokens + lost_tokens + pending_tokens` for each language and the total.
`token_yield = passed_tokens / emitted_tokens` is a fraction, null before any source exists.
Failed compilation charges every token in that artifact as loss. Pending checks remain pending;
neither source emission nor a replay proves compilation. `artifact_yield = passed_artifacts /
artifacts` additionally exposes emitter failures with no source-token denominator. Both
`emission_errors` and `compile_errors` are explicit counters. All-source emission failures have
null token yield and zero artifact yield, never 100%.

`emitted_tokens_per_second` divides cumulative output by elapsed wall time since the first
persisted source event, including service downtime. `assembly_tokens_per_second` divides
cumulative actual emitted tokens by measured emitter/token-counting time. They measure different
spans; neither is model inference token throughput, network retransmission throughput or fit
speed. `unique_programs` counts distinct captured program bytes, not semantic novelty or independent fits.
The older observer's dual-language credit/estimator is separate legacy telemetry;
use this endpoint for the complete source-output ledger.

Compiler checks do not run generated programs: Rust uses `--crate-type=lib --emit=metadata`,
Python uses built-in `compile`, C++ uses `-std=c++17 -fsyntax-only`, JavaScript uses `node --check`.
These prove syntax/types where applicable, not predictive parity, linking/execution, accuracy,
safety of running code or predictive quality. Existing emitter parity CI remains required.
Missing tools, nonzero status and timeout are failures. Diagnostic files are capped at 4 MiB by
Linux `prlimit`; each compiler process also has an 8 GiB address-space cap and a CPU-seconds
cap equal to its selected timeout. Returned diagnostics retain the first 64 KiB and exit status. Default per-check
GNU `timeout` is 15 seconds (configurable from 1 to 300). Its process group bounds compiler
children. Compiler paths default to PATH and can be overridden by `JOPE_GP_RUSTC`,
`JOPE_GP_PYTHON`, `JOPE_GP_CXX`, `JOPE_GP_NODE`; use trusted local compilers.

## Operations and verification

Keep `rustc`, `python3`, `g++`, `node`, GNU `timeout` and `prlimit` available. State is exclusive to
one service via a file lock. Default storage admission is one million occurrences and at most
4096 producer journals per run. A source program is at most 4 MiB and a complete event at most
32 MiB. Size/storage/corrupt-complete-record failures stop visibly; a partial final producer line
waits for completion. Partial producer tails are listed in `ingress.partial_journal_tails`; `observed_backlog_clear`
stays false while a tail, `ingress.backlogged_journals`, or validation/error backlog exists. This is an observation of current
queues, not proof that every producer has finished. A partial/corrupt service log is refused on restart: preserve the original
before recovery rather than silently trimming evidence. Journals/sources/check diagnostics/logs
are retained; plan storage and use a new declared run directory when archiving old output.
Every uncached compiler attempt retains a separate diagnostics directory. At the default one-million
occurrence limit this can approach four million directories; monitor both free bytes and inodes.
There is no automatic retention/deletion policy. Archive stopped runs externally before choosing
a new state directory. An event exceeding 32 MiB is a fatal admission failure before accounting;
its producer record remains intact and is retried on restart, so simply restarting cannot recover
that run. Preserve the journal/log, identify the offending occurrence from the terminal error,
and explicitly archive/isolate that input before starting a new declared run. No oversized
occurrence is silently skipped or credited as successfully emitted.

```sh
cargo test --locked -p jope-observatory --test gp_source
cargo test --locked -p jope-cli --test gp_source_capture
cargo test --locked -p jope-pipeline --test gp_source_capture
cargo test --locked -p jope-dime-regressor --test generalized_gp \
  generalized_source_capture_retains_parseable_generation_zero_and_successors -- --exact
# Finite ingestion/validation of current journals, with metrics on stdout:
cargo run --locked -p jope-observatory --bin gp-source-stream -- \
  --journal-dir /absolute/journals --state-dir /absolute/state --once
```

Use `--once` with quiescent journals to obtain a finite drain; it visits every bounded turn.
`--once` reports artifact failures in metrics and exits successfully if ingestion/validation
itself completed. It fails on service/journal/storage corruption. Do not interpret process exit
zero alone as 100% compilation yield; inspect counters/status. CI checks actual source/status
payloads and token conservation, not merely that the server or compiler command ran.
