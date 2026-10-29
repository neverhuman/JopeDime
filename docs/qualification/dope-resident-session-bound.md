# No-landmark resident session bound

This Rust-only qualification exercises the JopeDime direct Dope512 product through
its actual session startup algorithm. It complements the released resident512
forward benchmark. The new hardware test is opt-in; adding the test establishes
no measured latency result. Retention, none/full speedup, cold process startup,
and model loading remain separate qualification items.

## Input and admission

An operator supplies an already admitted, local **real S3 FIT** numeric CSV and
its JSON manifest through `JOPE_SESSION_BOUND_CSV` and
`JOPE_SESSION_BOUND_MANIFEST`. The manifest has exactly these fields:

| Field | Required value |
| --- | --- |
| `s3_uri` | Nonempty `s3://bucket/key` identifying the admitted source |
| `role` | `fit` |
| `csv_sha256` | Lowercase SHA256 of the exact CSV bytes |
| `target` | Explicit target column name |
| `columns` | Complete CSV header in exact order, including the target |
| `row_ids` | Unique physical source IDs in CSV row order, 1–1024 rows |
| `task` | `regression`, `classification` (two classes), or `multiclass` (at least three) |

All feature and target cells must be finite numbers. The target may occur at any
header position. Missing/extra rows, duplicate IDs, schema drift, a hash mismatch,
unsupported task or invalid numeric cell refuse the measurement. The harness
neither downloads nor discovers data, and does not open the CSV until the
manifest declares FIT. Do not supply sealed data or evaluation/test partitions.
The manifest is an **operator declaration**, not independent proof of S3 origin
or FIT authorization. A qualifying report must retain the separate admission and
S3 provenance evidence. Normal unit-test fixture bytes only test refusal/parser
logic; they establish no native coverage, latency or retention result.

Select `JOPE_SESSION_BOUND_DEVICE=cpu` or `cuda:N` explicitly. Unsupported or
unavailable CUDA errors, with no CPU fallback. CUDA qualification requires an
external nonblocking `flock -n ~/jope_plan/qualification.lock` reservation and the
host's current resource allocation. Missing environment inputs error when the
ignored test is explicitly invoked.

## Timing and acceptance

The released D0 checkpoint is loaded once and checked against SHA256
`d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c`.
The test takes six samples, discards the first and uses five measured samples.
CSV read, byte validation and numeric parsing are timed once; that same read
cost contributes to every complete sample. Input handoff clones and model load
are outside the spans. Each sample performs one real session preparation.

A private observer marks completion of actual preparation. Validation and
preparation precede that boundary; scope hashing, direct-consumer setup,
zero-landmark conditioning, pretrained forward, session creation and the first
host readback follow it. The final readback synchronizes CUDA completion.
Session drain/join and post-span assertions are excluded. No native landmark
panel runs in `none` mode. The public session entry uses the same internal
algorithm with an empty observer; it gains no clock or public API.

Preparation uses eight Rayon threads. The existing CPU encoder forward pool
continues to use one thread; CUDA uses the explicitly requested device. Both
thread counts are reported. This preserves production scheduling.

For the supplied numeric table with 1–1024 rows, all six samples must be finite
and nonnegative, with positive complete latency. After discarding the first:

- maximum post-preparation overhead must be **≤5 ms**;
- median complete latency (read/parse + validation/preparation + overhead) must
  be **≤30 ms**.

The report also includes interpolated p95 over the five measured totals, raw
samples, hashes, table shape, task, requested/actual device, recipe and checkpoint
binding. It emits no raw source URI, column names, cells or physical ID lists.
The run checks finite 512 coordinates, version zero, no observed landmarks,
the direct product, a single event and successful session completion. Repeated
runs must retain source and effective-conditioning digests.

Passing qualifies only the supplied fixture, device, build and hardware. It
establishes neither a general table-size guarantee nor the mission's ≥5×
none/full cold-start or quality/retention gates.

## Reproduction

Build on allocated CPU capacity, using the lane's existing target and required
environment (`CARGO_INCREMENTAL=0`,
`CARGO_PROFILE_DEV_DEBUG=line-tables-only`). Keep the host below its resource cap.
The deterministic control tests need no external data or accelerator:

```sh
cargo test --locked -p starforge --features dope --lib \
  dope_encoder::workflow::resident_release_tests -- --skip real_s3_fit_none_session_release_bound
```

Build the optimized library test once, selecting `dope-cuda` when qualifying
CUDA, and retain the compiler artifact record:

```sh
cargo test --locked --profile dist-ci -p starforge --features dope-cuda \
  --lib --no-run --message-format=json > target/session-bound-build.jsonl
```

Resolve the emitted `starforge` library test executable from that record; do not
assume a hashed filename. Set the three environment inputs, then execute that
binary with the following exact arguments (under the external reservation for
CUDA):

```text
--exact dope_encoder::workflow::resident_release_tests::real_s3_fit_none_session_release_bound --ignored --nocapture --test-threads=1
```

Retain the complete result and command environment alongside the source commit,
executable/checkpoint hashes, hardware identity, thread settings and separate
real-S3/FIT admission evidence. A nonzero result is a refusal or failed bound,
not a PASS. Run the ordinary encoder/session compatibility suites and repository
required gates on the final consolidation head. Hardware latency remains pending
until the explicit test actually passes under those conditions.
