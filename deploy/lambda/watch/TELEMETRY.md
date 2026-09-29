# Snapshot contract v1

The Rust viewer accepts a bounded local JSON file. It is a derived view, **never** the training authority or budget ledger. Publish it on the coordinator host and access the TUI through SSH. At most one aggregate snapshot per second is normally sufficient; the UI redraws at most four times per second and Ratatui emits only changed terminal cells.

Write a temporary file in the same directory, flush/fsync it, then rename atomically to `JOPE_WATCH_SNAPSHOT`. See `feed.py::atomic`. Never expose partially written JSON intentionally. Readers retain their last good frame on errors. Maximum 8 MiB, 20,000 views, 120 rate buckets, 512 history values/view, 64 KiB program/view. The binary rejects invalid/duplicate identities; choose bounded display histories and keep complete histories/models in S3.

Required top-level identity: `schema="jope.lambda-watch.snapshot.v1"`, unique `feed_id` for the producer instance, monotonic `sequence` within that instance, `campaign`, `mode` (`live`, `closed`, `replay`), positive integer `generated_utc_us`. `cursor_utc_us` is the time anchoring the rate arrays; `last_event_utc_us` is the latest observed event. Never relabel recorded events as live.

Other fields:

| Field | Contents |
|---|---|
| `status`, `scope`, `notice` | Short, truthful operator labels and explicit protocol/fidelity |
| `activity_label` | Exact meaning of `in_flight`, e.g. coordinator requests |
| `counts` | `dispatched`, `completed`, `failed`, `unknown`, `in_flight`, `jobs`, `job_failures`, `candidates`, `unknown_bills`, `archived` as unsigned integers |
| `budget` | `cap_usd`, nullable `reserved_usd`, observed `lambda_usd`, `s3_usd`, textual `scope`; charge each attempt exactly once |
| `rates` | `calls`, `jobs`, `candidates`, optional `active`: oldest-to-newest integer counts, normally 60 one-second UTC buckets ending at `cursor_utc_us // 1_000_000` |
| `tiers[]` | `mb`, `calls`, `cold`, `peak_mb`, `billed_ms`, `lambda_usd`; AWS memory semantics must be labeled |
| `p50_ms`, `p95_ms` | Nullable observed duration quantiles; declare the duration and sampling basis |
| `errors[]` | Up to 100 concise redacted diagnostics, no tokens/presigned URLs |

Each `views[]` row has a unique, stable `id` that binds dataset, repeat, outer fold and evaluation contract; human `name`, `task`, `fold`, `status`; nullable `generation` and `target_generation`; `candidates`, `rows`, `features`, `memory_mb`; nullable `duration_ms`; and integer `updated_utc_us`.

For quality, include `metric`, `direction` (`min` or `max`), nullable `score` and `baseline`, and `history` for **that same view/objective/fidelity**. A committed population controller should report committed generations; the legacy adapter explicitly reports candidate generation indices. Supply the corresponding complete executable `program`, `program_sha`, and `context` explaining selection versus audit role. Do not substitute or compare metrics with different directions/units; null means unavailable, never zero. No NaN/Infinity.

On unchanged data, atomically replace `<snapshot>.heartbeat` with:

```json
{"feed_id":"same producer instance","sequence":123,"generated_utc_us":1790618717413207}
```

Heartbeats are at most 2 KiB and must bind the currently published snapshot's feed ID and sequence. A different producer/sequence cannot revive an old snapshot. The viewer advances live rate windows with zeros during idle periods; replay time remains fixed. After eight seconds without a valid heartbeat/snapshot, it shows STALE FEED. Future clock skew above five seconds is flagged.

Do not publish heartbeat success while the authoritative source is inaccessible. Emit a source error; leave counters and budget uncertainties intact. A closed immutable campaign may keep a heartbeat alive to show that its viewer adapter is working, but must remain `mode="closed"`.

Raw receipts, exact UTC microseconds, monotonic duration strings, model artifacts, predictions, lineage and all KPI measurements continue to live in the training archive. The snapshot is a fast disposable index over that evidence. Microsecond precision does not assert synchronized clocks across Lambda environments.
