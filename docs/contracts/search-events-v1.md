# Search events v1

`jope.search-events/v1` is an additive, observational sidecar for the legacy canonical-pipeline
GP in `jope_pipeline::search`. `observations.json`, `jope.pipeline-search/v2` and proposal-ledger/v1
retain their original readers and bytes. The optional sidecar is `search-events.json` in each
cut's search directory; comparison extends that same stream. Disabling recording is an explicit
control (`SearchOptions::record_events = false`), not a claim of complete event evidence.

Each record has an external `event_id` computed over its event and original binding with I02's
`SHA256(schema_utf8 || 0x00 || u64_be(payload_length) || canonical_payload_bytes)`. The implementation
reuses the adopted codec source, pending the integrator's Cargo dependency hook. Exact float
measurements are `{ "f64_bits": "<16 lowercase hex digits>" }`; RNG seeds are 16-digit hex.
No raw floating-point JSON numbers, self hashes, signatures or later commit IDs enter preimages.
Strict parsing rejects duplicate keys, unknown fields, altered IDs and unknown versions.
A sorted capture inventory detects a lost event even when it was an unattempted duplicate or shadow draw.

A binding names the original run, task, context, source, experiment, evaluator, role/exposure,
profile, legal set and fidelity. `Known` references contain SHA-256 digests; unavailable references
carry `Unknown` with a reason. The legacy caller does not supply an authenticated experiment,
I04 exposure, evaluator or registered profile/legal-set receipt, so the default honestly records
those as unknown. It never claims a guessed profile ID/revision. The exact ordered input bits and
cut partition are hashed locally without exporting labels. Callers may supply an original binding;
its context and logger/search source must equal the actual implementation and cut inputs. Run IDs default to an output-location digest and may
be supplied explicitly for integration with an existing run manifest.

`SearchDecision` identifies every actual draw, even a duplicate, invalid operator or fallback.
Its draw ID is the enclosing Draw record’s `event_id`, avoiding a self-referential hash.
Its coordinate is unique within that run/cut stream. All candidates drawn before the legacy
admission loop are retained, including the tail refused by evaluation or seconds limits. Admission
records retain admitted, duplicate and budget dispositions. An operator failure has no invented
program or request. A migration retains its existing score and admitted/rejected disposition; it
never opens a physical evaluation. Neural incomplete/parser-rejected/shadow draws stay in the
existing proposal ledger and its versioned companion, rather than being manufactured as GP fits.

A `request_id` hashes the full binding plus the original program and canonical recipe hashes.
Repeated draws of the same subject under that binding join one logical request. A different run,
context, source, role, evaluator, fidelity or program changes that identity. The raw program hash
also participates because legacy GP deduplication uses exact canonical-pipeline text; I05 does
not broaden that rule to semantic equivalence.

An `attempt_id` hashes the binding, request and 1-based physical ordinal. Starts happen on the
actual evaluation worker after the existing deadline/cancellation check. Each retry names the
retained failed predecessor, keeps its original request, and increments the ordinal. Current GP
never automatically retries fits; the adapter supports explicit retry evidence without changing
that policy. Terminal records preserve measured, parse/fit/score failure, panic, timeout and
interruption statuses, original result references and stage costs. A missing terminal is pending;
`require_complete` refuses it. Proposal effort is separate from physical evaluation effort. A
proposal that starts no fit has no fabricated fit cost. Unknown wall/CPU/GPU/memory telemetry has
an explicit reason and never becomes zero. A separate observational timer measures the full parse/fit/predict/score wall span.
Legacy observation.seconds stops before final scoring and remains byte-compatible. CPU/GPU/memory,
proposal timing and observer publication effort remain unavailable; none is fabricated as zero.

`EditTransition` links the draw to ordered parent program/canonical hashes, the crossover donor,
sibling group, operator, stream seed, transform bound, post-canonical hash and legality/refusal.
Both parent choices survive even when they select the same parent. GP marginal child laws are
unknown in v1, with an explicit reason: seed likelihood, critic rank weight, posterior samples
and parent selection are not unconditional child density. Discrete probability and continuous
density have separate typed representations and subject checks. An attempted GP seed-density
substitution is refused. Unknown laws supply no unbiased policy-gradient/OPE claim.

Reconciliation verifies IDs, request joins, retries, edits and costs independently of delivery
or completion order. Identical delivery repeats are counted once and reported separately;
conflicting draw coordinates, starts, admissions or terminals refuse closure. Denominators are:
actual proposal draws, repeated-program draws, migration decisions separately, unique non-migration requests, physical evaluation starts,
failed physical attempts, logical completed requests, proposal-stage costs and terminal costs.
Four draws A,B,A,C with A failing then succeeding on a retry produce 4 draws, 1 duplicate,
3 requests, 4 physical attempts, 1 failure and 3 logical completions. Proposal effort is not part
of the evaluation-attempt denominator. Unknown evaluation wall costs are counted independently
of the sum of known wall costs.

`ProposalLedger::decision_links` exports `jope.proposal-decision-links/v1`, referencing every
existing neural/fallback/exploration draw, including shadow/refused draws and unchanged
reconciled denominators. Stable draw IDs bind the ledger digest and source coordinate. Optional
sorted downstream decision IDs add linkage; missing links carry a reason and confer no fitting
authority. Re-derivation against the original ledger detects omissions and alterations.

`jope_learning::progress::engine::search_events` verifies the sidecar digest and complete joins,
requires an explicit request-to-planned-work map and the same original run, verifies original
observation bytes/status/subject/known wall cost (at least the original observation span), and returns existing W8 events and retained blobs.
Each physical attempt maps to its original ordinal, preserving failed predecessors and one logical
completion through the production census. The full sidecar, costs and failed result bytes remain
in the blob set; no winner-only export or checkpoint rewrite occurs. Logical event identities are
stable across reordered/duplicate delivery. Retain blobs before publishing the returned objects.
This bridge does not issue I04 eligibility, exposure, HWM, training or promotion authority.

Recording consumes no RNG. Frozen-clock evaluation-budget fixtures compare complete legacy
observations, selection, migration and output bytes with logging on/off, including duplicate and
fallback streams and comparison remeasurement. Real logger overhead can change admissions when
wall time binds; no equality of real deadline decisions is promised. Sidecar writes are synced and
atomically replaced before success publication; an existing failed/partial sidecar refuses a new run in that destination and is preserved; I/O/recording errors fail the caller at that
boundary. Cancellation and total numerical failure preserve partial/failed event evidence even
when the legacy successful search receipt is not published. An uncatchable process/host failure
may lose the in-memory tail; pending/missing evidence cannot qualify as complete durability.

Coordinator integration still owns canonical schemas, coverage/discovery/inventory records,
shared CLI/receipt hooks and the full required gate. The separate typed kernel2 and engine GP
producers are outside I05's worker path allocation; this contract qualifies only the named legacy
pipeline route and the existing proposal-ledger export, not complete workspace instrumentation.
I04 acceptance and live resource/durability qualification remain separate integration gates.
