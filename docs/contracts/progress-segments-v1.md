# Progress segments v1 (draft)

This protocol retains immutable progress proof in hashed segments, integrity checked by a content-hashed root and an exact 256-way key index. It is additive and not wired into the current product `Outbox`, `Driver`, rebuild reader, or CLI. Existing records and acknowledgements remain untouched. Integration must route new outbox admission and historical consumers through this reader before claiming end-to-end bounded progress.

## Identity and bytes

A `SegmentRecord` carries the exact key, source key, evidence class, source digest, exact source proof bytes, and (only for `Committed`) an optional committed-update identity. The record rejects source bytes that do not match the stated digest. A segment is canonical JSON and content addressed by its canonical body. It records its generation and previous root digest. Records in one segment must have unique keys; retries with the same key are idempotent only when every field and proof byte agrees. A same-key conflict fails visibly.

The immutable root binds generation, the prior root, every retained segment reference, 256 shard references, total record count, and retained proof bytes. Each shard is sorted by exact key and carries the segment digest, row ordinal and record digest. The first two hex characters of SHA256(key) choose the shard; the complete key is compared after lookup, so the hash prefix is only an index and never membership authority. Lookup reads at most one shard and one segment, each under an explicit byte limit.

## Durability and recovery

The writer lock enforces one writer. Append validates all records and prospective capacity before writing. It fsyncs immutable segment bytes, changed shard pages and candidate root before atomically replacing `CURRENT.json`; this pointer replacement is the root commit point. A crash before pointer publication leaves the previous root current and any written immutable objects as accounted uncommitted leftovers. A crash after pointer publication reopens the complete new root. It never deletes an old segment, root, failure, unknown or divergent proof.

The current root and bounded exact-index pages are read at startup. The reader verifies canonical encodings, root and page digests, ordering, bucket placement, counts and caps. Segment bytes are verified on lookup. A missing CURRENT pointer with retained objects refuses implicit cold start. A malformed or missing referenced object is an error; no summary cache, maximum step, ack count or uncommitted source is promoted to committed progress.

## Limits and saturation

`SegmentLimits` requires explicit positive bounds for records per segment, segment bytes/count, per-shard bytes, total entries, startup reads, hot index bytes, root bytes and total local disk bytes. The 256-shard root is fixed. Append checks entry/segment/shard/root and reserved pointer-write headroom before publication. Old proof remains present when a cap is hit; new admission stops with an error. This draft does not provide a garbage collector or authorize remote/archive deletion. A finite local cap cannot promise infinite retention; an owner archive/retention protocol or stopped admission is required.

The `Usage` readback reports generation, retained entries/segments/proof bytes, startup bytes, hot index bytes, last lookup page/bytes and total observed disk bytes. These are bounded fixture measurements; callers still need a resource envelope and production measurements. The current old outbox does not yet use these limits, so they are not evidence that production startup/hot memory is bounded.

## Checkpoints and derived progress

A `SegmentCheckpoint` binds source commit/root and references for checkpoint, intent and occurrence evidence. `rebuild` deduplicates by exact update/evidence identity, rejects conflicting identity reuse, and increments committed updates only when an injected owner verifier confirms the actual source receipt set. Unsupported committed receipts are counted as held. Proposed, computed, applied, uncommitted, failed, unknown, inconclusive, divergent, quarantined, local-only and remote-pending states remain distinct. No checkpoint object certifies optimizer mutation or scientific improvement by itself.

The `CommittedUpdateVerifier` is a draft dependency boundary for the accepted I05/I16/I26 and future I12/I13 source semantics. The local fixtures use a test verifier; they do not prove authentic signed authority or production producer/consumer wiring.

## Rollback

Disable new segment admission, retain all segments and roots, and resume from a previously verified root only after the owning protocol validates its source closure. Do not remove records or reclaim disk by deleting failed, unknown, divergent, or unacknowledged proof. The coordinator must add the production module/caller hooks and coverage/inventory entries, then test the real outbox, rebuild, cadence, reconciliation and next-consumer routes.
