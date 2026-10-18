# Wave evidence archive and Git context v1

This contract defines a draft, additive evidence format for one training wave and each immutable version. It does not change training decisions, claim authority, promotion, serving, or retention. Implementations preserve computed, applied, and committed exposure as separate counters.

## Detailed local evidence

The learning adapter accepts a typed wave identity and ordered producer events. Every event carries sequence, attempt, rank, type, explicit fields/counters, and metrics with unit, denominator, status, source field, and a missing reason when unavailable. Unknown is represented as `missing(reason)`; it is never converted to zero. New waves require source/config/grammar/evaluator/registry/context/support/optimizer/RNG/sampler/data-role identity and per-event computed/applied/committed counters. Historical gaps are recorded as `HISTORICAL_PARTIAL` with their original missing reasons.

Events are sorted by `(sequence, rank, attempt)`, duplicate tuples are refused, and shards are capped at 512 rows and 8 MiB decoded canonical JSON. The worker writes a typed Parquet table with Zstandard pages into the existing content-addressed `$JOPE_STATE/blobs/sha256` namespace. The manifest is stored only after every shard. It records schema, wave/version/disposition, event census, missing fields, shard ordering, codec, row range, compressed and decoded lengths, compressed SHA-256, decoded member SHA-256, and optional physical object version. A new wave with missing required measurements is stored as `INCOMPLETE_BLOCKED`; it is not admitted as complete. Historical missing data remains `HISTORICAL_PARTIAL`. A local blob has no remote physical version and remains `LOCAL_COMMITTED_REMOTE_PENDING`; no local fixture qualifies S3 durability. Repeating archive creation is idempotent by content hash. Reconstruction checks manifest hash, exact blob key, file type, content hash/length, Parquet metadata/rows, decoded member hash, sequence boundaries and census.

The archive API accepts an already-produced event stream. It is not a trainer, observer, optimizer, retry scheduler, or service. Actual producer wiring and the I48 flush/terminal observer remain coordinator integration work. The adapter cannot invent absent telemetry. It never retrains on archive retry.

## Public Git projections

The records compiler accepts verified archive roots and a caller-authenticated public export policy. Required context slots are question, prospective hypothesis, controls, execution/missingness, effect/uncertainty, stage, changed/next-used model, counterevidence/limits, next falsifier, and reconstruction locator. Each metric is emitted only when its explicit metric ID is allowlisted and it carries a value, unit, numerator/denominator, status, source field, expression, evidence digest and evidence version. Numeric prose in the context is rejected so numbers cannot become unbound narrative. Sealed/private role is refused; unauthorized aggregate, identity, ancestry or evidence fields are refused.

Files are byte-stable for the same typed inputs and are returned to the caller; the compiler does not write the real wave namespace or publish Git. The manifest binds archive hashes and keeps unknown source commits null. It has no future Git commit, PR, or self-referential publication hash. The card is descriptive and cannot upgrade synthetic/shadow/adaptive evidence. The index row says `GIT_PUBLICATION_PENDING`; it does not assert a remote push.

UTF-8 caps: manifest 12 KiB, card 4 KiB, context 4 KiB, metrics 8 KiB, one index row 1 KiB, total 32 KiB including caller-declared additional wave-specific review bytes. No material limitation is truncated. If the total exceeds 32 KiB, a non-empty reviewed exception ID and reason are recorded in the manifest; individual file and index hard caps remain enforced. `docs/research/waves/INDEX.jsonl` is intentionally empty until the coordinator admits actual wave records.

## Draft limits

Remote storage/version APIs, accepted I13/I44/I46/I48 interfaces, authenticated role issuance, and a producer-to-archive observer are not available on this branch. A content hash proves bytes, not completeness, scientific validity, role authority, or remote durability. Coordinator integration must wire actual telemetry, verify remote versions, test lost-response bookkeeping, and update shared coverage/inventory records before using this format for a real wave.
