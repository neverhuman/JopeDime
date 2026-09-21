# The learning store — `jope.hwm/v1`, `jope.score/v1`, ledgers and pointers

This is the design **as built** by `crates/jope-learning` (WP11 steps L1 and L2 of the port
plan). It records what the GP high-water marks are, where every byte of them lives, how the
ledgers are chained, how the four preserved sources map into them, and which lineage-A state is
declared unrecoverable.

## Two records

**`jope.hwm/v1`** (`schemas/jope.hwm.v1.schema.json`) — one high-water mark: the best program
known for an exact cut under an evaluator identity.

```
identity        {task, dataset_hash, cut_id, evaluator_sha256, vocabulary_sha256, protocol}
identity_sha256 sha256(canonical JSON of identity)      canonical = sorted keys, compact, no whitespace
legacy_identity_sha256?  the source's own key (champions: Identity::key(), the directory name)
genome          {dsl, program_sha256 = sha256(dsl bytes), semantic_sha256?}
score           {metric, value, direction, units}
cost            {runtime_seconds?, search_seconds?}
evaluator       {sha256 | "unbound", version}
benchmark_valid sealed  replayable
provenance      {source, run_id, git_sha, bundle_id, record_sha256 = sha256 of the record blob}
prev_event_sha256   improvement_number   ts_unix
```

**`jope.score/v1`** (`schemas/jope.score.v1.schema.json`) — one evaluation, appended once, never
edited: `{dataset_hash, cut_id, method_id, evaluator_sha256, protocol, metric, value, direction,
cost?, benchmark_valid, receipt_sha256, run_id, git_sha, ts_unix, source{kind, track?, version?,
…}, row_sha256}` where `row_sha256` hashes the canonical row with itself blanked.

Both committed schema files are the schemars derivations of the Rust types
(`tests/schema_agreement.rs` fails if they drift); every object is `additionalProperties:
false`, every hash field carries its `^[0-9a-f]{64}$` pattern, and real records of every source
are validated against the committed files in the tests.

### Metric direction table

`crate::metric::Metric` / `Direction` (from `jope_task_elo.rs:31-36` and
`promotion_gate.py:22-27`; `oof_r2`, genebank's cross-validated r2, is the one addition):

| improves upwards | improves downwards |
|---|---|
| `r2`, `oof_r2`, `auc`, `macro_f1`, `negative_mase` | `rmse`, `logloss`, `mase` |

`score.direction` is written into every record so a reader without the table can rank; a record
whose direction disagrees with the table is invalid.

### Admission (`hwm::decide`)

1. An invalid record is rejected (`invalid`).
2. No incumbent → accepted.
3. An `unbound` evaluator never displaces a bound one (`unbound-cannot-displace-bound`), whatever
   the scores.
4. A different metric than the incumbent's is rejected (`metric-mismatch`).
5. Otherwise **strict** improvement under the direction: equal keeps the older record (`tie`,
   `champions.rs:452`), worse is `not-better`.

Rejected candidates are ledger events like accepted ones — nothing is dropped silently.

## Layout under `$JOPE_STATE` (mirrored 1:1 to the store URL)

```
blobs/sha256/<64hex>                            record blobs, content-addressed, written once
hwm/ledger/<yyyy>/<mm>/<ulid>.jsonl             HWM events, sealed batches (see below)
hwm/ledger/HEAD.json                            {batch, path, sha256, batches, events_total}
hwm/<task>/<dataset_hash>/<cut_id>/current.json the current best jope.hwm/v1 record for the cut
scores/ledger/<yyyy>/<mm>/<ulid>.jsonl          jope.score/v1 rows, sealed batches
scores/ledger/HEAD.json
scores/sotarena/<track>/<version>/scores.csv    the published file byte-identical + .sha256
hwm.lock  scores.lock                           flock held by every writer
```

### Sealed, hash-chained batches

A batch is one compact JSON event per line followed by a seal line
(`jope.ledger-seal/v1`): `{ledger, batch (the ULID), events, events_sha256 (sha256 of the event
lines), prev_batch, prev_batch_sha256 (sha256 of the previous batch *file*), source, tool,
ts_unix}`. The batch is written through a temporary file, fsynced and renamed; `HEAD.json` is
then replaced the same way. Batches are never rewritten. `verify` lists the batches in ULID
order and recomputes: the seal's event count and `events_sha256`, the link to the previous
batch's name and file hash, and that `HEAD.json` names the last batch with its hash and the
running counts — so a tampered byte, a truncated file, a removed batch or a stale head is a
named break. Every writer first proves the chain intact (`require_intact`) and refuses to append
to a broken ledger.

### Pointers

`current.json` moves only after its batch is durable, only when the record accepted for that
pointer differs from the one on disk, and only if the record on disk is still the one the
decision was made against (`state::write_pointer` compares `provenance.record_sha256`; anything
else is a refusal, the candidate is re-decided on the next run). Writers hold `hwm.lock` for the
whole ingest.

### Durable mirror

`--store <url>` / `JOPE_LEARNING_URL` pushes through `jope_datasets::store::Store` (thin adapter
in `mirror.rs`): blobs and batches with the create-only put (`If-None-Match: *`, an existing
object must hold the same bytes), `HEAD.json` and `current.json` by reading the object back and
comparing it to the bytes the writer decided against before replacing it. A server-side
conditional PUT with an expected version (`PutMode::Update`) is not exposed by `Store` yet and
the local-filesystem backend of `object_store` does not implement it; until it is, the
compare-then-put runs under the local lock, which is the only writer in this design.
`s3://` is configured through the environment and was not exercised in this step.

## Mapping table (as implemented in `sources/`)

| Source | task / dataset | `cut_id` = sha256(canonical{…}) | evaluator / protocol | score | chain | blob (`record_sha256`) | flags |
|---|---|---|---|---|---|---|---|
| `parallel.cut_champion.v1` (`local-winner-restore`, 11,030; `sealed-winner-restore`, 873) | `regression` / `identity.dataset` | `{family, partition, cut, prepared_sha256, training_sha256, validation_sha256}` | `identity.evaluator_sha256`, `identity.vocabulary_sha256`; protocol = `observation.evaluation_bias` (`adaptive_cut_heldout_teacher`) | `rmse` = `rmse_original_units`, lower, `original_target_units` | `prev_event_sha256` / `improvement_number` / `ts_unix` copied from the champion | the original `current.json` bytes | `legacy_identity_sha256 = Identity::key()` (must equal the file name); `sealed = partition == meta_test`; `replayable: true`; `benchmark_valid: false`; `provenance.source = champions` / `champions-sealed`, `run_id` = the observation path, `bundle_id` = the restore directory |
| genebank `best_scores` (4,948) | row `task` / `dataset_hash` | `{tier: score_tier, protocol: "genebank-gen1"}` | `unbound` / `unbound`; version `genebank-legacy`; protocol `genebank-gen1` | row `metric` (`oof_r2`, `r2`, `auc`), higher | assigned by the ledger | canonical JSON of the row | `replayable: false`, `benchmark_valid: false`; `ts_unix = achieved_at_ms / 1000`; `bundle_id` = sha256 of the db file; `semantic_sha256` kept when present |
| lineage-A snapshot (`JopeTestOnlyTrainingCorpus.v1` manifest + streamed rows) | `metadata.task_type` / `metadata.dataset_hash` | `{tier: "test_exact", protocol: strict_filter.selection_bias}` | `unbound`; version `snapshot-<watermark>` | strict-filter metric of the task (regression `r2`, binary `auc`), higher; per-(task, dataset) maximum over `valid` `test_exact` rows | assigned by the ledger | canonical JSON of the score-bearing row (embeddings dropped) | `replayable: false`; `provenance.source = snapshot-<watermark>`, `run_id = <generation>/<split>`, `bundle_id` = manifest sha256; the streamed splits must hash to `train_sha256` / `validation_sha256` |
| SOTArena `results/<version>/scores.csv` | — (`jope.score/v1`, not HWM) | `{track: track_id, protocol}` | `unbound`; `receipt_sha256 = source_sha256` | row metric/value | — | — | `benchmark_valid` from the file; `git_sha` of the results checkout; `ts_unix` from `source_revision`; `source.version` = the results directory |

### Engine types: mirrored, not linked

`champion.rs` mirrors `jope-engine`'s `champions::{Identity, Champion}` field-for-field
(`champions.rs:15-26`, `:66-77`) together with `score()` and `validate()`. Linking the engine
would pull the whole `jope-parallel` surface into the store's build for two structs, so the
canonicalisation is reproduced instead and **proved on the real files**:
`Identity::key() = sha256(serde_json::to_vec(identity))` equals every restore file name, and
`sha256(serde_json::to_vec(champion))` of every re-parsed `current.json` / `event-*.json` equals
the sha256 that names its own event file (this needs `serde_json`'s `float_roundtrip`, which the
crate enables to match the engine, and no `preserve_order`, so `Value` maps serialise sorted).
`hwm verify --events <dir>` performs this proof for every ingested champion and walks each chain
back to `improvement_number 0` checking identity, numbering, strict rmse improvement and
`previous_rmse_original_units` at every hop.

## Real ingest receipt (read-only inputs, scratch `$JOPE_STATE`)

See the end of this document — filled from the run recorded in the WP11 handoff.

## Lost lineage-A state — declared unrecoverable

Lineage A's control plane scheduled against files that no longer exist. What survives is the
four archived corpus snapshots, `encodings/`, `genebank.db` and the `D-runtime` task receipts;
the following were removed before any preservation pass and cannot be rebuilt from what remains:

| Lost | Consequence |
|---|---|
| `test_exact_gen0/corpus/strict-test-exact.sqlite` (labels) and `ops/pair-registry.sqlite` (identities) | no lineage-A label can be re-derived; the snapshot maxima above are the only per-dataset lineage-A marks and stay `unbound` |
| `training/immutable/checkpoints/*` descendants of `fbcd5459…` | no lineage-A checkpoint can be retroactively accepted; `--trainer v3-jsonl` checkpoints carry `lineage: a` and never become the serving pointer |
| `training/{learning,accepted,milestone,accepted-by-task}.json` | A's acceptance history is gone; nothing in `serving/` derives from it |
| `sealed-panel-pool.v2.json` and its consumption receipts | A's sealed gate cannot be honestly re-armed |
| `ops/task-elo.v1.json`, `catalog/current.json` | A's Elo ladders cannot be recomputed |

Archive receipts for what *was* preserved (`/home/ubuntu/preserved-archives/20260908-F-D-E/`,
`SHA256SUMS`, `catalog.json`, `README.md`; each row's `verification_receipt` under
`/home/ubuntu/storage-offload-20260908-1420/`):

| Archive | sha256 | Content |
|---|---|---|
| `snapshot-004754860-full_consolidation-eq32v32c128.tar.zst` | `e1370e816c566dce13e5ad9fdbe4fcc0a27af8dbb6f1dd8b5a77290070308058` | manifest + `corpus.{train,val}.jsonl` (753,613 regression/train rows) |
| `snapshot-008903292-…-ca24ddd8fa6aa.tar.zst` | `93721812a2a9ff8ad4339c7b1c36bb6675e02e0b26ca6cbb9c7d843a9783c374` | manifest + corpus |
| `snapshot-009658180-…-ca24ddd8fa6aa.tar.zst` | `006aed00ced36de5d0e4aa7bf4e0ca5e35cd3f546478d85a8154783a0f2fdd80` | manifest retained, corpus payloads removed |
| `snapshot-009915070-…-ca24ddd8fa6aa.tar.zst` | `5b04d3716aa2bd620f8837e2c0c70d7ab78d77525dcc7fe1e8d53593c239a7dc` | manifest + corpus (selected by `training/current-corpus.json`) |
| `D-runtime.tar.zst` | `f31d7066160e4a425a5258468e01bcb447c287b374678b97dc2283247948d974` | 2,595,995 task receipts (`scores/ledger` backfill candidates, lineage A) |
| `D-recovery.tar.zst` | `c7098e489e7866cd9b61060de18352265d2800794ea4c2a92026fecfce8d6750` | 1,038,888 recovery task files |
| `source-manifests.sqlite.zst` | `5162273d9aff0a48f55cc48f2b45145e75a7f3d42207b2e93fec796f4783299d` | the original file/directory inventory (index only) |

Lineage B is complete and is the canonical loop; its winners are the bound records above and
are the only ones `hwm freeze` exports by default (`replayable: false` records need
`--include-unbound`; `sealed` records are never exported).
