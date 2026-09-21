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

The root is `--state` when given, else ADR-0003's `jope_paths::jope_state()` — `$JOPE_STATE`, else
`$XDG_STATE_HOME/jope`, else `$HOME/.local/state/jope`, else `<jope_home>/state`. The rule is not
re-implemented here; `crates/jope-learning/tests/state_resolve.rs` asserts every step of it, and
that each step agrees with `jope-paths`, so this crate cannot drift onto a state tree the rest of
the product does not use.

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
comparing it against an expected-previous value before replacing it. That value is never an
*unchecked* read of the key about to be written: such a precondition satisfies itself, making the
write last-writer-wins, which is how a second host's batch gets orphaned.
`crates/jope-learning/tests/mirror_lost_update.rs` reproduces that race — two `$JOPE_STATE` trees,
one mirror; the second writer is refused and the first's batch stays the tip.

Nor is it the writer's own pre-write state. A precondition with no way to satisfy it is a wedge:
one `hwm ingest` without `--store`, or one transient push failure, and the local pre-append
`HEAD.json` never equals the mirror's again, so every later push and every retry fails — and there
is no push/pull/backfill verb to recover with. `sync.rs` reconciles instead. It reads the mirror's
`HEAD.json` and asks `Ledger::mirror_position` where that tip sits in the local chain:

| mirror's tip | meaning | what the push does |
|---|---|---|
| absent, or a batch this host holds, at the position and file hash its `HEAD.json` claims | **behind** — a gap | for every local batch after it: the blobs that batch names first, then the batch object (both immutable and content-addressed, so re-uploading is a no-op); then move `HEAD.json` conditionally on the tip just validated; then the `current.json` pointers those batches moved |
| anything else — a batch this host does not have, different bytes, a different depth or ledger | **diverged** | refuse, before the local append, naming both tips and what to run |

A pointer is caught up on the same rule: the mirror's `current.json` may be replaced only when the
record it holds appears *earlier* in this host's accepted history for that pointer. A record this
host's ledger never accepted is another writer's tip and is refused, not overwritten.

`HEAD.json` is the commit point, and the order around it is what makes a failed push *retryable*.
`sync::plan` derives the gap from the mirror's tip and nothing else, so anything a push skips under
a tip it has already moved is out of reach of every later push. Blobs therefore go up ahead of the
batch object that names them — free, because they are content-addressed, immutable and created with
`If-None-Match: *`, so a repeat is an `Exists` no-op — and a push that dies part-way leaves the tip
where it was, so the retry recomputes the same gap and closes it.

**What the mirror's `HEAD.json` names, the mirror holds** — and the push fails closed rather than
publish a batch for which that would stop being true. Ordering alone only covers a blob missing
from the *mirror*; a blob missing from the *local* store used to be skipped in silence, and
because the skip happened inside the backfill the batch object still went up, the tip still moved
past it, and no later push ever came back for it. So each blob a backfilled batch names is asked
of the local store first and of the mirror second, and if neither can supply it the push refuses
**before that batch's object is written**, naming the record and both places that do not have it.
The tip stays where it was; restore the blob (`hwm verify` lists it as `blob-missing`) and the next
push closes the same gap.

"The mirror already holds it" means the mirror's bytes *hash to the sha the event names*, not that
a GET returned something. This is the only path in the crate that accepts an object this host did
not itself write — `State::read_blob` re-hashes what it reads and `Mirror::put_immutable` compares
bytes on an `Exists`, and neither is on it — so identity is proved here or nowhere. A wrong object
under the immutable key (a truncated upload, a restore from the wrong bundle, a key reused by an
unrelated writer) is refused, the push fails before the batch object is written, and that object is
left exactly as it is: an immutable key is never overwritten to make a push succeed.

The `current.json` pointers deliberately stay *after* the move: the real corpus has ~12k of them and
each would cost a GET per push. A pointer write therefore fails on a mirror whose ledger has
**already committed**, and the mismatch it leaves outlives the run: the pointer stays stale — or
absent, for a cut the mirror has never held — until the next batch that *accepts* a record at it,
which catches it up under the same ancestor rule. A duplicate-only retry does not repair it,
because it seals a batch of rejects and moves no pointer, so a cut that has stopped improving can
stay stale indefinitely.

Being explicit about what that costs, because the earlier wording was not: `hwm verify` run against
that mirror **reports the mismatch** — `pointer-stale`, or `pointer-missing` for a cut it has never
held — and keeps reporting it while every later push exits 0. That is the verifier working, not a
verifier that is "unaffected": the ledger is authoritative and complete, the derived index is
behind it, and both facts are visible. It is an accepted limitation of a single-publisher,
compare-then-put mirror; moving the pointer writes ahead of the `HEAD.json` move would trade it for
something worse, because `check_pointer_ancestor` refuses a pointer another host owns and that
refusal would then wedge the ledger push itself. Blobs cannot be left in this category at all,
because nothing re-derives them: a later batch names its own records, never the ones already
committed.

Because the check runs before the local append, a diverged mirror costs nothing: no batch sealed,
no pointer moved, no object left on the mirror that its own `HEAD` cannot reach. A gap costs one
GET and the batches in it. `crates/jope-learning/tests/mirror_reconcile.rs` holds both: the
three-step wedge (mirrored ingest, unmirrored ingest, mirrored ingest) now succeeds and leaves the
mirror a complete copy; a genuinely diverged mirror still fails closed; and
`a_failed_blob_push_leaves_the_mirrors_head_where_it_was` injects an object-store failure mid-push
and asserts that while any blob a batch names is missing, the mirror's `HEAD.json` has not moved
past that batch — and that the next push closes the gap.
`a_blob_the_local_store_cannot_supply_is_never_published_over` removes a blob from the *local*
store instead and asserts the same tip invariant, with
`a_blob_only_the_mirror_still_has_does_not_wedge_the_push` as its control: a blob the mirror
already holds is not a gap and must not become a new one-way door.

### Refused at the door

Two classes of input never reach any ledger, because no later step could undo them:

* **A score that is not a finite number.** JSON has no NaN and no infinity; `serde_json` writes
  both as `null` and cannot read them back, so one such event makes every later read of an
  append-only, hash-chained ledger fail with `invalid type: null, expected f64` — the whole ledger,
  not just that event. `hwm ingest` therefore refuses the run before the lock, before the mirror
  preflight and before any blob, naming each offending candidate and field. It is *not* recorded as
  a rejection event: a rejection event is itself a ledger write, and writing it with a substituted
  score would put a number in the record that was never measured. `jope.hwm/v1` is unchanged — the
  fix is at the point of entry, not in the schema. Candidates that are merely not better are
  unaffected and keep their readable rejection event
  (`crates/jope-learning/tests/nonfinite_scores.rs`). A ledger already poisoned by an older build
  is not silently unreadable either: the reader names the batch position, the pointer, the
  identity, the record hash and the field, and `hwm verify` reports it as `event-unparseable`
  without aborting.
* **A path component that came out of an imported file.** `scores append` archives each published
  `scores.csv` under `<state>/scores/sotarena/<track_id>/<version>/`, and `track_id` is a column of
  that CSV. `Path::join` is not containment — an absolute component replaces the whole path, a `..`
  component walks out of it — so every such component is checked to be a plain segment
  (`[A-Za-z0-9._-]`, non-empty, not `.` or `..`) *before* the join, and the resulting directory is
  canonicalised and proved to be under the state root. The whole import is refused if any version
  in it is unsafe, so a good version is never archived before a later bad one is caught
  (`crates/jope-learning/tests/score_import.rs`).

Those archive writes also happen **inside the `scores` lock and after the divergence preflight**,
not before it: a run that exits 2 saying `Nothing was written` must not already have replaced last
week's archived source. And the mirror push is driven by `--store`, never by whether this run
happened to seal a batch — an import whose rows are all duplicates seals nothing, and gating the
push on that made the one command an operator would re-run to repair a lagging mirror the one
command that skipped it. `mirror` is `Some` exactly when a mirror was configured; a push with
nothing to do says so in its own counts (`"batches": 0`) instead of reporting `null`.

A server-side conditional PUT with an expected version (`PutMode::Update`) is not exposed by
`Store` yet and the local-filesystem backend of `object_store` does not implement it; until it is,
the compare-then-put is still TOCTOU between the GET and the PUT and is serialised only by the
local lock, which does not reach across hosts. What the precondition buys is that a *diverged*
mirror is refused rather than overwritten. `s3://` is configured through the environment and was
not exercised in this step.

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
