# Runbook — ingest the winners into the learning store

Rebuilds the high-water-mark (HWM) ledger and the score ledger from the preserved inputs, in the
order the lost-state contingency of the port plan prescribes. Every step is read-only on its
inputs and append-only on `$JOPE_STATE`; re-running a step is idempotent (every candidate is a
tie against itself and is logged as a rejected event, never dropped).

```sh
cargo build --release -p jope-learning --locked
JL=target/release/jope-learning
export JOPE_STATE=/path/to/state            # ADR-0003 default: $XDG_STATE_HOME/jope, else ~/.local/state/jope, else <jope_home>/state
# export JOPE_LEARNING_URL=s3://veox-jopedime-learning-use2   # optional durable mirror
```

`--store <url>` / `JOPE_LEARNING_URL` mirrors every batch, blob and pointer through
`object_store` (`file:///dir` for drills, `s3://` in production, credentials from the
environment). Without it the local tree is the only copy. Mixing the two is safe: a mirror left
behind by an unmirrored run (or a failed push) is caught up by the next `--store` run.

The order inside that push is not uniform, and the difference is what you are reading for when
something fails. **Content goes up before `HEAD.json`**: for every batch in the gap, the record
blobs that batch names, then the batch object, and only then the pointer move — so a failure
anywhere in that part leaves the mirror's tip where it was and costs nothing but a retry. **The
`current.json` pointers go up *after* `HEAD.json`**, not before it, and only for the pointers the
backfilled batches moved. A pointer write that fails therefore fails on a mirror whose ledger has
already committed, and it leaves a mismatch that this run cannot undo: `hwm verify` against that
mirror reports `pointer-stale` (or `pointer-missing`, for a cut it has never held) until some
later batch *accepts* a record at that pointer. A duplicate-only retry does not repair it — it
seals a batch of rejects, which moves no pointer — so the verifier keeps reporting the mismatch
while every push reports success.

That remains a limitation of publishing the derived index after the ledger, not a bug to work
around here: the ledger stays authoritative and the stale entry is a derived index. Do not try to
repair it by hand-editing the mirror's `current.json`.

## 1. Champions (lineage B, `parallel.cut_champion.v1`)

Inputs: the two restore directories (one `<Identity::key()>.json` per identity, the engine's
`current.json` bytes verbatim) and, for chain walking, the engine event directories
(`<key>/event-<sha256>.json`).

```sh
$JL hwm ingest --source champions <restore>/local-winner-restore --bundle-id local-winner-restore
$JL hwm ingest --source champions <restore>/sealed-winner-restore --bundle-id sealed-winner-restore
```

Expect `candidates 11030 / accepted 11030` then `873 / 873` with `scan.sealed 873`. The
identity key of every file must equal `Identity::key()` of its content, the observation must
re-validate (`program_sha256`, `observation_sha256`, strict teacher score) — anything else is
listed under `scan.rejects` and is not ingested. A layout of `<key>/current.json` directories
(the live `runs/champions` tree) is accepted too.

## 2. Genebank (`best_scores`, unbound)

```sh
$JL hwm ingest --source genebank /path/to/genebank.db
```

Opens the database read-only; `bundle_id` defaults to the file's sha256. Expect 4,948
candidates accepted (regression 4,406 / binary 542; tiers cv_robust 2,090 / test_exact 2,858),
all `evaluator: unbound`, `replayable: false`, `benchmark_valid: false`.

## 3. Lineage-A corpus snapshot (unbound, per-dataset `test_exact` maxima)

The archive holds `corpus-manifest.json` (5 KB) and two JSONL splits (2 GB + 38 GB). Extract
only the manifest; stream the rows through a pipe so nothing is written to disk:

```sh
A=/path/to/snapshot-004754860-full_consolidation-eq32v32c128
tar --zstd -xf $A.tar.zst --strip-components=1 -C /tmp/snap $(basename $A)/corpus-manifest.json
$JL hwm ingest --source snapshot /tmp/snap/corpus-manifest.json \
  --rows validation=<(tar --zstd -xOf $A.tar.zst $(basename $A)/corpus.val.jsonl) \
  --rows train=<(tar --zstd -xOf $A.tar.zst $(basename $A)/corpus.train.jsonl)
```

Each split is hashed as it streams and must equal the manifest's `train_sha256` /
`validation_sha256`, otherwise the ingest refuses (exit 2); per-task/split row counts are
compared with `counts` (`scan.counts_match_manifest`). The pass takes a few minutes for the
40 GB stream. Only `valid` rows of tier `test_exact` whose metric is the strict-filter metric
of their task (regression r2, binary auc) compete; one record per (task, dataset) is produced.

## 4. SOTArena results (score ledger, not HWM)

```sh
$JL scores append --source sotarena /path/to/sotarena/results
```

One sealed batch per `<version>/scores.csv` (today `tabular-regression-v1`, `-v2`);
`git_sha` is `git rev-parse HEAD` of the results checkout unless `--git-sha` is given. The
published file is copied byte-identical to `$JOPE_STATE/scores/sotarena/<track>/<version>/`
next to its sha256.

## 5. Verify

```sh
$JL hwm verify --events <mirror>/champions/runs/champions --events <sealed>/runs/champions
$JL scores verify-ledger
```

Both exit 0 only with zero breaks. `hwm verify` recomputes every batch seal and the batch
chain, replays every accepted event against the per-pointer chain (strict improvement, unbound
never displacing bound, `prev_event_sha256`/`improvement_number` continuity), re-hashes every
blob, compares every `current.json` on disk with the ledger, and — for champion records — walks
the engine chain from `current.json` back to `improvement_number 0` through the event files,
proving each `event-<sha>.json` re-serialises to its own name.

## 6. Inspect (read-only)

```sh
$JL hwm show <dataset_hash> [--task regression] [--cut <cut_id>] [--raw]   # --raw = the original bytes
$JL hwm find <identity_sha256 | Identity::key()>
$JL hwm status ; $JL scores status
$JL scores query --best-per-dataset [--version tabular-regression-v2]
$JL hwm blob <sha256>                              # one record's bytes, no provenance claim
```

`hwm status` is a bare `HEAD.json` read and makes no claim about the chain; `show`, `find` and
`blob` re-hash whatever they hand you. Use `blob` and `show --raw` to salvage individual records
from a store that does not verify — they read one record at a time and assert nothing about where
it came from.

## 7. Export a corpus

```sh
$JL hwm verify --events …                          # section 5 — must be exit 0 first
$JL hwm freeze --out <dir> [--include-unbound]     # snapshot restricted to HWM winners
```

**This is the export boundary, not an inspection.** `freeze` writes a corpus that later readers
consume *without* re-deriving it, and the `ledger_head` in its `manifest.json` is a provenance
claim about the chain it came from. It therefore verifies the batch chain first
(`require_intact`, the same check `ingest` makes) and exits 2 without writing anything if the
chain does not walk — there is no `--force`.

That check is the chain, not a clean bill of health for the tree. Run section 5's `hwm verify`
before publishing a corpus: it additionally re-hashes every blob, replays the per-pointer chains
and compares every `current.json` against the ledger, so a `pointer-orphan` or a `pointer-stale`
on an otherwise-intact chain shows up there and not in `freeze`. And a removed tail with
`HEAD.json` rewritten to match is invisible to both, for the reason given under *Failure modes*
— only the mirror catches that.

## Failure modes

| Symptom | Meaning | Action |
|---|---|---|
| exit 2 `chain broken` on ingest | a ledger batch or HEAD was altered | nothing is appended; restore the batch from the mirror or the bundle, re-run `verify` |
| `pointer … changed underneath us` | another writer moved the pointer between decision and write | re-run the ingest (the candidate is re-decided against the new incumbent) |
| exit 2 `holds a chain this host cannot continue` | the mirror's tip is a batch this host's ledger does not have (another host owns that chain) | nothing was written, locally or on the mirror; re-run without `--store` to record locally, or point `--store` at a mirror this host's chain covers |
| exit 2 `this host's ledger never accepted at that pointer` | the mirror's `current.json` holds a record from a chain this host has not ingested | ingest that source here first, or push to a mirror this host's chain covers; the batches and `HEAD.json` of that run did land |
| exit 2 `put blobs/sha256/…` or `put <ledger>/…jsonl` during a `--store` run — a **pre-`HEAD`** content failure | the push failed while uploading a blob or a batch object, which both happen before the pointer move | the local append stands; the mirror's `HEAD.json` did **not** move, so nothing reachable from it is unbacked and `hwm verify` against the mirror is clean. Re-run the same `--store` ingest: the next push recomputes the same gap and closes it |
| exit 2 `put hwm/<task>/…/current.json` or `mirror pointer changed` — a **post-`HEAD`** pointer failure | the push failed *after* `HEAD.json` committed; pointers go up last | the mirror's ledger is committed and correct, but its `current.json` index is now behind it. `hwm verify` against that mirror reports `pointer-stale` (or `pointer-missing`) and will keep reporting it: a plain retry re-runs the same duplicate rows, seals a batch of rejects and moves no pointer. Re-running is safe but does **not** repair the index; only a batch that *accepts* a record at that cut does. Do not hand-edit the mirror's `current.json` |
| the mirror's `current.json` is older than the mirror's own ledger | the same post-`HEAD` pointer failure, seen later | accepted limitation, not a wedge, and **visible**: `hwm verify` against the mirror reports `pointer-stale` for it. The ledger stays authoritative. To clear it, ingest a genuinely better record for that cut with `--store`; a cut that has stopped improving can stay stale indefinitely |
| exit 2 `names record … does not hold it — and neither does the mirror` | a batch the push was backfilling names a record blob that no longer exists locally and was never mirrored | the mirror's tip did **not** move and nothing was published. `hwm verify` lists the same record as `blob-missing`; restore it from a bundle or re-ingest its source, then re-run the `--store` push — it recomputes the same gap and closes it |
| exit 2 `the … bytes the mirror … holds at blobs/sha256/… hash to … not …, so that object is not this record` | the local blob is gone and the object under its immutable key on the mirror is **not** that record | nothing was published and **the mirror's object was not touched** — an immutable key is never overwritten to make a push succeed. Find out what wrote it before doing anything else; then restore the correct blob locally. Note that a correct local blob will *also* refuse (`exists … with different bytes`) until the wrong object is dealt with by whoever owns it |
| exit 2 `… is a symlink` / `is not a directory` on `scores append` | a component of `<state>/scores/sotarena/<track>/<version>` is not a real directory inside the state root | nothing was written, and nothing was created on the far side of it; the archive path must be real directories under `--state`. Remove the symlink or point `--state` somewhere clean |
| exit 2 `candidate(s) whose score is not a finite number` | a source row carries a NaN or an infinity, which JSON cannot represent and this ledger could never read back | nothing was written — no blob, no batch, no pointer, nothing on the mirror. The error names each candidate and field; fix or drop those rows at the source and re-run |
| exit 2 `unsafe track_id …` / `unsafe version …` on `scores append` | the imported `scores.csv` names an archive directory that is not a plain path component (absolute, `..`, nested, …) | nothing was written, for any version of that import; correct the `track_id` column in the published results and re-run |
| `"rows": 0` with `"mirror": {"batches": 0, …}` | every row was a duplicate **and** the mirror was already at this host's tip | a genuine no-op; `"mirror": null` (and only that) means no mirror was configured at all |
| exit 2 `refusing to freeze into … chain broken: N break(s)` | `hwm freeze` verifies the chain before exporting, and this one does not walk | nothing was written — no blobs copied, no `manifest.json`. Run `hwm verify` for the full report and repair the ledger; a corpus must not carry a `ledger_head` nobody checked. To pull individual records out of a damaged store, use `hwm blob` / `hwm show --raw`, which make no provenance claim |
| `rows hash to … but the manifest binds …` | the streamed split is not the archived one | check the archive against `SHA256SUMS`; do not pass `--rows` from any other copy |
| `scan.rejects` non-empty for champions | a file is not a valid `parallel.cut_champion.v1` or its name is not its identity key | inspect; never edit the file — rejected files are a preservation finding |

## Concurrent pointer updates

Pointer publication uses the version returned with the ancestry-checked read. S3 updates
use conditional PUT; missing pointers use create-only PUT. A conflict fails the push rather
than overwriting the winner. Reconcile the local chain against the winning remote tip before
retrying; a failed push may already have appended locally and uploaded immutable objects.
Local filesystem mirrors serialize cooperating pointer writers on the mirror root directory
and revalidate the version under that lock. Direct filesystem writes are outside this guard.
The ledger head and derived indexes are still separate writes; atomic pointer updates do not
remove the stale-index limitation described above. Live S3 qualification is separate from
the offline tests.
