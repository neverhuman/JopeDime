# Runbook — ingest the winners into the learning store

Rebuilds the high-water-mark (HWM) ledger and the score ledger from the preserved inputs, in the
order the lost-state contingency of the port plan prescribes. Every step is read-only on its
inputs and append-only on `$JOPE_STATE`; re-running a step is idempotent (every candidate is a
tie against itself and is logged as a rejected event, never dropped).

```sh
cargo build --release -p jope-learning --locked
JL=target/release/jope-learning
export JOPE_STATE=/path/to/state            # ADR-0003 default: $XDG_STATE_HOME/jope, else ~/.local/state/jope
# export JOPE_LEARNING_URL=s3://veox-jopedime-learning-use2   # optional durable mirror
```

`--store <url>` / `JOPE_LEARNING_URL` mirrors every batch, blob and pointer through
`object_store` (`file:///dir` for drills, `s3://` in production, credentials from the
environment). Without it the local tree is the only copy.

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

## 6. Inspect

```sh
$JL hwm show <dataset_hash> [--task regression] [--cut <cut_id>] [--raw]   # --raw = the original bytes
$JL hwm find <identity_sha256 | Identity::key()>
$JL hwm status ; $JL scores status
$JL scores query --best-per-dataset [--version tabular-regression-v2]
$JL hwm freeze --out <dir> [--include-unbound]     # snapshot restricted to HWM winners
```

## Failure modes

| Symptom | Meaning | Action |
|---|---|---|
| exit 2 `chain broken` on ingest | a ledger batch or HEAD was altered | nothing is appended; restore the batch from the mirror or the bundle, re-run `verify` |
| `pointer … changed underneath us` | another writer moved the pointer between decision and write | re-run the ingest (the candidate is re-decided against the new incumbent) |
| `rows hash to … but the manifest binds …` | the streamed split is not the archived one | check the archive against `SHA256SUMS`; do not pass `--rows` from any other copy |
| `scan.rejects` non-empty for champions | a file is not a valid `parallel.cut_champion.v1` or its name is not its identity key | inspect; never edit the file — rejected files are a preservation finding |
