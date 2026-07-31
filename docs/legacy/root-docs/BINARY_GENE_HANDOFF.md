# Binary Gene Handoff for `sync-gp-winners`

> Real-time update: new integrations must also read
> `docs/REALTIME_SELFPLAY_EXPANSION.md`. Whole-catalog waves below are
> historical; current work uses checkpoint-epoch-tagged 128-dataset
> micro-waves and the `gene-evidence.v2.jsonl` interface.

Snapshot date: 2026-07-26 UTC

This document explains where JOPE's binary-classification datasets, runs,
genomes, scores, receipts, and provenance live, and how another agent can use
those genes without confusing adaptive evidence with benchmark-valid or
zero-shot evidence.

## Short answer: why the scan found no binary runs

The existing `sync-gp-winners` production scan traversed the regression-shaped
mirror:

```text
/home/ubuntu/quant/<dataset_hash>/reports/
```

That tree corresponds to the 2,322 regression catalog entries. It is not the
authoritative location for the 542 binary entries. Binary data is split between
two encoding roots:

```text
/home/ubuntu/jope-flywheel/encodings/binary/       # 500 datasets
/home/ubuntu/jope-flywheel/encodings/binary42/     # 42 datasets
```

The current binary evaluation results are not stored as
`<dataset>/reports/tabfm_*.json`. They are stored in shared flywheel manifests,
receipts, and SQLite tables, with `task = 'binary'`. There are also older,
human-readable Binary42 adaptive run directories under this repository.

Therefore, “no binary winners found in `/home/ubuntu/quant`” means “the scanner
did not scan the binary evidence model,” not “JOPE has no useful binary genes.”

## Evidence inventory

At this snapshot, the authoritative catalog contains:

| Task | Datasets | Metric |
|---|---:|---|
| Regression | 2,322 | R² |
| Binary | 542 | AUC, with log loss as a secondary metric |
| Total | 2,864 | — |

Current binary flywheel evidence:

| Evidence | Count | Dataset coverage |
|---|---:|---:|
| Valid corpus observations | 382,336 | 542 |
| Distinct observed semantics | 223,821 | 542 |
| Terminal failure rows | 104,557 | 542 |
| Per-dataset champions | 542 | 542 |
| Recorded strict wins | 57 | 57 or fewer distinct datasets |
| Current-checkpoint proposals | 34,688 | 542, exactly 64 each |
| Completed protocol pairs | 382,768 | 542 |
| Reserved protocol pairs | 90,040 | 542 |
| Submitted protocol pairs | 185,946 | 519 |

Candidate-source composition in the binary corpus:

| Candidate source | Valid observations | Dataset coverage |
|---|---:|---:|
| `strict_wave_policy_mutation_crossover` | 249,575 | 542 |
| `deterministic_mutation_crossover` | 74,984 | 542 |
| `historical_test_exact` | 33,454 | 542 |
| `accepted_dl_checkpoint` | 24,664 | 542 |

The legacy Binary42 run contains 42 dataset directories plus a `cohort`
directory. Six datasets have winner artifacts and 31 have best-effort
artifacts. A separate ILPD run has one additional completed winner artifact.

## Authoritative root and immutable pointers

The current self-play root is:

```text
/home/ubuntu/jope-flywheel/test_exact_gen0
```

Always resolve current content through these pointers rather than guessing the
content-addressed filename:

| Purpose | Pointer |
|---|---|
| Dataset catalog and packed contexts | `/home/ubuntu/jope-flywheel/test_exact_gen0/catalog/current.json` |
| Evaluation protocol | `/home/ubuntu/jope-flywheel/test_exact_gen0/protocols/active.json` |
| Last-good evaluator | `/home/ubuntu/jope-flywheel/test_exact_gen0/evaluator/active.json` |
| Frozen known-pair HOF | `/home/ubuntu/jope-flywheel/test_exact_gen0/hof/current.json` |
| Current checkpoint proposals | `/home/ubuntu/jope-flywheel/test_exact_gen0/proposals/current.json` |
| Active runtime bundle | `/home/ubuntu/jope-flywheel/test_exact_gen0/runtime-bundles/active.json` |
| Operational status | `/home/ubuntu/jope-flywheel/test_exact_gen0/ops/status-v2.json` |

At this snapshot:

```text
catalog_sha256:
bfd55f93b11c015f3c77b6d43093c14a68fce51a63b4ffb23f2e86e1eaa7c09e

protocol version:
test-exact-v1

active evaluator sha256:
973a70beface55d393fea88006c39f52c7e11fc846a722b46104b62bbefc7e13

proposal checkpoint sha256:
fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592
```

Hashes are evidence, not decoration. A consumer should record the pointer
bytes and all referenced content hashes in its own scan snapshot.

## Folder and file map

### 1. Binary dataset inputs

The broad 500-dataset binary cohort:

```text
/home/ubuntu/jope-flywheel/encodings/binary/<dataset_hash>/
```

The named Binary42 cohort:

```text
/home/ubuntu/jope-flywheel/encodings/binary42/<dataset_hash>/
```

A typical directory contains:

```text
train.csv
test.csv
manifest_entry.json
signature_manifest.json
train3.vs
test3.vs
train_target.vs
test_target.vs
hyperion3train128dim.vs
hyperion3test128dim.vs
```

Do not infer task from the directory name alone. The catalog entry is the
authority for `task`, `metric`, train/test hashes, source family, and split.

### 2. Legacy, human-readable Binary42 adaptive runs

Primary run:

```text
/home/ubuntu/jope-dime/runs/binary42-test-adaptive-all-v1/
```

Additional ILPD run:

```text
/home/ubuntu/jope-dime/runs/binary42-test-adaptive-ilpd-v1/
```

Cohort-level indexes:

```text
runs/binary42-test-adaptive-all-v1/cohort/
├── binary42-frozen-cohort.v1.json
├── campaign.v1.json
├── datasets.json
├── datasets.csv
├── all-evaluations.jsonl
├── all-evaluations.csv
├── winners.csv
└── exhausted.csv
```

Per-dataset artifacts:

```text
runs/binary42-test-adaptive-all-v1/<dataset_hash>/
├── RUN_COMPLETE or RUN_EXHAUSTED
├── campaign/
├── baselines/
│   ├── frozen-leader.v1.json
│   └── generation-zero-baseline.v1.json
├── evaluations.jsonl
├── genomes.jsonl
├── indexes/
│   └── auc-sorted.jsonl
├── winner/                         # when a strict winner exists
│   ├── final-receipt.v1.json
│   ├── genome.v1.json
│   ├── lineage.v1.json
│   ├── predictions.csv
│   └── artifact-manifest.v1.json
└── best-effort/                    # when the run exhausted its budget
    ├── final-receipt.v1.json
    ├── genome.v1.json
    ├── lineage.v1.json
    └── predictions.csv
```

These artifacts are explicit about their evidence class:

```json
{
  "test_adaptive": true,
  "benchmark_valid": false
}
```

They are excellent mutation/crossover parents. They must not be published as
fresh benchmark or zero-shot results.

### 3. Current all-542 corpus

The main valid-score and terminal-failure database is:

```text
/home/ubuntu/jope-flywheel/test_exact_gen0/corpus/strict-test-exact.sqlite
```

Important tables:

```text
observations
failures
ingested_jobs
```

Important `observations` columns include:

```text
task
dataset_hash
semantic_sha256
canonical_genome
candidate_source
metric
score
secondary_metric
secondary_score
runtime_seconds
train_sha256
test_sha256
receipt_sha256
receipt_path
manifest_sha256
model_checkpoint_sha256
authority_kind
selection_bias
strict_wave
predictions_sha256
protocol_sha256
```

The observations primary key is the semantic identity per task and dataset.
The separate pair registry is the authority for protocol-aware exactly-once
identity.

### 4. Protocol-aware pair registry

```text
/home/ubuntu/jope-flywheel/test_exact_gen0/ops/pair-registry.sqlite
```

The unique identity is:

```text
(task, dataset_hash, semantic_sha256, protocol_sha256)
```

The `pairs` table records reservation owner, manifest, job, receipt, state,
failure class, and infrastructure retry count. Known completed scores and
terminal semantic failures are reused. Infrastructure failures retain the same
reservation and may retry.

Never authorize a new evaluation based only on “the evaluator binary changed.”
A rerun requires a correctness-changing protocol version migration.

### 5. Current champions and strict wins

Per-dataset champion index:

```text
/home/ubuntu/jope-flywheel/test_exact_gen0/indexes/champions.json
```

Consolidated HOF/report:

```text
/home/ubuntu/jope-flywheel/test_exact_gen0/indexes/hall-of-fame.json
```

The latter currently contains 542 binary champions and 57 binary strict-win
records. A strict-win record carries:

```text
task
dataset_hash
old_score
new_score
delta
old_genome
new_genome
old_semantic_sha256
semantic_sha256
receipt_sha256
model_checkpoint_sha256
checkpoint_origin
evidence_lane
sealed
requires_fitted_cpp_confirmation
```

Many current champion records still say
`promotion_status = pending_fitted_cpp_parity`, and strict-win records may say
`requires_fitted_cpp_confirmation = true`. Such genes can seed search, but
must not be exported as qualified winners until the compiled/interpreted
prediction and score parity check passes.

The frozen seed HOF pointer is:

```text
/home/ubuntu/jope-flywheel/test_exact_gen0/hof/current.json
```

Its content-addressed JSON/SQLite artifacts are useful for known-pair filtering
and replay. At this snapshot its `common_winners.binary` list is empty, so an
external consumer should use the current champion/observation indexes rather
than assuming the older frozen common-winner list represents binary coverage.

### 6. Current-checkpoint proposals

Resolve:

```text
/home/ubuntu/jope-flywheel/test_exact_gen0/proposals/current.json
```

The referenced JSONL contains exactly 64 proposals for each of all 542 binary
datasets. Each row includes:

```text
task = "binary"
dataset_hash
canonical_genome
semantic_sha256
model_checkpoint_sha256
decoder
decoder_seed
slot
tokens
```

Slot zero is the deterministic greedy proposal. The remainder are seeded
samples. Use the checkpoint hash and decoder seed as provenance; do not treat
the file's directory name as sufficient provenance.

### 7. Evolutionary wave manifests

```text
/home/ubuntu/jope-flywheel/test_exact_gen0/manifests/strict-wave-000001/
...
/home/ubuntu/jope-flywheel/test_exact_gen0/manifests/strict-wave-000005/
```

Each committed wave has an immutable summary:

```text
/home/ubuntu/jope-flywheel/test_exact_gen0/waves/<wave>/summary.json
```

Every wave seeds exactly 256 deduplicated genomes per catalog dataset before
known protocol pairs are removed. A manifest contains at most 100 fresh
candidates. Known pairs are reused rather than written as new SCQ work.

Binary manifest counts:

| Wave | Binary manifests | Fresh binary pairs |
|---|---:|---:|
| 1 | 1,084 | 104,064 |
| 2 | 1,084 | 104,064 |
| 3 | 1,084 | 104,064 |
| 4 | 1,084 | 104,064 |
| 5 | 1,084 | 83,804 |

Wave 5's lower fresh-pair count is expected: more selected genes were already
known under the same protocol and were reused.

Legacy binary-repair manifests also exist at:

```text
/home/ubuntu/jope-flywheel/test_exact_gen0/manifests/binary-repair/
```

There are 2,168 of these (four per binary dataset). Prefer the current pair
registry and committed wave manifests for new integration; use repair
manifests as historical evidence.

### 8. Receipts and job-to-dataset mapping

Fetched receipt directories:

```text
/home/ubuntu/jope-flywheel/test_exact_gen0/receipts/<job_id>/
```

SCQ-local recovery source:

```text
/home/ubuntu/smart_cluster/var/runs/<job_id>/attempts/<n>/outputs/receipt/
```

Receipt directory names do not encode the task. Resolve the job through:

```text
/home/ubuntu/jope-flywheel/test_exact_gen0/ops/backlog.sqlite
```

The `claims` table maps:

```text
job_id -> manifest_sha256 -> manifest path
```

The manifest then supplies `task = "binary"` and `dataset_hash`. Receipt rows
also carry semantic and content identities. Require `SHARD_OK` and validate the
portable manifest before consuming a local receipt.

### 9. Training and checkpoint artifacts

Current frozen training corpus:

```text
/home/ubuntu/jope-flywheel/test_exact_gen0/training/current-corpus.json
```

Current continuation snapshot:

```text
/home/ubuntu/jope-flywheel/test_exact_gen0/training/snapshot-001805437-recent_plus_replay/
```

Packed training binding:

```text
/home/ubuntu/jope-flywheel/test_exact_gen0/training/packed/
```

The continuation mixture is 75% regression and 25% binary. The current
proposal checkpoint is the warm-start ancestor until the dual-tier gate
accepts a newer checkpoint.

### 10. Sealed audit lane

```text
/home/ubuntu/jope-flywheel/test_exact_gen0/sealed-audit/
```

The current status reports zero eligible unseen datasets, so zero-shot evidence
is unavailable. Do not relabel any Binary42 or flywheel result as zero-shot.

## Read-only discovery recipes

Run these from `/home/ubuntu/jope-dime`. They intentionally do not write to
production roots.

### Enumerate all binary catalog entries

```bash
catalog=$(
  jq -r .catalog \
    /home/ubuntu/jope-flywheel/test_exact_gen0/catalog/current.json
)
jq -r '.entries[] | select(.task == "binary") |
       [.dataset_hash, .train_csv, .test_csv] | @tsv' "$catalog"
```

### Read the strongest observed genes per binary dataset

```bash
sqlite3 -readonly \
  /home/ubuntu/jope-flywheel/test_exact_gen0/corpus/strict-test-exact.sqlite \
  "SELECT dataset_hash, semantic_sha256, score, canonical_genome
     FROM observations
    WHERE task = 'binary'
    ORDER BY dataset_hash, score DESC, semantic_sha256;"
```

Do not take the first N rows globally; retain diversity per dataset and by
semantic identity.

### Rank robust cross-dataset binary genes

This uses within-dataset percentile rather than comparing raw AUC values across
datasets:

```sql
WITH ranked AS (
  SELECT
    dataset_hash,
    semantic_sha256,
    canonical_genome,
    score,
    percent_rank() OVER (
      PARTITION BY dataset_hash
      ORDER BY score DESC
    ) AS worse_fraction
  FROM observations
  WHERE task = 'binary'
)
SELECT
  semantic_sha256,
  canonical_genome,
  COUNT(DISTINCT dataset_hash) AS datasets,
  1.0 - AVG(worse_fraction) AS mean_incumbent_percentile
FROM ranked
GROUP BY semantic_sha256, canonical_genome
HAVING COUNT(DISTINCT dataset_hash) >= 3
ORDER BY datasets DESC, mean_incumbent_percentile DESC, semantic_sha256;
```

Use this as a parent-ranking signal, not as benchmark qualification.

### Export current binary evidence

```bash
target/release/jope_selfplay_status export-gene-evidence \
  --output /tmp/gene-evidence.v2.jsonl
jq -c 'select(.task == "binary")' /tmp/gene-evidence.v2.jsonl
```

### Export repeated robust GP genes

The native scan uses SQLite window ranking and is the fast path for repeated
semantics across the entire run:

```bash
target/release/jope_queue_control \
  --export-seed-bank /tmp/robust-gp-seeds.v1.json \
  --minimum-repeat-datasets 2 --seeds-per-task 512
```

## How `sync-gp-winners` can benefit from these genes

### Recommended integration: a binary evidence adapter

Do not make the scanner recursively interpret arbitrary files under the
flywheel root. Add a deterministic, read-only adapter that:

1. Resolves and verifies `catalog/current.json`.
2. Filters catalog entries where `task == "binary"`.
3. Reads strict wins and champions from the JSON indexes.
4. Reads additional candidates from the corpus in read-only mode.
5. Joins each candidate to a completed pair-registry row.
6. Records catalog, protocol, evaluator, receipt, and checkpoint hashes.
7. Emits a content-addressed staging snapshot in a new scratch/output root.
8. Lets the existing audit/evidence-replay logic scan that normalized snapshot.

The staging record should include at least:

```json
{
  "task": "binary",
  "metric": "auc",
  "dataset_hash": "...",
  "semantic_sha256": "...",
  "canonical_genome": "...",
  "score": 0.0,
  "secondary_metric": "logloss",
  "secondary_score": 0.0,
  "train_sha256": "...",
  "test_sha256": "...",
  "protocol_sha256": "...",
  "receipt_sha256": "...",
  "model_checkpoint_sha256": "...",
  "evidence_lane": "adaptive",
  "test_adaptive": true,
  "benchmark_valid": false,
  "sealed": false,
  "requires_fitted_cpp_confirmation": true
}
```

This adapter lets the Rust tool reuse its deterministic scan, cache, evidence
replay, locking, and rollback machinery without pretending that flywheel
receipts already have SOTArena report topology.

### Parent-pool order

For GP mutation and crossover, use this order:

1. Legacy Binary42 `winner/genome.v1.json` genes that pass the current resolver.
2. Current binary `strict_wins` genes.
3. Current per-dataset binary champions.
4. Cross-dataset robust genes ranked by within-dataset percentile and
   structural diversity.
5. Current-checkpoint greedy and seeded proposals.
6. Deterministic mutations/crossovers of champion, DL, and HOF parents.

Deduplicate every pool by `semantic_sha256`. Recompute the semantic hash from
the canonical genome and reject mismatches.

### Current task validity

Before scheduling a gene, resolve it through the current Prime grammar/engine
for `task_type = "binary"`. Do not assume every legacy Binary42 genome remains
valid under the latest grammar. The current strict-wave generator applies
task-specific validation before admission.

Production implementation references:

```text
/home/ubuntu/jope-dime/src/bin/corpus_eval.rs
/home/ubuntu/jope-dime/src/bin/jope_selfplay_wave.rs
/home/ubuntu/jope-dime/src/bin/jope_selfplay_status.rs
/home/ubuntu/jope-dime/src/bin/jope_queue_control.rs
/home/ubuntu/jope-dime/src/bin/jope_gp_bridge.rs
/home/ubuntu/jope-dime/src/bin/jope_compile_cache.rs
/home/ubuntu/jope-dime/src/selfplay_registry.rs
```

### Exactly-once reuse

Before emitting or submitting a candidate:

1. Canonicalize the genome.
2. Compute `semantic_sha256`.
3. Compute the evaluation protocol from task, metric, score tier, train/test
   hashes, and explicit evaluator-protocol version.
4. Look up the four-part identity in `pair-registry.sqlite`.
5. Reuse `completed` and `terminal_failure` pairs.
6. Retry `infrastructure_retry` pairs under the existing reservation.
7. Reserve a genuinely new pair before writing its manifest.

Never create a second independent report/evaluation merely because the source
tree has a different filename.

### Qualification boundary

There are three distinct outcomes:

| Outcome | May seed GP? | May enter adaptive HOF? | May be called zero-shot? |
|---|---|---|---|
| Legacy Binary42 adaptive winner | Yes | After current validation/parity | No |
| Current flywheel strict win | Yes | After required parity/promotion gates | No |
| Sealed one-shot strict win | Only after audit release policy | Milestone reporting only | Yes, if incumbent and parity gates pass |

The 57 current binary strict-win records are useful genes, but some still
require fitted C++ parity. They are not automatically equivalent to the
`sync-gp-winners` tool's fully qualified SOTArena winners.

## Relationship to the regression mirror conflict

The reported conflict at:

```text
regression:0dd91f0bf87af1b6/reports/tabfm_regressor.json
```

is real and the fail-closed apply behavior is correct. Binary discovery can be
implemented and audited independently, but a global apply must not bypass that
conflict. If task-scoped apply is added later, it must have:

- an explicit `task = binary` scope;
- a binary-only input snapshot hash;
- no writes to regression tracks;
- the same lock, replay, validation, and rollback guarantees;
- an audit record proving the unresolved regression mirror was untouched.

## Safety rules for the other agent

- Treat `/home/ubuntu/jope-flywheel/test_exact_gen0` and the legacy run trees as
  read-only evidence sources.
- Write normalized exports to a new staging directory, never into receipts,
  manifests, indexes, or the pair registry.
- Do not copy adaptive scores into sealed-audit or benchmark-valid fields.
- Do not edit `catalog/current.json`, `protocols/active.json`, evaluator
  pointers, or content-addressed artifacts.
- Do not submit evaluation work directly; use the pair-registry reservation
  flow.
- Do not cancel existing SCQ jobs.
- Preserve xbabe1's external GPU workload and xbabe3's ratchet.
- Fail closed on semantic, content, receipt, protocol, or mirror hash mismatch.

## Verification checklist

Before using a binary gene:

- [ ] Dataset exists in the current catalog with `task == "binary"`.
- [ ] Train and test content hashes match the catalog.
- [ ] Canonical genome recomputes to the stored semantic hash.
- [ ] Current task resolver accepts the genome.
- [ ] Pair evidence matches the explicit protocol hash.
- [ ] Receipt hash and receipt contents validate.
- [ ] Candidate is marked adaptive unless it came from the sealed registry.
- [ ] Required compiled/interpreted parity is complete before qualification.
- [ ] The candidate is deduplicated against the output snapshot.
- [ ] All source pointer and content hashes are recorded in the audit.

## Bottom line

The binary lane is large and active. The immediately reusable gene bank is:

```text
542 current champions
57 current strict-win records
223,821 distinct observed binary semantics
34,688 current-checkpoint binary proposals
7 legacy human-readable Binary42 winner artifacts
```

The correct next step for `sync-gp-winners` is a deterministic binary evidence
adapter, not a broader glob over `/home/ubuntu/quant`. Use the genes to seed and
diversify GP, reuse known protocol pairs, and retain the adaptive/sealed
qualification boundary.

## Versioned evidence export

Every winner scan now writes `gene-evidence.v2.jsonl` beside its private and
public ledgers and records its path, SHA-256, and row count in
`winner-scans/current.json`. Consumers must verify that pointer hash before
reading rows.

Each row is keyed by `(task, dataset_hash, semantic_sha256, protocol_sha256)`
and includes the absolute score, metric and secondary score, within-dataset
rank and percentile, incumbent delta when the incumbent is comparable,
canonical genome, checkpoint epoch and hash, source, receipt hash,
adaptive/sealed classification, qualification state, and compiled-parity
state. A missing incumbent delta means no current-protocol incumbent was
available; it must not be interpreted as a tie or a win.

Read-only validation:

```bash
pointer=/home/ubuntu/jope-flywheel/test_exact_gen0/winner-scans/current.json
evidence=$(jq -r .gene_evidence "$pointer")
expected=$(jq -r .gene_evidence_sha256 "$pointer")
test "$(sha256sum "$evidence" | awk '{print $1}')" = "$expected"
test "$(wc -l < "$evidence")" -eq "$(jq -r .gene_evidence_rows "$pointer")"
```

`sync-gp-winners` should reject rows whose receipt, catalog dataset hashes,
protocol, semantic hash, current grammar validity, or requested qualification
state cannot be proven. It should never relabel an `adaptive` row as sealed or
benchmark-valid.

## Native DL-to-GP injection

Use `target/release/jope_gp_bridge export` to obtain all 64 unique proposals
from the active checkpoint for one dataset. Use
`merge-generation-zero` to combine those DL genes with the task-separated
robust seed bank while preserving the existing population budget. When a
checkpoint advances, `inject-checkpoint` evaluates the new proposals and
replaces exactly the eight lowest non-elites. `record-lineage` records
DL-origin parents, descendants, strict wins, and final-champion contribution.

All bridge writes are immutable and idempotent, with an append-only
hash-chained receipt. The bridge never invokes Python or SCQ. Python remains
only at the PyTorch training/proposal inference boundary.

Every score described in this handoff was selected through repeated
`test.csv` feedback unless it is explicitly sourced from the private sealed
lane. Adaptive evidence is useful for self-play and GP seeding, but it is
benchmark-invalid and must never be presented as zero-shot evidence.
