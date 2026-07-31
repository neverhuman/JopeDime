# Real-Time Self-Play Expansion

This is the operator and consumer map for JOPE's internal, test-adaptive
self-play lane. The lane fits genomes on `train.csv`, selects them with
`test.csv`, and feeds those labels back into training and evolution. Every
adaptive artifact is therefore `test_adaptive: true`,
`benchmark_valid: false`, and `selection_bias: test-exact-reused`. Only the
separate sealed one-shot cohort can support zero-shot milestone evidence.

## Authority map

| Authority | Location |
|---|---|
| Regression candidates | `/home/ubuntu/remote_super/quant/regression/<dataset_id>/` |
| Binary candidates | `/home/ubuntu/remote_super/quant/binary/<dataset_id>/` |
| Regression mirror | `/home/ubuntu/quant/<dataset_id>/` or `/home/ubuntu/quant/regression/<dataset_id>/` |
| Binary mirror | `/home/ubuntu/quant/binary/<dataset_id>/` or `/home/ubuntu/quant/<dataset_id>/` |
| Expanded regression cache | `/home/ubuntu/jope-flywheel/encodings/regression-expanded/<dataset_id>/` |
| Expanded binary cache | `/home/ubuntu/jope-flywheel/encodings/binary-expanded/<dataset_id>/` |
| Active catalog | `/home/ubuntu/jope-flywheel/test_exact_gen0/catalog/current.json` |
| Expanded append-only catalogs | `/home/ubuntu/jope-flywheel/test_exact_gen0/catalog/v2/` |
| Pair/attempt registry | `/home/ubuntu/jope-flywheel/test_exact_gen0/ops/pair-registry.sqlite` |
| Corpus | `/home/ubuntu/jope-flywheel/test_exact_gen0/corpus/strict-test-exact.sqlite` |
| Current proposals | `/home/ubuntu/jope-flywheel/test_exact_gen0/proposals/current.json` |
| Checkpoint/cohort proposals | `/home/ubuntu/jope-flywheel/test_exact_gen0/proposals/by-checkpoint/` |
| Pending micro-cohort | `/home/ubuntu/jope-flywheel/test_exact_gen0/waves/pending/` |
| Committed waves | `/home/ubuntu/jope-flywheel/test_exact_gen0/waves/strict-wave-NNNNNN/` |
| Evaluation manifests | `/home/ubuntu/jope-flywheel/test_exact_gen0/manifests/strict-wave-NNNNNN/` |
| Checkpoints and pointers | `/home/ubuntu/jope-flywheel/test_exact_gen0/training/` |
| Receipts | `/home/ubuntu/jope-flywheel/test_exact_gen0/receipts/` and `/home/ubuntu/smart_cluster/var/runs/<job>/attempts/<n>/outputs/receipt/` |
| Freshness database/socket | `/home/ubuntu/jope-flywheel/test_exact_gen0/ops/freshness.sqlite`, `freshness.sock` |
| Operational status v3 | `/home/ubuntu/jope-flywheel/test_exact_gen0/ops/status-v3.json` |
| Compile cache | `/home/ubuntu/jope-flywheel/test_exact_gen0/compiled-cache/` and `target/release/jope_compile_cache` |
| Pinned SCQ node policy | `/home/ubuntu/jope-dime/manifests/selfplay/scq-nodes.v3.toml` |
| Pinned SCQ project policy | `/home/ubuntu/jope-dime/manifests/selfplay/scq-projects.v3.toml` |
| Deployed SCQ config | `/home/ubuntu/smart_cluster/etc/nodes.toml`, `projects.toml` |
| Freshness MCP runtime | `/home/ubuntu/jope-dime/manifests/selfplay/freshness-mcp-runtime.v2.json` and `target/release/jope_selfplay_mcp` |
| Rust GP seed bank | `/home/ubuntu/jope-flywheel/test_exact_gen0/indexes/robust-gp-seeds.v1.json` |
| GP proposal/injection bridge | `/home/ubuntu/jope-dime/target/release/jope_gp_bridge` |
| Rust production bundle | `/home/ubuntu/jope-dime/target/release/jope_selfplay_bundle` |

Historical evidence remains read-only:

- per-dataset `reports/` beneath `/home/ubuntu/quant` and
  `/home/ubuntu/remote_super/quant`;
- binary JAIR and NM material beneath
  `/home/ubuntu/remote_super/quant/binary/{JAIR,NM_RUN}`;
- Binary42 runs under
  `/home/ubuntu/jope-dime/runs/binary42-test-adaptive-{all,ilpd}-v1`;
- current flywheel indexes under
  `/home/ubuntu/jope-flywheel/test_exact_gen0/indexes`.

Old reports are candidate evidence, not automatically comparable incumbents.
Replay them under the current evaluator protocol before claiming a strict
prior-GP or TabFM win.

## Queue freshness

The queue keeper runs every five seconds. Given the 60-minute p95 completion
rate and effective cluster CPU-job slots:

```text
queued_target = clamp(256, 2048,
                      max(2 * effective_cluster_slots,
                          ceil(p95_jobs_per_minute * 15)))

queued_hard_cap = clamp(512, 4096,
                        max(queued_target,
                            ceil(p95_jobs_per_minute * 30)))
```

Only `queued` and `starting` count as useful depth. `running`, `dispatching`,
and `held` are separate counters. Launch-ready checked manifests outside SCQ
are limited to one target horizon.

Every new manifest binds an immutable checkpoint epoch and checkpoint hash,
catalog hash, evaluator protocol hash, generation timestamp, train/test
contents, seed, proposals, and hall-of-fame.

When the learning pointer advances, the freshness service stops old admission,
holds scoped queued/starting work on its next five-second cycle, previews
stale work, and preserves all
dispatching/running work. Application requires a preview receipt plus an
idempotency key. Cancellation happens only after the service re-reads SCQ and
sees `held`, verifies the submitted backlog claim, verifies every pair remains
submitted under that manifest, and finds no authoritative result. Checked
launch-ready claims outside SCQ follow the same audit with reserved pairs and
need no scheduler cancellation. The service then
finalizes the attempt as `superseded_unstarted`, returns each pair to
`available`, updates the claim, and retains every attempt, event, claim, and
receipt. Cancelled-but-not-finalized work is included in later previews, so an
interrupted retirement resumes safely.

The independent Rust service is
`systemd/jope-selfplay-freshness.service`. It owns the Unix socket under one
process lock; ingestion, training, evaluator canaries, and SCQ recovery do not
own freshness decisions. Scheduler jobs without a matching scoped backlog
claim and checkpoint-gate, sealed-audit, and canary manifests are outside its
hold/cancel surface.

## Freshness MCP

`target/release/jope_selfplay_mcp` is the stdio MCP server backed only by the
freshness socket. Its runtime manifest pins the Rust entrypoint and MCP
protocol/server versions. The complete tool surface is:

- `get_freshness_status`
- `get_active_checkpoint`
- `list_queue_epochs`
- `explain_job_provenance`
- `preview_stale_retirement`
- `apply_stale_retirement`
- `pause_admission`
- `resume_admission`
- `request_micro_wave`

Mutations require `preview_receipt_sha256` and `idempotency_key`. Identical
retries return the first result; key reuse for another mutation fails. There is
no arbitrary command, shell, submit, priority, hold, release, or cancel tool.

## Expanded cache and catalog

Read-only inventory:

```bash
/home/ubuntu/jope-dime/target/release/jope_selfplay_status status
```

Discovery accepts only dataset-shaped directories with `train.csv`,
`test.csv`, and `meta.json`. If a mirror copy exists, train and test hashes
must agree. Disagreement fails closed and writes an immutable receipt beneath
`test_exact_gen0/receipts/mirror-conflicts/`.

Each expanded cache contains materialized train/test CSVs, `meta.json`,
`signature_manifest.json`, compatible train encodings, train-target encoding,
and `cache-manifest.v1.json`. The cache manifest hashes every content file and
the encoder protocol. Materialization uses a per-dataset lock, temporary
directory, reflink-or-copy, complete validation, and atomic rename. Missing
encodings are generated by JOPE's pinned Hyperion and target-side train-only
tooling. Partial encoder output is not published.

Synthetic families derive from the generator recipe after seed and row-count
noise is removed. Recipe family, parameters, and generator identity remain.
Thus synthetic data is neither one global family nor one family per seed.

Before adaptive activation, freeze 256 family-disjoint datasets per task in a
private sealed manifest. Adaptive admission excludes those identities and
families. Never copy sealed identities, scores, contexts, or families into the
adaptive catalog, registry, HOF, proposals, or training corpus.

Adaptive tranches contain at most 2,048 entries. A v2 tranche activates only
after all cache, train/test, encoding, encoder-protocol, family, and split
hashes validate. Catalog files are immutable; only `current.json` advances.

## Micro-waves and proposals

A cohort is selected before proposal generation. It has exactly 64 regression
and 64 binary datasets. Per task it contains 16 rotating stable anchors, 32
least-observed/new datasets, and 16 hard/high-uncertainty datasets. Selection
is deterministic for the wave.

The checkpoint then generates proposals only for those 128 contexts. Slot zero
is greedy top-1; remaining slots are seeded nucleus samples. Proposal sets bind
checkpoint, catalog, and cohort hashes.

Each selected dataset has 256 unique semantics:

- 32 local elites;
- 64 current-checkpoint proposals;
- 96 robust/diverse task winners;
- 64 deterministic mutations/crossovers.

Manifests contain at most 100 candidates. Known authoritative protocol pairs
are reused; fresh work is reserved transactionally before publication.

The operational plan/build boundary is Rust:

```bash
target/release/jope_selfplay_wave plan \
  --catalog-pointer /path/to/catalog/current.json \
  --checkpoint-pointer /path/to/training/accepted.json \
  --corpus /path/to/strict-test-exact.sqlite \
  --sealed-cohort /path/to/sealed-cohort.json \
  --output-root /path/to/waves

# PyTorch proposal inference is the sole Python exception between these steps.

target/release/jope_selfplay_wave build \
  --cohort /path/to/pending/cohort.json \
  --catalog-pointer /path/to/catalog/current.json \
  --proposals-pointer /path/to/proposals/current.json \
  --corpus /path/to/strict-test-exact.sqlite \
  --robust-seed-bank /path/to/robust-gp-seeds.v1.json \
  --sealed-cohort /path/to/sealed-cohort.json \
  --output-root /path/to/waves
```

## Fitness and compile economy

`corpus_eval` is the broad compile-free engine. It fits once on `train.csv`
and predicts `test.csv`. Regression fitness is test R² (RMSE secondary);
binary fitness is test ROC AUC (log loss secondary). Training rows carry
absolute score, rank, percentile, incumbent regret, validity/failure,
checkpoint provenance, and adaptive labels. Datasets with no comparable
incumbent still train the ranker but cannot support strict prior-GP claims.

Promising semantics replay over a 32-dataset family-diverse panel through the
precompiled engine. Compile only a strict incumbent beater or dataset top-two,
never more than two finalists per dataset/checkpoint. Cache artifacts by:

```text
(task, dataset train hash, semantic hash, evaluator protocol,
 compiler bundle, target triple)
```

Never reuse dataset-fitted state across datasets. Champion qualification
requires compiled/interpreted prediction and score bit parity.

The Rust compile-economy controller is audit-only unless an explicit phase is
selected:

```bash
target/release/jope_compile_cache --request /path/to/request.json
target/release/jope_compile_cache --request /path/to/request.json --replay
target/release/jope_compile_cache --request /path/to/request.json --build \
  --replay-receipt /path/to/replay-receipt.json \
  --idempotency-key <stable-key>
```

Replay uses the precompiled `corpus_eval` engine on the deterministic
32-dataset family-diverse panel. Build admits at most two qualified finalists
per dataset/checkpoint and verifies fitted compiled/interpreted parity.

## GP gene injection

Export all 64 current-checkpoint seeds:

```bash
target/release/jope_gp_bridge export \
  --checkpoint-pointer /path/to/training/accepted.json \
  --proposals-pointer /path/to/proposals/current.json \
  --task binary --dataset <dataset_hash> \
  --output /path/to/gp-seeds.json \
  --receipts /path/to/gp-bridge-receipts.jsonl \
  --idempotency-key <stable-key>
```

Export fails unless slots `0..63` are unique and exactly match the requested
checkpoint/task/dataset. In generation zero replace 64 existing seed slots;
do not grow the population. On checkpoint promotion during a long run:

1. evaluate new proposals under the current dataset protocol;
2. choose the eight best new semantics with the task comparator;
3. replace the eight lowest non-elites;
4. retain all elites and the population budget;
5. record `checkpoint_proposal_injection` ancestry.

Propagate `dl_origin` through descendants. Report DL parents, descendants,
strict wins, and contribution to the final champion.

`gene-evidence.v2.jsonl` is the decision-complete consumer interface. Its key
is `(task, dataset_hash, semantic_sha256, protocol_sha256)`. It includes score,
rank, percentile, incumbent delta, genome, checkpoint provenance, receipt,
adaptive/sealed classification, and compiled parity state.

Build the repeated cross-dataset GP bank with the native scan and feed it to
both micro-wave construction and generation-zero GP merging:

```bash
target/release/jope_queue_control \
  --export-seed-bank /path/to/robust-gp-seeds.v1.json \
  --minimum-repeat-datasets 2 --seeds-per-task 512

target/release/jope_gp_bridge merge-generation-zero \
  --checkpoint-pointer /path/to/training/accepted.json \
  --proposals-pointer /path/to/proposals/current.json \
  --task binary --dataset <dataset_hash> \
  --population /path/to/generation-zero.json \
  --robust-seed-bank /path/to/robust-gp-seeds.v1.json \
  --output /path/to/seeded-generation-zero.json \
  --receipts /path/to/gp-bridge-receipts.jsonl \
  --idempotency-key <stable-key>
```

The scan ranks repeated semantics by task-separated, within-dataset
percentiles. Raw regression R² and binary AUC are never compared to each
other. The 64 newest-checkpoint DL proposals and robust across-run GP genes
are deduplicated without increasing the GP population budget.

## Capacity and rollout

The pinned desired policy calls for a two-second scheduler tick, one-second
telemetry, default multiplier 1×, and an eventual xbabe4 multiplier 2× (20
effective cores, desired target 40 CPU slots). That 40-slot target is not
live-enforced by the current SCQ build. After a compatible SCQ deployment, the
ramp is 30 slots for ten minutes before 40. Three consecutive unsafe
one-minute samples must roll it back to 20/1×. The pressure, memory, failure,
and joint runtime/throughput gates are reported by the native status/capacity
control plane.

The deployed SCQ binary in this snapshot does not contain the multiplier field
and its source tree is absent. The repository therefore ships and validates
the target config but does not restart the live daemon with an unknown field.
Merge the corresponding SCQ parser/scheduler change in its authoritative
source and only then install the pinned artifact. Until then, status reports
desired multiplier/slot target separately from physical/effective cores and
running jobs.

Roll out audit-first: snapshot; deploy freshness with
`JOPE_FRESHNESS_AUDIT_ONLY=1`; preview, then remove that override so the
five-second cycle holds scoped stale work; audit and retire only
confirmed-unstarted legacy work through a preview receipt; activate the
queue; deploy a compatible SCQ build before starting the xbabe4 ramp; freeze sealed data;
build the first tranche; start micro-waves; verify one 50,000-label
continuation and turnover; then run the two-hour fault-injected burn. Never
stop xbabe1's external GPU job, xbabe3's ratchet, or xbabe2's due trainer.

Build and activate the production runtime without operational Python:

```bash
cargo build --release --bins
target/release/jope_selfplay_bundle --activate
```

The bundler fails if any `jope-selfplay-*.service` `ExecStart` references
Python or if a Python file is outside its exact PyTorch training/proposal
allowlist. The old `scripts/{build,deploy}_selfplay_bundle.py` files are
historical test fixtures and are not production entrypoints. Review the first
freshness preview before removing `--audit-only` from the Rust freshness unit.
No bundle activation automatically applies a retirement receipt.
