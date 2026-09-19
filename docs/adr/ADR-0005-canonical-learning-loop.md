# ADR-0005 — The canonical learning loop is lineage B; lineage A is backfilled, its live state is lost

**Status:** accepted 2026-09-18; implementation in WP11 (`jope-learning`, `jope learn|hwm|scores`) ·
**Plan sections:** "Two learning lineages", "Learning loop: Canonical loop (ADR-0005)",
"Unified high-water-mark ledger", "Lost-state contingency", "Learning store" ·
**Scope:** `crates/jope-learning`, `crates/jope-engine` (`gp-service`, `cluster` feature), `python/src/jope/train`, the learning bucket

## Context

The red-team pass found two learning lineages, not one:

- **Lineage A — the xbabe2 self-play flywheel (Jul → 8 Sep 2026):** the root crate's control plane
  (38 `systemd/` units: queue → freshness → supply → ingestion → corpus-freezer → trainer-watchdog →
  promotion; `jope_selfplay_*`, `jope_task_learning_controller`) with training in **Python**
  (`test_exact_train.py` → `train_v3.py`, warm start `fbcd5459….model.pt`, SCQ evaluation), label
  authority `strict-test-exact.sqlite`, promotion gates in `jope_selfplay_promotion.rs`, pointers
  `training/learning.json` / `accepted-by-task.json`. **Its state plane is gone from every
  filesystem on xbabe2** (`test_exact_gen0/` holds only `ops/`; disk-wide and mount searches found
  no SQLite, pointers or `training/immutable/`; full member listings of `D-runtime.tar.zst` and
  `D-recovery.tar.zst` contain none of them). What survives: four frozen corpus snapshots
  (`snapshot-004754860 … snapshot-009915070`, `corpus.train.jsonl` ≈ 82.6 GB raw each +
  `corpus.val.jsonl` + manifest), the SCQ task scratch trees (`D-runtime.tar.zst` 48.9 GB,
  `D-recovery.tar.zst` 19.5 GB), the deprecated `genebank.db` (`best_scores` 4,948 rows,
  `observations` 2.5 M) and `ops/task-elo.v1.json`-style readouts.
- **Lineage B — the 8×A100 node flywheel (8 → 15 Sep 2026):** the `parallel/` plane (Rust tch
  trainer, `runs/flywheel-20260909`, DL → GP → DL cycles) whose complete, verified state was
  preserved at decommission and is present on xbabe2: accepted checkpoint **step-60000**
  (`parallel.checkpoint.v1`, identity = sha256 of `state.json` `8ac80c26…`) among 78 checkpoints;
  teaching corpora `snapshot-000097` (10,214,332 records) and `snapshot-000444` (11,111,243
  records); all **11,903 GP winners** as `parallel.cut_champion.v1` chains (11,030
  train/validation + 873 sealed `meta_test`); per-generation `generation.json` binding checkpoint ↔
  corpus ↔ contexts ↔ seeds; the training-input closure (`accepted60000-closure`); the controller
  journal and `state.json`; and a **step-60100** checkpoint already trained on the RTX 3090
  (`world 1`, microbatch 32, accumulation 8). Its promotion verifier is
  `promotion_tournament.py` (rule F10 over `parallel.promotion_evidence.v2`), its generation loop
  `continuous_learning.py`, and it has a non-SCQ executor, `jope-parallel gp-service`, whose only
  node binding is a hard-coded CPU list.

The first plan draft treated all of this as evidence to archive. Learned state is a capability,
not an artefact.

## Decision

1. **Canonical = lineage B.** The standalone loop is built on B's data model —
   `parallel.cut_champion.v1` high-water marks, `parallel.teaching_shards.v1` corpora,
   `parallel.checkpoint.v1` lineage — driven by the `continuous_learning.py` generation loop, the
   tch trainer and the F10 tournament. It is the only lineage whose serving checkpoint, corpus
   closure, contexts, winners and hash chains are complete and independently verified
   (`green-light-final.json`: 11,903 winners, 180,730 refs), and step-60000 is what
   `crates/jope-policy` loads. `serving/current.json` ← step-60000 (`state.json` sha `8ac80c26…`);
   `training-head.json` ← step-60100 if its `state.json` verifies, else 60000.
2. **Lineage A's control plane is kept, not scheduled.** Its binaries survive as `jope-selfplay`
   subcommands with their tests; no unit runs them by default, because they schedule against files
   that no longer exist and their gate cannot be honestly re-armed without the lost sealed-panel
   zero-prior-use receipts. Its second trainer is kept as `jope learn train --trainer v3-jsonl`
   (the only trainer covering binary/multiclass/timeseries, with surviving inputs); its checkpoints
   carry `lineage: a` in the ledger and **never become the serving pointer** until a B-style
   tournament exists for those tasks.
3. **Lineage A is backfilled with original timestamps**, not rebuilt from `observations`: the four
   corpus snapshots → `corpus/snapshots/xbabe2-testexact-<watermark>/` (re-packed to
   `teaching_shards.v1`, originals kept as blobs); `genebank.best_scores` (4,948 rows) → `hwm/`
   under the distinct evaluator id `genebank-legacy` (`evaluator: "unbound"`, `replayable: false`,
   `benchmark_valid: false`); per-dataset `test_exact` maxima from the snapshot rows; SCQ receipts
   from `D-runtime.tar.zst` → `scores/ledger` batches tagged `lineage=A` after a sampled restore
   proves the receipt schema.
4. **Declared unrecoverable** (recorded in `docs/learning.md` with the archive receipts and as
   `lost` rows in the capability matrix): `strict-test-exact.sqlite` labels, `ops/pair-registry.sqlite`
   identities, `training/immutable/checkpoints/*` descendants of `fbcd5459…`,
   `training/{learning,accepted,milestone,accepted-by-task}.json`, `sealed-panel-pool.v2.json` and
   its consumption receipts, `ops/task-elo.v1.json`, `catalog/current.json`. Therefore no lineage-A
   checkpoint can be retroactively "accepted" and A's Elo ladders cannot be recomputed; the
   closest recoverable A state is snapshot-009915070 plus the `fbcd5459…` warm-start model.
5. **One ledger, `jope.hwm/v1`.** `identity_sha256 = sha256(canonical{task, dataset_hash, cut_id,
   evaluator_sha256, vocabulary_sha256, protocol})`; strict improvement only, ties keep the older
   record (`champions.rs:452`); an unbound-evaluator record can never displace a bound one for the
   same (task, dataset_hash); every event, accepted or rejected, is appended to
   `hwm/ledger/<yyyy>/<mm>/<ulid>.jsonl` (hash-chained batches). SOTArena results are not HWM —
   they are `jope.score/v1` rows with `benchmark_valid: true`.
6. **Promotion rule F10, carried over unchanged** (`promotion_tournament.py`): 16
   `meta_validation` held-out cuts × 3 matched rng seeds, each proposal fitted once under the
   ordinary evaluator with a 30 s deadline; statistic = per-cut mean over seeds of
   ln(best_challenger / best_serving); tie when |ln| ≤ 1e-12; cluster bootstrap over cuts with the
   preregistered resamples/seed/alpha; **accepted iff** mean < 0 AND wins ≥ losses AND challenger
   `diagnostic_accepted` AND challenger failures ≤ serving failures; a held-out tie is decided by
   the 32-context training-subset CI90 excluding 0; anything malformed → `inconclusive`;
   `final_tests_opened` must be false; checkpoint payload shas must equal `state.json`;
   `proposal_protocol == training_context_neural_pool_v1`; regression only (other tasks advisory,
   documented not hidden). Pointer advance = flock + ancestor compare-and-swap locally, conditional
   PUT naming the expected previous version in S3, immutable objects, `push` never deletes.
   The lineage-A gate list (per-task delta < −0.005, `proposal_validity < 0.99`, strict wins < 1,
   equal-budget must-beat-ancestor, failure-rate delta > 0.0025, p95 latency ratio > 1.25, replay
   verification, conditionality margins) is retained with its tests for `--trainer v3-jsonl`.
7. **Executor:** default `local` = `gp-service`/`cycle` with the hard-coded node CPU list replaced
   by `--cpus <list>`/`JOPE_GP_CPUS` (the one code change any run needs); `scq` = the whole
   jain-smartcluster workspace vendored behind the `jope-engine` feature `cluster` (default off;
   never used in CI), because label admission and F10 evidence bind SCQ process receipts and
   re-implementing that behind a trait would be a new algorithm.
8. **Storage:** the fourth bucket `s3://veox-jopedime-learning-use2` with append-only ledgers,
   content-addressed blobs, one mutable pointer per concept, receipts binding every write to a git
   sha, a bundle id and an evaluator sha; `$JOPE_STATE` mirrors the same tree so every service runs
   locally and S3 is the durable copy. Training-resume inputs are warm (Standard/GIR); only
   audit-only material is cold. Written only by `JopeDimeTrainer` (main, self-hosted) and humans.

## Consequences

- The next policy is trained from the same data model that produced the current one, on one RTX
  3090 (`world 1`, microbatch 32 × accumulation 8, as already done for step-60100; the trainer
  asserts `state.json.global_batch = 2048` so sampler offsets stay comparable).
- Nothing about lineage A is hidden: its surviving data is queryable in the same ledger with an
  honest `unbound`/`lineage=A` marking, and its loss is a documented fact with a contingency, not
  a footnote.
- `jope learn status|pull|freeze|train|export|qualify|promote|rollback|push`,
  `jope hwm ingest|show|verify|freeze|replay`, `jope scores append|publish|query|verify-ledger`
  wrap existing programs (no new algorithms); harness jobs `learn-smoke` and `hwm-ingest-smoke`
  on `main` and the weekly `restore-drill` keep the loop demonstrably alive.
- Risks accepted with mitigations: Python trainer determinism (autotune/AMP off, fixed seed, a
  50-step twice-run parity test, else `lineage: a` non-promotable); unbound-evaluator pollution
  (`replayable: false` records excluded from `freeze` unless `--include-unbound`); regression-only
  promotion.

## Verification

WP11 acceptance, in order: L0 second copies (`HeadObject` sha == local for every object;
replication `COMPLETED` for `hwm/` and `waves/`); L1 schema round-trips and 10 mapped champion
fixtures byte-preserved; L2 11,903 + 4,948 + snapshot maxima ingested with verifying chains and
`jope hwm show` == the `local-winner-restore` bytes; L3 `learn-smoke` green with
`--executor local` and `cargo build --features cluster` green; L4 rollback restores the previous
pointer and a concurrent promote loses the CAS deterministically; L5 metrics JSONL and twice-run
parity for the v3-jsonl trainer; L6 every harness row green once. Plan "Verification" items 10–12
(a full `jope learn train` on xbabe2 whose export loads in `jope-policy`; zero chain breaks; Elo
recomputed from the ledger equals the published 2259.8 for `jope-dime`).
