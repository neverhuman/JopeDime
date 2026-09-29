# Regression evolution storage compatibility decision — 2026-09-28

Status: frozen experimental contract. Dated deployment, qualification and campaign receipts determine operational status. No canonical schema is replaced.

The existing source of truth is JopeDime at a8e4ce8f1f7cdf2f518e867147c4f0b038d966c5: schemas/jope.score.v1.schema.json, schemas/jope.hwm.v1.schema.json, crates/jope-learning/src/{record,canon,scores,hwm,champion,bank}.rs, ADR-0005, and the existing Lambda STORAGE.md. The live S3 census and sampled record receipts accompany this document. Counts are observations over a scan interval, not a bucket-wide atomic snapshot.

## Preserve these conventions

- Dataset bucket owns versioned source data, catalogs, collection manifests and content-addressed data blobs. Existing research results there are historical custody; do not relocate or delete them automatically.
- Learning bucket owns new experimental evaluations, immutable result blobs and publication manifests. Promotion to canonical scores/HWM/teacher data is a separate validated action.
- Keep content hashes, S3 VersionIds, byte lengths, evaluator and vocabulary identities, explicit comparison cuts, source lineage and previous-event links. Hash a program's canonical DSL bytes, never just its displayed nickname.
- Preserve immutable records and create-only COMMIT markers. Update only explicitly mutable pointers through compare-and-swap. Versioning alone is not a concurrent writer lock or a retention policy.
- Keep raw measured results authoritative. Parquet/SQLite are derived query indexes, recoverable from committed manifests. Never turn failed or missing outcomes into zero losses.
- Retain every successful fitted model and its predictions, including unsuccessful search candidates and successful folds of subsequently failed candidates. Elites alone are an inadequate recovery/learning archive.
- Never duplicate source data or fitted models merely to put them in another dataset/fold folder: manifests and indexes reference shared content-addressed objects.

## Existing readers must continue to work

| Contract | Remains unchanged | Additional information lives in |
|---|---|---|
| jope.score/v1 | Existing fields, metric enum, direction, cut_id, receipt hash, benchmark_valid, integer-second timestamp, Rust canonical serialization | jope.gp-evaluation/v1 envelope and separate event records |
| jope.hwm/v1 | Exact identity and optional ComparisonContext; strict improvement; sealed/admission semantics | Experimental generation snapshots; no direct HWM promotion |
| parallel.cut_champion.v1 | Original identity, DSL and source lineage | Version-pinned historical source references |
| jope.lambda-gp-* legacy diagnostics | Original records, caps, metrics and omissions retained verbatim | Import manifests with explicit cache eligibility and missing-field reasons |
| jope.lambda-watch.snapshot.v1 | Existing TUI schema | New producer emits this schema; snapshot is not scientific evidence |

The v1 score metric enum supports r2, oof_r2, auc, macro_f1, negative_mase, rmse, logloss and mase. MAE, MAPE, RMSLE, fold vectors, VIP stability and arbitrary added fields MUST NOT be inserted into a jope.score/v1 row. New research metrics belong to a versioned envelope. A future canonical metric extension requires a reviewed reader/schema migration.

Hashing trap: the score schema's field description says “without row_sha256”, but its top description and actual Rust compute_row_sha256() blank that field to the empty string before hashing. The Rust implementation is authoritative. Python's float JSON spelling is not automatically Rust's shortest-roundtrip spelling. Canonical score export must use/verify the native serializer. Envelope hashes are a separately named contract and do not pretend to be legacy row hashes.

## Additive experimental namespace

    s3://veox-jopedime-learning-use2/experiments/regression-evolution/v1/
      schemas/<schema-id>/<schema-content-sha>.json
      objects/<kind>/sha256/<content-sha>.<format>
      contexts/<context-sha>/HEAD.json
      campaigns/<campaign-id>/protocol.json
      campaigns/<campaign-id>/manifests/<manifest-sha>.json
      campaigns/<campaign-id>/COMMIT.json
      campaigns/<campaign-id>/tables/<table>/<content-sha>.parquet

Objects can include input closures, program catalogs, invocation packs, source snapshots, generation states and audit evidence. Immutable manifests name all dependencies by bucket/key/version/hash/size. A committed generation retains the exact ordered population, parents/operators/seeds, seen-program set/index, next deterministic proposal counter and complete result references. Generation labels count completed transitions, never requests merely submitted.

Context HEAD is the only mutable cache pointer. It binds a versioned immutable context, durable novelty admissions, completed result references and unresolved execution claims. Conditional writes serialize competing controllers; no read-then-unconditional-write. Unknown execution claims never expire into automatic retraining. Recover archived output first; require explicit reconciliation if an outcome cannot be established. Cache correctness takes precedence over apparent utilization.

The HEAD body contains a revision and one immutable event reference. Events contain deltas; every 64 revisions produces a full checkpoint. Reconstruct at most 64 events. Missing or corrupt event bodies fail closed. This limits versioned HEAD growth and avoids storing the growing seen set on every admission. Synthetic live contention qualification used 16 writers and admitted exactly one claim, including restart and closed-claim rejection.

Numeric input format JRG1 stores magic, little-endian u64 row count, u32 feature count, one ordered-fold byte, i32 fold assignments, and row-major f64 features plus target. Its separate membership JSON retains original row IDs and feature mapping. Full stored training inputs and the label-blind 4096-row/128-feature screening fallback have different explicit fidelity identities. Screening does not substitute for full-source evaluation. Existing canonical CSV preprocessing may predate the split and is labeled as a limitation.

## Measurement identity

The exact context includes: source versions/content hashes; selected training rows and their original IDs; outer repeat/fold; all inner train/validation memberships; feature recipe and target units; scorer and numerical policy; actual fitting seed semantics; fidelity/row/feature caps; evaluator binary hash; architecture; grammar version; fit deadline and resource policy when those can change results. Program identity is canonical parsed DSL, not algebraic simplification.

A reusable evaluation is (context hash, program hash) with a verified complete artifact closure. The same tree under changed rows, fold, preprocessing, evaluator or fidelity is a DIFFERENT scientific evaluation. Historical “good score” rows without these bindings are useful proposals, not cache hits. Preserve original and canonical program hashes when importing.

Avoid S3 LIST/HEAD per candidate. Load the context index once, reject repeats locally, then atomically reserve a batch of new identities in the shared context pointer before dispatch. Batch model/prediction artifacts into immutable packs and build compact metric tables. Bloom filters may accelerate negative lookups, but an authoritative exact set decides duplicates.

## CV, development outcomes, and teaching labels

Keep inner-CV outcomes, training-only confirmation outcomes and outer-development outcomes separate, with label_role, stage, metric, units, row-membership hash and exposure provenance. Optimizer selection never reads outer labels. Meta-training family development scores can later supervise a neural outcome head; they then cease to be an unbiased test of that learned policy. Meta-validation/meta-test families remain separated by source family and aliases.

Store raw fold metric vectors, paired baseline vectors, validation row IDs/predictions, uncertainty descriptors, candidate complexity, per-fit failures, feature representation and all resource observations. Correlated CV folds do not provide independent replications; SD/sqrt(K) is descriptive rather than a calibrated confidence interval. Retain negative examples, not only winners. Derived teacher views can separately target CV skill, held-out skill, gap, reliability and compute cost.

Permutation VIP/stability should be computed from archived fitted fold models and validation rows, so it requires predictions rather than duplicate fits. Preserve permutation seeds, original feature lineage and per-fold importance vectors; distinguish transformed coefficients from original-feature importance. Compare importance only across compatible representations.

## Version and migration rules

Every new envelope/table has an explicit schema ID and a checked schema document. Additive optional fields require a schema-document revision; changed field meanings, identity formulas, metric policies or required fields require a new major schema/protocol identity. No in-place rewriting of old records. A migration emits a new artifact, source hash/version, converter source hash, exclusions and roundtrip/reader checks. Old and new remain queryable through an explicit union view. No canonical HWM, score ledger, accepted policy or teaching pointer is changed by this experiment.

## Completion and evidence

Historical exclusion covers the indexed genebank, all 60,276 listed lineage-B observation files, frozen discovery programs and the completed Lambda candidate records. The index contains 8,345,677 program records and 10,544,544 dataset/program pairs. Opaque tar archives and every separate custody bucket have not been exhaustively decoded; do not claim account-wide historical completeness. No legacy fitness is reused without exact context bindings. Canonical score export remains disabled: the compatibility fixture checks unchanged JSON schemas and actual Rust Value serialization, not native ScoreRecord deserialization.

Each successful child fold is flushed and fsynced locally. The invocation uploads its bounded compressed archive before acknowledging success. A catastrophic runtime failure or failed upload can still leave an unknown attempt without a recovered model; claims remain blocked and the gap is reported. This is not a claim that arbitrary crashes preserve every physical intermediate fit. Archived complete fits include model, prediction vector, metric vector, row memberships and microsecond UTC timestamps; monotonic durations do not assume synchronized cross-host clocks.

Outer-development scoring and permutation VIP are separate post-freeze assessment work. They are not implemented by the fitting worker and are never implied by the presence of nullable outer-score fields. No canonical teacher admission occurs here.

Before broad dispatch: finish live layout audit; retain representative old records and schema hashes; validate new envelopes and the compatibility mapping; prove concurrent novelty admission and crash/retry behavior; test fold isolation and numerical replay; qualify real Lambda archives and cost accounting. Before claiming completion: compare frozen corpus denominator with per-view generation dispositions, restore committed artifacts from S3, and publish actual cost/coverage plus remaining failures. Bucket replication/lifecycle gaps are recommendations, not implicitly authorized historical data migration.
