# Storage and schema addendum — 2026-09-29

Applies to campaign `regression-20260928-v3`. This document supplements the frozen `regression-lambda-v3/SCHEMA-EVOLUTION.md` and `WORKER-PLAN.md`; their original bytes and scientific protocol identity remain preserved. No historical record is rewritten. Legacy score, HWM and TUI schemas remain unchanged.

## Actual physical layout

All paths below are relative to `s3://veox-jopedime-learning-use2/experiments/regression-evolution/v1/`.

| Data | Actual path or reference |
|---|---|
| Worker invocation pack | `attempts/<attempt-id>.json.gz` |
| Shared immutable artifact | `objects/<kind>/sha256/<content-sha>.<format>` |
| Parquet query table | `objects/query-<table>/sha256/<content-sha>.parquet` |
| Incremental query segment | `objects/query-segments/sha256/<content-sha>.json` |
| Query catalog | `objects/query-catalogs/sha256/<content-sha>.json` |
| Schema bundle and registry | Version-pinned references in the foundation manifest |
| Context admission pointer | `contexts/<context-sha>/HEAD.json` |
| Frozen protocol | `campaigns/regression-20260928-v3/protocol.json` |
| Publication snapshot | `campaigns/regression-20260928-v3/manifests/<content-sha>.json` |
| Terminal publication marker | `campaigns/regression-20260928-v3/COMMIT.json` |

The frozen design document showed proposed `schemas/...` and `campaigns/.../tables/...` hierarchies. These are logical categories, not the deployed physical locations. Resolve exact schema/table objects through the foundation/query manifests. No existing reader or object was moved to implement this clarification.

## Validation and version policy

Five core JSON Schema documents formally validate evaluations, fitted folds, invocation packs, context HEADs and context events. Two unchanged legacy schemas are retained for compatibility checks. Other operational envelopes have explicit schema IDs, source-pinned producers and documented contracts; **not every operational envelope has a separate formal JSON Schema**. That distinction supersedes the frozen design's aspirational statement that every record already has a checked schema document.

The v4 validator compiles the same frozen schema rules, admits only a specified shared keyword subset, and disables default insertion. Thirty-one valid/invalid compatibility cases matched the reference validator. Independent archive identity and metric replay checks still run. Validation performance changed; fit algorithms, objective, archive fields and protocol semantics did not.

Changes to field meaning, required fields, measurement identity or metric policy require a new major schema/protocol identity. New optional fields require a documented schema revision. Derived outputs must retain source versions/hashes and converter identity. A new result must never borrow an old score identity merely because the dataset display name is the same.

## Derived query table contracts

Each Parquet file embeds `jope.schema`, `jope.derived=true`, and source semantics in Arrow metadata. Exact columns and Arrow types are also stored in each segment's table reference. The five schema IDs are `jope.regression-query-<table>/v1`.

| Table | Grain and essential fields |
|---|---|
| `programs` | Program hash plus canonical DSL. Deduplicate across segments by hash when querying. |
| `evaluations` | One attempted program/context, with attempt ID, status, complete-fold count, fitness, fold dispersion, archive version/hash and record index. Null outer metrics explicitly mean not evaluated. |
| `fold_metrics` | Program/context/attempt/inner-fold; train/CV/baseline metric columns, model hash/codec, memberships' row counts, raw memory observation and fit timestamps. Exact row IDs, targets, predictions and model remain in the pack. |
| `invocations` | One accepted archive, its AWS request ID, handler/controller times, observed billing, memory, counts and archive locator. Calls without accepted archives are absent here; the separate CloudWatch REPORT export is needed for their observed billing. |
| `events` | UTC-microsecond events with attempt ID, and program/context/inner-fold where applicable. Use explicit event names to distinguish dispatch, fitting and durable verification. |

Join dataset/fold/recipe identity through the context hash and version-pinned context/membership references. Use original source feature maps to interpret VIP. Full stored training and the label-blind screened fallback are distinct fidelities; do not silently union their scores as equivalent evaluations.

Event meanings: `controller_dispatch` is the recorded dispatch boundary; `handler_start` is handler entry; `fit_start` and `fit_end` surround a successful native fit; `fold_archived_locally` is measured before local serialization; `handler_compute_end` precedes compression/upload; `archive_verified` follows controller read-back validation. UTC resolution is microseconds, **not a claim of microsecond accuracy or synchronized cross-host clocks**. Native monotonic durations are stored as decimal text nanoseconds. Compute throughput separately for invocations, candidate evaluations and successful fold fits.

RMSE/MAE use the stored target units. R² is null for undefined constant-target cases. `mape_fraction_nonzero_targets` excludes zero targets and is a fraction, not percentage points. `rmsle_nonnegative_only` has a restricted domain; its presence is not permission to compare it with a differently defined official scorer. The search currently uses RMSE-based fitness even for the 27 BeyondArena views whose declared headline metric is MAE/MAPE/RMSLE.

## Assessment companion

`assess_leaders.py` and the separate Rust assessment executable restore archived typed models and calculate marginal permutation importance from validation rows; no training call is made. The assessment manifest and per-fold outputs bind source pack/model/input hashes, feature mapping, row sample, permutation seeds and replay tolerance. The qualified policy uses at most 256 validation rows, 32 features and three permutations, and records incomplete feature coverage explicitly. Five generation-2 leaders passed; this is not all-view VIP qualification. Outer-development scores remain unavailable; no canonical teacher admission occurs.

## Resource and execution amendments

The original 25 GiB envelope and 256 in-flight archive admissions were qualification limits. Recorded production amendments authorize a 128 GiB accounted-storage envelope, 200 GiB free-space floor, 512 controller/archive slots and 896 reserved Lambda executions spread across four owned ARM tiers. The $1,000 total Lambda ceiling, prior reservation of $13.6535432, full-timeout reservation policy and fixed 2026-09-29 05:00 UTC deadline are unchanged. S3, transfer and logging costs are separate.

Coordinator v1 exhausted file descriptors before admitting new production work. v2 encountered SQLite writer contention; its admitted attempts were reconciled with zero new Lambda calls. v3 fixed accounting but hit expensive reference-schema validation. It drained normally on a source guard, then v4 resumed existing contexts with the compiled equivalent validator. Resource amendments, frozen source hashes, pre-dispatch dispositions and reconciliation receipts preserve this history. Operational pause dispositions are not scientific poor-score outcomes.

During execution, the budget's `unknown` count includes outstanding unsettled admissions, so it is not a count of permanently lost models. At terminal closure, unresolved claims remain blocked; they are never zero scores or automatic retries. Snapshot summaries read an interval of concurrent state, so their component counts need not form an atomic balance sheet. The terminal query proof checks exact equality of accepted and indexed attempt identities.

## Publication, recovery and remaining limits

Publication manifests bind source/binaries, schemas, audit/history closure, query catalog, leaders, accounting and model references by S3 version/hash/size. A sampled independent read-back proof is included; it is not a fresh rehash of every payload. The detached closeout supervisor waits for dispatch closure, finishes query indexing, publishes COMMIT and sets only the four owned functions' reserved concurrency to zero. Failed publication remains explicit and does not manufacture a successful COMMIT.

Core sources are frozen under `~/JopeDime/.agent/experiments/regression-lambda-v3/`; revisions and this addendum live in adjacent `regression-lambda-v3-ops/`. Private env files are excluded from source archives. The env names an existing credential profile and filesystem paths; keys are not embedded.

This campaign does not claim account-wide historical tree coverage, automatic replication, a canonical teacher dataset, all-view VIP, outer-label scoring, or completion of 100 generations. Opaque historical archives remain an explicit deduplication gap. A catastrophic runtime/upload failure can leave a physical fit without a recovered model. Next-scale work should compact derived Parquet shards, introduce paged manifest roots with a new schema, add family/alias admission, and qualify per-fold durable uploads.
