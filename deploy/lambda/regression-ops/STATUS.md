# Regression campaign and S3 audit handoff

**Audit and storage implementation complete; broad evolution remains running.** The latest verified publication snapshot was measured around **2026-09-29 00:50 UTC**. It is not a claim that all datasets reached 100 generations.

| Measurement | Snapshot |
|---|---:|
| Frozen regression source views | 3,533 |
| Eligible views | 3,502 |
| Explicit exclusions | 31 |
| Views with a committed population | 1,849 |
| Generation distribution | 1,844 at generation 0; 5 at generation 2 |
| Accepted invocation archives | 8,371 |
| Available AWS REPORT records | 8,374 |
| Observed Lambda list-price estimate | $0.9337 |
| Conservative total reservation, including prior runs | $21.0259 |
| Approved controlled-Lambda ceiling | $1,000 |
| Accounted experiment storage | 31.78 GB decimal |

Snapshot components are read while work continues, so they are not an atomic balance sheet. In-flight/unsettled admissions contribute to the budget's `unknown` count. Unresolved terminal claims remain explicit and blocked from automatic retries. Observed cost comes from available AWS billed-duration reports at ARM list prices; missing reports, S3, data transfer and logs are not included. The local reservation is deliberately much larger than observed usage and is not an invoice.

The independent plot snapshot at 00:48 UTC covers 7,997 indexed invocation archives, 46,380 complete three-fold evaluations, and 149,904 retained successful fold fits. It also retains 16,926 candidate timeouts and 664 failed candidates. The observed one-second peak is 538 fold completions and 96 handler starts. These are burst maxima, not sustained rates. Its reconstructed peak compute overlap is 284 handlers, with one-second rounding and accepted-archive coverage only; it is not AWS measured concurrency.

That plot snapshot has $0.8698 of observed accepted-archive Lambda charges, about $0.00001875 per complete three-fold evaluation when those charges are spread over completed evaluations. This early mixed-program/small-input sample is not a price promise for the remaining larger datasets. Full-campaign CloudWatch reporting also covers observed invocations that did not yield accepted archives.

## What was preserved and added

The current-key audit counted 976,592 objects / 583.41 GB in the four primary buckets. It also sampled 3,913 source JSON/JSONL objects and inventoried bucket controls. Old source, research, staging, models, canonical score/HWM contracts and historical records remain intact. The legacy metric enum and native Rust score hashing semantics are preserved.

New experiments use version-pinned immutable packs and manifests under:

`s3://veox-jopedime-learning-use2/experiments/regression-evolution/v1/`

The directly discoverable schema entry point is:

`campaigns/regression-20260928-v3/schema-registry/v1.json`

It links five formal new schemas, two unchanged legacy schemas, and the audit/addendum documents. Parquet tables physically live under `objects/query-<table>/sha256/`, with catalog references providing campaign organization. Historical folders were not migrated.

The latest verified campaign manifest is:

`campaigns/regression-20260928-v3/manifests/45ebf4a2a6eac2bf937fcf18a56f001ddc48b0cfc85be5ca43f47f4bb9d8bbe3.json`

Version ID: `6h4EgRDg9T4SNVTwJw4tznRmabwu1drq`. Exact key/version/hash references and the sampled restoration proof are in `publication-latest.json` and `publication-index.json` alongside this report. The manifest contains source/binary, data, model, query and evidence references. Read-back validates publication roots and representative fitted archives; it is not an independent second download of every payload.

## Execution and recovery

Core code: `~/JopeDime/.agent/experiments/regression-lambda-v3/`.

Post-freeze operations, assessments and documentation: `~/JopeDime/.agent/experiments/regression-lambda-v3-ops/`.

Runtime ledger/evidence: `~/scratch/jopedime-regression-lambda-v3-20260928/`.

The private `.env` is mode 0600 and contains only profile/run/campaign settings. Both source lanes are Git-ignored; source snapshots are retained in S3. The ARM ZIP is 1,645,898 bytes and uses no ECR image. Product Git HEAD remains `a8e4ce8f1f7cdf2f518e867147c4f0b038d966c5`; only the shared agent journal has tracked changes.

Running processes at handoff:

- v4 coordinator PID 1458505; `production-v4.log`.
- Derived query collector PID 754334; `query-tables.log`.
- Closeout supervisor PID 2059359; `closeout.log`.

PIDs are observations, not reusable authority. Match command, Linux boot ID and process start ticks before signaling; the supervisor records these in `closeout-launch.json`.

Stages are breadth-first generation 0, then 10, 25, 50, 100 and at most 256 for surviving views. The fixed deadline is **2026-09-29 05:00 UTC / September 28, 11:00 p.m. Denver**, with earlier stops for budget, 128 GiB accounted storage, 200 GiB host free-space floor or integrity failure. It is possible to stop below the desired generation count.

The supervisor performs no training. After the coordinator closes it stops the derived indexer, finishes indexing accepted archives, verifies exact attempt-set equality, runs terminal publication, and sets only the four owned Lambda functions' reserved concurrency to zero. Success writes `closeout-complete.json` and a create-only campaign `COMMIT.json`; failure writes `closeout-failed.json` without claiming successful publication. Full terminal closure has not happened yet. The snapshot publication path passed live S3 restoration checks; supervisor process-identity and index-equality checks passed local qualification.

For the existing colorful Rust TUI:

```sh
rtk proxy ssh -t xbabe2 '~/JopeDime/.agent/experiments/lambda-watch/watch --snapshot-only --env ~/JopeDime/.agent/experiments/regression-lambda-v3-ops/watch.env'
```

Its new feed passed the viewer's `--check` with 2,439 tracked view entries in live mode. View entries include pending views, not just committed generations. Refresh age and schema are visible; the TUI is an operator view, not the scientific ledger. Some counters cover the current coordinator instance, whereas the durable tables cover earlier revisions too.

## Scientific and storage boundaries

- Indexed historical exclusion covers 8,345,677 program records and 10,544,544 dataset/program pairs. Opaque historical archives and separate custody buckets remain an explicit coverage gap. Canonical syntax identity does not prove algebraic novelty.
- Existing historical scores lack exact full context and are proposal-only. CV champion seeds directly map to 2,090 of 3,533 source views; not every BeyondArena view starts from a historical champion.
- CV metrics, folds, models, predictions, resource observations and failures are retained separately. The worker does not receive outer test labels. No new outer-development labels or canonical teacher admission have been produced.
- A separate Rust tool restored and replayed five generation-2 leaders and computed validation permutation importance without fitting again. This does not constitute all-view VIP assessment.
- Fitness currently uses CV RMSE-to-baseline skill plus dispersion, including on the 27 BeyondArena views whose official headline metric is MAE/MAPE/RMSLE. Metric-specific optimization remains future work.
- Canonical source preprocessing may predate the split; full and screened fidelities are explicitly distinct. The family/alias registry and clean meta-learning admission gate remain necessary.
- A catastrophic process or upload failure can strand a physical fit before its archive is durable. Such unknowns are never converted to zero scores or automatically retrained.
- Existing replication does not cover this experiment prefix. No retention, replication, lifecycle or historical data migration was silently introduced.

Next scale improvements are Parquet compaction, paged manifest roots with an explicit new reader/schema, per-fold durable output, algorithm-aware memory routing, and family/alias-aware teacher admission. These preserve the useful existing S3 design while addressing its actual growth limits.
