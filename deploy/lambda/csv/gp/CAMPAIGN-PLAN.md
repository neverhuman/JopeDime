# GP corpus campaign: execution and review plan

## Objective and completion contract

Run a real, bounded GP training/search event on each runnable S3 dataset source; account for every census entry, including protected or unusable inputs. Save complete replayable fitted artifacts, candidate lineage, audit predictions, high-resolution timing, failures, and resource/cost receipts to an immutable S3 archive and a verified local mirror. Keep total newly admitted Lambda liability below $100. Produce burst plots and machine-readable timing history.

The frozen census contains 3,859 canonical datasets and 142 additional BeyondArena source entries, with 3,722 named views. One canonical stored training CSV or one preregistered BeyondArena view per source is the event, not all folds/views. Cross-corpus overlap is unresolved; 4,001 is a source-entry count, not 4,001 independent families. Seven known sealed canonical sources and seven zero-feature sources are explicit exclusions. Unknown family authority remains explicit and non-teaching.

Completion requires a disposition for all 4,001 entries, real local/ARM canary parity, validated model/prediction replay and metrics for every successful event, exact input/source/function bindings, complete attempt accounting, successful conditional S3 writes and readback, restoration into a fresh directory from the commit anchor, and cleanup readbacks for temporary compute. A failure is a recorded result, not fabricated success. Count full searches, partially completed searches, no-offspring fits, unsupported inputs, failed/unknown attempts, and retries separately.

Non-goals: canonical HWM admission, teacher-corpus ingestion, neural optimizer training, serving-policy promotion, SOTA claims, full-capacity GP grammar, all 3,722 BeyondArena views, unrelated account resource changes, or a global AWS prepaid spending limit. The task is experimental GP screening and profiling. Original dataset preprocessing may precede the stored split; experimental classification/timeseries is distinct from the qualified regression-only product engine.

## Read first and ownership

On `xbabe2`, read in order:

1. `/home/ubuntu/JopeDime/AGENTS.md`, `AGENT_CHAT.md`, and `MASTER_PLAN.md` product target.
2. `docs/adr/ADR-0005-canonical-learning-loop.md`, `docs/learning.md`, `docs/datasets.md`.
3. `crates/feat-math/src/evolve.rs`, the `fit_table` and `BoundModel` implementation under `src/advanced_v3/`, and `crates/jope-engine/src/task_contract.rs`.
4. `.agent/experiments/lambda-csv/gp/FLYWHEEL.md`, `STORAGE.md`, `inventory.py`, `curation.py`, `control.py`, `campaign.py`, `budget.py`, `writer.py`, and `store.py`.
5. `.agent/experiments/lambda-csv/rust/src/gp.rs`, `gp_v2.rs`, `Cargo.toml`, `Cargo.lock`, plus frozen manifests under `gp/corpus/`.

Canonical checkout observed at `4d9a27e64d31c57b6573a84e35f22c69ebb2b1b6`. All experiment source stays under `/home/ubuntu/JopeDime/.agent/experiments/lambda-csv/`. This is ignored staging, not merged product code. Root owns deployment, budgets, invocation ledger and S3 mutations. Disjoint workers own runtime source, inventory/curation/storage code, and independent review documentation respectively. No worker creates a new worktree or writes product source, Git refs/index, canonical learning pointers, dataset buckets or other agents' lanes. `AGENT_CHAT.md` appends use its flock.

Bulk inputs and receipts live under `/home/ubuntu/scratch/jopedime-lambda-gp-20260928/`, outside Git. Reuse `lambda-csv/rust/target`, with `CARGO_INCREMENTAL=0`, `CARGO_PROFILE_DEV_DEBUG=line-tables-only`, at most two compiler jobs and no global `CARGO_TARGET_DIR` override. Never delete old failed-build or failed-canary evidence.

## Spend and concurrency controls

`jopedime-lambda-monthly-100` monitors monthly account Lambda service charges at $100. Its automatic action at $80 attaches `JopeDimeLambdaBudgetStop` to `JopeDimeLambdaDispatcher`. The SNS topic is a machine notification destination, with no configured human email subscription. Budgets are delayed and cannot guarantee an account-wide hard cutoff; unrelated existing Lambda functions remain untouched.

Every controller invocation reserves integer nanodollars in `admission.sqlite3` using `BEGIN IMMEDIATE` and `synchronous=FULL` before the SDK call. Budget and deadline are immutable on resume. Reservations include all in-flight, failed and ambiguous calls and are never refunded, even after cheap success. No automatic SDK invocation retries. A retry is a new attempt against the same ledger. The bound uses a conservative x86 price ceiling even for ARM, two configured handler deadlines plus 130 seconds for initialization/reset, and request overhead. The controller verifies published Lambda code SHA, architecture, memory and timeout against the frozen version before dispatch.

The campaign deadline is `2026-09-28T23:00:00Z`; maximum 10,000 attempted invocations, $100 cumulative reservation, batch size 1–5. Four ARM memory tiers: 512/1024/2048/3072 MiB, timeout 180 seconds. Observed account concurrency is 1,000 with four reserved elsewhere; at most 896 are reservable while leaving 100 unreserved. Allocate 768/64/32/32 across our tiers, use only observed available capacity, and retain all unrelated resources. No provisioned concurrency, event sources, asynchronous retries, ECR, VPC or recurring scheduler.

Stop new admission on cap, deadline, STOP file, source/input/version mismatch, loss of scoped permissions, secret-bearing output, or ten invocation-level infrastructure failures. Expected per-job deferrals do not count as infrastructure failures. Wait for in-flight work, retain uncertainty, then preserve logs and clean up owned resources.

## Ordered execution

1. **Inventory:** pin catalog, manifests, metadata and training object versions/hashes; reconcile every source ID. Read no original canonical test CSV. Preserve historical family protection and original task. Freeze `train-jobs.json` and its SHA before compilation.
2. **Curate extra sources:** an isolated data preparer streams hash-verified raw Parquet, selects official training-row positions before any values reach GP, and emits only a bounded deterministic sample (maximum 4,096 rows) and metadata-selected maximum 512 original features. Keep official train order, exact sampled positions, original column names/types and source bindings. Stateless encodings and independent reversible target codes must not learn from audit labels. Record group/time qualification limits. The Rust learning worker sees only the derived training CSV. Test poisoning of raw outer features/labels.
3. **Runtime:** use existing `feat_math::evolve` mutation/crossover and actual `fit_table`. Twelve candidate attempts; compact width-preserving transformations and task-bound ridge-family heads. Fit-only top-64 variance projection; all admitted CSV rows retained. Fixed 60/20/20 fitting/selection/audit split; fit vocabulary/preprocessing on fitting rows. Freeze the winning program using selection only, then evaluate audit once. Capture candidate parentage and successful evolutionary offspring. Preserve original task, surrogate/alias notes, schema and projection. Save complete `BoundModel` plus native roundtrip check. Require serde_json `float_roundtrip`; otherwise real fitted vocabulary hashes and prediction replay can change.
4. **Local gates:** unit tests for disjoint folds, corrupt hashes, forbidden grammar, sealed admission, raw audit poisoning, fitted-state bit-preserving replay, timestamp spans/failures and extra-bucket boundaries. Then five real-data canaries across tasks, including the largest 138 MB/20,532-column dataset. Verify status, twelve attempts, offspring, model reconstruction, independent prediction hashes and metrics. Exit status alone is insufficient for older CLI receipts.
5. **AWS gates:** deploy a tiny static ARM `bootstrap` ZIP on `provided.al2023`; pin numbered versions and package hashes. Execution role grants only owned log-stream writes, no S3. Coordinator supplies short-lived, versioned, exact-key presigned GETs from the immutable admission manifest; never store/log those URLs. Verify all role policies/trust. Invoke task/wide-data canaries at bounded concurrency and compare with local outputs before the burst.
6. **Burst:** run a small live pilot, then all remaining jobs with metadata-based tier routing and batches capped by rows/bytes. Schedule as many as the observed account allowance permits. A trusted completed receipt can suppress duplicate work; an unreceipted, wrong-version or uncertain result cannot. Explicitly retry deferred or recoverable jobs as singletons with fresh reservations; inspect numerical/input failures before retrying. Record counts of completed searches and partial work.
7. **Archive:** use the scoped archive writer to conditionally create objects only inside `experiments/lambda-gp/v1/campaigns/gp-corpus-20260928-v1/` in the learning bucket. Live-test duplicate identity, denied unconditional overwrite, denied deletion and full readback. Batch complete records into deterministic compressed JSONL shards, indexed by dataset/job/attempt. Store input/preprocessing bindings, source/package snapshots, coverage, attempts, models, telemetry and derived Parquet/plots. Upload dependencies, then immutable manifest, then COMMIT. No canonical store pointer moves.
8. **Verify and close:** independently recompute all audit metrics, IEEE prediction hashes, search selection and coverage; retain exact source/build/config identities. Restore every committed object into a fresh local directory using an externally retained COMMIT SHA, verify compressed/decompressed hashes and counts. Remove owned Lambda functions/runtime and temporary archive writer roles after preserving receipts; keep the explicit monthly budget/dispatcher controls. Record stopped-head handoff and final source/evidence locations.

## Durable return and recovered coordinator incident

The initial large dispatch hit the controller's soft file-descriptor limit of 1,024. Preserve its 506 admissions: 500 are proven not dispatched, six remain uncertain, and two additional unattributed executions have actual CloudWatch billing reports. None of these reservations is refunded. The five initial successful canaries remain separate. `coordinator-incident.json` and the original log are permanent evidence.

The repaired controller closes SQLite connections explicitly, permits at most 16 simultaneous connections, raises its own descriptor soft limit to 65,536, and preloads credentials before starting workers. An independent 896-worker test holds 1,792 real socket descriptors while reserving/completing 896 local test admissions, and verifies descriptor count returns to baseline. This test makes no AWS training calls.

The hardened runtime requires an exact-object conditional presigned PUT and a unique attempt ID. It writes the full raw result to the campaign's `raw-attempts/<attempt>.json` before returning. The API response adds the object VersionId, SHA256, bytes and upload timing. The controller reads that exact version and validates the complete raw body against the result before marking success. Runtime IAM remains logs-only; temporary signed capabilities supply precisely scoped reads and writes. No URLs are written to receipts. Archive writes and readback GETs are included in cost reporting.

Do not modify the original campaign manifest to deploy the hardened code. `canonical-durable` and `extra` wave manifests pin new numbered versions while retaining the original budget, deadline and ledger. A new wave is not a new spending allowance. Recovered output must retain the original attempt/request IDs and any unknown billing status; do not relabel uncertainty as a clean success without evidence.

## Logging contract

Use UTC epoch microseconds for cross-process joins, integer nanoseconds encoded as strings where JSON/JavaScript cannot represent them exactly, and monotonic durations for intervals. Timestamp precision is not a claim of microsecond clock synchronization. Record coordinator NTP state and AWS HTTP Date; retain clock-domain metadata. Never subtract monotonic readings from different processes.

Preserve queue, worker admission, request-start, response-headers, response-complete or failure times; Lambda handler start/end, environment ID and cold status; per-dataset download/prepare/search boundaries; per-candidate fit/predict boundaries, ancestry, success/failure and monotonic intervals. Keep AWS RequestId, published version, billed duration, initialization, configured memory, reported peak memory and process-lifetime RSS high-water distinctly. HWM is not a per-job peak in a reused process.

Derived spans/events must distinguish Lambda invocations, dataset training events and candidate fits. Produce one-second starts/completions, active concurrency via start/end sweep, latency distributions, task/memory breakdowns and cost-per-event history. State whether counts include retries and whether the window is dispatch, handler or actual job execution. Keep authoritative raw receipts; plots and Parquet are rebuildable views.

`handler_end_utc_us` ends computation before the archive PUT; `response_ready_utc_us` ends the full observed handler span. Retain both. Structured CloudWatch events link handler, job and candidate starts/ends plus archive start/completion by attempt ID and AWS RequestId. Export complete CloudWatch messages, event IDs, log streams, service timestamps and ingestion timestamps with `collect_logs.py`; reconcile every distinct REPORT once and retain the two original unattributed reports explicitly. Raw microseconds and nanosecond strings must survive Parquet conversion and JavaScript plotting without silent rounding.

## Validation commands and handoff artifacts

All shell commands use `rtk`. From the remote source directory:

```sh
rtk proxy python3 -m unittest -v test_control test_store
rtk proxy python3 budget.py .env
rtk proxy python3 writer.py setup
rtk proxy python3 writer.py canary
rtk proxy python3 campaign.py deploy --binary ../rust/target/aarch64-unknown-linux-musl/release/bootstrap-gp
rtk proxy python3 campaign.py run --ids 0,11,37,543,1417 --batch-size 1 --concurrency 4 --phase canary
rtk proxy python3 campaign.py run --concurrency 896 --phase corpus
rtk proxy python3 campaign.py deploy-extra --wave canonical-durable --binary ../rust/target/aarch64-unknown-linux-musl/release/bootstrap-gp
rtk proxy python3 campaign.py run --wave canonical-durable --ids 0,11,37,543,1417 --batch-size 1 --concurrency 4 --phase canary-durable
rtk proxy python3 campaign.py run --wave canonical-durable --concurrency 896 --phase corpus
rtk proxy python3 campaign.py deploy-extra --wave extra --binary ../rust/target/aarch64-unknown-linux-musl/release/bootstrap-gp-extra
rtk proxy python3 campaign.py run --wave extra --ids 2,4,20 --batch-size 1 --concurrency 3 --phase canary-extra
rtk proxy python3 campaign.py run --wave extra --concurrency 142 --phase corpus
rtk proxy python3 collect_logs.py
rtk proxy python3 campaign.py status
rtk proxy python3 campaign.py cleanup
```

These are one campaign's commands, not a recipe for silently resetting an exhausted ledger. A deploy readback already present means resume the pinned campaign. ARM cross-compilation requires the retained Zig wrappers and `RUSTFLAGS='-C link-self-contained=no'`; exact successful commands and tool versions belong in build receipts.

Expected handoff: source and dependency hash manifest, frozen job/campaign manifests, raw attempts and reservation database, complete coverage and timing JSONL/Parquet, independent verification, source/evidence bundle, S3 COMMIT URI/hash/version, fresh restore proof, budget/IAM/cleanup readbacks, interactive burst review and static export figures. Explain experimental limitations and next admission work without claiming policy learning or promotion happened.
