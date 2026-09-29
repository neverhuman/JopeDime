# Tiny Rust Lambda: test and sizing plan

Date: 2026-09-28. Owner request: prove one S3 CSV/fold on Lambda, keep Lambda code under `/home/ubuntu/JopeDime`, use an env file without keys in Git, minimize package and successful-call cost, and use measured history to choose resources for GP programs.

## 1. Objective, boundaries, and current evidence

Build a small native Rust evaluator deployed as a ZIP on `provided.al2023`. Fetch a versioned training CSV from S3, verify its hash, evaluate one fixed fold, and preserve enough evidence to select the cheapest configuration that reliably completes. Package size is a measured constraint; billed cost and successful completion are the optimization targets.

The initial Python/XGBoost infrastructure experiment completed 12 live invocations. It is historical evidence, not the proposed production implementation. Its image, ECR repository, function, IAM role/policy, and CloudWatch log group were deleted and absence checked. All 12 model/prediction hashes matched the local image; independent metric replay, CSV corruption refusal, and validation-label poisoning passed. See `results.json`, `verification.json`, `cleanup-readback.json` and the measured report alongside this plan.

The subsequent native probe uses JopeDime's existing `feat_math::boost2::HistGbm2Model`, not an implementation marketed as XGBoost. XGBoost itself is a C++ library; a Rust wrapper would remove Python but retain that native dependency. Pure Rust `histgbm2` is the closer match for the requested deployment. It is still only one GP-relevant model head, not an entire GP search or arbitrary GP-program qualification.

**Non-goals for this experiment:** full corpus sweep; policy optimization; HWM/learning-store admission; outer-test or sealed-label use; production scheduler; provisioned concurrency; GPU support; VPC/NAT deployment; replacing unrelated JopeDime workloads; automatic merging of another owner's work.

**Assumptions:** account `714044927654`; region `us-east-2`; deployment profile `veox-jepson-prod`; current dataset bucket accessible; CPU-only jobs; input identity frozen before fitting. Small row-random fold is an infrastructure benchmark. Parkinsons has repeated subjects, so these scores are not a patient-group generalization result.

**Dependencies:** existing host AWS CLI/profile provider, Cargo 1.97.1, musl target/linker, `zip`, `jq`, AWS permissions to create/delete the owned function and scoped execution role. Runtime uses the official Rust Lambda runtime client and official SigV4 signer, plus a small HTTPS client. No complete AWS S3 SDK is bundled in the pilot. Lambda uses temporary role credentials, never the deployment profile's administrator keys.

## 2. Ownership and repository placement

Canonical repository is `/home/ubuntu/JopeDime` (case-sensitive; `/home/ubuntu/JopeDIme` does not exist). At inspection its HEAD was `1238d524961c1e09fb8afe534c514a8758ef8b81`, branch `main`, with only the shared ledger modified. Four worktrees were occupied; PR #120 was owned by another lane. Those observations expire and must be refreshed before product source/index changes.

Current standalone experiment source lives in:

```
/home/ubuntu/JopeDime/.agent/experiments/lambda-csv/
  .env                  # local mode 0600, ignored; AWS profile reference only
  .env.example          # safe settings template
  .gitignore
  env.sh                # allowlisted parser; does not source/eval the env file
  test-env.sh
  deploy-rust.sh        # AWS CLI deployment, bounded invokes, cleanup trap
  rust/Cargo.toml
  rust/Cargo.lock
  rust/src/main.rs
  rust/target/          # single reused experiment build directory
  TEST-AND-SIZING-PLAN.md
```

This directory is already ignored by the repository's `.agent/` rule. It is experiment staging, not a fifth checkout or merged product feature. Evidence and downloaded data stay outside the repository at `/home/ubuntu/scratch/jopedime-lambda-csv-20260928/`. Preserve both the source and evidence before retiring the experiment. Do not imply ignored source is backed up by Git.

Final tracked home, once a lane is available under the repository's queue rule: `/home/ubuntu/JopeDime/deploy/lambda-csv/`. Keep the small crate independently locked with its own `[workspace]` initially; use `../../crates/feat-math` after import. The experiment's current path dependency is `../../../../crates/feat-math`. Do not import its build output, `.env`, receipts, AWS credentials, or CSV payload into Git.

One owner edits the harness. Do not change existing agent branches, other scratch campaigns, `.agent/worktrees/`, `.claude/worktrees`, product dataset files, canonical score stores, generated `target/health` evidence, or the historical S3 input objects. Do not blanket-delete `target/`, prune Docker globally, switch shared GitHub identity, or merge PR #120 to unblock this unrelated experiment. No parallel implementation packets are needed now; future dispatcher and telemetry work can run separately only after interfaces are frozen and write scopes are disjoint.

## 3. Read-first order for the implementation worker

1. `/home/ubuntu/.codex/RTK.md` if present on the executing host; this session's equivalent is `/Users/bentaylor/.codex/RTK.md`. Prefix shell execution with `rtk` (`rtk proxy` for unsupported commands).
2. `/home/ubuntu/JopeDime/AGENTS.md`; then any narrower `AGENTS.md` at the proposed destination.
3. Latest `/home/ubuntu/JopeDime/AGENT_CHAT.md`, live PR queue, `git worktree list`, branch/HEAD and dirty state. Claim exact paths while holding `AGENT_CHAT.md.lock`; preserve prior claims.
4. `/home/ubuntu/JopeDime/MASTER_PLAN.md`, `docs/adr/ADR-0005-canonical-learning-loop.md`, `docs/learning.md`, `docs/parity/capability-matrix.md`. Treat historical PASS as dated evidence, not current qualification.
5. `/home/ubuntu/JopeDime/docs/datasets.md`, `crates/jope-datasets/src/catalog.rs`, `crates/jope-datasets/src/credentials.rs` for identity and transport boundaries.
6. `/home/ubuntu/JopeDime/crates/feat-math/Cargo.toml`, `build.rs`, `src/boost2.rs`, `src/folds.rs`, `src/kernel/pipeline.rs`, `src/kernel/program.rs`, and `src/model.rs` for native evaluator boundaries. The crate build script compiles an unrelated C++ emitter artifact; verify linker elimination from the deployed Rust-only path rather than claiming the entire source tree lacks C++.
7. The current harness, lockfile, this plan, `rust-local-baseline.json`, live receipts, and both cleanup readbacks. Confirm binary and source hashes before reusing a result.

## 4. Frozen pilot identity and resource envelope

Dataset `21fbfe8b53dbcd60`, Parkinsons Telemonitoring. S3 bucket `veox-jopedime-datasets-use2`. Training object:

```
blobs/sha256/d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931
version: ymyHYuYxnSHXPoK18mKgrPdHBQ4.q2Nq
bytes: 259938
shape: 1000 rows × 22 columns (21 features, last-column target)
```

Fold: sort row IDs by SHA-256 of UTF-8 `20260928:<row_index>`, select every fifth sorted ID starting at zero for validation, sort each partition back into row order. Result: 800 train, 200 validation. Record a canonical JSON hash over both ID arrays. No outer `test` blob is fetched or granted to the execution role. A future worker must preserve existing group/time-aware folds where supplied; this pilot algorithm must not replace those folds globally.

Native model: `HistGbm2Params { bags: 1, rounds: 500, leaves: 31, ..Default::default() }`. Record all resolved defaults, actual trees fitted, early-stopping behavior, model hash, predictions, and row IDs. Inner early stopping receives only the training partition. Numerical input is f64; the original XGBoost proof used f32. Do not interpret cross-algorithm scores or timing as a controlled implementation-speed comparison.

Current aggregate experiment cap: 18 synchronous invocations (12 original + at most 6 native), reserved concurrency 1, no event triggers/retries, timeout 120 seconds, maximum configured memory 6144 MiB across the whole experiment, estimated spend ceiling $2. Native sweep was planned as 128, 256, 512 MiB, one cold and one warm invocation each. The first native invocation exposed a missing S3 payload-checksum signing header before fitting; its failed receipt is retained in `rust-run/`. The repair sets `PayloadChecksumKind::XAmzSha256` and uses `rust-run-repair/`, preserving the aggregate six-call limit by counting durable intent files across both directories. Thus the repaired sweep has cold/warm pairs at 128 and 256 MiB and only a cold call at 512 MiB. Do not invent a warm 512 result. Record an intent before every invoke; timeout, crash, and unknown outcome consume the attempt. Do not silently restart a campaign directory or reset the budget. Native `deploy-rust.sh` refuses an existing run directory and an exhausted aggregate budget.

Current ceilings bound model compute far below $2 at published rates. They are not an AWS-wide account budget. A larger validation matrix needs a separately frozen invocation count, worst-case cost, and deadline; the current cap is not permission for an unbounded sweep.

## 5. Packaging and configuration work

1. Keep the runtime as a single native `bootstrap` executable, ZIP at archive root, executable mode preserved. Use `provided.al2023` and `x86_64` for the first qualified package.
2. Build musl with locked dependencies, `opt-level=3`, LTO, one codegen unit, stripping and `panic=abort`. One reused build directory; `CARGO_INCREMENTAL=0`; at most two compiler jobs. Avoid `target-cpu=native`, which could produce host-only instructions. Measure `opt-level=z` separately later: reducing bytes can increase expensive CPU time.
3. Capture `file`, `readelf -d`, binary bytes, ZIP bytes, ELF architecture, Rust toolchain, source-tree manifest, Cargo.lock SHA, and binary SHA. ZIP contents must contain only bootstrap. Proposed initial package gate: <= 5 MiB ZIP, no Python, no shared-library requirement, no credentials or env file. Smaller is desirable, but correctness and cost gates take priority over a byte target.
4. Env file is required by deployment tooling, parsed as data with an allowlist and mode 0600. Current variables: `AWS_PROFILE`, `AWS_REGION`, `AWS_ACCOUNT_ID`, `FUNCTION_NAME`, `RUN_DIR`. Do not source the file or echo it. Reject raw credential fields. Publish only `.env.example`.
5. Lambda obtains temporary credentials from its execution role. The role grants only `s3:GetObjectVersion` on the exact training blob/version and writes to its own log group. Deployment credentials never enter bootstrap, ZIP, Lambda configuration, or receipts.
6. Pilot dataset constants are intentionally frozen. Generalization should replace them with a validated job manifest (section 8), not arbitrary URLs or shell command strings. Preserve deny-by-default object/version boundaries.
7. ECR is unnecessary for a Rust ZIP. A Docker `FROM scratch` image can also be tiny, but Lambda container images still need ECR. A `scratch` container requires a runtime client and suitable TLS root certificates; static linking alone does not implement the Lambda Runtime API. Keep the ZIP path unless measurements demonstrate a reason to change.

## 6. Ordered test gates and exact commands

Run these on xbabe2. Source root below is the current ignored experiment directory; change it only after the repository ownership/import step is complete.

```sh
rtk proxy bash -n /home/ubuntu/JopeDime/.agent/experiments/lambda-csv/env.sh /home/ubuntu/JopeDime/.agent/experiments/lambda-csv/deploy-rust.sh
rtk proxy bash /home/ubuntu/JopeDime/.agent/experiments/lambda-csv/test-env.sh
rtk proxy git -C /home/ubuntu/JopeDime check-ignore .agent/experiments/lambda-csv/.env

rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only CARGO_BUILD_JOBS=2 JOPE_SOURCE_HEAD=1238d524961c1e09fb8afe534c514a8758ef8b81 /home/ubuntu/.cargo/bin/cargo +1.97.1 test --manifest-path /home/ubuntu/JopeDime/.agent/experiments/lambda-csv/rust/Cargo.toml --release --locked --target x86_64-unknown-linux-musl
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only CARGO_BUILD_JOBS=2 JOPE_SOURCE_HEAD=1238d524961c1e09fb8afe534c514a8758ef8b81 /home/ubuntu/.cargo/bin/cargo +1.97.1 build --manifest-path /home/ubuntu/JopeDime/.agent/experiments/lambda-csv/rust/Cargo.toml --release --locked --target x86_64-unknown-linux-musl

rtk proxy /home/ubuntu/JopeDime/.agent/experiments/lambda-csv/rust/target/x86_64-unknown-linux-musl/release/bootstrap --local /home/ubuntu/scratch/jopedime-lambda-csv-20260928/train.csv
rtk proxy bash /home/ubuntu/JopeDime/.agent/experiments/lambda-csv/deploy-rust.sh /home/ubuntu/JopeDime/.agent/experiments/lambda-csv/.env
```

The last command is a one-shot campaign command, not a repeat-until-green recipe. Current receipts make another invocation refuse before creation. The installed `.env` points to the repair directory; `.env.example` retains the original directory. Missing artifacts or failed tests are blockers, not skipped PASS.

Gate A — configuration: valid profile file loads; missing file, raw credential field, duplicate setting, executable expression, unknown setting and world-readable file are refused before AWS mutation. `bash -n` passes. Confirm Git exclusion and package exclusion independently.

Gate B — evaluator: corrupted CSV refused; row shape/finite values enforced; train/validation IDs disjoint and complete; fixed fold hash matches the original proof. Poison only validation labels and prove model/prediction hashes unchanged while metrics change. Future GP proof must poison outer-test/selection labels as appropriate too.

Gate C — packaging: ELF is static and correctly targeted; ZIP has exactly bootstrap; local run executes without AWS credentials or network. Preserve prediction/model hashes and parameter resolution. Never use a local timing under a heavily loaded host as a fair Lambda speed comparison.

Gate D — real AWS: function reaches Active; versioned S3 GET succeeds under the restricted role; planned memory tiers produce observed cold/warm results or an explicit missing-sample record if the budget is consumed by a failure (as at 512 MiB in this pilot); all successful fits match local model/prediction hashes (or a predeclared numerical tolerance if a future architecture changes arithmetic). Retrieve `REPORT` billed duration and Max Memory Used, not merely handler wall time. Mark failed, timed out, throttled, malformed or unavailable results explicitly. A missing sample cannot support that tier's warm-cost claim.

Gate E — independent replay: recalculate RMSE and R² from the frozen CSV, recorded row IDs and predictions; verify model/prediction/fold identities across runs. Check actual fitted tree count and early stopping to prove comparable work. Record test count and failures; absence of results is failure.

Gate F — cleanup: preserve responses and logs before deletion; remove only owned function, log group, inline policy, role, and any packaging resource created by this campaign. Re-read AWS and require function/role not-found and zero matching log groups. Preserve an explicit cleanup failure receipt; a shell trap alone is not proof of deletion. No ECR resource should be created by the Rust path.

## 7. Cost measurement and optimization criteria

For x86 first-tier Ohio pricing verified through AWS Pricing API:

```
lambda_usd = ceil(billed_ms) / 1000 * memory_mib / 1024 * 0.0000166667
             + 0.0000002
s3_get_usd = 0.0000004 per GET
```

Use actual AWS `Billed Duration`, already including billed initialization; never add Init Duration twice. Keep client latency, handler duration, fit duration and billed duration distinct. These are list-price estimates before free tier, discounts and tax, not an invoice. Include logging, S3 operations, retries and any stored outputs in broader system estimates. Input download and parse costs matter when fits become very short.

Report cold and warm separately. One cold/one warm native sample is a feasibility probe, not a reliable p95 or an established optimum. For later qualification use at least five independently cold environments and twenty warm runs per finalist, randomized tier order, with bounded concurrency and repeated identical work. Force cold starts by publishing new versions or changing an isolated test function; sequential warm calls do not establish burst-start behavior.

Compare 128/256/512/1024/1769/3072/6144/10240 MiB as a candidate future grid only after budget freeze. Lambda CPU allocation scales with memory; 1769 MiB corresponds to approximately one vCPU. Single-thread workloads may waste additional CPU entitlement at larger sizes. For parallel GP kernels, jointly vary explicit thread count and memory while preserving identical work and recording actual parallelism. A 128 MiB function can be cheapest, slower at equal cost, or fail; measure instead of assuming.

Compare arm64 separately after native dependencies and parity pass; use its region-specific rate and real timings. Lower per-GB-second price alone does not establish lower total cost.

Candidate objective:

```
expected_cost(tier) = p_cold * cost_cold(tier) + (1 - p_cold) * cost_warm(tier)
                     + expected_retry_cost(tier) + I/O_and_logging_cost(tier)
choose cheapest tier satisfying:
  predicted upper memory bound <= 0.75 * configured_memory
  predicted upper runtime bound <= 0.80 * timeout
  successful completion probability >= declared reliability target
```

Headroom fractions are conservative starting policies, not measured guarantees. Calibrate them using held-out workloads; record the policy version. Package-size reductions matter chiefly for initialization/transfer and deployment, while compute cost depends on billed GB-seconds. Do not optimize away evidence needed to detect failure.

## 8. History schema and Rust instrumentation

Create these tracked files during the later implementation import:

```
deploy/lambda-csv/rust/src/job.rs        # JobV1, validated input and work limits
deploy/lambda-csv/rust/src/receipt.rs    # ReceiptV1 and failure categories
deploy/lambda-csv/rust/src/telemetry.rs  # monotonic phase clocks and memory
deploy/lambda-csv/rust/src/evaluator.rs  # typed native evaluator dispatch
deploy/lambda-csv/rust/src/sizing.rs     # deterministic conservative selection
deploy/lambda-csv/schemas/job.v1.json
deploy/lambda-csv/schemas/receipt.v1.json
deploy/lambda-csv/tests/                 # admission, boundary, replay and sizing fixtures
docs/experiments/lambda-csv.md
```

Suggested typed contracts (names are implementation targets, not already existing APIs):

```rust
struct JobV1 {
    job_id: String,
    dataset_sha256: String,
    s3_bucket: String,
    s3_key: String,
    s3_version: String,
    fold_sha256: String,
    algorithm_id: String,
    program_sha256: Option<String>,
    params_sha256: String,
    source_sha: String,
    limits: WorkLimits,
}
enum Outcome { Success, OutOfMemory, Timeout, Rejected, InvalidData,
               EvaluatorFailure, InfrastructureFailure, Unknown }
```

Receipt required fields:

- Identity: schema/version, job/attempt/invocation IDs, source and binary hashes, lockfile hash, region, architecture, package version, algorithm/program/parameter hashes, dataset/S3 version/fold hashes.
- Input/work descriptors: encoded bytes, rows, training rows, raw features, dtype width, missing/categorical counts, sparse density, encoded width, GP operation/node count, dense expansion bound, ensemble size, tree/round/depth/leaf limits, thread count, declared caches and output limits. File bytes alone do not predict decoded memory or computational complexity.
- Resources: configured memory/timeout/tmp, cold flag/environment identity, start/end times, monotonic download/parse/prepare/fit/predict/serialize phases, CPU time, process RSS/HWM, child rusage, Lambda Max Memory Used, Init Duration, handler duration and billed duration.
- Outcome: normalized class plus original AWS error type, retries, partial-work counters, actual trees/generations, correctness flags, model/prediction identities, sample status, estimated charged cost and price-table version.

**Peak memory trap:** `/proc/self/status` VmHWM and Lambda's reported maximum can reflect the lifetime of a reused execution environment. Never treat successive warm HWM readings as independent per-job peaks. Current pilot labels its field `process_lifetime_hwm_kib` explicitly and compares identical work. For varied GP programs, run the evaluator in a fresh child and use `wait4` rusage for each child plus runtime/S3 buffers. Measure aggregate simultaneous child memory separately if parallel children are allowed. Sampling RSS can miss brief spikes; retain high-water evidence and sampled values with their measurement method.

Use monotonic clocks for durations and UTC for joining records. Keep telemetry overhead measured with an empty evaluator. A child-process boundary also lets the parent serialize a failure when the child exits; parent OOM or whole-Lambda timeout still requires coordinator-side receipts. Add invocation log joins by RequestId. Do not rely on the failed function to report its own OOM.

Store compact append-only receipts per attempt. Initially aggregate local JSONL under custody to avoid a DynamoDB write or numerous S3 PUTs dominating tiny fits. Batch upload compressed records later to a separately claimed experiment prefix, with manifests and hashes. Never store raw CSV rows, credentials, signed Authorization headers, presigned URLs or whole events in logs. Keep limited immutable prediction evidence for verification; production receipts can retain hashes plus separately stored predictions.

## 9. Sizing policy: safe table first, learned predictor later

Phase 1: nearest conservative historical envelope keyed by algorithm family, code/architecture version, rows/features/encoded-width bucket, dtype, and complexity knobs. Select the lowest estimated-cost tier among those with adequate successful evidence. For unseen or out-of-distribution jobs, use a bounded conservative tier or refuse Lambda if a known bound exceeds service limits. Never extrapolate an unbounded GP feature expansion from CSV size.

Phase 2: fit separate upper-quantile memory and runtime models from history, conditioning runtime on memory/CPU/thread settings. Use algorithm-aware terms: dense `n*p*width`, kernel `n*n*width`, transformed width, tree rounds/leaves, ensemble multiplicity, and GP structure. Purely empirical prediction must be combined with static expansion/operation limits. Do not use test-set labels or measured validation quality as sizing features.

Phase 3: shadow decisions on held-out dataset families and unseen GP structures. Avoid splitting repeated runs of the same dataset between training and validation; that exaggerates generalization. Evaluate memory-bound coverage, timeout/OOM rate, cost regret versus the measured cheapest safe tier, confidence calibration, and refusal rate. Preserve policy version and predicted bounds in every receipt. Activate routing only after predeclared thresholds pass.

Timeouts and OOMs are censored observations: runtime/required memory is at least the failed bound, not zero. Retain failures and unknowns; otherwise the predictor learns only easy jobs. Investigate preemption, malformed data and bad programs separately from true resource shortage. In production allow at most one explicit higher-tier retry when safe and idempotent; count its cost in the objective. No retry of hash mismatch or invalid program. Deterministic job IDs and conditional output creation prevent duplicate completion/admission.

Represent tiers as pinned Lambda versions/aliases or separate tier functions with identical code/config hash. Do not mutate one shared function's memory while concurrent jobs execute. Pilot updates are safe because concurrency is one and calls are sequential. Versioned aliases also make the routing decision auditable.

## 10. GP qualification matrix and difficult cases

After the native boosted-tree pilot, admit one fixed existing GP program through the real `feat-math` evaluator API. Pin program bytes, evaluator source, vocab/protocol, fold, random seed, parameters, thread count and feature-expansion limits. Compare local and Lambda predictions and metrics before measuring a matrix. A standalone model head does not qualify arbitrary GP execution.

Cover at least: simple linear/head-only program; bounded feature expansion; tree ensemble; row-quadratic kernel model; deep composition; near-memory-limit sparse-to-dense case. Use small/medium/large admitted datasets across narrow/wide schemas and natural missing/categorical patterns. Avoid changing semantics by truncating rows/features merely to finish; a rejected full job remains rejected. Define deterministic missing-value and categorical mappings with training-only fitting.

Exercise malformed CSV, ragged rows, NaN/Inf target, hash/version mismatch, unavailable S3 object, expired/denied access, oversized manifest, multiplication overflow in memory estimates, constant target (R² policy), too-small training fold, singular solver, unsupported operation, child crash, OOM, near-timeout, cold burst, duplicate invocation and cleanup interruption. Test admission/resource failure mechanisms on synthetic data before expensive real workloads. Unknown outcomes never enter canonical success records.

Jobs exceeding the 15-minute/10-GiB Lambda envelope should be explicitly routed to an appropriate batch/host executor, or partitioned only where the algorithm's semantics support it. GP search state can stay in a coordinator, with one bounded evaluation or bounded batch per invocation. Measure batching because many sub-millisecond evaluations can be dominated by requests, S3 and logging. Cache only immutable versioned input, cap cache memory/disk, and distinguish cache-hit from cold/fresh-fetch evidence.

## 11. Completion criteria and stop conditions

Pilot complete only when: env and evaluator checks pass; static ZIP identity is saved; all scheduled native invokes have an explicit outcome; local/Lambda prediction parity and independent metric replay pass for successes; cost table distinguishes cold/warm; AWS cleanup absence is verified; source/evidence paths and remaining limitations are recorded. The full adaptive system is complete only after real GP qualification, historical-data collection, held-out sizing validation, failure attribution, bounded retries and concurrency-safe routing pass separately.

Stop new compute on exhausted invocation/cost/deadline budget, credential/account/region mismatch, overlapping ownership, hash mismatch, unbounded feature allocation, unexpected secret-bearing output, repeated infrastructure failure, or cleanup failure. Preserve evidence and mark partial/failed honestly. Do not widen permissions, fetch outer tests, or reset campaign identity to make a test pass.

## 12. Logging, publication and final handoff

Append progress and stopped-state receipts to `AGENT_CHAT.md` under its lock, naming exact HEAD, owned paths, current attempt count, resource names and blockers. Preserve command/exit status, UTC times, resource budgets, source/binary/data/fold identities, tests executed and skipped, raw AWS reports, pricing response and cleanup readbacks. Hash the delivered source and receipts; document any build/preflight failure rather than overwriting it with success.

For a tracked import, first refresh remote main and claims, obtain a free existing lane, preserve current ledger changes, and create a focused claimed branch. Import only source/template/schema/docs files. Run targeted Rust and env tests, `git diff --check`, package/secret-content checks, then the repository's exact-head CI and independent-review lifecycle. Never self-approve or declare an ignored scratch experiment merged. Keep unrelated campaign source, generated files and data outside the change.

Final handoff must state: measured cheapest **tested** tier; cold/warm cost and latency; package and memory sizes; number of successful/failed/unknown calls; which algorithm actually ran; GP and architecture limitations; source location and Git status; config file handling; source and evidence hashes; cloud resources remaining; next bounded experiment. Leave no monitoring/recurring compute behind.

## References

- [AWS Rust ZIP deployment](https://docs.aws.amazon.com/lambda/latest/dg/rust-package.html)
- [AWS Rust runtime client](https://github.com/aws/aws-lambda-rust-runtime)
- [AWS custom runtime requirements](https://docs.aws.amazon.com/lambda/latest/dg/runtimes-custom.html)
- [Lambda quotas and CPU allocation](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html)
- [Lambda pricing](https://aws.amazon.com/lambda/pricing/)
- [AWS SigV4 Rust signing API](https://docs.rs/aws-sigv4/latest/aws_sigv4/http_request/index.html)

Pricing and limits were checked during this experiment; recheck when a future campaign freezes its budget.
