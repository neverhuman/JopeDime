# Cost minimization extension — 2026-09-28

This extends `../TEST-AND-SIZING-PLAN.md` following the explicit request to minimize per-training-event cost, including 2–5 different datasets per Lambda call. S3 writes remain deferred until vetting. This is a new bounded campaign; the earlier 18-call campaign is closed and its receipts remain unchanged.

## Objective and comparison unit

Find the lowest measured list-price cost for successful native training while preserving input/fold/parameters/model results. A training event is one dataset, one frozen inner fold, one `feat-math.histgbm2` fit. A comparison cohort contains exactly the same five distinct events. Report both total cohort cost and cohort cost divided by five. Do not claim a batch beats a harder singleton simply because most datasets in the batch are easier.

Five pinned training inputs are in `jobs.json`. Each records S3 key, version, SHA-256, encoded bytes, rows and columns. The cohort includes Parkinsons (1,000 rows, 21 features), Chile88 (1,000 × 7), Chatfield (195 × 12), pwLinear (166 × 10), and barium (109 × 30). CSVs stay outside Git. No test blob or arbitrary user-supplied URL is admitted.

Parameters remain `bags=1`, `rounds=500`, `leaves=31`, with the existing resolved defaults. Existing early stopping is preserved. Local baselines retained 500/31/30/196/393 trees respectively. Every comparison must retain the same actual tree count and predictions, not just the same maximum round setting. ARM permits a predeclared prediction tolerance `1e-10 * (1 + abs(reference))`; record whether exact hashes also match.

## Paths and ownership

Active source: `/home/ubuntu/JopeDime/.agent/experiments/lambda-csv/rust/src/batch.rs` and `cost/` below the same source root. Both architectures reuse `rust/target/`; no new checkout/worktree or global target override. Product source and other lanes remain read-only. Staged source is ignored; tracked import remains subject to the repository's queue, claim and reviewed-PR lifecycle.

Evidence: `/home/ubuntu/scratch/jopedime-lambda-cost-20260928/`. Resource prefix: `jope-lambda-cost-20260928-*`. Configuration: `cost/.env` mode 0600, same allowlisted profile-only parser policy; safe template `cost/.env.example`. No static keys or `.env` in either ZIP.

Read first: the prior plan and reports, root AGENTS, latest ledger/PR/worktree state, the new `campaign.json`, `jobs.json`, local baselines, source/package SHA manifest, and `cost/run.sh`. Freeze source, lockfile, target flags, parameters, data identities, rate table and budget before deployment.

## Resource envelope

Maximum 60 synchronous invocations; planned 59. Final result: 60 successful Lambda invocations and 175 successful training events. The first invocation trained successfully but a jq summary-expression syntax error stopped the controller; original raw receipts and cleanup were retained in `aws-run/`. Corrected controller used `aws-run-attempt2/`, counted the earlier intent, and completed the planned 59-call matrix without resetting the 60-call cap. No training failure occurred. Each timeout 120 s, maximum configured memory 3072 MiB; actual sweep max 2048 MiB. Reserved concurrency 1 per function; coordinator invokes sequentially. Estimated campaign ceiling $2; measured Lambda + GET list-price estimate $0.0016732936. Deadline frozen in `campaign.json` at 2026-09-28T18:36:11.169909Z. No automatic invocation retries, triggers, provisioned concurrency, or monitoring. Both functions and owned role/policy/log groups were deleted and absence verified.

Every invoke has an intent file saved before dispatch. Unknown outcomes consume budget. The run directory is an exclusive campaign lock, not a restartable loop. Preserve failed build/deploy/run artifacts. Current ARM cross-build required removing the incompatible cc-rs target spelling and setting `-C link-self-contained=no` to let Zig supply one set of musl startup objects; failed builds cost no Lambda invocations and their logs remain preserved.

## Implementation and validation sequence

1. Generalize the native wrapper to the five compile-time-admitted job identities. Validate one to five distinct job indices. Fixed region/bucket/version paths; role grants only exact `GetObjectVersion` access. Sequential fits release per-fit model state. Reuse the HTTPS agent inside each batch.
2. Preserve local baselines and fold/hash/label-poison tests. Build x86_64 and ARM musl binaries with the same source and lockfile; inspect ELF architecture, no `NEEDED` libraries, archive contents and hashes. No ECR. Use at most two build jobs and one reused target directory.
3. Screen memory tiers 512/1024/1769/2048 MiB on each architecture. For each setting invoke the same five-job batch once cold and once warm: 16 calls. This is screening, not reliable tail-latency estimation.
4. Select the minimum observed warm cohort cost from the screening results, retaining all alternative costs and cold costs. Record the selected architecture/memory in `screen-winner.json`; it is a candidate, not proof of a global optimum.
5. Reconfigure/warm that candidate with one five-job call. Compare batch sizes 1/2/3/5 across the identical five-job cohort, three repeats each, rotating size order. Sizes 2 and 3 have a final partial batch. Calls: `(5 + 3 + 2 + 1) * 3 = 33`. Score each size by total five-job cohort cost, not mean invocation cost.
6. Enable `CACHE_INPUTS=1` on the finalist. Populate cache with one five-job call, then run three five-job cached calls and five cached singleton calls: nine calls. Cache contains only five version/hash-pinned parsed input matrices, never fitted models. In current code, job index is safe because its complete manifest is compiled into the deployment version. General routing must use full content/version keys and bounded eviction.
7. Every job records `cache_hit`, `s3_get_count`, S3 bytes read, download/parse/fit/predict/job times, input dimensions, actual trees, resolved parameters, model/prediction/fold hashes and memory observations. Every invocation preserves architecture, configured memory, cold flag, request ID, billed duration, whole-Lambda peak memory, and cost. Do not charge five GETs to a cache-hit batch or advertise cached costs as fresh-read costs.
8. Verify every prediction and metric against the pinned local baseline, every cohort covers all five jobs exactly once, cold/warm state is observed, and no failure is silently excluded. Final tests include source/ZIP identity and actual cleanup absence readbacks.

## Build and run commands

Commands execute on xbabe2, using the existing Rust source directory. The host-specific Zig wrapper lives under `cost/`; Zig 0.13.0 was installed only in the campaign's `tools/` directory. No global Python/runtime package is deployed to Lambda.

```sh
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only CARGO_BUILD_JOBS=2 JOPE_SOURCE_HEAD=1238d524961c1e09fb8afe534c514a8758ef8b81 /home/ubuntu/.cargo/bin/cargo +1.97.1 build --manifest-path /home/ubuntu/JopeDime/.agent/experiments/lambda-csv/rust/Cargo.toml --release --locked --bin bootstrap-batch --target x86_64-unknown-linux-musl

rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only CARGO_BUILD_JOBS=2 JOPE_SOURCE_HEAD=1238d524961c1e09fb8afe534c514a8758ef8b81 CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_RUSTFLAGS='-C link-self-contained=no' CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER=/home/ubuntu/JopeDime/.agent/experiments/lambda-csv/cost/zig-arm-cc.sh CC_aarch64_unknown_linux_musl=/home/ubuntu/JopeDime/.agent/experiments/lambda-csv/cost/zig-arm-cc.sh AR_aarch64_unknown_linux_musl=/home/ubuntu/JopeDime/.agent/experiments/lambda-csv/cost/zig-arm-ar.sh /home/ubuntu/.cargo/bin/cargo +1.97.1 build --manifest-path /home/ubuntu/JopeDime/.agent/experiments/lambda-csv/rust/Cargo.toml --release --locked --bin bootstrap-batch --target aarch64-unknown-linux-musl

rtk proxy bash -n /home/ubuntu/JopeDime/.agent/experiments/lambda-csv/cost/run.sh
rtk proxy bash /home/ubuntu/JopeDime/.agent/experiments/lambda-csv/cost/run.sh /home/ubuntu/JopeDime/.agent/experiments/lambda-csv/cost/.env
```

The last command is single-use for this campaign. It now refuses because the aggregate 60-call budget is exhausted. Existing run directories require inspection/handoff, not deletion and rerun. Source and plan edits stay under the already claimed ignored directory; no product source/index mutation or competing lane is authorized by this protocol.

## Cost accounting

Ohio first-tier rates were checked using AWS Pricing API: x86 $0.0000166667/GB-s; ARM $0.0000133334/GB-s; request $0.0000002; S3 GET $0.0000004. Formula:

```
cohort_cost = sum(billed_seconds * configured_MiB/1024 * architecture_rate
                 + 0.0000002 + actual_S3_GET_count * 0.0000004)
cost_per_training = cohort_cost / five_successful_training_events
```

Do not add Init Duration again to billed duration. Report per-call and per-cohort distributions, medians/ranges and all raw samples; three repeated cohorts do not establish p95/p99 reliability. Also report singleton cost by dataset to show workload variation. Account for failed attempts separately when estimating delivered-success cost. List-price estimates exclude free-tier credits, discounts, taxes, logs, storage, result I/O and coordinator charges; do not label them a final invoice.

Caching saves S3 GET and parse/transfer CPU but does not save training work. Cold cache-fill cost remains real and must be amortized with an explicit hit/cold fraction. Invocation reuse, traffic distribution and input reuse determine realized savings. The five-job mixture cannot predict arbitrary GP work without the broader held-out sizing study in the main plan.

## S3 writes: deferred vetting gate and cost design

No S3 PUT is performed or granted by this campaign. Future output writing requires a reviewed tracked implementation, input/model parity, complete failure/cleanup receipts, output identity and retry rules, an explicitly admitted destination prefix, and a scoped writer role. Preserve a separate read-only mode.

After vetting, prefer one immutable output object per completed batch containing individually indexed job IDs, input/fold/program/evaluator identities, metrics, checksums, and bounded model artifacts or references. Conditional create (`If-None-Match: *`) and deterministic batch/job identity prevent retries from silently overwriting or double-admitting outputs. Partial failures need per-job status and retry only for missing jobs. Verify returned version/checksum and sampled full readback before canonical admission. Never write results into input blob keys or promote scratch results to the learning store automatically.

Verified S3 Standard PUT price is $0.005/1,000 requests. One PUT per event adds **$5 per million events**; one PUT per five-event batch adds **$1 per million events**, before bytes stored, any additional manifest writes, encryption or reads. For very cheap models, write overhead can exceed compute. Count actual object count; five model objects plus a manifest is not one batched write. Compression and retention policy affect byte costs and CPU time and require their own measurements.

Request-only fresh-read floor for five separate S3 inputs in one Lambda call is $0.44 per million training events (five GETs plus one Lambda request divided by five), before compute. Cached-input request-only floor is $0.04 per million events, but excludes cache population and assumes reuse. These are cost components, not achievable total training prices.

## Stop and handoff

Stop on budget/deadline, account/region mismatch, shape/hash/version/score mismatch, unbounded input/memory, missing billing evidence, or cleanup failure. Keep failed calls and builds. Final handoff names the cheapest measured safe configuration for this cohort, fresh versus cached cost, batching gain against the same cohort, actual per-dataset singleton costs, package/peak memory, all tested architectures/tiers, missing capabilities and sample limitations. Preserve sources, commands, lockfiles, source/binary/input hashes, raw receipts and pricing responses. Update the shared ledger under lock and leave no ongoing Lambda resources or S3 writes.
