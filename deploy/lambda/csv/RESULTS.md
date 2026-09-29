# Rust Lambda feasibility: measured results

Native Rust deployment works without ECR. A ZIP containing one statically linked `bootstrap` fetched a version-pinned CSV from S3 and trained JopeDime's existing `feat-math.histgbm2` implementation successfully. Five successful native runs produced exactly the same model and prediction hashes as the local Rust executable; independent RMSE/R² replay passed.

The ZIP is **1,442,120 bytes (1.375 MiB)**; executable **2,904,912 bytes (2.770 MiB)**. ELF inspection found no required shared libraries. The archive contains only `bootstrap`, no Python, configuration file, keys, CSV, or native library payloads. AWS reported **23 MB peak memory** on every successful native call. Process-only HWM was roughly 5–6.3 MiB; this is not the whole Lambda footprint.

## Native measurements

One training CSV: 1,000 rows × 21 features, fixed 800/200 inner fold, f64 input, 500 trees, 31 leaves, one bag, single-thread native fitting. `histgbm2` is JopeDime's own boosted-tree implementation, not XGBoost and not a complete GP search.

| Configured memory | State | Fit seconds | AWS billed seconds | Estimated USD / 1,000 identical calls* |
|---|---|---:|---:|---:|
| 128 MB | Cold | 24.658 | 25.327 | $0.05336 |
| 128 MB | Warm | 24.321 | 24.864 | $0.05240 |
| 256 MB | Cold | 12.555 | 12.893 | $0.05432 |
| 256 MB | Warm | 12.359 | 12.625 | $0.05320 |
| 512 MB | Cold | 5.999 | 6.215 | $0.05239 |

*Lambda compute + request + one S3 GET, first-tier Ohio list prices, before free tier/discounts/tax. Excludes logs, stored outputs, orchestration, retries and network egress. These are extrapolations from individual calls, not a 1,000-call load test. Cold initialization was 35–42 ms and is already included in billed duration. There is no 512 MB warm sample: one failed setup call consumed that reserved attempt.

**Interpretation:** 512 MB was fastest and narrowly cheapest among these individual measurements, roughly four times faster than 128 MB at essentially the same price. The difference in cost is far too small, and sample count too low, to establish a reliable optimum. CPU entitlement rather than memory capacity dominated this job. Do not extrapolate this result to arbitrary GP programs or big/wide data. If choosing a next-test tier for this exact workload, start at 512 MB and repeat cold/warm measurements; it is not yet a production sizing policy.

RMSE 0.02319893631773873, R² 0.9889014955812375. These scores confirm replay, not patient-group predictive validity. Only the training CSV was used; outer-test data was neither fetched nor authorized to the role. The row-random infrastructure fold is not a clinical/generalization benchmark.

## Verification and retained failure

- Three Rust tests passed at the corrected source: disjoint/complete deterministic fold, corrupted CSV refusal, validation-label poisoning with unchanged model/predictions.
- Env checks accepted the valid file and rejected missing configuration, credential-bearing fields, duplicates, executable syntax and loose permissions.
- All five successful native calls matched local model/prediction hashes. Their fold hash also matched the original XGBoost experiment. Metrics were recalculated independently from recorded predictions and CSV rows.
- The first native call failed during S3 access before fitting because the generic SigV4 settings omitted S3's payload-checksum header. Fix: `PayloadChecksumKind::XAmzSha256`. Original failed response, billing log and source version were retained; there was no silent retry or budget reset.
- Native total: six attempted calls, five successful fits, one retained setup failure. Overall total: 18 calls including the original 12-call Python/XGBoost proof. The original cap is exhausted; the launcher refuses additional calls from this campaign.
- AWS readbacks confirmed temporary function and role absent and no matching log group. No ECR resources were used by Rust. Original Python-prototype ECR repository and related resources were also deleted and absence verified. Existing `jain-infer` functions were untouched.

Native Lambda compute/request estimate, including failed call: **$0.00026418**. Combined Lambda estimate for all 18 calls: **$0.00145890**. This is a pricing estimate from AWS billing logs, not a statement of final account charges.

## Code, configuration and delivery status

All active Lambda source is under `xbabe2:/home/ubuntu/JopeDime/.agent/experiments/lambda-csv/`. This is an ignored experiment directory inside the requested repository. Existing four worktrees and another owner's PR prevent a new tracked lane under the repository contract; no new product branch/PR was opened and nothing was merged. The detailed plan specifies eventual tracked import into `deploy/lambda-csv/` once a lane is released. Archived Python-prototype and first-Rust-attempt source live under `versions/` there.

Installed `.env` has mode 0600 and selects AWS profile, account, region, function name and receipt directory. The parser rejects static-key fields and never sources/evals the file. Lambda uses its scoped execution role's temporary credentials. `.env` is ignored and absent from the ZIP; `.env.example` is the safe template. No account keys were committed, copied into artifacts, or embedded in the package.

Remote evidence: `/home/ubuntu/scratch/jopedime-lambda-csv-20260928/`; successful repaired run in `rust-run-repair/`, initial failure in `rust-run/`. The local delivery retains the same JSON/log receipts, package manifest and tested ZIP. Package source SHA manifest is `rust-source-manifest.json`; native dependency source is pinned to repository HEAD `1238d524961c1e09fb8afe534c514a8758ef8b81`. The Mac copy's Cargo path dependency resolves only after placing the code in its documented repository location.

## Adaptive sizing: implementation still to come

The detailed [test and sizing plan](TEST-AND-SIZING-PLAN.md) covers exact files, ownership, commands, tests, schemas, failure handling, resource limits and handoff gates. Proposed path:

1. Qualify a real fixed GP program through the native evaluator and preserve prediction parity.
2. Record dataset shape/decoded width, algorithm/program complexity, memory tier, thread count, cold/warm state, per-phase clocks, per-job memory peaks, billed duration and all failures.
3. Use conservative historical envelopes first. Select the lowest expected-cost tier with memory and timeout headroom; minimum MB alone is not the objective.
4. Train upper-quantile resource predictors only after enough varied measurements; validate on held-out dataset families and GP structures. Treat OOM/timeout records as censored lower bounds, not zeros.
5. Route through pinned tier versions/aliases. Record sizing-policy version, prediction bounds and actual outcome. Refuse oversized work or send it to a batch/host executor; bound retries and their costs.

Warm process HWM can include earlier jobs. The plan calls for isolated child-process rusage plus runtime overhead for variable GP workloads. Five successful calls on one tiny dataset are insufficient to train a reliable predictor. ARM, general GP programs, large inputs, concurrent cold bursts and production routing remain untested.

## Original XGBoost proof

The historical container packaged Python 3.12, `xgboost-cpu` 3.0.5, NumPy and SciPy. It performed 12 successful calls at 1024/3072/6144 MB. The compressed ECR image was 261,747,044 bytes. Best tested warm Lambda-only cost was about $0.03224 per 1,000 calls at 1024 MB, with 1.927-second median billed time and 176 MB maximum AWS-reported memory.

Do not compare that fit cost directly against native `histgbm2` as a language benchmark: algorithms, defaults, arithmetic precision and execution differ. Its useful result is that versioned S3, fitting, score replay and AWS billing worked; its container is superseded by the native ZIP architecture.

References: [AWS Rust ZIP deployment](https://docs.aws.amazon.com/lambda/latest/dg/rust-package.html), [Lambda limits](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-limits.html), [Lambda pricing](https://aws.amazon.com/lambda/pricing/). Region-specific pricing API response is retained as `pricing.json`.
