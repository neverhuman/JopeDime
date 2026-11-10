# JopeDime Carbon — Final Synthesized Lambda (+ Spot Hybrid Trigger) Inference, Neural Gen-0 & GP Evolution Plan (v4)

- **Document ID**: `CARBON-FINAL-SPEC-v4`
- **Date**: 2026-11-10
- **Controlling Upstream Baseline**: `origin/main` at `8a7f68d926520b2dc542cf1ee51dbc97e97d92c7` (canonical `$REPO_ROOT` is currently detached at `9483d5854f` with other owners' staged whitepaper work; **all implementation lanes branch from `origin/main`**).
- **Synthesized Sources**:
  1. [`LAMBDA_INFERENCE_CARBON_PLAN.md`](README.md) (v2/v3 — Antigravity CTO adaptive compute, Spot hybrid economics & sealed-capsule analysis)
  2. `carbon_lambda_plan_fable5p2.md` (Claude Fable 5.2 — live `origin/main` `jope-service`/`r882`/`JMAT v1` architecture, 3-role S3 ACLs, zero-SDK Lambda runtime, and `compat_freeze` protocol)
  3. `$JOPE_EVIDENCE_DIR/CARBON-gpt6p1sol-20261110/` (Codex `gpt6p1sol` — operational preflight, 3-shard `ready_for_review` CI gate, worktree/PR-queue discipline, HOLDOUT closure isolation, scoped `jain-lambda-dev-operator` audit, and `$10/UTC day` hard spend circuit breaker)
- **Single-Agent Execution Goal**: Enable **one agent** to prepare a verified clean `.agent/worktrees/carbon` lane and deliver Phases `C0`–`C12` (`PR-0`..`PR-8`) end-to-end through JopeDime's reviewed PR lifecycle without re-inventing existing `origin/main` crates, violating `compat_freeze`, leaking GP IP, or tripping host/CI/AWS traps.

---

## 1. Executive Synthesis: Why v4 Is Strictly Stronger Than Any Single Prior Plan

Each of the three Carbon artifacts solved critical pieces of the problem while harboring at least one major blind spot. Combining their verified strengths — plus Alton's Spot-hybrid & clean-PR operational model — produces a zero-gap specification:

| Architecture Dimension | `LAMBDA_INFERENCE_CARBON_PLAN` (v2) | `carbon_lambda_plan_fable5p2.md` | `CARBON-gpt6p1sol-20261110` | **BEST & FINAL v4 Synthesis Decision** |
| --- | --- | --- | --- | --- |
| **1. Codebase Grounding (`origin/main` `8a7f68d926`)** | Built from detached `9483d5854f`; proposed re-writing `jope-fused`, `JRG2`, and custom API from scratch. | **100% anchored to `origin/main`**: reuses `jope-service`, `jope-api`, `r882` Candle ranker, `JMAT v1`, `trait Executor`, and `jope-appliance`. | Anchored to `origin/main` **plus** verified live GitHub CI (`ci.yml`), 43 worktrees vs 8-lane cap, and open PR queue (`#2184`, `#2188`). | **Reuse `jope-service` + `jope-api` + `r882` + `JMAT v1` + `trait Executor`**; add leaf `crates/jope-prep` + `crates/jope-carbon`, and enforce `gpt6p1sol`'s exact `C0` preflight & CI rules. |
| **2. Client Landmarking vs. Local DOPE Weights** | Proposed shipping DOPE weights to public Free/Pro clients. | **Caught license/rule blocker**: `r882` `dope.safetensors` is `LicenseRef-Jain-Internal-Model` (`redistribution: internal`) and `route.rs:5` states *"Nobody downloads model weights"*; DOPE CPU forward is `88.8 µs` on Lambda, whereas `landmarks.rs` (`v256 + 64×17` = `6.4 KB`) is pure `std` math! | Required client-side deterministic landmarking (`<= 1,024` FIT rows) and HOLDOUT/TEST exclusion from conditioning. | **Default public contract = `jope-prep` Client Landmarking (`v256: [f32; 256]` + `columns: [[f32; 17]; 64]` = `6,368 B`, zero weights shipped, WASM-ready)**; optional `PrecomputedDope` variant reserved strictly for signed internal/SDK callers. |
| **3. Compute Shape & Cost (Alton's CTO Q1)** | Proved single-Lambda multi-vCPU avoids coordinator idle wait & S3 re-read on small/medium workloads. | Designed `256 MB` `carbon-edge` + `engine` + sync `eval-{s,m,l,xl}` wave fan-out (`W=16..128`) + S3 Range-read progress (`$0` watch billing), and **caught `896` reserved concurrency held by legacy `jope-gp-hwm-hybrid-x86-20261001`!** | Added `$100` dev campaign / `$10 per UTC day` hard spend circuit breaker and pre-reservation ledger. | **Adaptive 3-Mode Planner (`Solo` vs `Hybrid Coordinator+Eval` vs `Wide Fan-Out`)**:<br>• **Solo** (1 invocation, `ProcessExecutor(vCPUs)`, `0` idle wait, `0` S3 re-fetch) for all `fast` and small `medium` (`< 12 vCPU-s`).<br>• **Fan-Out** (`LambdaExecutor` across `eval-{xs,s,m,l,xl}` from **`512 MB`** to `10,240 MB` + 1 local worker on `engine` so coordinator CPU never sits idle) for `medium`/`high`/`ultra` (`W=16..128`, capped to `W<=64` until legacy `896` reservation is lowered).<br>• **S3 Range-read progress** (`$0` watch cost). |
| **4. Stop Early ("See It Now") & Restart (Alton's CTO Q2)** | Sub-second mid-wave cancellation + `AES-256-GCM-SIV` KMS-sealed capsules. | Mapped cleanly onto `jope-service`'s **4 native states**: mid-run `POST /v1/predictions`, `control finish` (`effort_reduced`), `control pause`/`continue` (`$0` paused cost), and `warm fork` (`seed_from_parent`). | Required **immutable provisional prediction** mid-run + strict **HOLDOUT/TEST closure isolation** until finalization + durable outbox recovery. | **Deliver all 4 native controls** (`provisional mid-run predict`, sub-second `finish` via `Executor::revoke`, `$0` `pause`/`continue` checkpointing, and `warm fork` upgrade `fast -> high/ultra`) **with KMS/SIV encryption and strict HOLDOUT isolation**. |
| **5. GP IP Protection (Cloud + Enterprise VPC/On-Prem)** | Sealed embedded weights (`jope-assets`) + `AES-256-GCM-SIV` encrypted internal state (`0` plaintext DSL/weights on disk). | Built GP-free `carbon-edge`, `eval` zero-S3-IAM wall, and `carbon serve` + `--role eval` cluster mode, **but left Enterprise `internal/` as plaintext JSON on disk (`OD-6` gap)**. | Required owner-scoped opaque `ModelId` handles, zero `model.jope` export, and offline `>20k`-row qualification. | **Combine all 6 IP walls**: GP-free `carbon-edge` & thin `jopedime` CLI, `eval` zero-S3-IAM, owner-scoped opaque `ModelId` handles, **PLUS close `OD-6`** by sealing embedded `r882` weights (`jope-assets`) and encrypting Enterprise `internal/` store blobs with `AES-256-GCM-SIV`! |
| **6. AWS IAM, Toolchain & CI Governance** | Audited 17 regions, ARM64 Graviton pricing, and KMS Multi-Region keys. | Designed zero-AWS-SDK `aws-sigv4` + ~150-line `ureq` Runtime API client, Zig `aarch64-unknown-linux-musl` build, and **single-shot `Cargo.lock` `compat_freeze` (`ADR-0004`)**. | Audited live `jain-lambda-dev-operator` role (`ListFunctions` gap), 3-shard `ci.yml` (`ready_for_review` PR trigger; non-main `workflow_dispatch` refused!), and `SNAPSHOT.json` freeze on `deploy/lambda/**`. | **Zero-SDK `aws-sigv4`+`ureq` binary, single `Cargo.lock` regeneration in `C1`, all scripts under `deploy/carbon/` (`deploy/lambda/**` untouched), scoped `jain-lambda-dev-operator` policy patch, and exact 3-shard `ready_for_review` PR lifecycle.** |
| **7. Spot vs. Lambda+Spot Hybrid (Alton's CTO Q3)** | Not quantified. | Built `--role eval` HTTP worker protocol (`C5`) reusable on Spot, but did not analyze Spot break-even or hybrid failover. | Focused on Lambda-only budget cap (`$10/day`). | **Ship 100% Pure Lambda Day-1 + passive `SpotAdvisor` telemetry; enable ~250-LoC stateless `HybridExecutor(SpotEvalPool, LambdaOverflow)` per hot cell ONLY when 3 hard triggers fire (`>= $750/mo/cell` + `eta >= 35%` / `p20 >= 4 vCPU` + `0` NAT Gateway overhead)!** |

---

## 2. Verified `origin/main` (`8a7f68d926`) & Host Ground Truth (`G1`–`G18`)

Every design choice in this plan rests on facts verified directly against `origin/main` (`8a7f68d926520b2dc542cf1ee51dbc97e97d92c7`), GitHub Actions, and AWS account `714044927654`:

- **`G1` — Existing v1 Product Engine (`crates/jope-service` & `crates/jope-api`)**: `origin/main` already ships `jope-api` (wire types + OpenAPI [`schemas/jope.api.v1.openapi.json`](../../../schemas/jope.api.v1.openapi.json); decimal strings for `u64`; zero DSL/operator/lineage fields) and `jope-service` (datasets, `JMAT v1`, deterministic split, `discover` wave loop, `predict`, `export`, and HTTP router). Carbon **wraps** `jope-service` rather than forking it.
- **`G2` — Fused DOPE + JopeDime Ranker (`r882`) Already Ported to Pure Rust (Candle CPU)**:
  - Bundle [`models/bundles/r882.json`](../../../models/bundles/r882.json) (`dope-fused-ranker/h40b`, `10.5 MB` total in `s3://veox-jopedime-models-use2` and `models-usw2`): `dope.safetensors` (`1,778,896 B`), `jope.safetensors` (`7,498,904 B`), `bank.jsonl.zst` (`691,829 B`), `featurizer.json` (`2,487 B`), plus 50 golden contexts (`max_abs_diff <= 1e-4`).
  - [`crates/jope-service/src/ranker/cuda/forward.rs`](../../../crates/jope-service/src/ranker/cuda/forward.rs) (`FusedForward::stages(v256, col) -> StageOutput { hidden256, dope629, lm48, lm48_z, tokens, mask }`, `CpuAdapter` — pure `candle_core::Device::Cpu` despite module path) and [`crates/jope-service/src/run/learned.rs`](../../../crates/jope-service/src/run/learned.rs) (`AdmittedRanker::r882`) already run end-to-end in Rust: cold load `67.5 ms`, DOPE stage `88.8 µs`, full candidate slate `1.89 ms` on 1 CPU thread (`31.8 MiB` RSS delta).
- **`G3` — Neural Generation-0 Seeding Gap**: `r882` currently ranks evolutionary candidate slates in rounds $\ge 1$, **not** Round 0 (`"seed round does not invoke learned admission"`, `quality_qualified: false` in `models/bundles/r882.json`). Phase `C3` adds `SeededSpace` (2-wave `bank.nearest(lm48, 4)` + `r882` Gen-0 seeding) and qualifies it via `E1` + `CQ1`.
- **`G4` — Client Landmarking & Weight-Redistribution Rule**:
  - [`crates/jope-service/src/ranker/landmarks.rs`](../../../crates/jope-service/src/ranker/landmarks.rs) (`fit_transform_and_extract_bundle_high`) deterministically subsamples $\le 1,024$ **FIT-split** rows (`seed=42`) and computes `v256: [f32; 256]` + `columns: [[f32; 17]; 64]` = **`6,368` bytes** using `std`-only math (zero Candle, WASM-ready).
  - In `models/bundles/r882.json`, `dope.safetensors` carries `"license_class": "internal", "redistribution": "internal"`, and [`crates/jope-appliance/src/cloud/route.rs`](../../../crates/jope-appliance/src/cloud/route.rs) codifies the 2026-11-06 rule: *"Nobody downloads model weights."* Shipping `jope-prep` landmarking (`6.4 KB` output) to public clients offloads 100% of `O(n·p²)` descriptor/landmarking work to the client while shipping **zero** model weights.
- **`G5` — `JMAT v1` Binary Matrix Format**: [`crates/jope-service/src/dataset/jmat.rs`](../../../crates/jope-service/src/dataset/jmat.rs) already defines JopeDime's `mmap`-ready binary matrix format (`f64` numeric + dictionary-coded categoricals). Client `jope-prep` writes `JMAT v1` (compressed with `zstd -3` on upload) directly — no competing table format needed.
- **`G6` — `trait Executor` & Wave Planner Seam**: [`crates/jope-service/src/discover/loop/executor.rs`](../../../crates/jope-service/src/discover/loop/executor.rs) (`evaluate`, `fitted_model`, `revoke`) and `discover/loop/mod.rs` (`planned_workers <= 1024`, `R = (planned_workers / 4).max(1)` rounds in flight) already support batched wave execution (`Effort::Fast = 24 evals / 10s`, `Medium = 128 / 60s`, `High = 512 / 300s`, `Ultra = 4,096 / 1,800s` in [`crates/jope-service/src/discover/effort.rs`](../../../crates/jope-service/src/discover/effort.rs)).
- **`G7` — Public CLI Currently Leaks GP Engine**: [`crates/jope-appliance/Cargo.toml`](../../../crates/jope-appliance/Cargo.toml) unconditionally links `jope-runtime` (and transitively `feat-math`) even in `cloud-tls` builds. Phase `C7` gates `jope-runtime`/`jope-service` behind feature `native` (off in public builds, verified by `ops/ci/lite-client.sh`).
- **`G8` — `feat-math` & `Cargo.lock` `compat_freeze` (`ADR-0004`)**: [`crates/feat-math/tests/compat_freeze.rs`](../../../crates/feat-math/tests/compat_freeze.rs) hashes `crates/feat-math/src/**` AND workspace `Cargo.lock` (`compat_freeze.sha256`). **Zero changes** are permitted in `crates/feat-math/src/**`. All new Carbon crates/dependencies land in **one single `Cargo.lock` transition in Phase `C1`** using `JOPE_COMPAT_FREEZE_REGENERATE=identity` and an `ADR-0004` addendum.
- **`G9` — Single `rustls` Crypto Provider**: Workspace standardizes on `rustls` `0.23` with `aws_lc_rs` (`ureq = { version = "3", default-features = false, features = ["rustls-no-provider", "gzip"] }`). Never enable `ring` alongside `aws_lc_rs`.
- **`G10` — `deploy/lambda/**` Is Frozen (`SNAPSHOT.json`) & No New Python**: `deploy/lambda/**` is digest-locked by `deploy/lambda/SNAPSHOT.json` (`ops/ci/lambda-snapshot-guard.sh`), and `ops/ci/PYTHON_EXCEPTIONS.md` forbids new `.py` files. All Carbon tooling lives in Rust (`jope-carbon`) and shell scripts under `deploy/carbon/`.
- **`G11` — CI Trigger & 3-Shard Topology**: [`.github/workflows/ci.yml`](../../../.github/workflows/ci.yml) line 122 **refuses manual `workflow_dispatch` on non-`main` branches** (`if [ "${GITHUB_REF}" != "refs/heads/main" ]; then exit 1; fi`) and runs **3** Rust test shards (`shard: [1, 2, 3]`) plus `lite-client`, `observatory-ui`, `hygiene`, `jankurai-audit`, and `aggregate`. On PR branches, full non-draft CI triggers automatically via `pull_request` (`opened`, `synchronize`, `reopened`, `ready_for_review`).
- **`G12` — Governance Inventory Sync**: Adding workspace crates requires synchronized updates to `Cargo.toml`, `deploy/ci/rust-test-scope.sh` (`product` and `deferred`), `deploy/ci/coverage.json`, `deploy/ci/ci-plan.py` + `crates/jope-records/src/ci_plan.rs`, `docs/parity/test-inventory.json`, and `README.md`.
- **`G13` — AWS Opt-In Regions (17 Enabled)**: `us-east-1`, `us-east-2`, `us-west-1`, `us-west-2`, `ca-central-1`, `sa-east-1`, `eu-west-1`, `eu-west-2`, `eu-west-3`, `eu-central-1`, `eu-north-1`, `ap-northeast-1`, `ap-northeast-2`, `ap-northeast-3`, `ap-southeast-1`, `ap-southeast-2`, `ap-south-1`.
- **`G14` — Account Concurrency Trap (`896` Reserved!)**: Account `714044927654` has default `1,000` concurrent executions per region, **896 of which are currently reserved** by legacy `jope-gp-hwm-hybrid-x86-20261001` in 8 primary regions (`us-east-1/2`, `us-west-1/2`, `eu-west-1`, `eu-central-1`, `ap-northeast-1`, `ap-southeast-1`), leaving only **`~100–104` unreserved slots**! `dev` caps wave width at `W <= 64` (zero impact on existing functions); `prod` wave width `W = 128` requires Owner Action `OA-2` (lowering legacy reservation on idle functions + requesting AWS quota `5,000`).
- **`G15` — Scoped Operator Role vs. Admin Profile**: Host has both `veox-jepson-prod` (`AdministratorAccess` in `714044927654`) and scoped role `arn:aws:sts::714044927654:assumed-role/jain-lambda-dev-operator` (`--profile jain-lambda-dev-operator`, currently missing `lambda:ListFunctions` / `carbon-dev-*` permissions). Phase `C9` provisions a least-privilege `carbon-dev-*` policy on `jain-lambda-dev-operator` so routine `dev` operations never require admin credentials.
- **`G16` — Live Worktree & PR Queue Discipline**: Canonical `$REPO_ROOT` currently has `43` worktree blocks registered (above the 8-lane cap) and open PRs (`#2184`, `#2188`). Phase `C0` enforces resolving/reconciling the PR queue and worktree custody before opening the first implementation PR.
- **`G17` — Cost Ledger & Daily Spend Breaker**: Prior Carbon guidance (`gpt6p1sol`) establishes a strict **`$100.00` total `dev` campaign ceiling** and **`$10.00 / UTC day` hard circuit breaker** for `dev` and `prod` beta, tracked in an append-only ledger (`never` reset without explicit owner approval).
- **`G18` — Zero `/tmp` on Host, `target/` Evidence Custody**: Host build/test state never touches `/tmp` (Lambda's own container `/tmp` inside AWS or `target/carbon-emu/<run>/tmp` in local CI is isolated). Zig toolchain installs under `~/.local/share/jope-tools/`.

---

## 3. System Architecture, Two-Binary Split & CTO Compute Topology

### 3.1 Architecture Dataflow (Cloud & Enterprise Identical `v1` Contract)

```
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ PUBLIC CLIENT: `jopedime` CLI (`cloud-tls`, NO `native` GP, NO model weights)│
 │  1. Deterministic FIT/HOLDOUT split (`jope-prep::split`, seed = run_seed)    │
 │  2. Client landmarking on <=1,024 FIT rows -> `ConditioningBundle` (6,368 B) │
 │     (`v256: [f32; 256]` + `columns: [[f32; 17]; 64]`; HOLDOUT excluded!)     │
 │  3. Serialize `fit.jmat.zst` & `holdout.jmat.zst` (`JMAT v1`)                │
 │  4. Smart Cell Router: verify P-256 `cells.json`, 2-probe `/v1/ping` race    │
 └──────────────┬───────────────────────────────────────────────┬───────────────┘
                │ HTTPS Function URL (`AuthType: NONE`)         │ Presigned S3 PUT (upload)
                │ (`carbon-<stage>-edge:live`, 256 MB ARM64)    │ & Presigned Range GET (watch)
                ▼                                               ▼
 ┌──────────────────────────────┐               ┌───────────────────────────────┐
 │ `carbon-edge` (GP-FREE BIN)  │               │ Regional Cell Bucket          │
 │ - Free rate/PoW + $10/d cap  │               │ `jopedime-carbon-<stage>-<r>` │
 │ - Pro P-256 token verify     │               │  ├─ cas/<sha256>.jmat         │
 │ - Presigns CAS upload/watch  │───Async──────▶│  ├─ runs/<id>/public/*        │
 │ - IAM DENIES `*/internal/*`! │  Invoke       │  └─ runs/<id>/internal/* (KMS)│
 └──────────────────────────────┘               └───────▲───────────────▲───────┘
                                                        │               │ Presigned
                                            Full Bucket │               │ `fit.jmat` ONLY
                                            + SSE-KMS   │               │ (ZERO S3 IAM!)
                                                        ▼               │
                                ┌─────────────────────────────────┐     │
                                │ `carbon-<stage>-engine-{free,pro}`    │
                                │ (1,024–3,008 MB ARM64, `r882`)  │     │
                                │ - DOPE stage (88.8 µs) + Jope   │     │
                                │ - Gen-0 `SeededSpace` (`r882`)  │     │
                                │ - Adaptive Planner:             │     │
                                │   * Mode A: SOLO (`ProcessExec`)│     │
                                │   * Mode B: FAN-OUT (`LambdaEx`)├─────┴──────────────┐
                                │ - Epoch handoff (<810s) + finish│  Sync Invoke       │
                                └─────────────────────────────────┘  (up to W=128)     ▼
                                                                      ┌──────────────────────────┐
                                                                      │ `carbon-<stage>-eval-*`  │
                                                                      │ `xs` 512M | `s` 1024M    │
                                                                      │ `m` 1769M | `l` 3008M    │
                                                                      │ `xl` 10240M (6 vCPU)     │
                                                                      │ - `/tmp/cas` mmap cache  │
                                                                      │ - Fits GP expr on FIT    │
                                                                      │ - Returns scalar metrics │
                                                                      └──────────────────────────┘
```

### 3.2 Two Binaries from One Crate (`crates/jope-carbon`) + Leaf `crates/jope-prep`
1. **`crates/jope-prep`** (`std` + `serde` + `sha2` + `zstd` only; WASM-compatible; zero `feat-math`, zero `candle`, zero `jope-runtime`):
   - Houses deterministic split, `JMAT v1` encoder/header validator, and `landmarks::fit_transform_and_extract_bundle_high` (extracted cleanly from `jope-service` and re-exported there so zero existing callers break).
2. **`carbon-edge`** (`cargo build --release -p jope-carbon --bin carbon-edge --no-default-features --features edge`):
   - Deployed to `carbon-<stage>-edge` (`256 MB` ARM64, `provided.al2023`, `timeout = 15 s`, Function URL `AuthType: NONE` on alias `:live`).
   - **Physically contains zero GP DSL, zero `feat-math`, zero `jope-service`, and zero `r882` weights** (proven in CI gate `CT3` via `cargo tree` + symbol scan). Even a total memory-read vulnerability in the public Internet endpoint cannot disclose GP operators or neural weights.
3. **`carbon`** (`cargo build --release -p jope-carbon --bin carbon`, default features `engine,eval,serve,emu`):
   - Single static `aarch64-unknown-linux-musl` (`~28 MB` with sealed `r882` bundle) and `x86_64-unknown-linux-musl` binary dispatching on `CARBON_ROLE` / CLI subcommand:
     - `CARBON_ROLE=engine`: Coordinator Lambda (`Solo` or `Fan-Out` wave loop, `r882` ranker, HOLDOUT finalization, epoch checkpointing, and `predict` worker).
     - `CARBON_ROLE=eval`: Stateless parallel candidate evaluator Lambda (`eval-{xs,s,m,l,xl}`).
     - `jope-carbon __worker`: Subprocess sandbox entry point reused from `jope-service::Executor` so segfaults/OOMs/timeouts (`RLIMIT_AS`) in candidate evaluation never crash the Lambda runtime parent.
     - `jope-carbon serve`: **Enterprise VPC / On-Prem single-binary appliance** (with `--role eval --join <coordinator_url>` horizontal worker clustering).
     - `jope-carbon emu`: **Full local-Lambda CI emulator** (Runtime API + Invoke API + Function URL + SigV4 + S3 ACLs + fault injector).
   - **Zero AWS SDK Bloat**: Uses workspace `aws-sigv4` + `ureq` (`rustls` `aws_lc_rs`) and a ~150-line Lambda Runtime HTTP client against `http://${AWS_LAMBDA_RUNTIME_API}/2018-06-01/runtime/invocation/...`. Cold start stays `< 90 ms` (`edge`) and `< 320 ms` (`engine` including `67.5 ms` `r882` init).

### 3.3 Answering Alton's CTO Question #1: Two Compute Sizes vs. One, GPU Reality & Data-Transfer Math

1. **AWS Lambda Has Zero GPUs — and Our Workload Doesn't Want One**:
   - AWS Lambda (`provided.al2023`) offers `x86_64` and `arm64` Graviton3 CPUs (`128 MB` to `10,240 MB` / 6 vCPUs), **no GPUs**.
   - On 1 ARM64 vCPU, `r882` DOPE forward takes **`88.8 µs`** and scoring an entire 16-candidate JopeDime slate takes **`1.89 ms`** (`< 0.15%` of run wall time!). Over **99.5%** of compute is fitting/scoring GP candidate trees (`feat-math` + Ridge/Logistic/GBDT leaf heads) on tabular `f64` arrays — where Graviton3 NEON SIMD (`20%` cheaper per GB-s than `x86_64`) beats GPU PCIe launch overhead on $\le 1\text{M}$ rows.
2. **Why Pure Fan-Out Wastes Money on Small Runs, While Pure Solo Caps Throughput on Large Runs**:
   - **Pure Fan-Out Penalty on Small Runs (`fast` / small `medium`)**: If a `1,024 MB` `engine` coordinator fans out 24 tiny `5 ms` evaluations to separate `eval` Lambdas, you pay (a) `engine` wall-clock idle wait while children execute (`double-billing`), (b) `24` Lambda request fees + JSON/IPC serialization, and (c) cold-start `/tmp` hydration on evaluators.
   - **Pure Solo Ceiling on Large Runs (`medium` on $\ge 25\text{k}$ rows, `high`, `ultra`)**: One `10,240 MB` Lambda tops out at **6 vCPUs**. Running `ultra` (`4,096` evaluations) on 6 vCPUs takes $12\times$–$20\times$ longer than fanning out across `W = 64..128` parallel `eval` Lambdas (**`64–384` aggregate vCPUs**!), and S3 inside the **same AWS region** is **$0.00/GB** (`15–40 ms` for a `10 MB` `fit.jmat` downloaded **once** per warm `eval` container into `/tmp/cas/<sha256>.jmat` and `mmap`'d across all subsequent waves!).
3. **The Optimal Solution — Carbon's Adaptive 3-Mode Execution Planner (`crates/jope-carbon/src/planner.rs`)**:
   - Let estimated total fit work be $\widehat{W}_{\text{cpu}} = E \cdot \hat{c}(n_{\text{fit}}, p)$, where $E \in \{24, 128, 512, 4096\}$ is effort budget and $\hat{c}(n, p) \approx 2.5 \times 10^{-7} \cdot n \cdot p$ vCPU-seconds:
     - **Mode A — `Solo` (Single Lambda Invocation, Zero Fan-Out, Zero Double-Billing, Zero S3 Re-Read)**:
       - Selected whenever **`tier == Free`**, **`effort == Fast` (`24` evals)**, OR **`\widehat{W}_{\text{cpu}} < 12.0` vCPU-s** (e.g., `medium` `128` evals on $\le 5,000 \times 32$ rows = `~2.0 s` wall time!).
       - Runs entirely inside `engine` (`1,024 MB` Free / `3,008 MB` 2-vCPU Pro) using `ProcessExecutor(vCPUs)`. Total cost for a `fast` run: **`~$0.00004`–`$0.00012`** (`< 0.012 cents`)!
     - **Mode B — `Hybrid Fan-Out` (`LambdaExecutor` + 1 Local Coordinator Worker)**:
       - Selected for Pro `medium` ($\widehat{W}_{\text{cpu}} \ge 12$ vCPU-s), `high` (`512` evals, `W = 32`), and `ultra` (`4,096` evals, `W = 64` in `dev` / `128` in `prod`).
       - `engine-pro` runs at **`1,024 MB`** (~0.58 vCPU, minimal coordinator cost) **and evaluates 1 candidate per wave locally** while dispatching the remaining `W - 1` candidates concurrently to right-sized `carbon-<stage>-eval-{xs,s,m,l,xl}` Lambdas! Coordinator CPU is never wasted idle.
   - **5 Right-Sized Evaluator Memory Classes (`xs` Added for Lowest Cost!)**:
     - Working-set formula: `peak_bytes = 2.6 * fit_jmat_bytes + 96 MiB <= 0.65 * memory_mb`:

| Class | Function | Memory (ARM64) | vCPUs | Dataset Fit Envelope (`fit_jmat_bytes`) | Cost / 100 ms |
| --- | --- | --- | --- | --- | --- |
| **`xs`** | `carbon-<stage>-eval-xs` | **`512 MB`** | 0.29 | $\le 80\text{ MiB}$ (e.g. $\le 2,500 \times 64$ numeric) | `$0.00000067` |
| **`s`** | `carbon-<stage>-eval-s` | **`1,024 MB`** | 0.58 | $\le 210\text{ MiB}$ ($\le 20\text{k} \times 64$) | `$0.00000133` |
| **`m`** | `carbon-<stage>-eval-m` | **`1,769 MB`** | **1.00** | $\le 400\text{ MiB}$ ($\le 100\text{k} \times 128$) | `$0.00000236` |
| **`l`** | `carbon-<stage>-eval-l` | **`3,008 MB`** | **2.00** | $\le 700\text{ MiB}$ ($\le 350\text{k} \times 256$) | `$0.00000401` |
| **`xl`** | `carbon-<stage>-eval-xl` | **`10,240 MB`** | **6.00** | $\le 2.4\text{ GiB}$ (up to $1\text{M} \times 512$) | `$0.00001365` |

   - **Zero-Lambda-Cost Progress Watching**: After each wave, `engine` writes `runs/<id>/public/snapshot.json` and appends `runs/<id>/public/events.ndjson` to S3 (`$0.000005` per PUT). Clients watch progress via presigned HTTP `Range: bytes=<offset>-` reads directly against S3 — **zero Lambda invocations and `$0.00` compute while a user watches a 20-minute `ultra` run!**
   - **Empirical Crossover Gate (`T1`)**: Phase `C9` runs benchmark `T1` comparing `Solo (3,008 MB / 10,240 MB)` vs `Fan-Out (eval-xs/s/m)` across `(rows × effort)` grids and freezes the exact empirical crossover constant in `planner.rs`.

---

## 4. Client Conditioning (`jope-prep`), Neural Gen-0 (`SeededSpace`) & GP Evolution

### 4.1 Client Conditioning Contract (`ConditioningInput` — Default Landmarks + Internal Optional DOPE)
In `crates/jope-api/src/dataset.rs` (additive `#[serde(default)]` field on `DatasetUploadRequest` / `CreateDatasetResponse`):
```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ClientConditioning {
    /// Default for ALL public Free/Pro/WASM clients: zero model weights on client!
    /// Computed by `jope-prep::extract_conditioning` strictly on <=1,024 FIT-split rows.
    LandmarksV1 {
        fit_row_digest: String,        // SHA-256 of FIT row indices + schema (proves HOLDOUT exclusion)
        v256: Vec<f32>,                // len == 256
        columns: Vec<[f32; 17]>,       // len <= 64 (sorted canonical column slots)
    },
    /// Reserved strictly for signed internal/Enterprise-SDK callers that already hold licensed DOPE.
    PrecomputedDopeV1 {
        bundle_digest: String,         // Must match active r882 `dope.safetensors` sha256
        fit_row_digest: String,
        dope629: Vec<f32>,             // len == 629
        hidden256: Vec<f32>,           // len == 256
    },
}
```
- **Validation & Determinism**:
  - `jope-prep` computes `SplitPlan` (`FIT` vs `HOLDOUT`) deterministically from `(row_count, target_hash, split_seed)` **before** landmarking, guaranteeing **zero HOLDOUT/TEST rows ever enter `v256` or `columns`** (`gpt6p1sol` closure invariant!).
  - On `engine` startup for a run, if `LandmarksV1` is present and `fit_row_digest` matches `fit.jmat`'s header digest (and all floats are finite), `engine` skips `landmarks::fit_transform_and_extract_bundle_high` (`O(n·p²)` saved!) and runs `FusedForward::stages(&v256, &col)` in **`88.8 µs`** on CPU to obtain `(hidden256, dope629, lm48, lm48_z, tokens, mask)`. If absent (e.g., raw `curl` upload), `engine` computes `LandmarksV1` from `fit.jmat` as a fallback.

### 4.2 Neural Generation-0 (`SeededSpace`) + Multi-Round Wave Evolution
Currently on `origin/main` (`G3`), Round 0 evaluates only fixed curated templates without consulting `r882` or `bank.jsonl.zst`. Phase `C3` adds [`crates/jope-service/src/discover/loop/native/seeded.rs`](../../../crates/jope-service/src/discover/loop/native) (`SeededSpace<S: SearchSpace>`), activated when `JOPE_RANKER_SEED_GEN0=1` (default in Carbon once qualified by `E1` + `CQ1`):
1. **Stage 1 (`88.8 µs` on `engine`)**: Run `FusedForward::stages(&v256, &col)` $\rightarrow$ `lm48_z` + `tokens`. Query `r882.bank.nearest(&lm48_z, 4)` (`692 KB` compressed bank of historical champion GP programs) and adapt column references to the dataset's top-variance typed columns (`<= 64` window).
2. **Gen-0 Wave 0 (Anchor Calibration — `8` Parallel Evaluations)**:
   - Evaluate 4 curated structural baselines (linear/ridge, tree stump, interaction pair, missingness-aware baseline) + the 4 `bank.nearest` historical champions in parallel on `FIT` CV folds.
   - These measured CV utilities populate the **4 context anchor slots** required by `AdmittedRanker::r882` (`CpuAdapter`), calibrating dataset noise scale before neural ranking!
3. **Gen-0 Wave 1 (Neural `r882` Slate Admission — `4 * R` Admitted Seeds)**:
   - Deterministically generate `16 * R` diverse depth-1..3 grammar candidates (`R = (W / 4).max(1)` rounds in flight).
   - Score each 16-candidate slate with `AdmittedRanker::r882` conditioned on `(tokens, mask, 4 measured Wave-0 anchors)` (`~1.89 ms` per slate on CPU) and admit the top `4 * R` programs for parallel evaluation across the Lambda `eval` fleet!
4. **Generations $1 \dots G$ (`fast` / `medium` / `high` / `ultra`)**:
   - Standard `jope-service` evolutionary rounds (`select_parents` tournament + lexicon crossover/mutation/constant jitter $\rightarrow$ `r882` 8-candidate neural filter admitting top-4 per round $\rightarrow$ `R` rounds batched into a single `W`-wide `Executor` wave):
     - `fast`: `24` evals (`Solo`, `W = vCPUs`, `~0.8–3.5 s`)
     - `medium`: `128` evals (`Solo` if small else `Fan-Out W = 16`, `~3–12 s`)
     - `high`: `512` evals (`Fan-Out W = 32`, `~10–35 s`)
     - `ultra`: `4,096` evals (`Fan-Out W = 64` `dev` / `128` `prod`, multi-epoch if $> 810\text{ s}$)

---

## 5. Answering Alton's CTO Question #2: Stop Early ("See It Now"), Zero-Cost Pause/Resume, Warm Upgrade & HOLDOUT Isolation

`jope-service` already defines `ControlAction::{Pause, Continue, Finish, Cancel}` and `POST /v1/runs/{id}/fork { warm_start: true }`. Carbon wires **all four user workflows** end-to-end across Lambda with sub-second responsiveness and strict HOLDOUT isolation:

| User Intent | API / CLI Action | Exact Carbon Lambda Mechanics | Latency & Billing |
| --- | --- | --- | --- |
| **1. "Let me test predictions right now *without* stopping search"** | `jopedime predict --model <snapshot.best.model_id> --provisional` | Every time a wave discovers a new FIT-CV champion, `engine` atomically writes `internal/models/<model_id>.sealed` (fitted on `FIT` only, labeled `status: "provisional_fit_only"`) and updates `public/snapshot.json`. User can run `predict` immediately while `engine` keeps evolving! | **`~150–400 ms`**; search continues uninterrupted. |
| **2. "Stop early right now & give me the final holdout-verified model"** | `jopedime run control <id> finish` (`POST /v1/runs/{id}/control {"action":"finish"}`) | 1. `carbon-edge` writes `runs/<id>/public/control.json` (`seq`, `action: "finish"`).<br>2. `LambdaExecutor` polls `control.json` every **`500 ms`** during in-flight waves (`Solo` checks atomic flag between sub-batches) and immediately invokes `Executor::revoke` (drops unfinished `ureq` evaluator sockets & kills local `__worker` children; late results discarded!).<br>3. `engine` loads `holdout.jmat` **for the first and only time**, scores the current best candidate on `HOLDOUT`, refits final deployment parameters, seals `internal/models/<final_model_id>.sealed`, and marks run `succeeded` (`effort_reduced: true`). | **`< 1.0 s`** to halt wave + finalization; zero remaining waves billed! |
| **3. "Pause now ($0 cost) and resume the exact run hours/days later"** | `jopedime run control <id> pause` $\rightarrow$ later `continue` | 1. On `pause`, `engine` aborts in-flight evaluators, serializes deterministic search state (`rng`, archive, `r882` anchor history, completed evals, remaining budget) via `Store::export_run` into KMS-encrypted `runs/<id>/internal/checkpoint/<epoch>-<wave>.sealed`, marks `paused`, and **exits the Lambda**.<br>2. While paused: **zero Lambdas running (`$0.00/hr`, S3 checkpoint `< 150 KB` = `$0.000003/mo`)**.<br>3. On `continue`, `carbon-edge` CAS-writes `epochs/<n+1>.claim` and async-invokes `engine`, which `import_run`s the sealed checkpoint and resumes at the exact next wave! | **`$0.00`** while paused; deterministic continuation (`CR3` & `CR4`). |
| **4. "I ran `fast` (or stopped early); now evolve further to `high`/`ultra`"** | `jopedime run fork <id> --effort high --warm-start` | Calls `POST /v1/runs/{id}/fork` (`discover/warm.rs::seed_from_parent`): creates a new child `RunId` seeded with the parent's top archive champions as Wave-0 elites (`monotonic champion guarantee`: child's starting best $\ge$ parent's best) and runs the new effort budget! | Preserves prior run immutable + upgrades quality seamlessly (`CR4`). |

- **Automatic 15-Minute Lambda Ceiling Continuation**: `engine` sets `timeout = 900 s` (`15 min`) and triggers the exact same checkpoint-and-handoff mechanism at **`t = 810 s`** (`90 s` safety margin): writes `checkpoint/<epoch>.sealed`, creates `epochs/<epoch+1>.claim` (`If-None-Match: "*"` conditional S3 PUT preventing duplicate async retries from ever spawning two engines!), self-invokes `engine` asynchronously for `epoch + 1`, and exits `0`. Thus `ultra` runs of 30–60 minutes span multiple Lambda epochs seamlessly (`CR2`).
- **Strict HOLDOUT/TEST Closure Invariant (`gpt6p1sol`)**:
  - `cas/<fit_sha>.jmat` and `cas/<holdout_sha>.jmat` are stored as **separate CAS blobs**.
  - Evaluators (`carbon-<stage>-eval-*`) have **zero S3 IAM permissions**; `LambdaExecutor` passes them a presigned GET URL **solely for `cas/<fit_sha>.jmat`**. It is cryptographically impossible for an evaluator, `r882` ranker, or evolutionary operator to read `holdout.jmat`! Only `engine`'s single post-search finalization step reads `holdout.jmat`.

---

## 6. Six-Wall Defense-in-Depth GP IP Protection (Cloud + Enterprise Hardened)

Our GP grammar (`feat-math` operator set, tree structure, mutation/crossover schedules, archive lineage, and `r882` weights/bank) is JopeDime's crown-jewel IP. Carbon enforces **six independent walls** so no single bug, compromised endpoint, or curious Enterprise admin can extract GP internals:

1. **Wall 1 — Wire Schema Scrubbing (`CT2`)**: `jope-api` structs (`RunSnapshot`, `RunEvent`, `ModelSummary`, `PredictionResponse`, `ProblemDetail`) contain **only** numeric progress counters, fold/holdout scalar metrics, permutation/ablation **input-column** importances, and opaque IDs (`run_...`, `mdl_...`). Any string matching `(pipe`, `feat_`, operator tokens, AST nodes, or `safetensors` paths is structurally unrepresentable and verified by black-box byte scan `CT2`.
2. **Wall 2 — Physically GP-Free Public Edge Binary (`CT3`)**: `carbon-edge` compiles without `jope-service`, `jope-runtime`, or `feat-math`. CI gate `CT3` runs `cargo tree -p jope-carbon --no-default-features --features edge` and `nm -D` / string scans to assert zero GP or Candle symbols exist in the Internet-facing binary.
3. **Wall 3 — 3-Role S3 Prefix Isolation & KMS Encryption (`CT4`)**:
   - `carbon-<stage>-edge-role`: `s3:GetObject`/`PutObject` on `cas/*` and `runs/*/public/*`, plus **explicit IAM `Deny`** on `arn:aws:s3:::jopedime-carbon-*/runs/*/internal/*`!
   - `carbon-<stage>-eval-role`: **CloudWatch Logs ONLY (`AWSLambdaBasicExecutionRole`) — ZERO `s3:*` IAM permissions!** Evaluators read `cas/<fit_sha>.jmat` solely via short-lived (`15 min`) presigned GET URLs minted by `engine`.
   - `carbon-<stage>-engine-role`: Full access to its regional cell bucket + regional KMS key (`SSE-KMS` mandatory on `runs/*/internal/*`).
4. **Wall 4 — Opaque Owner-Scoped `ModelId` Handles & Safe Source Export (`CT5`)**:
   - `POST /v1/predictions` takes an opaque `ModelId` (`mdl_<ulid>`), verifies `owner_hash == caller_owner_hash` (`gpt6p1sol` cross-tenant isolation), loads `runs/<id>/internal/models/<mdl>.sealed` inside `engine`, computes predictions, and returns **predictions only**.
   - Binary `kind: "model"` (`model.jope`) export is **disabled** (`403 ExportKindDisabled`). Pro/Enterprise `POST /v1/models/{id}/exports` supports only self-contained numeric evaluation code (`python` / `c` / `sql`, gated by `jope_service::export::lint_shipped_text` so zero search history, operator registry, or `r882` weights appear).
5. **Wall 5 — GP-Free Public `jopedime` CLI (`CT6`)**: `crates/jope-appliance` disables `native` in release builds (`cloud-tls` + `jope-prep` only). Verified by existing `ops/ci/lite-client.sh` + `deploy/appliance/dependencies-deny.toml`.
6. **Wall 6 — Enterprise `carbon serve` Sealed Weights & Encrypted `internal/` Disk Store (Closes `fable5p2` `OD-6`!)**:
   - Embedded `r882` bundle bytes inside the `carbon` binary are encrypted at build time (`AES-256-GCM-SIV` + split-mask key derivation in `crates/jope-carbon/src/seal.rs`) and decrypted purely into `mlock`'d anonymous RAM (`memfd` / `Zeroizing<Vec<u8>>`), never written to disk and unreadable via `strings` or `safetensors` scanners on the binary!
   - Unlike `fable5p2` (which left `internal/` as plaintext JSON on disk in Enterprise mode), `carbon serve` encrypts **every** `runs/*/internal/**` checkpoint and `.sealed` model blob at rest with `AES-256-GCM-SIV` (`--seal-key-env CARBON_SEAL_KEY` or auto-generated host key with KMS option), sets `PR_SET_DUMPABLE=0`, and disables `/debug` & pprof endpoints. Even root inspection of `--data-dir` on an Enterprise host sees zero plaintext `(pipe X ...)` expressions!

---

## 7. Product Tiers (`Free` / `Pro` / `Enterprise`), Abuse Defense & Cost Ledger

### 7.1 Reconciled Tier Matrix

| Dimension | **Free (Anonymous)** | **Pro (Authenticated)** | **Enterprise (VPC / On-Prem)** |
| --- | --- | --- | --- |
| **Authentication** | Anonymous: device keypair (`~/.config/jopedime/anon.key`) + SHA-256 `/24` IPv4 or `/48` IPv6 bucket + adaptive Hashcash PoW (`18–22 bits` under load) | `Authorization: Bearer <p256-jwt>` minted by `app.veox.ai` (`sub`, `tier: "pro"`, `exp <= 1h`, verified offline by `carbon-edge` using pinned P-256 pubkey) | Customer-deployed `jope-carbon serve` (`--auth-jwks` / `--api-key-hash` / mTLS inside customer VPC or air-gapped datacenter) |
| **Max Dataset Rows × Cols** | **`20,000` rows × `64` cols** (`<= 10 MiB` compressed `JMAT`; aligns with `jope-service` 20k rung-0 cap & `r882` 64-col window) | **`1,000,000` rows × `512` cols** (`<= 250 MiB` compressed `JMAT` via S3 multipart presigned upload) | **Unlimited** (bounded only by customer RAM/disk; offline qualified at `> 20,000` rows in `CR5`) |
| **Allowed Efforts** | `fast` (`24` evals, `10 s` cap) | `fast` (`24`), `medium` (`128`), `high` (`512`), `ultra` (`4,096`) | `fast`, `medium`, `high`, `ultra`, + custom `--max-evaluations` / `--wall-seconds` |
| **Compute Execution Mode** | **Always `Solo`** on `carbon-<stage>-engine-free` (`1,024 MB` for $\le 5\text{k}$ rows, `1,769 MB` for $5\text{k}–20\text{k}$ rows; **zero fan-out amplification**) | **Adaptive `Solo` or `Fan-Out`** (`W = 16` `medium`, `W = 32` `high`, `W = 64/128` `ultra`) across `eval-{xs,s,m,l,xl}` (`512 MB`–`10,240 MB`) | Local `ProcessExecutor(--workers N)` on 1 host **OR** horizontal VPC cluster (`--role eval --join http://coord:8480`) |
| **Lifecycle Controls** | `predict` (`<= 100/day`), early `finish` | Provisional mid-run `predict`, `finish`, `$0` `pause`/`continue`, `warm fork`, numeric code export (`python`/`c`/`sql`) | All Pro controls + `AES-256-GCM-SIV` local disk encryption + S3-compatible (`MinIO`/VPC S3) backend |
| **Rate & Concurrency Caps** | `1` active run / IP-device, `5` runs / hr, `20` runs / UTC day; isolated `engine-free` concurrency pool (`15` per cell) so Free floods **never** starve Pro! | `4` concurrent runs / user, `200` runs / day (`fair-share` per-tenant semaphore in DynamoDB) | Customer-controlled |
| **Artifact Retention** | S3 lifecycle auto-expire **`24 hours`** | S3 lifecycle **`30 days`** (pinned models `90 days`) | Customer-controlled |

### 7.2 Hard Spend Circuit Breakers & Campaign Cost Ledger (`gpt6p1sol` Requirement)
1. **Runtime Global Free-Tier Spend Breaker (DynamoDB Atomic Counter)**:
   - Table `carbon-<stage>-quotas` tracks `PK = "SPEND#<utc-yyyy-mm-dd>"`, incrementing `estimated_microdollars` before every `engine` dispatch.
   - Default hard daily ceilings: **`$10.00 / UTC day` total in `dev`** and **`$10.00 / UTC day` Free-tier pool during `prod` beta** (configurable via `CARBON_FREE_DAILY_CAP_USD`). When tripped, `carbon-edge` returns `429 ProblemDetail { code: "free_capacity_exhausted", retry_after_s }` immediately at the `256 MB` (`< 5 ms`) edge.
2. **Operator Pre-Reservation Ledger (`deploy/carbon/ledger.sh`)**:
   - Append-only JSONL ledger stored outside ephemeral paths (`~/.local/state/jopedime/carbon-spend-ledger.jsonl` + S3 mirror) enforcing the **`$100.00` cumulative `dev` campaign cap** and **`$10.00 / UTC day` cap**.
   - Every live `deploy/carbon/smoke.sh` or benchmark invocation (`T1–T3`, `E1`, `CQ1`) reserves worst-case `(memory_gb * timeout_s * invocations * $0.0000133334)` **before** invoking AWS Lambda and reconciles actual billed `Duration` / `Billed Duration` from CloudWatch report lines afterward. **Never reset prior ledger totals without explicit owner sign-off.**

### 7.3 End-to-End AWS Lambda Unit Economics (`arm64` Graviton3: `$0.0000133334 / GB-s`)

| Workload Scenario | Mode & Sizing | Wall Time | Total Lambda + S3 Cost |
| --- | --- | --- | --- |
| **Free `fast` (`24` evals, `2k × 32`)** | `Solo` (`1,024 MB`, 1 invoke) | `~1.1 s` | **`$0.000020`** (`50,000` runs per `$1`!) |
| **Free `fast` (`24` evals, `20k × 64`)** | `Solo` (`1,769 MB`, 1 invoke) | `~3.8 s` | **`$0.000095`** (`10,500` runs per `$1`) |
| **Pro `medium` (`128` evals, `5k × 32`)** | `Solo` (`3,008 MB` 2-vCPU, 1 invoke) | `~2.4 s` | **`$0.000102`** (saves `3.1×` vs Fan-Out!) |
| **Pro `medium` (`128` evals, `50k × 64`)** | `Fan-Out` (`W=16` × `eval-m` `1,769 MB`) | `~6.5 s` | **`$0.00145`** (`0.15 cents`) |
| **Pro `high` (`512` evals, `100k × 128`)** | `Fan-Out` (`W=32` × `eval-l` `3,008 MB`) | `~22 s` | **`$0.0148`** (`1.5 cents`) |
| **Pro `ultra` (`4,096` evals, `250k × 128`)** | `Fan-Out` (`W=128` × `eval-l` `3,008 MB`) | `~78 s` | **`$0.142`** (`14.2 cents`) |
| **Online `predict` (`10k × 64`, warm)** | `Solo` (`1,024 MB`, 1 invoke) | `~65 ms` | **`$0.0000015`** (`660,000` predicts per `$1`!) |

---

## 8. 17-Region Cell Architecture & Client-Side Smart Router

### 8.1 Self-Contained Cell Design (Zero Cross-Region Data Transfer Fees)
Each AWS region is an autonomous **Carbon Cell** (`cell_id = "aws-<region>"`) containing:
1. Regional bucket `jopedime-carbon-<stage>-<region>` (`BlockPublicAccess=ALL`, `SSE-KMS` on `internal/*`, lifecycle rules for Free `1d` / Pro `30d`, seeded once with `models/r882/*` `10.5 MB`).
2. Regional DynamoDB table `carbon-<stage>-quotas` (`PAY_PER_REQUEST`, TTL enabled — local to each cell, `$0` idle cost).
3. Regional Lambdas (`arm64`, `provided.al2023`, alias `:live`):
   - `carbon-<stage>-edge` (`256 MB`, timeout `15 s`, Function URL `AuthType: NONE`, `InvokeMode: BUFFERED`)
   - `carbon-<stage>-engine-free` (`1,769 MB`, timeout `30 s`, reserved concurrency `15`)
   - `carbon-<stage>-engine-pro` (`1,024 MB` fan-out / `3,008 MB` solo, timeout `900 s`)
   - `carbon-<stage>-eval-{xs,s,m,l,xl}` (`512`, `1,024`, `1,769`, `3,008`, `10,240 MB`, timeout `120 s`)

### 8.2 Phased Rollout Across All 17 Enabled AWS Regions
- **`dev` Stage (2 Cells, Immediate after `C0`–`C8`, Zero Impact on Legacy `896` Reservation)**:
  - `us-east-2` (primary) + `eu-west-1` (cross-continent smart-routing & failover proof), with `W <= 64` fitting comfortably inside the `~100` unreserved concurrency slots.
- **`prod` Wave 1 (4 Global Anchor Cells, Gated on `OA-1..OA-4`)**:
  - `us-east-2`, `us-west-2`, `eu-west-1`, `ap-southeast-1`.
- **`prod` Wave 2 (Remaining 13 Enabled Regions = All 17 Regions Total)**:
  - Americas: `us-east-1`, `us-west-1`, `ca-central-1`, `sa-east-1`
  - Europe: `eu-west-2`, `eu-west-3`, `eu-central-1`, `eu-north-1`
  - Asia-Pacific: `ap-northeast-1`, `ap-northeast-2`, `ap-northeast-3`, `ap-southeast-2`, `ap-south-1`
  - Provisioned idempotently via `deploy/carbon/sync-cell.sh --stage prod --region <r>` (total idle cost across all 17 regions is **`<$0.50/month`** because Lambda, S3 empty buckets, and on-demand DynamoDB cost `$0` at rest!).

### 8.3 Smart Client Router (`crates/jope-appliance/src/cloud/cells.rs`)
1. **Signed Cell Directory (`jope.carbon-cells/v1`)**: Client fetches `cells.json` (with embedded static fallback in binary) and verifies its **P-256 ECDSA signature** against a compile-time public key so DNS/MITM tampering can never redirect uploads to an attacker host (`CN1`).
2. **Health & Headroom Endpoint (`GET /v1/ping`)**: Served by `carbon-edge` in `< 2 ms`, returning `{ cell_id, region, status: "ok"|"drain", eval_headroom, throttle_ewma_bp, ts_ms }`.
3. **2-Round Latency + Capacity Race**:
   - Filter cells by `status == "ok"`, `eval_headroom >= required_workers(effort)`, and optional Enterprise/GDPR residency flag (`--regions eu-*` or `JOPE_REGIONS=us-*`).
   - Probe top-6 geographically nearest candidates concurrently (`timeout = 600 ms`); run a 2nd warm sample on top-3 so TLS/cold-edge jitter is eliminated.
   - Score: `effective_ms = warm_rtt_ms * (1.0 + 2.5 * throttle_rate) + (if headroom < 2*W { 35.0 } else { 0.0 })`.
   - Pin the winning `cell_url` for the entire dataset upload + run lifecycle (`DatasetId` & `RunId` carry cell prefix `use2_...` so subsequent `watch`/`predict`/`fork` calls automatically route to the owning cell!). Cache winner for `6 hours` keyed by local network interface hash.
   - **Pre-Start Automatic Failover**: If `POST /v1/datasets` or `POST /v1/runs` returns `5xx`/`429` (`cell_draining`), client transparently fails over to rank-2 cell before search starts.

---

## 9. Enterprise Downloadable Local Lambda Server (`jope-carbon serve`)

Enterprise customers never send rows or conditioning vectors to our cloud. They download **either** the single static binary (`jope-carbon`, `linux-amd64` / `linux-arm64`) **or** the minimal `FROM scratch` OCI container (`ghcr.io/neverhuman/jopedime-carbon-enterprise:<tag>`) and point the **exact same `jopedime` CLI** at their VPC URL (`JOPE_API_URL=https://jope.internal.corp`):

1. **Single-Node Appliance Mode**:
   ```sh
   CARBON_SEAL_KEY="$(cat /etc/jope/seal.key)" \
   jope-carbon serve \
     --listen 0.0.0.0:8480 \
     --data-dir /var/lib/jopedime \
     --workers 32 \
     --tls-cert /etc/jope/tls.crt --tls-key /etc/jope/tls.key
   ```
   - Uses `ProcessExecutor(32)` + embedded sealed `r882` bundle in RAM + `AES-256-GCM-SIV` encrypted `internal/` directory on disk. Supports **unlimited rows** (qualified offline at `> 20,000` rows in CI gate `CR5` with zero outbound sockets).
2. **Multi-Node VPC Cluster Mode (Horizontal Scale Matching Cloud Lambda!)**:
   - Coordinator node runs `jope-carbon serve --cluster-secret-file /etc/jope/cluster.key --s3-url s3://corp-jope-bucket` (or built-in CAS HTTP server).
   - Auto-scaling Kubernetes / ECS / EC2 worker nodes run `jope-carbon serve --role eval --join https://coord.internal:8480 --workers 16`, pulling waves over HMAC-authenticated HTTP using the exact same `RemoteWaveExecutor` protocol as AWS Lambda!

---

## 10. Complete File-by-File Engineering Blueprint

### 10.1 New & Modified Repository Files (Anchored to `origin/main` `8a7f68d926`)

| Path | Action | Responsibility |
| --- | --- | --- |
| **`crates/jope-prep/Cargo.toml`**, **`src/{lib,split,jmat_lite,landmarks}.rs`** | **NEW** | GP-free, Candle-free, WASM-clean client prep crate: deterministic `FIT`/`HOLDOUT` split, `JMAT v1` writer + `zstd`, and `landmarks::fit_transform_and_extract_bundle_high` (`v256` + `64×17`, `6,368 B`) strictly on `FIT` rows. |
| **`crates/jope-api/src/{dataset,run,ping,cells}.rs`**, **`schemas/jope.api.v1.openapi.json`** | **EDIT** | Add `ClientConditioning` (`LandmarksV1` / `PrecomputedDopeV1`), `GET /v1/ping` (`PingResponse`), `SignedCellManifest` (`jope.carbon-cells/v1`), and `provisional: bool` on `PredictionRequest`. Re-verify via `deploy/ci/check-api-schema.sh`. |
| **`crates/jope-service/src/ranker/landmarks.rs`** | **EDIT** | Re-export `jope_prep::landmarks::*` (eliminating duplicate code while preserving 100% of existing tests). |
| **`crates/jope-service/src/discover/loop/native/seeded.rs`**, **`mod.rs`** | **NEW / EDIT** | Implement `SeededSpace<S>`: 2-wave Gen-0 neural seeding (`Wave 0`: 4 baselines + `bank.nearest(lm48, 4)` anchors; `Wave 1`: `AdmittedRanker::r882` slate admission) gated on `JOPE_RANKER_SEED_GEN0`. |
| **`crates/jope-service/src/service/hooks.rs`**, **`store.rs`** | **NEW / EDIT** | Add `RunObserver` trait (`on_wave_committed(&RunSnapshot, &[RunEvent], Option<&SealedChampion>)`, `poll_external_control() -> Option<ControlAction>`, `should_yield_epoch() -> bool`) + pluggable encrypted blob hook on `Store::{export_run, import_run}` and separate `fit.jmat` / `holdout.jmat` paths. |
| **`crates/jope-carbon/Cargo.toml`**, **`build.rs`** | **NEW** | Defines `--bin carbon-edge` (`features = ["edge"]`, GP-free) and `--bin carbon` (`engine`, `eval`, `serve`, `emu`). `build.rs` encrypts `r882` bundle at build time when `CARBON_EMBED_R882_DIR` is set (graceful dev fallback to `JOPE_RANKER_BUNDLE_DIR`). |
| **`crates/jope-carbon/src/runtime.rs`** | **NEW** | ~150-line custom AWS Lambda Runtime API loop (`GET /runtime/invocation/next`, `POST .../response`, `POST .../error`) over `ureq`. Zero `lambda_runtime` / Tokio bloat. |
| **`crates/jope-carbon/src/aws/{sigv4,s3,lambda,ddb}.rs`** | **NEW** | Minimal `aws-sigv4` + `ureq` (`rustls` `aws_lc_rs`) clients for S3 `GetObject`/`PutObject`/`HeadObject` + `If-None-Match: "*"` conditional PUT + presigning, Lambda sync/async `Invoke`, and DynamoDB `UpdateItem` quota/spend counters. |
| **`crates/jope-carbon/src/seal.rs`** | **NEW** | `AES-256-GCM-SIV` sealing/unsealing for embedded `r882` weights in RAM (`Zeroizing`) and `internal/` checkpoints/models on disk or S3, plus `PR_SET_DUMPABLE=0`. |
| **`crates/jope-carbon/src/planner.rs`**, **`src/spot_advisor.rs`** | **NEW** | CTO Adaptive 3-Mode Planner (`Solo` vs `Hybrid Fan-Out`), memory-class selector (`eval-{xs,s,m,l,xl}`), Day-1 passive `SpotAdvisor` counterfactual spend/duty-cycle telemetry, and ~250-LoC `HybridExecutor(SpotEvalPool, LambdaExecutor)` overflow seam. |
| **`crates/jope-carbon/src/{edge,engine,eval,executor,serve}.rs`** | **NEW** | `carbon-edge` HTTP handler; `engine` wave coordinator, 500 ms `control.json` poller, `<810 s` epoch handoff, and `predict` handler; `eval` `/tmp/cas` `mmap` evaluator; `LambdaExecutor` (`impl jope_service::Executor`); and Enterprise `serve` + `--role eval` cluster mode. |
| **`crates/jope-carbon/src/emu/{mod,runtime_api,s3_mem,iam_acl,faults}.rs`** | **NEW** | Full local-Lambda CI emulator spawning real `carbon-edge` & `carbon` child processes with per-role S3 prefix ACLs (`edge` denied `internal/*`, `eval` denied all IAM S3), `RLIMIT_AS` memory caps, and fault injection (`429`, killed wave children, duplicate async retries, Spot ITN preemption failover). |
| **`crates/jope-appliance/Cargo.toml`**, **`src/cloud/{cells,prep,watch}.rs`** | **EDIT / NEW** | Make `native` (`jope-service`/`jope-runtime`) optional and **disabled** in public release builds; add `jope-prep` client landmarking + `JMAT.zst` upload, P-256 smart cell router, and presigned S3 Range-read watcher. |
| **`deploy/carbon/{build-musl.sh,package.sh,iam-policies.json,sync-cell.sh,ledger.sh,spot-advisor.sh,smoke.sh,sign-cells.sh,Dockerfile.enterprise}`** | **NEW** | Reproducible Zig `aarch64-unknown-linux-musl` builder (`~/.local/share/jope-tools/zig`), deterministic zip packager, 3-role IAM policies + scoped `jain-lambda-dev-operator` policy, idempotent multi-region deployer, `$10/day` / `$100` pre-reservation ledger, Spot break-even evaluator, smoke runner, and `FROM scratch` Enterprise image. (`deploy/lambda/**` stays 100% untouched!) |
| **`docs/plans/carbon-lambda-20261110/README.md`** | **NEW** | Tracked copy of this controlling Carbon v4 specification committed inside the clean `.agent/worktrees/carbon` PR lane. |
| **`Cargo.toml`**, **`Cargo.lock`**, **`crates/feat-math/tests/compat_freeze.sha256`**, **`docs/adr/ADR-0004-*.md`**, **`deploy/ci/{rust-test-scope.sh,coverage.json,ci-plan.py}`**, **`crates/jope-records/src/ci_plan.rs`**, **`docs/parity/test-inventory.json`**, **`README.md`**, **`.github/workflows/ci.yml`** | **EDIT** | Single-shot workspace & governance registration in `C1` + dedicated `carbon-emu` & `lite-client` CI verification in `C8`. |

---

## 11. Local-Lambda CI (`carbon emu`) & Live Cloud Qualification Matrix

### 11.1 Mandatory PR CI Gates (Zero AWS Credentials Required in GitHub Actions)
Every PR runs `jope-carbon emu` inside GitHub Actions (`target/carbon-emu/`, **never `/tmp`**), launching the **compiled `carbon-edge` and `carbon` binaries** as real OS processes speaking the AWS Lambda Runtime HTTP API (`2018-06-01`) and Function URL v2 JSON payloads:

| Gate ID | Automated CI Test / Script | Exact Acceptance Proof |
| --- | --- | --- |
| **`CS1–CS3`** | `cargo test -p jope-prep` | Bit-exact match between `jope-prep::landmarks` and `jope-service` golden fixtures (`max_abs_diff == 0.0`); deterministic `FIT`/`HOLDOUT` split proves **zero** HOLDOUT rows touch `LandmarksV1`; WASM target `cargo check -p jope-prep --target wasm32-unknown-unknown` passes. |
| **`CR1`** | `carbon_emu::e2e_solo_and_fanout` | Full anonymous `Free` (`Solo`) and authenticated `Pro` (`Fan-Out` across `eval-xs`/`eval-s` workers) lifecycle: presigned upload $\rightarrow$ `SeededSpace` Gen-0 $\rightarrow$ multi-wave GP $\rightarrow$ `Range` watch $\rightarrow$ HOLDOUT finalization $\rightarrow$ `POST /v1/predictions`. |
| **`CR2`** | `carbon_emu::epoch_continuation_fencing` | Sets artificial epoch ceiling `t = 3 s` on a `12 s` run: proves checkpoint seal, `If-None-Match: "*"` `epochs/1.claim` fencing (injecting duplicate async invoke proves second engine exits cleanly without double-running!), and exact completion. |
| **`CR3`** | `carbon_emu::early_finish_and_provisional_predict` | Issues mid-run provisional `predict` (`status == "provisional_fit_only"`), then sends `control finish` mid-wave: proves `< 1 s` `Executor::revoke` of in-flight evaluators, single HOLDOUT evaluation, and `effort_reduced: true`. |
| **`CR4`** | `carbon_emu::pause_zero_compute_and_warm_fork` | Sends `pause`, asserts **zero** running worker processes and encrypted checkpoint on S3, sends `continue` to completion, then runs `fork { warm_start: true, effort: "medium" }` and asserts `child.initial_best >= parent.best`. |
| **`CR5`** | `carbon_emu::enterprise_serve_offline_cluster` | Starts `jope-carbon serve` + 2 `--role eval` workers inside unshared loopback namespace (`0` external network routes) on a **`25,000 × 32`** synthetic dataset (`> 20k` rows) and asserts encrypted `internal/` disk blobs contain **zero** plaintext `(pipe` or `.safetensors` headers! |
| **`CM1`** | `carbon_emu::memory_rlimit_classes` | Enforces Linux `RLIMIT_AS` matching `512 / 1,024 / 1,769 MB` on evaluator children across synthetic worst-case tables (`peak_rss <= 0.70 * class_mb`). |
| **`CT2–CT6`** | `ops/ci/carbon-ip-gate.sh` + `ops/ci/lite-client.sh` | **CT2**: Regex/AST scan of every HTTP response, S3 `public/*` object, error body, and OpenAPI schema (`0` DSL/operator tokens). **CT3**: `cargo tree` & symbol audit of `carbon-edge` (`0` `feat-math`/`jope-service`/`candle`). **CT4**: `carbon-edge` attempt to read `runs/*/internal/*` returns `403 AccessDenied`; `eval` attempt to read `holdout.jmat` fails (`403`). **CT5**: Cross-tenant `predict` returns `404`; `kind: "model"` export returns `403`. **CT6**: Public `jopedime` binary passes `lite-client.sh`. |
| **`CN1`** | `jope_appliance::cells::router_tests` | Verifies P-256 signature rejection on tampered `cells.json`, latency + headroom selection, `--regions eu-*` pinning, and automatic pre-start failover on `503`/`drain`. |
| **`CF1`** | `carbon_emu::fault_injection` | Injects `25%` Lambda `429 TooManyRequestsException` + `SIGKILL` on 2 `eval` children mid-wave + simulated Spot `--role eval` preemption fallback to `LambdaExecutor`: proves zero corrupted state and seamless wave completion. |

### 11.2 Live Host-Executed AWS `dev` & `prod` Qualification (`deploy/carbon/smoke.sh`)
Executed from canonical host (since CI containers hold zero AWS credentials per `AGENTS.md`), strictly gated by `deploy/carbon/ledger.sh`:
- **`CX1` (`dev` 2-Cell Live Proof — `us-east-2` & `eu-west-1`, Est. Spend `< $0.35`)**: End-to-end Free `Solo` (`fast`), Pro `Fan-Out` (`medium` `W=16`, `high` `W=32`), mid-run provisional `predict`, early `finish`, `pause`/`continue`, `warm fork`, cross-cell latency routing + forced `drain` failover, and `$10/day` Free quota tripwire test.
- **`T1` (`Solo` vs `Fan-Out` Empirical Cost/Latency Matrix — Est. Spend `< $0.80`)**: Benchmarks `(1k, 5k, 20k, 50k rows) × (fast, medium, high)` across `Solo` (`1,769 / 3,008 / 10,240 MB`) vs `Fan-Out` (`eval-xs/s/m`) and records exact billed GB-s & wall-clock p50/p95 into `docs/product/carbon-benchmarks.json`.
- **`E1` & `CQ1` (Neural Gen-0 + Wave Width Qualification — Est. Spend `< $4.50`)**: Evaluates `SeededSpace` (`r882` Gen-0 + learned rounds) vs heuristic baseline across preregistered benchmark tasks (`E1`: wave width `W=16/32` vs serial baseline; `CQ1`: win-rate & time-to-target lift) before flipping `quality_qualified: true` on `r882`.

---

## 12. Answering Alton's CTO Question #3: Dedicated Graviton Spot vs. Lambda + Spot Hybrid Architecture

### 12.1 Exact Unit Economics: AWS Lambda (`arm64`) vs. EC2 Graviton Spot (`c7g` / `c8g`)
Comparing identical Graviton3/4 ARM64 cores in `us-east-2`:
- **AWS Lambda (`arm64`, `1,769 MB` = `1.0` vCPU)**:
  $$1.72754\text{ GiB} \times 3600\text{ s} \times \$0.0000133334 = \mathbf{\$0.0829 / \text{billed vCPU-hour}} \quad (\mathbf{\$0.00/\text{hr}}\text{ when idle!})$$
- **EC2 Graviton3 Spot (`c7g.4xlarge` = `16` vCPUs, `32 GiB` RAM, diversified across `c7g`/`c6g`/`m7g`)**:
  - On-Demand: `$0.580/hr` (`$0.03625/vCPU-hr`). Typical Spot (`~65%` discount): **`~$0.200/hr` = `$0.0125 / provisioned vCPU-hour`**.
  - **At `100%` busy ($\eta = 1.0$)**: Spot (`$0.0125/vCPU-hr`) is **`6.6×` cheaper** than Lambda (`$0.0829/vCPU-hr`), *plus* all 16 cores share one Linux page-cache copy of `cas/<fit_sha>.jmat` instead of 16 separate container downloads!
  - **At Low / Spiky Utilization ($\eta < 0.16$)**: Effective Spot cost per *useful* vCPU-hour is $\$0.0125 / \eta + C_{\text{fixed}}$. Below **`15.1%` average busy utilization**, keeping warm Spot capacity **costs MORE than pure Lambda**, before counting a single minute of engineering/ops overhead — and adding a single AWS NAT Gateway (`$32.85/mo + $0.045/GB`) or ALB (`$16.40/mo`) in 17 regions would burn **`$830+/month` fixed** while Lambda idles at **`$0.00/month`**!

| Utilization ($\eta$) of Provisioned Spot Pool | Effective Spot Cost / Useful vCPU-hr | Lambda Cost / Billed vCPU-hr | Winner & Net Compute Savings |
| --- | --- | --- | --- |
| **$\eta = 5\%$** (spiky early product / off-peak regions) | `$0.2500` | **`$0.0829`** | **Pure Lambda (`3.0×` cheaper than Spot!)** |
| **$\eta = 15.1\%$** (raw compute break-even point) | `$0.0828` | `$0.0829` | **Tie on raw compute $\rightarrow$ Lambda wins (zero ops)** |
| **$\eta = 35\%$** (ASG `0 <-> N` buffered by Lambda) | **`$0.0357`** | `$0.0829` | **Lambda + Spot Hybrid (`57%` cheaper)** |
| **$\eta = 65\%$** (sustained regional daytime/global load) | **`$0.0192`** | `$0.0829` | **Lambda + Spot Hybrid (`77%` cheaper, `4.3×`!)** |

### 12.2 Why Lambda + Spot Hybrid Is *Low* Complexity in Carbon (~250 LoC, Zero Stateful Spot Risk)
A hybrid architecture is **only** complex if you put stateful services, TLS ingress, or database/checkpoint coordination onto preemptible Spot nodes. **Carbon never does that**:
1. **`carbon-edge` (`256 MB`) and `carbon-engine` (stateful GP coordinator + `r882` ranker) ALWAYS stay 100% on Lambda!**
   - Edge + coordinator account for **`< 6%`** of total GB-seconds on `high`/`ultra` runs (`94%+` of compute is inside stateless `eval` workers!). Keeping `engine` on Lambda preserves `$0` idle cost, managed multi-AZ HA, and S3 `<810 s` epoch checkpointing.
2. **Spot Nodes Run the *Exact Same* Stateless Binary We Already Build in Phase `C5` (`jope-carbon serve --role eval`)**:
   - In Phase `C5` (Enterprise VPC cluster mode), we already implement `RemoteWaveExecutor` talking HMAC-authenticated HTTP to `jope-carbon serve --role eval` workers that fetch `cas/<fit_sha>.jmat` via presigned S3 GET URLs (**zero IAM credentials on Spot nodes!**).
   - Wrapping `RemoteWaveExecutor` + `LambdaExecutor` into `HybridExecutor` takes **~250 lines of Rust**:
     - **Instant Cold-Burst Masking ("Lambda Hides the 60s Spot Boot")**: Spot ASG runs at **`min = 0`** when quiet (`$0.00/hr` idle!). When a burst of `ultra` runs arrives and `SpotPool` has 0 warm slots, `HybridExecutor` immediately dispatches Waves 0–3 to **Lambda `eval-*`** (`< 100 ms` start, zero user wait!) while writing a capacity hint that scales the Spot Fleet `0 -> K`. ~55 seconds later, warm `c7g.4xlarge` Spot workers register in DynamoDB `carbon-<stage>-workers` (`TTL = 30 s` heartbeat) and absorb `85–95%` of waves at `$0.0125/vCPU-hr`!
     - **Zero-Impact Spot Preemption**: Spot workers poll EC2 IMDSv2 `/latest/meta-data/spot/instance-action` every `5 s`. On a 2-minute interruption notice, the worker flips `draining = true` (finishing in-flight `50 ms–3 s` candidate fits cleanly). Even on an instant hard kill (`TCP reset` / `timeout`), `HybridExecutor` retries only the unfinished candidate IDs on **`LambdaExecutor` within the same wave** — **zero failed runs, zero lost generations!**
     - **Zero NAT Gateway Cost Trap**: Spot workers deploy in an isolated VPC using a **free S3 + DynamoDB VPC Gateway Endpoint (`$0.00/hr`, `$0.00/GB`)** and IPv6/Security-Group-locked HMAC ingress (no NAT Gateway, no ALB!).

### 12.3 Day-1 `SpotAdvisor` Telemetry & The 3 Hard Quantitative Triggers (`deploy/carbon/spot-advisor.sh`)
We **do not deploy any Spot infrastructure in Phases `C1`–`C12`**. Instead, `carbon-engine` emits a 4-field `spot_shadow` struct into every run completion log (`cell_id`, `eval_vcpu_sec`, `lambda_eval_micros_usd`, `wave_concurrency_peak`), and `deploy/carbon/spot-advisor.sh` evaluates **per regional cell** whether all three gates are met:
1. **Trigger Gate `S-1` (Dollar ROI Floor)**: Single-cell `eval-*` Lambda spend **`>= $750 / month`** (`>= $25 / UTC day` trailing 14-day average, ~300 Lambda vCPU-hrs/day $\approx$ `$500+/mo` net Spot savings in that single cell). Below `$25/day`, engineering attention is worth far more than Spot savings.
2. **Trigger Gate `S-2` (Sustained Duty-Cycle Floor)**: Trailing 7-day simulated `min=0` Spot ASG (`60 s` boot + `10 min` scale-in cooldown) achieves **`eta >= 35%` busy utilization** (or rolling `p20` concurrent demand `>= 4` vCPUs for `>= 8 hrs/day`).
3. **Trigger Gate `S-3` (Early Override — Quota / Throttle Pressure)**: Regional Lambda concurrency utilization exceeds **`60%` of regional quota** or `throttle_ewma > 50 bp` (`0.5%`) for `> 15 min/day` while awaiting an AWS quota raise (Spot `c7g` instances bypass Lambda concurrency limits completely).
- **Per-Cell Granularity**: When `us-east-2` crosses `S-1 + S-2`, **only `us-east-2`** flips `CARBON_SPOT_POOL_ENABLED=1`; the other 16 lower-volume cells remain **100% pure serverless Lambda (`$0` idle)**!

---

## 13. Clean Worktree Reclamation & `.agent/worktrees/carbon` Preparation Runbook

### 13.1 Verified Live Host & Queue Audit (Recorded 2026-11-10T18:07Z)
A direct audit of `$REPO_ROOT` (`origin/main` = `8a7f68d926520b2dc542cf1ee51dbc97e97d92c7`) established:
1. **Canonical `$REPO_ROOT` Must Not Be Mutated**: Detached at `9483d5854f` with 62 staged `docs/whitepaper/*` files and dirty `MASTER_CLEAN_UP.md` + `dope_freeze_guard.py` owned by other sessions.
2. **`.claude/worktrees` Symlink Verified**: `$REPO_ROOT/.claude/worktrees -> ../.agent/worktrees` resolves cleanly to `$REPO_ROOT/.agent/worktrees`.
3. **Open PR Queue (`2` PRs Finishing CI)**:
   - `PR #2188` (`fix/jankurai-8a7f68d9` @ `b7ded34e39`, `grok-4.7`) — `hygiene` & `jankurai-audit` running.
   - `PR #2184` (`feat/comparators-run-stage` @ `7618d26542`) — all 3 Rust shards, `hygiene`, `jankurai-audit`, `typescript`, `macos`, `windows` GREEN; `go-parity` finishing.
   - Per `AGENTS.md`, wait for `#2188` and `#2184` to land (or explicit owner override) before opening Carbon's first active PR so the repo maintains one active implementation PR at a time.
4. **Exact Classification of the `39` `.agent/worktrees/*` Lanes**:
   - **`26` Dirty or Locked Lanes (`DO NOT TOUCH`)**: `bat-02`, `claude-resfit-2071`, `codex-value-plan-20261102`, `elo-current-20261109`, `h225-dsc01`, `h323-opus`, `inputs-t0545`, `jk-cap-*` (14 lanes), `jk-evidence-excl`, `jrb-01-learner`, `pub-02`, `qlt-02-t0545`, `rnk-11`, `svc-08`.
   - **`4` Active / Unmerged Clean Lanes (`DO NOT TOUCH`)**: `jankurai-8a7f68d9` (open PR `#2188`), `audit-main-20261110` (`b2f51bc0d5` > `#2173`), `jankurai-main-20261110` (`415dde34ca` > `#2179`), `jk-cap-falsegreen` (`4edb38890a`).
   - **`9` Exact-SHA Squash-Merged / Closed Clean Unlocked Lanes (`SAFE TO PRESERVE & RECLAIM / REUSE`)**:
     - `bloat-next-20261110` (`58b1e699f6` == **Merged PR `#2181`**, clean, unlocked, **has warm Cargo `target/`** $\rightarrow$ **REUSE IN-PLACE as `.agent/worktrees/carbon`!**)
     - `jankurai-1db31b33` (`81871381cf` == Merged PR `#2187`), `jankurai-9a4bc4f8` (`c3e987ccd0` == Merged PR `#2186`), `jankurai-2b868999` (`3c0a5a18ff` == Merged PR `#2185`), `jankurai-e6e79d81` (`f2e825ce57` == Merged PR `#2183`), `jankurai-f6460a4e` (`44be8c43df` == Merged PR `#2182`), `bloat-trim-20261110` (`b26cfaa4e3` == Merged PR `#2178`), `ci-mirror-2175` (`bb9ade2a60` == Closed PR `#2175`), `waves-to-main` (`61eb119e2a` == Merged PR `#2168`).

### 13.2 Exact Step-by-Step Commands to Prepare Clean `.agent/worktrees/carbon` (Zero Net New Lanes, Zero `/tmp`, Zero Lost Commits)

```sh
# 1. Verify symlink & create custody receipt directory (NEVER /tmp)
test -L $REPO_ROOT/.claude/worktrees
test "$(readlink -f $REPO_ROOT/.claude/worktrees)" = "$REPO_ROOT/.agent/worktrees"
CUSTODY="$JOPE_EVIDENCE_DIR/carbon-worktree-prep-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$CUSTODY"

# 2. Bundle all 9 closed-PR branches BEFORE moving/removing a single lane, then restore-verify in --no-local --bare
git -C $REPO_ROOT bundle create "$CUSTODY/closed-prs-9.bundle" \
  chore/bloat-next-20261110 fix/jankurai-1db31b33 fix/jankurai-9a4bc4f8 \
  fix/jankurai-2b868999 fix/jankurai-e6e79d81 fix/jankurai-f6460a4e \
  chore/bloat-trim-20261110 ci/browser-deps-mirror-fallback research/flywheel-waves-to-main
git clone --no-local --bare "$CUSTODY/closed-prs-9.bundle" "$CUSTODY/verified.git"
for sha in 58b1e699f69f09eb6ae8aeae1d85531a20767e1b 81871381cf48057a4cefb445580dfcc3c0c6ad32 \
           c3e987ccd08eba2195242610bfd8716c2a97a671 3c0a5a18ff345c6bc9d2896f72bf3181bb035218 \
           f2e825ce57bba591104cba095438783d7b84d513 44be8c43dfb5906b8be7980917a56d68a39e7d99 \
           b26cfaa4e3bfd22e0dc8502e5fb7a312ade23ac6 bb9ade2a6096178dafdd93b6f280da525f633760 \
           61eb119e2a00f7ddc771a92d8c28d4dd8112a31a; do
  git -C "$CUSTODY/verified.git" cat-file -e "${sha}^{commit}"
done
sha256sum "$CUSTODY/closed-prs-9.bundle" > "$CUSTODY/SHA256SUMS"

# 3. Verify zero live PIDs / zero unique target/health evidence, convert bloat-next-20261110 -> carbon in-place!
#    (Reuses its existing Cargo target/ directory -> 0 new build-dir sprawl, 15+ min faster first compile!)
git -C $REPO_ROOT fetch origin main
git -C $REPO_ROOT worktree move \
  $REPO_ROOT/.agent/worktrees/bloat-next-20261110 \
  $REPO_ROOT/.agent/worktrees/carbon
git -C $REPO_ROOT/.agent/worktrees/carbon checkout -B feat/carbon-c0-architecture-plan origin/main

# 4. Retire the remaining 8 verified closed/merged clean lanes (preserving any target/health/* first if present)
#    and log reclamation in AGENT_CHAT.md + synchronized MASTER_CLEAN_UP.md under flock!
for wt in jankurai-1db31b33 ci-mirror-2175 jankurai-9a4bc4f8 jankurai-2b868999 \
          jankurai-e6e79d81 jankurai-f6460a4e bloat-trim-20261110 waves-to-main; do
  test ! -d "$REPO_ROOT/.agent/worktrees/$wt/target/health" && \
    git -C $REPO_ROOT worktree remove "$REPO_ROOT/.agent/worktrees/$wt"
done
```

---

## 14. Complete PR-by-PR Completion & Guarded Merge Playbook (`PR-0`..`PR-8` + Cloud `C9`–`C12`)

Every PR reuses the **single** `$REPO_ROOT/.agent/worktrees/carbon` checkout (`CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only`, `CARGO_TARGET_DIR` unset) and follows `AGENTS.md`'s exact reviewed lifecycle:

### 14.1 PR Stack Breakdown (Focused, Self-Contained, Independently Revertible)

| PR | Branch Name (reusing `.agent/worktrees/carbon`) | Phases Covered | Acceptance Commands & Exact Gates |
| --- | --- | --- | --- |
| **`PR-0`** | `docs/carbon-lambda-architecture-v4` | **Clean Plan & ADR Landing**: Commit `docs/plans/carbon-lambda-20261110/README.md` (this v4 spec) + `docs/adr/ADR-0007-carbon-lambda-cells.md` so the architecture contract is tracked on `main` (and remove untracked scratch copy once merged). | `git diff --check`, link check, PR `hygiene` + `jankurai-audit` + `aggregate` green. |
| **`PR-1`** | `feat/carbon-c1-prep-and-scaffold` | **Phase `C1` (`jope-prep` + `jope-carbon` Scaffold + Single `Cargo.lock` Update)**: Extract `jope-prep` (`split`, `jmat_lite`, `landmarks`), re-export in `jope-service`, add `jope-carbon` binary targets (`carbon-edge`, `carbon`), run `JOPE_COMPAT_FREEZE_REGENERATE=identity` **once** with `ADR-0004` addendum, and sync `rust-test-scope.sh`, `coverage.json`, `ci-plan.py`, `ci_plan.rs`, `test-inventory.json`, `README.md`. | `cargo test -p jope-prep`, `cargo check -p jope-prep --target wasm32-unknown-unknown`, `cargo test -p feat-math --test compat_freeze`, `deploy/ci/ci-plan.py` check (`CS1–CS3`). |
| **`PR-2`** | `feat/carbon-c2-c3-api-and-seeded-gen0` | **Phases `C2` + `C3` (`jope-api` v1 + `SeededSpace` Gen-0 + HOLDOUT Split)**: Add `ClientConditioning`, `/v1/ping`, `SignedCellManifest`, provisional `predict`, `SeededSpace<S>` (Wave 0 `bank.nearest(lm48, 4)` anchors + Wave 1 `r882` admission), `RunObserver` hooks, and separated `fit.jmat`/`holdout.jmat`. | `deploy/ci/check-api-schema.sh`, `cargo test -p jope-api -p jope-service` (deterministic replay + zero-HOLDOUT-closure tests + `CT2` schema lint). |
| **`PR-3`** | `feat/carbon-c4-lambda-roles-and-planner` | **Phase `C4` (`carbon-edge`, `engine`, `eval`, `planner`, `seal`, `spot_advisor`)**: ~150-LoC Runtime API client, SigV4 S3/Lambda/DDB clients, `AES-256-GCM-SIV` sealing, Adaptive `Solo`/`Hybrid Fan-Out` planner (`xs..xl`), `<810 s` `epochs/<n>.claim` handoff, `<1 s` `finish`/`pause`, owner-scoped `predict`, and passive `SpotAdvisor` + `HybridExecutor` seam. | `cargo test -p jope-carbon`, `CT3` symbol/dependency scan confirming `carbon-edge` has zero `feat-math`/`jope-service`/`candle`. |
| **`PR-4`** | `feat/carbon-c5-c6-enterprise-and-emu` | **Phases `C5` + `C6` (Enterprise `serve` Cluster + Full `carbon emu` CI)**: `jope-carbon serve` + `--role eval --join` VPC worker mode + `AES-256-GCM-SIV` disk store (`Dockerfile.enterprise`), plus full `jope-carbon emu` local-Lambda harness (Runtime API, Function URL, per-role S3 ACLs, `RLIMIT_AS`, 429/kill/Spot-preemption fault injection). | `CR1–CR5`, `CM1`, `CT2–CT5`, `CF1` (`> 20k`-row air-gapped Enterprise cluster + disk ciphertext verification). |
| **`PR-5`** | `feat/carbon-c7-c8-thin-cli-and-ci` | **Phases `C7` + `C8` (GP-Free `jopedime` CLI, Smart Router & Musl CI)**: Gate `native` off in public `jope-appliance`, wire `jope-prep` + P-256 2-probe cell router + S3 Range-read watcher, add `deploy/carbon/{build-musl.sh,package.sh,sync-cell.sh,ledger.sh,spot-advisor.sh,smoke.sh}`, and wire `carbon-emu` + `carbon-ip-gate` into `.github/workflows/ci.yml`. | `ops/ci/lite-client.sh` (`CT6`), `CN1` router tests, reproducible `aarch64-unknown-linux-musl` zip build, and **all 3 Rust CI shards + `aggregate` green**. |
| **`PR-6`..`8`** | `feat/carbon-c9-c12-cloud-qualification` | **Phases `C9`–`C12` (Gated on `OA-1`..`OA-3`)**: Patch `jain-lambda-dev-operator`, deploy `dev` (`us-east-2` + `eu-west-1`, `W <= 64`) under `$10/day` ledger, commit empirical `T1` `Solo`/`Fan-Out` threshold (`PR-6`), run `E1` + `CQ1` Gen-0 qualification (`PR-7`), and roll out `prod` Wave 1 (`4` cells) $\rightarrow$ Wave 2 (`17` regions) + signed `cells.json` (`PR-8`). | Live receipts under `$JOPE_EVIDENCE_DIR/carbon-c9..c12/`, ledger reconciliation (`<= $10/UTC day`), `docs/learning.md` & `capability-matrix.md` updates. |

### 14.2 Exact Per-PR Verification, Review & Guarded Merge Checklist (`AGENTS.md` Compliant)
For **every** PR (`PR-0` through `PR-8`) inside `$REPO_ROOT/.agent/worktrees/carbon`:
1. **Claim & Pre-Push Check**:
   - Claim `.agent/worktrees/carbon` + branch + `HEAD` under `flock` on `$REPO_ROOT/AGENT_CHAT.md.lock`.
   - Run `git diff --check`, targeted `CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test ...`, and verify `deploy/lambda/SNAPSHOT.json` (`ops/ci/lambda-snapshot-guard.sh`) and `compat_freeze` pass locally.
2. **Push & Trigger Full 3-Shard PR CI (Remember `G11`: Never `workflow_dispatch` Non-`main`!)**:
   ```sh
   $HOME/.local/bin/gh-role writer api user --jq .login   # verify jepsontaylor
   git -C $REPO_ROOT/.agent/worktrees/carbon push -u origin HEAD
   $HOME/.local/bin/gh-role writer pr create --repo neverhuman/JopeDime \
     --base main --head "$(git -C $REPO_ROOT/.agent/worktrees/carbon branch --show-current)" \
     --title "..." --body "..."
   ```
   - Opening the non-draft PR automatically triggers `.github/workflows/ci.yml` across **all 3 Rust shards (`rust-test (1..3)`), `rust-check`, `lite-client`, `hygiene`, `jankurai-audit`, and `aggregate`** at the exact frozen `FULL_SHA`.
3. **Substantive Same-Agent Exact-SHA Review (`2026-10-21` Owner Ruling)**:
   - Freeze `FULL_SHA="$(git -C $REPO_ROOT/.agent/worktrees/carbon rev-parse HEAD)"`, inspect full `gh-role writer pr diff`, verify zero skipped/cancelled required checks, and post a substantive head-bound `COMMENT` review (`commit_id: $FULL_SHA`, honestly labeled same-agent review; security review included whenever workflows, IAM, or `Cargo.lock` change).
4. **Guarded Merge & Next-PR Reset**:
   ```sh
   $HOME/.local/bin/gh-role writer pr merge "$PR_NUM" --repo neverhuman/JopeDime \
     --squash --match-head-commit "$FULL_SHA"
   # Verify merged_at + main-guard pass on resulting origin/main, then advance .agent/worktrees/carbon:
   git -C $REPO_ROOT fetch origin main
   git -C $REPO_ROOT/.agent/worktrees/carbon checkout -B <next-pr-branch> origin/main
   ```
   - Log stopped `HEAD`, CI run ID, merged commit SHA, and receipt path in `AGENT_CHAT.md` under `flock` after each PR lands.
