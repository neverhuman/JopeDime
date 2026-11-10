# ADR-0007 — Carbon: Regional Lambda Cells, Client Landmarking, Sealed Enterprise Appliance & Spot-Hybrid Trigger

**Status:** accepted 2026-11-10 (owner architecture direction; implementation tracked in
[docs/plans/carbon-lambda-20261110/README.md](../plans/carbon-lambda-20261110/README.md)) ·
**Builds on:** [ADR-0003](ADR-0003-runtime-roots.md),
[ADR-0004](ADR-0004-evaluator-identity-known-table.md),
[ADR-0005](ADR-0005-canonical-learning-loop.md), and
[ADR-0006](ADR-0006-rust-only-stack.md) ·
**Scope:** `crates/jope-prep` (planned), `crates/jope-carbon` (planned), `crates/jope-api`,
`crates/jope-service`, `crates/jope-appliance`, `deploy/carbon/` (planned), `.github/workflows/ci.yml`

## Context

`jope-service` and `jope-api` on `main` provide the v1 product engine, `JMAT v1` binary matrix
storage (`crates/jope-service/src/dataset/jmat.rs`), the pure-Rust Candle CPU `r882` fused
DOPE + JopeDime ranker (`models/bundles/r882.json`, `crates/jope-service/src/ranker/`),
deterministic wave execution (`crates/jope-service/src/discover/loop/executor.rs`), and run
lifecycle controls (`pause`, `continue`, `finish`, `cancel`, and `warm fork`).

To serve anonymous Free, authenticated Pro, and self-hosted Enterprise users globally with low
latency, strict search-IP protection, and near-zero idle infrastructure spend, we evaluated three
competing architecture proposals (`LAMBDA_INFERENCE_CARBON_PLAN`, `fable5p2`, and `gpt6p1sol`)
across four core engineering tensions:

1. **Client Landmarking vs. Weight Redistribution:** `r882`'s `dope.safetensors` carries
   `LicenseRef-Jain-Internal-Model` (`redistribution: internal`), and the 2026-11-06 product rule
   (`crates/jope-appliance/src/cloud/route.rs`) forbids public weight downloads. Meanwhile,
   DOPE's CPU forward pass takes `88.8 µs` on 1 ARM64 vCPU, whereas computing dataset descriptors
   and landmarks (`v256: [f32; 256]` + `columns: [[f32; 17]; 64]` = `6,368 B`) is pure `std` math.
2. **Single-Lambda (`Solo`) vs. Multi-Lambda Wave Fan-Out:** Pure fan-out incurs coordinator
   wait billing and cold hydration overhead on small runs (`fast` / small `medium`), whereas a
   single Lambda caps out at 6 vCPUs (`10,240 MB`) and limits throughput on `high` / `ultra` runs.
3. **Search-IP Protection Across Cloud & Customer-Hosted Enterprise:** Public endpoints, public
   CLIs, evaluator IAM roles, and customer-operated VPC/on-prem disks must never expose internal
   program representations, operator registries, lineage archives, or plaintext `r882` weights.
4. **Pure Serverless Lambda vs. Dedicated EC2 Graviton Spot Hybrid:** Graviton3 Spot (`~$0.0125`
   per provisioned vCPU-hr) is `6.6x` cheaper than Graviton3 Lambda (`$0.0829` per billed vCPU-hr)
   at `100%` utilization, but costs more than Lambda below `15.1%` utilization and introduces
   fixed network burn if NAT Gateways or ALBs are provisioned prematurely across 17 regions.

## Decision

1. **Wrap `jope-service` and `jope-api`; Do Not Fork the Engine.**
   - Extract `crates/jope-prep` (`std` + `serde` + `sha2` + `zstd`, WASM-compatible, zero
     `feat-math`, zero `candle`) for deterministic `FIT`/`HOLDOUT` splitting, `JMAT v1` writing,
     and `FIT`-only `LandmarksV1` extraction (`6,368 B`). `jope-service` re-exports `jope-prep`.
   - Public Free/Pro clients send `LandmarksV1` + compressed `fit.jmat.zst` and `holdout.jmat.zst`
     and download zero model weights. `PrecomputedDopeV1` is reserved strictly for signed internal
     callers.
   - All new workspace dependencies register in **one single `Cargo.lock` transition** in Phase
     `C1` under [ADR-0004](ADR-0004-evaluator-identity-known-table.md)
     (`JOPE_COMPAT_FREEZE_REGENERATE=identity`), with zero edits to `crates/feat-math/src/**` and
     zero edits to `deploy/lambda/**` (`SNAPSHOT.json` preserved).

2. **Two Binaries from `crates/jope-carbon` (`aws-sigv4` + `ureq`, Single `aws_lc_rs` Provider).**
   - `carbon-edge` (`256 MB` ARM64, Function URL): Physically compiled without `jope-service`,
     `jope-runtime`, `feat-math`, or `candle`. Enforces Free PoW/rate quotas + a hard daily spend
     circuit breaker (`$10.00 / UTC day` in `dev` and `prod` beta), verifies Pro P-256 tokens,
     presigns CAS uploads and S3 `Range` progress reads (`$0` Lambda watch billing), and carries
     an explicit IAM `Deny` on `runs/*/internal/*`.
   - `carbon` (`CARBON_ROLE=engine|eval|__worker|serve|emu`): Static `aarch64-unknown-linux-musl`
     and `x86_64-unknown-linux-musl` binary driving coordinator runs, stateless evaluators,
     Enterprise `serve` clusters, and the local-Lambda CI emulator (`carbon emu`).

3. **Adaptive `Solo` vs. `Hybrid Fan-Out` Compute Planner & Five Memory Classes (`xs`..`xl`).**
   - Free runs (`<= 20,000 x 64`, `fast`) and small Pro runs (`< 12 vCPU-s` estimated fit work,
     calibrated empirically by benchmark `T1`) execute in **`Solo` mode** inside a single `engine`
     Lambda (`ProcessExecutor(vCPUs)` — zero coordinator wait double-billing, zero S3 re-fetch).
   - Larger Pro `medium`, `high`, and `ultra` runs execute in **`Hybrid Fan-Out` mode**: a
     `1,024 MB` `engine` evaluates 1 candidate per wave locally while `LambdaExecutor` fans out
     `W - 1` candidates across right-sized `carbon-<stage>-eval-{xs,s,m,l,xl}` Lambdas (`512`,
     `1,024`, `1,769`, `3,008`, `10,240 MB`) with zero S3 IAM permissions (presigned `fit.jmat`
     GET only; `holdout.jmat` stays outside evaluator/ranker closure until finalization).

4. **Neural Generation-0 (`SeededSpace`) & Four Native Run Controls.**
   - `SeededSpace` evaluates 4 structural baselines + `r882.bank.nearest(lm48, 4)` in Wave 0 to
     calibrate noise anchors, then uses `AdmittedRanker::r882` (`FusedForward` + `CpuAdapter`) in
     Wave 1 to admit top candidates across `R` in-flight rounds (qualified via `E1` + `CQ1`).
   - Supports mid-run provisional `POST /v1/predictions` (`provisional_fit_only`), `< 1 s`
     `control finish` (`Executor::revoke` + single HOLDOUT evaluation), `$0`-paused
     `control pause`/`continue` (also used for `< 810 s` multi-epoch continuation fenced by
     conditional `If-None-Match: "*"` `epochs/<n>.claim`), and `warm fork` upgrades.

5. **Six-Wall IP Defense Including Sealed Enterprise Storage.**
   - Public `jopedime` builds disable `native` (`ops/ci/lite-client.sh`).
   - Enterprise `jope-carbon serve` (+ `--role eval --join` VPC worker clustering) embeds `r882`
     encrypted in `Zeroizing` RAM and encrypts all `runs/*/internal/**` disk state with
     `AES-256-GCM-SIV`, sets `PR_SET_DUMPABLE=0`, and disables binary `kind: "model"` export.

6. **17-Region Autonomous Cells & Day-1 `SpotAdvisor` 3-Gate Hybrid Trigger.**
   - Clients select cells via a P-256 signed `jope.carbon-cells/v1` manifest and a 2-sample
     `/v1/ping` latency + headroom race (`dev`: `us-east-2` + `eu-west-1` at `W <= 64`; `prod`:
     4 Wave-1 cells expanding to all 17 enabled AWS regions after lowering idle legacy `896`
     reserved concurrency).
   - Phases `C1`–`C12` ship **100% pure Lambda** with passive `spot_shadow` telemetry. Per-cell
     stateless Graviton Spot overflow (`HybridExecutor` wrapping Phase `C5`'s `serve --role eval`
     with `min=0` ASG, free S3 Gateway Endpoint, zero NAT Gateway, and instant `LambdaExecutor`
     fallback on preemption) is enabled **only** in an individual cell when:
     1. **Gate `S-1`**: Single-cell `eval-*` spend `>= $750/month` (`$25/day` 14-day mean), **and**
     2. **Gate `S-2`**: Simulated `min=0` Spot ASG busy ratio `eta >= 35%` (or `p20 >= 4` vCPUs),
     3. *Or* **Gate `S-3`**: Regional Lambda concurrency exceeds `60%` of quota / `0.5%` throttle.

## Consequences

- Zero duplicate engine maintenance: `jope-service` remains the single source of truth for
  `JMAT v1`, `r882`, determinism, and qualification gates.
- PR CI tests the full multi-Lambda Runtime API, S3 prefix ACLs, memory limits, and preemption
  failover hermetically via `jope-carbon emu` under `target/carbon-emu/` without AWS credentials.
- Idle global footprint across all 17 regions stays near `$0/month` while preserving a ~250-LoC
  per-cell path to `57%–77%` Graviton Spot evaluator savings once sustained regional load arrives.
