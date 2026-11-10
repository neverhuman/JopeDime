# `jope-carbon`

Serverless cell runtime, adaptive multi-class execution planner, authenticated state sealer, Enterprise VPC/on-premises server, and local multi-cell Lambda emulator for the Carbon execution plane ([ADR-0007](../../docs/adr/ADR-0007-carbon-lambda-cells.md), [Carbon plan](../../docs/plans/carbon-lambda-20261110/README.md)):

- **Compile-time IP boundary (`edge` vs `engine`/`eval`)**: `carbon-edge` (`--no-default-features --features edge`) compiles strictly against `jope-api`, `jope-prep`, `aws-lc-rs`, `rustls`, and `ureq` with zero dependency path to `jope-service`, `jope-runtime`, `feat-math`, or `candle-core` (`CT3`).
- **AWS Lambda Custom Runtime (`2018-06-01`) cells** ([`lambda_runtime`](src/lambda_runtime.rs), [`edge_router`](src/edge_router.rs), [`engine_worker`](src/engine_worker.rs), [`eval_worker`](src/eval_worker.rs)): `carbon-edge`, `carbon engine` (Wave-0 conditioning, cooperative early-stop `/stop` + Lambda deadline drain, cross-container `/resume`, separated outer-holdout evaluation, and sealed server-side `/predict`), and stateless `carbon eval` shard workers (`XS` `512 MB` .. `XL` `7,076 MB` with warm `fit.jmat` caching).
- **Enterprise VPC / on-premises single binary (`carbon serve`)** ([`serve`](src/serve.rs)): runs the identical `/v1/carbon/*` contract locally with `CarbonProductTier::no_limit()` and `CarbonFsCas` persistence.
- **Credential-free CI multi-cell emulator (`carbon emu`)** ([`emu`](src/emu.rs)) & **Thin Client SDK** ([`client`](src/client.rs)): boots real `2018-06-01` Lambda Runtime API queues and HTTP CAS across `edge` (including OS-isolated `carbon-edge` subprocess), `engine`, and `HybridFanOut` `eval` workers with multi-region `/v1/carbon/ping` failover.
- **Authenticated state sealing** ([`seal`](src/seal.rs)) & **3-Gate Spot-Hybrid advisor** ([`spot_advisor`](src/spot_advisor.rs)): `AES-256-GCM` (`CCKP` v1) context-bound checkpoint/champion encryption and `S-1`/`S-2`/`S-3` monthly break-even evaluation.

## Verification

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-carbon
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo build --locked -p jope-carbon --bin carbon-edge --no-default-features --features edge
```
