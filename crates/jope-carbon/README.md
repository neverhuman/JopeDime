# `jope-carbon`

Serverless cell runtime, adaptive multi-class execution planner, envelope sealer, and Enterprise/CI server foundation for the Carbon execution plane ([ADR-0007](../../docs/adr/ADR-0007-carbon-lambda-cells.md), [Carbon plan](../../docs/plans/carbon-lambda-20261110/README.md)):

- **Compile-time IP boundary (`edge` vs `engine`/`eval`)**: `carbon-edge` (`--no-default-features --features edge`) compiles strictly against `jope-api`, `jope-prep`, `aws-lc-rs`, `rustls`, and `ureq` with zero dependency path to `jope-service`, `jope-runtime`, `feat-math`, or `candle-core`.
- **Product & effort tier admission** ([`role`](src/role.rs)): enforces `Free` (`<= 5,000` rows, `<= 64` features, `fast`/`medium`), `Pro` (`<= 100,000` rows, `<= 512` features, `fast`..`ultra`), and `Enterprise` (`no_limit()` on rows and features, VPC/on-prem `carbon serve`).
- **Adaptive Solo vs Hybrid Fan-Out planner** ([`planner`](src/planner.rs)): selects Arm64 Graviton2 evaluation memory classes (`512 MB` `XS` through `7,076 MB` `XL`) from FIT working-set footprint and chooses `SoloSingleLambda` vs `HybridFanOut` from explicit wall-clock and GB-second cost crossover rules.
- **CTO Spot-Hybrid 3-Gate advisor** ([`spot_advisor`](src/spot_advisor.rs)): evaluates monthly telemetry windows against spend (`S-1`), load-shape duty-cycle (`S-2`), and burstiness (`S-3`) gates before recommending warm standby or optional Enterprise Spot overflow.
- **Authenticated state sealing** ([`seal`](src/seal.rs)): encrypts and authenticates opaque checkpoint payloads (`AES-256-GCM` + context-bound `HMAC-SHA256` key derivation via `aws-lc-rs`, `Zeroizing` key material) so search state never leaves the trusted boundary in cleartext.

## Verification

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test -p jope-carbon
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo build --locked -p jope-carbon --bin carbon-edge --no-default-features --features edge
```
