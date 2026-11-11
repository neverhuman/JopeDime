# `jope-carbon`

Zero-knowledge serverless cell runtime, `r882` Free/Pro neural-fusion GP orchestrator, adaptive multi-class execution planner, authenticated state sealer, live AWS Lambda/S3 SigV4 transport, CLI runner, 16-vector adversarial red-team harness, Enterprise VPC/on-premises server, and local multi-cell Lambda emulator for the Carbon execution plane ([ADR-0007](../../docs/adr/ADR-0007-carbon-lambda-cells.md), [deployment runbook](../../docs/runbooks/carbon-lambda-deploy.md), [Carbon plan](../../docs/plans/carbon-lambda-20261110/README.md)).

## Package Organization

- **`src/security/`** ([`auth`](src/security/auth.rs), [`user_envelope`](src/security/user_envelope.rs), [`seal`](src/security/seal.rs), [`sigv4`](src/security/sigv4.rs)):
  - `ctk1.<payload>.<hmac>` tenant-scoped capability tokens (`Free`, `Pro`, `Enterprise`) and anonymous Free token issuance (`POST /v1/carbon/auth/anonymous`).
  - Strict zero-knowledge per-user envelope encryption (`CUEN\x01` + `CUDK\x01`): client CSPRNG `U_DEK` wrapped exclusively to `carbon-engine`'s NIST P-256 `ECDH` KEK (`ECDH-ES + HKDF-SHA256 + AES-256-GCM`), slot-scoped HKDF subkeys (`FitJmat`, `HoldoutJmat`, `LandmarkJlmb`, `PredictRows`, `PredictOutput`, `CheckpointSeal`), and ephemeral client response wrapping on `/predict`.
  - `CCKP\x02` domain-separated (`CheckpointArchive` vs `ChampionModel`) and `U_DEK`-double-bound `AES-256-GCM` state sealing.
  - Size-pinned AWS SigV4 `PUT` presigning (`Content-Length` + `x-amz-server-side-encryption: AES256` signed headers).
- **`src/cells/`** ([`edge`](src/cells/edge.rs), [`engine`](src/cells/engine.rs), [`eval`](src/cells/eval.rs), [`engine_invoker`](src/cells/engine_invoker.rs), [`lambda_runtime`](src/cells/lambda_runtime.rs)):
  - **Compile-time IP boundary (`CT3`)**: `carbon-edge` (`--no-default-features --features edge`) compiles strictly against `jope-api`, `jope-prep`, `aws-lc-rs`, `rustls`, and `ureq` with zero dependency path to `jope-service`, `jope-runtime`, `feat-math`, or `candle-core`, and holds zero `CARBON_ENGINE_KEK_HEX`.
  - **`carbon-engine`**: unwraps `CUDK\x01` in `Zeroizing` RAM, strips `wrapped_u_dek_b64` before writing `request.json`, loads the owner-signed `r882` DOPE + JopeDime neural fusion bundle (`/opt/jope-models/r882` Lambda Layer), runs `Wave0SeedOnly` (`Free`) or `MultiWaveLearned` (`Pro` / `Enterprise`), supports async `202 Accepted` execution (`Event`), cooperative `/stop`, cross-container `/resume`, post-selection outer-holdout scoring, E2E encrypted `/predict`, and cryptographic `DELETE /v1/carbon/runs/{id}` shredding.
  - **`carbon-eval`**: stateless candidate shard worker receiving only a 60-second `FitJmat`-only HKDF subkey grant (`Zeroizing` RAM cache `<= 2` entries, `<= 60 s` TTL).
- **`src/storage/`** ([`cas`](src/storage/cas.rs)): `CarbonCasStore` (`CarbonFsCas`, `CarbonHttpCasClient`, `CarbonS3CasClient`) with strict tenant/run/SHA-256 key validation and deletion primitives.
- **`src/modes/`** ([`planner`](src/modes/planner.rs), [`spot_advisor`](src/modes/spot_advisor.rs), [`serve`](src/modes/serve.rs), [`emu`](src/modes/emu.rs)): Solo vs. Hybrid Fan-Out planner, 3-Gate (`S-1`/`S-2`/`S-3`) Spot-Hybrid advisor, Enterprise VPC `carbon serve` (`CarbonProductTier::no_limit()`), and credential-free CI multi-cell emulator `carbon emu`.
- **`src/cli/`** ([`client`](src/cli/client.rs), [`runner`](src/cli/runner.rs), [`redteam`](src/cli/redteam.rs), [`deploy`](src/cli/deploy.rs)): Zero-knowledge `CarbonThinClient` SDK, `carbon package` (cosign `r882` + `CT3` + deterministic PKZIP builder), `carbon deploy` (declarative [`deploy/carbon/cloudformation-cell.json`](../../deploy/carbon/cloudformation-cell.json) ChangeSet + Secrets Manager + `:live` alias cutover), `carbon rollback`, `carbon run`, `carbon bench`, `carbon token-mint`, and the 16-vector (`V-01`..`V-16`) adversarial `carbon redteam` harness.

## Verification & Deployment

See [`docs/runbooks/carbon-lambda-deploy.md`](../../docs/runbooks/carbon-lambda-deploy.md) for full `arm64` Graviton2 cross-compilation, `r882` Lambda Layer packaging, IAM/S3 provisioning, zero-knowledge P-256 KEK setup, `carbon redteam` verification, CLI benchmarking, and Enterprise VPC setup.

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-carbon
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo build --locked -p jope-carbon --bin carbon-edge --no-default-features --features edge
```
