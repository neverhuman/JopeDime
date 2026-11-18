# Carbon Integration Guide for `app.veox.ai` & `jAIn` (Auth, Output Schema, Envelope Crate & AWS Capacity)

Operational and integration contract between Carbon (`crates/jope-carbon`, `crates/jope-envelope`) and `app.veox.ai` / `jAIn` ([ADR-0007](../adr/ADR-0007-carbon-lambda-cells.md)).

---

## 1. Production Anonymous Gate (`CARBON_ALLOW_PUBLIC_FREE_MINT`)

- **`prod` (`EnvironmentName = "prod"`)**: `deploy/carbon/cloudformation-cell.json` sets `CARBON_ALLOW_PUBLIC_FREE_MINT=false` on `EdgeFunction` (`AllowPublicFreeMint = "auto"` -> `"false"` when `IsProd`).
  - `POST /v1/carbon/auth/anonymous` returns `HTTP 403 Forbidden` (`{"error":"anonymous_disabled", ...}`).
  - `POST /v1/carbon/auth/register` returns `HTTP 403 Forbidden` (`{"error":"registration_disabled", ...}`).
  - Every `prod` upload, run, status, prediction, explanation, export, and shred call requires a valid `ctk2.*` (asymmetric `app.veox.ai`-minted) or `ctk1.*` (operator/service) capability token.
- **`dev` (`EnvironmentName = "dev"`)**: `CARBON_ALLOW_PUBLIC_FREE_MINT=true` remains open for developer smoke tests and open CLI testing.

---

## 2. Credentials for `app.veox.ai`: Asymmetric `ctk2.*` (Target) & `ctk1.*` (`veox-app` Immediate Bridge)

### Path A (Recommended, Zero Shared HMAC Secret): Asymmetric P-256 `ctk2.*` Tokens (`ADR-0007`)

`app.veox.ai` holds an ECDSA NIST P-256 (`ES256`) private key and mints short-lived per-user tokens locally; `carbon-edge` pins only the uncompressed SEC1 public key(s) (`0x04 || X || Y`, 65 bytes Base64) in `CARBON_APP_AUTH_PUBKEYS` (or `app_auth_pubkeys_b64` in `jopedime/carbon/prod/edge-secrets`). `app.veox.ai` never holds `auth_secret_hex`.

1. **Generate a P-256 signing keypair (`0600` PKCS#8 PEM + Base64 SEC1 public key)**:
   ```sh
   carbon token-keygen --out ./veox-app-signing-active.pem --kid veox-2026-11
   ```
   The JSON stdout receipt prints `public_key_sec1_b64` (never the private scalar).
2. **Pin the public key on `carbon-edge` (supports comma-separated `[active, previous]` zero-downtime rotation)**:
   - Either set CloudFormation parameter `AppAuthPubkeysB64="<active_pub_b64>,<previous_pub_b64>"` on `jopedime-carbon-prod`, or store `"app_auth_pubkeys_b64": "<active_pub_b64>,<previous_pub_b64>"` inside `jopedime/carbon/prod/edge-secrets` and bump `EdgeSecretsVersionId`.
3. **Mint per-user `ctk2.*` tokens in `app.veox.ai` (Rust via `jope-envelope`, or CLI)**:
   - Wire format: `ctk2.<base64url_no_pad(claims_json)>.<base64url_no_pad(raw_r_s_64_bytes)>` over ASCII signing input `"ctk2.<claims_b64url>"`.
   - Claims JSON (`AppCapabilityClaims`):
     ```json
     {
       "tenant_id": "veox_usr_01j...",
       "tier": "pro",
       "exp_epoch_secs": 1795003600,
       "iss": "app.veox.ai",
       "kid": "veox-2026-11"
     }
     ```
   - CLI offline minting:
     ```sh
     carbon token-mint \
       --signing-key-file ./veox-app-signing-active.pem \
       --tenant veox_usr_01j9example \
       --tier pro \
       --ttl-secs 3600 \
       --issuer app.veox.ai \
       --kid veox-2026-11
     ```
   - **Enforced guardrails on `carbon-edge`**:
     - `tenant_id` must match `^[A-Za-z0-9_-]{3,64}$`; every CAS blob (`blobs/<tenant_id>/<upload_id>/<sha256>`) and run object (`runs/<tenant_id>/<run_id>/*`) is strictly isolated to that `tenant_id`.
     - `ctk2.*` tokens can grant only `free` or `pro` (`Enterprise` and `admin=true` are structurally rejected).
     - Remaining lifetime (`exp_epoch_secs - now_epoch_secs`) is capped at `86,400 s` (`24 h`) by default (`DEFAULT_MAX_CTK2_TTL_SECS`).

### Path B (Immediate Worker-as-Client Bridge): `veox-app` Pro `ctk1.*` Token & Rotation

While `ai-veox-app` runs in worker-as-client mode (`PR #155`), mint a dedicated `veox-app` Pro `ctk1.*` token directly from the regional `prod` Secrets Manager vault (all Wave-1 `prod` regions share the synchronized `auth_secret_hex`):

```sh
AWS_PROFILE=veox-jepson-prod \
  carbon token-mint \
  --env prod \
  --region us-east-2 \
  --tenant veox-app \
  --tier pro \
  --ttl-secs 2592000
```

- **Zero-Downtime Secret Rotation (`[active, previous]`)**:
  1. In `jopedime/carbon/prod/edge-secrets`, move current `auth_secret_hex` into `previous_auth_secret_hex`, set a fresh 32-byte CSPRNG hex string as `auth_secret_hex`, and update `EdgeSecretsVersionId`.
  2. Both old and newly minted `ctk1.*` tokens verify cleanly in constant time during the overlap window.
  3. Issue the new `veox-app` `ctk1.*` token, roll `ai-veox-app` workers, then remove `previous_auth_secret_hex` after the overlap window expires.

---

## 3. Output Contract (`jope.carbon.fit_receipt.v1`), `holdout_rows_sha256`, & `release_id` Policy

### `carbon fit --format json` / `carbon run --format json` Schema (`v1`)

`CarbonCliRunReceipt` carries explicit schema and split-provenance fields (additive, preserving all existing `v1.2.x` fields):

| Field | Type | Semantics |
| --- | --- | --- |
| `schema` | `string` | `"jope.carbon.fit_receipt.v1"` |
| `schema_version` | `u32` | `1` (incremented on any breaking change; additive optional fields preserve `schema_version = 1`) |
| `release_id` | `string \| null` | Active cell `ReleaseId` read from `x-carbon-release-id` / `/v1/carbon/version` |
| `split_mode` | `string` | `"random"` (deterministic hash split) or `"forward"` (chronological tail split) |
| `seed` | `u64` | Deterministic split & search seed |
| `holdout_rows_sha256` | `string` | Lowercase 64-hex SHA-256 over the ordered held-out rows (`compute_canonical_holdout_rows_sha256`), binding Carbon's `outer_holdout_score` and `comparator_baselines` to the exact same evaluation rows for `baseline-protocol/v1` |
| `holdout_jmat_sha256` | `string` | Lowercase 64-hex SHA-256 of the plaintext `holdout.jmat` binary artifact |

### Canonical `holdout_rows_sha256` Definition

`compute_canonical_holdout_rows_sha256(holdout_indices, holdout_x, holdout_y)` hashes:
1. Domain tag `b"jope.carbon.holdout_rows.v1\0"`;
2. `u64::to_le_bytes(holdout_row_count)` and `u64::to_le_bytes(encoded_feature_count)`;
3. For each held-out row `i` in deterministic split order:
   - `u64::to_le_bytes(holdout_indices[i] as u64)` (`0`-based row index in the input table),
   - Little-endian IEEE-754 `f64` bytes for every encoded feature `holdout_x[i][j]` (`-0.0` canonicalized to `+0.0` `0x0000_0000_0000_0000`, all `NaN` bit patterns canonicalized to `0x7ff8_0000_0000_0000`),
   - Canonical little-endian IEEE-754 `f64` bytes for `holdout_y[i]`.

### Stability & `prod` `release_id` Notice Policy

1. **Additive-Only within `schema_version = 1`**: Existing fields on `CarbonCliRunReceipt` and `CarbonSavedModelHandle` (`jope-carbon-model-handle-v1`) are never renamed, retyped, or removed without bumping `schema_version` and providing advance notice + dual-read support.
2. **Staged `dev` -> `prod` Promotion**: Every cell or `r882` model change deploys to `dev` first, passes `carbon deploy --check-drift` + 17-vector `carbon redteam`, and notifies `veox-app` owners with the new `release_id`, `git_sha`, and `r882_bundle_digest` before shifting `prod` `:live` aliases (`carbon rollback --env prod` restores prior aliases in `< 1 s`).

---

## 4. Auto-Shred on Failed Runs, Standalone `--save` Handles, & JS/C++ Export

1. **Automatic CAS Shred on Mid-Flight Failure**: `execute_carbon_cli_run_with_ux` arms an upload-scoped `ShredOnDropGuard` (`DELETE /v1/carbon/runs/{id}?tenant_id=...&upload_id=...&fit_sha256=...&holdout_sha256=...`) immediately after `upload_prepared` completes. If `start_run`, `resume_run`, `/predict`, or local handle persistence fails, the client immediately deletes both staged `CUEN\x01` ciphertext blobs and `_uploads/*.uid` pointers instead of waiting for the 24-hour S3 lifecycle rule.
2. **Self-Contained `--save` Handles for Ephemeral Free Sessions**: Single-run ephemeral `anon_*` tokens are now embedded in `--save <path>` handles automatically (while long-lived Pro/registered tokens remain stripped unless `--embed-token` is passed, with automatic fallback to `$CARBON_CONFIG_DIR/runs/<id>.json` and `carbon-credentials.json`).
3. **CLI Flag Symmetry**: `predict`, `status`, `shred`, `explain`, `export`, and `resume` all accept `--cloud`, `--tier` (`--free`/`--pro`), `--clouds-manifest`, `--endpoint`, and `--tenant` (including stateless `carbon shred --run <id> --cloud prod --token ...` when no local handle exists).
4. **Standalone Fitted `JS` & `C++17` Export**: `carbon export --lang js` and `carbon export --lang cpp` now support `gbm` (`histgbm`), `histgbm2`, `blend` (`ensemble`), `ridge_gcv`, `krr`, `gpr`, and `ksmooth` in addition to `ridge` and `elasticnet`.

---

## 5. Production Spend Caps, Lambda Concurrency Headroom (`896` Slots), & Classification Roadmap

### Verified AWS Account & Per-Tier Limits (Audited `2026-11-17`, Account `714044927654`)

| Control | Live Value | Notes |
| --- | --- | --- |
| AWS Account Monthly Budget (`jope-monthly-1000usd`) | **`$1,000.00 / month`** | Account-wide AWS Budgets threshold (`~$0.001–$0.015` per typical `fast`/`medium` Carbon run). |
| Pro Tier Per-Run Admission (`CarbonProductTier::Pro`) | **`<= 100,000` rows, `<= 2,000` features, `<= 256 MiB` upload slot** | Supports `fast`, `medium`, `high`, and `ultra` efforts (`HybridFanOut` up to `4` shards/wave on `prod`), `10,000` rows/call and `1,000` `/predict` calls/run. |
| Edge Sliding-Window Rate Limits (`CarbonRateLimiter`) | `300/min` `uploads/prepare`, `120/min` `POST /runs`, `120/min` `DELETE /runs/{id}` | Per-`tenant_id` (`veox_usr_*` per-user `ctk2.*` tokens get independent per-user rate buckets!). |
| Regional Lambda Concurrent Executions Quota (`L-B99A9384`) | **`1,000` / region** across all 5 Wave-1 `prod` regions | `us-east-2`, `us-east-1`, `us-west-2`, `eu-west-1`, `eu-central-1`. |

### Resolving the `896` Reserved Concurrency Bottleneck (Reclaimed `2026-11-18`)

Live inspection across all 5 Wave-1 `prod` regions (`us-east-2`, `us-east-1`, `us-west-2`, `eu-west-1`, `eu-central-1`) plus `ca-central-1` confirmed:
- `jopedime-carbon-prod-{edge,engine,eval}` have `ReservedConcurrentExecutions = None` (drawing from each region's unreserved pool).
- Retired campaign function **`jope-gp-hwm-hybrid-x86-20261001`** previously held **`ReservedConcurrentExecutions = 896`** (`0` invocations over 14 days), leaving only `100–104` unreserved concurrent executions per region.
- On `2026-11-18`, [`deploy/carbon/reclaim-idle-concurrency.sh`](../../deploy/carbon/reclaim-idle-concurrency.sh) verified `0` invocations and released `jope-gp-hwm-hybrid-x86-20261001`'s `896` reservation across all 6 regions, restoring **`996–1,000` unreserved concurrent slots per region (`~5,000` across Wave-1 `prod`)**.

```sh
AWS_PROFILE=veox-jepson-prod bash deploy/carbon/reclaim-idle-concurrency.sh \
  --json-out docs/carbon/evidence/concurrency-reclaim.json
```

### Pinned `app.veox.ai` `ctk2.*` Public Verification Keys (`2026-11-18`)

- **`dev` (`jopedime-carbon-dev`, `us-east-2`)**: Tracked in [`deploy/carbon/pubkeys/veox-app-dev.pub`](../../deploy/carbon/pubkeys/veox-app-dev.pub) (`kid = veox-app-dev-2026-11`), wired automatically by `deploy/carbon/release-and-promote.sh`.
- **`prod` (`jopedime-carbon-prod`, Wave-1 5 regions)**: Tracked in [`deploy/carbon/pubkeys/veox-app-prod.pub`](../../deploy/carbon/pubkeys/veox-app-prod.pub) (`kid = veox-app-prod-2026-11`), wired automatically by `deploy/carbon/release-and-promote.sh`.

### Classification Status & Roadmap

- **Current (`v1.3.0`)**:
  - **Regression**: `45` built-in objectives (`rmse`, `mae`, `huber`, `pinball`, `tail_var`, `asymmetric_mse`, and custom `@file.json` expressions).
  - **2-Class Binary Classification**: Universal 2-class categorical/boolean/numeric target conditioning (`jope-prep::target`: `yes`/`no`, `true`/`false`, `+`/`-`, `<=50K`/`>50K`, `0`/`1`, arbitrary 2-class strings) with deterministic stratified splits (`SplitPlan::stratified_binary`), `binary_target` metrics (`holdout_roc_auc`, `holdout_brier`, `holdout_logloss`, `holdout_accuracy`), and `prediction,predicted_probability,predicted_label` CSV scoring.
- **Planned Next Milestone (Multiclass `>= 3` Classes & Dedicated Binary Neural GP Bundle)**:
  1. Export dedicated binary/multiclass neural GP landmark weights alongside `r882`.
  2. Multi-column class probability vectors and `>= 3`-class macro-F1 / log-loss search heads (until then, `>= 3`-class tables decline cleanly to the `veox` appliance lane).

---

## 6. Standalone WASM-Ready Client Envelope Crate (`crates/jope-envelope`, `ADR-0007 C7`)

To enable `jAIn` (desktop/Tauri and browser `wasm32-unknown-unknown`) and `app.veox.ai` to encrypt locally and mint `ctk2.*` tokens without pulling `ureq`, `rustls`, `aws-lc-sys` (C/CMake), `tokio`, or `clap`, the client cryptography now lives in **`crates/jope-envelope`** alongside `crates/jope-prep`:

- **Properties**: `#![forbid(unsafe_code)]`, pure Rust (`p256`, `aes`, `sha2`, `base64`, `zeroize`), `0` C/ASM FFI, `0` `getrandom` dependency on `wasm32-unknown-unknown` (callers pass explicit CSPRNG bytes from `crypto.getRandomValues` in WASM), bit-for-bit wire-identical to `jope-carbon`'s `aws-lc-rs` `CarbonEngineKek` implementation.
- **Core APIs**:
  - `jope_envelope::ClientEnvelopeKeypair` (`from_secret_bytes`, `pubkey_sec1_bytes`, `pubkey_b64`)
  - `jope_envelope::wrap_u_dek_to_pubkey` (`CUDK\x01` `ECDH-ES P-256 + HKDF-SHA256 + AES-256-GCM`) & `unwrap_u_dek_with_keypair`
  - `jope_envelope::encrypt_cuen_payload` (`CUEN\x01` slot-separated `HKDF-SHA256 + AES-256-GCM`) & `decrypt_cuen_payload`
  - `jope_envelope::AppSigningKey`, `AppCapabilityClaims`, `AppTokenTier`, `verify_ctk2_token_with_pubkeys`
- **WASM Verification Command**:
  ```sh
  cargo check --target wasm32-unknown-unknown -p jope-prep -p jope-envelope
  ```
