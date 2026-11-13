//! Carbon serverless cell v1 wire contracts (`ADR-0007`).
//!
//! Public request, capability-auth, per-user zero-knowledge envelope (`CUEN\x01` / `CUDK\x01`),
//! progress, prediction, cryptographic shred, and multi-region discovery payloads shared by
//! WASM/native clients, `carbon-edge` regional gateways, and Enterprise `carbon serve` instances.
//! No candidate structure, expression, or internal search representation appears in any wire payload.

use crate::{
    decimal::DecimalU64,
    ids::{DatasetId, RunId},
    Effort,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// `CarbonTierWire` is an enum naming the product tier enforced at the Carbon edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CarbonTierWire {
    Free,
    Pro,
    Enterprise,
}

/// `CarbonRunStateWire` is an enum representing the lifecycle state of a Carbon run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CarbonRunStateWire {
    Queued,
    Running,
    Stopping,
    Paused,
    Completed,
    Failed,
}

/// `CarbonExecutionModeWire` is an enum describing whether a Carbon run executes in-process
/// on one `engine` worker or fans out shards across stateless `eval` workers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CarbonExecutionModeWire {
    SoloSingleLambda,
    HybridFanOut,
}

/// `CarbonAnonymousAuthResponse` is a struct returned by `POST /v1/carbon/auth/anonymous`
/// issuing a short-lived HMAC-signed Free-tier capability token bound to a fresh `tenant_id`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CarbonAnonymousAuthResponse {
    pub tenant_id: String,
    pub capability_token: String,
    pub tier: CarbonTierWire,
    pub expires_epoch_secs: DecimalU64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_kek_pubkey_b64: Option<String>,
}

/// `CarbonRegisterFreeRequest` is a struct submitted to `POST /v1/carbon/auth/register`
/// to register a persistent `30`-day Free-tier capability token with an optional user label.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CarbonRegisterFreeRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// `CarbonRegisterFreeResponse` is a struct returned by `POST /v1/carbon/auth/register`
/// issuing a `30`-day HMAC-signed Free-tier capability token bound to a unique `free_<...>` `tenant_id`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CarbonRegisterFreeResponse {
    pub tenant_id: String,
    pub capability_token: String,
    pub tier: CarbonTierWire,
    pub expires_epoch_secs: DecimalU64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_kek_pubkey_b64: Option<String>,
}

/// `CarbonUploadSlotWire` is a struct declaring the SHA-256 digest and byte length of a
/// client-encoded (and `CUEN\x01`-encrypted) `JMAT` v1 split artifact before presigning.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CarbonUploadSlotWire {
    pub sha256: String,
    pub bytes: DecimalU64,
}

/// `CarbonPrepareUploadRequest` is a struct requesting authenticated upload URLs for
/// `fit.jmat` and `holdout.jmat`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CarbonPrepareUploadRequest {
    pub tenant_id: String,
    pub tier: CarbonTierWire,
    pub fit_rows: u32,
    pub holdout_rows: u32,
    pub feature_count: u32,
    pub fit_jmat: CarbonUploadSlotWire,
    pub holdout_jmat: CarbonUploadSlotWire,
    #[serde(default)]
    pub encrypted_cuen: bool,
}

/// `CarbonPresignedTarget` is a struct returning one time-bounded, size-pinned upload destination.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CarbonPresignedTarget {
    pub url: String,
    pub method: String,
    pub sha256: String,
    #[serde(default)]
    pub max_bytes: DecimalU64,
    pub expires_epoch_secs: DecimalU64,
}

/// `CarbonPrepareUploadResponse` is a struct returning presigned `PUT` URLs for the
/// disjoint `FIT` and `HOLDOUT` blobs plus the `carbon-engine` P-256 KEK public key.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CarbonPrepareUploadResponse {
    pub upload_id: String,
    pub fit_target: CarbonPresignedTarget,
    pub holdout_target: CarbonPresignedTarget,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_kek_pubkey_b64: Option<String>,
}

/// `CarbonCreateRunRequest` is a struct submitting a conditioned Carbon search run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CarbonCreateRunRequest {
    pub run_id: RunId,
    pub dataset_id: DatasetId,
    pub tenant_id: String,
    pub tier: CarbonTierWire,
    pub fit_jmat_sha256: String,
    pub holdout_jmat_sha256: String,
    pub fit_rows: u32,
    pub holdout_rows: u32,
    pub feature_names: Vec<String>,
    pub target_name: String,
    pub objective: String,
    pub effort: Effort,
    pub seed: DecimalU64,
    /// Base64-encoded `JLMB` v1 `FitLandmarkBundle` (or `CUEN\x01(LandmarkJlmb)` ciphertext when `encrypted_cuen == true`).
    pub landmark_bundle_b64: String,
    #[serde(default)]
    pub encrypted_cuen: bool,
    /// Base64-encoded `133`-byte `CUDK\x01` envelope wrapping the per-run `U_DEK` to `carbon-engine`'s P-256 KEK.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wrapped_u_dek_b64: Option<String>,
    /// Optional upload session identifier (`upl_<ulid>`) scoping S3 CAS blobs to `(tenant_id, upload_id)`
    /// so `DELETE /v1/carbon/runs/{id}` destroys blobs without cross-run collision (`GAP-S3-3`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upload_id: Option<String>,
}

/// `CarbonEarlyStopRequest` is a struct signaling cooperative wave-boundary drain ("see it now").
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CarbonEarlyStopRequest {
    pub tenant_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// `CarbonResumeRunRequest` is a struct resuming a paused Carbon run from its sealed checkpoint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CarbonResumeRunRequest {
    pub tenant_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<Effort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_waves: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wrapped_u_dek_b64: Option<String>,
}

/// `CarbonPublicWaveProgress` is a struct reporting strictly opaque scalar wave progress,
/// outer-holdout metrics, authenticated neural bundle identity, and sealed-state digests
/// with zero candidate structure fields.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CarbonPublicWaveProgress {
    pub run_id: RunId,
    pub state: CarbonRunStateWire,
    pub execution_mode: CarbonExecutionModeWire,
    pub eval_memory_mb: u32,
    pub eval_shards: u32,
    #[serde(default)]
    pub eval_invocations: DecimalU64,
    #[serde(default)]
    pub eval_total_wall_ms: DecimalU64,
    pub wave_index: u32,
    pub waves_target: u32,
    pub candidates_evaluated: DecimalU64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_cv_loss: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_cv_metric: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outer_holdout_loss: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outer_holdout_metric: Option<f64>,
    pub neural_seed_applied: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub neural_guidance_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bundle_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier_applied: Option<CarbonTierWire>,
    #[serde(default)]
    pub encrypted_cuen: bool,
    pub can_resume: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sealed_checkpoint_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sealed_champion_sha256: Option<String>,
    pub elapsed_ms: DecimalU64,
    pub region: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_message: Option<String>,
}

fn is_false(v: &bool) -> bool {
    !*v
}

/// `CarbonPreparePredictRequest` is a struct requesting a size/SHA-pinned presigned `PUT` URL
/// for uploading large `CUEN\x01(PredictFeatures)` batches directly to S3 CAS (`GAP-S3-2`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CarbonPreparePredictRequest {
    pub tenant_id: String,
    pub predict_cuen: CarbonUploadSlotWire,
}

/// `CarbonPreparePredictResponse` is a struct returning a presigned S3/gateway `PUT` target for
/// large encrypted `/predict` batches.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CarbonPreparePredictResponse {
    pub upload_id: String,
    pub input_target: CarbonPresignedTarget,
}

/// `CarbonPredictRequest` is a struct submitting feature rows (either cleartext in local VPC mode
/// or `CUEN\x01`-encrypted under `U_DEK` inline or via S3 CAS) for champion inference.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CarbonPredictRequest {
    pub tenant_id: String,
    #[serde(default)]
    pub feature_names: Vec<String>,
    #[serde(default)]
    pub rows: Vec<Vec<Option<f64>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypted_rows_cuen_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypted_rows_cas_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predict_upload_id: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub prefer_cas_output: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_response_pubkey_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wrapped_u_dek_b64: Option<String>,
}

/// `CarbonPredictResponse` is a struct returning scalar predictions (or an E2E `CUEN\x01`
/// response inline or via presigned S3 CAS `GET`) without exposing model internals.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CarbonPredictResponse {
    pub run_id: RunId,
    #[serde(default)]
    pub predictions: Vec<f64>,
    pub predictions_sha256: String,
    pub row_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypted_predictions_cuen_b64: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypted_predictions_cas_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypted_predictions_cas_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_cudk_b64: Option<String>,
}

/// `CarbonShredRunResponse` is a struct returned by `DELETE /v1/carbon/runs/{id}` confirming
/// cryptographic destruction of user ciphertexts, sealed state, and wrapped key envelopes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CarbonShredRunResponse {
    pub run_id: RunId,
    pub shredded: bool,
    pub objects_deleted: u32,
}

/// `CarbonPingResponse` is a struct returned by `/v1/carbon/ping` for smart-routing RTT probes
/// and advertising the regional `carbon-engine` P-256 KEK public key.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CarbonPingResponse {
    pub region: String,
    pub healthy: bool,
    pub server_epoch_ms: DecimalU64,
    pub enterprise_self_hosted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_kek_pubkey_b64: Option<String>,
}

/// `CarbonVersionResponse` is a struct returned by `GET /v1/carbon/version` exposing build
/// commit provenance, release tag, deployed `ReleaseId`, and `r882` bundle digest (`GAP-V3`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct CarbonVersionResponse {
    pub version: String,
    pub release_tag: String,
    pub git_sha: String,
    pub build_target: String,
    pub region: String,
    pub enterprise_self_hosted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub release_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r882_bundle_digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_kek_pubkey_b64: Option<String>,
    pub server_epoch_ms: DecimalU64,
}

/// `CarbonRegionDescriptor` is a struct describing one regional Carbon Lambda cell endpoint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CarbonRegionDescriptor {
    pub region: String,
    pub geography: String,
    pub endpoint_url: String,
    pub priority_rank: u32,
}

/// `CarbonRegionCatalog` is a struct served at `/.well-known/carbon-regions.json` listing
/// all active regional Carbon cells for latency-probing client failover.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CarbonRegionCatalog {
    pub catalog_version: u32,
    pub regions: Vec<CarbonRegionDescriptor>,
}

/// Canonical 17-region AWS Graviton2 (`arm64`) cell topology from `ADR-0007`.
pub const CARBON_DEFAULT_REGIONS: &[(&str, &str)] = &[
    ("us-east-2", "americas"),
    ("us-east-1", "americas"),
    ("us-west-2", "americas"),
    ("us-west-1", "americas"),
    ("ca-central-1", "americas"),
    ("sa-east-1", "americas"),
    ("eu-west-1", "europe"),
    ("eu-central-1", "europe"),
    ("eu-west-2", "europe"),
    ("eu-west-3", "europe"),
    ("eu-north-1", "europe"),
    ("ap-northeast-1", "apac"),
    ("ap-northeast-2", "apac"),
    ("ap-southeast-1", "apac"),
    ("ap-southeast-2", "apac"),
    ("ap-south-1", "apac"),
    ("me-central-1", "mea"),
];

impl CarbonRegionCatalog {
    /// Construct the canonical 17-region catalog parameterized by `domain_suffix`
    /// (for example `"carbon.jopedime.com"` -> `"https://us-east-2.carbon.jopedime.com"`).
    pub fn default_17_region_catalog(domain_suffix: &str) -> Self {
        let trimmed = domain_suffix.trim_matches('/');
        let regions = CARBON_DEFAULT_REGIONS
            .iter()
            .enumerate()
            .map(|(idx, &(region, geography))| CarbonRegionDescriptor {
                region: region.to_owned(),
                geography: geography.to_owned(),
                endpoint_url: format!("https://{region}.{trimmed}"),
                priority_rank: (idx + 1) as u32,
            })
            .collect();
        Self {
            catalog_version: 1,
            regions,
        }
    }
}
