//! Versioned, bundle-relative model artifact metadata.
//!
//! The manifest is intentionally boring: every runtime file is named relative to the
//! bundle root and is checked before a model is loaded.  This prevents a locally-trained
//! checkpoint or an operator's absolute path from silently becoming a product dependency.

use anyhow::{bail, Context, Result};
use rustix::fs::{open, openat, Mode, OFlags};
use rustix::io::Errno;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::File;
use std::io::Read;
use std::os::fd::AsFd;
use std::path::{Component, Path, PathBuf};

mod integrity;
pub(crate) mod loading;
mod projection;
mod validation;

pub use projection::{
    JopePolicyAssetsIdentity, JopePolicyProjectionLayout, VerifiedJopePolicyAssets,
};

use integrity::*;
use validation::{verify_jope_metadata, verify_operator_registry, verify_target_metadata};

pub const MODEL_BUNDLE_V1: &str = "model-bundle/v1";
pub const TENSOR_INVENTORY_V1: &str = "sha256-name-nul-dtype-nul-shape-csv-lf-v1";
const JOPE_BUNDLE_ID: &str = "jope-prime-gen31-v2";
const JOPE_SOURCE_CHECKPOINT: &str = "runs/self_play_v33/gen31.pt";
const JOPE_SOURCE_CHECKPOINT_SHA256: &str =
    "6a20443443d4d7edc75aa18ef2825c3f240ea92b02f09573c6f53c41e76f60d0";
const JOPE_POLICY_VERSION: &str = "policy_v3.2";
const JOPE_DROPPED_TRAINING_TENSORS: [&str; 7] = [
    "loss_log_vars.ablation_bt",
    "loss_log_vars.reward_bt",
    "loss_log_vars.reward_listwise",
    "loss_log_vars.routing_ce",
    "loss_log_vars.runtime_huber",
    "loss_log_vars.seq_nll",
    "loss_log_vars.stability_bce",
];
const JOPE_POLICY_METADATA: &str = "artifacts/model-bundle/jope_policy.meta.json";
const JOPE_POLICY_ARTIFACT: &str = "artifacts/model-bundle/jope_policy.safetensors";
const JOPE_PREPROCESSING_ARTIFACT: &str = "artifacts/model-bundle/preprocessing.safetensors";
const TARGET_METADATA: &str = "artifacts/model-bundle/target_encoder_fused_v1.meta.json";
const OPERATOR_REGISTRY: &str = "artifacts/model-bundle/operator_registry.json";
const JOPE_VOCABULARY_SHA256: &str =
    "000128f29eecd6f3345fca2ff2230729c913cc45bc2882cb0e8f962f90b1c40e";
const JOPE_VOCABULARY_CANONICAL_SHA256: &str =
    "0701a24881e1e6c9899da746e405873b1e58935ffd19cb60073b8c824281ee7d";
const JOPE_GRAMMAR_SHA256: &str =
    "a455ee2a435cf105d961ff42a397673779ec25a7c06cacf7061b9f1471c9c180";
// The frozen manifest above records the original lossy serde_json parse. Keep
// that declaration unchanged, but verify content with explicit roundtrip f64s.
const JOPE_GRAMMAR_CONTENT_SHA256: &str =
    "7664781efab15adebe94829467e37221bb475e5ee6e0e6cde74cf47045b1aa30";
const OPERATOR_REGISTRY_SHA256: &str =
    "62c7ab26dfcf0f6c0047fe6c9eec192b0440a7080bac92d772c08eed9279c8cb";
const OPERATOR_REGISTRY_SEMANTIC_SHA256: &str =
    "81c79a29141421642a067549009c6491ac30cde98e48d7d4480824dbff98a823";
const FOUNDATION_ARTIFACT: &str = "artifacts/starforge/starlight_core_regressor.safetensors";
const FOUNDATION_SHA256: &str = "f663ae49cc0f1504f07800a69dd6c31ca22fc473f26bbc0684eac25129a04128";
const FOUNDATION_RECIPE_SHA256: &str =
    "551137f5e0ee2f56c0bd182a385480a0686ce7689a5ea8cf5dee7a801de8d9f0";
const FOUNDATION_TENSOR_INVENTORY_SHA256: &str =
    "f4939f17b047ba1b7d6470a72013379499349f315441fca59a511161592e2d03";
const FOUNDATION_BYTES: u64 = 233_183_532;
const FOUNDATION_TENSORS: usize = 504;
const MODEL_BUNDLE_MANIFEST_SHA256: &str =
    "5e7884ec1c0e0b955a2381d5b69c54beddad9ec66a1980c94ad279d974486d05";
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_NOTICE_BYTES: u64 = 64 * 1024;
const MAX_MODEL_FILE_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelPolicyEligibilityV1 {
    pub eligible: bool,
    pub open_only: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelLicenseV1 {
    pub schema: String,
    pub component_id: String,
    pub component_kind: String,
    pub sha256: String,
    pub spdx_expression: String,
    pub license_class: String,
    pub source_revision: String,
    pub source_uri: String,
    pub redistribution: String,
    pub model_lineage: String,
    pub notice_paths: Vec<String>,
    pub policy_eligibility: ModelPolicyEligibilityV1,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct EncoderIdentityV1 {
    pub version: String,
    pub artifact: String,
    pub sha256: String,
    pub source_revision: String,
    pub source_uri: String,
    pub recipe_sha256: String,
    pub recipe: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ModelFileV1 {
    pub path: String,
    pub kind: String,
    pub shape: Vec<usize>,
    pub bytes: u64,
    pub sha256: String,
    pub tensor_count: usize,
    pub tensor_inventory_sha256: Option<String>,
    pub license: ModelLicenseV1,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ModelBundleV1 {
    pub schema: String,
    pub bundle_id: String,
    pub source_checkpoint: String,
    pub source_checkpoint_sha256: String,
    pub vocabulary_sha256: String,
    pub vocabulary_canonical_sha256: String,
    pub grammar_sha256: String,
    pub registry_sha256: String,
    pub registry_semantic_sha256: String,
    pub policy_version: String,
    pub tensor_inventory_algorithm: String,
    pub encoders: BTreeMap<String, EncoderIdentityV1>,
    pub files: Vec<ModelFileV1>,
    pub notices: BTreeMap<String, String>,
}

pub struct LoadedModelBundle {
    root: PathBuf,
    root_dir: File,
    pub manifest: ModelBundleV1,
    target_metadata: TargetEncoderMetadataV1,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TensorSpec {
    name: String,
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct JopePolicyMetadata {
    config: JopePolicyConfig,
    dropped_training_tensors: Vec<String>,
    grammar_specs: Vec<serde_json::Value>,
    model_version: String,
    source_checkpoint: String,
    source_checkpoint_sha256: String,
    vocab: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct JopePolicyConfig {
    d_model: usize,
    n_layers: usize,
    n_heads: usize,
    ffn_dim: usize,
    dataset_dim: usize,
    target_dim: usize,
    reward_ensemble_k: usize,
    embedding_rows: usize,
    routing_classes: usize,
    vocab_size: usize,
    max_seq_len: usize,
    padding_idx: usize,
    layernorm_eps: f64,
    gelu: String,
    norm_first: bool,
    attn_scale: f64,
}

#[derive(Clone, Debug, Deserialize)]
struct TargetEncoderMetadataV1 {
    schema: String,
    artifact: String,
    artifact_bytes: u64,
    artifact_sha256: String,
    source_bundle: TargetSourceBundle,
    source_revision: String,
    source_uri: String,
    seed: u64,
    input_preprocessing: TargetInputPreprocessing,
    license: ModelLicenseV1,
    model_config: TargetModelConfig,
    normalization: TargetNormalization,
    block_layout: Vec<TargetBlock>,
    vector_dim: usize,
    signature_dim: usize,
    classical_component_dim: usize,
    actual_svd_dim: usize,
    tensor_count: usize,
    tensor_inventory_algorithm: String,
    tensor_inventory_sha256: String,
    tensors: Vec<TensorSpec>,
    feature_names: Vec<String>,
    fit_records: Vec<String>,
    source_epochs: TargetSourceEpochs,
    task_support: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct TargetSourceEpochs {
    best: usize,
    #[serde(rename = "final")]
    final_value: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct TargetSourceBundle {
    format: String,
    version: usize,
    components: Vec<TargetSourceFile>,
    oracle_files: Vec<TargetSourceFile>,
}

#[derive(Clone, Debug, Deserialize)]
struct TargetSourceFile {
    path: String,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct TargetInputPreprocessing {
    binary: String,
    regression: String,
    non_finite: String,
    empty_after_filter: String,
    dtype: String,
    neural_seed: String,
    signature_seed: String,
    signature_views: usize,
    window_size: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct TargetModelConfig {
    attention_heads: usize,
    attention_layers: usize,
    embedding_dim: usize,
    fourier_frequencies: usize,
    hidden_dim: usize,
    histogram_bins: usize,
    rbf_bins: usize,
    rich_stats: bool,
    token_dim: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct TargetNormalization {
    blocks: String,
    epsilon: f64,
    #[serde(rename = "final")]
    final_value: String,
    neural_window_pool: String,
}

#[derive(Clone, Debug, Deserialize)]
struct TargetBlock {
    name: String,
    start: usize,
    end: usize,
    source_dim: usize,
    padded_dim: Option<usize>,
    weight: f64,
}

#[cfg(test)]
mod tests;
