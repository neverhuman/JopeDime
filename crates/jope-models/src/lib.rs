//! JOPE-Dime model bundles: content-addressed weights in one bucket
//! (`blobs/sha256/<sha256>` + `bundles/<bundle_id>/manifest.json`), committed manifests under
//! `models/` in the repository, and a local cache under `$JOPE_HOME/models`.
//!
//! The transport, the `.partial` + hash-then-rename download and the blob directory come from
//! [`jope_datasets`] (`store`, `blobs`, `paths`); this crate adds the bundle manifest
//! (`jope.model-bundle/v2`), the index (`jope.model-index/v1`), the bundle cache layout and the
//! `fetch` / `verify` / `publish` / `path` commands.

#![warn(missing_docs)]

pub mod cache;
pub mod cli;
/// Frozen 42-dataset golden embedding cohort shared with the study tooling.
pub mod embeddings_golden;
pub mod fetch;
pub mod index;
pub mod manifest;
pub mod publish;
pub mod repo;

use anyhow::Result;
use jope_datasets::store::Store;

/// Default models bucket (`JOPE_MODELS_URL` overrides).
pub const DEFAULT_MODELS_URL: &str = "s3://veox-jopedime-models-use2";

/// Canonical bundle identifier for the v2 unified regression + binary ICL foundation bundle.
pub const UNIFIED_ICL_REGBIN_V2_BUNDLE_ID: &str = "unified-icl-regbin-v2";
/// Semantic release version of the `unified-icl-regbin-v2` model bundle.
pub const UNIFIED_ICL_REGBIN_V2_VERSION: &str = "2.0.0";
/// Pinned SHA-256 digest of the `unified-icl-regbin-v2` bundle manifest (`jope.model-bundle/v2`).
pub const UNIFIED_ICL_REGBIN_V2_MANIFEST_SHA256: &str =
    "cc04e4ab815b82c9f8df58f239c7ce86ac942386744317cd04fd8de70ae3b677";
/// Pinned SHA-256 digest of `unified_icl_regbin_M.safetensors` (`Stage 2B FullS3` 3-valley soup, 25.0M params).
pub const UNIFIED_ICL_REGBIN_M_V2_SHA256: &str =
    "51dcb550184377b8de42a0bb9418f72b5aa7ecc2eba4a127eda8e3edb55815de";
/// Pinned SHA-256 digest of `unified_icl_regbin_L.safetensors` (`Stage 2B FullS3` 3-valley soup, 109.1M params).
pub const UNIFIED_ICL_REGBIN_L_V2_SHA256: &str =
    "fcfb119e6f2a7f0d5418e7d98a1560d173b72cdbb998af5681fa140ba86e10ab";
/// Pinned SHA-256 digest of `unified_icl_regbin_M_stage2a_noarena.safetensors` (`Stage 2A NoArena`, 25.0M params).
pub const UNIFIED_ICL_REGBIN_M_V2_STAGE2A_SHA256: &str =
    "96b4b4959109d689f2424d21a50e549760cf05f3433494c548c6832678eb916b";
/// Pinned SHA-256 digest of `unified_icl_regbin_L_stage2a_noarena.safetensors` (`Stage 2A NoArena`, 109.1M params).
pub const UNIFIED_ICL_REGBIN_L_V2_STAGE2A_SHA256: &str =
    "9cf2bb3f2d56cb7d2fba2a959029c15715c8414b0104b2cdc12a0bc5204c0b33";
/// Pinned SHA-256 digest of frozen `DOPE v1.1.0` `dope.safetensors` (`dope-core@15e12aa3dc33`).
pub const UNIFIED_ICL_REGBIN_DOPE_SHA256: &str =
    "0f5e98fa29cf9c45bfaac5a3cf001b13a0de37d7b6e154c582d7ec689b21a84d";

/// `JOPE_MODELS_URL`, else the default bucket.
pub fn models_url() -> String {
    std::env::var("JOPE_MODELS_URL")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_MODELS_URL.to_string())
}

/// Open the models store (`JOPE_MODELS_URL`; unsigned S3 requests with `JOPE_MODELS_ANON=1`).
pub fn open_store(url: Option<&str>) -> Result<Store> {
    let url = url.map_or_else(models_url, str::to_string);
    let anonymous = std::env::var("JOPE_MODELS_ANON").is_ok_and(|v| v == "1");
    Store::open_with(&url, anonymous)
}

/// The object key of a bundle manifest.
pub fn manifest_key(bundle_id: &str) -> String {
    format!("bundles/{bundle_id}/manifest.json")
}
