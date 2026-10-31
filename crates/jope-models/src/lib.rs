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
pub mod fetch;
pub mod index;
pub mod manifest;
pub mod publish;
pub mod repo;

use anyhow::Result;
use jope_datasets::store::Store;

/// Default models bucket (`JOPE_MODELS_URL` overrides).
pub const DEFAULT_MODELS_URL: &str = "s3://veox-jopedime-models-use2";

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
