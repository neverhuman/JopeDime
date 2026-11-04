//! Language-neutral, immutable fitted objects. The manifest is the commit marker.
//!
//! The envelope can retain any versioned GP state; execution requires a registered decoder.
//! SHA-256 authenticates bytes against the manifest, not the author of those bytes.
use crate::fitted::FittedModel;
use crate::{sha256_hex, Failure};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[path = "calibration_platt.rs"]
pub mod calibration_platt;

pub const INSPECTION_SCHEMA: &str = "jope.model-object-inspection/v1";
pub const SCHEMA: &str = "jope.model-object/v1";
pub const APPLIANCE: &str = "org.jopedime/appliance-fitted/v1";
pub const BOUND: &str = "org.jopedime/battle-artifact/v2";
pub const MATH_PIPELINE: &str = "org.jopedime/math-pipeline/v1";
pub const MANIFEST: &str = "manifest.json";
pub const DEFAULT_ROOT: &str = "models";
pub const MAX_MANIFEST_BYTES: u64 = 1 << 20;
pub const MAX_FILES: usize = 256;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Producer {
    pub name: String,
    pub version: String,
    pub language: String,
}
impl Default for Producer {
    fn default() -> Self {
        Self {
            name: "jopedime".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            language: "rust".into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Part {
    pub path: String,
    pub media_type: String,
    pub bytes: u64,
    pub sha256: String,
}
impl Part {
    pub fn new(path: &str, media_type: &str, data: &[u8]) -> Self {
        Self {
            path: path.into(),
            media_type: media_type.into(),
            bytes: data.len() as u64,
            sha256: sha256_hex(data),
        }
    }
}

/// Flat filenames and separate buffers let every language implement the envelope without a
/// binary archive library, JSON float canonicalization, pickle, or source-code execution.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema: String,
    pub minimum_reader_version: u32,
    /// Namespaced state decoder, e.g. `org.jopedime/battle-artifact/v2`.
    pub format: String,
    pub producer: Producer,
    pub primary: String,
    /// All semantics needed for execution. An unknown requirement prevents loading.
    pub required_capabilities: Vec<String>,
    pub files: Vec<Part>,
    /// Extension metadata is retained; it must never override required capabilities.
    pub metadata: Value,
}

#[derive(Debug)]
pub struct ModelObject {
    id: String,
    manifest: Manifest,
    manifest_bytes: Vec<u8>,
    parts: BTreeMap<String, Vec<u8>>,
}

/// What [`ModelObject::predict`] reads for a Platt-calibrated object: the head's input schema,
/// plus the fitted header when the object is in appliance format (bound heads have none).
pub struct CalibratedInput {
    pub schema: feat_math::advanced_v3::DataSchema,
    pub genome_sha256: String,
    pub header: Option<crate::fitted::Header>,
}

impl CalibratedInput {
    /// Load a prediction table, ignoring the fitted target and auxiliary columns like a fitted
    /// model does. A bound head records neither, so every non-feature column is refused.
    pub fn table(
        &self,
        csv: &crate::table::CsvTable,
    ) -> Result<(feat_math::advanced_v3::RawTable, Vec<String>)> {
        match &self.header {
            Some(h) => crate::table::prediction_ignoring(
                csv,
                &self.schema,
                &h.target,
                &h.auxiliary_columns,
            ),
            None => {
                if let Some(column) = csv.headers.iter().find(|h| h.is_empty()) {
                    return Err(Failure::input(format!(
                        "input column {column:?} is not a feature of this model"
                    )));
                }
                crate::table::prediction_ignoring(csv, &self.schema, "", &[])
            }
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Inspection {
    pub schema: &'static str,
    pub id: String,
    pub integrity_verified: bool,
    pub executable: bool,
    pub unsupported_capabilities: Vec<String>,
    pub format: String,
    pub primary: String,
    pub files: Vec<Part>,
    pub metadata: Value,
}

fn hex_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn filename(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value != "."
        && value != ".."
        && !value.starts_with('.')
        && !value.ends_with('.')
        && !device_name(value)
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}
pub(crate) fn device_name(value: &str) -> bool {
    let stem = value.split('.').next().unwrap_or("").to_ascii_uppercase();
    matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && matches!(stem.as_bytes()[3], b'1'..=b'9'))
}
fn token(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"./:_-".contains(&b))
}

impl Manifest {
    pub fn validate(&self, max_bytes: u64) -> Result<()> {
        let bad = |s| Failure::integrity(format!("model object: {s}"));
        if self.schema != SCHEMA || self.minimum_reader_version != 1 {
            return Err(bad("unsupported envelope version"));
        }
        if !token(&self.format)
            || !self.format.contains('/')
            || self.format.split('/').any(str::is_empty)
            || self.producer.name.is_empty()
            || self.producer.version.is_empty()
            || self.producer.language.is_empty()
        {
            return Err(bad("invalid format or producer"));
        }
        let mut names = BTreeSet::new();
        let mut total = 0u64;
        if self.files.is_empty() || self.files.len() > MAX_FILES {
            return Err(bad("file count exceeds bounds"));
        }
        for part in &self.files {
            if !filename(&part.path)
                || part.path.eq_ignore_ascii_case(MANIFEST)
                || !names.insert(part.path.to_ascii_lowercase())
                || !hex_hash(&part.sha256)
                || part.media_type.is_empty()
            {
                return Err(bad("invalid or duplicate file descriptor"));
            }
            total = total
                .checked_add(part.bytes)
                .ok_or_else(|| bad("size overflow"))?;
        }
        if total > max_bytes || !self.files.iter().any(|part| part.path == self.primary) {
            return Err(bad("invalid primary or object exceeds byte limit"));
        }
        let requirements: BTreeSet<_> = self.required_capabilities.iter().collect();
        if requirements.len() != self.required_capabilities.len()
            || requirements.iter().any(|r| !token(r))
        {
            return Err(bad("invalid or duplicate capability requirement"));
        }
        Ok(())
    }
}

/// Read one regular file, bounded even if it grows after the metadata check.
pub fn read_bounded(path: &Path, max_bytes: u64) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|e| Failure::input(format!("{}: {e}", path.display())))?;
    if !metadata.file_type().is_file() || metadata.len() > max_bytes {
        return Err(Failure::integrity(
            "expected bounded regular model file (no symlinks)",
        ));
    }
    let mut bytes = Vec::new();
    File::open(path)
        .and_then(|file| {
            file.take(max_bytes.saturating_add(1))
                .read_to_end(&mut bytes)
        })
        .map_err(|e| Failure::input(format!("{}: {e}", path.display())))?;
    if bytes.len() as u64 > max_bytes {
        return Err(Failure::integrity("model file grew beyond its byte limit"));
    }
    Ok(bytes)
}

mod calibration;
mod construction;
mod inference;

/// Create-only exports: a failed/repeated export cannot overwrite training data or a model.
pub fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let temporary = parent.join(format!(
        ".jope-write-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&temporary)
        .map_err(|e| Failure::output(format!("{}: {e}", temporary.display())))?;
    let result = file
        .write_all(bytes)
        .and_then(|()| file.sync_all())
        .and_then(|()| fs::hard_link(&temporary, path));
    // Ownership follows successful create_new; never remove an existing destination.
    let _ = fs::remove_file(&temporary);
    result.map_err(|e| Failure::output(format!("{}: {e}", path.display())))
}

// A failed atomic create may mean a concurrent writer committed exactly these bytes.
fn publish_identical(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Err(error) = write_new(path, bytes) {
        if read_bounded(path, bytes.len() as u64).is_ok_and(|existing| existing == bytes) {
            return Ok(());
        }
        return Err(error);
    }
    Ok(())
}

/// Backwards-compatible entry point for inference: a standalone `.jope` or a committed object.
pub fn load_fitted(path: &Path, max_bytes: u64) -> Result<FittedModel> {
    if path.is_dir() {
        ModelObject::read(path, max_bytes)?.fitted()
    } else {
        FittedModel::from_bytes(&read_bounded(path, max_bytes)?)
    }
}

#[cfg(test)]
mod tests;
