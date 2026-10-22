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

pub const SCHEMA: &str = "jope.model-object/v1";
pub const APPLIANCE: &str = "org.jopedime/appliance-fitted/v1";
pub const BOUND: &str = "org.jopedime/battle-artifact/v2";
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
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
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
                || part.path == MANIFEST
                || !names.insert(&part.path)
                || !hex_hash(&part.sha256)
                || part.media_type.is_empty()
            {
                return Err(bad("invalid or duplicate file descriptor"));
            }
            total = total
                .checked_add(part.bytes)
                .ok_or_else(|| bad("size overflow"))?;
        }
        if total > max_bytes || !names.contains(&self.primary) {
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

impl ModelObject {
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    pub fn manifest_bytes(&self) -> &[u8] {
        &self.manifest_bytes
    }
    pub fn parts(&self) -> &BTreeMap<String, Vec<u8>> {
        &self.parts
    }

    pub fn new(
        manifest: Manifest,
        parts: BTreeMap<String, Vec<u8>>,
        max_bytes: u64,
    ) -> Result<Self> {
        let bytes = serde_json::to_vec_pretty(&manifest)?;
        Self::decode(bytes, parts, max_bytes)
    }

    /// Identity is SHA-256 of the *exact* manifest bytes. Readers never reserialize JSON to
    /// derive it, so Python/JS/C++/Rust do not need a shared float/string formatter.
    pub fn decode(
        manifest_bytes: Vec<u8>,
        parts: BTreeMap<String, Vec<u8>>,
        max_bytes: u64,
    ) -> Result<Self> {
        if manifest_bytes.len() as u64 > MAX_MANIFEST_BYTES {
            return Err(Failure::integrity("manifest exceeds its byte limit"));
        }
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|e| Failure::integrity(format!("model manifest: {e}")))?;
        manifest.validate(max_bytes)?;
        if parts.len() != manifest.files.len() {
            return Err(Failure::integrity(
                "model object has missing or unlisted parts",
            ));
        }
        for part in &manifest.files {
            let bytes = parts
                .get(&part.path)
                .ok_or_else(|| Failure::integrity("missing part"))?;
            if bytes.len() as u64 != part.bytes || sha256_hex(bytes) != part.sha256 {
                return Err(Failure::integrity(format!(
                    "part {} checksum/length mismatch",
                    part.path
                )));
            }
        }
        Ok(Self {
            id: format!("sha256-{}", sha256_hex(&manifest_bytes)),
            manifest,
            manifest_bytes,
            parts,
        })
    }

    /// Wrap the entire native fitted file; no fit or parameter conversion occurs.
    pub fn from_fitted(model: &FittedModel) -> Result<Self> {
        let bytes = model.to_bytes()?;
        let manifest = Manifest {
            schema: SCHEMA.into(),
            minimum_reader_version: 1,
            format: APPLIANCE.into(),
            producer: Producer::default(),
            primary: "model.jope".into(),
            required_capabilities: vec![],
            files: vec![Part::new(
                "model.jope",
                "application/vnd.jopedime.fitted",
                &bytes,
            )],
            metadata: json!({"task": model.header.task, "target": model.header.target,
                "method": model.header.method.id, "input_schema": model.header.input_schema,
                "retained_training_context": model.header.retained_training_context,
                "source_sha40": model.header.created_by.source_sha40}),
        };
        Self::new(
            manifest,
            BTreeMap::from([("model.jope".into(), bytes)]),
            crate::limits::Limits::default().max_model_bytes,
        )
    }

    /// All current bound GP forms use the existing validated state codec. Future forms use
    /// new format/capability identifiers instead of silently changing this decoder.
    pub fn from_bound(model: &feat_math::advanced_v3::BoundModel) -> Result<Self> {
        model.validate_binding()?;
        let bytes = serde_json::to_vec(model)?;
        Self::new(
            Manifest {
                schema: SCHEMA.into(),
                minimum_reader_version: 1,
                format: BOUND.into(),
                producer: Producer::default(),
                primary: "state.json".into(),
                required_capabilities: vec![],
                files: vec![Part::new("state.json", "application/json", &bytes)],
                metadata: json!({"input_schema": model.schema, "task": model.task}),
            },
            BTreeMap::from([("state.json".into(), bytes)]),
            crate::limits::Limits::default().max_model_bytes,
        )
    }

    fn unsupported(&self) -> Vec<String> {
        let mut out = self.manifest.required_capabilities.clone();
        if !matches!(self.manifest.format.as_str(), APPLIANCE | BOUND) {
            out.push(format!("decoder:{}", self.manifest.format));
        }
        out
    }

    pub fn fitted(&self) -> Result<FittedModel> {
        self.require_executable()?;
        if self.manifest.format != APPLIANCE {
            return Err(Failure::backend(
                "this command needs an appliance fitted object",
            ));
        }
        FittedModel::from_bytes(&self.parts[&self.manifest.primary])
    }

    pub fn bound(&self) -> Result<feat_math::advanced_v3::BoundModel> {
        self.require_executable()?;
        if self.manifest.format == APPLIANCE {
            return Ok(self.fitted()?.model);
        }
        let model: feat_math::advanced_v3::BoundModel =
            serde_json::from_slice(&self.parts[&self.manifest.primary])
                .map_err(|e| Failure::integrity(format!("bound state: {e}")))?;
        model
            .validate_binding()
            .map_err(|e| Failure::integrity(format!("bound state: {e:#}")))?;
        Ok(model)
    }

    /// Lower a loaded model using the existing portable emitter's explicit coverage gate.
    pub fn export_source(&self, language: &str) -> Result<Vec<u8>> {
        let model = self.bound()?;
        let reason = feat_math::advanced_v3::fitted_portable_unsupported(&model)
            .map_err(|e| Failure::integrity(e.to_string()))?;
        if let Some(reason) = reason {
            return Err(Failure::backend(format!(
                "portable source export unsupported: {reason:?}"
            )));
        }
        let sources = feat_math::advanced_v3::emit_fitted_portable(&model)?;
        let source = match language {
            "python" => sources.python,
            "rust" => sources.rust,
            "javascript" => sources
                .javascript
                .ok_or_else(|| Failure::backend("JavaScript export unsupported for this head"))?,
            _ => {
                return Err(Failure::input(
                    "source language must be python, rust or javascript",
                ))
            }
        };
        Ok(source.into_bytes())
    }

    fn require_executable(&self) -> Result<()> {
        let unsupported = self.unsupported();
        if !unsupported.is_empty() {
            return Err(Failure::backend(format!(
                "unsupported model capabilities: {}",
                unsupported.join(", ")
            )));
        }
        Ok(())
    }

    pub fn inspect(&self) -> Result<Inspection> {
        let unsupported = self.unsupported();
        if unsupported.is_empty() {
            self.bound()?;
        }
        Ok(Inspection {
            schema: SCHEMA,
            id: self.id.clone(),
            integrity_verified: true,
            executable: unsupported.is_empty(),
            unsupported_capabilities: unsupported,
            format: self.manifest.format.clone(),
            primary: self.manifest.primary.clone(),
            files: self.manifest.files.clone(),
            metadata: self.manifest.metadata.clone(),
        })
    }

    /// `manifest.json` is written last using an atomic link. A failed or interrupted publisher
    /// leaves an uncommitted directory which readers refuse; it never damages another object.
    pub fn save(&self, root: &Path, max_bytes: u64) -> Result<PathBuf> {
        self.manifest.validate(max_bytes)?;
        fs::create_dir_all(root)
            .map_err(|e| Failure::output(format!("{}: {e}", root.display())))?;
        let destination = root.join(&self.id);
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        match builder.create(&destination) {
            Ok(()) => (),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let existing = Self::read(&destination, max_bytes)?;
                if existing.manifest_bytes != self.manifest_bytes {
                    return Err(Failure::integrity("existing object differs"));
                }
                return Ok(destination);
            }
            Err(e) => return Err(Failure::output(format!("{}: {e}", destination.display()))),
        }
        for (name, data) in &self.parts {
            write_new(&destination.join(name), data)?;
        }
        write_new(&destination.join(".manifest.pending"), &self.manifest_bytes)?;
        fs::hard_link(
            destination.join(".manifest.pending"),
            destination.join(MANIFEST),
        )
        .map_err(|e| Failure::output(format!("commit model manifest: {e}")))?;
        fs::remove_file(destination.join(".manifest.pending"))
            .map_err(|e| Failure::output(e.to_string()))?;
        #[cfg(unix)]
        {
            File::open(&destination)
                .and_then(|f| f.sync_all())
                .map_err(|e| Failure::output(e.to_string()))?;
        }
        #[cfg(unix)]
        {
            File::open(root)
                .and_then(|f| f.sync_all())
                .map_err(|e| Failure::output(e.to_string()))?;
        }
        Ok(destination)
    }

    pub fn read(path: &Path, max_bytes: u64) -> Result<Self> {
        if !fs::symlink_metadata(path)
            .map_err(|e| Failure::input(e.to_string()))?
            .file_type()
            .is_dir()
        {
            return Err(Failure::integrity(
                "model object must be a directory (no symlinks)",
            ));
        }
        let manifest_bytes = read_bounded(&path.join(MANIFEST), MAX_MANIFEST_BYTES)?;
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|e| Failure::integrity(format!("model manifest: {e}")))?;
        manifest.validate(max_bytes)?;
        let id = format!("sha256-{}", sha256_hex(&manifest_bytes));
        if path.file_name().and_then(|n| n.to_str()) != Some(&id) {
            return Err(Failure::integrity(
                "object directory does not match manifest identity",
            ));
        }
        let mut parts = BTreeMap::new();
        for part in &manifest.files {
            parts.insert(
                part.path.clone(),
                read_bounded(&path.join(&part.path), part.bytes)?,
            );
        }
        Self::decode(manifest_bytes, parts, max_bytes)
    }
}

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
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
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

/// Backwards-compatible entry point for inference: a standalone `.jope` or a committed object.
pub fn load_fitted(path: &Path, max_bytes: u64) -> Result<FittedModel> {
    if path.is_dir() {
        ModelObject::read(path, max_bytes)?.fitted()
    } else {
        FittedModel::from_bytes(&read_bounded(path, max_bytes)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn future() -> ModelObject {
        let state = b"{\"variant\":\"future\",\"operators\":[\"org.example/solver/v7\"]}".to_vec();
        let buffer = vec![0, 255, 0, 128, 1, 7];
        ModelObject::new(Manifest {
            schema: SCHEMA.into(), minimum_reader_version: 1,
            format: "org.example/future-gp/v7".into(), producer: Producer::default(),
            primary: "state.json".into(), required_capabilities: vec!["org.example/solver/v7".into()],
            files: vec![Part::new("state.json", "application/json", &state), Part::new("weights.bin", "application/octet-stream", &buffer)],
            metadata: json!({"extension": {"unicode": "λ 日本", "float_bits": "8000000000000000"}}),
        }, BTreeMap::from([("state.json".into(), state), ("weights.bin".into(), buffer)]), 1024).unwrap()
    }

    #[test]
    fn future_variants_retain_all_bytes_but_require_their_decoder() {
        let object = future();
        let root = tempfile::tempdir().unwrap();
        let path = object.save(root.path(), 1024).unwrap();
        let loaded = ModelObject::read(&path, 1024).unwrap();
        assert_eq!(object.manifest_bytes, loaded.manifest_bytes);
        assert_eq!(object.parts, loaded.parts);
        let inspection = loaded.inspect().unwrap();
        assert!(inspection.integrity_verified);
        assert!(!inspection.executable);
        assert_eq!(inspection.unsupported_capabilities.len(), 2);
        assert_eq!(
            crate::error::kind_of(&loaded.bound().unwrap_err()),
            crate::FailureKind::Backend
        );
        assert_eq!(object.save(root.path(), 1024).unwrap(), path);
    }

    #[test]
    fn manifest_identity_uses_retained_bytes_including_foreign_whitespace() {
        let object = future();
        let mut bytes = b" \n".to_vec();
        bytes.extend_from_slice(&serde_json::to_vec(&object.manifest).unwrap());
        bytes.push(b'\n');
        let foreign = ModelObject::decode(bytes.clone(), object.parts.clone(), 1024).unwrap();
        assert_ne!(foreign.id, object.id);
        assert_eq!(foreign.id, format!("sha256-{}", sha256_hex(&bytes)));
        let root = tempfile::tempdir().unwrap();
        let path = foreign.save(root.path(), 1024).unwrap();
        assert_eq!(
            ModelObject::read(&path, 1024).unwrap().manifest_bytes,
            bytes
        );
    }

    #[test]
    fn truncation_substitution_wrong_directory_and_uncommitted_store_are_refused() {
        let object = future();
        let root = tempfile::tempdir().unwrap();
        let path = object.save(root.path(), 1024).unwrap();
        fs::write(path.join("weights.bin"), [0, 255]).unwrap();
        assert!(ModelObject::read(&path, 1024).is_err());
        assert!(object.save(root.path(), 1024).is_err());
        assert_eq!(fs::read(path.join("weights.bin")).unwrap(), [0, 255]);
        let root = tempfile::tempdir().unwrap();
        let path = object.save(root.path(), 1024).unwrap();
        let renamed = root.path().join(format!("sha256-{}", "0".repeat(64)));
        fs::rename(&path, &renamed).unwrap();
        assert!(ModelObject::read(&renamed, 1024).is_err());
        fs::remove_file(renamed.join(MANIFEST)).unwrap();
        assert!(ModelObject::read(&renamed, 1024).is_err());
    }

    #[test]
    fn envelope_bounds_and_paths_are_checked_before_part_access() {
        let object = future();
        for bad_name in [
            "../escape",
            "/abs",
            "x/y",
            "x\\y",
            ".hidden",
            "C:state",
            MANIFEST,
        ] {
            let mut manifest = object.manifest.clone();
            manifest.files[0].path = bad_name.into();
            assert!(manifest.validate(1024).is_err(), "{bad_name}");
        }
        let mut manifest = object.manifest.clone();
        manifest.minimum_reader_version = 2;
        assert!(manifest.validate(1024).is_err());
        manifest = object.manifest.clone();
        manifest.format = "/invalid/".into();
        assert!(manifest.validate(1024).is_err());
        manifest = object.manifest.clone();
        manifest.files.push(manifest.files[0].clone());
        assert!(manifest.validate(1024).is_err());
        manifest = object.manifest.clone();
        manifest
            .required_capabilities
            .push(manifest.required_capabilities[0].clone());
        assert!(manifest.validate(1024).is_err());
        manifest = object.manifest.clone();
        manifest.files[0].bytes = u64::MAX;
        assert!(manifest.validate(u64::MAX).is_err());
        assert!(object.manifest.validate(2).is_err());
        assert!(ModelObject::decode(
            vec![b' '; MAX_MANIFEST_BYTES as usize + 1],
            BTreeMap::new(),
            1024
        )
        .is_err());
        let mut parts = object.parts.clone();
        parts.insert("unlisted.bin".into(), vec![]);
        assert!(ModelObject::decode(object.manifest_bytes.clone(), parts, 1024).is_err());
    }

    #[test]
    fn unknown_required_capabilities_are_never_ignored_for_a_known_format() {
        let object = future();
        let mut manifest = object.manifest.clone();
        manifest.format = APPLIANCE.into();
        let object = ModelObject::new(manifest, object.parts, 1024).unwrap();
        assert!(!object.inspect().unwrap().executable);
        assert!(object.fitted().is_err());
    }

    #[test]
    fn atomic_create_only_exports_preserve_existing_files() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("export.bin");
        write_new(&path, b"original").unwrap();
        assert!(write_new(&path, b"replacement").is_err());
        assert_eq!(fs::read(path).unwrap(), b"original");
        assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_parts_and_object_directories_are_refused() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let object = future();
        let root = tempfile::tempdir().unwrap();
        let path = object.save(root.path(), 1024).unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o700
        );
        let alias = root.path().join("alias");
        symlink(&path, &alias).unwrap();
        assert!(ModelObject::read(&alias, 1024).is_err());
        let payload = path.join("weights.bin");
        fs::rename(&payload, root.path().join("external.bin")).unwrap();
        symlink(root.path().join("external.bin"), &payload).unwrap();
        assert!(ModelObject::read(&path, 1024).is_err());
    }
}
