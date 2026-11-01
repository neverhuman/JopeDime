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
        let manifest: Manifest = crate::strict_json::decode(&manifest_bytes)
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
        Self::from_fitted_bytes(model.to_bytes()?)
    }

    /// Import an already serialized fitted file without normalizing its JSON state bytes.
    pub fn from_fitted_bytes(bytes: Vec<u8>) -> Result<Self> {
        let model = FittedModel::from_bytes(&bytes)?;
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
        let mut out: Vec<_> = self
            .manifest
            .required_capabilities
            .iter()
            .filter(|c| c.as_str() != calibration_platt::CAPABILITY)
            .cloned()
            .collect();
        if !matches!(
            self.manifest.format.as_str(),
            APPLIANCE | BOUND | MATH_PIPELINE
        ) {
            out.push(format!("decoder:{}", self.manifest.format));
        }
        out
    }

    /// A versioned required section: neither its bytes nor its capability may be omitted.
    pub fn platt_calibration(&self) -> Result<Option<calibration_platt::Calibration>> {
        let required = self
            .manifest
            .required_capabilities
            .iter()
            .any(|c| c == calibration_platt::CAPABILITY);
        match (required, self.parts.get(calibration_platt::PART)) {
            (false, None) => Ok(None),
            (true, Some(bytes)) => {
                if bytes.len() > 64 * 1024 {
                    return Err(Failure::integrity("Platt section exceeds 64 KiB"));
                }
                let calibration: calibration_platt::Calibration = crate::strict_json::decode(bytes)
                    .map_err(|e| Failure::integrity(format!("Platt section: {e}")))?;
                calibration
                    .validate()
                    .map_err(|e| Failure::integrity(format!("Platt section: {e:#}")))?;
                Ok(Some(calibration))
            }
            _ => Err(Failure::integrity(
                "Platt section and required capability must occur together",
            )),
        }
    }

    fn platt_head(
        &self,
        calibration: &calibration_platt::Calibration,
    ) -> Result<feat_math::advanced_v3::BoundModel> {
        let model = match self.manifest.format.as_str() {
            APPLIANCE => {
                let fitted = FittedModel::from_bytes(&self.parts[&self.manifest.primary])?;
                if fitted.header.class_order != calibration.class_order {
                    return Err(Failure::integrity(
                        "Platt class order differs from fitted head",
                    ));
                }
                fitted.model
            }
            BOUND => crate::strict_json::decode(&self.parts[&self.manifest.primary])
                .map_err(|e| Failure::integrity(format!("bound state: {e}")))?,
            _ => {
                return Err(Failure::integrity(
                    "Platt section requires an appliance or bound binary head",
                ))
            }
        };
        model
            .validate_binding()
            .map_err(|e| Failure::integrity(format!("Platt head: {e:#}")))?;
        if model.task.kind != feat_math::advanced_v3::TaskKind::Binary {
            return Err(Failure::integrity(
                "Platt section is only valid for binary heads",
            ));
        }
        Ok(model)
    }

    /// Attach a final map learned from out-of-fold scores, preserving every native state byte.
    /// For bare bound heads, `calibration.class_order` declares the native IDs' 0/1 label order.
    pub fn with_platt_calibration(
        &self,
        calibration: calibration_platt::Calibration,
    ) -> Result<Self> {
        self.require_executable()?;
        if self.platt_calibration()?.is_some() {
            return Err(Failure::input(
                "model object already has a Platt calibration",
            ));
        }
        calibration.validate()?;
        self.platt_head(&calibration)?;
        let bytes = serde_json::to_vec(&calibration)?;
        let mut manifest = self.manifest.clone();
        manifest
            .required_capabilities
            .push(calibration_platt::CAPABILITY.into());
        manifest.files.push(Part::new(
            calibration_platt::PART,
            "application/json",
            &bytes,
        ));
        let mut parts = self.parts.clone();
        parts.insert(calibration_platt::PART.into(), bytes);
        Self::new(
            manifest,
            parts,
            crate::limits::Limits::default().max_model_bytes,
        )
    }

    /// The input contract of a Platt-calibrated object, or `None` when it has no calibration.
    /// It exposes no executable head, so prediction still goes through [`Self::predict`].
    pub fn calibrated_input(&self) -> Result<Option<CalibratedInput>> {
        let Some(calibration) = self.platt_calibration()? else {
            return Ok(None);
        };
        self.require_executable()?;
        let head = self.platt_head(&calibration)?;
        let header = match self.manifest.format.as_str() {
            APPLIANCE => Some(FittedModel::from_bytes(&self.parts[&self.manifest.primary])?.header),
            _ => None,
        };
        Ok(Some(CalibratedInput {
            schema: head.schema,
            genome_sha256: head.genome_sha256,
            header,
        }))
    }

    /// Predict through the complete object. Calibrated binary rows retain explicit class order.
    /// Math pipelines keep their separate numeric-input API, and a bare uncalibrated binary
    /// bound object lacks class names; wrap it in an appliance object or attach its calibration.
    pub fn predict(
        &self,
        table: &feat_math::advanced_v3::RawTable,
    ) -> Result<crate::methods::native::Predictions> {
        use crate::methods::native::{predict_bound, Predictions};
        self.require_executable()?;
        if let Some(calibration) = self.platt_calibration()? {
            let head = self.platt_head(&calibration)?;
            let Predictions::Probabilities { classes, values } =
                predict_bound(&head, table, &calibration.class_order)?
            else {
                return Err(Failure::integrity(
                    "Platt head did not return binary probabilities",
                ));
            };
            let values = values
                .into_iter()
                .map(|row| {
                    let p = calibration.sigmoid.apply(row[1])?;
                    Ok(vec![1.0 - p, p])
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(Predictions::Probabilities { classes, values });
        }
        match self.manifest.format.as_str() {
            APPLIANCE => self.fitted()?.predict(table),
            BOUND => predict_bound(&self.bound()?, table, &[]),
            _ => Err(Failure::backend(
                "use the registered decoder's prediction API for this object format",
            )),
        }
    }

    pub fn fitted(&self) -> Result<FittedModel> {
        self.require_executable()?;
        if self.platt_calibration()?.is_some() {
            return Err(Failure::backend("standalone fitted export would drop Platt calibration; use the complete object and ModelObject::predict"));
        }
        if self.manifest.format != APPLIANCE {
            return Err(Failure::backend(
                "this command needs an appliance fitted object",
            ));
        }
        FittedModel::from_bytes(&self.parts[&self.manifest.primary])
    }

    pub fn bound(&self) -> Result<feat_math::advanced_v3::BoundModel> {
        self.require_executable()?;
        if self.platt_calibration()?.is_some() {
            return Err(Failure::backend(
                "bound/source export would drop Platt calibration; export the complete object",
            ));
        }
        if self.manifest.format == MATH_PIPELINE {
            return Err(Failure::backend(
                "a math-pipeline object has no bound state; export object or javascript instead",
            ));
        }
        if self.manifest.format == APPLIANCE {
            let fitted = self.fitted()?;
            if fitted
                .header
                .output_transform
                .is_some_and(|map| !map.is_identity())
            {
                return Err(Failure::backend(
                    "bound/source export cannot preserve this model's output calibration; export object or fitted state instead",
                ));
            }
            return Ok(fitted.model);
        }
        let model: feat_math::advanced_v3::BoundModel =
            crate::strict_json::decode(&self.parts[&self.manifest.primary])
                .map_err(|e| Failure::integrity(format!("bound state: {e}")))?;
        model
            .validate_binding()
            .map_err(|e| Failure::integrity(format!("bound state: {e:#}")))?;
        Ok(model)
    }

    /// Decode a `math-pipeline/v1` object: a fitted feat-math pipeline behind a numeric input
    /// contract. See [`crate::math_pipeline`].
    pub fn math_pipeline(&self) -> Result<crate::math_pipeline::MathPipeline> {
        self.require_executable()?;
        if self.platt_calibration()?.is_some() {
            return Err(Failure::backend(
                "math-pipeline decode/source export cannot preserve Platt calibration; export the complete object",
            ));
        }
        if self.manifest.format != MATH_PIPELINE {
            return Err(Failure::backend(
                "this command needs a math-pipeline object",
            ));
        }
        crate::math_pipeline::MathPipeline::decode(&self.parts, &self.manifest.primary)
    }

    /// Lower a loaded model using the existing portable emitter's explicit coverage gate.
    pub fn export_source(&self, language: &str) -> Result<Vec<u8>> {
        if self.manifest.format == MATH_PIPELINE {
            let pipeline = self.math_pipeline()?;
            return match language {
                "javascript" => pipeline.export_javascript(),
                "rust" => crate::fitted_rs::emit_pipeline(&pipeline)
                    .map(String::into_bytes)
                    .map_err(|error| Failure::backend(error.to_string())),
                "python" => crate::fitted_py::emit_math_pipeline(&pipeline),
                _ => Err(Failure::input(
                    "source language must be python, rust or javascript",
                )),
            };
        }
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

    /// Verify the complete executable state without reducing an appliance to its native model.
    pub fn verify_executable(&self) -> Result<()> {
        self.require_executable()?;
        if let Some(calibration) = self.platt_calibration()? {
            self.platt_head(&calibration)?;
        } else if self.manifest.format == APPLIANCE {
            self.fitted()?;
        } else if self.manifest.format == MATH_PIPELINE {
            self.math_pipeline()?;
        } else {
            self.bound()?;
        }
        Ok(())
    }

    pub fn inspect(&self) -> Result<Inspection> {
        let unsupported = self.unsupported();
        if unsupported.is_empty() {
            self.verify_executable()?;
        }
        Ok(Inspection {
            schema: INSPECTION_SCHEMA,
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
        // Decode known state before any filesystem mutation. Unsupported codecs remain opaque.
        self.inspect()?;
        if root.to_str().is_none() {
            return Err(Failure::input("model root must be UTF-8"));
        }
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
                if !fs::symlink_metadata(&destination)?.file_type().is_dir() {
                    return Err(Failure::integrity(
                        "object destination is not a regular directory",
                    ));
                }
                if fs::symlink_metadata(destination.join(MANIFEST)).is_ok() {
                    let existing = Self::read(&destination, max_bytes)?;
                    if existing.manifest_bytes != self.manifest_bytes {
                        return Err(Failure::integrity("existing object differs"));
                    }
                    return Ok(destination);
                }
            }
            Err(e) => return Err(Failure::output(format!("{}: {e}", destination.display()))),
        }
        // Resume an interrupted publication only when every retained declared part agrees.
        // Never overwrite or reclaim state; a concurrent identical publisher is safe.
        for (name, data) in &self.parts {
            if let Ok(metadata) = fs::symlink_metadata(destination.join(name)) {
                if !metadata.file_type().is_file()
                    || read_bounded(&destination.join(name), data.len() as u64)? != *data
                {
                    return Err(Failure::integrity(
                        "incomplete object contains conflicting state",
                    ));
                }
            }
        }
        for (name, data) in &self.parts {
            publish_identical(&destination.join(name), data)?;
        }
        publish_identical(&destination.join(MANIFEST), &self.manifest_bytes)?;
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
        let manifest: Manifest = crate::strict_json::decode(&manifest_bytes)
            .map_err(|e| Failure::integrity(format!("model manifest: {e}")))?;
        manifest.validate(max_bytes)?;
        let id = format!("sha256-{}", sha256_hex(&manifest_bytes));
        let resolved = fs::canonicalize(path).map_err(|e| Failure::input(e.to_string()))?;
        if resolved.file_name().and_then(|n| n.to_str()) != Some(&id) {
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
mod tests {
    use super::*;

    #[test]
    fn math_pipeline_decode_and_javascript_export_refuse_a_missing_required_platt_part() {
        use crate::math_pipeline::{INPUT_PART, PIPELINE_PART};

        let parts: BTreeMap<String, Vec<u8>> = BTreeMap::from([
            (
                PIPELINE_PART.into(),
                include_bytes!("../tests/fixtures/math-pipeline/pipeline.json").to_vec(),
            ),
            (
                INPUT_PART.into(),
                include_bytes!("../tests/fixtures/math-pipeline/input.json").to_vec(),
            ),
        ]);
        let mut manifest = Manifest {
            schema: SCHEMA.into(),
            minimum_reader_version: 1,
            format: MATH_PIPELINE.into(),
            producer: Producer::default(),
            primary: PIPELINE_PART.into(),
            required_capabilities: vec![],
            files: parts
                .iter()
                .map(|(path, bytes)| Part::new(path, "application/json", bytes))
                .collect(),
            metadata: Value::Null,
        };
        let legacy = ModelObject::new(manifest.clone(), parts.clone(), 1 << 20).unwrap();
        assert!(legacy.inspect().unwrap().executable);
        legacy.verify_executable().unwrap();
        legacy.math_pipeline().unwrap();
        assert!(!legacy.export_source("javascript").unwrap().is_empty());

        manifest
            .required_capabilities
            .push(calibration_platt::CAPABILITY.into());
        let invalid = ModelObject::new(manifest, parts, 1 << 20).unwrap();
        assert!(invalid.platt_calibration().is_err());
        assert!(invalid.inspect().is_err());
        assert!(invalid.verify_executable().is_err());
        let decoded = invalid.math_pipeline();
        let exported = invalid.export_source("javascript");
        let rust_exported = invalid.export_source("rust");
        assert!(
            decoded.is_err() && exported.is_err() && rust_exported.is_err(),
            "missing required Platt part accepted: math_pipeline={}, javascript_export={}, rust_export={}",
            decoded.is_ok(),
            exported.is_ok(),
            rust_exported.is_ok()
        );
        for error in [
            decoded.unwrap_err(),
            exported.unwrap_err(),
            rust_exported.unwrap_err(),
        ] {
            assert_eq!(crate::error::kind_of(&error), crate::FailureKind::Integrity);
            assert!(error.to_string().contains("must occur together"));
        }
    }

    #[test]
    fn loaded_math_pipeline_object_exports_rust_and_legacy_version_is_refused() {
        use crate::math_pipeline::{INPUT_PART, PIPELINE_PART};
        use std::{fs, process::Command};

        fn fixture_object(input: Vec<u8>) -> ModelObject {
            let parts = BTreeMap::from([
                (
                    PIPELINE_PART.to_owned(),
                    include_bytes!("../tests/fixtures/math-pipeline/pipeline.json").to_vec(),
                ),
                (INPUT_PART.to_owned(), input),
            ]);
            let manifest = Manifest {
                schema: SCHEMA.into(),
                minimum_reader_version: 1,
                format: MATH_PIPELINE.into(),
                producer: Producer::default(),
                primary: PIPELINE_PART.into(),
                required_capabilities: vec![],
                files: parts
                    .iter()
                    .map(|(path, bytes)| Part::new(path, "application/json", bytes))
                    .collect(),
                metadata: Value::Null,
            };
            ModelObject::new(manifest, parts, 1 << 20).unwrap()
        }

        let mut input: serde_json::Value =
            serde_json::from_slice(include_bytes!("../tests/fixtures/math-pipeline/input.json"))
                .unwrap();
        input["prediction_version"] = serde_json::json!(2);
        let object = fixture_object(serde_json::to_vec(&input).unwrap());
        let root = tempfile::tempdir().unwrap();
        let path = object.save(root.path(), 1 << 20).unwrap();
        let loaded = ModelObject::read(&path, 1 << 20).unwrap();
        let source = loaded.export_source("rust").unwrap();
        let source = String::from_utf8(source).expect("Rust export is UTF-8");
        let pipeline = loaded.math_pipeline().unwrap();
        let table = crate::table::parse_csv(
            include_bytes!("../tests/fixtures/math-pipeline/rows.csv"),
            &crate::limits::Limits::default(),
        )
        .unwrap();
        let (expected, _) = pipeline.predict_table(&table).unwrap();
        let headers = format!(
            "[{}]",
            table
                .headers
                .iter()
                .map(|header| format!("{header:?}"))
                .collect::<Vec<_>>()
                .join(",")
        );
        let rows = format!(
            "vec![{}]",
            table
                .rows
                .iter()
                .map(|row| format!(
                    "vec![{}]",
                    row.iter()
                        .map(|cell| format!("{:?}", cell.as_deref()))
                        .collect::<Vec<_>>()
                        .join(",")
                ))
                .collect::<Vec<_>>()
                .join(",")
        );
        let expected = format!("vec!{:?}", expected);
        let program = format!(
            "{source}\nfn main() {{ let got = predict_table(&{headers}, &{rows}).unwrap(); let expected = {expected}; assert_eq!(got.len(), expected.len()); for (actual, expected) in got.iter().zip(expected) {{ assert!((actual - expected).abs() <= 1e-12 + 1e-10 * expected.abs()); }} }}"
        );
        let out_dir = std::env::current_dir()
            .unwrap()
            .join("target/exp-03-f01-rust-export");
        fs::create_dir_all(&out_dir).unwrap();
        let source_path = out_dir.join("fixture_export.rs");
        let binary_path = out_dir.join("fixture_export");
        fs::write(&source_path, program).unwrap();
        let compile = Command::new("rustc")
            .args(["--edition=2021", "-D", "warnings"])
            .arg(&source_path)
            .arg("-o")
            .arg(&binary_path)
            .output()
            .expect("run rustc for the exported module");
        assert!(
            compile.status.success(),
            "rustc failed: {}",
            String::from_utf8_lossy(&compile.stderr)
        );
        let run = Command::new(&binary_path)
            .output()
            .expect("run the exported module");
        assert!(
            run.status.success(),
            "exported Rust failed: {}",
            String::from_utf8_lossy(&run.stderr)
        );

        let legacy =
            fixture_object(include_bytes!("../tests/fixtures/math-pipeline/input.json").to_vec());
        let error = legacy.export_source("rust").unwrap_err();
        assert!(error.to_string().contains("prediction_version=2"));
    }

    fn binary_fixture() -> (FittedModel, feat_math::advanced_v3::RawTable) {
        use crate::{api, budget::Budget, fitted::CreatedBy, limits::Limits, table, task};
        let mut csv = String::from("x,y\n");
        for i in 0..40 {
            csv.push_str(&format!(
                "{},{}\n",
                i % 10,
                if i % 10 > 4 { "yes" } else { "no" }
            ));
        }
        let parsed = table::parse_csv(csv.as_bytes(), &Limits::default()).unwrap();
        let training = table::training(
            &parsed,
            "y",
            task::TaskChoice::Fixed(task::TaskKind::Binary),
            &Limits::default(),
        )
        .unwrap();
        let fitted = api::fit(
            &training,
            &api::FitOptions {
                method: "logistic".into(),
                ..Default::default()
            },
            &Budget::new(None, 1),
            &parsed.sha256,
            CreatedBy::default(),
        )
        .unwrap()
        .fitted;
        let query = table::parse_csv(b"x\n0\n3\n6\n9\n", &Limits::default()).unwrap();
        (
            fitted,
            table::prediction(&query, &training.schema, &training.target)
                .unwrap()
                .0,
        )
    }

    fn platt_fixture() -> calibration_platt::Calibration {
        let mut scores = Vec::new();
        let mut labels = Vec::new();
        let mut folds = Vec::new();
        for fold in 0..3 {
            for i in 0..20 {
                scores.push(if i < 10 { 0.001 } else { 0.999 });
                labels.push(if i % 10 < (if i < 10 { 2 } else { 8 }) {
                    "yes"
                } else {
                    "no"
                });
                folds.push(fold);
            }
        }
        calibration_platt::cross_fit(&scores, &labels, ["no", "yes"], &folds)
            .unwrap()
            .calibration
    }

    #[test]
    fn a_platt_part_without_its_required_capability_cannot_be_silently_stripped() {
        let (fitted, _) = binary_fixture();
        let object = ModelObject::from_fitted(&fitted).unwrap();
        let bytes = serde_json::to_vec(&platt_fixture()).unwrap();
        let mut manifest = object.manifest.clone();
        let mut parts = object.parts.clone();
        manifest.files.push(Part::new(
            calibration_platt::PART,
            "application/json",
            &bytes,
        ));
        parts.insert(calibration_platt::PART.into(), bytes);
        let object = ModelObject::new(manifest, parts, 1 << 20).unwrap();
        assert!(
            object.fitted().is_err(),
            "calibration part was discarded while exporting the bare fitted head"
        );
        assert!(object.bound().is_err());
    }

    #[test]
    fn platt_object_round_trip_predicts_with_the_stored_map_and_retains_native_bytes() {
        use crate::methods::native::Predictions;
        let (fitted, query) = binary_fixture();
        let raw = fitted.predict(&query).unwrap();
        let Predictions::Probabilities { classes, values } = &raw else {
            panic!("binary fixture");
        };
        let calibration = platt_fixture();
        let expected = Predictions::Probabilities {
            classes: classes.clone(),
            values: values
                .iter()
                .map(|row| {
                    let p = calibration.apply(row[1]).unwrap();
                    vec![1.0 - p, p]
                })
                .collect(),
        };
        assert_ne!(
            serde_json::to_vec(&raw).unwrap(),
            serde_json::to_vec(&expected).unwrap()
        );
        for object in [
            ModelObject::from_fitted(&fitted).unwrap(),
            ModelObject::from_bound(&fitted.model).unwrap(),
        ] {
            let before = object.parts.clone();
            let calibrated = object.with_platt_calibration(calibration.clone()).unwrap();
            for (name, bytes) in before {
                assert_eq!(calibrated.parts[&name], bytes);
            }
            assert_eq!(object.platt_calibration().unwrap(), None);
            assert_eq!(
                calibrated.manifest.required_capabilities,
                vec![calibration_platt::CAPABILITY]
            );
            assert_eq!(
                calibrated.platt_calibration().unwrap(),
                Some(calibration.clone())
            );
            assert!(calibrated.inspect().unwrap().executable);
            assert_eq!(
                serde_json::to_vec(&calibrated.predict(&query).unwrap()).unwrap(),
                serde_json::to_vec(&expected).unwrap()
            );
            assert!(calibrated.fitted().is_err());
            assert!(calibrated.bound().is_err());
            for language in ["python", "rust", "javascript"] {
                assert!(calibrated.export_source(language).is_err());
            }
            assert!(calibrated
                .with_platt_calibration(calibration.clone())
                .is_err());
            let directory = tempfile::tempdir().unwrap();
            let path = calibrated.save(directory.path(), 1 << 20).unwrap();
            let loaded = ModelObject::read(&path, 1 << 20).unwrap();
            assert_eq!(loaded.manifest_bytes, calibrated.manifest_bytes);
            assert_eq!(loaded.parts, calibrated.parts);
            assert_eq!(
                serde_json::to_vec(&loaded.predict(&query).unwrap()).unwrap(),
                serde_json::to_vec(&expected).unwrap()
            );
            assert!(load_fitted(&path, 1 << 20).is_err());
        }
        let legacy = ModelObject::from_fitted(&fitted).unwrap();
        assert_eq!(
            legacy.fitted().unwrap().to_bytes().unwrap(),
            fitted.to_bytes().unwrap()
        );
        assert_eq!(
            serde_json::to_vec(&legacy.predict(&query).unwrap()).unwrap(),
            serde_json::to_vec(&raw).unwrap()
        );
    }

    #[test]
    fn calibrated_input_reports_the_head_contract_without_exporting_it() {
        use crate::{limits::Limits, table};
        let (fitted, query) = binary_fixture();
        let calibration = platt_fixture();
        let legacy = ModelObject::from_fitted(&fitted).unwrap();
        assert!(legacy.calibrated_input().unwrap().is_none());
        let csv = table::parse_csv(b"x\n0\n3\n6\n9\n", &Limits::default()).unwrap();
        let labelled = table::parse_csv(b"x,y\n0,no\n", &Limits::default()).unwrap();
        for (object, appliance) in [
            (ModelObject::from_fitted(&fitted).unwrap(), true),
            (ModelObject::from_bound(&fitted.model).unwrap(), false),
        ] {
            let calibrated = object.with_platt_calibration(calibration.clone()).unwrap();
            let input = calibrated.calibrated_input().unwrap().unwrap();
            assert_eq!(input.schema, fitted.model.schema);
            assert_eq!(input.genome_sha256, fitted.model.genome_sha256);
            assert_eq!(input.header.as_ref(), appliance.then_some(&fitted.header));
            let (table, ignored) = input.table(&csv).unwrap();
            assert!(ignored.is_empty());
            assert_eq!(
                serde_json::to_vec(&calibrated.predict(&table).unwrap()).unwrap(),
                serde_json::to_vec(&calibrated.predict(&query).unwrap()).unwrap()
            );
            match input.table(&labelled) {
                Ok((_, ignored)) => assert!(appliance && ignored == ["y"]),
                Err(_) => assert!(!appliance, "an appliance head ignores its target"),
            }
        }
    }

    #[test]
    fn invalid_platt_sections_fail_before_store_mutation() {
        let (fitted, _) = binary_fixture();
        let object = ModelObject::from_fitted(&fitted).unwrap();
        let calibration = platt_fixture();
        let mut reordered = calibration.clone();
        reordered.class_order.swap(0, 1);
        assert!(object.with_platt_calibration(reordered).is_err());
        let mut negative = calibration.clone();
        negative.sigmoid.slope = -1.0;
        assert!(object.with_platt_calibration(negative).is_err());
        let calibrated = object.with_platt_calibration(calibration.clone()).unwrap();
        for value in [
            json!({"schema":"jope.platt-calibration/v2"}),
            json!({"unknown_required_field":true}),
            json!({"sigmoid":{"intercept":0.0,"slope":-1.0,"scale":1.0,"center":0.0}}),
            json!({"cross_fit_folds":1}),
            json!({"class_order":["yes","no"]}),
            json!({"score_kind":"raw-margin/v999"}),
        ] {
            let mut encoded = serde_json::to_value(&calibration).unwrap();
            for (key, value) in value.as_object().unwrap() {
                encoded[key] = value.clone();
            }
            let bytes = serde_json::to_vec(&encoded).unwrap();
            let mut manifest = calibrated.manifest.clone();
            *manifest
                .files
                .iter_mut()
                .find(|p| p.path == calibration_platt::PART)
                .unwrap() = Part::new(calibration_platt::PART, "application/json", &bytes);
            let mut parts = calibrated.parts.clone();
            parts.insert(calibration_platt::PART.into(), bytes);
            let invalid = ModelObject::new(manifest, parts, 1 << 20).unwrap();
            let directory = tempfile::tempdir().unwrap();
            let store = directory.path().join("store");
            assert!(invalid.save(&store, 1 << 20).is_err(), "accepted {encoded}");
            assert!(!store.exists());
        }
        let mut manifest = calibrated.manifest.clone();
        manifest.files.retain(|p| p.path != calibration_platt::PART);
        let mut parts = calibrated.parts.clone();
        parts.remove(calibration_platt::PART);
        let missing = ModelObject::new(manifest, parts, 1 << 20).unwrap();
        assert!(missing.inspect().is_err());
        let mut manifest = calibrated.manifest.clone();
        manifest.required_capabilities[0] = "org.jopedime/platt-calibration/v2".into();
        let future = ModelObject::new(manifest, calibrated.parts.clone(), 1 << 20).unwrap();
        assert!(!future.inspect().unwrap().executable);
        assert!(future.verify_executable().is_err());
        let genome = feat_math::GenomeIr::parse("(pipe X (ridge))").unwrap();
        let x: Vec<_> = (0..16).map(|i| vec![i as f64]).collect();
        let y: Vec<_> = (0..16).map(|i| i as f64).collect();
        let reg = feat_math::advanced_v3::fit_legacy_numeric_bound(&genome, &x, &y).unwrap();
        assert!(ModelObject::from_bound(&reg)
            .unwrap()
            .with_platt_calibration(calibration)
            .is_err());
    }

    #[test]
    fn portable_parts_refuse_device_names_trailing_dots_and_case_collisions() {
        let object = future();
        for name in [
            "NUL",
            "con.json",
            "COM1.bin",
            "lpt9",
            "AUX",
            "PRN.txt",
            "state.",
            "MANIFEST.JSON",
        ] {
            let mut manifest = object.manifest.clone();
            manifest.files[0].path = name.into();
            manifest.primary = name.into();
            assert!(
                manifest.validate(1024).is_err(),
                "accepted nonportable part {name}"
            );
        }
        let mut manifest = object.manifest.clone();
        manifest.files[1].path = "STATE.JSON".into();
        assert!(
            manifest.validate(1024).is_err(),
            "accepted case-insensitive collision"
        );
    }

    #[test]
    fn an_interrupted_publication_resumes_without_overwriting_retained_parts() {
        let object = future();
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join(object.id());
        fs::create_dir(&path).unwrap();
        fs::write(path.join("weights.bin"), &object.parts["weights.bin"]).unwrap();
        object.save(root.path(), 1024).unwrap();
        assert_eq!(ModelObject::read(&path, 1024).unwrap().parts, object.parts);
    }

    #[test]
    fn concurrent_identical_publication_and_conflicting_recovery_preserve_bytes() {
        let object = future();
        let root = tempfile::tempdir().unwrap();
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| scope.spawn(|| object.save(root.path(), 1024)))
                .collect();
            for handle in handles {
                handle.join().unwrap().unwrap();
            }
        });
        assert_eq!(
            ModelObject::read(&root.path().join(object.id()), 1024)
                .unwrap()
                .parts,
            object.parts
        );
        let damaged = tempfile::tempdir().unwrap();
        let path = damaged.path().join(object.id());
        fs::create_dir(&path).unwrap();
        fs::write(path.join("weights.bin"), b"retained conflicting bytes").unwrap();
        assert!(object.save(damaged.path(), 1024).is_err());
        assert_eq!(
            fs::read(path.join("weights.bin")).unwrap(),
            b"retained conflicting bytes"
        );
        assert!(!path.join(MANIFEST).exists());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let path = root.path().join(object.id());
            assert_eq!(
                fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o700
            );
            assert_eq!(
                fs::metadata(path.join(MANIFEST))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
    }

    #[test]
    fn malformed_known_state_is_refused_before_creating_a_store() {
        let mut manifest = future().manifest;
        let bytes = b"not a fitted file".to_vec();
        manifest.format = APPLIANCE.into();
        manifest.required_capabilities.clear();
        manifest.primary = "model.jope".into();
        manifest.files = vec![Part::new(
            "model.jope",
            "application/vnd.jopedime.fitted",
            &bytes,
        )];
        let object = ModelObject::new(
            manifest,
            BTreeMap::from([("model.jope".into(), bytes)]),
            1024,
        )
        .unwrap();
        let root = tempfile::tempdir().unwrap();
        let store = root.path().join("store");
        assert!(object.save(&store, 1024).is_err());
        assert!(!store.exists());
    }

    #[test]
    fn duplicate_json_metadata_is_refused_before_cross_language_disagreement() {
        let object = future();
        let text = String::from_utf8(object.manifest_bytes).unwrap();
        let duplicated = text.replace(
            "\"extension\": {",
            "\"extension\": {\"unicode\":\"other\"}, \"extension\": {",
        );
        assert!(ModelObject::decode(duplicated.into_bytes(), object.parts, 1024).is_err());
    }

    #[test]
    fn new_gp_fields_cannot_be_silently_dropped_by_an_older_decoder() {
        let genome = feat_math::GenomeIr::parse("(pipe X (ridge))").unwrap();
        let x = (0..16)
            .map(|i| vec![i as f64, (i % 3) as f64])
            .collect::<Vec<_>>();
        let y = (0..16).map(|i| i as f64 * 0.5).collect::<Vec<_>>();
        let fitted = feat_math::advanced_v3::fit_legacy_numeric_bound(&genome, &x, &y).unwrap();
        let mut value = serde_json::to_value(&fitted).unwrap();
        let head = value
            .pointer_mut("/predictor/Legacy/Pipeline/head/Ridge")
            .unwrap();
        head.as_object_mut()
            .unwrap()
            .insert("future_required_state".into(), json!({"offset": 123}));
        let bytes = serde_json::to_vec(&value).unwrap();
        let mut manifest = ModelObject::from_bound(&fitted).unwrap().manifest;
        manifest.files = vec![Part::new("state.json", "application/json", &bytes)];
        let object = ModelObject::new(
            manifest,
            BTreeMap::from([("state.json".into(), bytes)]),
            1 << 20,
        )
        .unwrap();
        assert!(
            object.bound().is_err(),
            "older decoder discarded required fitted state"
        );
    }

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
