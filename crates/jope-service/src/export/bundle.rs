//! Deterministic export bundle contents and publication contract.
//!
//! ZIP encoding is kept behind `BundleWriter` so the assembly contract can be exercised
//! independently from the store and archive implementation.

use anyhow::{ensure, Context, Result};
use feat_math::{genome::GenomeIr, heads::FittedHead, model::Model};
use jope_api::Language;
use jope_runtime::model_object::ModelObject;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::{Cursor, Write},
};

pub const BUNDLE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleManifest {
    pub schema_version: u32,
    pub model_object_digest: String,
    pub retains_training_data: bool,
    /// Archive path to the lower-case SHA-256 digest of that entry's exact bytes.
    pub entries: BTreeMap<String, String>,
}

impl BundleManifest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == BUNDLE_SCHEMA_VERSION,
            "unsupported export bundle schema"
        );
        ensure!(
            is_sha256(&self.model_object_digest),
            "invalid model object digest"
        );
        ensure!(!self.entries.is_empty(), "export bundle has no entries");
        for (path, digest) in &self.entries {
            ensure!(safe_entry(path), "invalid export bundle path: {path}");
            ensure!(
                is_sha256(digest),
                "invalid digest for export bundle entry: {path}"
            );
        }
        ensure!(
            !self.entries.contains_key("manifest.json"),
            "manifest cannot hash itself"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bundle {
    /// BTreeMap order is the required archive entry order.
    pub entries: BTreeMap<String, Vec<u8>>,
    pub manifest: BundleManifest,
}

impl Bundle {
    fn validate(&self) -> Result<()> {
        self.manifest.validate()?;
        let manifest_bytes = self
            .entries
            .get("manifest.json")
            .context("bundle manifest missing")?;
        ensure!(
            manifest_bytes == &serde_json::to_vec(&self.manifest)?,
            "bundle manifest entry differs from validated manifest"
        );
        ensure!(
            self.entries.len() == self.manifest.entries.len() + 1,
            "bundle contains unlisted or missing entries"
        );
        for (path, digest) in &self.manifest.entries {
            let bytes = self
                .entries
                .get(path)
                .with_context(|| format!("bundle entry missing: {path}"))?;
            ensure!(
                sha256_hex(bytes) == *digest,
                "bundle entry digest mismatch: {path}"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LanguageFiles {
    pub predictor: Vec<u8>,
    /// Typescript adds a declaration beside the JavaScript predictor.
    pub declaration: Option<Vec<u8>>,
}

/// Build the sorted, hashed archive contents. Predictor bytes are produced by the fitted
/// runtime emitters before calling this function; keeping them as inputs makes unsupported
/// requested languages fail before any publisher is invoked.
pub fn assemble(
    model_object: &[u8],
    requested: &[Language],
    emitted: &BTreeMap<LanguageKey, LanguageFiles>,
    parity_receipt: &[u8],
    retains_training_data: bool,
) -> Result<Bundle> {
    ensure!(!model_object.is_empty(), "model object is empty");
    ensure!(!requested.is_empty(), "no export languages requested");
    serde_json::from_slice::<serde_json::Value>(parity_receipt)
        .context("parity receipt must be valid JSON")?;
    let mut unique = requested.iter().map(LanguageKey::from).collect::<Vec<_>>();
    unique.sort();
    unique.dedup();
    ensure!(
        unique.len() == requested.len(),
        "duplicate export language requested"
    );

    // Resolve every required input before constructing any publishable artifact.
    let selected = unique
        .iter()
        .map(|language| {
            let files = emitted.get(language).with_context(|| {
                format!("required language is unsupported: {}", language.as_str())
            })?;
            ensure!(
                !files.predictor.is_empty(),
                "empty predictor for {}",
                language.as_str()
            );
            if *language == LanguageKey::Typescript {
                ensure!(
                    files.declaration.as_ref().is_some_and(|d| !d.is_empty()),
                    "missing TypeScript declaration"
                );
            }
            Ok((*language, files))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut entries = BTreeMap::new();
    entries.insert("model.jope".into(), model_object.to_vec());
    entries.insert("parity_receipt.json".into(), parity_receipt.to_vec());
    for (language, files) in selected {
        let directory = language.as_str();
        let extension =
            if language == LanguageKey::Javascript || language == LanguageKey::Typescript {
                "mjs"
            } else if language == LanguageKey::Python {
                "py"
            } else {
                "rs"
            };
        entries.insert(
            format!("src/{directory}/predictor.{extension}"),
            files.predictor.clone(),
        );
        entries.insert(
            format!("src/{directory}/README.md"),
            usage_readme(language, retains_training_data).into_bytes(),
        );
        if let Some(declaration) = &files.declaration {
            entries.insert("src/typescript/predictor.d.ts".into(), declaration.clone());
        }
    }

    let hashes = entries
        .iter()
        .map(|(path, bytes)| (path.clone(), sha256_hex(bytes)))
        .collect();
    let manifest = BundleManifest {
        schema_version: BUNDLE_SCHEMA_VERSION,
        model_object_digest: sha256_hex(model_object),
        retains_training_data,
        entries: hashes,
    };
    manifest.validate()?;
    entries.insert("manifest.json".into(), serde_json::to_vec(&manifest)?);
    Ok(Bundle { entries, manifest })
}

/// Writer owns publication of the completed ZIP archive. Tests use an in-memory stub.
pub trait BundleWriter {
    type Published;
    fn publish(&mut self, archive: &[u8]) -> Result<Self::Published>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublishedBundle {
    pub blob_id: String,
    pub sha256: String,
    pub size: u64,
}

pub struct StoreBundleWriter<'a> {
    pub store: &'a crate::store::Store,
}

impl BundleWriter for StoreBundleWriter<'_> {
    type Published = PublishedBundle;

    fn publish(&mut self, archive: &[u8]) -> Result<Self::Published> {
        let reference = self.store.publish_blob(Cursor::new(archive))?;
        let sha256: String = reference
            .sha256
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        Ok(PublishedBundle {
            blob_id: format!("sha256:{sha256}"),
            sha256,
            size: reference.bytes,
        })
    }
}

pub fn publish<W: BundleWriter>(bundle: &Bundle, writer: &mut W) -> Result<W::Published> {
    bundle.validate()?;
    writer.publish(&zip_bytes(bundle)?)
}

pub fn assemble_and_publish<W: BundleWriter>(
    model_object: &[u8],
    requested: &[Language],
    emitted: &BTreeMap<LanguageKey, LanguageFiles>,
    parity_receipt: &[u8],
    retains_training_data: bool,
    writer: &mut W,
) -> Result<W::Published> {
    let bundle = assemble(
        model_object,
        requested,
        emitted,
        parity_receipt,
        retains_training_data,
    )?;
    publish(&bundle, writer)
}

pub fn publish_stored_model(
    store: &crate::store::Store,
    model_blob: &crate::store::BlobRef,
    requested: &[Language],
    checks: &[ParityCheck],
    parity_receipt: &[u8],
) -> Result<PublishedBundle> {
    let model_bytes = store.read_blob(model_blob)?;
    let (emitted, retains_training_data) = emit_fitted_sources(&model_bytes, requested, checks)?;
    let bundle = assemble(
        &model_bytes,
        requested,
        &emitted,
        parity_receipt,
        retains_training_data,
    )?;
    publish(&bundle, &mut StoreBundleWriter { store })
}

#[derive(Clone, Debug)]
pub struct ParityCheck {
    pub inputs: Vec<f64>,
    pub expected: f64,
}

/// Emit supported requested languages from a fitted `model.jope` blob. Unknown bound-model
/// layouts refuse rather than emitting a predictor with different learned state.
pub fn emit_fitted_sources(
    model_jope: &[u8],
    requested: &[Language],
    checks: &[ParityCheck],
) -> Result<(BTreeMap<LanguageKey, LanguageFiles>, bool)> {
    let object = ModelObject::from_fitted_bytes(model_jope.to_vec())?;
    let fitted = object.fitted()?;
    let genome = GenomeIr::parse(&fitted.header.method.recipe)
        .context("fitted header recipe is not an exportable genome")?;
    let model = legacy_model(&fitted.model)?;
    let retains_training_data = retains_training_data(&model);
    let width = fitted.header.input_schema.len();
    let js_checks = checks
        .iter()
        .map(|row| jope_runtime::fitted_js::CheckRow {
            inputs: row.inputs.clone(),
            expected: row.expected,
        })
        .collect::<Vec<_>>();
    let py_checks = checks
        .iter()
        .map(|row| jope_runtime::fitted_py::CheckRow {
            inputs: row.inputs.clone(),
            expected: row.expected,
        })
        .collect::<Vec<_>>();
    let rs_checks = checks
        .iter()
        .map(|row| jope_runtime::fitted_rs::CheckRow {
            inputs: row.inputs.clone(),
            expected: row.expected,
        })
        .collect::<Vec<_>>();
    let mut emitted = BTreeMap::new();
    for language in requested.iter().map(LanguageKey::from) {
        let files = match language {
            LanguageKey::Javascript | LanguageKey::Typescript => {
                let predictor =
                    jope_runtime::fitted_js::emit_with_checks(&genome, &model, width, &js_checks)?;
                LanguageFiles {
                    predictor: predictor.into_bytes(),
                    declaration: if language == LanguageKey::Typescript {
                        Some(crate::export::dts::generate(&fitted.header)?.into_bytes())
                    } else {
                        None
                    },
                }
            }
            LanguageKey::Python => LanguageFiles {
                predictor: jope_runtime::fitted_py::emit_with_checks(
                    &genome, &model, width, &py_checks,
                )?
                .into_bytes(),
                declaration: None,
            },
            LanguageKey::Rust => LanguageFiles {
                predictor: jope_runtime::fitted_rs::emit_with_checks(
                    &genome, &model, width, &rs_checks,
                )?
                .into_bytes(),
                declaration: None,
            },
        };
        emitted.insert(language, files);
    }
    Ok((emitted, retains_training_data))
}

fn legacy_model(bound: &feat_math::advanced_v3::BoundModel) -> Result<Model> {
    let state = serde_json::to_value(bound)?;
    let legacy = state
        .get("predictor")
        .and_then(|predictor| predictor.get("Legacy"))
        .cloned()
        .context("fitted model is outside the fitted source emitter contract")?;
    serde_json::from_value(legacy).context("decode retained legacy fitted model")
}

fn retains_training_data(model: &Model) -> bool {
    let Model::Pipeline { head, .. } = model else {
        return false;
    };
    matches!(
        head.as_ref(),
        FittedHead::Krr(_) | FittedHead::Gpr(_) | FittedHead::GprKernel(_) | FittedHead::Ksmooth(_)
    )
}

fn zip_bytes(bundle: &Bundle) -> Result<Vec<u8>> {
    let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let timestamp = zip::DateTime::from_date_and_time(1980, 1, 1, 0, 0, 0)?;
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated)
        .compression_level(Some(6))
        .last_modified_time(timestamp);
    for (path, bytes) in &bundle.entries {
        writer.start_file(path, options)?;
        writer.write_all(bytes)?;
    }
    Ok(writer.finish()?.into_inner())
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LanguageKey {
    Javascript,
    Python,
    Rust,
    Typescript,
}

impl From<&Language> for LanguageKey {
    fn from(language: &Language) -> Self {
        match language {
            Language::Javascript => Self::Javascript,
            Language::Python => Self::Python,
            Language::Rust => Self::Rust,
            Language::Typescript => Self::Typescript,
        }
    }
}

impl LanguageKey {
    fn as_str(self) -> &'static str {
        match self {
            Self::Javascript => "javascript",
            Self::Python => "python",
            Self::Rust => "rust",
            Self::Typescript => "typescript",
        }
    }
}

fn usage_readme(language: LanguageKey, retains_training_data: bool) -> String {
    let notice = if retains_training_data {
        "\n**Privacy notice:** this fitted model retains training rows in its learned state. Treat the bundle as containing training data.\n"
    } else {
        ""
    };
    format!(
        "# {} predictor\n\nThis directory contains a standalone fitted predictor. Use `predictor.{}` with the prediction input contract for the bundled model.\n{notice}",
        language.as_str(),
        if language == LanguageKey::Javascript || language == LanguageKey::Typescript { "mjs" }
        else if language == LanguageKey::Python { "py" } else { "rs" },
    )
}

fn safe_entry(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
        && path
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._/-".contains(&b))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    Sha256::digest(bytes)
        .iter()
        .flat_map(|byte| {
            [
                HEX[(byte >> 4) as usize] as char,
                HEX[(byte & 0x0f) as usize] as char,
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sources() -> BTreeMap<LanguageKey, LanguageFiles> {
        BTreeMap::from([
            (
                LanguageKey::Javascript,
                LanguageFiles {
                    predictor: b"export function predict() {}".to_vec(),
                    declaration: None,
                },
            ),
            (
                LanguageKey::Python,
                LanguageFiles {
                    predictor: b"def predict(row): return 0".to_vec(),
                    declaration: None,
                },
            ),
            (
                LanguageKey::Rust,
                LanguageFiles {
                    predictor: b"pub fn predict() {}".to_vec(),
                    declaration: None,
                },
            ),
            (
                LanguageKey::Typescript,
                LanguageFiles {
                    predictor: b"export function predict() {}".to_vec(),
                    declaration: Some(b"export declare function predict(): number;".to_vec()),
                },
            ),
        ])
    }

    #[test]
    fn manifest_round_trips_and_hashes_every_file() {
        let requested = [Language::Javascript, Language::Typescript];
        let a = assemble(
            b"model object",
            &requested,
            &sources(),
            b"{\"checks\":[]}",
            false,
        )
        .unwrap();
        let b = assemble(
            b"model object",
            &requested,
            &sources(),
            b"{\"checks\":[]}",
            false,
        )
        .unwrap();
        assert_eq!(a.entries, b.entries);
        let archive = zip_bytes(&a).unwrap();
        assert_eq!(archive, zip_bytes(&b).unwrap());
        let mut zip = zip::ZipArchive::new(Cursor::new(archive)).unwrap();
        let names = (0..zip.len())
            .map(|index| zip.by_index(index).unwrap().name().unwrap().into_owned())
            .collect::<Vec<_>>();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
        for index in 0..zip.len() {
            let file = zip.by_index(index).unwrap();
            assert_eq!(
                file.last_modified(),
                Some(zip::DateTime::from_date_and_time(1980, 1, 1, 0, 0, 0).unwrap())
            );
            assert_eq!(file.compression(), zip::CompressionMethod::Deflated);
        }
        let parsed: BundleManifest = serde_json::from_slice(&a.entries["manifest.json"]).unwrap();
        parsed.validate().unwrap();
        assert_eq!(parsed, a.manifest);
        assert_eq!(parsed.entries["model.jope"], sha256_hex(b"model object"));
        assert!(a.entries.contains_key("src/typescript/predictor.d.ts"));
        assert!(a.entries.contains_key("src/typescript/README.md"));
    }

    #[test]
    fn unsupported_required_language_fails_before_writer_publication() {
        struct Stub {
            calls: usize,
        }
        impl BundleWriter for Stub {
            type Published = ();
            fn publish(&mut self, _: &[u8]) -> Result<()> {
                self.calls += 1;
                Ok(())
            }
        }
        let absent = BTreeMap::new();
        let mut writer = Stub { calls: 0 };
        let result = assemble_and_publish(
            b"model",
            &[Language::Rust],
            &absent,
            b"{}",
            false,
            &mut writer,
        );
        assert!(result.is_err());
        assert_eq!(writer.calls, 0);
    }

    #[test]
    fn store_publication_returns_content_identity_and_archive_bytes() {
        let root = tempfile::tempdir().unwrap();
        let store = crate::store::Store::open(root.path()).unwrap();
        let bundle =
            assemble(b"model object", &[Language::Rust], &sources(), b"{}", false).unwrap();
        let expected = zip_bytes(&bundle).unwrap();
        let published = publish(&bundle, &mut StoreBundleWriter { store: &store }).unwrap();
        assert_eq!(published.blob_id, format!("sha256:{}", published.sha256));
        assert_eq!(published.size, expected.len() as u64);
        let digest = (0..32)
            .map(|index| {
                u8::from_str_radix(&published.sha256[index * 2..index * 2 + 2], 16).unwrap()
            })
            .collect::<Vec<_>>();
        let reference = crate::store::BlobRef {
            sha256: digest.try_into().unwrap(),
            bytes: published.size,
        };
        assert_eq!(store.read_blob(&reference).unwrap(), expected);
    }

    #[test]
    fn manifest_rejects_traversal_and_bad_hashes() {
        let mut manifest = BundleManifest {
            schema_version: 1,
            model_object_digest: "a".repeat(64),
            retains_training_data: false,
            entries: BTreeMap::from([("../evil".into(), "b".repeat(64))]),
        };
        assert!(manifest.validate().is_err());
        manifest.entries = BTreeMap::from([("model.jope".into(), "bad".into())]);
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn retention_flag_tracks_training_row_heads() {
        let model = |head: &str, value: serde_json::Value| -> Model {
            serde_json::from_value(serde_json::json!({
                "Pipeline": {"target": null, "ops": [], "head": {head: value}}
            }))
            .unwrap()
        };
        let krr = serde_json::json!({"means": [], "scales": [], "basis": [], "alpha": [], "gamma": 0.0, "y_mean": 0.0});
        let gpr = serde_json::json!({"means": [], "scales": [], "basis": [], "alpha": [], "ls": [], "sf2": 0.0, "y_mean": 0.0});
        let ksmooth = serde_json::json!({"means": [], "scales": [], "z": [], "ys": [], "k": 1, "gamma": 1.0, "y_mean": 0.0});
        let ridge = serde_json::json!({"means": [], "scales": [], "w": []});
        assert!(retains_training_data(&model("Krr", krr)));
        assert!(retains_training_data(&model("Gpr", gpr)));
        assert!(retains_training_data(&model("Ksmooth", ksmooth)));
        assert!(!retains_training_data(&model("Ridge", ridge)));
    }

    #[test]
    fn retained_legacy_bound_model_reaches_all_fitted_emitters() {
        let genome = GenomeIr::parse("(pipe X (ridge))").unwrap();
        let bound = feat_math::advanced_v3::fit_legacy_numeric_bound(
            &genome,
            &[vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
            &[0.0, 1.0, 2.0, 3.0],
        )
        .unwrap();
        let model = legacy_model(&bound).unwrap();
        assert!(!retains_training_data(&model));
        assert!(
            !jope_runtime::fitted_js::emit_with_checks(&genome, &model, 1, &[])
                .unwrap()
                .is_empty()
        );
        assert!(
            !jope_runtime::fitted_py::emit_with_checks(&genome, &model, 1, &[])
                .unwrap()
                .is_empty()
        );
        assert!(
            !jope_runtime::fitted_rs::emit_with_checks(&genome, &model, 1, &[])
                .unwrap()
                .is_empty()
        );
    }
}
