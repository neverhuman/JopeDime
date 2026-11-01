//! Deterministic export bundle contents and publication contract.
//!
//! ZIP encoding is kept behind `BundleWriter` so the assembly contract can be exercised
//! independently from the store and archive implementation.

use anyhow::{ensure, Context, Result};
use feat_math::{genome::GenomeIr, heads::FittedHead, model::Model};
use jope_api::Language;
use jope_runtime::{fitted::Header, model_object::ModelObject};
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
            if language == LanguageKey::Typescript {
                // NodeNext pairs an .mjs module with a .d.mts declaration. Keep the
                // EXP-06 .d.ts artifact as well for direct consumers of the declaration.
                entries.insert("src/typescript/predictor.d.mts".into(), declaration.clone());
            }
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
                let mut predictor =
                    jope_runtime::fitted_js::emit_with_checks(&genome, &model, width, &js_checks)?;
                if language == LanguageKey::Typescript {
                    append_typescript_api(&mut predictor, &fitted.header)?;
                }
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

/// Add the named-row API declared beside the TypeScript bundle's fitted JavaScript module.
/// The underlying fitted runtime remains the authority for prediction math and width checks.
fn append_typescript_api(source: &mut String, header: &Header) -> Result<()> {
    let columns = serde_json::to_string(
        &header
            .input_schema
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>(),
    )?;
    source.push_str(&format!(
        r#"
// Named-row API paired with predictor.d.ts. Prediction math remains in the fitted runtime above.
const JOPE_PREDICTOR_COLUMNS={columns};
function jopePredictRows(rows){{
  if(!Array.isArray(rows))throw new TypeError("predictBatch expects an array of rows");
  const columns=JOPE_PREDICTOR_COLUMNS.length,flat=new Float64Array(rows.length*columns);
  for(let i=0;i<rows.length;i++){{
    const row=rows[i];
    if(row===null || typeof row!=="object" || Array.isArray(row))throw new TypeError("predict expects a row object");
    for(let j=0;j<columns;j++){{
      const name=JOPE_PREDICTOR_COLUMNS[j],value=row[name];
      // The fitted runtime is numeric: null is the missing sentinel, and anything else that is
      // not a number (a category string, a missing key) is refused rather than read as NaN.
      if(value!==null && typeof value!=="number")throw new TypeError(`column ${{JSON.stringify(name)}} must be a number or null; this fitted predictor takes numeric inputs only`);
      flat[i*columns+j]=value===null?Number.NaN:value;
    }}
  }}
  const output=new Float64Array(rows.length);
  const status=modelPredictNumericBatch(modelFittedHandle(),flat,flat.length,rows.length,columns,null,0,output,output.length);
  if(status!==0)throw new Error(`fitted predictor failed with status ${{status}}`);
  return Array.from(output);
}}
export function predict(row){{return jopePredictRows([row])[0];}}
export function predictBatch(rows){{return jopePredictRows(rows);}}
"#
    ));
    Ok(())
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

pub(crate) fn retains_training_data(model: &Model) -> bool {
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
    use serde_json::json;
    use std::process::Command;

    fn typescript_header() -> Header {
        serde_json::from_value(json!({
            "schema": "jope.appliance-fitted/v1",
            "minimum_reader_version": 1,
            "product": "jopedime",
            "product_version": "1.0.0",
            "runtime_abi": "jope.appliance-runtime/v1",
            "method": {"id":"ridge", "family":"linear", "configuration":"default", "recipe":"ridge", "implementation":"fixture"},
            "task":"regression", "metric":"rmse", "class_order":[], "target":"target",
            "input_schema":[
                {"name":"age", "kind":"numeric"},
                {"name":"region", "kind":"categorical"},
                {"name":"quote\"field", "kind":"numeric"}
            ],
            "preprocessing":{"raw_cell_encoding":"jope.raw-cell-encoding/v1", "input_policy":"named-columns"},
            "recipe_sha256":"00", "required_asset_hashes":[], "fitted_sections":[],
            "training_procedure":{"rows":1, "fitting_input_sha256":"00", "training_file_sha256":"00", "method_choice":"requested", "selection":null, "budget":{}, "final_refit":true},
            "retained_training_context":{"neural_support_rows":0, "note":"fixture"},
            "created_by":{"source_sha40":"0000000000", "package_manifest_sha256":null, "package_profile":null}
        }))
        .unwrap()
    }

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

    #[test]
    fn typescript_bundle_consumer_compiles_and_calls_named_predictors() {
        let Some(tsc) = std::env::var_os("JOPE_TSC") else {
            return;
        };
        let header = typescript_header();
        let genome = GenomeIr::parse("(pipe X (ridge))").unwrap();
        let bound = feat_math::advanced_v3::fit_legacy_numeric_bound(
            &genome,
            &[
                vec![0.0, 0.0, 0.0],
                vec![1.0, 1.0, 1.0],
                vec![2.0, 2.0, 2.0],
            ],
            &[0.0, 1.0, 2.0],
        )
        .unwrap();
        let model = legacy_model(&bound).unwrap();
        let mut predictor =
            jope_runtime::fitted_js::emit_with_checks(&genome, &model, 3, &[]).unwrap();
        append_typescript_api(&mut predictor, &header).unwrap();
        let emitted = BTreeMap::from([(
            LanguageKey::Typescript,
            LanguageFiles {
                predictor: predictor.into_bytes(),
                declaration: Some(crate::export::dts::generate(&header).unwrap().into_bytes()),
            },
        )]);
        let bundle = assemble(
            b"fitted model fixture",
            &[Language::Typescript],
            &emitted,
            b"{}",
            false,
        )
        .unwrap();
        let archive_bytes = zip_bytes(&bundle).unwrap();
        let mut archive = zip::ZipArchive::new(Cursor::new(archive_bytes)).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let evidence_dir =
            std::env::var_os("JOPE_TS_BUNDLE_EVIDENCE_DIR").map(std::path::PathBuf::from);
        if let Some(path) = &evidence_dir {
            std::fs::create_dir_all(path).unwrap();
            std::fs::write(path.join("manifest.json"), &bundle.entries["manifest.json"]).unwrap();
        }
        for entry_name in [
            "src/typescript/predictor.mjs",
            "src/typescript/predictor.d.ts",
            "src/typescript/predictor.d.mts",
        ] {
            let mut entry = archive.by_name(entry_name).unwrap();
            let destination = dir
                .path()
                .join(std::path::Path::new(entry_name).file_name().unwrap());
            let mut output = std::fs::File::create(destination).unwrap();
            std::io::copy(&mut entry, &mut output).unwrap();
            if let Some(path) = &evidence_dir {
                std::fs::copy(
                    dir.path()
                        .join(std::path::Path::new(entry_name).file_name().unwrap()),
                    path.join(std::path::Path::new(entry_name).file_name().unwrap()),
                )
                .unwrap();
            }
        }
        let consumer = dir.path().join("fitted-consumer.mts");
        let consumer_source = include_str!("../../../../deploy/ci/typescript/fitted-consumer.mts");
        std::fs::write(&consumer, consumer_source).unwrap();
        if let Some(path) = &evidence_dir {
            std::fs::write(path.join("fitted-consumer.mts"), consumer_source).unwrap();
        }
        let output = Command::new(tsc)
            .args([
                "--strict",
                "--noEmit",
                "--target",
                "ES2020",
                "--module",
                "NodeNext",
                "--moduleResolution",
                "NodeNext",
            ])
            .arg(&consumer)
            .output()
            .unwrap();
        if let Some(path) = &evidence_dir {
            let log = format!(
                "command: tsc --strict --noEmit --target ES2020 --module NodeNext --moduleResolution NodeNext fitted-consumer.mts\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            std::fs::write(path.join("tsc.log"), log).unwrap();
        }
        assert!(
            output.status.success(),
            "tsc --strict failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let runtime = dir.path().join("exercise.mjs");
        std::fs::write(
            &runtime,
            "import {predict,predictBatch} from './predictor.mjs';\nconst row={age:1,region:2,'quote\\\"field':3};\nconst single=predict(row),batch=predictBatch([row]);\nif(!Number.isFinite(single)||batch.length!==1||batch[0]!==single)throw Error('predict API mismatch');\nfor(const bad of [{...row,region:'north'},{age:1,'quote\\\"field':3}]){let refused=false;try{predict(bad);}catch(error){refused=error instanceof TypeError;}if(!refused)throw Error('non-numeric or missing input was not refused: '+JSON.stringify(bad));}\n",
        )
        .unwrap();
        let output = Command::new("node").arg(runtime).output().unwrap();
        assert!(
            output.status.success(),
            "generated predictor runtime failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        if let Some(path) = evidence_dir {
            std::fs::write(
                path.join("node-runtime.log"),
                "node exercised exported predict and predictBatch with the same fitted row; result finite and equal\n",
            )
            .unwrap();
        }
    }
}

/// Measurements from an explicit, trusted host qualification run. There is no public
/// endpoint that accepts a caller's claimed parity. Exporting never launches a compiler.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredModelParity {
    pub schema: String,
    pub model_digest: String,
    pub emitter_version: String,
    pub refit: jope_api::Refit,
    pub source_sha256: BTreeMap<String, String>,
    pub toolchains: BTreeMap<String, String>,
    /// Raw feature cells, in the frozen input-contract order; no target values.
    pub rows: Vec<Vec<Option<String>>>,
    pub engine: Vec<f64>,
    pub observed: BTreeMap<String, Vec<f64>>,
}

impl MeasuredModelParity {
    fn validate(&self, object: &ModelObject, refit: &serde_json::Value) -> Result<()> {
        ensure!(
            self.schema == "jope.measured-model-parity/v1",
            "unknown parity protocol"
        );
        ensure!(self.model_digest == object.id(), "parity model mismatch");
        ensure!(
            self.emitter_version == env!("CARGO_PKG_VERSION"),
            "parity emitter mismatch"
        );
        ensure!(
            *refit == serde_json::to_value(self.refit)?,
            "parity refit mismatch"
        );
        ensure!(
            self.rows.len() == 64 && self.engine.len() == 64,
            "parity requires 64 check rows"
        );
        let expected_keys = ["python".to_string(), "rust".to_string()];
        for keys in [
            self.source_sha256.keys().cloned().collect::<Vec<_>>(),
            self.observed.keys().cloned().collect(),
            self.toolchains.keys().cloned().collect(),
        ] {
            ensure!(
                keys == expected_keys,
                "parity requires Python and Rust measurements"
            );
        }
        let pipeline = object.math_pipeline()?;
        let borrowed = self
            .rows
            .iter()
            .map(|r| r.iter().map(Option::as_deref).collect())
            .collect::<Vec<_>>();
        let engine = pipeline.predict_cells(&borrowed)?;
        ensure!(
            engine == self.engine && engine.iter().all(|v| v.is_finite()),
            "parity engine values differ"
        );
        for language in &expected_keys {
            ensure!(
                self.source_sha256[language] == sha256_hex(&object.export_source(language)?),
                "parity source mismatch"
            );
            ensure!(
                !self.toolchains[language].trim().is_empty(),
                "missing parity toolchain identity"
            );
            let observed = &self.observed[language];
            ensure!(observed.len() == 64, "parity output count differs");
            ensure!(
                observed
                    .iter()
                    .zip(&engine)
                    .all(|(a, b)| a.is_finite() && (a - b).abs() <= 1e-12 + 1e-10 * b.abs()),
                "exported predictor parity failed"
            );
        }
        Ok(())
    }
}

/// Bind actual toolchain measurements to a completed model in the trusted local store.
/// Callers retain the toolchain execution receipts. The service rechecks all bindings
/// and the engine values on every new export; absence of evidence stays Unavailable.
pub fn record_model_parity(
    store: &crate::store::Store,
    model_artifact: &jope_api::ids::ArtifactId,
    object: &ModelObject,
    receipt: &MeasuredModelParity,
) -> Result<jope_api::ids::ArtifactId> {
    use crate::store::{Record, RecordKey};
    let key = RecordKey::Artifact(model_artifact.clone());
    let original = store.get(&key)?.context("parity model artifact missing")?;
    ensure!(
        original.data["kind"] == "finalized_model" && original.data["model_digest"] == object.id(),
        "parity needs the completed model"
    );
    let refit = original
        .data
        .get("refit")
        .cloned()
        .unwrap_or(serde_json::json!("fit"));
    receipt.validate(object, &refit)?;
    let bytes = serde_json::to_vec(receipt)?;
    ensure!(
        bytes.len() <= 16 * 1024 * 1024,
        "parity receipt exceeds bound"
    );
    let blob = store.publish_blob(std::io::Cursor::new(bytes))?;
    let digest = hex_blob(&blob.sha256);
    let id: jope_api::ids::ArtifactId = format!("art_{}", &digest[..24]).parse()?;
    store.transaction(|tx| {
        let mut current = tx.get(&key)?.ok_or(crate::store::StoreError::Invalid(
            "parity model disappeared",
        ))?;
        if current != original {
            return Err(crate::store::StoreError::Invalid("parity model changed"));
        }
        let parity_key = RecordKey::Artifact(id.clone());
        let parity = Record::metadata(
            serde_json::json!({"kind":"model_parity","model_digest":object.id(),"archive":blob}),
            vec![blob.clone()],
        );
        if let Some(old) = tx.get(&parity_key)? {
            if old != parity {
                return Err(crate::store::StoreError::Integrity(
                    "parity artifact collision",
                ));
            }
        } else {
            tx.put(&parity_key, &parity)?;
        }
        current.data["parity_receipt"] = serde_json::to_value(&id)?;
        tx.put(&key, &current)
    })?;
    Ok(id)
}

fn hex_blob(bytes: &[u8]) -> String {
    bytes.iter().map(|v| format!("{v:02x}")).collect()
}

/// Return only measured evidence for the exact frozen model, mode and source bytes.
pub(crate) fn verified_model_parity(
    store: &crate::store::Store,
    original: &crate::store::Record,
    object: &ModelObject,
    languages: &[jope_api::Language],
) -> Result<Option<(jope_api::ids::ArtifactId, String, Vec<u8>)>> {
    use crate::store::RecordKey;
    let Some(value) = original.data.get("parity_receipt") else {
        return Ok(None);
    };
    let id: jope_api::ids::ArtifactId = serde_json::from_value(value.clone())?;
    let record = store
        .get(&RecordKey::Artifact(id.clone()))?
        .context("parity receipt missing")?;
    ensure!(
        record.data["kind"] == "model_parity" && record.data["model_digest"] == object.id(),
        "parity record mismatch"
    );
    let blob = record
        .blobs
        .first()
        .filter(|b| b.bytes <= 16 * 1024 * 1024)
        .context("parity blob missing or too large")?;
    let bytes = store.read_blob(blob)?;
    let receipt: MeasuredModelParity = jope_runtime::strict_json::decode(&bytes)?;
    let refit = original
        .data
        .get("refit")
        .cloned()
        .unwrap_or(serde_json::json!("fit"));
    receipt.validate(object, &refit)?;
    if languages.is_empty()
        || languages
            .iter()
            .any(|l| !matches!(l, jope_api::Language::Python | jope_api::Language::Rust))
    {
        return Ok(None);
    }
    ensure!(
        id.to_string() == format!("art_{}", &sha256_hex(&bytes)[..24]),
        "parity content identity differs"
    );
    Ok(Some((
        id,
        format!("measured-model:{}", sha256_hex(&bytes)),
        bytes,
    )))
}
