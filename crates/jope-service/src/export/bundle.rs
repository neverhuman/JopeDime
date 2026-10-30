//! Deterministic export bundle contents and publication contract.
//!
//! ZIP encoding is kept behind `BundleWriter` so the assembly contract can be exercised
//! independently from the store and archive implementation.

use anyhow::{ensure, Context, Result};
use jope_api::Language;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const BUNDLE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleManifest {
    pub schema_version: u32,
    pub model_object_digest: String,
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
) -> Result<Bundle> {
    ensure!(!model_object.is_empty(), "model object is empty");
    ensure!(!requested.is_empty(), "no export languages requested");
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
            usage_readme(language).into_bytes(),
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
        entries: hashes,
    };
    manifest.validate()?;
    entries.insert("manifest.json".into(), serde_json::to_vec(&manifest)?);
    Ok(Bundle { entries, manifest })
}

/// Writer owns the final encoding and store publication. Tests use an in-memory stub.
pub trait BundleWriter {
    type Published;
    fn publish(&mut self, entries: &BTreeMap<String, Vec<u8>>) -> Result<Self::Published>;
}

pub fn publish<W: BundleWriter>(bundle: &Bundle, writer: &mut W) -> Result<W::Published> {
    bundle.manifest.validate()?;
    ensure!(
        bundle.entries.contains_key("manifest.json"),
        "bundle manifest missing"
    );
    writer.publish(&bundle.entries)
}

pub fn assemble_and_publish<W: BundleWriter>(
    model_object: &[u8],
    requested: &[Language],
    emitted: &BTreeMap<LanguageKey, LanguageFiles>,
    parity_receipt: &[u8],
    writer: &mut W,
) -> Result<W::Published> {
    let bundle = assemble(model_object, requested, emitted, parity_receipt)?;
    publish(&bundle, writer)
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

fn usage_readme(language: LanguageKey) -> String {
    format!(
        "# {} predictor\n\nThis directory contains a standalone fitted predictor. Use `predictor.{}` with the prediction input contract for the bundled model.\n",
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
        let a = assemble(b"model object", &requested, &sources(), b"{\"checks\":[]}").unwrap();
        let b = assemble(b"model object", &requested, &sources(), b"{\"checks\":[]}").unwrap();
        assert_eq!(a.entries, b.entries);
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
            fn publish(&mut self, _: &BTreeMap<String, Vec<u8>>) -> Result<()> {
                self.calls += 1;
                Ok(())
            }
        }
        let absent = BTreeMap::new();
        let mut writer = Stub { calls: 0 };
        let result = assemble_and_publish(b"model", &[Language::Rust], &absent, b"{}", &mut writer);
        assert!(result.is_err());
        assert_eq!(writer.calls, 0);
    }

    #[test]
    fn manifest_rejects_traversal_and_bad_hashes() {
        let mut manifest = BundleManifest {
            schema_version: 1,
            model_object_digest: "a".repeat(64),
            entries: BTreeMap::from([("../evil".into(), "b".repeat(64))]),
        };
        assert!(manifest.validate().is_err());
        manifest.entries = BTreeMap::from([("model.jope".into(), "bad".into())]);
        assert!(manifest.validate().is_err());
    }
}
