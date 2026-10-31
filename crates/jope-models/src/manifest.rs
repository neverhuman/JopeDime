//! `jope.model-bundle/v2`: the committed/published description of one bundle — every file with
//! its sha256 (= blob key), size, kind and `jain.model-license/v1` block, plus embedded notices.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeSet;

/// Bundle manifest schema identifier.
pub const SCHEMA_ID: &str = "jope.model-bundle/v2";
/// Per-file license metadata schema identifier.
pub const LICENSE_SCHEMA_ID: &str = "jain.model-license/v1";
/// Embedded JSON Schema document used to validate bundle manifests.
pub const SCHEMA_JSON: &str = include_str!("../../../schemas/jope.model-bundle.v2.schema.json");

/// Parse and return the embedded bundle manifest JSON Schema.
pub fn schema() -> Value {
    serde_json::from_str(SCHEMA_JSON).expect("committed schema parses")
}

/// License metadata associated with one bundle file.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct License {
    /// License schema identifier.
    pub schema: String,
    /// SPDX expression describing the file license.
    pub spdx_expression: String,
    /// License classification used by the bundle policy.
    pub license_class: String,
    /// `permitted` | `internal` | `prohibited` | `conditional`.
    pub redistribution: String,
    /// Paths of notices embedded in the bundle manifest.
    #[serde(default)]
    pub notice_paths: Vec<String>,
    /// Additional forward-compatible license fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// One file referenced by a model bundle manifest.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct BundleFile {
    /// Bundle-relative path (`bundles/<id>/<path>` in the cache).
    pub path: String,
    /// Lowercase SHA-256 digest of the file contents.
    pub sha256: String,
    /// File size in bytes.
    pub bytes: u64,
    /// `blobs/sha256/<sha256>`.
    pub blob: String,
    /// File kind label from the bundle manifest.
    pub kind: String,
    /// Optional license metadata.
    #[serde(default)]
    pub license: Option<License>,
    /// Additional forward-compatible file fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// An embedded notice included with a bundle.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Notice {
    /// Safe bundle-relative path where the notice is materialized.
    pub path: String,
    /// Notice text.
    pub text: String,
}

/// Validated description of the files and notices in a model bundle.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct BundleManifest {
    /// Bundle schema identifier.
    pub schema: String,
    /// Stable bundle identifier.
    pub bundle_id: String,
    /// Files included in the bundle.
    pub files: Vec<BundleFile>,
    /// Notices embedded in the manifest.
    #[serde(default)]
    pub notices: Vec<Notice>,
    /// Additional forward-compatible manifest fields.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// A safe bundle-relative path: non-empty normal components only (no `/` prefix, `.`, `..`,
/// empty segments or backslashes).
pub fn safe_relative(path: &str) -> bool {
    !path.is_empty()
        && !path.contains('\\')
        && path
            .split('/')
            .all(|c| !c.is_empty() && c != "." && c != "..")
}

/// Return whether `s` is a 64-character lowercase hexadecimal SHA-256 digest.
pub fn valid_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

impl BundleFile {
    /// `redistribution: prohibited` — never fetched without `--accept-restricted`.
    pub fn restricted(&self) -> bool {
        self.license
            .as_ref()
            .is_some_and(|l| l.redistribution == "prohibited")
    }

    /// One line for the pre-download listing: `<path>  <spdx> / <redistribution>`.
    pub fn license_line(&self) -> String {
        match &self.license {
            Some(l) => format!(
                "{}  {} / {}{}",
                self.path,
                l.spdx_expression,
                l.redistribution,
                if self.restricted() {
                    "  [restricted]"
                } else {
                    ""
                }
            ),
            None => format!("{}  (no licence block: {})", self.path, self.kind),
        }
    }
}

impl BundleManifest {
    /// Parse and validate: JSON Schema, then the invariants the schema cannot express.
    pub fn parse(bytes: &[u8]) -> Result<BundleManifest, Vec<String>> {
        let value: Value = serde_json::from_slice(bytes).map_err(|e| vec![format!("json: {e}")])?;
        Self::validate_value(&value)
    }

    /// Validate a JSON value against the schema and bundle invariants.
    pub fn validate_value(value: &Value) -> Result<BundleManifest, Vec<String>> {
        let mut errs = jope_datasets::schema::validate(&schema(), value);
        let doc: Option<BundleManifest> = match serde_json::from_value(value.clone()) {
            Ok(m) => Some(m),
            Err(e) => {
                errs.push(format!("deserialize: {e}"));
                None
            }
        };
        if let Some(m) = &doc {
            errs.extend(m.invariant_errors());
        }
        match (errs.is_empty(), doc) {
            (true, Some(m)) => Ok(m),
            _ => Err(errs),
        }
    }

    /// Return manifest consistency violations not expressible in the JSON Schema.
    pub fn invariant_errors(&self) -> Vec<String> {
        let mut errs = Vec::new();
        if self.schema != SCHEMA_ID {
            errs.push(format!("schema {} != {SCHEMA_ID}", self.schema));
        }
        let notice_paths: BTreeSet<&str> = self.notices.iter().map(|n| n.path.as_str()).collect();
        if notice_paths.len() != self.notices.len() {
            errs.push("duplicate notice paths".into());
        }
        for n in &self.notices {
            if !safe_relative(&n.path) {
                errs.push(format!("notice {}: unsafe path", n.path));
            }
        }
        let mut seen = BTreeSet::new();
        for f in &self.files {
            if !safe_relative(&f.path) {
                errs.push(format!("{}: unsafe path", f.path));
            }
            if f.path.ends_with(".partial") {
                errs.push(format!("{}: reserved path", f.path));
            }
            if !seen.insert(f.path.as_str()) {
                errs.push(format!("{}: duplicate path", f.path));
            }
            if !valid_sha256(&f.sha256) {
                errs.push(format!("{}: sha256 is not 64 lowercase hex", f.path));
            }
            if f.blob != jope_datasets::blobs::blob_key(&f.sha256) {
                errs.push(format!("{}: blob key != blobs/sha256/<sha256>", f.path));
            }
            if f.bytes == 0 {
                errs.push(format!("{}: bytes must be > 0", f.path));
            }
            if let Some(l) = &f.license {
                if l.schema != LICENSE_SCHEMA_ID {
                    errs.push(format!("{}: licence schema {}", f.path, l.schema));
                }
                for np in &l.notice_paths {
                    if !notice_paths.contains(np.as_str()) {
                        errs.push(format!("{}: notice {np} is not embedded", f.path));
                    }
                }
            }
        }
        errs
    }

    /// Find a file by its exact bundle-relative path.
    pub fn file(&self, path: &str) -> Option<&BundleFile> {
        self.files.iter().find(|f| f.path == path)
    }

    /// The files named by `--component` selectors (an exact path, or a directory prefix such
    /// as `model-bundle/`); no selector means every file.
    pub fn select(&self, components: &[String]) -> Result<Vec<&BundleFile>> {
        if components.is_empty() {
            return Ok(self.files.iter().collect());
        }
        let mut out: Vec<&BundleFile> = Vec::new();
        for c in components {
            let dir = c.trim_end_matches('/').to_string() + "/";
            let matched: Vec<&BundleFile> = self
                .files
                .iter()
                .filter(|f| f.path == *c || f.path.starts_with(&dir))
                .collect();
            if matched.is_empty() {
                return Err(anyhow!(
                    "{}: no such component in bundle {} (paths: {})",
                    c,
                    self.bundle_id,
                    self.files
                        .iter()
                        .map(|f| f.path.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            for f in matched {
                if !out.iter().any(|o| o.path == f.path) {
                    out.push(f);
                }
            }
        }
        Ok(out)
    }

    /// Sum the declared byte lengths of all files.
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.bytes).sum()
    }

    /// The set of `spdx_expression`s (`none` for files without a licence block), as the index
    /// records them.
    pub fn license_set(&self) -> BTreeSet<String> {
        self.files
            .iter()
            .map(|f| {
                f.license
                    .as_ref()
                    .map_or_else(|| "none".to_string(), |l| l.spdx_expression.clone())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn committed_manifests_validate() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/bundles");
        let mut n = 0;
        for entry in std::fs::read_dir(&dir).unwrap() {
            let p = entry.unwrap().path();
            if p.extension().is_some_and(|e| e == "json") {
                let m = BundleManifest::parse(&std::fs::read(&p).unwrap())
                    .unwrap_or_else(|e| panic!("{}: {}", p.display(), e.join("; ")));
                assert!(!m.files.is_empty());
                n += 1;
            }
        }
        assert!(n >= 3, "expected the three committed bundles, found {n}");
    }

    #[test]
    fn invariants() {
        assert!(safe_relative("a/b.safetensors"));
        assert!(!safe_relative("/a"));
        assert!(!safe_relative("a/../b"));
        assert!(!safe_relative("a//b"));
        assert!(!safe_relative(""));
        let text = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/bundles/hyperion-v7.json"),
        )
        .unwrap();
        let mut v: Value = serde_json::from_slice(&text).unwrap();
        v["files"][0]["blob"] = Value::String("blobs/sha256/deadbeef".into());
        let errs = BundleManifest::validate_value(&v).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("blob")), "{errs:?}");
        let mut v: Value = serde_json::from_slice(&text).unwrap();
        v["files"][1]["path"] = v["files"][0]["path"].clone();
        let errs = BundleManifest::validate_value(&v).unwrap_err();
        assert!(errs.iter().any(|e| e.contains("duplicate")), "{errs:?}");
        let mut v: Value = serde_json::from_slice(&text).unwrap();
        v["files"][1]["license"]["redistribution"] = Value::String("maybe".into());
        assert!(BundleManifest::validate_value(&v).is_err());
        let m = BundleManifest::parse(&text).unwrap();
        assert_eq!(m.select(&[]).unwrap().len(), 2);
        assert_eq!(
            m.select(&["model.safetensors".to_string()]).unwrap().len(),
            1
        );
        assert!(m.select(&["nope".to_string()]).is_err());
        assert!(!m.files[1].restricted());
        assert!(m.files[1].license_line().contains("internal"));
    }
}
