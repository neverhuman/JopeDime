use anyhow::{ensure, Context, Result};
use feat_math::advanced_v3::{
    fitted_portable_templates, ColumnSpec, PortableUnsupported, FITTED_PORTABLE_EMITTER,
    FITTED_PUBLIC_CPP_EMITTER, NUMERICAL_POLICY,
};
use jope_datasets::catalog::Task;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Python,
    Rust,
    Cpp17,
    Cpp20,
    #[serde(rename = "javascript")]
    JavaScript,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ParityPolicy {
    ExactBits,
    AbsoluteRelative { absolute: f64, relative: f64 },
}

impl Backend {
    pub fn numerical_policy(self) -> String {
        if self == Self::JavaScript {
            NUMERICAL_POLICY.replace("platform-libm-v1", "ecmascript-math-es2019-v1")
        } else {
            NUMERICAL_POLICY.into()
        }
    }
    pub(crate) fn emitter(self) -> &'static str {
        if matches!(self, Self::Cpp17 | Self::Cpp20) {
            FITTED_PUBLIC_CPP_EMITTER
        } else {
            FITTED_PORTABLE_EMITTER
        }
    }
    pub(crate) fn runtime(self) -> &'static str {
        match self {
            Self::JavaScript => "ES2019 or later; Node.js 12 or later; ECMAScript Math",
            Self::Python => "Python 3.10 or later; host libm",
            Self::Rust => "Rust edition 2021; host libm",
            Self::Cpp17 => "C++17 plus Python 3.10 ctypes typed-table host; host libm",
            Self::Cpp20 => "C++20 plus Python 3.10 ctypes typed-table host; host libm",
        }
    }
    pub(crate) fn host_boundary(self) -> &'static str {
        if matches!(self, Self::Cpp17 | Self::Cpp20) {
            "python-ctypes-typed-table"
        } else {
            "raw-rows"
        }
    }
    pub fn parity_policy(self) -> ParityPolicy {
        match self {
            Self::JavaScript => ParityPolicy::AbsoluteRelative {
                absolute: 1e-12,
                relative: 1e-12,
            },
            _ => ParityPolicy::ExactBits,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicManifest {
    pub schema: String,
    pub backend: Backend,
    pub task: Task,
    pub columns: Vec<ColumnSpec>,
    pub classes: Vec<String>,
    pub raw_cell_encoding: String,
    pub input_policy: String,
    pub numerical_policy: String,
    pub emitter: String,
    pub template_sha256: BTreeMap<String, String>,
    pub runtime: String,
    pub host_boundary: String,
    pub parity_policy: ParityPolicy,
    pub qualification: String,
    pub entrypoint: String,
    pub files: BTreeMap<String, String>,
}

/// W9-T01: named-column input mode. A host-side adapter over the positional raw-row ABI that
/// every backend ships, so emitted source, bundle identities and qualification plans are
/// unchanged.
impl PublicManifest {
    /// Positional raw rows, in this manifest's column order, from named records: a JSON array
    /// of objects, each naming every column exactly once with a string or null. Key order is
    /// irrelevant, so columns of one kind can never be swapped silently, which the positional
    /// mode cannot detect. A missing, unknown or repeated name, any other value type, a record
    /// that is not an object, or a manifest whose column names are not distinct is refused
    /// before anything is served. The rows then pass the unchanged positional boundary and its
    /// validators.
    pub fn positional_rows(&self, records: &[u8]) -> Result<Vec<Vec<Option<String>>>> {
        let names: Vec<&str> = self.columns.iter().map(|c| c.name.as_str()).collect();
        ensure!(
            names
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                == names.len(),
            "manifest column names are not distinct; named input is ambiguous"
        );
        let records: Vec<NamedRecord> = serde_json::from_slice(records)
            .context("named records must be a JSON array of objects of strings or nulls")?;
        records
            .into_iter()
            .enumerate()
            .map(|(index, record)| {
                let mut row: Vec<Option<Option<String>>> = vec![None; names.len()];
                for (name, value) in record.0 {
                    let position = names
                        .iter()
                        .position(|column| *column == name)
                        .with_context(|| format!("record {index}: unknown column {name:?}"))?;
                    ensure!(
                        row[position].replace(value).is_none(),
                        "record {index}: column {name:?} named twice"
                    );
                }
                row.into_iter()
                    .zip(&names)
                    .map(|(value, name)| {
                        value.with_context(|| format!("record {index}: column {name:?} missing"))
                    })
                    .collect()
            })
            .collect()
    }
}

/// One named record with its entries in document order, repeated names kept so that they
/// can be refused rather than silently collapsed.
struct NamedRecord(Vec<(String, Option<String>)>);
impl<'de> Deserialize<'de> for NamedRecord {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Entries;
        impl<'de> serde::de::Visitor<'de> for Entries {
            type Value = NamedRecord;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("an object of column name to string or null")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<NamedRecord, A::Error> {
                let mut entries = Vec::new();
                while let Some(entry) = map.next_entry::<String, Option<String>>()? {
                    entries.push(entry);
                }
                Ok(NamedRecord(entries))
            }
        }
        deserializer.deserialize_map(Entries)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicBundle {
    pub manifest: PublicManifest,
    pub files: BTreeMap<String, String>,
}
impl PublicBundle {
    pub fn identity(&self) -> Result<String> {
        Ok(digest(&serde_json::to_vec(self)?))
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.manifest.schema == "jope.fitted-public-bundle/v1"
                && self.manifest.qualification == "emitted-unqualified"
                && self.manifest.raw_cell_encoding == RAW_CELL_ENCODING
                && matches!(
                    self.manifest.input_policy.as_str(),
                    INPUT_POLICY | FITTED_INPUT_POLICY
                )
                && self.manifest.numerical_policy == self.manifest.backend.numerical_policy()
                && self.manifest.emitter == self.manifest.backend.emitter()
                && self.manifest.runtime == self.manifest.backend.runtime()
                && self.manifest.host_boundary == self.manifest.backend.host_boundary()
                && self.manifest.template_sha256 == templates(self.manifest.backend)
                && self.manifest.parity_policy == self.manifest.backend.parity_policy(),
            "unsupported public bundle contract"
        );
        let expected: &[&str] = match self.manifest.backend {
            Backend::Python => &["model.py"],
            Backend::Rust => &["model.rs"],
            Backend::JavaScript => &["model.js"],
            Backend::Cpp17 | Backend::Cpp20 => &["model.cpp", "codec.json", "boundary.py"],
        };
        ensure!(
            self.files.len() == expected.len()
                && expected.iter().all(|name| self.files.contains_key(*name)),
            "public bundle file allowlist differs"
        );
        ensure!(
            self.manifest.files
                == self
                    .files
                    .iter()
                    .map(|(path, text)| (path.clone(), digest(text.as_bytes())))
                    .collect(),
            "public file checksum differs"
        );
        Ok(())
    }
}
