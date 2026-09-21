use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const SCHEMA: &str = "jope.synthetic.v1";
pub const GENERATOR: &str = "jope.synthetic.splitmix64.integer.v1";
pub const ROWS: usize = 96;
pub const MAX_FILE_BYTES: u64 = 256 * 1024;
pub const MAX_CORPUS_BYTES: u64 = 10 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    Binary,
    Regression,
    TimeSeries,
    Multiclass,
}

impl Task {
    pub const ALL: [Self; 4] = [
        Self::Binary,
        Self::Regression,
        Self::TimeSeries,
        Self::Multiclass,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Binary => "binary",
            Self::Regression => "regression",
            Self::TimeSeries => "time_series",
            Self::Multiclass => "multiclass",
        }
    }

    /// Ten Git fixtures total (not ten per family). Real tables stay in S3.
    pub fn scenarios(self) -> &'static [&'static str] {
        match self {
            Self::Binary => &["linear", "imbalanced", "outliers"],
            Self::Regression => &["linear", "sparse_signal", "extrapolation"],
            Self::TimeSeries => &["trend", "lag_dependence"],
            Self::Multiclass => &["three_classes", "many_classes"],
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub kind: String,
    pub license: String,
    pub generator: String,
    pub source: String,
    pub external_inputs: Vec<String>,
}

impl Default for Provenance {
    fn default() -> Self {
        Self {
            kind: "synthetic".into(),
            license: "Apache-2.0".into(),
            generator: GENERATOR.into(),
            source: "crates/jope-cli/src/generator".into(),
            external_inputs: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    pub schema: String,
    pub provenance: Provenance,
    pub datasets: Vec<Entry>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub id: String,
    pub task: Task,
    pub scenario: String,
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
    pub seed: u64,
    pub rows: usize,
    pub expected_behavior: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Dataset {
    pub schema: String,
    pub id: String,
    pub task: Task,
    pub scenario: String,
    pub seed: u64,
    pub columns: Vec<Column>,
    /// Empty for numerical targets; otherwise the complete, ordered class vocabulary.
    pub classes: Vec<i64>,
    pub rows: Vec<Row>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Column {
    pub name: String,
    pub kind: Kind,
    pub nullable: bool,
    pub categories: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Integer,
    Categorical,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub id: usize,
    pub features: Vec<Value>,
    pub target: i64,
    pub split: Split,
    pub group: Option<String>,
    pub timestamp: Option<i64>,
    pub series: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Train,
    Validation,
    Test,
}

pub fn serialize<T: Serialize>(value: &T) -> anyhow::Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub fn digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}
