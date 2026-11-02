#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Asset {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub id: String,
    pub group: String,
    pub outer: u8,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dataset {
    pub id: String,
    pub family: String,
    pub features: Asset,
    pub labels: Asset,
    pub rows: Vec<Row>,
    /// "untouched", "adaptive_development", or "unknown", supported by the exposure asset.
    pub evaluation_exposure: String,
    pub checkpoint_exposure: String,
    pub encoder_exposure: String,
    pub exposure_evidence: Asset,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Arm {
    pub id: String,
    pub checkpoint: Asset,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub max_searches: u64,
    pub candidates_per_search: u64,
    pub candidate_fit_seconds: u64,
    pub max_fit_seconds: u64,
    pub cpus: u32,
    pub memory_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Specification {
    pub schema: String,
    pub round: String,
    pub source_sha: String,
    pub procedure: Asset,
    pub encoder: Asset,
    pub assets: Vec<Asset>,
    /// Parent and challenger are mandatory; controls are retained but never pooled into the KPI.
    pub arms: Vec<Arm>,
    pub seeds: Vec<u64>,
    pub datasets: Vec<Dataset>,
    pub limits: Limits,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Search {
    pub dataset: String,
    pub seed: u64,
    pub outer: u8,
    pub audit: Option<u8>,
    pub train: Vec<String>,
    pub score: Vec<String>,
    /// Five inner selection partitions wholly within train. Full search repeats for each Search.
    pub inner_score: Vec<Vec<String>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub schema: String,
    pub specification: Specification,
    pub searches: Vec<Search>,
    pub failures: Vec<String>,
    pub planned_units: usize,
    pub planned_searches: u64,
    pub estimated_candidate_fits: u64,
    pub estimated_fit_seconds: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prediction {
    pub schema: String,
    pub plan_sha256: String,
    pub search_sha256: String,
    pub arm: String,
    pub checkpoint_sha256: String,
    pub procedure_sha256: String,
    pub recipe_sha256: String,
    pub recipe: Asset,
    pub selection_receipt: Asset,
    pub fitting_rows: Vec<String>,
    pub selection_label_rows: Vec<String>,
    pub predictions: Vec<f64>,
    pub completed_sequence: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionReceipt {
    pub schema: String,
    pub search_sha256: String,
    pub procedure_sha256: String,
    pub recipe_sha256: String,
    pub inner_score: Vec<Vec<String>>,
    pub label_dependencies: Vec<String>,
    pub complete_search_repeated: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FailedPrediction {
    pub search_sha256: String,
    pub arm: String,
    pub status: String,
    pub reason: String,
    pub completed_sequence: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Freeze {
    pub schema: String,
    pub plan_sha256: String,
    pub predictions: Vec<Asset>,
    pub failures: Vec<FailedPrediction>,
    pub frozen_sequence: u64,
}
/// Supplied by separately admitted custody, with its digest pinned outside the evidence
/// bundle. A self-authored JSON file is not independent proof of chronological isolation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Release {
    pub schema: String,
    pub plan_sha256: String,
    pub freeze_sha256: String,
    pub released_sequence: u64,
    pub first_scored_result_sequence: u64,
    pub executor_source_sha: String,
    pub isolation_evidence: Asset,
    pub all_predictions_committed_before_scoring: bool,
    pub scoring_labels_excluded_from_selection_and_fitting: bool,
    pub no_cross_fold_result_feedback: bool,
    pub independently_verified: bool,
}
