//! `jope.pipeline-receipt/v1`: what the run did, stage by stage.
//!
//! The committed JSON Schema under `schemas/` is the public contract; the Rust
//! type derives `JsonSchema` so `tests/schema_agreement.rs` can assert the two
//! agree on every object's required set and property list, exactly as
//! `jope-datasets` does for its catalog. Validation reuses
//! `jope_datasets::schema`, the workspace's JSON Schema 2020-12 subset validator.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const RECEIPT_SCHEMA_ID: &str = "jope.pipeline-receipt/v1";

/// The committed receipt schema (JSON Schema 2020-12).
pub const RECEIPT_SCHEMA_JSON: &str =
    include_str!("../../../schemas/jope.pipeline-receipt.v1.schema.json");

/// A stage either ran, was deliberately skipped with a reason, or failed.
pub const STATUSES: [&str; 3] = ["passed", "skipped", "failed"];

/// One pipeline stage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Stage {
    pub name: String,
    /// `passed`, `skipped` or `failed`.
    pub status: String,
    /// Why, for anything that is not a plain pass. A skip always carries one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub seconds: f64,
}

/// One file of the dataset, as hashed at resolution time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DatasetFile {
    pub name: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Dataset {
    pub id: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{16}$"))]
    pub dataset_hash: String,
    /// Where the dataset was read from, as given on the command line.
    pub path: String,
    pub task: String,
    pub rows: u64,
    pub columns: u64,
    pub training_rows: u64,
    pub files: Vec<DatasetFile>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Prepared {
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub sha256: String,
    pub seed: u64,
    pub rows: u64,
    /// The partition each cut holds one fold out of.
    pub folds: u64,
    /// The cut indices that were built, in build order.
    pub cuts: Vec<u64>,
}

/// One cut's search, as its `search.json` published it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CutSearch {
    pub cut: u64,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub search_sha256: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub observations_sha256: String,
    /// Rows the fits on this cut saw, and rows they were scored on. Disjoint.
    pub train_rows: u64,
    pub hold_rows: u64,
    pub generations: u64,
    /// Fits attempted. One fit is one evaluation, whatever its outcome, so this
    /// is exactly `successful + failed`.
    pub completed_evaluations: u64,
    pub successful_measurements: u64,
    pub failed_or_unavailable_measurements: u64,
    /// What ended this cut: `evaluations`, `seconds`, or `search_space_exhausted`
    /// when the grammar stopped yielding candidates it had not already measured.
    pub stopped_by: String,
    /// The best held-out RMSE measured on this cut, in the dataset's target units.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_validation_rmse: Option<f64>,
    /// That same candidate's held-out R^2.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_validation_r2: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_program: Option<String>,
}

/// The selected program and its measured evidence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Best {
    pub genome: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub program_sha256: String,
    /// Mean held-out RMSE over the cuts this program was measured on, in the
    /// dataset's own target units. There is no normalisation to undo: the search
    /// fits the committed rows as they are.
    pub validation_rmse: f64,
    /// Mean held-out R^2 over the same cuts — the fits never saw these rows.
    pub validation_r2: f64,
    /// Mean in-sample R^2, for the gap against `validation_r2`.
    pub train_r2: f64,
    /// Fit-and-score seconds this program cost across those cuts.
    pub seconds: f64,
    pub cuts_measured: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Search {
    /// The search implementation that produced these measurements.
    pub engine: String,
    pub rng_seed: u64,
    /// The `--budget` bounds, per cut.
    pub evaluations: u64,
    pub seconds: u64,
    /// Candidates alive per generation, and the transform depth they may reach.
    pub population: u64,
    pub max_transforms: u64,
    /// The genome generation 0 always contains, so a measured champion is never
    /// worse than it.
    pub bootstrap_genome: String,
    pub cuts: Vec<CutSearch>,
    /// How the emitted program was chosen. `min_mean_validation_rmse` is the only
    /// rule applied here; no fold-pure lower-confidence-bound selector exists in
    /// this workspace. `unmeasured_fallback_bootstrap_genome` means nothing was
    /// measured at all, which the search stage now refuses to return.
    pub selection: String,
    /// The genome handed to the emitters, always present.
    pub selected_genome: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub selected_program_sha256: String,
    /// Absent when no cut produced a successful measurement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best: Option<Best>,
}

/// One emitted source file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Artifact {
    pub language: String,
    /// Relative to the run's `emit/` directory. `cpp17` and `cpp20` share one
    /// path: there is a single C++ emitter and the standard is a compile flag.
    pub path: String,
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub sha256: String,
    pub bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub build: Option<String>,
}

/// A requested language the selected genome is not covered for.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Exclusion {
    pub language: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Receipt {
    pub schema: String,
    pub ok: bool,
    pub tool_version: String,
    /// The checkout this ran from, when it could be determined.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_sha: Option<String>,
    pub created_at: String,
    /// The `--policy` the run was given: `auto`, `skip` or `require`.
    pub policy: String,
    pub dataset: Dataset,
    pub prepared: Prepared,
    pub stages: Vec<Stage>,
    pub search: Search,
    pub artifacts: Vec<Artifact>,
    pub emit_exclusions: Vec<Exclusion>,
    /// sha256 over every cut's `observations.json` digest, concatenated as hex
    /// in cut order: one value binding the run to all of its evidence.
    #[schemars(regex(pattern = r"^[0-9a-f]{64}$"))]
    pub observations_sha256: String,
}

impl Receipt {
    /// Structural invariants the schema cannot express.
    pub fn invariant_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema != RECEIPT_SCHEMA_ID {
            errors.push(format!(
                "schema is {:?}, expected {RECEIPT_SCHEMA_ID:?}",
                self.schema
            ));
        }
        for stage in &self.stages {
            if !STATUSES.contains(&stage.status.as_str()) {
                errors.push(format!(
                    "stage {}: status {:?} is not one of {STATUSES:?}",
                    stage.name, stage.status
                ));
            }
            if stage.status != "passed" && stage.reason.as_ref().is_none_or(String::is_empty) {
                errors.push(format!(
                    "stage {}: a {} stage must record a reason",
                    stage.name, stage.status
                ));
            }
        }
        for required in ["dataset", "prepare", "propose", "search", "select", "emit"] {
            if !self.stages.iter().any(|s| s.name == required) {
                errors.push(format!("stage {required} is missing"));
            }
        }
        if self.search.cuts.len() != self.prepared.cuts.len() {
            errors.push(format!(
                "{} cut searches for {} prepared cuts",
                self.search.cuts.len(),
                self.prepared.cuts.len()
            ));
        }
        for cut in &self.search.cuts {
            if cut.completed_evaluations
                != cut.successful_measurements + cut.failed_or_unavailable_measurements
            {
                errors.push(format!(
                    "cut {}: {} evaluations but {} + {} outcomes",
                    cut.cut,
                    cut.completed_evaluations,
                    cut.successful_measurements,
                    cut.failed_or_unavailable_measurements
                ));
            }
            if cut.successful_measurements > 0 && cut.best_program.is_none() {
                errors.push(format!(
                    "cut {}: {} measurement(s) but no best program",
                    cut.cut, cut.successful_measurements
                ));
            }
        }
        let measured: u64 = self
            .search
            .cuts
            .iter()
            .map(|c| c.successful_measurements)
            .sum();
        if measured > 0 && self.search.selection == crate::select::UNMEASURED_FALLBACK {
            errors.push(format!(
                "{measured} successful measurement(s) but the selection is the unmeasured fallback"
            ));
        }
        // A passed emit stage with nothing emitted would be a receipt that says the run produced
        // code while listing none, which no reader could catch without re-deriving it.
        if let Some(emit) = self.stages.iter().find(|s| s.name == "emit") {
            if emit.status == "passed" && self.artifacts.is_empty() {
                errors.push("stage emit passed but the receipt names no artifact".to_owned());
            }
            if emit.status == "failed" && !self.artifacts.is_empty() {
                errors.push(format!(
                    "stage emit failed but the receipt names {} artifact(s)",
                    self.artifacts.len()
                ));
            }
        }
        if self.ok && self.stages.iter().any(|s| s.status == "failed") {
            errors.push("ok is true but a stage failed".to_owned());
        }
        if let Some(best) = &self.search.best {
            if best.genome != self.search.selected_genome {
                errors.push("the selected genome is not the best one".to_owned());
            }
            if best.cuts_measured == 0 {
                errors.push("a best program measured on no cut".to_owned());
            }
        } else if self.search.selection != crate::select::UNMEASURED_FALLBACK {
            errors.push(format!(
                "selection {:?} without a best program",
                self.search.selection
            ));
        }
        errors
    }
}

/// The committed schema, parsed.
pub fn schema() -> serde_json::Value {
    serde_json::from_str(RECEIPT_SCHEMA_JSON).expect("committed receipt schema parses")
}

/// The schema schemars derives from [`Receipt`].
pub fn derived_schema() -> serde_json::Value {
    schemars::schema_for!(Receipt).to_value()
}

/// Validate one receipt document: the committed schema, then the Rust type, then
/// the invariants above.
pub fn validate_value(value: &serde_json::Value) -> Result<Receipt, Vec<String>> {
    let errors = jope_datasets::schema::validate(&schema(), value);
    if !errors.is_empty() {
        return Err(errors);
    }
    let receipt: Receipt =
        serde_json::from_value(value.clone()).map_err(|error| vec![error.to_string()])?;
    let errors = receipt.invariant_errors();
    if errors.is_empty() {
        Ok(receipt)
    } else {
        Err(errors)
    }
}

/// Read and validate a receipt file.
pub fn read(path: &std::path::Path) -> anyhow::Result<Receipt> {
    let bytes =
        std::fs::read(path).map_err(|error| anyhow::anyhow!("read {}: {error}", path.display()))?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| anyhow::anyhow!("parse {}: {error}", path.display()))?;
    validate_value(&value)
        .map_err(|errors| anyhow::anyhow!("{}: {}", path.display(), errors.join("; ")))
}
