use crate::{decimal::DecimalU64, ids::*, Warning};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    Auto,
    Regression,
    Binary,
    /// Reserved in the v1 vocabulary; runtime support is deferred by t525u.
    Multiclass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Effort {
    Fast,
    Medium,
    High,
    Ultra,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Language {
    Python,
    Rust,
    Javascript,
    Typescript,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Guidance {
    Auto,
    Learned,
    Heuristic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LandmarkMode {
    Full,
    None,
    /// The plan handler refuses this with CAPABILITY_UNAVAILABLE in v1.
    Progressive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SplitKind {
    Random,
    Stratified,
    Grouped,
    TimeOrdered,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SplitSpec {
    pub kind: SplitKind,
    pub holdout_fraction: f64,
    pub group_column: Option<String>,
    pub time_column: Option<String>,
}

/// The objective grammar is owned by jope-runtime; the API transports it without
/// importing the evaluator. A string is a built-in name or name:param=value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum ObjectiveSpec {
    Name(String),
    Specification(BTreeMap<String, serde_json::Value>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Budget {
    pub evaluations: Option<DecimalU64>,
    pub wall_seconds: Option<DecimalU64>,
}

/// Operation 6 uses this same body for a dry-run plan and a real start.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RunSpec {
    pub dataset: DatasetId,
    pub target: String,
    pub task: Task,
    pub ignore_columns: Vec<String>,
    pub test_dataset: Option<DatasetId>,
    pub split: SplitSpec,
    pub objective: ObjectiveSpec,
    pub effort: Effort,
    pub budget: Budget,
    pub landmarks: LandmarkMode,
    pub guidance: Guidance,
    pub languages: Vec<Language>,
    pub require_all_languages: bool,
    /// `auto`, `cpu`, or an explicit device such as `cuda:0`.
    pub device: String,
    pub seed: DecimalU64,
    pub dry_run: bool,
    pub plan_digest: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResolvedTask {
    pub value: Task,
    pub inferred: bool,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Minimize,
    Maximize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResolvedObjective {
    pub name: String,
    pub digest: String,
    pub direction: Direction,
    pub fitting: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RowCounts {
    pub total: DecimalU64,
    pub fit: DecimalU64,
    pub holdout: DecimalU64,
    pub test: Option<DecimalU64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResolvedEffort {
    pub name: Effort,
    pub evaluations: DecimalU64,
    pub wall_seconds: DecimalU64,
    pub cv_folds: u32,
    pub rounds_in_flight: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResolvedGuidance {
    pub value: Guidance,
    pub prior_only: bool,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResolvedLandmarks {
    pub mode: LandmarkMode,
    pub sample_rows: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResolvedLanguages {
    pub requested: Vec<Language>,
    pub excluded_operations: Vec<String>,
    pub intersection_empty: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Placement {
    pub dope: String,
    pub ranker: String,
    pub evaluation: String,
    pub workers: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RunLimits {
    pub eval_timeout_s: u32,
    pub worker_rss_bytes: DecimalU64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResourceEstimates {
    pub peak_ram_bytes: DecimalU64,
    pub disk_bytes: DecimalU64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Plan {
    pub plan_digest: String,
    pub task: ResolvedTask,
    pub objective: ResolvedObjective,
    pub rows: RowCounts,
    pub effort: ResolvedEffort,
    pub guidance: ResolvedGuidance,
    pub landmarks: ResolvedLandmarks,
    pub languages: ResolvedLanguages,
    pub placement: Placement,
    pub limits: RunLimits,
    pub estimates: ResourceEstimates,
    pub warnings: Vec<Warning>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RunState {
    Running,
    Pausing,
    Paused,
    Completed,
    Cancelled,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RunPhase {
    Preparing,
    Encoding,
    Seeding,
    Searching,
    Finalizing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CompletionReason {
    BudgetEvaluations,
    BudgetTime,
    Converged,
    EffortReduced,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BudgetUsed {
    pub evaluations: DecimalU64,
    /// Elapsed seconds may be fractional (the spec example is "23.4").
    pub wall_seconds: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BudgetTotal {
    pub evaluations: DecimalU64,
    pub wall_seconds: DecimalU64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BestAlgorithm {
    pub algorithm_id: AlgorithmId,
    pub metric: String,
    pub cv_value: f64,
    pub model_artifact: Option<ArtifactId>,
}

/// The watermark, rather than revision, is the long-poll/attach cursor (§4.4).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RunSnapshot {
    pub run_id: RunId,
    pub revision: DecimalU64,
    pub state: RunState,
    pub phase: RunPhase,
    pub pending: Option<String>,
    pub effort: Effort,
    pub budget_used: BudgetUsed,
    pub budget_total: BudgetTotal,
    pub best: Option<BestAlgorithm>,
    pub event_watermark: DecimalU64,
    pub warnings: Vec<Warning>,
    pub completion_reason: Option<CompletionReason>,
    /// Present only for engine evaluations using the versioned Q1 receipt path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation_receipt: Option<EvaluationReceipt>,
}

/// Durable, request-bound result from one engine evaluation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EvaluationReceipt {
    pub schema: String,
    pub request_sha256: String,
    pub identity: EvaluationIdentity,
    pub charged: DecimalU64,
    pub attempts: Vec<EvaluationAttempt>,
    pub champion_program: Option<String>,
    pub adapter: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EvaluationIdentity {
    pub source_commit: String,
    pub plan_sha256: String,
    pub pins: BTreeMap<String, String>,
    pub dataset_id: String,
    pub dataset_sha256: String,
    pub cut_sha256: String,
    pub guidance: String,
    pub metric: String,
    pub seed: DecimalU64,
    pub requested_budget: DecimalU64,
    pub dataset_binding_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EvaluationAttempt {
    pub attempt: DecimalU64,
    pub outcome: EvaluationAttemptOutcome,
    pub fold_scores: Vec<f64>,
    pub loss: Option<f64>,
    pub program_sha256: Option<String>,
    pub failure: Option<EvaluationFailure>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationAttemptOutcome {
    Completed,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EvaluationFailure {
    pub code: String,
    pub reason: String,
}
