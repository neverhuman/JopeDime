use crate::{decimal::DecimalU64, error::ErrorCode, ids::*, *};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Only durable events have this envelope. Telemetry is a separate SSE payload.
/// Publication/sequence allocation belongs to SVC-04, after transaction commit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PublicEvent {
    pub run_id: RunId,
    pub sequence: DecimalU64,
    /// RFC 3339 UTC timestamp; produced by the service clock.
    pub time: String,
    #[serde(flatten)]
    pub event: EventBody,
}

/// Adjacent tagging makes mismatched kind/body pairs unrepresentable.
/// Appendix B's telemetry entry is ephemeral and cannot acquire a durable ID.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", content = "body")]
pub enum EventBody {
    #[serde(rename = "run.state")]
    RunState(RunStateBody),
    #[serde(rename = "run.config")]
    RunConfig(RunConfigBody),
    #[serde(rename = "plan")]
    Plan(Box<Plan>),
    #[serde(rename = "experiment.started")]
    ExperimentStarted(ExperimentStartedBody),
    #[serde(rename = "experiment.finished")]
    ExperimentFinished(ExperimentFinishedBody),
    #[serde(rename = "algorithm.accepted")]
    AlgorithmAccepted(AlgorithmAcceptedBody),
    #[serde(rename = "reasoning")]
    Reasoning(ReasoningBody),
    #[serde(rename = "warning")]
    Warning(Warning),
    #[serde(rename = "export.completed")]
    ExportCompleted(ExportCompletedBody),
    #[serde(rename = "run.finished")]
    RunFinished(RunFinishedBody),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RunStateBody {
    pub state: RunState,
    pub phase: RunPhase,
    pub pending: Option<String>,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RunConfiguration {
    pub revision: DecimalU64,
    pub effort: Effort,
    pub budget: BudgetTotal,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RunConfigBody {
    pub before: RunConfiguration,
    pub after: RunConfiguration,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ExperimentStartedBody {
    pub experiment_id: String,
    pub hypothesis: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentOutcome {
    Improved,
    NotImproved,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PublicFailure {
    pub code: ErrorCode,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ExperimentFinishedBody {
    pub experiment_id: String,
    pub outcome: ExperimentOutcome,
    pub metric: String,
    pub value: Option<f64>,
    pub failure: Option<PublicFailure>,
    /// Actual per-attempt details when the evaluator retained them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempt_receipt: Option<Box<EvaluationAttempt>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AlgorithmAcceptedBody {
    pub algorithm_id: AlgorithmId,
    pub metric: String,
    pub cv_value: f64,
    pub previous_value: Option<f64>,
    pub summary: String,
    pub model_artifact: Option<ArtifactId>,
    /// Primary-language source, limited to 12 lines by the projection producer.
    pub excerpt: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ReasoningBody {
    pub stage: String,
    pub algorithm_id: Option<AlgorithmId>,
    pub summary: String,
    pub evidence: Option<ComparisonEvidence>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ComparisonEvidence {
    pub metric: String,
    pub scope: String,
    pub before: Option<f64>,
    pub after: Option<f64>,
    pub comparable: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ExportCompletedBody {
    pub artifact_id: ArtifactId,
    pub kind: ExportKind,
    pub languages: Vec<Language>,
    pub parity_status: ParityStatus,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct FinalScores {
    pub cv: Option<f64>,
    pub holdout: Option<f64>,
    pub test: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Nondeterminism {
    pub count: DecimalU64,
    pub experiment_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct RunFinishedBody {
    pub completion_reason: CompletionReason,
    pub champion: Option<BestAlgorithm>,
    pub scores: FinalScores,
    pub budget_used: BudgetUsed,
    pub nondeterminism: Nondeterminism,
    /// Full Q1 request receipt, atomically committed with this terminal event.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation_receipt: Option<Box<EvaluationReceipt>>,
}

/// Ephemeral SSE frame: deliberately lacks `sequence` and therefore no SSE id.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TelemetryEvent {
    pub run_id: RunId,
    pub time: String,
    #[serde(flatten)]
    pub event: TelemetryBody,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", content = "body")]
pub enum TelemetryBody {
    #[serde(rename = "telemetry")]
    Telemetry(Vec<MetricSample>),
}
