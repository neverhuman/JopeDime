pub const DEFAULT_THROTTLE_SECS: f64 = 1.0;

/// Wire-schema version of [`Event`]. Bumped whenever fields are added/changed so the web cockpit and
/// the Python SDK can detect skew. Committed schema lives in `contracts/progress-event.schema.json`.
pub const EVENT_SCHEMA_VERSION: u32 = 13;
pub const MIN_READABLE_EVENT_SCHEMA_VERSION: u32 = 12;

/// One important base column surfaced at the end of a run.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EnrichedFeature {
    /// Human-readable transformed input name.
    pub name: String,
    /// Normalized importance mass (memory-vector weight; higher = more important).
    pub importance: f64,
    /// Origin. Jain 10.0.0 emits only `"base"`.
    #[serde(deserialize_with = "deserialize_base_feature_source")]
    pub source: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HyperionPreviewPoint {
    pub row: u64,
    pub score: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HyperionConcernRow {
    pub row: u64,
    pub score: f64,
    pub z: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HyperionConcernRange {
    pub start: u64,
    pub end: u64,
    pub max_score: f64,
    pub max_z: f64,
}

/// Compact row-level preview for Hyperion output. It carries only summed Hyperion values and row
/// indices; raw row contents and full row-level arrays are deliberately not streamed.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HyperionPreview {
    pub rows: u64,
    pub threshold_z: f64,
    pub points: Vec<HyperionPreviewPoint>,
    pub top_rows: Vec<HyperionConcernRow>,
    pub concern_ranges: Vec<HyperionConcernRange>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GpuBadge {
    pub available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub free_mb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_mb: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
}

/// One progress record. Valid phases are 0, 2, 3, 4, 5, and 6. Phase 1 is retired and is never
/// renumbered or reused.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Event {
    pub phase: u8,     // 0=HYPERION(encode) | 2..=6, excluding retired phase 1
    pub event: String, // "start" | "gen" | "genprog" | "block" | "model" | "eval" | "improved" | "done" | "warning"
    pub t: f64,        // seconds since pipeline start

    // generation counters (p3/p4/p6)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gen: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gens: Option<u64>,

    // Phase 2 (importance epochs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocks_total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_frac: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top: Option<Vec<(u32, f64)>>,

    // Phase 3/4 (selection)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_val_auc: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_holdout_auc: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_val_auc: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_size: Option<u64>,

    // Fine-grained progress / estimates
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eta: Option<f64>, // estimated seconds remaining (for this phase)
    // Final model selection (phase 5): which model this record is about / the selected winner.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    // Fast-selection feature counts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accepted_features: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_features: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_features: Option<u64>,

    /// Task-aware label ("auc" | "r2") for the score carried in `best_val_auc`/`best_holdout_auc`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>,
    /// Phase-0 Hyperion/scoring row count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rows: Option<u64>,
    /// Phase-0 Hyperion input feature column count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cols: Option<u64>,
    /// Phase-0 Hyperion output embedding dimension.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_dim: Option<u64>,
    /// Phase-0 Hyperion status: "auto" | "on" | "off" | "used" | "skipped" | "failed".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperion_status: Option<String>,
    /// Phase-0 Hyperion engine: "v7" | "projection".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperion_engine: Option<String>,
    /// Phase-0 compute device used/intended by the engine: "cpu" | "cuda:0".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperion_device: Option<String>,
    /// Concrete cellmat backend and timing/recovery telemetry for the encoding stage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperion_backend: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperion_elapsed_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperion_fallback: Option<bool>,
    /// Phase-0 compact row-sum concern preview.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperion_preview: Option<HyperionPreview>,
    /// Number of CV folds used by a scored event. Added for the Hyperion gate probe.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cv_folds: Option<u64>,
    /// Hyperion probe sweep count. The standard gate always probes with 3 sweeps.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperion_probe_sweeps: Option<u64>,
    /// Hyperion deep sweep budget used after a passing gate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperion_deep_sweeps: Option<u64>,
    /// Candidate-vs-base Jboost CV lift from the Hyperion gate probe.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperion_lift: Option<f64>,
    /// True only when the strict Hyperion lift gate passed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperion_gate_passed: Option<bool>,
    /// True when Hyperion coordinates were available to the separate outlier-scoring path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperion_outlier_signal: Option<bool>,

    // Schema v5 additions (end-of-training payoff).
    /// Per-fold CV scores for the model named by `model` (phase 5). Length = number of CV folds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub folds: Option<Vec<f64>>,
    /// Vanilla holdout score: the final metric of a single model trained on the RAW base features only,
    /// on the same holdout. Paired with `best_holdout_auc` to show the feature-engineering lift.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_auc: Option<f64>,
    /// Top important base columns at end of run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_features: Option<Vec<EnrichedFeature>>,

    // Schema v6 additions (phase-6 invention: pipeline-genome search).
    /// Full emitted Python source for a phase-6 search candidate, live improvement, or final CV winner.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_code: Option<String>,
    /// Canonical phase-6 DSL for a search candidate, live improvement, finalist, or final CV winner.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub genome: Option<String>,
    /// Cumulative genomes evaluated by the invention search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evals: Option<u64>,
    /// Live genome evaluations per second.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evals_per_sec: Option<f64>,
    /// Completed algorithm evaluations per second. `evals_per_sec` remains as a compatibility
    /// alias for older consumers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub algorithms_per_sec: Option<f64>,
    /// Cumulative full-artifact token credit (each evaluated genome counts as its complete emitted
    /// Python + Rust source).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_total: Option<f64>,
    /// Live token-credit rate per second.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_sec: Option<f64>,
    /// Current population fitness snapshot (sorted desc, downsampled) for live visuals.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pop_fitness: Option<Vec<f64>>,
    /// Human-readable note (e.g. why the phase-6 invention stage was skipped) — shown
    /// verbatim in the cockpit so a skipped/aborted stage is never silent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// Runtime GPU capability badge attached to the phase-6 invention stream.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu: Option<GpuBadge>,
    /// CV score of the individual candidate carried by a phase-6 `"eval"` churn event (kept
    /// separate from `best_val_auc` so it never pollutes the best-over-generations curve).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_score: Option<f64>,

    // Schema v13 additions (Rust-authoritative model diagnostics).
    /// Typed model-diagnostics/v1 snapshot. Point estimates may be followed by a higher revision
    /// carrying deterministic uncertainty intervals.
    #[serde(skip_serializing_if = "Option::is_none")]
    // Port note: feat-core types this as jain-math's `feat_math::DiagnosticBundle`; see the
    // `crate::diagnostics` shim.
    pub diagnostics: Option<crate::diagnostics::DiagnosticBundle>,
    /// Task-correct primary CV score. Legacy `best_val_auc` remains during the compatibility window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_cv: Option<f64>,
    /// Task-correct primary holdout score. Legacy `best_holdout_auc` remains compatible.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_holdout: Option<f64>,
    /// Canonical primary metric ID (`binary.roc_auc` or `regression.r2`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metric_name: Option<String>,

    /// Event schema version (always serialized). See [`EVENT_SCHEMA_VERSION`].
    #[serde(default)]
    pub v: u32,
}

fn deserialize_base_feature_source<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let source = String::deserialize(deserializer)?;
    if source != "base" {
        return Err(serde::de::Error::custom(format!(
            "progress top_features source must be \"base\", found {source:?}; {}",
            crate::RETRAIN_WITH_JAIN_10
        )));
    }
    Ok(source)
}

impl Event {
    pub fn new(phase: u8, event: &str, t: f64) -> Self {
        Event {
            phase,
            event: event.to_string(),
            t,
            v: EVENT_SCHEMA_VERSION,
            ..Default::default()
        }
    }

    /// Enforce the v12/v13 wire contract at every decoding boundary.
    pub fn validate_contract(&self) -> Result<(), String> {
        if !(MIN_READABLE_EVENT_SCHEMA_VERSION..=EVENT_SCHEMA_VERSION).contains(&self.v) {
            return Err(format!(
                "incompatible progress schema v{}; expected v{} or v{}; {}",
                self.v,
                MIN_READABLE_EVENT_SCHEMA_VERSION,
                EVENT_SCHEMA_VERSION,
                crate::RETRAIN_WITH_JAIN_10
            ));
        }
        if self.v == MIN_READABLE_EVENT_SCHEMA_VERSION
            && (self.diagnostics.is_some()
                || self.primary_cv.is_some()
                || self.primary_holdout.is_some()
                || self.metric_name.is_some())
        {
            return Err(format!(
                "progress v12 cannot carry v13 diagnostic fields; {}",
                crate::RETRAIN_WITH_JAIN_10
            ));
        }
        if let Some(value) = self
            .primary_cv
            .into_iter()
            .chain(self.primary_holdout)
            .find(|value| !value.is_finite())
        {
            return Err(format!(
                "progress primary score must be finite, found {value}; {}",
                crate::RETRAIN_WITH_JAIN_10
            ));
        }
        if let Some(diagnostics) = &self.diagnostics {
            diagnostics.validate().map_err(|error| {
                format!(
                    "invalid model-diagnostics/v1 payload: {error}; {}",
                    crate::RETRAIN_WITH_JAIN_10
                )
            })?;
        }
        if !matches!(self.phase, 0 | 2 | 3 | 4 | 5 | 6) {
            return Err(format!(
                "incompatible progress phase {}; phase 1 was retired in Jain 10.0.0; {}",
                self.phase,
                crate::RETRAIN_WITH_JAIN_10
            ));
        }
        if let Some(engine) = self
            .hyperion_engine
            .as_deref()
            .filter(|engine| !matches!(*engine, "v7" | "projection"))
        {
            return Err(format!(
                "progress hyperion_engine must be \"v7\" or \"projection\", found {engine:?}; {}",
                crate::RETRAIN_WITH_JAIN_10
            ));
        }
        if let Some(feature) = self
            .top_features
            .as_ref()
            .and_then(|features| features.iter().find(|feature| feature.source != "base"))
        {
            return Err(format!(
                "progress top_features source must be \"base\", found {:?}; {}",
                feature.source,
                crate::RETRAIN_WITH_JAIN_10
            ));
        }
        Ok(())
    }

    pub fn from_json_line(line: &str) -> Result<Self, String> {
        let event: Self = serde_json::from_str(line).map_err(|error| {
            format!(
                "invalid progress-v12-or-v13 JSON: {error}; {}",
                crate::RETRAIN_WITH_JAIN_10
            )
        })?;
        event.validate_contract()?;
        Ok(event)
    }
}
