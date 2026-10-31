//! Progress events + sinks. Events are emitted by every phase and consumed by the live TUI through
//! an in-process channel or by a lightweight human stderr sink.
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::sync::Mutex;

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

/// One per-cell confidence update from the encoding sweep, serialized compactly as
/// `[row, col, confidence]` (tuple struct → JSON array, ~60% smaller than an object).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SweepCell(pub u32, pub u32, pub f32);

/// One live frame from the phase-0 HYPERION composite sweep — a SIDE-CHANNEL type, deliberately NOT
/// part of the governed [`Event`] contract (no schema-version bump; the web nests it under a
/// `"sweep"` payload key, mirroring the `reasoning.note` precedent).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SweepFrame {
    /// Current sweep index k (0-based).
    pub sweep: u32,
    /// Total sweeps K.
    pub sweeps: u32,
    /// Fraction of THIS sweep completed.
    pub frac: f64,
    /// Overall encode progress in \[0,1\] (rescaled across the train/hold share) — drives the UI %.
    pub done_frac: f64,
    /// Original matrix dims (target column dropped).
    pub rows: u64,
    pub cols: u64,
    /// Coordinate space of `cells` (== rows/cols when small; UI-grid-binned for large matrices).
    pub grid_rows: u32,
    pub grid_cols: u32,
    /// This chunk's running-mean confidence updates at ORIGINAL (or grid-binned) positions, in \[0,1\].
    pub cells: Vec<SweepCell>,
    /// Current running 1×128 dataset vector = mean over rows of the running row embeddings.
    pub vector: Vec<f32>,
    /// Seconds since pipeline start.
    pub t: f64,
}

/// A progress sink. Must be Send+Sync (held by the driver across phases).
pub trait Sink: Send + Sync {
    fn emit(&self, e: &Event);
    /// Cooperative cancellation hook. The pipeline checks this at phase/generation boundaries and
    /// halts promptly. Defaults to `false` so existing sinks are unaffected.
    fn is_cancelled(&self) -> bool {
        false
    }
    /// Cooperative one-shot phase advance hook. A sink returns true when the current long-running
    /// phase should stop at its next safe boundary and let the pipeline continue with partial/best
    /// results. This is intentionally separate from hard cancellation.
    fn take_advance_request(&self, _phase: u8) -> bool {
        false
    }
    /// Phase-0 encode side-channel: per-chunk sweep frames from the HYPERION composite sweep.
    /// Default no-op so existing sinks are unaffected. NOTE: wrapper sinks (e.g. the report tee)
    /// MUST forward this explicitly or frames are silently dropped.
    fn emit_sweep(&self, _f: &SweepFrame) {}
}

/// Discards events.
pub struct NullSink;
impl Sink for NullSink {
    fn emit(&self, _e: &Event) {}
}

pub struct RecordingSink<'a> {
    inner: &'a dyn Sink,
    events: Mutex<Vec<Event>>,
}

impl<'a> RecordingSink<'a> {
    pub fn new(inner: &'a dyn Sink) -> Self {
        Self {
            inner,
            events: Mutex::new(Vec::new()),
        }
    }

    pub fn events(&self) -> Vec<Event> {
        self.events
            .lock()
            .expect("progress recorder poisoned")
            .clone()
    }
}

impl Sink for RecordingSink<'_> {
    fn emit(&self, e: &Event) {
        self.events
            .lock()
            .expect("progress recorder poisoned")
            .push(e.clone());
        self.inner.emit(e);
    }

    fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }

    fn take_advance_request(&self, phase: u8) -> bool {
        self.inner.take_advance_request(phase)
    }

    fn emit_sweep(&self, f: &SweepFrame) {
        self.inner.emit_sweep(f);
    }
}

/// Machine-readable sink: serializes each `Event` as one JSON line (JSONL). Used by the SageMaker
/// container path (`JAIN_PROGRESS=jsonl`). The `json diagnostics` stream lets
/// log consumers parse typed stage, ETA, metric, and model-diagnostic events.
/// The `Event` struct already derives `Serialize`; optional fields are skipped when `None`, so each
/// line is a compact sparse object. Writes to its sink (stdout by default) and flushes per line so a
/// streaming reader sees progress live.
pub struct JsonlSink {
    writer: Mutex<Box<dyn Write + Send>>,
}

impl JsonlSink {
    /// JSONL to stdout. Human progress stays on stderr, so both streams can be merged without
    /// interleaving the two formats.
    pub fn stdout() -> Self {
        Self::new(Box::new(std::io::stdout()))
    }

    pub fn new(writer: Box<dyn Write + Send>) -> Self {
        Self {
            writer: Mutex::new(writer),
        }
    }
}

impl Sink for JsonlSink {
    fn emit(&self, e: &Event) {
        if let Ok(line) = serde_json::to_string(e) {
            if let Ok(mut w) = self.writer.lock() {
                let _ = writeln!(w, "{line}");
                let _ = w.flush();
            }
        }
    }

    fn emit_sweep(&self, f: &SweepFrame) {
        // Nested under "sweep" so JSONL consumers can distinguish frames from contract Events.
        if let Ok(line) = serde_json::to_string(&serde_json::json!({ "sweep": f })) {
            if let Ok(mut w) = self.writer.lock() {
                let _ = writeln!(w, "{line}");
                let _ = w.flush();
            }
        }
    }
}

/// Stable run metadata used only by human progress sinks.
#[derive(Clone, Debug, Default)]
pub struct RunContext {
    pub command: Option<String>,
    pub dataset: Option<String>,
    pub task: Option<String>,
    pub rows: Option<u64>,
    pub hold_rows: Option<u64>,
    pub features: Option<u64>,
    pub selection_mode: Option<String>,
}

impl RunContext {
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: Some(command.into()),
            ..Self::default()
        }
    }

    pub fn dataset(mut self, dataset: impl Into<String>) -> Self {
        self.dataset = Some(dataset.into());
        self
    }

    pub fn task(mut self, task: impl Into<String>) -> Self {
        self.task = Some(task.into());
        self
    }

    pub fn rows(mut self, train: usize, hold: usize) -> Self {
        self.rows = Some(train as u64);
        self.hold_rows = Some(hold as u64);
        self
    }

    pub fn features(mut self, features: usize) -> Self {
        self.features = Some(features as u64);
        self
    }

    pub fn selection_mode(mut self, mode: impl Into<String>) -> Self {
        self.selection_mode = Some(mode.into());
        self
    }
}

#[derive(Clone, Debug, Default)]
pub struct HumanProgressState {
    phase_started_at: [Option<f64>; 8],
    last_progress_at: Option<f64>,
    last_line: Option<String>,
}

impl HumanProgressState {
    pub fn render(
        &mut self,
        e: &Event,
        context: &RunContext,
        phase_label: &str,
        throttle_secs: f64,
    ) -> Option<String> {
        if e.event == "start" {
            if let Some(slot) = self.phase_started_at.get_mut(e.phase as usize) {
                *slot = Some(e.t);
            }
        }

        let immediate = is_immediate(e);
        if !immediate {
            if let Some(last) = self.last_progress_at {
                if e.t - last < throttle_secs {
                    return None;
                }
            }
        }

        let phase_elapsed = if e.event == "done" {
            self.phase_started_at
                .get(e.phase as usize)
                .and_then(|t| *t)
                .map(|start| (e.t - start).max(0.0))
        } else {
            None
        };
        let line = progress_line(e, context, phase_label, phase_elapsed);
        if !immediate && self.last_line.as_ref() == Some(&line) {
            return None;
        }
        if !immediate {
            self.last_progress_at = Some(e.t);
        }
        self.last_line = Some(line.clone());
        Some(line)
    }
}

/// Lightweight stderr sink for tests and diagnostics that should not depend on the CLI crate.
pub struct StderrProgressSink {
    context: RunContext,
    state: Mutex<HumanProgressState>,
}

impl StderrProgressSink {
    pub fn new(context: RunContext) -> Self {
        Self {
            context,
            state: Mutex::new(HumanProgressState::default()),
        }
    }
}

impl Sink for StderrProgressSink {
    fn emit(&self, e: &Event) {
        let phase = phase_label(e.phase);
        let mut state = self.state.lock().expect("progress lock poisoned");
        if let Some(line) = state.render(e, &self.context, phase, DEFAULT_THROTTLE_SECS) {
            eprintln!("{line}");
        }
    }
}

pub fn phase_label(phase: u8) -> &'static str {
    match phase {
        0 => "HYPERION",
        2 => "IMPORT",
        3 => "SELECT",
        4 => "MIMO",
        5 => "FINAL",
        6 => "INVENT",
        _ => "PIPE",
    }
}

fn is_immediate(e: &Event) -> bool {
    matches!(
        e.event.as_str(),
        "start" | "done" | "model" | "warning" | "warn" | "seed"
    )
}

fn progress_line(
    e: &Event,
    context: &RunContext,
    phase: &str,
    phase_elapsed: Option<f64>,
) -> String {
    let mut parts = vec![format!("[{phase}] {}", event_label(e))];
    if e.event == "seed" {
        // native jope inference seed for GP gen-0; #1 = the rank-0 (selection_key argmax). The
        // human line carries the seed count only; seed program text is never printed.
        parts.push(format!(
            "jope inference · seed #{}",
            e.gen.map(|g| g + 1).unwrap_or(1)
        ));
    }
    if e.event == "start" {
        context_parts(context, &mut parts);
    }
    if e.event != "seed" {
        if let Some(model) = &e.model {
            parts.push(format!("model={model}"));
        }
    }
    if let Some(metric) = &e.metric {
        parts.push(format!("metric={metric}"));
    }
    if let Some(status) = &e.hyperion_status {
        parts.push(format!("status={status}"));
    }
    if let Some(engine) = &e.hyperion_engine {
        parts.push(format!("engine={engine}"));
    }
    if let Some(device) = &e.hyperion_device {
        parts.push(format!("device={device}"));
    }
    if let Some(rows) = e.rows {
        parts.push(format!("rows={rows}"));
    }
    if let Some(cols) = e.cols {
        parts.push(format!("cols={cols}"));
    }
    if let Some(dim) = e.output_dim {
        parts.push(format!("output_dim={dim}"));
    }
    if let Some(preview) = &e.hyperion_preview {
        parts.push(format!(
            "preview_points={} concerns={}",
            preview.points.len(),
            preview.concern_ranges.len()
        ));
    }
    if let (Some(gen), Some(gens)) = (e.gen, e.gens) {
        parts.push(format!("gen={}/{}", gen + 1, gens));
    } else if let Some(gens) = e.gens {
        parts.push(format!("gens={gens}"));
    }
    if let (Some(done), Some(total)) = (e.done, e.blocks_total) {
        let label = if e.phase == 2 {
            "blocks"
        } else if e.event == "genprog" {
            "scored"
        } else {
            "done"
        };
        let frac = e
            .done_frac
            .unwrap_or_else(|| done as f64 / total.max(1) as f64);
        parts.push(format!("{label}={done}/{total} {}", bar(frac)));
        let elapsed = phase_elapsed_for_rate(e, phase_elapsed);
        if done > 0 && elapsed > 0.0 {
            parts.push(format!("rate={:.1}/s", done as f64 / elapsed));
        }
    }
    if let Some(eta) = e.eta {
        parts.push(format!("eta={}", fmt_secs(eta.max(0.0))));
    }
    if let Some(accepted) = e.accepted_features {
        parts.push(format!("accepted={accepted}"));
    }
    if let Some(selected) = e.selected_features.or(e.best_size) {
        parts.push(format!("selected={selected}"));
    }
    if let Some(final_features) = e.final_features {
        parts.push(format!("final_features={final_features}"));
    }
    if let Some(top) = &e.top {
        let top = top
            .iter()
            .take(3)
            .map(|(idx, pct)| format!("#{idx}:{pct:.3}"))
            .collect::<Vec<_>>()
            .join(",");
        if !top.is_empty() {
            parts.push(format!("top={top}"));
        }
    }
    if let Some(cv) = e.best_val_auc {
        parts.push(format!("cv={cv:.5}"));
    }
    if let Some(holdout) = e.best_holdout_auc {
        parts.push(format!("holdout={holdout:.5}"));
    }
    if let Some(mean) = e.mean_val_auc {
        parts.push(format!("mean_cv={mean:.5}"));
    }
    if let Some(size) = e.mean_size {
        parts.push(format!("mean_size={size}"));
    }
    if let Some(elapsed) = phase_elapsed {
        parts.push(format!("elapsed={}", fmt_secs(elapsed)));
    }
    parts.push(format!("t={}", fmt_secs(e.t)));
    parts.join(" ")
}

fn event_label(e: &Event) -> &'static str {
    match (e.event.as_str(), e.model.as_ref()) {
        ("model", _) => "candidate",
        ("done", Some(_)) if e.phase == 5 => "selected",
        ("genprog", _) => "progress",
        (other, _) => match other {
            "start" => "start",
            "done" => "done",
            "gen" => "gen",
            "block" => "progress",
            "warning" | "warn" => "warning",
            "seed" => "seed",
            _ => "event",
        },
    }
}

fn context_parts(context: &RunContext, parts: &mut Vec<String>) {
    if let Some(command) = &context.command {
        parts.push(format!("cmd={command}"));
    }
    if let Some(dataset) = &context.dataset {
        parts.push(format!("dataset={dataset}"));
    }
    if let Some(task) = &context.task {
        parts.push(format!("task={task}"));
    }
    if let Some(rows) = context.rows {
        parts.push(format!("rows={rows}"));
    }
    if let Some(hold_rows) = context.hold_rows {
        parts.push(format!("hold={hold_rows}"));
    }
    if let Some(features) = context.features {
        parts.push(format!("features={features}"));
    }
    if let Some(mode) = &context.selection_mode {
        parts.push(format!("selection={mode}"));
    }
}

fn phase_elapsed_for_rate(e: &Event, phase_elapsed: Option<f64>) -> f64 {
    phase_elapsed.unwrap_or(e.t).max(1e-9)
}

fn fmt_secs(secs: f64) -> String {
    if secs >= 60.0 {
        let total = secs.round() as u64;
        format!("{}m{:02}s", total / 60, total % 60)
    } else {
        format!("{secs:.1}s")
    }
}

fn bar(frac: f64) -> String {
    let width = 18usize;
    let fill = (frac.clamp(0.0, 1.0) * width as f64).round() as usize;
    format!(
        "[{}{}] {:>3.0}%",
        "#".repeat(fill),
        ".".repeat(width.saturating_sub(fill)),
        100.0 * frac.clamp(0.0, 1.0)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_sink_round_trips_an_event() {
        // The SDK parses these lines; prove a representative Event serializes to one JSON line and
        // deserializes back field-for-field (sparse: None fields are absent).
        let mut e = Event::new(5, "model", 41.7);
        e.done_frac = Some(0.5);
        e.eta = Some(12.5);
        e.best_val_auc = Some(0.71234);
        e.best_holdout_auc = Some(0.70011);
        e.best_size = Some(19);
        e.model = Some("catboost".to_string());
        e.metric = Some("auc".to_string());
        e.rows = Some(100);
        e.cols = Some(12);
        e.output_dim = Some(128);
        e.hyperion_status = Some("used".to_string());
        e.hyperion_engine = Some("v7".to_string());
        e.hyperion_device = Some("cpu".to_string());
        e.hyperion_preview = Some(HyperionPreview {
            rows: 100,
            threshold_z: 3.5,
            points: vec![HyperionPreviewPoint {
                row: 42,
                score: 12.5,
            }],
            top_rows: vec![HyperionConcernRow {
                row: 42,
                score: 12.5,
                z: 4.1,
            }],
            concern_ranges: vec![HyperionConcernRange {
                start: 41,
                end: 43,
                max_score: 12.5,
                max_z: 4.1,
            }],
        });

        let buf = std::sync::Arc::new(Mutex::new(Vec::<u8>::new()));
        struct SharedWriter(std::sync::Arc<Mutex<Vec<u8>>>);
        impl Write for SharedWriter {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let sink = JsonlSink::new(Box::new(SharedWriter(buf.clone())));
        sink.emit(&e);

        let out = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
        assert!(out.ends_with('\n'), "each event is a line");
        assert_eq!(out.lines().count(), 1);
        let back = Event::from_json_line(out.trim()).unwrap();
        assert_eq!(back.phase, 5);
        assert_eq!(back.event, "model");
        assert_eq!(back.t, 41.7);
        assert_eq!(back.done_frac, Some(0.5));
        assert_eq!(back.eta, Some(12.5));
        assert_eq!(back.best_val_auc, Some(0.71234));
        assert_eq!(back.best_holdout_auc, Some(0.70011));
        assert_eq!(back.best_size, Some(19));
        assert_eq!(back.model.as_deref(), Some("catboost"));
        assert_eq!(back.v, EVENT_SCHEMA_VERSION);
        assert_eq!(back.metric.as_deref(), Some("auc"));
        assert_eq!(back.rows, Some(100));
        assert_eq!(back.cols, Some(12));
        assert_eq!(back.output_dim, Some(128));
        assert_eq!(back.hyperion_status.as_deref(), Some("used"));
        assert_eq!(back.hyperion_engine.as_deref(), Some("v7"));
        assert_eq!(back.hyperion_device.as_deref(), Some("cpu"));
        assert_eq!(
            back.hyperion_preview.as_ref().map(|p| p.points.len()),
            Some(1)
        );
        // Track the constant instead of a hard-coded number so schema bumps can't strand this test.
        assert!(
            out.contains(&format!("\"v\":{EVENT_SCHEMA_VERSION}")),
            "schema version always serialized"
        );
    }

    fn context() -> RunContext {
        RunContext::new("bench")
            .dataset("demo.csv")
            .task("Binary")
            .rows(100, 25)
            .features(12)
            .selection_mode("mvp-fast")
    }

    #[test]
    fn human_progress_formats_phase2_mvp_and_final_events() {
        let ctx = context();
        let mut state = HumanProgressState::default();

        let mut e = Event::new(2, "block", 3.0);
        e.done = Some(4);
        e.blocks_total = Some(10);
        e.done_frac = Some(0.4);
        e.eta = Some(6.0);
        let line = state
            .render(&e, &ctx, "IMPORT", DEFAULT_THROTTLE_SECS)
            .unwrap();
        assert!(line.contains("blocks=4/10"));
        assert!(line.contains("eta=6.0s"));

        let mut e = Event::new(3, "done", 5.0);
        e.model = Some("mvp-fast".into());
        e.accepted_features = Some(20);
        e.selected_features = Some(7);
        e.final_features = Some(19);
        let line = state
            .render(&e, &ctx, "SELECT", DEFAULT_THROTTLE_SECS)
            .unwrap();
        assert!(line.contains("accepted=20"));
        assert!(line.contains("selected=7"));
        assert!(line.contains("final_features=19"));

        let mut e = Event::new(5, "model", 6.0);
        e.model = Some("catboost".into());
        e.best_val_auc = Some(0.71);
        let line = state
            .render(&e, &ctx, "FINAL", DEFAULT_THROTTLE_SECS)
            .unwrap();
        assert!(line.contains("[FINAL] candidate"));
        assert!(line.contains("model=catboost"));
    }

    #[test]
    fn progress_v12_rejects_retired_phase_and_prior_schema() {
        let phase_one = r#"{"phase":1,"event":"start","t":0.0,"v":12}"#;
        let error = Event::from_json_line(phase_one).unwrap_err();
        assert!(error.contains("phase 1 was retired"));
        assert!(error.contains("retrain with Jain 10.0.0"));

        let v11 = r#"{"phase":2,"event":"start","t":0.0,"v":11}"#;
        let error = Event::from_json_line(v11).unwrap_err();
        assert!(error.contains("expected v12"));
        assert!(error.contains("retrain with Jain 10.0.0"));
    }

    #[test]
    fn human_progress_throttles_repetitive_progress_but_not_done() {
        let ctx = context();
        let mut state = HumanProgressState::default();
        let mut e = Event::new(2, "block", 1.0);
        e.done = Some(1);
        e.blocks_total = Some(10);
        assert!(state
            .render(&e, &ctx, "IMPORT", DEFAULT_THROTTLE_SECS)
            .is_some());

        let mut e2 = Event::new(2, "block", 1.5);
        e2.done = Some(2);
        e2.blocks_total = Some(10);
        assert!(state
            .render(&e2, &ctx, "IMPORT", DEFAULT_THROTTLE_SECS)
            .is_none());

        let mut done = Event::new(2, "done", 1.6);
        done.done = Some(10);
        done.blocks_total = Some(10);
        assert!(state
            .render(&done, &ctx, "IMPORT", DEFAULT_THROTTLE_SECS)
            .is_some());
    }

    #[test]
    fn human_seed_line_carries_the_count_but_never_the_genome() {
        let ctx = context();
        let mut state = HumanProgressState::default();
        let genome = "(pipe X (robust_scale) (histgbm depth=3 lr=0.1 rounds=40))";
        let mut e = Event::new(6, "seed", 2.0);
        e.genome = Some(genome.to_string());
        e.gen = Some(2);
        e.model = Some("jope".to_string());
        let line = state
            .render(&e, &ctx, "INVENT", DEFAULT_THROTTLE_SECS)
            .expect("seed events render immediately");
        assert!(line.contains("jope inference · seed #3"), "{line}");
        assert!(!line.contains(genome), "{line}");
        for token in ["pipe", "histgbm", "robust_scale", "("] {
            assert!(!line.contains(token), "genome text leaked: {line}");
        }
    }
}
