//! Measured train-side OOF admission. A candidate supplies predictions, never eligibility policy
//! or trusted metric values. The policy, targets and trivial baseline are persisted before optimization.
use crate::{
    canon::{canonical_json, canonical_sha256, is_hex64, sha256_hex},
    ledger::{now_unix, Ledger},
    metric::Metric,
    record::{HwmIdentity, HwmRecord},
    state::{PointerKey, State},
};
use anyhow::{bail, ensure, Context, Result};
use jope_datasets::catalog::Task;
use jope_pipeline::evaluation_score::{score_predictions, Predictions, NUMERICAL_POLICY};
use jope_pipeline::execution::{Artifact, WorkerJob, WorkerOutcome, EXECUTION_SCHEMA};
use jope_pipeline::experiment::{
    EvaluationRequest, EvaluationStage, EvaluationStatus, ExecutionWindow, Experiment, Panel,
    PanelDataset,
};
use jope_pipeline::experiment_v2::EvaluationResult;
use jope_pipeline::prepared_data::parse_development;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::Path,
};

mod context;
mod inner;
pub use context::{ContextEvidence, ContextPlan};
pub use inner::{
    derive_inner_partition, freeze_inner_optimization, register_inner_candidate, InnerEvidence,
    InnerPartition,
};

const FREEZE: &str = "jope.optimization-freeze/v1";
const PREDICTIONS: &str = "jope.train-oof-predictions/v1";
const QUALIFICATION: &str = "jope.measured-teaching/v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetRow {
    pub id: String,
    pub group: String,
    pub fold: u8,
    pub target: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FreezePlan {
    pub comparison: HwmIdentity,
    /// `test` observed by optimization is development data; sealed data is never eligible.
    pub original_label_role: String,
    pub panel_sha256: String,
    pub dataset_id: String,
    pub evaluation_seed: u64,
    /// Required for teaching admission; a freeze without it is diagnostic only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<ContextPlan>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenFold {
    pub fold: u8,
    pub training_rows_sha256: String,
    pub scoring_rows_sha256: String,
    pub baseline_predictions_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FrozenPolicy {
    schema: String,
    plan_sha256: String,
    comparison: HwmIdentity,
    label_role: String,
    panel_sha256: String,
    dataset: PanelDataset,
    original_label_role: String,
    label_burn_sha256: Option<String>,
    guard: GuardPolicy,
    evaluation_seed: u64,
    targets_sha256: String,
    folds: Vec<FrozenFold>,
    created_at_unix: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    inner_partition: Option<InnerPartition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    context: Option<ContextPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FreezeReceipt {
    pub comparison_sha256: String,
    pub freeze_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoldPredictions {
    pub schema: String,
    pub comparison_sha256: String,
    pub program_sha256: String,
    pub fold: u8,
    pub training_rows_sha256: String,
    pub scoring_rows_sha256: String,
    pub row_ids: Vec<String>,
    pub predictions_sha256: String,
    pub evaluation_request_sha256: String,
    pub worker_job_sha256: String,
    pub fitted_artifact_sha256: String,
    pub execution_receipt_sha256: String,
    pub started_receipt_sha256: String,
    pub fit_started_unix: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OofEvidence {
    pub freeze: FreezeReceipt,
    pub experiment_blob_sha256: String,
    pub window_sha256: String,
    /// Exactly one content-addressed FoldPredictions blob per fold, in fold order.
    pub fold_predictions_sha256: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inner: Option<InnerEvidence>,
    /// Mandatory for qualification; absent only on legacy/diagnostic submissions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<ContextEvidence>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qualification {
    pub schema: String,
    pub comparison: HwmIdentity,
    pub freeze: FreezeReceipt,
    pub record_sha256: String,
    pub program_sha256: String,
    pub targets_sha256: String,
    pub baseline_predictions_sha256: Vec<String>,
    pub fold_predictions_sha256: Vec<String>,
    pub guard: GuardPolicy,
    pub label_role: String,
    pub original_label_role: String,
    pub label_burn_sha256: Option<String>,
    pub candidate_loss: f64,
    pub baseline_loss: f64,
    pub improved_folds: usize,
    pub fold_losses: Vec<(f64, f64)>,
    /// Scores were recomputed; process execution provenance still comes from the worker receipts.
    pub measurement: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_fitting_rows_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_outer_fold: Option<u8>,
    /// Verified context receipt; stored legacy qualifications without one never revalidate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_receipt_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredQualification {
    evidence: OofEvidence,
    qualification: Qualification,
}

#[derive(Clone, Debug, Serialize)]
pub struct TeachingManifest {
    pub schema: &'static str,
    pub excluded_diagnostic: usize,
    pub exclusions: Vec<(String, String)>,
    pub entries: Vec<Qualification>,
}

/// Fixed, versioned admission policy. Caller evidence never supplies thresholds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuardPolicy {
    pub schema: String,
    pub folds: usize,
    pub minimum_improved_folds: usize,
    pub aggregation: String,
    pub baseline: String,
    pub absolute_gain: f64,
    pub relative_gain: f64,
}
impl GuardPolicy {
    fn current() -> Self {
        Self {
            schema: "jope.teaching-guard/v1".into(),
            folds: 5,
            minimum_improved_folds: 4,
            aggregation: "mean_fold_loss".into(),
            baseline: "training_mean_or_class_frequency".into(),
            absolute_gain: 1e-12,
            relative_gain: 1e-6,
        }
    }
    fn improves(&self, candidate: f64, baseline: f64) -> bool {
        candidate.is_finite()
            && baseline.is_finite()
            && baseline - candidate > self.absolute_gain + self.relative_gain * baseline.abs()
    }
    fn admits(&self, measured: &Qualification) -> bool {
        measured.fold_losses.len() == self.folds
            && measured.improved_folds >= self.minimum_improved_folds
            && self.improves(measured.candidate_loss, measured.baseline_loss)
    }
}

fn authenticated_rows(state: &State, plan: &FreezePlan) -> Result<(PanelDataset, Vec<TargetRow>)> {
    let panel: Panel = blob(state, &plan.panel_sha256)?;
    panel.validate()?;
    let dataset = panel
        .datasets
        .into_iter()
        .find(|dataset| dataset.id == plan.dataset_id)
        .context("dataset absent from frozen panel")?;
    ensure!(
        dataset.source.sha256 == plan.comparison.dataset_hash
            && dataset.split.sha256 == plan.comparison.cut_id
            && dataset.task.as_str() == plan.comparison.task
            && dataset.classes
                == plan
                    .comparison
                    .comparison
                    .as_ref()
                    .context("missing comparison")?
                    .class_order,
        "comparison differs from authenticated panel dataset"
    );
    let bytes = state
        .read_blob(&dataset.split.development_sha256)?
        .context("prepared development bytes unavailable")?;
    let data = parse_development(&bytes, &dataset)?;
    let rows = data
        .rows
        .into_iter()
        .map(|row| TargetRow {
            id: row.id,
            group: row.group,
            fold: row.fold,
            target: row.target,
        })
        .collect();
    Ok((dataset, rows))
}

fn burn_path(state: &State, dataset: &str, cut: &str) -> Result<std::path::PathBuf> {
    ensure!(
        is_hex64(dataset) && is_hex64(cut),
        "invalid label partition identity"
    );
    Ok(state
        .root()
        .join("eligibility/burns")
        .join(dataset)
        .join(format!("{cut}.json")))
}
/// Test-headline callers must check this before presenting a partition as untouched.
pub fn require_unburned_test_partition(state: &State, dataset: &str, cut: &str) -> Result<()> {
    let ledger = Ledger::open(state.root(), "eligibility");
    ledger.require_intact()?;
    let mut burned = burn_path(state, dataset, cut)?.exists();
    for event in ledger.events()? {
        if let Some(sha) = event
            .get("label_burn_sha256")
            .and_then(|value| value.as_str())
        {
            let receipt: serde_json::Value = blob(state, sha)?;
            burned |= receipt
                .get("dataset_sha256")
                .and_then(|value| value.as_str())
                == Some(dataset)
                && receipt.get("cut_id").and_then(|value| value.as_str()) == Some(cut);
        }
    }
    ensure!(
        !burned,
        "partition labels were consumed by optimization and are development data"
    );
    Ok(())
}

pub fn effective_label_role(original: &str, optimization_visible: bool) -> Result<&'static str> {
    match (original, optimization_visible) {
        ("sealed" | "meta_test", _) => bail!("sealed labels cannot enter optimization or teaching"),
        ("train", _) => Ok("train"),
        ("development" | "validation", _) | ("test", true) => Ok("development"),
        ("test", false) => Ok("test"),
        _ => bail!("unknown label role"),
    }
}

fn task(identity: &HwmIdentity) -> Result<Task> {
    let context = identity
        .comparison
        .as_ref()
        .context("legacy comparison is diagnostic only")?;
    ensure!(
        context.numerical_policy == NUMERICAL_POLICY
            && context.stage == "train_oof"
            && context.aggregation == "mean_fold_loss"
            && context.label_role == "development"
            && context.recipe_encoding == "native-canonical-dsl-sha256/v1"
            && !context.sealed
            && !context.benchmark_valid,
        "numerical policy differs from scorer"
    );
    ensure!(
        is_hex64(&identity.evaluator_sha256)
            && is_hex64(&identity.vocabulary_sha256)
            && is_hex64(&identity.dataset_hash)
            && is_hex64(&identity.cut_id)
            && !identity.protocol.trim().is_empty(),
        "incomplete comparison identity"
    );
    ensure!(
        context
            .checkpoint_sha256
            .as_ref()
            .is_none_or(|sha| is_hex64(sha)),
        "invalid checkpoint identity"
    );
    match identity.task.as_str() {
        "regression" => Ok(Task::Regression),
        "binary" => Ok(Task::Binary),
        "multiclass" => Ok(Task::Multiclass),
        _ => bail!("task has no admitted OOF scorer"),
    }
}

fn blob<T: for<'de> Deserialize<'de>>(state: &State, sha: &str) -> Result<T> {
    ensure!(is_hex64(sha), "invalid artifact identity");
    let bytes = state
        .read_blob(sha)?
        .context("required prediction/target artifact is missing")?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn require_blob(state: &State, sha: &str) -> Result<()> {
    ensure!(is_hex64(sha), "invalid fitted artifact identity");
    ensure!(
        !state
            .read_blob(sha)?
            .context("required fitted evidence is missing")?
            .is_empty(),
        "empty fitted evidence"
    );
    Ok(())
}

fn freeze_path(state: &State, identity: &str) -> Result<std::path::PathBuf> {
    ensure!(is_hex64(identity), "invalid comparison digest");
    Ok(state
        .root()
        .join("eligibility/freezes")
        .join(identity)
        .join("freeze.json"))
}

fn immutable_write(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::create_dir_all(path.parent().context("immutable output parent missing")?)?;
    if path.exists() {
        ensure!(
            fs::read(path)? == bytes,
            "immutable eligibility evidence already differs"
        );
        return Ok(());
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::File::open(path.parent().unwrap())?.sync_all()?;
    Ok(())
}

fn fold_rows(rows: &[TargetRow], fold: u8) -> (Vec<&TargetRow>, Vec<&TargetRow>) {
    rows.iter().partition(|row| row.fold != fold)
}

fn rows_sha(rows: &[&TargetRow]) -> Result<String> {
    canonical_sha256(&rows.iter().map(|row| &row.id).collect::<Vec<_>>())
}

fn baseline(
    task: Task,
    classes: &[String],
    fitting: &[&TargetRow],
    n: usize,
) -> Result<Predictions> {
    ensure!(!fitting.is_empty() && n > 0, "empty OOF partition");
    match task {
        Task::Regression => {
            let ys = fitting
                .iter()
                .map(|row| row.target.parse::<f64>().context("invalid target"))
                .collect::<Result<Vec<_>>>()?;
            ensure!(ys.iter().all(|y| y.is_finite()), "nonfinite target");
            let mean = ys.iter().map(|y| y / ys.len() as f64).sum::<f64>();
            ensure!(mean.is_finite(), "nonfinite trivial mean");
            Ok(Predictions::Values(vec![mean; n]))
        }
        Task::Binary | Task::Multiclass => {
            let mut counts = vec![0usize; classes.len()];
            for row in fitting {
                let index = classes
                    .iter()
                    .position(|class| class == &row.target)
                    .context("unknown training class")?;
                counts[index] += 1;
            }
            ensure!(
                counts.iter().all(|count| *count > 0),
                "training fold lacks a frozen class"
            );
            let probabilities = counts
                .iter()
                .map(|count| *count as f64 / fitting.len() as f64)
                .collect();
            Ok(Predictions::Probabilities {
                classes: classes.to_vec(),
                values: vec![probabilities; n],
            })
        }
        _ => bail!("unsupported trivial baseline task"),
    }
}

/// Call before optimization. Repeating the same plan retains the original receipt/time;
/// no plan is reconstructed from a candidate record, and differing plans cannot overwrite it.
pub fn freeze_optimization(state: &State, plan: &FreezePlan) -> Result<FreezeReceipt> {
    freeze_impl(state, plan, None)
}
fn freeze_impl(
    state: &State,
    plan: &FreezePlan,
    inner_partition: Option<InnerPartition>,
) -> Result<FreezeReceipt> {
    let selected_task = task(&plan.comparison)?;
    if let Some(context) = &plan.context {
        context.validate()?;
    }
    let classes = &plan.comparison.comparison.as_ref().unwrap().class_order;
    let label_role = effective_label_role(&plan.original_label_role, true)?;
    ensure!(
        label_role == plan.comparison.comparison.as_ref().unwrap().label_role,
        "frozen label role differs from comparison identity"
    );
    let identity = plan.comparison.sha256()?;
    let (dataset, rows) = match &inner_partition {
        Some(inner) => inner::authenticated_inner_rows(state, plan, inner)?,
        None => authenticated_rows(state, plan)?,
    };
    let plan_sha256 = match &inner_partition {
        Some(inner) => canonical_sha256(&(plan, inner))?,
        None => canonical_sha256(plan)?,
    };
    let path = freeze_path(state, &identity)?;
    let _hwm_lock = state.lock("hwm")?;
    let _lock = state.lock("eligibility")?;
    Ledger::open(state.root(), "eligibility").require_intact()?;
    if path.exists() {
        let bytes = fs::read(&path)?;
        let existing: FrozenPolicy = serde_json::from_slice(&bytes)?;
        ensure!(
            existing.schema == FREEZE && existing.plan_sha256 == plan_sha256,
            "optimization freeze is already pinned to different inputs"
        );
        let receipt = FreezeReceipt {
            comparison_sha256: identity,
            freeze_sha256: sha256_hex(&bytes),
        };
        load_freeze(state, &receipt)?;
        state.put_blob(&bytes)?;
        return Ok(receipt);
    }
    ensure!(
        !state
            .pointers()?
            .iter()
            .any(|(key, _)| key.identity_sha256.as_ref() == Some(&identity)),
        "cannot freeze policy after an incumbent exists"
    );
    let mut ids = BTreeSet::new();
    let mut groups = BTreeMap::new();
    ensure!(!rows.is_empty(), "no training rows");
    for row in &rows {
        ensure!(
            !row.id.is_empty() && ids.insert(&row.id) && !row.group.is_empty() && row.fold < 5,
            "invalid/duplicate OOF row"
        );
        ensure!(
            *groups.entry(&row.group).or_insert(row.fold) == row.fold,
            "group crosses OOF folds"
        );
    }
    let mut folds = Vec::new();
    for fold in 0..5 {
        let (train, hold) = fold_rows(&rows, fold);
        let predictions = baseline(selected_task, classes, &train, hold.len())?;
        let targets = hold
            .iter()
            .map(|row| row.target.clone())
            .collect::<Vec<_>>();
        score_predictions(selected_task, classes, &targets, &predictions)?;
        folds.push(FrozenFold {
            fold,
            training_rows_sha256: rows_sha(&train)?,
            scoring_rows_sha256: rows_sha(&hold)?,
            baseline_predictions_sha256: state.put_blob(&canonical_json(&predictions)?)?,
        });
    }
    let label_burn_sha256 = if plan.original_label_role == "test" {
        let bytes = canonical_json(&serde_json::json!({"schema":"jope.label-burn/v1",
            "dataset_sha256":plan.comparison.dataset_hash, "cut_id":dataset.split.sha256,
            "panel_sha256":plan.panel_sha256, "development_sha256":dataset.split.development_sha256,
            "original_role":"test", "effective_role":"development"}))?;
        immutable_write(
            &burn_path(state, &plan.comparison.dataset_hash, &dataset.split.sha256)?,
            &bytes,
        )?;
        Some(state.put_blob(&bytes)?)
    } else {
        None
    };
    if let Some(inner) = &inner_partition {
        state.put_blob(&canonical_json(&inner.fitting_rows)?)?;
    }
    let policy = FrozenPolicy {
        schema: FREEZE.into(),
        plan_sha256,
        comparison: plan.comparison.clone(),
        label_role: label_role.into(),
        panel_sha256: plan.panel_sha256.clone(),
        dataset,
        original_label_role: plan.original_label_role.clone(),
        label_burn_sha256,
        guard: GuardPolicy::current(),
        evaluation_seed: plan.evaluation_seed,
        targets_sha256: state.put_blob(&canonical_json(&rows)?)?,
        folds,
        created_at_unix: now_unix(),
        inner_partition,
        context: plan.context.clone(),
    };
    let bytes = canonical_json(&policy)?;
    immutable_write(&path, &bytes)?;
    state.put_blob(&bytes)?;
    Ledger::open(state.root(), "eligibility").append(
        &[serde_json::json!({
            "outcome":"frozen", "comparison_sha256":identity, "freeze_sha256":sha256_hex(&bytes),
            "original_label_role":policy.original_label_role, "label_role":policy.label_role,
            "label_burn_sha256":policy.label_burn_sha256, "guard":policy.guard
        })],
        serde_json::json!({"source":"controller-owned-preoptimization-freeze"}),
    )?;
    Ok(FreezeReceipt {
        comparison_sha256: identity,
        freeze_sha256: sha256_hex(&bytes),
    })
}

fn load_freeze(state: &State, receipt: &FreezeReceipt) -> Result<FrozenPolicy> {
    ensure!(is_hex64(&receipt.freeze_sha256), "invalid freeze receipt");
    let bytes = fs::read(freeze_path(state, &receipt.comparison_sha256)?)?;
    ensure!(
        sha256_hex(&bytes) == receipt.freeze_sha256,
        "independent optimization freeze changed"
    );
    let policy: FrozenPolicy = serde_json::from_slice(&bytes)?;
    let ledger = Ledger::open(state.root(), "eligibility");
    ledger.require_intact()?;
    ensure!(
        ledger.events()?.iter().any(
            |event| event.get("outcome").and_then(|value| value.as_str()) == Some("frozen")
                && event.get("freeze_sha256").and_then(|value| value.as_str())
                    == Some(&receipt.freeze_sha256)
        ),
        "freeze has no durable preoptimization ledger receipt"
    );
    ensure!(
        policy.schema == FREEZE
            && policy.comparison.sha256()? == receipt.comparison_sha256
            && matches!(policy.label_role.as_str(), "train" | "development"),
        "invalid frozen comparison policy"
    );
    Ok(policy)
}

#[cfg(test)]
fn combine(parts: &[Predictions]) -> Result<Predictions> {
    match parts.first().context("missing OOF predictions")? {
        Predictions::Values(_) => {
            let mut out = Vec::new();
            for part in parts {
                let Predictions::Values(values) = part else {
                    bail!("mixed OOF prediction kinds")
                };
                out.extend(values.iter().copied());
            }
            Ok(Predictions::Values(out))
        }
        Predictions::Probabilities { classes, .. } => {
            let mut out = Vec::new();
            for part in parts {
                let Predictions::Probabilities {
                    classes: order,
                    values,
                } = part
                else {
                    bail!("mixed OOF prediction kinds")
                };
                ensure!(order == classes, "OOF class order changed");
                out.extend(values.iter().cloned());
            }
            Ok(Predictions::Probabilities {
                classes: classes.clone(),
                values: out,
            })
        }
    }
}

/// Bind scored bytes to retained worker outputs, including native JSON artifact envelopes.
fn verify_measured_outputs(
    state: &State,
    worker: &WorkerOutcome,
    candidate: &FoldPredictions,
    predictions: &Predictions,
    inner_fit: bool,
) -> Result<()> {
    let mut found_actual_fit = !inner_fit;
    let mut found_predictions = worker
        .outputs
        .iter()
        .any(|output| output.sha256 == candidate.predictions_sha256);
    let mut found_model = worker
        .outputs
        .iter()
        .any(|output| output.sha256 == candidate.fitted_artifact_sha256);
    for output in &worker.outputs {
        if found_predictions && found_model && found_actual_fit {
            break;
        }
        // Binary model files may be large; only bounded JSON envelopes can embed predictions/state.
        if output.path.extension().and_then(|value| value.to_str()) != Some("json")
            || state
                .read_blob(&output.sha256)?
                .context("retained output missing")?
                .len() as u64
                > 128 * 1024 * 1024
        {
            continue;
        }
        let Ok(envelope) = serde_json::from_slice::<serde_json::Value>(
            &state
                .read_blob(&output.sha256)?
                .context("retained output missing")?,
        ) else {
            continue;
        };
        if let Some(values) = envelope.get("predictions") {
            if serde_json::from_value::<Predictions>(values.clone())
                .ok()
                .as_ref()
                == Some(predictions)
            {
                found_predictions = true;
            }
        }
        if envelope
            .get("schema")
            .and_then(|value| value.as_str())
            .is_some_and(|schema| {
                matches!(
                    schema,
                    "jope.panel-rust-worker-output/v1" | "jope.panel-synthesized-worker-output/v1"
                )
            })
            && envelope
                .get("artifact_sha256")
                .and_then(|value| value.as_str())
                == Some(&candidate.fitted_artifact_sha256)
        {
            let stored: serde_json::Value = blob(state, &candidate.fitted_artifact_sha256)?;
            if envelope.get("artifact") == Some(&stored) {
                found_model = true;
                if inner_fit {
                    ensure!(
                        envelope["schema"] == "jope.panel-synthesized-worker-output/v1"
                            && envelope["physical_fit_starts"].as_u64() == Some(1)
                            && envelope["artifact"]["recipe"]["executable_sha256"]
                                == candidate.program_sha256,
                        "inner fit lacks actual one-fit synthesized output"
                    );
                    found_actual_fit = true;
                }
            }
        }
    }
    ensure!(
        found_predictions && found_model && found_actual_fit,
        "scored predictions or fitted state are absent from the bound worker outputs"
    );
    Ok(())
}

fn measured(
    state: &State,
    record: &HwmRecord,
    evidence: &OofEvidence,
    live: bool,
) -> Result<Qualification> {
    record.validate()?;
    ensure!(
        !record.sealed && !record.benchmark_valid && record.replayable,
        "sealed or unbound record cannot teach"
    );
    let selected_task = task(&record.identity)?;
    let expected_metric = match selected_task {
        Task::Regression => Metric::Rmse,
        Task::Binary => Metric::Auc,
        Task::Multiclass => Metric::Logloss,
        _ => unreachable!(),
    };
    ensure!(
        record.score.metric == expected_metric,
        "metric differs from measured task protocol"
    );
    let policy = load_freeze(state, &evidence.freeze)?;
    ensure!(
        policy.guard == GuardPolicy::current(),
        "unknown teaching guard"
    );
    let context_receipt_sha256 = context::verify(state, &policy, record, evidence)?;
    let experiment: Experiment = blob(state, &evidence.experiment_blob_sha256)?;
    let window: ExecutionWindow = blob(state, &evidence.window_sha256)?;
    experiment.validate()?;
    window.validate(&experiment)?;
    let inner_binding = inner::bind(state, &policy, evidence, &experiment, &window, live)?;
    ensure!(
        (inner_binding.is_some()
            || (experiment.checkpoints.get("teaching_freeze")
                == Some(&evidence.freeze.freeze_sha256)
                && experiment.checkpoints.get("teaching_vocabulary")
                    == Some(&record.identity.vocabulary_sha256)))
            && experiment.panel_sha256 == policy.panel_sha256
            && experiment.protocol == record.identity.protocol
            && experiment.seed == policy.evaluation_seed
            && experiment.numerical_policy == NUMERICAL_POLICY
            && window.started_unix_ms >= policy.created_at_unix * 1000,
        "experiment does not commit to the preoptimization freeze"
    );
    ensure!(
        record.identity == policy.comparison
            && record.identity_sha256 == evidence.freeze.comparison_sha256,
        "stale comparison identity"
    );
    ensure!(
        evidence.fold_predictions_sha256.len() == 5 && policy.folds.len() == 5,
        "complete five-fold OOF required"
    );
    let classes = &record.identity.comparison.as_ref().unwrap().class_order;
    let rows: Vec<TargetRow> = blob(state, &policy.targets_sha256)?;
    let mut fold_losses = Vec::new();
    let mut improved_folds = 0;
    for (fold, frozen) in policy.folds.iter().enumerate() {
        ensure!(frozen.fold as usize == fold, "frozen fold order changed");
        let (train, hold) = fold_rows(&rows, fold as u8);
        let candidate: FoldPredictions = blob(state, &evidence.fold_predictions_sha256[fold])?;
        ensure!(
            candidate.schema == PREDICTIONS
                && candidate.fold as usize == fold
                && candidate.comparison_sha256 == record.identity_sha256
                && candidate.program_sha256 == record.genome.program_sha256
                && candidate.fit_started_unix >= policy.created_at_unix,
            "prediction provenance differs from pre-optimization freeze"
        );
        ensure!(
            candidate.training_rows_sha256 == frozen.training_rows_sha256
                && candidate.training_rows_sha256 == rows_sha(&train)?
                && candidate.scoring_rows_sha256 == frozen.scoring_rows_sha256
                && candidate.scoring_rows_sha256 == rows_sha(&hold)?
                && candidate.row_ids == hold.iter().map(|row| row.id.clone()).collect::<Vec<_>>(),
            "OOF row identity or order changed"
        );
        require_blob(state, &candidate.fitted_artifact_sha256)?;
        let fitted: serde_json::Value = blob(state, &candidate.fitted_artifact_sha256)?;
        ensure!(
            ((fitted.get("schema").and_then(|v| v.as_str()) == Some("jope.panel-fitted-rust/v1")
                && fitted.pointer("/method/dsl").and_then(|v| v.as_str())
                    == Some(&record.genome.dsl))
                || (inner_binding.is_some()
                    && fitted.get("schema").and_then(|v| v.as_str())
                        == Some("jope.panel-fitted-synthesized/v1")
                    && fitted
                        .pointer("/recipe/executable_canonical")
                        .and_then(|v| v.as_str())
                        == Some(&record.genome.dsl)))
                && fitted
                    .pointer("/model/genome_sha256")
                    .and_then(|v| v.as_str())
                    == Some(&record.genome.program_sha256),
            "native fitted recipe differs from teaching candidate"
        );
        let request: EvaluationRequest = blob(state, &candidate.evaluation_request_sha256)?;
        let result: EvaluationResult = blob(state, &candidate.execution_receipt_sha256)?;
        request.validate()?;
        result.validate(&request)?;
        ensure!(
            result.reused_from_request_sha256.is_none(),
            "reused predictions are not a new OOF teaching observation"
        );
        let worker: WorkerOutcome = blob(state, &result.worker_receipt_sha256)?;
        let job: WorkerJob = blob(state, &candidate.worker_job_sha256)?;
        if live {
            worker.verify_outputs()?;
        }
        ensure!(
            worker.schema == EXECUTION_SCHEMA && !worker.outputs.is_empty(),
            "invalid worker receipt"
        );
        for artifact in worker
            .outputs
            .iter()
            .chain(worker.stdout.iter())
            .chain(worker.stderr.iter())
        {
            retain_artifact(state, artifact, live)?;
        }
        let started: serde_json::Value = blob(state, &candidate.started_receipt_sha256)?;
        let started_job: WorkerJob =
            serde_json::from_value(started.get("job").context("started job missing")?.clone())?;
        let started_ms = started
            .get("started_unix_ms")
            .and_then(|v| v.as_u64())
            .context("start time missing")?;
        let deadline_ms = started
            .get("fit_deadline_unix_ms")
            .and_then(|v| v.as_u64())
            .context("fit deadline missing")?;
        ensure!(
            started_job.identity()? == job.identity()?
                && started.get("job_sha256").and_then(|v| v.as_str()) == Some(&worker.job_sha256)
                && started_ms >= window.started_unix_ms
                && candidate.fit_started_unix == started_ms / 1000
                && deadline_ms > started_ms
                && deadline_ms <= window.deadline_unix_ms
                && deadline_ms - started_ms <= experiment.resources.per_fit_seconds * 1000,
            "worker started receipt differs from original admission"
        );
        ensure!(
            worker.request_sha256 == request.identity()?
                && worker.job_sha256 == job.identity()?
                && job.request.identity()? == request.identity()?
                && worker.command_status == EvaluationStatus::Completed
                && worker.exit_code == Some(0)
                && worker.signal.is_none()
                && worker.reason.is_none(),
            "OOF lacks a clean bound worker execution"
        );
        retain_artifact(state, &job.command.program, live)?;
        ensure!(
            !job.command.inputs.is_empty(),
            "worker input closure is missing"
        );
        for input in &job.command.inputs {
            retain_artifact(state, input, live)?;
        }
        ensure!(
            worker
                .outputs
                .iter()
                .map(|artifact| artifact.path.clone())
                .collect::<BTreeSet<_>>()
                == job
                    .outputs
                    .iter()
                    .map(|path| job.directory.join(path))
                    .collect::<BTreeSet<_>>()
                && worker.outputs.len() == job.outputs.len(),
            "worker output inventory changed"
        );
        ensure!(
            result.consumption.wall_seconds == worker.consumption.wall_seconds
                && result.consumption.cpu_seconds == worker.consumption.cpu_seconds
                && result.consumption.peak_rss_bytes == worker.consumption.peak_rss_bytes
                && result.consumption.cgroup_peak_memory_bytes
                    == worker.consumption.cgroup_peak_memory_bytes
                && result.consumption.sampled_peak_rss_bytes
                    == worker.consumption.sampled_peak_rss_bytes
                && result.consumption.accounting_reason == worker.consumption.accounting_reason,
            "resource receipt changed or unknown consumption was replaced"
        );
        let context = record.identity.comparison.as_ref().unwrap();
        if let Some(binding) = &inner_binding {
            inner::validate_execution(
                binding,
                &request,
                &job,
                &worker,
                started_ms,
                deadline_ms,
                &record.genome.program_sha256,
            )?;
        }
        ensure!(
            request.experiment_sha256 == experiment.identity()?
                && request.dataset_sha256 == record.identity.dataset_hash
                && (inner_binding.is_some() || request.split_sha256 == record.identity.cut_id)
                && request.fitting_rows_sha256 == frozen.training_rows_sha256
                && request.scoring_rows_sha256 == frozen.scoring_rows_sha256
                && request.fitting_rows == train.len()
                && request.scoring_rows == hold.len()
                && request.recipe_sha256 == record.genome.program_sha256
                && request.evaluator_sha256 == record.identity.evaluator_sha256
                && request.checkpoint_sha256 == context.checkpoint_sha256
                && request.task == selected_task
                && request.classes == *classes
                && request.fold as usize == fold
                && request.stage == EvaluationStage::TrainOof
                && (inner_binding.is_some() || request.seed == policy.evaluation_seed)
                && request.numerical_policy == context.numerical_policy,
            "execution request differs from frozen train-side OOF comparison"
        );
        ensure!(
            result.status == EvaluationStatus::Completed
                && result.predictions_sha256.as_ref() == Some(&candidate.predictions_sha256)
                && result.fitted_state_sha256.as_ref() == Some(&candidate.fitted_artifact_sha256),
            "OOF execution is incomplete or names different artifacts"
        );
        let predictions: Predictions = blob(state, &candidate.predictions_sha256)?;
        verify_measured_outputs(
            state,
            &worker,
            &candidate,
            &predictions,
            inner_binding.is_some(),
        )?;
        if inner_binding.is_some() {
            ensure!(
                result.secondary.get("physical_fit_starts") == Some(&1.),
                "inner physical fit count changed"
            );
        }
        let baseline_predictions: Predictions = blob(state, &frozen.baseline_predictions_sha256)?;
        ensure!(
            baseline_predictions == baseline(selected_task, classes, &train, hold.len())?,
            "baseline differs from train-only trivial predictor"
        );
        let targets = hold
            .iter()
            .map(|row| row.target.clone())
            .collect::<Vec<_>>();
        let c = score_predictions(selected_task, classes, &targets, &predictions)?;
        let b = score_predictions(selected_task, classes, &targets, &baseline_predictions)?;
        ensure!(
            result.primary == Some(c.value),
            "execution score differs from stored predictions"
        );
        improved_folds += usize::from(policy.guard.improves(c.loss, b.loss));
        fold_losses.push((c.loss, b.loss));
    }
    let candidate_loss = fold_losses
        .iter()
        .map(|(candidate, _)| candidate)
        .sum::<f64>()
        / 5.0;
    let baseline_loss = fold_losses
        .iter()
        .map(|(_, baseline)| baseline)
        .sum::<f64>()
        / 5.0;
    let candidate_value = if selected_task == Task::Binary {
        1.0 - candidate_loss
    } else {
        candidate_loss
    };
    ensure!(
        candidate_value == record.score.value,
        "claimed HWM score differs from measured mean-fold predictions"
    );
    Ok(Qualification {
        schema: QUALIFICATION.into(),
        comparison: record.identity.clone(),
        freeze: evidence.freeze.clone(),
        record_sha256: record.provenance.record_sha256.clone(),
        program_sha256: record.genome.program_sha256.clone(),
        targets_sha256: policy.targets_sha256,
        baseline_predictions_sha256: policy
            .folds
            .iter()
            .map(|fold| fold.baseline_predictions_sha256.clone())
            .collect(),
        fold_predictions_sha256: evidence.fold_predictions_sha256.clone(),
        guard: policy.guard,
        label_role: policy.label_role,
        original_label_role: policy.original_label_role,
        label_burn_sha256: policy.label_burn_sha256,
        candidate_loss,
        baseline_loss,
        improved_folds,
        fold_losses,
        context_fitting_rows_sha256:policy.inner_partition.as_ref().map(|p|p.fitting_rows_sha256()).transpose()?,
        context_outer_fold:policy.inner_partition.as_ref().map(|p|p.outer_fold),
        context_receipt_sha256: Some(context_receipt_sha256),
        measurement:
            "recomputed-from-pinned-targets-and-predictions; analytic-baseline-worker-fits=0; execution receipts retained separately"
                .into(),
    })
}

/// Recompute a retained admission without opening mutable worker paths or changing ledgers.
pub fn revalidate_qualification(state: &State, record: &HwmRecord) -> Result<Qualification> {
    let stored: StoredQualification =
        serde_json::from_slice(&fs::read(qualification_path(state, record))?)?;
    let current = measured(state, record, &stored.evidence, false)?;
    ensure!(
        current == stored.qualification && current.guard.admits(&current),
        "retained teaching qualification changed"
    );
    Ok(current)
}
fn qualification_path(state: &State, record: &HwmRecord) -> std::path::PathBuf {
    state
        .root()
        .join("eligibility/qualified")
        .join(&record.identity_sha256)
        .join(format!("{}.json", record.provenance.record_sha256))
}

/// Recompute every fold and mean loss; no structural-only witness can call itself qualified.
pub fn qualify(state: &State, record: &HwmRecord, evidence: &OofEvidence) -> Result<Qualification> {
    let _lock = state.lock("eligibility")?;
    let ledger = Ledger::open(state.root(), "eligibility");
    ledger.require_intact()?;
    let measurement = measured(state, record, evidence, true);
    let outcome = (|| -> Result<Qualification> {
        let qualification = measurement
            .as_ref()
            .map_err(|error| anyhow::anyhow!("{error:#}"))?
            .clone();
        ensure!(qualification.guard.admits(&qualification), "no eligible signal: mean-fold gain and four of five gains above the frozen numerical margin are required");
        require_blob(state, &record.provenance.record_sha256)?;
        let stored = StoredQualification {
            evidence: evidence.clone(),
            qualification: qualification.clone(),
        };
        immutable_write(
            &qualification_path(state, record),
            &canonical_json(&stored)?,
        )?;
        Ok(qualification)
    })();
    ledger.append(&[serde_json::json!({"schema":"jope.teaching-decision/v1",
        "comparison_sha256":record.identity_sha256, "record_sha256":record.provenance.record_sha256,
        "evidence_sha256":canonical_sha256(evidence)?, "guard":GuardPolicy::current(),
        "outcome":if outcome.is_ok(){"qualified"}else{"rejected"},
        "reason":outcome.as_ref().err().map(|error| format!("{error:#}")),
        "measurement":measurement.as_ref().ok()
    })], serde_json::json!({"source":"measured-teaching-scorer"}))?;
    outcome
}

pub(crate) fn require_qualified(state: &State, record: &HwmRecord) -> Result<()> {
    record.validate()?;
    let stored: StoredQualification =
        serde_json::from_slice(&fs::read(qualification_path(state, record))?)?;
    let measured = measured(state, record, &stored.evidence, false)?;
    ensure!(
        measured == stored.qualification && measured.guard.admits(&measured),
        "unqualified OOF candidate"
    );
    Ok(())
}

fn retain_artifact(state: &State, artifact: &Artifact, live: bool) -> Result<()> {
    if live {
        artifact.verify()?;
        let bytes = fs::read(&artifact.path)?;
        ensure!(
            state.put_blob(&bytes)? == artifact.sha256,
            "artifact changed during retention"
        );
    } else {
        require_blob(state, &artifact.sha256)?;
    }
    Ok(())
}

/// Explicit historical replay: every entry must have an intact accepted ledger
/// event and recomputable measured evidence. Current-pointer admission is unchanged.
pub fn revalidate_replay(state: &State, entries: &[Qualification]) -> Result<TeachingManifest> {
    Ledger::open(state.root(), "hwm").require_intact()?;
    Ledger::open(state.root(), "eligibility").require_intact()?;
    ensure!(!entries.is_empty(), "empty replay admission");
    let events = crate::hwm::events(state)?;
    for entry in entries {
        let mut found = false;
        for event in &events {
            if event.outcome != "accepted"
                || event.pointer != PointerKey::of(&event.record)
                || event.record.sealed
                || event.record.identity != entry.comparison
            {
                continue;
            }
            event.record.validate()?;
            if revalidate_qualification(state, &event.record)? == *entry {
                found = true;
                break;
            }
        }
        ensure!(
            found,
            "replay lacks exact accepted historical qualification"
        );
    }
    Ok(TeachingManifest {
        schema: "jope.qualified-teaching-freeze/v1".into(),
        entries: entries.to_vec(),
        excluded_diagnostic: 0,
        exclusions: vec![],
    })
}

/// Only exact current pointers with recomputable measured admissions can enter a teaching manifest.
pub fn freeze_teaching(state: &State) -> Result<TeachingManifest> {
    Ledger::open(state.root(), "hwm").require_intact()?;
    Ledger::open(state.root(), "eligibility").require_intact()?;
    let accepted: BTreeSet<_> = crate::hwm::events(state)?
        .into_iter()
        .filter(|event| {
            event.outcome == "accepted" && event.pointer == PointerKey::of(&event.record)
        })
        .map(|event| {
            (
                event.record.identity_sha256,
                event.record.provenance.record_sha256,
            )
        })
        .collect();
    let mut manifest = TeachingManifest {
        schema: "jope.qualified-teaching-freeze/v1",
        excluded_diagnostic: 0,
        exclusions: Vec::new(),
        entries: Vec::new(),
    };
    for (key, record) in state.pointers()? {
        let candidate = (|| -> Result<Qualification> {
            record.validate()?;
            ensure!(
                key == PointerKey::of(&record)
                    && record.identity.comparison.is_some()
                    && !record.sealed,
                "legacy, mismatched or sealed diagnostic record"
            );
            ensure!(
                accepted.contains(&(
                    record.identity_sha256.clone(),
                    record.provenance.record_sha256.clone()
                )),
                "teaching pointer lacks an accepted exact-identity ledger event"
            );
            require_blob(state, &record.provenance.record_sha256)?;
            let stored: StoredQualification =
                serde_json::from_slice(&fs::read(qualification_path(state, &record))?)?;
            let qualified = measured(state, &record, &stored.evidence, false)?;
            ensure!(
                qualified == stored.qualification && qualified.guard.admits(&qualified),
                "stored teaching qualification changed"
            );
            Ok(qualified)
        })();
        match candidate {
            Ok(qualified) => manifest.entries.push(qualified),
            Err(error) => {
                manifest.excluded_diagnostic += 1;
                manifest.exclusions.push((key.rel(), format!("{error:#}")));
            }
        }
    }
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        encoder_receipt::{Encoder, EncoderCapability, Inputs, Member, Product, Scope},
        hwm::{self, Candidate},
        record::{ComparisonContext, Cost, Evaluator, Genome, Provenance, Score, HWM_SCHEMA},
    };
    use jope_pipeline::execution::{
        Artifact, ObservedConsumption, WorkerCommand, EXECUTION_SCHEMA,
    };
    use jope_pipeline::experiment::REQUEST_SCHEMA;
    use jope_pipeline::experiment_v2::{Consumption, RESULT_SCHEMA};
    use serde_json::json;

    struct Fixture {
        _directory: tempfile::TempDir,
        state: State,
        plan: FreezePlan,
        record: HwmRecord,
        evidence: OofEvidence,
        rows: Vec<TargetRow>,
    }

    fn context_plan() -> ContextPlan {
        ContextPlan {
            source_family: "synthetic-fixture-family".into(),
            encoder: Encoder {
                artifact_sha256: sha256_hex(b"fixture context encoder"),
                configuration_sha256: sha256_hex(b"fixture context configuration"),
                seed: 7,
                product: Product::TeachingContext2048,
            },
        }
    }

    /// Producer-side scope, built from the fixture rows rather than the receiver's derivation.
    fn context_scope(comparison: &HwmIdentity, rows: &[TargetRow]) -> Scope {
        Scope {
            comparison: comparison.clone(),
            source_family: context_plan().source_family,
            outer_fold: None,
            inner_fold: None,
            fitting: rows
                .iter()
                .map(|row| Member {
                    row: row.id.clone(),
                    group: row.group.clone(),
                })
                .collect(),
            forbidden: vec![],
        }
    }

    fn context_output(width: usize, offset: f32) -> Vec<u8> {
        (0..width)
            .flat_map(|index| (offset + index as f32 / 64.).to_le_bytes())
            .collect()
    }

    /// Retain one synthetic producer's receipt, fitted state and output; nothing is encoded.
    fn produce_context(
        state: &State,
        scope: Scope,
        encoder: Encoder,
        parent_receipts: Vec<String>,
        output: &[u8],
    ) -> ContextEvidence {
        let capability = EncoderCapability::bind(
            scope,
            encoder,
            Inputs {
                features_sha256: sha256_hex(b"fixture projected features"),
                labels_sha256: Some(sha256_hex(b"fixture projected labels")),
                parent_receipts,
            },
        )
        .unwrap();
        let fitted = format!("synthetic fitted context state/{}", output.len()).into_bytes();
        let receipt = capability.receipt(&fitted, output).unwrap();
        ContextEvidence {
            receipt_sha256: state.put_blob(&canonical_json(&receipt).unwrap()).unwrap(),
            fitted_state_sha256: state.put_blob(&fitted).unwrap(),
            output_sha256: state.put_blob(output).unwrap(),
        }
    }

    fn valid_context(
        state: &State,
        comparison: &HwmIdentity,
        rows: &[TargetRow],
    ) -> ContextEvidence {
        let parent = produce_context(
            state,
            context_scope(comparison, rows),
            Encoder {
                product: Product::DatasetSignature512,
                ..context_plan().encoder
            },
            vec![],
            &context_output(512, 0.25),
        );
        produce_context(
            state,
            context_scope(comparison, rows),
            context_plan().encoder,
            vec![parent.receipt_sha256],
            &context_output(2048, 0.5),
        )
    }

    fn fixture(selected_task: Task, improved: usize) -> Fixture {
        fixture_with(selected_task, improved, Some(context_plan()))
    }

    // All inputs/predictions are synthetic; these tests establish admission mechanics,
    // not corpus admission, real fitted-model accuracy, or a completed live search.
    fn fixture_with(selected_task: Task, improved: usize, context: Option<ContextPlan>) -> Fixture {
        let fixture_root =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp/teaching-fixtures");
        fs::create_dir_all(&fixture_root).unwrap();
        let directory = tempfile::tempdir_in(fs::canonicalize(&fixture_root).unwrap()).unwrap();
        let state = State::at(directory.path());
        let classes: Vec<String> = match selected_task {
            Task::Regression => vec![],
            Task::Binary => vec!["negative".into(), "positive".into()],
            Task::Multiclass => vec!["α".into(), "001".into(), "class-C".into()],
            _ => unreachable!(),
        };
        let comparison = HwmIdentity {
            task: selected_task.as_str().into(),
            dataset_hash: "d".repeat(64),
            cut_id: "c".repeat(64),
            evaluator_sha256: "e".repeat(64),
            vocabulary_sha256: "a".repeat(64),
            protocol: "synthetic-train-oof-v1".into(),
            comparison: Some(ComparisonContext {
                numerical_policy: NUMERICAL_POLICY.into(),
                stage: "train_oof".into(),
                aggregation: "mean_fold_loss".into(),
                label_role: "development".into(),
                sealed: false,
                benchmark_valid: false,
                recipe_encoding: "native-canonical-dsl-sha256/v1".into(),
                class_order: classes.clone(),
                checkpoint_sha256: None,
            }),
        };
        let targets = if classes.is_empty() {
            vec!["-1".to_string(), "1".to_string()]
        } else {
            classes.clone()
        };
        let rows = (0..5)
            .flat_map(|fold| {
                targets
                    .iter()
                    .enumerate()
                    .map(move |(index, target)| TargetRow {
                        id: format!("{:064x}", fold as usize * 100 + index),
                        group: format!("{:064x}", 1000 + fold as usize * 100 + index),
                        fold,
                        target: target.clone(),
                    })
            })
            .collect::<Vec<_>>();
        let prepared_rows = rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                json!({
                    "id":row.id, "source_index":index, "group":row.group, "fold":row.fold,
                    "target":row.target, "values":[index.to_string()]
                })
            })
            .collect::<Vec<_>>();
        let table = json!({"schema":"jope.prepared-table/v1", "dataset":"fixture", "task":selected_task,
            "source_sha256":comparison.dataset_hash, "partition":"development",
            "features":[{"name":"x", "kind":"numeric"}], "classes":classes, "rows":prepared_rows});
        let development_sha = state.put_blob(&canonical_json(&table).unwrap()).unwrap();
        let dataset = json!({"id":"fixture", "task":selected_task,
            "metric":jope_pipeline::experiment::Metric::for_task(selected_task).unwrap(),
            "source":{"url":"synthetic", "sha256":comparison.dataset_hash, "bytes":1,
                "metadata_url":"synthetic", "metadata_sha256":"9".repeat(64)},
            "rows":rows.len()+1, "features":[{"name":"x", "kind":"numeric"}], "classes":classes,
            "group_columns":[], "split":{"protocol":"synthetic", "sha256":comparison.cut_id,
                "development_sha256":development_sha, "sealed_sha256":"8".repeat(64),
                "development_rows":rows.len(), "sealed_rows":1, "fold_rows":vec![targets.len();5]},
            "license":{"original":"synthetic", "evidence_url":"synthetic", "repository":"synthetic", "note":"synthetic"},
            "checkpoint_exposure":"synthetic", "description":"synthetic", "group_limitations":"synthetic"});
        let mut datasets = Vec::new();
        for (kind_index, kind) in [Task::Regression, Task::Binary, Task::Multiclass]
            .into_iter()
            .enumerate()
        {
            for index in 0..10 {
                let mut value = dataset.clone();
                if !(kind == selected_task && index == 0) {
                    value["id"] = json!(format!("synthetic-{kind_index}-{index}"));
                    value["task"] = json!(kind);
                    value["metric"] =
                        json!(jope_pipeline::experiment::Metric::for_task(kind).unwrap());
                    value["source"]["sha256"] =
                        json!(format!("{:064x}", kind_index * 10 + index + 100));
                    value["classes"] = match kind {
                        Task::Regression => json!([]),
                        Task::Binary => json!(["0", "1"]),
                        _ => json!(["0", "1", "2"]),
                    };
                }
                datasets.push(value);
            }
        }
        let panel = json!({"schema":"jope.benchmark-panel/v1", "panel_id":"synthetic",
            "repository":"synthetic", "revision":"a".repeat(40), "seed":7,
            "selection":"explicit synthetic qualification fixture", "folds":5, "datasets":datasets});
        let plan = FreezePlan {
            comparison: comparison.clone(),
            original_label_role: "test".into(),
            panel_sha256: state.put_blob(&canonical_json(&panel).unwrap()).unwrap(),
            dataset_id: "fixture".into(),
            evaluation_seed: 7,
            context,
        };
        let freeze = freeze_optimization(&state, &plan).unwrap();
        let frozen = load_freeze(&state, &freeze).unwrap();
        assert_eq!(frozen.label_role, "development");
        let dsl = "(pipe X (synthetic-fixture))".to_string();
        let program = sha256_hex(dsl.as_bytes());
        let fitted_bytes = canonical_json(&json!({"schema":"jope.panel-fitted-rust/v1",
            "method":{"dsl":dsl}, "model":{"genome_sha256":program},
            "fixture_note":"synthetic artifact consistency test, not a live fit"}))
        .unwrap();
        let fitted = state.put_blob(&fitted_bytes).unwrap();
        let experiment: Experiment = serde_json::from_value(json!({
            "schema":"jope.experiment/v1", "panel_sha256":plan.panel_sha256,
            "methods_sha256":"7".repeat(64), "source_sha":"a".repeat(40),
            "checkpoints":{"teaching_freeze":freeze.freeze_sha256,"teaching_vocabulary":comparison.vocabulary_sha256}, "seed":7,
            "resources":{"workers":1,"cpus":1,"memory_bytes":1048576,"swap_bytes":0,
                "gpu_tasks":0,"threads_per_worker":1,"total_seconds":60,"reporting_seconds":5,"per_fit_seconds":30},
            "expected_artifacts":["synthetic"],"protocol":"synthetic-train-oof-v1", "numerical_policy":NUMERICAL_POLICY
        })).unwrap();
        let window = ExecutionWindow {
            experiment_sha256: experiment.identity().unwrap(),
            started_unix_ms: frozen.created_at_unix * 1000,
            deadline_unix_ms: frozen.created_at_unix * 1000 + 60000,
        };
        let mut evidence = OofEvidence {
            freeze,
            experiment_blob_sha256: state
                .put_blob(&canonical_json(&experiment).unwrap())
                .unwrap(),
            window_sha256: state.put_blob(&canonical_json(&window).unwrap()).unwrap(),
            fold_predictions_sha256: Vec::new(),
            inner: None,
            context: Some(valid_context(&state, &comparison, &rows)),
        };
        let mut all_targets = Vec::new();
        let mut all_predictions = Vec::new();
        let mut scores = Vec::new();
        for fold in 0..5 {
            let (train, hold) = fold_rows(&rows, fold);
            let predictions = if (fold as usize) < improved {
                match selected_task {
                    Task::Regression => Predictions::Values(
                        hold.iter().map(|row| row.target.parse().unwrap()).collect(),
                    ),
                    _ => Predictions::Probabilities {
                        classes: classes.clone(),
                        values: hold
                            .iter()
                            .map(|row| {
                                classes
                                    .iter()
                                    .map(|class| f64::from(class == &row.target))
                                    .collect()
                            })
                            .collect(),
                    },
                }
            } else {
                baseline(selected_task, &classes, &train, hold.len()).unwrap()
            };
            let labels: Vec<_> = hold.iter().map(|row| row.target.clone()).collect();
            let score = score_predictions(selected_task, &classes, &labels, &predictions).unwrap();
            let predictions_sha256 = state
                .put_blob(&serde_json::to_vec(&predictions).unwrap())
                .unwrap();
            let request = EvaluationRequest {
                schema: REQUEST_SCHEMA.into(),
                experiment_sha256: experiment.identity().unwrap(),
                dataset_sha256: comparison.dataset_hash.clone(),
                split_sha256: comparison.cut_id.clone(),
                fitting_rows_sha256: rows_sha(&train).unwrap(),
                scoring_rows_sha256: rows_sha(&hold).unwrap(),
                fitting_rows: train.len(),
                scoring_rows: hold.len(),
                recipe_sha256: program.clone(),
                evaluator_sha256: comparison.evaluator_sha256.clone(),
                checkpoint_sha256: None,
                task: selected_task,
                metric: score.metric,
                classes: classes.clone(),
                fold,
                stage: EvaluationStage::TrainOof,
                seed: 7,
                numerical_policy: NUMERICAL_POLICY.into(),
            };
            let job_directory = state.root().join(format!("fixture-worker-{fold}"));
            fs::create_dir(&job_directory).unwrap();
            let input_path = job_directory.join("input.json");
            fs::write(&input_path, canonical_json(&request).unwrap()).unwrap();
            let output_path = job_directory.join("predictions.json");
            fs::write(&output_path, serde_json::to_vec(&predictions).unwrap()).unwrap();
            let fitted_path = job_directory.join("fitted.bin");
            fs::write(&fitted_path, &fitted_bytes).unwrap();
            let fixture_program = Artifact {
                path: input_path.clone(),
                sha256: sha256_hex(&fs::read(&input_path).unwrap()),
            };
            let job = WorkerJob {
                request: request.clone(),
                directory: job_directory,
                outputs: vec!["predictions.json".into(), "fitted.bin".into()],
                arm: None,
                command: WorkerCommand {
                    program: fixture_program.clone(),
                    args: vec![],
                    inputs: vec![fixture_program],
                    environment: BTreeMap::new(),
                },
            };
            let outcome = WorkerOutcome {
                schema: EXECUTION_SCHEMA.into(),
                job_sha256: job.identity().unwrap(),
                request_sha256: request.identity().unwrap(),
                command_status: EvaluationStatus::Completed,
                reason: None,
                exit_code: Some(0),
                signal: None,
                outputs: vec![
                    Artifact {
                        path: output_path,
                        sha256: predictions_sha256.clone(),
                    },
                    Artifact {
                        path: fitted_path,
                        sha256: fitted.clone(),
                    },
                ],
                stdout: None,
                stderr: None,
                consumption: ObservedConsumption {
                    wall_seconds: Some(0.01),
                    cpu_seconds: None,
                    accounting_reason: "synthetic fixture, CPU unmeasured".into(),
                    ..Default::default()
                },
            };
            let result = EvaluationResult {
                schema: RESULT_SCHEMA.into(),
                worker_receipt_sha256: state.put_blob(&canonical_json(&outcome).unwrap()).unwrap(),
                reused_from_request_sha256: None,
                request_sha256: request.identity().unwrap(),
                status: EvaluationStatus::Completed,
                reason: None,
                primary: Some(score.value),
                secondary: BTreeMap::new(),
                scored_rows: hold.len(),
                predictions_sha256: Some(predictions_sha256.clone()),
                fitted_state_sha256: Some(fitted.clone()),
                consumption: Consumption {
                    wall_seconds: Some(0.01),
                    cpu_seconds: None,
                    accounting_reason: "synthetic fixture, CPU unmeasured".into(),
                    ..Default::default()
                },
            };
            let prediction = FoldPredictions {
                schema: PREDICTIONS.into(),
                comparison_sha256: comparison.sha256().unwrap(),
                program_sha256: program.clone(),
                fold,
                training_rows_sha256: request.fitting_rows_sha256.clone(),
                scoring_rows_sha256: request.scoring_rows_sha256.clone(),
                row_ids: hold.iter().map(|row| row.id.clone()).collect(),
                predictions_sha256,
                evaluation_request_sha256: state
                    .put_blob(&canonical_json(&request).unwrap())
                    .unwrap(),
                worker_job_sha256: state.put_blob(&canonical_json(&job).unwrap()).unwrap(),
                fitted_artifact_sha256: fitted.clone(),
                execution_receipt_sha256: state
                    .put_blob(&canonical_json(&result).unwrap())
                    .unwrap(),
                started_receipt_sha256: state
                    .put_blob(
                        &canonical_json(&json!({"job":job,
                    "job_sha256":job.identity().unwrap(),"started_unix_ms":window.started_unix_ms,
                    "fit_deadline_unix_ms":window.started_unix_ms+30000,"cgroup":"synthetic"}))
                        .unwrap(),
                    )
                    .unwrap(),
                fit_started_unix: frozen.created_at_unix,
            };
            evidence.fold_predictions_sha256.push(
                state
                    .put_blob(&canonical_json(&prediction).unwrap())
                    .unwrap(),
            );
            all_targets.extend(labels);
            all_predictions.push(predictions);
            scores.push(score.value);
        }
        score_predictions(
            selected_task,
            &classes,
            &all_targets,
            &combine(&all_predictions).unwrap(),
        )
        .unwrap();
        let metric = match selected_task {
            Task::Regression => Metric::Rmse,
            Task::Binary => Metric::Auc,
            _ => Metric::Logloss,
        };
        let record = HwmRecord {
            schema: HWM_SCHEMA.into(),
            identity_sha256: comparison.sha256().unwrap(),
            identity: comparison,
            legacy_identity_sha256: None,
            genome: Genome {
                dsl,
                program_sha256: program,
                semantic_sha256: None,
            },
            score: Score {
                metric,
                value: scores.iter().sum::<f64>() / 5.0,
                direction: metric.direction(),
                units: "synthetic".into(),
            },
            cost: Cost::default(),
            evaluator: Evaluator {
                sha256: "e".repeat(64),
                version: "synthetic".into(),
            },
            benchmark_valid: false,
            sealed: false,
            replayable: true,
            provenance: Provenance {
                source: "synthetic".into(),
                run_id: "fixture".into(),
                git_sha: "unbound".into(),
                bundle_id: "unbound".into(),
                record_sha256: state.put_blob(b"synthetic candidate").unwrap(),
            },
            prev_event_sha256: None,
            improvement_number: 0,
            ts_unix: now_unix(),
        };
        Fixture {
            _directory: directory,
            state,
            plan,
            record,
            evidence,
            rows,
        }
    }

    #[test]
    fn all_tasks_require_measured_aggregate_and_four_of_five_fold_gains() {
        for task in [Task::Regression, Task::Binary, Task::Multiclass] {
            for gains in [0, 3, 4, 5] {
                let f = fixture(task, gains);
                let result = qualify(&f.state, &f.record, &f.evidence);
                if gains < 4 {
                    assert!(result.is_err(), "no-signal/three-fold witness admitted");
                } else {
                    let qualified = result.unwrap();
                    assert_eq!(qualified.improved_folds, gains);
                    assert!(qualified.candidate_loss < qualified.baseline_loss);
                    hwm::ingest(
                        &f.state,
                        None,
                        vec![Candidate {
                            record: f.record.clone(),
                            blob: b"synthetic candidate".to_vec(),
                            chain_from_source: false,
                        }],
                        json!({"kind":"synthetic"}),
                    )
                    .unwrap();
                    let manifest = freeze_teaching(&f.state).unwrap();
                    assert_eq!(manifest.entries.len(), 1);
                    assert_eq!(manifest.entries[0], qualified);
                }
            }
        }
    }

    #[test]
    fn replay_requires_accepted_history_and_recomputes_retained_measurements() {
        let f = fixture(Task::Regression, 5);
        let qualified = qualify(&f.state, &f.record, &f.evidence).unwrap();
        assert!(revalidate_replay(&f.state, std::slice::from_ref(&qualified)).is_err());
        hwm::ingest(
            &f.state,
            None,
            vec![Candidate {
                record: f.record.clone(),
                blob: b"synthetic candidate".to_vec(),
                chain_from_source: false,
            }],
            json!({"kind":"synthetic"}),
        )
        .unwrap();
        assert_eq!(
            revalidate_replay(&f.state, std::slice::from_ref(&qualified))
                .unwrap()
                .entries,
            vec![qualified.clone()]
        );
        let mut changed = qualified;
        changed.candidate_loss += 0.1;
        assert!(revalidate_replay(&f.state, &[changed]).is_err());
    }

    #[test]
    fn freeze_is_independent_and_cannot_be_recreated_from_a_candidate() {
        let f = fixture(Task::Regression, 5);
        assert_eq!(
            freeze_optimization(&f.state, &f.plan).unwrap(),
            f.evidence.freeze
        );
        let mut changed = f.plan.clone();
        changed.evaluation_seed = 100;
        assert!(freeze_optimization(&f.state, &changed).is_err());
        let mut fake = f.evidence.clone();
        fake.freeze.freeze_sha256 = canonical_sha256(&f.record).unwrap();
        assert!(qualify(&f.state, &f.record, &fake).is_err());
        let mut claimed = f.record.clone();
        claimed.score.value = 0.5;
        assert!(qualify(&f.state, &claimed, &f.evidence).is_err());
    }

    #[test]
    fn stale_identity_missing_fold_and_reordered_rows_cannot_teach() {
        let f = fixture(Task::Regression, 5);
        let mut stale = f.record.clone();
        stale.identity.vocabulary_sha256 = "f".repeat(64);
        stale.identity_sha256 = stale.identity.sha256().unwrap();
        assert!(qualify(&f.state, &stale, &f.evidence).is_err());
        let mut legacy = f.record.clone();
        legacy.identity.comparison = None;
        legacy.identity_sha256 = legacy.identity.sha256().unwrap();
        assert!(qualify(&f.state, &legacy, &f.evidence).is_err());
        let mut incomplete = f.evidence.clone();
        incomplete.fold_predictions_sha256.pop();
        assert!(qualify(&f.state, &f.record, &incomplete).is_err());
        let mut reordered: FoldPredictions =
            blob(&f.state, &f.evidence.fold_predictions_sha256[0]).unwrap();
        reordered.row_ids.reverse();
        let mut evidence = f.evidence.clone();
        evidence.fold_predictions_sha256[0] = f
            .state
            .put_blob(&canonical_json(&reordered).unwrap())
            .unwrap();
        assert!(qualify(&f.state, &f.record, &evidence).is_err());
    }

    #[test]
    fn structural_receipt_without_actual_predictions_or_complete_execution_is_ineligible() {
        let f = fixture(Task::Regression, 5);
        let mut prediction: FoldPredictions =
            blob(&f.state, &f.evidence.fold_predictions_sha256[0]).unwrap();
        prediction.predictions_sha256 = "f".repeat(64);
        let mut evidence = f.evidence.clone();
        evidence.fold_predictions_sha256[0] = f
            .state
            .put_blob(&canonical_json(&prediction).unwrap())
            .unwrap();
        assert!(qualify(&f.state, &f.record, &evidence).is_err());
        let mut result: EvaluationResult =
            blob(&f.state, &prediction.execution_receipt_sha256).unwrap();
        result.status = EvaluationStatus::Failed;
        result.primary = None;
        result.scored_rows = 0;
        result.reason = Some("synthetic failure".into());
        prediction.execution_receipt_sha256 =
            f.state.put_blob(&canonical_json(&result).unwrap()).unwrap();
        evidence.fold_predictions_sha256[0] = f
            .state
            .put_blob(&canonical_json(&prediction).unwrap())
            .unwrap();
        assert!(qualify(&f.state, &f.record, &evidence).is_err());
    }

    #[test]
    fn unknown_accounting_and_unbound_fitted_outputs_cannot_be_substituted() {
        let f = fixture(Task::Regression, 5);
        let mut prediction: FoldPredictions =
            blob(&f.state, &f.evidence.fold_predictions_sha256[0]).unwrap();
        let original: EvaluationResult =
            blob(&f.state, &prediction.execution_receipt_sha256).unwrap();
        let mut fabricated = original.clone();
        fabricated.consumption.cpu_seconds = Some(0.0);
        prediction.execution_receipt_sha256 = f
            .state
            .put_blob(&canonical_json(&fabricated).unwrap())
            .unwrap();
        let mut evidence = f.evidence.clone();
        evidence.fold_predictions_sha256[0] = f
            .state
            .put_blob(&canonical_json(&prediction).unwrap())
            .unwrap();
        assert!(qualify(&f.state, &f.record, &evidence).is_err());

        let mut unrelated = original;
        prediction.fitted_artifact_sha256 = f
            .state
            .put_blob(b"unrelated synthetic fitted state")
            .unwrap();
        unrelated.fitted_state_sha256 = Some(prediction.fitted_artifact_sha256.clone());
        prediction.execution_receipt_sha256 = f
            .state
            .put_blob(&canonical_json(&unrelated).unwrap())
            .unwrap();
        evidence.fold_predictions_sha256[0] = f
            .state
            .put_blob(&canonical_json(&prediction).unwrap())
            .unwrap();
        let error = qualify(&f.state, &f.record, &evidence).unwrap_err();
        assert!(
            error.to_string().contains("expected value")
                || error.to_string().contains("native fitted recipe")
                || error
                    .to_string()
                    .contains("absent from the bound worker outputs")
        );
    }

    #[test]
    fn burns_rejections_guard_margin_and_unqualified_incumbents_are_durable() {
        let f = fixture(Task::Binary, 0);
        assert!(require_unburned_test_partition(
            &f.state,
            &f.record.identity.dataset_hash,
            &f.record.identity.cut_id
        )
        .is_err());
        assert!(qualify(&f.state, &f.record, &f.evidence).is_err());
        let ledger = Ledger::open(f.state.root(), "eligibility");
        ledger.require_intact().unwrap();
        let decisions = ledger.events().unwrap();
        assert!(decisions.iter().any(|event| event["outcome"] == "rejected"
            && event["measurement"]["improved_folds"] == 0
            && event["guard"]["aggregation"] == "mean_fold_loss"));
        let report = hwm::ingest(
            &f.state,
            None,
            vec![Candidate {
                record: f.record.clone(),
                blob: b"synthetic candidate".to_vec(),
                chain_from_source: false,
            }],
            json!({"source":"synthetic"}),
        )
        .unwrap();
        assert_eq!(report.rejected, 1);
        assert!(f
            .state
            .read_pointer(&PointerKey::of(&f.record))
            .unwrap()
            .is_none());
        let policy = GuardPolicy::current();
        assert!(!policy.improves(0.5 - 1e-8, 0.5));
        assert!(policy.improves(0.49, 0.5));
    }

    #[test]
    fn old_execution_cannot_be_attached_after_freeze_and_pruning_keeps_measured_evidence() {
        let f = fixture(Task::Regression, 5);
        let mut evidence = f.evidence.clone();
        let mut experiment: Experiment = blob(&f.state, &evidence.experiment_blob_sha256).unwrap();
        experiment.checkpoints.remove("teaching_freeze");
        evidence.experiment_blob_sha256 = f
            .state
            .put_blob(&canonical_json(&experiment).unwrap())
            .unwrap();
        assert!(qualify(&f.state, &f.record, &evidence).is_err());
        let qualified = qualify(&f.state, &f.record, &f.evidence).unwrap();
        // Simulate mutable journal output pruning after verified bytes entered content-addressed custody.
        for fold in 0..5 {
            fs::remove_dir_all(f.state.root().join(format!("fixture-worker-{fold}"))).unwrap();
        }
        require_qualified(&f.state, &f.record).unwrap();
        assert!(qualified.guard.admits(&qualified));
    }

    #[test]
    fn tampered_prediction_blob_fails_closed() {
        let f = fixture(Task::Regression, 5);
        let prediction: FoldPredictions =
            blob(&f.state, &f.evidence.fold_predictions_sha256[0]).unwrap();
        fs::write(
            f.state.blob_path(&prediction.predictions_sha256),
            b"{\"kind\":\"values\",\"data\":[null,null]}",
        )
        .unwrap();
        assert!(qualify(&f.state, &f.record, &f.evidence).is_err());
        assert_eq!(effective_label_role("test", true).unwrap(), "development");
        assert_eq!(effective_label_role("test", false).unwrap(), "test");
        assert!(effective_label_role("sealed", true).is_err());
    }

    fn inner_fixture(
        f: &Fixture,
    ) -> (
        FreezePlan,
        InnerPartition,
        FreezeReceipt,
        jope_pipeline::search_plan::SelectionPlan,
    ) {
        let partition = derive_inner_partition(
            &f.state,
            &f.plan.panel_sha256,
            "fixture",
            0,
            7,
            &"b".repeat(64),
            &"a".repeat(64),
        )
        .unwrap();
        let mut plan = f.plan.clone();
        plan.comparison.cut_id = partition.identity().unwrap();
        let freeze = freeze_inner_optimization(&f.state, &plan, &partition).unwrap();
        let old: FoldPredictions = blob(&f.state, &f.evidence.fold_predictions_sha256[0]).unwrap();
        let mut outer: EvaluationRequest = blob(&f.state, &old.evaluation_request_sha256).unwrap();
        outer.stage = EvaluationStage::Screen;
        outer.fitting_rows_sha256 = partition.fitting_rows_sha256().unwrap();
        outer.scoring_rows_sha256 = canonical_sha256(&partition.hold_rows).unwrap();
        outer.fitting_rows = partition.fitting_rows.len();
        outer.scoring_rows = partition.hold_rows.len();
        let selection = jope_pipeline::search_plan::SelectionPlan::new(
            outer,
            partition.fitting_rows.clone(),
            partition.hold_rows.clone(),
            partition.inner_validation_rows.clone(),
            partition.fitting_groups.clone(),
            partition.hold_groups.clone(),
        )
        .unwrap();
        (plan, partition, freeze, selection)
    }

    #[test]
    fn inner_partition_excludes_outer_hold_and_pins_comparison_not_candidate() {
        let f = fixture(Task::Regression, 5);
        let (plan, partition, freeze, _) = inner_fixture(&f);
        let policy = load_freeze(&f.state, &freeze).unwrap();
        let targets: Vec<TargetRow> = blob(&f.state, &policy.targets_sha256).unwrap();
        assert_eq!(
            targets.iter().map(|r| &r.id).collect::<Vec<_>>(),
            partition.fitting_rows.iter().collect::<Vec<_>>()
        );
        assert!(targets.iter().all(|r| !partition.hold_rows.contains(&r.id)));
        assert_eq!(targets.len(), 8);
        assert_eq!(
            partition
                .inner_validation_rows
                .iter()
                .map(Vec::len)
                .sum::<usize>(),
            8
        );
        assert_eq!(
            f.state.read_blob(&freeze.freeze_sha256).unwrap().unwrap(),
            fs::read(freeze_path(&f.state, &freeze.comparison_sha256).unwrap()).unwrap()
        );
        assert_eq!(
            blob::<Vec<String>>(&f.state, &partition.fitting_rows_sha256().unwrap()).unwrap(),
            partition.fitting_rows
        );
        let mut changed = partition.clone();
        changed.evaluation_seed += 1;
        assert_ne!(changed.identity().unwrap(), plan.comparison.cut_id);
        changed = partition.clone();
        changed.runtime_inventory_sha256 = "e".repeat(64);
        assert_ne!(changed.identity().unwrap(), plan.comparison.cut_id);
        changed = partition.clone();
        changed.executor_program_sha256 = "f".repeat(64);
        assert_ne!(changed.identity().unwrap(), plan.comparison.cut_id);
        assert!(serde_json::to_value(&partition)
            .unwrap()
            .get("candidate_sha256")
            .is_none());
        // User-visible test labels have been burned on the original partition,
        // even though teaching has a distinct inner comparison identity.
        fs::remove_file(
            burn_path(
                &f.state,
                &plan.comparison.dataset_hash,
                &partition.original_split_sha256,
            )
            .unwrap(),
        )
        .unwrap();
        assert!(require_unburned_test_partition(
            &f.state,
            &plan.comparison.dataset_hash,
            &partition.original_split_sha256
        )
        .is_err());
    }

    #[test]
    fn inner_admission_refuses_retrofitted_unpinned_experiment() {
        let f = fixture(Task::Regression, 5);
        let (_, _, freeze, _) = inner_fixture(&f);
        let policy = load_freeze(&f.state, &freeze).unwrap();
        let experiment: Experiment = blob(&f.state, &f.evidence.experiment_blob_sha256).unwrap();
        let window: ExecutionWindow = blob(&f.state, &f.evidence.window_sha256).unwrap();
        let mut evidence = f.evidence.clone();
        evidence.freeze = freeze;
        evidence.inner = Some(InnerEvidence {
            selection_sha256: "a".repeat(64),
            frozen_recipe_sha256: "a".repeat(64),
            search_receipt_sha256: "a".repeat(64),
            runtime_sha256: "a".repeat(64),
            arm_window_sha256: "a".repeat(64),
            memory_policy_sha256: "a".repeat(64),
            candidate_freeze_sha256: "a".repeat(64),
        });
        assert!(
            inner::bind(&f.state, &policy, &evidence, &experiment, &window, false)
                .err()
                .unwrap()
                .to_string()
                .contains("pre-admission")
        );
    }

    #[test]
    fn inner_fit_binding_rejects_seed_deadline_memory_and_late_freeze_changes() {
        let f = fixture(Task::Regression, 5);
        let (_, _, _, selection) = inner_fixture(&f);
        let experiment: Experiment = blob(&f.state, &f.evidence.experiment_blob_sha256).unwrap();
        let old: FoldPredictions = blob(&f.state, &f.evidence.fold_predictions_sha256[0]).unwrap();
        let result: EvaluationResult = blob(&f.state, &old.execution_receipt_sha256).unwrap();
        let mut worker: WorkerOutcome = blob(&f.state, &result.worker_receipt_sha256).unwrap();
        let mut job: WorkerJob = blob(&f.state, &old.worker_job_sha256).unwrap();
        let arm = jope_pipeline::execution::ArmBudget {
            id_sha256: "1".repeat(64),
            seconds: 30,
        };
        let candidate = "2".repeat(64);
        let request = selection
            .selection_requests(&f.record.genome.program_sha256, &candidate)
            .unwrap()
            .remove(0);
        job.request = request.clone();
        job.arm = Some(arm.clone());
        let binding = inner::BoundInner {
            selection,
            candidate_sha256: candidate,
            arm,
            arm_start: 1000,
            arm_deadline: 5000,
            frozen_at: 4500,
            program_sha256: f.record.genome.program_sha256.clone(),
            memory: jope_pipeline::execution::fit_memory_allocation(&experiment).unwrap(),
            runtime: json!({"program":job.command.program, "argument_prefix":job.command.args,
                "environment":job.command.environment,"inputs":job.command.inputs}),
        };
        worker.consumption.gate_released_unix_ms = Some(1100);
        worker.consumption.observed_exit_unix_ms = Some(4000);
        worker.consumption.fit_memory_limit_bytes = Some(binding.memory.per_fit_bytes);
        let verify = |r: &EvaluationRequest, w: &WorkerOutcome, deadline| {
            inner::validate_execution(
                &binding,
                r,
                &job,
                w,
                1000,
                deadline,
                &f.record.genome.program_sha256,
            )
        };
        verify(&request, &worker, 4100).unwrap();
        let mut stale = request.clone();
        stale.seed ^= 1;
        assert!(verify(&stale, &worker, 4100).is_err());
        assert!(verify(&request, &worker, 6000).is_err());
        let mut bad = worker.clone();
        bad.consumption.fit_memory_limit_bytes = None;
        assert!(verify(&request, &bad, 4100).is_err());
        bad = worker.clone();
        bad.consumption.observed_exit_unix_ms = Some(4600);
        assert!(verify(&request, &bad, 4800).is_err());
        bad = worker;
        bad.consumption.observed_exit_unix_ms = Some(4101);
        assert!(verify(&request, &bad, 4100).is_err());
    }

    mod provenance;
}
