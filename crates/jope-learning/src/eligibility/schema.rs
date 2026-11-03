use super::*;

pub(super) const FREEZE: &str = "jope.optimization-freeze/v1";
pub(super) const PREDICTIONS: &str = "jope.train-oof-predictions/v1";
pub(super) const QUALIFICATION: &str = "jope.measured-teaching/v1";
pub(super) const DIAGNOSTICS: &str = "jope.teaching-diagnostics/v1";
/// Rust fitted artifacts: v1 names no fitting input (unbound, as every artifact before v2); v2
/// names the exact fitting input it was fitted on (`prepared_data::fitting_input_sha256`), W3-T04.
pub(super) const RUST_FITTED: &str = "jope.panel-fitted-rust/v1";
pub(super) const RUST_FITTED_V2: &str = "jope.panel-fitted-rust/v2";

/// Owner decision D4, 2026-09-26: evidence made before a provenance requirement existed stays
/// quarantined. Nothing upgrades it in place, and no route re-admits it; requalification is a
/// refusal with this message, not a procedure, and happens only by producing a new, fully
/// provenanced qualification.
pub(crate) fn quarantined(reason: &str) -> String {
    format!(
        "{reason}. Quarantined: nothing upgrades it in place, and no route re-admits it. \
         Requalification happens only by producing a new, fully provenanced qualification: a new \
         record, qualified with evidence carrying this provenance under a freeze that \
         preregisters it (freezes are immutable, so a comparison frozen without it needs a new \
         comparison identity)"
    )
}

/// `qualify` of a record whose retained qualification differs from the new measurement.
pub(super) fn requalification_refused(record: &HwmRecord) -> String {
    quarantined(&format!(
        "RequalificationRefused: record {} already has a different retained qualification for \
         comparison {}, and retained qualifications are immutable",
        record.provenance.record_sha256, record.identity_sha256
    ))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `TargetRow` is a record for the target row contract in this module: Measured train-side OOF admission. A candidate supplies predictions, never eligibility policy or trusted metric values. The policy, targets and trivial baseline are persisted before optimization.
pub struct TargetRow {
    /// Stable identifier for this value.
    pub id: String,
    /// Group recorded for this target row.
    pub group: String,
    /// Fold recorded for this target row.
    pub fold: u8,
    /// Target recorded for this target row.
    pub target: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `FreezePlan` is a record for the freeze plan contract in this module: Measured train-side OOF admission. A candidate supplies predictions, never eligibility policy or trusted metric values. The policy, targets and trivial baseline are persisted before optimization.
pub struct FreezePlan {
    /// Comparison recorded for this freeze plan.
    pub comparison: HwmIdentity,
    /// `test` observed by optimization is development data; sealed data is never eligible.
    pub original_label_role: String,
    /// SHA-256 digest identifying the associated panel.
    pub panel_sha256: String,
    /// Identifier for the associated dataset.
    pub dataset_id: String,
    /// Evaluation seed recorded for this freeze plan.
    pub evaluation_seed: u64,
    /// Required for teaching admission; a freeze without it is diagnostic only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<ContextPlan>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `FrozenFold` is a record for the frozen fold contract in this module: Measured train-side OOF admission. A candidate supplies predictions, never eligibility policy or trusted metric values. The policy, targets and trivial baseline are persisted before optimization.
pub struct FrozenFold {
    /// Fold recorded for this frozen fold.
    pub fold: u8,
    /// SHA-256 digest identifying the associated training rows.
    pub training_rows_sha256: String,
    /// SHA-256 digest identifying the associated scoring rows.
    pub scoring_rows_sha256: String,
    /// SHA-256 digest identifying the associated baseline predictions.
    pub baseline_predictions_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct FrozenPolicy {
    pub(super) schema: String,
    pub(super) plan_sha256: String,
    pub(super) comparison: HwmIdentity,
    pub(super) label_role: String,
    pub(super) panel_sha256: String,
    pub(super) dataset: PanelDataset,
    pub(super) original_label_role: String,
    pub(super) label_burn_sha256: Option<String>,
    pub(super) guard: GuardPolicy,
    pub(super) evaluation_seed: u64,
    pub(super) targets_sha256: String,
    pub(super) folds: Vec<FrozenFold>,
    pub(super) created_at_unix: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) inner_partition: Option<InnerPartition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) context: Option<ContextPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `FreezeReceipt` is a record for the freeze receipt contract in this module: Measured train-side OOF admission. A candidate supplies predictions, never eligibility policy or trusted metric values. The policy, targets and trivial baseline are persisted before optimization.
pub struct FreezeReceipt {
    /// SHA-256 digest identifying the associated comparison.
    pub comparison_sha256: String,
    /// SHA-256 digest identifying the associated freeze.
    pub freeze_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `FoldPredictions` is a record for the fold predictions contract in this module: Measured train-side OOF admission. A candidate supplies predictions, never eligibility policy or trusted metric values. The policy, targets and trivial baseline are persisted before optimization.
pub struct FoldPredictions {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// SHA-256 digest identifying the associated comparison.
    pub comparison_sha256: String,
    /// SHA-256 digest of the canonical program source.
    pub program_sha256: String,
    /// Fold recorded for this fold predictions.
    pub fold: u8,
    /// SHA-256 digest identifying the associated training rows.
    pub training_rows_sha256: String,
    /// SHA-256 digest identifying the associated scoring rows.
    pub scoring_rows_sha256: String,
    /// Collection of row IDs values carried by this fold predictions.
    pub row_ids: Vec<String>,
    /// SHA-256 digest identifying the associated predictions.
    pub predictions_sha256: String,
    /// SHA-256 digest identifying the associated evaluation request.
    pub evaluation_request_sha256: String,
    /// SHA-256 digest identifying the associated worker job.
    pub worker_job_sha256: String,
    /// SHA-256 digest identifying the associated fitted artifact.
    pub fitted_artifact_sha256: String,
    /// SHA-256 digest identifying the associated execution receipt.
    pub execution_receipt_sha256: String,
    /// SHA-256 digest identifying the associated started receipt.
    pub started_receipt_sha256: String,
    /// Fit started unix recorded for this fold predictions.
    pub fit_started_unix: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `OofEvidence` is a record for the OOF evidence contract in this module: Measured train-side OOF admission. A candidate supplies predictions, never eligibility policy or trusted metric values. The policy, targets and trivial baseline are persisted before optimization.
pub struct OofEvidence {
    /// Freeze recorded for this OOF evidence.
    pub freeze: FreezeReceipt,
    /// SHA-256 digest identifying the associated experiment blob.
    pub experiment_blob_sha256: String,
    /// SHA-256 digest identifying the associated window.
    pub window_sha256: String,
    /// Exactly one content-addressed FoldPredictions blob per fold, in fold order.
    pub fold_predictions_sha256: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional inner value; `None` means it is not available.
    pub inner: Option<InnerEvidence>,
    /// Mandatory for qualification; absent only on legacy/diagnostic submissions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<ContextEvidence>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `Qualification` is a record for the qualification contract in this module: Measured train-side OOF admission. A candidate supplies predictions, never eligibility policy or trusted metric values. The policy, targets and trivial baseline are persisted before optimization.
pub struct Qualification {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Comparison recorded for this qualification.
    pub comparison: HwmIdentity,
    /// Freeze recorded for this qualification.
    pub freeze: FreezeReceipt,
    /// SHA-256 digest identifying the associated record.
    pub record_sha256: String,
    /// SHA-256 digest of the canonical program source.
    pub program_sha256: String,
    /// SHA-256 digest identifying the associated targets.
    pub targets_sha256: String,
    /// SHA-256 digest identifying the associated baseline predictions.
    pub baseline_predictions_sha256: Vec<String>,
    /// SHA-256 digest identifying the associated fold predictions.
    pub fold_predictions_sha256: Vec<String>,
    /// Guard recorded for this qualification.
    pub guard: GuardPolicy,
    /// Label role recorded for this qualification.
    pub label_role: String,
    /// Original label role recorded for this qualification.
    pub original_label_role: String,
    /// SHA-256 digest identifying the associated label burn.
    pub label_burn_sha256: Option<String>,
    /// Candidate loss recorded for this qualification.
    pub candidate_loss: f64,
    /// Baseline loss recorded for this qualification.
    pub baseline_loss: f64,
    /// Improved folds recorded for this qualification.
    pub improved_folds: usize,
    /// Collection of fold losses values carried by this qualification.
    pub fold_losses: Vec<(f64, f64)>,
    /// Scores were recomputed; process execution provenance still comes from the worker receipts.
    pub measurement: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// SHA-256 digest identifying the associated context fitting rows.
    pub context_fitting_rows_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional context outer fold value; `None` means it is not available.
    pub context_outer_fold: Option<u8>,
    /// Verified context receipt; stored legacy qualifications without one never revalidate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_receipt_sha256: Option<String>,
    /// The W2 handoff the context was admitted under (W3-T02, W2-T07); stored qualifications
    /// without one never revalidate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_handoff_sha256: Option<String>,
    /// W3-T04: `Some(true)` when every fold's fitted artifact names, and was checked against, its
    /// fold's authenticated fitting input (`jope.panel-fitted-rust/v2`). Absent when any fold's
    /// artifact is unbound (v1, or the inner route), as for every qualification before it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fitted_inputs_bound: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Optional diagnostics value; `None` means it is not available.
    pub diagnostics: Option<Diagnostics>,
}

/// Descriptive statistics recomputed with the qualification. Admission reads only
/// `fold_losses` under `guard` (mean-fold loss); nothing here can change a decision.
/// Pooled figures weight rows rather than folds and exist only for regression.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Diagnostics {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Number of fold rows represented here.
    pub fold_rows: Vec<usize>,
    /// Candidate minus train-only baseline loss per fold; positive is worse.
    pub fold_loss_deltas: Vec<f64>,
    /// Adverse folds recorded for this diagnostics.
    pub adverse_folds: usize,
    /// Worst fold loss delta recorded for this diagnostics.
    pub worst_fold_loss_delta: f64,
    /// Optional pooled RMSE value; `None` means it is not available.
    pub pooled_rmse: Option<f64>,
    /// Optional pooled baseline RMSE value; `None` means it is not available.
    pub pooled_baseline_rmse: Option<f64>,
    /// `1 - SSE/SST` about the pooled held-out target mean; uses held labels, never admits.
    pub pooled_r2: Option<f64>,
}

pub(super) fn diagnostics(
    fold_losses: &[(f64, f64)],
    fold_rows: Vec<usize>,
    pooled: &[(f64, f64, f64)],
) -> Result<Diagnostics> {
    let fold_loss_deltas: Vec<f64> = fold_losses.iter().map(|(c, b)| c - b).collect();
    let n = pooled.len() as f64;
    let squared = |f: fn(&(f64, f64, f64)) -> f64| pooled.iter().map(|p| f(p).powi(2)).sum::<f64>();
    let (pooled_rmse, pooled_baseline_rmse, pooled_r2) = if pooled.is_empty() {
        (None, None, None)
    } else {
        let mean = pooled.iter().map(|p| p.0).sum::<f64>() / n;
        let sse = squared(|(t, c, _)| t - c);
        let sst = pooled
            .iter()
            .map(|(t, _, _)| (t - mean).powi(2))
            .sum::<f64>();
        (
            Some((sse / n).sqrt()),
            Some((squared(|(t, _, b)| t - b) / n).sqrt()),
            (sst > 0.0).then(|| 1.0 - sse / sst),
        )
    };
    let value = Diagnostics {
        schema: DIAGNOSTICS.into(),
        fold_rows,
        adverse_folds: fold_loss_deltas.iter().filter(|d| **d > 0.0).count(),
        worst_fold_loss_delta: fold_loss_deltas.iter().copied().fold(f64::MIN, f64::max),
        fold_loss_deltas,
        pooled_rmse,
        pooled_baseline_rmse,
        pooled_r2,
    };
    ensure!(
        value.fold_loss_deltas.iter().all(|d| d.is_finite())
            && [pooled_rmse, pooled_baseline_rmse, pooled_r2]
                .into_iter()
                .flatten()
                .all(f64::is_finite),
        "nonfinite teaching diagnostic"
    );
    Ok(value)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StoredQualification {
    evidence: OofEvidence,
    qualification: Qualification,
}

#[derive(Clone, Debug, Serialize)]
/// `TeachingManifest` is a record for the teaching manifest contract in this module: Measured train-side OOF admission. A candidate supplies predictions, never eligibility policy or trusted metric values. The policy, targets and trivial baseline are persisted before optimization.
pub struct TeachingManifest {
    /// Schema identifier and version for the serialized value.
    pub schema: &'static str,
    /// Excluded diagnostic recorded for this teaching manifest.
    pub excluded_diagnostic: usize,
    /// Collection of exclusions values carried by this teaching manifest.
    pub exclusions: Vec<(String, String)>,
    /// Collection of entries values carried by this teaching manifest.
    pub entries: Vec<Qualification>,
}

/// Fixed, versioned admission policy. Caller evidence never supplies thresholds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuardPolicy {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// Folds recorded for this guard policy.
    pub folds: usize,
    /// Minimum improved folds recorded for this guard policy.
    pub minimum_improved_folds: usize,
    /// Aggregation recorded for this guard policy.
    pub aggregation: String,
    /// Baseline recorded for this guard policy.
    pub baseline: String,
    /// Absolute gain recorded for this guard policy.
    pub absolute_gain: f64,
    /// Relative gain recorded for this guard policy.
    pub relative_gain: f64,
}
impl GuardPolicy {
    pub(super) fn current() -> Self {
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
    pub(super) fn improves(&self, candidate: f64, baseline: f64) -> bool {
        candidate.is_finite()
            && baseline.is_finite()
            && baseline - candidate > self.absolute_gain + self.relative_gain * baseline.abs()
    }
    pub(super) fn admits(&self, measured: &Qualification) -> bool {
        measured.fold_losses.len() == self.folds
            && measured.improved_folds >= self.minimum_improved_folds
            && self.improves(measured.candidate_loss, measured.baseline_loss)
    }
}
