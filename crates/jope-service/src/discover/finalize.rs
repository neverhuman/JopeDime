//! Finalization of the numeric M1 champion. Prepared partitions are read only after
//! the durable once-only claim and the row-audit finalization boundary.
//!
//! Keep one `Finalizer` per store and share `journal()` with its subscribers. HTTP,
//! CLI and constrained-split preparation remain adapters; this API never searches.
use super::{
    audit::{AuditRowId, RowAudit, RowAuditPhase},
    r#loop::{Champion, Executor, Plan},
};
use crate::{
    dataset::jmat::{JmatCell, Matrix},
    events::{EventLog, RunUpdate},
    objective::regression::RegressionObjective,
    store::{BlobRef, Record, RecordKey, Store, StoreError},
    worker::{Attempt, Request, WorkerFailure},
};
use anyhow::{bail, ensure, Context, Result};
use jope_api::{
    ids::{ArtifactId, RunId},
    CompletionReason, EventBody, FinalScores, Refit, RunFinishedBody, RunPhase, RunSnapshot,
    RunState,
};
use jope_runtime::{
    math_pipeline::{
        InputContract, NumericColumn, PipelineState, TargetScale, INPUT_PART, PIPELINE_PART,
    },
    model_object::{Manifest, ModelObject, Part, Producer, MATH_PIPELINE, SCHEMA},
    table::CsvTable,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Read, path::PathBuf, sync::Arc};

const HANDOFF_MAX: u64 = 16 * 1024 * 1024;

/// Every file contains exactly these dataset-local source rows in this order.
/// The split preparer supplies the provenance roster; no payload is read here.
#[derive(Clone, Debug, Serialize)]
pub struct Partition {
    pub path: PathBuf,
    pub sha256: String,
    pub rows: Vec<AuditRowId>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FinalizationRequest {
    pub run_id: RunId,
    pub fit: Partition,
    pub holdout: Partition,
    pub test: Option<Partition>,
    pub refit: Refit,
    /// Runtime model-object root, not a client-selected output file.
    pub model_root: PathBuf,
    pub scratch_root: PathBuf,
    pub max_input_bytes: u64,
    pub max_model_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FinalizationReceipt {
    pub run_id: RunId,
    pub algorithm_id: jope_api::ids::AlgorithmId,
    pub model_artifact: ArtifactId,
    pub model_digest: String,
    pub model_path: PathBuf,
    pub refit: Refit,
    pub scores: FinalScores,
    pub completion_reason: CompletionReason,
    pub budget_bound_by: Option<String>,
    pub budget_used: jope_api::BudgetUsed,
    pub nondeterminism: jope_api::Nondeterminism,
    /// HOLDOUT describes this FIT-only model, even when the export is refitted.
    pub holdout_model_digest: String,
    pub holdout_scope: String,
    pub test_scope: Option<String>,
    pub fit_rows: usize,
    pub holdout_rows: usize,
    pub test_rows: usize,
    pub audit: Value,
}

pub struct Finalizer {
    store: Arc<Store>,
    journal: Arc<EventLog>,
}
impl Finalizer {
    pub(crate) fn with_journal(store: Arc<Store>, journal: Arc<EventLog>) -> Self {
        Self { store, journal }
    }
    pub fn new(store: Arc<Store>) -> Self {
        Self {
            journal: Arc::new(EventLog::new(Arc::clone(&store))),
            store,
        }
    }
    pub fn journal(&self) -> &Arc<EventLog> {
        &self.journal
    }

    /// A failed/interrupted claim is never automatically retried: its protected
    /// read outcome may be unknown. A completed replay verifies immutable output.
    pub fn finalize(
        &self,
        request: &FinalizationRequest,
        executor: &impl Executor,
    ) -> Result<FinalizationReceipt> {
        let record = self
            .store
            .get(&RecordKey::Run(request.run_id.clone()))?
            .context("run missing")?;
        let identity = digest(&serde_json::to_vec(&(
            request,
            &record.data["checkpoint"]["plan"],
            &record.data["checkpoint"]["champion"],
        ))?);
        let key = RecordKey::Command {
            principal: "discover.finalization.v1".into(),
            idempotency_key: request.run_id.to_string(),
        };
        if let Some(record) = self.store.get(&key)? {
            ensure!(
                record.data["identity"] == identity,
                "finalization request changed"
            );
            ensure!(
                record.data["state"] == "completed",
                "finalization outcome failed or unknown; protected rows will not be reread"
            );
            let receipt: FinalizationReceipt =
                serde_json::from_value(record.data["receipt"].clone())?;
            let object = ModelObject::read(&receipt.model_path, request.max_model_bytes)?;
            ensure!(
                object.id() == receipt.model_digest,
                "finalized object identity changed"
            );
            object.verify_executable()?;
            return Ok(receipt);
        }
        let snapshot = self
            .journal
            .snapshot(&request.run_id)?
            .context("run missing")?;
        ensure!(
            snapshot.state == RunState::Running
                || (snapshot.state == RunState::Paused
                    && record.run.as_ref().is_some_and(|r| r.interrupted)),
            "run cannot finalize"
        );
        let checkpoint = record.data["checkpoint"].clone();
        let plan: Plan = serde_json::from_value(checkpoint["plan"].clone())?;
        let champion: Champion = serde_json::from_value(checkpoint["champion"].clone())
            .context("finite champion missing")?;
        let reason: CompletionReason = serde_json::from_value(checkpoint["completed"].clone())
            .context("search is not complete")?;
        ensure!(
            plan.run_id == request.run_id
                && champion.loss.is_finite()
                && champion.value.is_finite(),
            "invalid completed champion"
        );
        self.store.verify_blob(&champion.model)?;
        ensure!(
            snapshot
                .best
                .as_ref()
                .is_some_and(|best| best.algorithm_id.to_string()
                    == champion.candidate.algorithm_id
                    && best.cv_value.to_bits() == champion.value.to_bits()),
            "snapshot and selected champion differ"
        );
        ensure!(
            request.fit.path == plan.matrix_path && request.fit.sha256 == plan.matrix_sha256,
            "FIT differs from the search plan"
        );
        ensure!(
            request
                .fit
                .rows
                .iter()
                .chain(&request.holdout.rows)
                .all(|r| r.dataset == plan.dataset_id),
            "training roster dataset mismatch"
        );
        ensure!(
            request.fit.rows.len() >= 2
                && !request.holdout.rows.is_empty()
                && request.test.as_ref().is_none_or(|p| !p.rows.is_empty()),
            "empty finalization partition"
        );
        ensure!(
            request.max_input_bytes > 0
                && request.max_model_bytes > 0
                && request.model_root.is_absolute()
                && request.scratch_root.is_absolute(),
            "invalid finalization limits or roots"
        );
        for part in [&request.fit, &request.holdout]
            .into_iter()
            .chain(request.test.as_ref())
        {
            ensure!(
                part.path.is_absolute() && is_digest(&part.sha256),
                "invalid prepared partition"
            );
        }
        let fitness = jope_runtime::fitness::Fitness::parse(&plan.objective)?;
        ensure!(
            fitness.columns().is_empty() && !fitness.requires_order(),
            "M1 finalization requires a column-free, order-independent regression objective"
        );
        let mut audit = RowAudit::new(
            ids(&request.fit),
            ids(&request.holdout),
            request.test.as_ref().map(ids).unwrap_or_default(),
        )?;
        self.store.transaction(|tx| {
            if tx.get(&key)?.is_some() {
                return Err(StoreError::Invalid("finalization already claimed"));
            }
            let current = tx
                .get(&RecordKey::Run(request.run_id.clone()))?
                .ok_or(StoreError::Invalid("run disappeared"))?;
            if current != record
                || current
                    .run
                    .as_ref()
                    .is_none_or(|r| r.revision != snapshot.revision.0 || r.state != snapshot.state)
            {
                return Err(StoreError::Invalid("run changed before finalization"));
            }
            tx.put(
                &key,
                &Record::metadata(json!({"identity":identity,"state":"started"}), vec![]),
            )
        })?;
        audit.begin_finalization();
        let result = self.execute(
            request,
            &plan,
            &champion,
            &snapshot,
            checkpoint.clone(),
            reason,
            executor,
            &mut audit,
            &key,
            &identity,
        );
        if let Err(error) = &result {
            let failure = json!({"identity":identity,"state":"failed_or_unknown","error":error.to_string(),"audit":audit});
            let mut failed = snapshot.clone();
            if let Some(revision) = failed.revision.0.checked_add(1) {
                failed.revision.0 = revision;
                failed.state = RunState::Failed;
                failed.phase = RunPhase::Finalizing;
                let _ = self.journal.commit(
                    RunUpdate {
                        expected_revision: Some(snapshot.revision.0),
                        snapshot: failed,
                        checkpoint,
                        blobs: record.blobs,
                    },
                    vec![],
                    |tx| {
                        let old = tx
                            .get(&key)?
                            .ok_or(StoreError::Invalid("finalization claim missing"))?;
                        if old.data["state"] != "started" {
                            return Err(StoreError::Invalid("finalization already resolved"));
                        }
                        tx.put(&key, &Record::metadata(failure.clone(), old.blobs))
                    },
                );
            }
            // Stale/cancelled or unknown commit: preserve a completed receipt and
            // never overwrite a newer controller state, but fence further reads.
            let _ = self.store.transaction(|tx| {
                if let Some(old) = tx.get(&key)? {
                    if old.data["state"] == "started" {
                        tx.put(&key, &Record::metadata(failure, old.blobs))?;
                    }
                }
                Ok(())
            });
        }
        result
    }

    fn ensure_current(&self, expected: &RunSnapshot) -> Result<()> {
        let current = self
            .journal
            .snapshot(&expected.run_id)?
            .context("run disappeared")?;
        ensure!(
            current.revision == expected.revision
                && current.state == expected.state
                && current.event_watermark == expected.event_watermark,
            "controller changed during finalization"
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn execute(
        &self,
        request: &FinalizationRequest,
        plan: &Plan,
        champion: &Champion,
        snapshot: &RunSnapshot,
        checkpoint: Value,
        reason: CompletionReason,
        executor: &impl Executor,
        audit: &mut RowAudit,
        key: &RecordKey,
        identity: &str,
    ) -> Result<FinalizationReceipt> {
        fs::create_dir_all(&request.scratch_root)?;
        ensure!(
            fs::symlink_metadata(&request.scratch_root)?.is_dir(),
            "scratch must be a real directory"
        );
        let scratch = tempfile::Builder::new()
            .prefix("finalize-")
            .tempdir_in(&request.scratch_root)?;
        let objective = RegressionObjective::parse(&plan.objective)?;
        let (fit_table, fit_object) = audit.with_rows(
            RowAuditPhase::FinalizationFit,
            &ids(&request.fit),
            |_| -> Result<_> {
                let table = load(&request.fit, plan, request.max_input_bytes)?;
                let object = refit(
                    executor,
                    plan,
                    champion,
                    &table,
                    &request.fit.path,
                    &request.fit.sha256,
                    Refit::Fit,
                    scratch.path(),
                    0,
                    request.max_model_bytes,
                )?;
                Ok((table, object))
            },
        )??;
        self.ensure_current(snapshot)?;
        let (holdout_table, holdout) = audit.with_rows(
            RowAuditPhase::HoldoutScore,
            &ids(&request.holdout),
            |_| -> Result<_> {
                let table = load(&request.holdout, plan, request.max_input_bytes)?;
                let score = score(&fit_object, &table, &plan.target, &objective)?;
                Ok((table, score))
            },
        )??;
        let mut union_ids = ids(&request.fit);
        union_ids.extend(ids(&request.holdout));
        self.ensure_current(snapshot)?;
        let object = match request.refit {
            Refit::Fit => None,
            Refit::FitHoldout => Some(audit.with_rows(
                RowAuditPhase::FinalizationRefit,
                &union_ids,
                |_| -> Result<_> {
                    let mut table = fit_table.clone();
                    table.rows.extend(holdout_table.rows.clone());
                    let bytes = crate::dataset::jmat::encode(&table)?;
                    ensure!(
                        bytes.len() as u64 <= request.max_input_bytes,
                        "refit matrix exceeds input bound"
                    );
                    let path = scratch.path().join("fit-holdout.jmat");
                    fs::write(&path, &bytes)?;
                    refit(
                        executor,
                        plan,
                        champion,
                        &table,
                        &path,
                        &digest(&bytes),
                        Refit::FitHoldout,
                        scratch.path(),
                        1,
                        request.max_model_bytes,
                    )
                },
            )??),
        };
        let exported = object.as_ref().unwrap_or(&fit_object);
        self.ensure_current(snapshot)?;
        let test = request
            .test
            .as_ref()
            .map(|part| {
                audit.with_rows(RowAuditPhase::TestScore, &ids(part), |_| -> Result<_> {
                    let table = load(part, plan, request.max_input_bytes)?;
                    score(exported, &table, &plan.target, &objective)
                })
            })
            .transpose()?
            .transpose()?;
        self.ensure_current(snapshot)?;
        let fit_path = fit_object.save(&request.model_root, request.max_model_bytes)?;
        let model_path = if object.is_some() {
            exported.save(&request.model_root, request.max_model_bytes)?
        } else {
            fit_path
        };
        let artifact: ArtifactId = format!("art_{}", &exported.id()[7..31]).parse()?;
        let nondeterminism: jope_api::Nondeterminism = serde_json::from_value(
            json!({"count":checkpoint["nondeterminism"].as_array().context("limit inventory missing")?.len().to_string(),"experiment_ids":checkpoint["nondeterminism"]}),
        )?;
        let receipt = FinalizationReceipt {
            run_id: request.run_id.clone(),
            algorithm_id: champion.candidate.algorithm_id.parse()?,
            model_artifact: artifact.clone(),
            model_digest: exported.id().into(),
            model_path,
            refit: request.refit,
            completion_reason: reason,
            budget_bound_by: match reason {
                CompletionReason::BudgetTime => Some("time".into()),
                CompletionReason::BudgetEvaluations => Some("evaluations".into()),
                _ => None,
            },
            budget_used: snapshot.budget_used.clone(),
            nondeterminism: nondeterminism.clone(),
            scores: FinalScores {
                cv: Some(champion.value),
                holdout: Some(holdout),
                test,
            },
            holdout_model_digest: fit_object.id().into(),
            holdout_scope: "holdout_scored_with_fit_only_model".into(),
            test_scope: request.test.as_ref().map(|_| {
                match request.refit {
                    Refit::Fit => "test_scored_with_fit_model",
                    Refit::FitHoldout => "test_scored_with_fit_holdout_model",
                }
                .into()
            }),
            fit_rows: request.fit.rows.len(),
            holdout_rows: request.holdout.rows.len(),
            test_rows: request.test.as_ref().map_or(0, |p| p.rows.len()),
            audit: serde_json::to_value(&audit)?,
        };
        let mut blobs = vec![champion.model.clone()];
        let fit_refs = publish(&self.store, &fit_object)?;
        blobs.extend(fit_refs);
        let object_refs = publish(&self.store, exported)?;
        blobs.extend(object_refs.clone());
        let mut finished = snapshot.clone();
        finished.revision.0 = finished
            .revision
            .0
            .checked_add(1)
            .context("revision exhausted")?;
        finished.state = RunState::Completed;
        finished.phase = RunPhase::Finalizing;
        finished.completion_reason = Some(reason);
        finished
            .best
            .as_mut()
            .context("snapshot champion missing")?
            .model_artifact = Some(artifact.clone());
        let event = EventBody::RunFinished(RunFinishedBody {
            completion_reason: reason,
            champion: finished.best.clone(),
            scores: receipt.scores.clone(),
            budget_used: finished.budget_used.clone(),
            nondeterminism,
            evaluation_receipt: finished.evaluation_receipt.clone(),
        });
        self.journal.commit(RunUpdate { expected_revision: Some(snapshot.revision.0), snapshot: finished, checkpoint, blobs }, vec![event], |tx| {
            let old = tx.get(key)?.ok_or(StoreError::Invalid("finalization claim missing"))?;
            if old.data["state"] != "started" || old.data["identity"] != identity { return Err(StoreError::Invalid("finalization claim changed")); }
            let artifact_key = RecordKey::Artifact(artifact);
            let record = Record::metadata(json!({"kind":"finalized_model","model_digest":exported.id(),"refit":request.refit,"manifest":exported.manifest()}), object_refs.clone());
            if let Some(old) = tx.get(&artifact_key)? { if old != record { return Err(StoreError::Integrity("finalized artifact collision")); } } else { tx.put(&artifact_key, &record)?; }
            tx.put(key, &Record::metadata(json!({"identity":identity,"state":"completed","receipt":receipt}), object_refs))
        })?;
        Ok(receipt)
    }
}

fn ids(part: &Partition) -> Vec<AuditRowId> {
    part.rows.clone()
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn load(part: &Partition, plan: &Plan, maximum: u64) -> Result<CsvTable> {
    let bytes = jope_runtime::model_object::read_bounded(&part.path, maximum)?;
    ensure!(digest(&bytes) == part.sha256, "partition digest mismatch");
    let matrix = Matrix::open(bytes)?;
    ensure!(
        matrix.rows() == part.rows.len(),
        "partition roster length differs from matrix"
    );
    let columns = plan
        .features
        .iter()
        .chain(std::iter::once(&plan.target))
        .map(|name| {
            matrix
                .columns()
                .iter()
                .position(|c| &c.name == name)
                .context("finalization column missing")
        })
        .collect::<Result<Vec<_>>>()?;
    let mut rows = Vec::with_capacity(matrix.rows());
    for row in 0..matrix.rows() {
        rows.push(columns.iter().map(|&column| match matrix.cell(row, column) {
            Some(JmatCell::Numeric(v)) if v.is_finite() => Ok(Some(v.to_string())),
            _ => bail!("M1 finalization requires complete finite numeric features and target; frozen math-pipeline cannot preserve mean-imputed missing values"),
        }).collect::<Result<Vec<_>>>()?);
    }
    let headers = plan
        .features
        .iter()
        .chain(std::iter::once(&plan.target))
        .cloned()
        .collect();
    Ok(CsvTable {
        headers,
        rows,
        sha256: part.sha256.clone(),
        bytes: matrix.bytes().len() as u64,
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    attempt: Attempt,
    model_json: String,
    model_digest: String,
    predictions_digest: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeFit {
    version: u32,
    features: Vec<String>,
    dsl: String,
    min: Vec<f64>,
    max: Vec<f64>,
    fill: Vec<f64>,
    target_min: f64,
    target_scale: f64,
    model: feat_math::Model,
}

#[allow(clippy::too_many_arguments)]
fn refit(
    executor: &impl Executor,
    plan: &Plan,
    champion: &Champion,
    table: &CsvTable,
    matrix_path: &std::path::Path,
    matrix_sha: &str,
    kind: Refit,
    scratch: &std::path::Path,
    stage: u32,
    max_model: u64,
) -> Result<ModelObject> {
    let directory = scratch.join(format!("fit-{stage}"));
    fs::create_dir(&directory)?;
    let output = fs::canonicalize(directory)?.join("model.json");
    // The existing private refit mode fits every mapped row and ignores folds.
    // Keep its required two-fold envelope constant-size under the frame bound.
    let maximum = HANDOFF_MAX.min(plan.evaluation_rss_bytes).min(max_model);
    let request = Request {
        attempt: Attempt {
            run: plan.run_id.clone(),
            round: u64::MAX,
            slot: stage,
            attempt: 1,
        },
        matrix_path: matrix_path.into(),
        fold_spec: json!({"version":1,"matrix_sha256":matrix_sha,"target":plan.target,"features":plan.features,"folds":[{"repeat":0,"train":[0],"validation":[1]},{"repeat":0,"train":[1],"validation":[0]}],"refit_output":output,"refit_max_bytes":maximum}),
        dsl: champion.candidate.dsl.clone(),
        objective: json!(plan.objective),
        seed: plan.seed,
        limits: crate::worker::Limits {
            wall_ms: plan.evaluation_wall_ms,
            rss_bytes: plan.evaluation_rss_bytes,
        },
    };
    let response = executor.evaluate(&request);
    ensure!(
        response.attempt == request.attempt && response.valid() && response.failure.is_none(),
        "bounded finalization fit failed: {:?}",
        response.failure.unwrap_or(WorkerFailure::Protocol)
    );
    ensure!(
        response.scores.len() == 3,
        "finalization worker score shape changed"
    );
    let meta = fs::symlink_metadata(&output)?;
    ensure!(
        meta.is_file() && meta.len() <= maximum,
        "invalid private fit handoff"
    );
    let mut bytes = vec![];
    fs::File::open(&output)?
        .take(maximum + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= maximum,
        "private fit handoff exceeds bound"
    );
    let envelope: Envelope = serde_json::from_slice(&bytes)?;
    ensure!(
        envelope.attempt == request.attempt
            && envelope.predictions_digest == response.predictions_digest
            && digest(envelope.model_json.as_bytes()) == envelope.model_digest,
        "private fit fence or digest mismatch"
    );
    let native: NativeFit = serde_json::from_str(&envelope.model_json)?;
    ensure!(
        native.version == 1
            && native.features == plan.features
            && native.dsl == champion.candidate.dsl
            && native.fill.len() == plan.features.len()
            && native.fill.iter().all(|x| x.is_finite())
            && native.min.len() == plan.features.len()
            && native.max.len() == plan.features.len(),
        "private native model contract mismatch"
    );
    let input = InputContract::new(
        plan.features
            .iter()
            .enumerate()
            .map(|(i, name)| NumericColumn {
                name: name.clone(),
                min: native.min[i],
                max: native.max[i],
                constant: native.min[i] == native.max[i],
            })
            .collect(),
        (0..plan.features.len()).collect(),
        TargetScale {
            name: plan.target.clone(),
            min: native.target_min,
            scale: native.target_scale,
        },
    )?;
    let parts: BTreeMap<String, Vec<u8>> = BTreeMap::from([
        (
            PIPELINE_PART.into(),
            serde_json::to_vec(&PipelineState {
                dsl: native.dsl,
                model: native.model,
            })?,
        ),
        (INPUT_PART.into(), serde_json::to_vec(&input)?),
    ]);
    let object = ModelObject::new(
        Manifest {
            schema: SCHEMA.into(),
            minimum_reader_version: 1,
            format: MATH_PIPELINE.into(),
            producer: Producer::default(),
            primary: PIPELINE_PART.into(),
            required_capabilities: vec![],
            files: parts
                .iter()
                .map(|(path, data)| Part::new(path, "application/json", data))
                .collect(),
            metadata: json!({"refit":kind,"run_id":plan.run_id,"algorithm_id":champion.candidate.algorithm_id,"problem_digest":plan.problem_digest,"binary_version":plan.binary_version,"fitting":"selection_only","fit_stage":"champion_refit","prediction_version":2,"holdout_score_scope":"fit_only_model","training_rows":table.rows.len(),"missing_input_policy":"registered math-pipeline decoder: scaled minimum"}),
        },
        parts,
        max_model,
    )?;
    object.verify_executable()?;
    // Verify that frozen serialization preserves the actual bounded fit, including
    // target inversion. Training predictions never choose a new champion.
    let predictions = predict(&object, table)?;
    let target = table
        .headers
        .iter()
        .position(|c| c == &plan.target)
        .context("target missing")?;
    let mut measured = Vec::with_capacity(predictions.len() * 16);
    for (row, prediction) in table.rows.iter().zip(predictions) {
        let label: f64 = row[target].as_deref().context("target missing")?.parse()?;
        measured.extend(label.to_le_bytes());
        measured.extend(prediction.to_le_bytes());
    }
    ensure!(
        digest(&measured) == response.predictions_digest,
        "frozen export contract cannot exactly represent this native fit"
    );
    Ok(object)
}
fn predict(object: &ModelObject, table: &CsvTable) -> Result<Vec<f64>> {
    jope_runtime::budget::Budget::new(None, 1)
        .install(|| object.math_pipeline()?.predict_table(table).map(|x| x.0))?
}
fn score(
    object: &ModelObject,
    table: &CsvTable,
    target: &str,
    objective: &RegressionObjective,
) -> Result<f64> {
    let predictions = predict(object, table)?;
    let column = table
        .headers
        .iter()
        .position(|c| c == target)
        .context("target missing")?;
    let labels = table
        .rows
        .iter()
        .map(|r| {
            r[column]
                .as_deref()
                .context("target missing")?
                .parse::<f64>()
                .map_err(anyhow::Error::from)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(objective.evaluate(&labels, &predictions)?.value)
}
fn publish(store: &Store, object: &ModelObject) -> Result<Vec<BlobRef>> {
    std::iter::once(object.manifest_bytes())
        .chain(object.parts().values().map(Vec::as_slice))
        .map(|bytes| store.publish_blob(bytes).map_err(anyhow::Error::from))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discover::r#loop::{evaluate_native, Discovery, HeuristicAdmission, SearchSpace};
    use crate::worker::{Response, Scores};
    use rand_chacha::ChaCha12Rng;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    };
    const TRAIN: &[u8] =
        include_bytes!("../../../../test-support/datasets/v1/d2-regression/train.csv");
    const TRAIN_SHA: &str = "d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931";

    struct Ridge;
    impl SearchSpace for Ridge {
        fn canonical(&self, dsl: &str) -> Result<String> {
            Ok(feat_math::GenomeIr::parse(dsl)?.canonical())
        }
        fn seeds(&self) -> Result<Vec<String>> {
            Ok(vec![self.canonical("(pipe X (ridge))")?])
        }
        fn propose(&self, _: &[String], _: usize, _: &mut ChaCha12Rng) -> Result<Vec<String>> {
            Ok(vec![])
        }
    }
    #[derive(Default)]
    struct Native {
        calls: Mutex<Vec<(u64, u32, PathBuf)>>,
        failure: Option<WorkerFailure>,
    }
    impl Executor for Native {
        fn evaluate(&self, request: &Request) -> Response {
            self.calls.lock().unwrap().push((
                request.attempt.round,
                request.attempt.slot,
                request.matrix_path.clone(),
            ));
            if let Some(reason) = self.failure {
                return Response::failed(request.attempt.clone(), reason, 0);
            }
            let matrix = Matrix::open(fs::read(&request.matrix_path).unwrap()).unwrap();
            match evaluate_native(request, &matrix) {
                Ok(Scores {
                    values,
                    fold_scores,
                    predictions_digest,
                }) => Response {
                    attempt: request.attempt.clone(),
                    scores: values,
                    fold_scores,
                    predictions_digest,
                    wall_ms: 0,
                    failure: None,
                },
                Err(reason) => Response::failed(request.attempt.clone(), reason, 0),
            }
        }
        fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
            let temp = tempfile::tempdir()?;
            let output = temp.path().join("model.json");
            let mut request = request.clone();
            request.fold_spec["refit_output"] = json!(output);
            request.fold_spec["refit_max_bytes"] = json!(HANDOFF_MAX);
            ensure!(
                self.evaluate(&request).failure.is_none(),
                "native fixture fit failed"
            );
            let envelope: Envelope = serde_json::from_slice(&fs::read(output)?)?;
            Ok(envelope.model_json.into_bytes())
        }
    }
    struct Fixture {
        directory: tempfile::TempDir,
        store: Arc<Store>,
        plan: Plan,
        request: FinalizationRequest,
    }
    fn fixture(missing: bool) -> Fixture {
        assert_eq!(digest(TRAIN), TRAIN_SHA);
        let directory = tempfile::tempdir().unwrap();
        let mut headers: Vec<_> = (0..21).map(|i| format!("x{i}")).collect();
        headers.push("target".into());
        let mut csv = headers.join(",").into_bytes();
        csv.push(b'\n');
        csv.extend_from_slice(TRAIN);
        let table =
            jope_runtime::table::parse_csv(&csv, &jope_runtime::limits::Limits::default()).unwrap();
        // Source is only the approved TRAIN object; logical roles use disjoint
        // source ordinals. No external HOLDOUT/TEST/sealed object is opened.
        let (roster, _) = crate::discover::splits::random_split(
            table.rows.len(),
            table.rows.len() * 4 / 5,
            73,
            "dsc09/train-only-fixture",
        )
        .unwrap();
        let dataset: jope_api::ids::DatasetId = "ds_21fbfe8b53dbcd605d79486b".parse().unwrap();
        let part = |name: &str, range: std::ops::Range<usize>, absent: bool| {
            let source: Vec<_> = range.map(|i| roster[i]).collect();
            let mut selected = table.clone();
            selected.rows = source.iter().map(|&row| table.rows[row].clone()).collect();
            if absent {
                selected.rows[0][0] = None;
            }
            let bytes = crate::dataset::jmat::encode(&selected).unwrap();
            let path = directory.path().join(name);
            fs::write(&path, &bytes).unwrap();
            Partition {
                path,
                sha256: digest(&bytes),
                rows: source
                    .into_iter()
                    .map(|row| AuditRowId {
                        dataset: dataset.clone(),
                        row: row as u64,
                    })
                    .collect(),
            }
        };
        let fit = part("fit.jmat", 0..40, missing);
        let holdout = part("holdout.jmat", 40..50, false);
        let test = part("test.jmat", 50..60, false);
        let plan = Plan {
            run_id: RunId::new(),
            dataset_id: dataset,
            problem_digest: digest(b"DSC09 TRAIN-only logical finalization fixture"),
            bundle_digest: digest(b"heuristic M1"),
            binary_version: std::env::var("JOPE_DSC09_SOURCE_SHA")
                .unwrap_or_else(|_| "dsc09-unit-fixture".into()),
            matrix_path: fit.path.clone(),
            matrix_sha256: fit.sha256.clone(),
            target: "target".into(),
            features: headers[..21].to_vec(),
            objective: "rmse".into(),
            seed: 73,
            planned_workers: 8,
            effort: jope_api::Effort::Fast,
            evaluations: 1,
            wall_seconds: 120,
            evaluation_wall_ms: 10_000,
            evaluation_rss_bytes: 512 * 1024 * 1024,
            groups: None,
            times: None,
            gap_rows: 0,
        };
        let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
        let request = FinalizationRequest {
            run_id: plan.run_id.clone(),
            fit,
            holdout,
            test: Some(test),
            refit: Refit::FitHoldout,
            model_root: directory.path().join("models"),
            scratch_root: directory.path().join("scratch"),
            max_input_bytes: 1024 * 1024,
            max_model_bytes: HANDOFF_MAX,
        };
        Fixture {
            directory,
            store,
            plan,
            request,
        }
    }
    fn search(f: &Fixture, executor: &impl Executor) -> Champion {
        let mut d = Discovery::open(Arc::clone(&f.store), f.plan.clone()).unwrap();
        d.advance(&Ridge, &HeuristicAdmission, executor, 1, usize::MAX)
            .unwrap();
        assert!(d.completed().is_some());
        d.champion().unwrap().clone()
    }
    fn claim_key(run: &RunId) -> RecordKey {
        RecordKey::Command {
            principal: "discover.finalization.v1".into(),
            idempotency_key: run.to_string(),
        }
    }

    #[test]
    fn finalization_audits_each_real_partition_and_declares_the_refit_and_score_scopes() {
        for kind in [Refit::Fit, Refit::FitHoldout] {
            let mut f = fixture(false);
            f.request.refit = kind;
            let native = Native::default();
            let selected = search(&f, &native);
            assert!(native
                .calls
                .lock()
                .unwrap()
                .iter()
                .all(|(_, _, path)| path == &f.plan.matrix_path));
            let finalizer = Finalizer::new(Arc::clone(&f.store));
            let receipt = finalizer.finalize(&f.request, &native).unwrap();
            assert_eq!(receipt.scores.cv, Some(selected.value));
            assert_eq!(
                receipt.completion_reason,
                CompletionReason::BudgetEvaluations
            );
            assert_eq!(receipt.budget_bound_by.as_deref(), Some("evaluations"));
            assert_eq!(receipt.budget_used.evaluations.0, 1);
            assert_eq!(receipt.nondeterminism.count.0, 0);
            assert!(receipt.scores.holdout.unwrap().is_finite());
            assert!(receipt.scores.test.unwrap().is_finite());
            let exported = ModelObject::read(&receipt.model_path, HANDOFF_MAX).unwrap();
            exported.verify_executable().unwrap();
            assert_eq!(
                exported.manifest().metadata["refit"],
                serde_json::to_value(kind).unwrap()
            );
            assert_eq!(
                exported.manifest().metadata["training_rows"],
                if kind == Refit::Fit { 40 } else { 50 }
            );
            assert_eq!(
                receipt.model_digest == receipt.holdout_model_digest,
                kind == Refit::Fit
            );
            let entries = receipt.audit["entries"].as_array().unwrap();
            assert_eq!(
                entries
                    .iter()
                    .filter(|e| e["phase"] == "test_score")
                    .count(),
                1
            );
            for entry in entries {
                let phase = entry["phase"].as_str().unwrap();
                for row in entry["rows"].as_array().unwrap() {
                    let id = row["row"].as_u64().unwrap();
                    if f.request.holdout.rows.iter().any(|r| r.row == id) {
                        assert!(matches!(phase, "holdout_score" | "finalization_refit"));
                    }
                    if f.request
                        .test
                        .as_ref()
                        .unwrap()
                        .rows
                        .iter()
                        .any(|r| r.row == id)
                    {
                        assert_eq!(phase, "test_score");
                    }
                }
            }
            let snapshot = finalizer
                .journal()
                .snapshot(&f.plan.run_id)
                .unwrap()
                .unwrap();
            assert_eq!(snapshot.state, RunState::Completed);
            assert_eq!(snapshot.phase, RunPhase::Finalizing);
            assert_eq!(
                snapshot.best.unwrap().algorithm_id.to_string(),
                selected.candidate.algorithm_id
            );
            let events = finalizer
                .journal()
                .page(&f.plan.run_id, 0, 256, &[])
                .unwrap()
                .unwrap();
            assert_eq!(
                events
                    .events
                    .iter()
                    .filter(|e| matches!(e.event, EventBody::RunFinished(_)))
                    .count(),
                1
            );
        }
    }

    #[test]
    fn completed_replay_and_recovery_do_not_read_or_score_protected_partitions_again() {
        let f = fixture(false);
        let native = Native::default();
        search(&f, &native);
        let finalizer = Finalizer::new(Arc::clone(&f.store));
        let first = finalizer.finalize(&f.request, &native).unwrap();
        let calls = native.calls.lock().unwrap().len();
        fs::rename(
            &f.request.holdout.path,
            f.directory.path().join("holdout-unavailable"),
        )
        .unwrap();
        fs::rename(
            &f.request.test.as_ref().unwrap().path,
            f.directory.path().join("test-unavailable"),
        )
        .unwrap();
        drop(finalizer);
        drop(f.store);
        let recovered = Arc::new(Store::open(f.directory.path().join("store")).unwrap());
        let finalizer = Finalizer::new(recovered);
        let replay = finalizer.finalize(&f.request, &native).unwrap();
        assert_eq!(first.model_digest, replay.model_digest);
        assert_eq!(calls, native.calls.lock().unwrap().len());
        let mut changed = f.request.clone();
        changed.refit = Refit::Fit;
        assert!(finalizer.finalize(&changed, &native).is_err());
        assert_eq!(calls, native.calls.lock().unwrap().len());
    }

    #[test]
    fn failed_bounded_fit_preserves_audit_and_never_retries_or_reads_holdout_or_test() {
        let f = fixture(false);
        search(&f, &Native::default());
        let failing = Native {
            failure: Some(WorkerFailure::Timeout),
            ..Native::default()
        };
        fs::rename(
            &f.request.holdout.path,
            f.directory.path().join("holdout-unavailable"),
        )
        .unwrap();
        fs::rename(
            &f.request.test.as_ref().unwrap().path,
            f.directory.path().join("test-unavailable"),
        )
        .unwrap();
        let finalizer = Finalizer::new(Arc::clone(&f.store));
        assert!(finalizer.finalize(&f.request, &failing).is_err());
        let record = f.store.get(&claim_key(&f.plan.run_id)).unwrap().unwrap();
        assert_eq!(record.data["audit"]["entries"].as_array().unwrap().len(), 1);
        assert_eq!(
            record.data["audit"]["entries"][0]["phase"],
            "finalization_fit"
        );
        assert_eq!(
            finalizer
                .journal()
                .snapshot(&f.plan.run_id)
                .unwrap()
                .unwrap()
                .state,
            RunState::Failed
        );
        assert!(finalizer.finalize(&f.request, &Native::default()).is_err());
        assert!(!f.request.model_root.exists());
    }

    #[test]
    fn bad_rosters_refuse_before_claim_and_missing_values_refuse_without_format_drift() {
        let f = fixture(false);
        let native = Native::default();
        search(&f, &native);
        let finalizer = Finalizer::new(Arc::clone(&f.store));
        let mut bad = f.request.clone();
        bad.holdout.rows[0] = bad.fit.rows[0].clone();
        assert!(finalizer.finalize(&bad, &native).is_err());
        assert!(f.store.get(&claim_key(&f.plan.run_id)).unwrap().is_none());
        let f = fixture(true);
        let native = Native::default();
        search(&f, &native);
        let calls = native.calls.lock().unwrap().len();
        let error = Finalizer::new(Arc::clone(&f.store))
            .finalize(&f.request, &native)
            .unwrap_err();
        assert!(error.to_string().contains("complete finite numeric"));
        assert_eq!(calls, native.calls.lock().unwrap().len());
    }

    #[test]
    fn unspecified_test_is_absent_and_an_interrupted_claim_never_becomes_a_zero_score() {
        let mut f = fixture(false);
        f.request.test = None;
        let native = Native::default();
        search(&f, &native);
        let receipt = Finalizer::new(Arc::clone(&f.store))
            .finalize(&f.request, &native)
            .unwrap();
        assert_eq!(receipt.scores.test, None);
        assert_eq!(receipt.test_scope, None);
        let f = fixture(false);
        search(&f, &native);
        let run = f
            .store
            .get(&RecordKey::Run(f.plan.run_id.clone()))
            .unwrap()
            .unwrap();
        let identity = digest(
            &serde_json::to_vec(&(
                &f.request,
                &run.data["checkpoint"]["plan"],
                &run.data["checkpoint"]["champion"],
            ))
            .unwrap(),
        );
        f.store
            .transaction(|tx| {
                tx.put(
                    &claim_key(&f.plan.run_id),
                    &Record::metadata(json!({"identity":identity,"state":"started"}), vec![]),
                )
            })
            .unwrap();
        let before = native.calls.lock().unwrap().len();
        assert!(Finalizer::new(Arc::clone(&f.store))
            .finalize(&f.request, &native)
            .unwrap_err()
            .to_string()
            .contains("unknown"));
        assert_eq!(before, native.calls.lock().unwrap().len());
    }

    #[test]
    fn a_controller_change_before_holdout_prevents_further_reads_and_late_publication() {
        struct Pause {
            store: Arc<Store>,
            native: Native,
            count: AtomicUsize,
        }
        impl Executor for Pause {
            fn evaluate(&self, request: &Request) -> Response {
                let response = self.native.evaluate(request);
                if self.count.fetch_add(1, Ordering::SeqCst) == 0 {
                    let log = EventLog::new(Arc::clone(&self.store));
                    let mut snapshot = log.snapshot(&request.attempt.run).unwrap().unwrap();
                    let record = self
                        .store
                        .get(&RecordKey::Run(request.attempt.run.clone()))
                        .unwrap()
                        .unwrap();
                    let revision = snapshot.revision.0;
                    snapshot.revision.0 += 1;
                    snapshot.state = RunState::Paused;
                    log.commit(
                        RunUpdate {
                            expected_revision: Some(revision),
                            snapshot,
                            checkpoint: record.data["checkpoint"].clone(),
                            blobs: record.blobs,
                        },
                        vec![],
                        |_| Ok(()),
                    )
                    .unwrap();
                }
                response
            }
            fn fitted_model(&self, _: &Request) -> Result<Vec<u8>> {
                bail!("unused")
            }
        }
        let f = fixture(false);
        search(&f, &Native::default());
        let executor = Pause {
            store: Arc::clone(&f.store),
            native: Native::default(),
            count: AtomicUsize::new(0),
        };
        let finalizer = Finalizer::new(Arc::clone(&f.store));
        assert!(finalizer
            .finalize(&f.request, &executor)
            .unwrap_err()
            .to_string()
            .contains("controller changed"));
        assert_eq!(
            finalizer
                .journal()
                .snapshot(&f.plan.run_id)
                .unwrap()
                .unwrap()
                .state,
            RunState::Paused
        );
        assert!(!f.request.model_root.exists());
        let claim = f.store.get(&claim_key(&f.plan.run_id)).unwrap().unwrap();
        assert_eq!(claim.data["audit"]["entries"].as_array().unwrap().len(), 1);
    }

    #[test]
    #[ignore = "operational gate: build actual jopedime, set JOPE_DSC09_APPLIANCE_EXE and execute explicitly"]
    fn real_appliance_workers_finalize_and_jopedime_predict_loads_the_model_object() {
        let executable = PathBuf::from(
            std::env::var_os("JOPE_DSC09_APPLIANCE_EXE")
                .expect("actual built appliance executable required"),
        );
        let f = fixture(false);
        let executor =
            super::super::r#loop::ProcessExecutor::new(&executable, 1, &f.request.scratch_root)
                .unwrap();
        search(&f, &executor);
        let receipt = Finalizer::new(Arc::clone(&f.store))
            .finalize(&f.request, &executor)
            .unwrap();
        let part = f.request.test.as_ref().unwrap();
        let table = load(part, &f.plan, f.request.max_input_bytes).unwrap();
        let csv_path = f.directory.path().join("predict.csv");
        let mut csv = table.headers.join(",") + "\n";
        for row in &table.rows {
            csv.push_str(
                &row.iter()
                    .map(|v| v.as_deref().unwrap())
                    .collect::<Vec<_>>()
                    .join(","),
            );
            csv.push('\n');
        }
        fs::write(&csv_path, csv).unwrap();
        let output_path = f.directory.path().join("predictions.csv");
        let result = std::process::Command::new(&executable)
            .args(["predict", "--model"])
            .arg(&receipt.model_path)
            .args(["--input"])
            .arg(&csv_path)
            .args(["--output"])
            .arg(&output_path)
            .args(["--threads", "1", "--device", "cpu", "--json"])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let cli_receipt: Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(cli_receipt["outcome"], "succeeded");
        assert!(output_path.is_file());
        let object = ModelObject::read(&receipt.model_path, HANDOFF_MAX).unwrap();
        let expected = predict(&object, &table).unwrap();
        let actual =
            jope_runtime::table::read_csv(&output_path, &jope_runtime::limits::Limits::default())
                .unwrap();
        let observed: Vec<f64> = actual
            .rows
            .iter()
            .map(|r| r[0].as_deref().unwrap().parse().unwrap())
            .collect();
        assert_eq!(expected, observed);
        if let Some(root) = std::env::var_os("JOPE_DSC09_RECEIPT_DIR") {
            let root = PathBuf::from(root);
            fs::create_dir_all(&root).unwrap();
            fs::write(
                root.join("finalization.json"),
                serde_json::to_vec_pretty(&receipt).unwrap(),
            )
            .unwrap();
            fs::write(root.join("predict-receipt.json"), &result.stdout).unwrap();
            fs::copy(&output_path, root.join("predictions.csv")).unwrap();
            fs::copy(&csv_path, root.join("predict-input.csv")).unwrap();
            // Preserve executable model parts and the exact TRAIN-derived input
            // mappings before fixture cleanup; receipts alone are not replayable.
            object.save(&root.join("models"), HANDOFF_MAX).unwrap();
            for part in [&f.request.fit, &f.request.holdout]
                .into_iter()
                .chain(f.request.test.as_ref())
            {
                fs::copy(&part.path, root.join(part.path.file_name().unwrap())).unwrap();
            }
            fs::write(
                root.join("source.json"),
                serde_json::to_vec_pretty(&json!({"source_commit":f.plan.binary_version,"source_train_sha256":TRAIN_SHA,"source_train_path":"test-support/datasets/v1/d2-regression/train.csv","external_holdout_or_test_reads":0,"plan":f.plan})).unwrap(),
            ).unwrap();
            fs::write(
                root.join("input-identities.json"),
                serde_json::to_vec_pretty(&f.request).unwrap(),
            )
            .unwrap();
        }
    }
}
