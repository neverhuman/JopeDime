//! Exact fitted winner export preparation and read-only report attachment.
//! This module never fits a recipe or changes a screening/confirmation rank.
use super::{
    campaign::{self, CampaignPlan},
    inference::{
        self, Backend, ExportOutcome, FittedInferenceView, PublicBundle, QualificationPlan,
    },
    protocol::{ConfirmedRecipe, Observation},
};
use anyhow::{ensure, Context, Result};
use clap::Args;
use jope_pipeline::{
    execution::{Artifact, WorkerJob, WorkerOutcome},
    experiment::{EvaluationStage, EvaluationStatus},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

pub const ORDER: [Backend; 5] = [
    Backend::Python,
    Backend::Rust,
    Backend::Cpp17,
    Backend::Cpp20,
    Backend::JavaScript,
];
pub fn name(backend: Backend) -> &'static str {
    match backend {
        Backend::Python => "python",
        Backend::Rust => "rust",
        Backend::Cpp17 => "cpp17",
        Backend::Cpp20 => "cpp20",
        Backend::JavaScript => "javascript",
    }
}
#[derive(Args)]
pub struct ExportArgs {
    #[command(flatten)]
    pub campaign: super::campaign_cli::ReportArgs,
    #[arg(long)]
    pub dataset: String,
    /// Export this existing fitted fold; no whole-data or replacement fit is made.
    #[arg(long, default_value_t = 0)]
    pub fold: u8,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum Entry {
    Unsupported {
        backend: Backend,
        reason: Value,
    },
    Prepared {
        backend: Backend,
        bundle: Box<PublicBundle>,
        qualification: Box<QualificationPlan>,
    },
}
impl Entry {
    pub fn backend(&self) -> Backend {
        match self {
            Self::Unsupported { backend, .. } | Self::Prepared { backend, .. } => *backend,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedExport {
    pub schema: String,
    pub campaign_plan: Artifact,
    pub source_sha40: String,
    pub program: Artifact,
    pub dataset: String,
    pub observation: Observation,
    pub original_job: WorkerJob,
    pub original_outcome: Artifact,
    pub original_outputs: Vec<Artifact>,
    pub entries: Vec<Entry>,
    pub preparation_wall_seconds: f64,
    pub preparation_cpu_seconds: f64,
}
impl PreparedExport {
    pub fn identity(&self) -> Result<String> {
        campaign::digest(self)
    }
    pub fn verify(&self) -> Result<()> {
        ensure!(
            self.schema == "jope.fitted-winner-export/v1",
            "export schema changed"
        );
        ensure!(
            [self.preparation_wall_seconds, self.preparation_cpu_seconds]
                .iter()
                .all(|v| v.is_finite() && *v >= 0.),
            "invalid export preparation cost"
        );
        self.campaign_plan.verify()?;
        self.program.verify()?;
        self.original_outcome.verify()?;
        let campaign: CampaignPlan = serde_json::from_slice(&fs::read(&self.campaign_plan.path)?)?;
        ensure!(
            campaign.experiment.source_sha == self.source_sha40
                && campaign.experiment.identity()? == self.observation.request.experiment_sha256
                && serde_json::to_value(&campaign.runtimes["rust"].program)?
                    == serde_json::to_value(&self.program)?,
            "export source/campaign changed"
        );
        campaign.validate()?;
        self.observation.validate()?;
        ensure!(
            self.observation.result.status == EvaluationStatus::Completed,
            "no completed fitted predictor"
        );
        let original = self
            .observation
            .result
            .reused_from_request_sha256
            .as_ref()
            .cloned()
            .unwrap_or(self.observation.request.identity()?);
        ensure!(
            self.original_job.request.identity()? == original,
            "original fit request changed"
        );
        let outcome: WorkerOutcome =
            serde_json::from_slice(&fs::read(&self.original_outcome.path)?)?;
        ensure!(
            outcome.job_sha256 == self.original_job.identity()?
                && outcome.request_sha256 == original
                && outcome.command_status == EvaluationStatus::Completed
                && outcome.exit_code == Some(0)
                && outcome.signal.is_none()
                && campaign::digest(&outcome)? == self.observation.result.worker_receipt_sha256
                && serde_json::to_value(&outcome.outputs)?
                    == serde_json::to_value(&self.original_outputs)?,
            "original fitted receipt changed"
        );
        outcome.verify_outputs()?;
        let journal = self
            .original_outcome
            .path
            .parent()
            .and_then(Path::parent)
            .context("original journal absent")?;
        super::receipt_timing::completed_fit(journal, &self.original_job, &outcome)?;
        let output_path = self.original_job.directory.join("output.json");
        let output_artifact = artifact(&output_path)?;
        ensure!(
            self.original_outputs
                .iter()
                .any(|a| a.path == output_artifact.path && a.sha256 == output_artifact.sha256),
            "native fitted output not authenticated"
        );
        let output: Value = serde_json::from_slice(&fs::read(&output_path)?)?;
        ensure!(
            self.observation.result.predictions_sha256.as_ref()
                == Some(&campaign::digest(&serde_json::from_value::<
                    super::score::Predictions,
                >(
                    output["predictions"].clone()
                )?)?),
            "original fitted predictions changed"
        );
        let fitted = self
            .observation
            .result
            .fitted_state_sha256
            .as_ref()
            .context("original fitted checksum absent")?;
        if output["contract"] == super::methods::ADAPTER_CONTRACT {
            ensure!(
                output["fitted"]["path"] == "output.json.fitted.pkl"
                    && output["fitted"]["format"] == "jope.panel-fitted-python/v1"
                    && output["fitted"]["preprocessing_policy"] == super::methods::INPUT_POLICY
                    && output["fitted"]["sha256"] == *fitted
                    && output["fitted"]["reload_verified"] == false
                    && self.original_outputs.iter().any(|a| a.path
                        == self.original_job.directory.join("output.json.fitted.pkl")
                        && a.sha256 == *fitted),
                "external complete fitted artifact changed"
            );
            ensure!(
                self.entries
                    .iter()
                    .all(|e| matches!(e, Entry::Unsupported { .. })),
                "external winner cannot be replaced by standalone recipe"
            );
        } else {
            ensure!(
                output["artifact_sha256"] == *fitted,
                "native fitted artifact identity changed"
            );
        }
        for item in &self.original_outputs {
            item.verify()?;
        }
        verify_entries(
            &self.entries,
            &output,
            fitted,
            &location(journal, &self.observation)?.join("public"),
        )
    }
    /// Reconstruct executable plans, including native reference predictions, only before
    /// launching the explicit qualification command. The report path re-emits source text
    /// to bind it to the fitted state but never predicts or compiles.
    pub fn verify_executable_plans(&self) -> Result<()> {
        let output: Value =
            serde_json::from_slice(&fs::read(self.original_job.directory.join("output.json"))?)?;
        let rows = self.entries.iter().find_map(|entry| match entry {
            Entry::Prepared { qualification, .. } => {
                qualification.files.get("qualification-input.json")
            }
            _ => None,
        });
        let Some(rows) = rows else {
            return Ok(());
        };
        let rows: Vec<Vec<Option<String>>> = serde_json::from_str(rows)?;
        let fitted = FittedOutput::parse(&output)?;
        let Some(view) = fitted.view()? else {
            anyhow::bail!("external state cannot carry executable standalone plans")
        };
        let entries = prepare_entries(&view, &rows)?;
        ensure!(
            serde_json::to_value(entries)? == serde_json::to_value(&self.entries)?,
            "executable export differs from exact fitted state"
        );
        Ok(())
    }
}
/// The caller has already authenticated the original panel observations. This
/// attaches only the selected existing predictor; it never emits or compiles.
pub fn report(
    journal: &Path,
    recipes: &[ConfirmedRecipe],
    observations: &[Observation],
) -> Result<Value> {
    let Some(winner) = winner(recipes)? else {
        return Ok(
            json!({"winner":null,"status":"unavailable_no_complete_five_fold_recipe","ranking_changed":false}),
        );
    };
    let mut folds = Vec::new();
    for fold in 0..5 {
        let observation = observations
            .iter()
            .find(|o| {
                o.method == winner.method
                    && o.request.fold == fold
                    && o.request.stage == EvaluationStage::Confirm
            })
            .context("winner confirmation fold absent")?;
        let directory = location(journal, observation)?;
        let path = directory.join("prepared.json");
        let backends = if path.is_file() {
            let prepared: PreparedExport = serde_json::from_slice(&fs::read(path)?)?;
            ensure!(
                serde_json::to_value(&prepared.observation)? == serde_json::to_value(observation)?,
                "stale winner export evidence"
            );
            prepared.verify()?;
            super::fitted_export_run::recover(&directory, &prepared)?
        } else {
            json!({"backends":ORDER.iter().map(|b|json!({"backend":b,"status":if winner.method.starts_with("external."){"unsupported"}else{"unattempted"},"reason":if winner.method.starts_with("external."){Some("external fitted pickle retained; reload is not verified, so standalone lowering is unavailable")}else{None}})).collect::<Vec<_>>(),"execution_cost":null})
        };
        folds.push(json!({"fold":fold,"request_sha256":observation.request.identity()?,"fitted_sha256":observation.result.fitted_state_sha256,"native_fitted_result_retained":true,"inference":backends}));
    }
    Ok(
        json!({"winner":winner.method,"selection":"complete five-fold mean loss, tie by method identity","mean_loss":winner.mean_loss,"ranking_changed":false,"folds":folds,"qualification_order":ORDER}),
    )
}

mod prepare;
#[cfg(test)]
mod tests;
mod validation;

pub use prepare::prepare;
use prepare::{materialize, prepare_entries, winner};
use validation::verify_entries;
pub use validation::{artifact, location};
