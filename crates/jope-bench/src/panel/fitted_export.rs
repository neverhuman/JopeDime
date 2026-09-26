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
                    && output["fitted"]["reload_verified"] == true
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
pub fn artifact(path: &Path) -> Result<Artifact> {
    let path = fs::canonicalize(path)?;
    let value = Artifact {
        sha256: super::methods::digest(&fs::read(&path)?),
        path,
    };
    value.verify()?;
    Ok(value)
}
pub fn location(journal: &Path, observation: &Observation) -> Result<PathBuf> {
    Ok(journal
        .join("fitted-inference")
        .join(observation.request.identity()?))
}
/// The original worker output, deserialized once; standalone views borrow from it.
enum FittedOutput {
    Registered(Box<super::worker::RustWorkerOutput>),
    Synthesized(Box<super::synthesized::SynthesizedWorkerOutput>),
    External,
}
impl FittedOutput {
    fn parse(output: &Value) -> Result<Self> {
        Ok(if output["schema"] == "jope.panel-rust-worker-output/v1" {
            Self::Registered(Box::new(serde_json::from_value(output.clone())?))
        } else if output["schema"] == "jope.panel-synthesized-worker-output/v1" {
            Self::Synthesized(Box::new(serde_json::from_value(output.clone())?))
        } else {
            Self::External
        })
    }
    fn view(&self) -> Result<Option<FittedInferenceView<'_>>> {
        Ok(match self {
            Self::Registered(value) => Some(FittedInferenceView::registered(&value.artifact)?),
            Self::Synthesized(value) => Some(FittedInferenceView::synthesized(&value.artifact)?),
            Self::External => None,
        })
    }
}
/// Bind every backend entry to the exact authenticated fitted output and to the
/// public files materialized under `public`. Public source is re-emitted from the
/// fitted state and compared by identity, and each executed plan must run exactly the
/// shipped files; nothing is compiled, predicted or fitted here. An unsupported entry
/// must carry the reason the fitted state itself yields, so support is neither hidden
/// nor invented, and a same-recipe state swap cannot survive with rebound checksums.
fn verify_entries(
    entries: &[Entry],
    output: &Value,
    fitted_sha256: &str,
    public: &Path,
) -> Result<()> {
    ensure!(entries.len() == ORDER.len(), "backend inventory incomplete");
    let fitted = FittedOutput::parse(output)?;
    let view = fitted.view()?;
    if let Some(view) = &view {
        ensure!(
            view.fitted_sha256() == fitted_sha256,
            "fitted state differs from the selected fitted checksum"
        );
    }
    for (entry, backend) in entries.iter().zip(ORDER) {
        ensure!(
            entry.backend() == backend,
            "backend qualification order changed"
        );
        let derived = match &view {
            Some(view) => Some(inference::export_inference(view, backend)?),
            None => None,
        };
        if let (Entry::Unsupported { reason, .. }, Some(derived)) = (entry, &derived) {
            let ExportOutcome::Unsupported { reason: actual, .. } = derived else {
                anyhow::bail!("fitted state supports a backend recorded as unsupported")
            };
            ensure!(
                serde_json::to_value(actual)? == *reason,
                "unsupported reason differs from the fitted state"
            );
        }
        if let Entry::Prepared {
            bundle,
            qualification,
            ..
        } = entry
        {
            bundle.validate()?;
            let Some(ExportOutcome::Emitted { bundle: rebuilt }) = derived else {
                anyhow::bail!("prepared backend is not emitted from the exact fitted state")
            };
            ensure!(
                rebuilt.identity()? == bundle.identity()?,
                "public bundle differs from the exact fitted state"
            );
            inference::verify_plan_binding(qualification, bundle)?;
            let public = public.join(name(backend));
            for (file, bytes) in &bundle.files {
                ensure!(
                    fs::read(public.join(file))? == bytes.as_bytes(),
                    "materialized public bundle changed"
                );
            }
            let manifest: Value = serde_json::from_slice(&fs::read(public.join("manifest.json"))?)?;
            ensure!(
                manifest == serde_json::to_value(&bundle.manifest)?,
                "materialized public manifest changed"
            );
            ensure!(
                serde_json::to_value(&qualification.expected)? == output["predictions"],
                "qualification reference differs from original fitted predictions"
            );
            ensure!(
                qualification.backend == backend
                    && qualification.bundle_sha256 == bundle.identity()?
                    && qualification.fitted_sha256 == fitted_sha256
                    && qualification.parity_policy == backend.parity_policy(),
                "exact fitted export identity changed"
            );
        }
    }
    Ok(())
}
/// Write each prepared public bundle once (0600, create-new); an existing file must match.
fn materialize(public: &Path, entries: &[Entry]) -> Result<()> {
    for entry in entries {
        if let Entry::Prepared {
            backend, bundle, ..
        } = entry
        {
            bundle.validate()?;
            let public = public.join(name(*backend));
            fs::create_dir_all(&public)?;
            for (name, bytes) in &bundle.files {
                use std::io::Write;
                use std::os::unix::fs::OpenOptionsExt;
                let path = public.join(name);
                if path.try_exists()? {
                    ensure!(
                        fs::read(&path)? == bytes.as_bytes(),
                        "existing public model differs"
                    );
                } else {
                    let mut file = fs::OpenOptions::new()
                        .create_new(true)
                        .write(true)
                        .mode(0o600)
                        .open(path)?;
                    file.write_all(bytes.as_bytes())?;
                    file.sync_all()?;
                }
            }
            campaign::immutable(&public.join("manifest.json"), &bundle.manifest)?;
        }
    }
    Ok(())
}
fn winner(recipes: &[ConfirmedRecipe]) -> Result<Option<&ConfirmedRecipe>> {
    for recipe in recipes {
        ensure!(
            recipe
                .mean_loss
                .is_none_or(|loss| loss.is_finite() && recipe.completed_folds == 5),
            "invalid five-fold loss"
        );
    }
    Ok(recipes
        .iter()
        .filter(|r| r.mean_loss.is_some())
        .min_by(|a, b| {
            a.mean_loss
                .unwrap()
                .total_cmp(&b.mean_loss.unwrap())
                .then(a.method.cmp(&b.method))
        }))
}
fn prepare_entries(
    view: &FittedInferenceView<'_>,
    rows: &[Vec<Option<String>>],
) -> Result<Vec<Entry>> {
    ORDER
        .into_iter()
        .map(
            |backend| match inference::export_inference(view, backend)? {
                ExportOutcome::Unsupported { reason, .. } => Ok(Entry::Unsupported {
                    backend,
                    reason: serde_json::to_value(reason)?,
                }),
                ExportOutcome::Emitted { bundle } => {
                    let qualification =
                        Box::new(inference::qualification_plan_for_view(view, &bundle, rows)?);
                    Ok(Entry::Prepared {
                        backend,
                        bundle,
                        qualification,
                    })
                }
            },
        )
        .collect()
}
pub fn prepare(args: ExportArgs) -> Result<Value> {
    let preparation_start = std::time::Instant::now();
    let preparation_cpu_start = campaign::cpu_seconds()?;
    ensure!(args.fold < 5, "fold must be0..4");
    let campaign_plan = artifact(&args.campaign.plan)?;
    let campaign: CampaignPlan = serde_json::from_slice(&fs::read(&campaign_plan.path)?)?;
    ensure!(
        campaign.identity()? == args.campaign.plan_sha256,
        "campaign identity changed"
    );
    let journal = fs::canonicalize(&args.campaign.journal)?;
    let report = campaign::report(
        &campaign,
        &args.campaign.manifest,
        &args.campaign.datasets,
        &journal,
    )?;
    let dataset_report = report["datasets"]
        .as_array()
        .context("dataset report absent")?
        .iter()
        .find(|r| r["dataset"] == args.dataset)
        .context("dataset absent")?;
    let recipes: Vec<ConfirmedRecipe> = serde_json::from_value(dataset_report["recipes"].clone())
        .context("five-fold confirmation unavailable")?;
    let winner =
        winner(&recipes)?.context("no complete five-fold winner; no replacement fitted")?;
    let observations: Vec<Observation> =
        serde_json::from_value(dataset_report["confirmation"].clone())?;
    let observation = observations
        .into_iter()
        .find(|o| o.method == winner.method && o.request.fold == args.fold)
        .context("winner fitted fold absent")?;
    ensure!(
        observation.result.status == EvaluationStatus::Completed,
        "winner fit incomplete"
    );
    let existing = location(&journal, &observation)?.join("prepared.json");
    if existing.try_exists()? {
        let previous: PreparedExport = serde_json::from_slice(&fs::read(&existing)?)?;
        ensure!(
            serde_json::to_value(&previous.observation)? == serde_json::to_value(&observation)?,
            "retained export changed selected predictor"
        );
        previous.verify()?;
        return Ok(
            json!({"ok":true,"prepared":existing,"prepared_sha256":previous.identity()?,"reused":true,"fits":0,"compiler_commands":0,"ranking_changed":false,"preparation_cpu_seconds_this_invocation":campaign::cpu_seconds()? - preparation_cpu_start,"preparation_wall_seconds_this_invocation":preparation_start.elapsed().as_secs_f64()}),
        );
    }
    let original = observation
        .result
        .reused_from_request_sha256
        .as_ref()
        .cloned()
        .unwrap_or(observation.request.identity()?);
    let mut jobs = BTreeMap::new();
    for filename in [
        "screen-jobs.json",
        "asm-screen-jobs.json",
        "confirm-jobs.json",
    ] {
        let path = journal.join(filename);
        if !path.is_file() {
            continue;
        }
        for job in serde_json::from_slice::<Vec<WorkerJob>>(&fs::read(path)?)? {
            if job.request.identity()? == original {
                jobs.insert(job.identity()?, job);
            }
        }
    }
    ensure!(
        jobs.len() == 1,
        "missing or ambiguous original fitted worker"
    );
    let (_, original_job) = jobs.into_iter().next().unwrap();
    let original_outcome = artifact(
        &journal
            .join("finished")
            .join(format!("{}.json", original_job.identity()?)),
    )?;
    let outcome: WorkerOutcome = serde_json::from_slice(&fs::read(&original_outcome.path)?)?;
    outcome.verify_outputs()?;
    let output: Value =
        serde_json::from_slice(&fs::read(original_job.directory.join("output.json"))?)?;
    let spec = campaign
        .panel
        .datasets
        .iter()
        .find(|d| d.id == args.dataset)
        .context("dataset specification absent")?;
    let data = super::data::load_development(&args.campaign.datasets, spec)?;
    let cut = data.cut(args.fold)?;
    ensure!(
        cut.hold.rows_sha256 == observation.request.scoring_rows_sha256,
        "export prediction rows changed"
    );
    let rows: Vec<_> = cut.hold.rows.iter().map(|r| r.values.clone()).collect();
    let fitted_sha = observation
        .result
        .fitted_state_sha256
        .clone()
        .context("fitted hash absent")?;
    let entries = if output["schema"] == "jope.panel-rust-worker-output/v1" {
        let output: super::worker::RustWorkerOutput = serde_json::from_value(output)?;
        ensure!(
            output.artifact_sha256 == fitted_sha
                && super::methods::digest(&output.artifact.bytes()?) == fitted_sha,
            "native fitted artifact changed"
        );
        prepare_entries(&FittedInferenceView::registered(&output.artifact)?, &rows)?
    } else if output["schema"] == "jope.panel-synthesized-worker-output/v1" {
        let output: super::synthesized::SynthesizedWorkerOutput = serde_json::from_value(output)?;
        ensure!(
            output.artifact_sha256 == fitted_sha
                && super::methods::digest(&output.artifact.bytes()?) == fitted_sha,
            "generated fitted artifact changed"
        );
        prepare_entries(&FittedInferenceView::synthesized(&output.artifact)?, &rows)?
    } else {
        ensure!(
            output["contract"] == super::methods::ADAPTER_CONTRACT
                && output["fitted"]["sha256"] == fitted_sha
                && output["fitted"]["reload_verified"] == true,
            "external native fitted receipt changed"
        );
        ORDER.into_iter().map(|backend|Entry::Unsupported{backend,reason:json!({"kind":"external_library_state","detail":"complete native fitted pickle and reload remain available; standalone lowering is not implemented"})}).collect()
    };
    let mut prepared = PreparedExport {
        schema: "jope.fitted-winner-export/v1".into(),
        campaign_plan,
        source_sha40: campaign.experiment.source_sha.clone(),
        program: campaign.runtimes["rust"].program.clone(),
        dataset: args.dataset,
        observation,
        original_job,
        original_outcome,
        original_outputs: outcome.outputs,
        entries,
        preparation_wall_seconds: 0.,
        preparation_cpu_seconds: 0.,
    };
    let directory = location(&journal, &prepared.observation)?;
    fs::create_dir_all(&directory)?;
    materialize(&directory.join("public"), &prepared.entries)?;
    prepared.verify()?;
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(&directory, fs::Permissions::from_mode(0o700))?;
    prepared.preparation_wall_seconds = preparation_start.elapsed().as_secs_f64();
    prepared.preparation_cpu_seconds = campaign::cpu_seconds()? - preparation_cpu_start;
    campaign::immutable(&directory.join("prepared.json"), &prepared)?;
    Ok(
        json!({"ok":true,"prepared":directory.join("prepared.json"),"prepared_sha256":prepared.identity()?,"dataset":prepared.dataset,"method":winner.method,"fold":args.fold,"native_fitted_sha256":fitted_sha,"public_bundle_root":directory.join("public"),"fits":0,"compiler_commands":0,"ranking_changed":false,"backends":prepared.entries.iter().map(|e|json!({"backend":e.backend(),"status":if matches!(e,Entry::Unsupported{..}){"unsupported"}else{"unattempted"}})).collect::<Vec<_>>()}),
    )
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
            json!({"backends":ORDER.iter().map(|b|json!({"backend":b,"status":if winner.method.starts_with("external."){"unsupported"}else{"unattempted"},"reason":if winner.method.starts_with("external."){Some("external native pickle retained; standalone lowering not implemented")}else{None}})).collect::<Vec<_>>(),"execution_cost":null})
        };
        folds.push(json!({"fold":fold,"request_sha256":observation.request.identity()?,"fitted_sha256":observation.result.fitted_state_sha256,"native_fitted_result_retained":true,"inference":backends}));
    }
    Ok(
        json!({"winner":winner.method,"selection":"complete five-fold mean loss, tie by method identity","mean_loss":winner.mean_loss,"ranking_changed":false,"folds":folds,"qualification_order":ORDER}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    fn recipe(method: &str, loss: Option<f64>) -> ConfirmedRecipe {
        ConfirmedRecipe {
            method: method.into(),
            diagnostic: false,
            completed_folds: if loss.is_some() { 5 } else { 0 },
            mean_loss: loss,
            fold_losses: [loss; 5],
        }
    }
    #[test]
    fn fitted_winner_uses_losses_and_identity_never_backend_support() {
        let recipes = vec![
            recipe("external.catboost.default", Some(0.2)),
            recipe("rust.ridge", Some(0.3)),
            recipe("asm.neural", None),
        ];
        assert_eq!(
            winner(&recipes).unwrap().unwrap().method,
            "external.catboost.default"
        );
        assert_eq!(
            winner(&[recipe("z", Some(1.0)), recipe("a", Some(1.0))])
                .unwrap()
                .unwrap()
                .method,
            "a"
        );
        assert!(winner(&[recipe("unavailable", None)]).unwrap().is_none());
    }
    #[test]
    fn fitted_winner_refuses_incomplete_or_nonfinite_means() {
        let mut bad = recipe("x", Some(0.0));
        bad.completed_folds = 4;
        assert!(winner(&[bad]).is_err());
        assert!(winner(&[recipe("x", Some(f64::NAN))]).is_err());
    }

    use super::super::inference::tests::fixture;
    use super::super::methods::{digest, FittedRustArtifact};
    use jope_datasets::catalog::Task;
    use jope_pipeline::experiment::{Column, ColumnKind};

    fn head_state(value: &mut Value) -> Option<&mut serde_json::Map<String, Value>> {
        match value {
            Value::Object(map) => {
                if map.contains_key("means") && map.contains_key("w") {
                    return Some(map);
                }
                map.values_mut().find_map(head_state)
            }
            Value::Array(items) => items.iter_mut().find_map(head_state),
            _ => None,
        }
    }
    /// Two fitted states of one registered recipe. B differs from A only in the learned
    /// weight of the numeric column, and every row sets that column to its fitted mean,
    /// so both states predict bit-identical values on these convenient rows.
    fn same_recipe_pair() -> (
        FittedRustArtifact,
        FittedRustArtifact,
        Vec<Vec<Option<String>>>,
    ) {
        let (a, _) = fixture(Task::Regression, "ridge");
        let mut value = serde_json::to_value(&a).unwrap();
        let head = head_state(&mut value).expect("ridge head state");
        let mean = head["means"][0].as_f64().unwrap();
        let weights = head.get_mut("w").unwrap().as_array_mut().unwrap();
        let weight = weights[1].as_f64().unwrap();
        weights[1] = json!(weight * 2.0 + 1.0);
        let b: FittedRustArtifact = serde_json::from_value(value).unwrap();
        b.validate().unwrap();
        assert_eq!(a.method, b.method, "one recipe");
        let amount = format!("{mean:?}");
        let rows = vec![
            vec![Some(amount.clone()), Some("NA".into())],
            vec![Some(amount.clone()), None],
            vec![Some(amount), Some("unseen".into())],
        ];
        (a, b, rows)
    }
    fn raw_columns() -> Vec<Column> {
        vec![
            Column {
                name: "amount".into(),
                kind: ColumnKind::Numeric,
            },
            Column {
                name: "country".into(),
                kind: ColumnKind::Categorical,
            },
        ]
    }
    /// The authenticated original worker output that `verify` reads from output.json.
    fn worker_output(
        artifact: &FittedRustArtifact,
        rows: &[Vec<Option<String>>],
    ) -> (Value, String) {
        let sha = digest(&artifact.bytes().unwrap());
        let predictions = artifact.predict_raw(&raw_columns(), rows).unwrap();
        (
            json!({"schema":"jope.panel-rust-worker-output/v1","input_sha256":"0".repeat(64),
                "artifact_sha256":sha,"predictions":predictions,"artifact":artifact}),
            sha,
        )
    }
    /// Unvalidated write, so a tampered entry reaches `verify_entries` itself.
    fn write_public(public: &Path, entries: &[Entry]) {
        for entry in entries {
            if let Entry::Prepared {
                backend, bundle, ..
            } = entry
            {
                let directory = public.join(name(*backend));
                fs::create_dir_all(&directory).unwrap();
                for (file, text) in &bundle.files {
                    fs::write(directory.join(file), text).unwrap();
                }
                fs::write(
                    directory.join("manifest.json"),
                    serde_json::to_vec(&bundle.manifest).unwrap(),
                )
                .unwrap();
            }
        }
    }
    /// Re-seal a prepared entry around substituted public files, keeping checksums,
    /// bundle identity and executed plan files internally consistent. The only
    /// remaining inconsistency is with the exact fitted state.
    fn reseal(entry: &Entry, files: BTreeMap<String, String>) -> Entry {
        let Entry::Prepared {
            backend,
            bundle,
            qualification,
        } = entry
        else {
            panic!("prepared entry")
        };
        let mut bundle = bundle.clone();
        let mut qualification = qualification.clone();
        for name in bundle.files.keys() {
            qualification.files.remove(name);
        }
        qualification.files.extend(files.clone());
        bundle.manifest.files = files
            .iter()
            .map(|(name, text)| (name.clone(), digest(text.as_bytes())))
            .collect();
        bundle.files = files;
        qualification.bundle_sha256 = bundle.identity().unwrap();
        Entry::Prepared {
            backend: *backend,
            bundle,
            qualification,
        }
    }
    fn prepared(entry: &Entry) -> (&PublicBundle, &QualificationPlan) {
        let Entry::Prepared {
            bundle,
            qualification,
            ..
        } = entry
        else {
            panic!("prepared entry")
        };
        (bundle, qualification)
    }

    #[test]
    fn same_recipe_state_swaps_and_missing_stages_fail_even_with_equal_predictions() {
        let (a, b, rows) = same_recipe_pair();
        let (a_bytes, b_bytes) = (a.bytes().unwrap(), b.bytes().unwrap());
        let entries_a =
            prepare_entries(&FittedInferenceView::registered(&a).unwrap(), &rows).unwrap();
        let entries_b =
            prepare_entries(&FittedInferenceView::registered(&b).unwrap(), &rows).unwrap();
        let (output_a, sha_a) = worker_output(&a, &rows);
        let (output_b, sha_b) = worker_output(&b, &rows);
        // Convenient rows: identical native predictions, distinct fitted states and bundles.
        assert_eq!(output_a["predictions"], output_b["predictions"]);
        assert_ne!(sha_a, sha_b);
        let mean: f64 = rows[0][0].as_ref().unwrap().parse().unwrap();
        let off = vec![vec![Some(format!("{:?}", mean + 2.0)), None]];
        let (far, _) = worker_output(&a, &off);
        let (far_b, _) = worker_output(&b, &off);
        assert_ne!(
            far["predictions"], far_b["predictions"],
            "states differ elsewhere"
        );
        for (ea, eb) in entries_a.iter().zip(&entries_b) {
            let ((bundle_a, plan_a), (bundle_b, plan_b)) = (prepared(ea), prepared(eb));
            assert_ne!(bundle_a.identity().unwrap(), bundle_b.identity().unwrap());
            assert_eq!(
                serde_json::to_value(&plan_a.expected).unwrap(),
                serde_json::to_value(&plan_b.expected).unwrap()
            );
            assert_ne!(plan_a.fitted_sha256, plan_b.fitted_sha256);
        }

        // Parity alone cannot tell the states apart: B's emitted Python passes A's plan.
        let scratch =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/fitted-inference-tests");
        fs::create_dir_all(&scratch).unwrap();
        let root = tempfile::tempdir_in(scratch).unwrap();
        let (_, plan_a) = prepared(&entries_a[0]);
        let (bundle_b, _) = prepared(&entries_b[0]);
        let run = root.path().join("parity-blind");
        fs::create_dir(&run).unwrap();
        for (file, text) in plan_a.files.iter().chain(&bundle_b.files) {
            fs::write(run.join(file), text).unwrap();
        }
        let output = std::process::Command::new("python3")
            .arg("qualification.py")
            .current_dir(&run)
            .output()
            .unwrap();
        assert!(output.status.success());
        inference::verify_qualification(plan_a, &plan_a.identity().unwrap(), &output.stdout)
            .expect("equal predictions pass parity");

        // Positive control through the production materializer.
        let public = root.path().join("intact").join("public");
        materialize(&public, &entries_a).unwrap();
        verify_entries(&entries_a, &output_a, &sha_a, &public).unwrap();

        let python = 0;
        let rust = 1;
        let cpp17 = 2;
        let (bundle_a, _) = prepared(&entries_a[python]);
        let mut attempts: Vec<(&str, Vec<Entry>)> = Vec::new();
        // Same-recipe state swap: B's public code, every identity rebound except the fitted one.
        let mut swap = entries_a.clone();
        swap[python] = reseal(
            &entries_a[python],
            prepared(&entries_b[python]).0.files.clone(),
        );
        attempts.push(("state-swap", swap));
        // The executed plan runs B's code while the shipped bundle stays A's.
        let mut executed = entries_a.clone();
        if let Entry::Prepared { qualification, .. } = &mut executed[python] {
            qualification.files.insert(
                "model.py".into(),
                prepared(&entries_b[python]).0.files["model.py"].clone(),
            );
        }
        attempts.push(("executed-file-swap", executed));
        // Whole receipt swap: B's complete entry carries B's fitted checksum.
        let mut receipt = entries_a.clone();
        receipt[python] = entries_b[python].clone();
        attempts.push(("receipt-swap", receipt));
        // Missing qualification stage: the harness file is absent from the plan.
        let mut stage = entries_a.clone();
        if let Entry::Prepared { qualification, .. } = &mut stage[python] {
            qualification.files.remove("qualification.py");
        }
        attempts.push(("missing-harness-stage", stage));
        // Missing wrapper: the C++ typed-table host boundary is dropped and resealed.
        let mut wrapper = entries_a.clone();
        let mut files = prepared(&entries_a[cpp17]).0.files.clone();
        files.remove("boundary.py");
        wrapper[cpp17] = reseal(&entries_a[cpp17], files);
        attempts.push(("missing-wrapper", wrapper));
        // Corrupted fitted constant in the shipped source, resealed.
        let mut constant = entries_a.clone();
        let mut files = bundle_a.files.clone();
        let source = files.get_mut("model.py").unwrap();
        let start = source.find("\\\"weights\\\":[").unwrap() + "\\\"weights\\\":[".len();
        let end = start + source[start..].find(',').unwrap();
        let weight: f64 = source[start..end].parse().unwrap();
        source.replace_range(start..end, &format!("{:?}", weight + 1.0));
        constant[python] = reseal(&entries_a[python], files);
        attempts.push(("corrupted-constant", constant));
        // Omitted operation: the missing-indicator encoding line is removed from Rust source.
        let mut omitted = entries_a.clone();
        let mut files = prepared(&entries_a[rust]).0.files.clone();
        let source = files.get_mut("model.rs").unwrap();
        let line = "x.push(if absent { 1.0 } else { 0.0 });\n";
        assert!(source.contains(line));
        *source = source.replacen(line, "", 1);
        omitted[rust] = reseal(&entries_a[rust], files);
        attempts.push(("omitted-operation", omitted));
        // Hidden support: an emitted backend is recorded as unsupported.
        let mut hidden = entries_a.clone();
        hidden[python] = Entry::Unsupported {
            backend: Backend::Python,
            reason: json!({"kind":"cpp_matrix"}),
        };
        attempts.push(("hidden-support", hidden));
        // Poisoned native reference.
        let mut poisoned = entries_a.clone();
        if let Entry::Prepared { qualification, .. } = &mut poisoned[python] {
            qualification.expected = serde_json::from_value(far_b["predictions"].clone()).unwrap();
        }
        attempts.push(("poisoned-reference", poisoned));

        let mut accepted = Vec::new();
        for (label, entries) in &attempts {
            let public = root.path().join(label).join("public");
            write_public(&public, entries);
            if verify_entries(entries, &output_a, &sha_a, &public).is_ok() {
                accepted.push(*label);
            }
        }
        // No authority-bearing input changed: both fitted subjects and A's receipt are intact.
        assert_eq!(a.bytes().unwrap(), a_bytes);
        assert_eq!(b.bytes().unwrap(), b_bytes);
        assert_eq!(worker_output(&a, &rows).0, output_a);
        verify_entries(&entries_a, &output_a, &sha_a, &public).unwrap();
        assert!(
            accepted.is_empty(),
            "accepted tampered entries: {accepted:?}"
        );
    }
}
