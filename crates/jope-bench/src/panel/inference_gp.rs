//! Training-only timing adapter: production GP operators and five-fold selection.
use super::*;
use jope_pipeline::{
    execution::{ArmBudget, WorkerJob, WorkerOutcome},
    experiment::EvaluationStatus,
    neural_search::{self, Backend, Candidate, Feedback, Variation},
    search_plan::{OofEvidence, SelectionPlan},
};
use std::time::{SystemTime, UNIX_EPOCH};
fn clock() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_millis() as u64
}
pub struct TimingBackend<'a> {
    pub input: &'a GpInput,
    pub root: PathBuf,
    pub cutoff: u64,
}
impl Backend for TimingBackend<'_> {
    fn now_ms(&self) -> u64 {
        clock()
    }
    fn begin_arm(&mut self, arm: &ArmBudget) -> Result<u64> {
        let path = self.root.join(&arm.id_sha256).join("deadline.json");
        fs::create_dir_all(path.parent().unwrap())?;
        if path.exists() {
            return Ok(
                serde_json::from_slice::<Value>(&fs::read(path)?)?["deadline"]
                    .as_u64()
                    .context("deadline")?,
            );
        }
        let deadline = (clock() + arm.seconds * 1000).min(self.cutoff);
        write(&path, &json!({"deadline":deadline}))?;
        Ok(deadline)
    }
    fn evaluate(
        &mut self,
        arm: &ArmBudget,
        selection: &SelectionPlan,
        candidates: &[Candidate],
    ) -> Result<Vec<Feedback>> {
        let deadline = self.begin_arm(arm)?;
        let mut all = Vec::new();
        for c in candidates {
            let directory = self.root.join(&arm.id_sha256).join(&c.id_sha256);
            fs::create_dir_all(&directory)?;
            let saved = directory.join("feedback.json");
            if saved.exists() {
                all.push(serde_json::from_slice(&fs::read(saved)?)?);
                continue;
            }
            let measured = (|| -> Result<Feedback> {
                let mut losses = Vec::new();
                let mut fitted = Vec::new();
                let mut starts = Vec::new();
                let mut receipts = Vec::new();
                for (fold, (pin, labels)) in self.input.folds.iter().enumerate() {
                    ensure!(clock() < deadline, "original arm deadline reached");
                    pin.verify()?;
                    let value: super::super::synthesized::SynthesizedFoldInput =
                        serde_json::from_slice(&fs::read(&pin.path)?)?;
                    let req = super::super::synthesized::SynthesizedFitRequest {
                        task: value.task,
                        schema: value.schema,
                        classes: value.classes,
                        train: value.train,
                        targets: value.targets,
                        predict: value.predict,
                        recipe: c.recipe.clone(),
                        scheduling_seed: 0,
                    };
                    let dir = directory.join(format!("fold-{fold}"));
                    fs::create_dir_all(&dir)?;
                    write(&dir.join("input.json"), &req)?;
                    let args = vec![
                        std::env::current_exe()?.display().to_string(),
                        "benchmark".into(),
                        "synthesized-fit-worker".into(),
                        "--input".into(),
                        dir.join("input.json").display().to_string(),
                        "--output".into(),
                        dir.join("output.json").display().to_string(),
                    ];
                    let (_, r) = export::command(
                        &dir,
                        &args,
                        30.min(deadline.saturating_sub(clock()).div_ceil(1000).max(1)),
                        "fit",
                    )?;
                    let output: super::super::synthesized::SynthesizedWorkerOutput =
                        serde_json::from_slice(&fs::read(dir.join("output.json"))?)?;
                    ensure!(
                        output.input_sha256 == digest(&serde_json::to_vec(&req)?)
                            && output.artifact.recipe == c.recipe,
                        "fit identity changed"
                    );
                    output.artifact.validate()?;
                    losses.push(
                        super::super::score::score_predictions(
                            req.task,
                            &[],
                            labels,
                            &output.predictions,
                        )?
                        .loss,
                    );
                    ensure!(
                        output.artifact_sha256 == digest(&output.artifact.bytes()?),
                        "fitted hash changed"
                    );
                    starts.push(output.physical_fit_starts);
                    fitted.push(output.artifact_sha256);
                    receipts.push(r);
                }
                let oof = OofEvidence::for_candidate(
                    selection,
                    &c.program_sha256,
                    &c.id_sha256,
                    starts.clone(),
                    fitted,
                )?;
                Ok(Feedback {
                    candidate_sha256: c.id_sha256.clone(),
                    status: EvaluationStatus::Completed,
                    loss: Some(losses.iter().sum::<f64>() / 5.0),
                    physical_fit_starts: Some(starts.iter().sum()),
                    worker_receipt_sha256: Some(digest(&serde_json::to_vec(&receipts)?)),
                    replay_evidence_sha256: None,
                    fitted_artifact_sha256: Some(oof.fitted_bundle_sha256()?),
                    reason: None,
                    oof: Some(oof),
                    retryable_inventory: false,
                })
            })();
            let feedback = match measured {
                Ok(f) => f,
                Err(e) => Feedback {
                    candidate_sha256: c.id_sha256.clone(),
                    status: if clock() >= deadline {
                        EvaluationStatus::BudgetExhausted
                    } else {
                        EvaluationStatus::Failed
                    },
                    loss: None,
                    physical_fit_starts: None,
                    worker_receipt_sha256: None,
                    replay_evidence_sha256: None,
                    fitted_artifact_sha256: None,
                    reason: Some(format!("{e:#}")),
                    oof: None,
                    retryable_inventory: false,
                },
            };
            write(&saved, &feedback)?;
            all.push(feedback);
        }
        Ok(all)
    }
    fn produce(&mut self, _: WorkerJob) -> Result<WorkerOutcome> {
        anyhow::bail!("timing adapter cannot produce admission evidence")
    }
    fn persist(&mut self, arm: &ArmBudget, name: &str, value: &Value) -> Result<()> {
        let dir = self.root.join(&arm.id_sha256);
        fs::create_dir_all(&dir)?;
        write(&dir.join(format!("{name}.json")), value)
    }
    fn load(&self, arm: &ArmBudget, name: &str) -> Result<Option<Value>> {
        let p = self.root.join(&arm.id_sha256).join(format!("{name}.json"));
        if p.exists() {
            Ok(Some(serde_json::from_slice(&fs::read(p)?)?))
        } else {
            Ok(None)
        }
    }
}
pub fn arm(
    backend: &mut TimingBackend<'_>,
    vocab: &jope_policy::grammar::BoundVocabulary,
    proposal: Option<&jope_policy::seeds::Proposal>,
) -> Result<neural_search::SearchReceipt> {
    let mut variation = neural_search::RegressionVariation { vocabulary: vocab };
    ensure!(
        variation.identity() == backend.input.pair.variation_identity
            && variation.generic_policy() == backend.input.pair.generic_sampling_policy,
        "variation identity changed"
    );
    neural_search::timing_arm(
        backend,
        &backend.input.pair.clone(),
        &mut variation,
        proposal,
    )
}
