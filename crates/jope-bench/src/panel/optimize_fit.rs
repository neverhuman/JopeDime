//! Exact training-only fold inputs, shared across recipe specifications. Large
//! raw tables are written once per fold rather than copied into every queued job.
use super::{
    campaign::Runtime,
    data::{DevelopmentCut, DevelopmentData, PreparedRow},
    synthesized::{SynthesizedFitSpec, SynthesizedFoldInput, SynthesizedWorkerOutput},
};
use anyhow::{ensure, Context, Result};
use jope_datasets::catalog::Task;
use jope_pipeline::{
    execution::{Artifact, WorkerCommand, WorkerJob, WorkerOutcome},
    experiment::{EvaluationRequest, EvaluationStatus},
    neural_search::{Candidate, Feedback},
    search_plan::{OofEvidence, SelectionPlan},
    sha256_hex,
};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

pub const INNER_FOLD_POLICY: &str = jope_pipeline::search_plan::INNER_GROUP_POLICY;
pub fn inner_folds(rows: &[PreparedRow]) -> Result<Vec<Vec<String>>> {
    jope_pipeline::search_plan::grouped_inner_folds(
        &rows
            .iter()
            .map(|r| (r.id.clone(), r.group.clone()))
            .collect::<Vec<_>>(),
    )
}

pub fn selection(cut: &DevelopmentCut, outer: EvaluationRequest) -> Result<SelectionPlan> {
    SelectionPlan::new(
        outer,
        cut.train.rows.iter().map(|r| r.id.clone()).collect(),
        cut.hold.rows.iter().map(|r| r.id.clone()).collect(),
        inner_folds(&cut.train.rows)?,
        cut.train.rows.iter().map(|r| r.group.clone()).collect(),
        cut.hold.rows.iter().map(|r| r.group.clone()).collect(),
    )
}

pub fn digest(value: &impl Serialize) -> Result<String> {
    Ok(sha256_hex(&serde_json::to_vec(value)?))
}
pub fn artifact(path: PathBuf) -> Result<Artifact> {
    let value = Artifact {
        sha256: sha256_hex(&fs::read(&path)?),
        path,
    };
    value.verify()?;
    Ok(value)
}
pub fn immutable(path: &Path, value: &impl Serialize) -> Result<Artifact> {
    let bytes = serde_json::to_vec(value)?;
    immutable_bytes(path, &bytes)
}
pub fn immutable_bytes(path: &Path, bytes: &[u8]) -> Result<Artifact> {
    if path.try_exists()? {
        ensure!(
            fs::symlink_metadata(path)?.is_file()
                && !fs::symlink_metadata(path)?.file_type().is_symlink(),
            "nonphysical private artifact"
        );
        ensure!(
            fs::read(path)? == bytes,
            "immutable optimization evidence changed: {}",
            path.display()
        );
    } else {
        use std::io::Write;
        fs::create_dir_all(path.parent().context("artifact parent")?)?;
        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::File::open(path.parent().context("artifact parent")?)?.sync_all()?;
    }
    artifact(path.into())
}

pub fn input_artifact(write: bool, path: &Path, value: &impl Serialize) -> Result<Artifact> {
    if write {
        immutable(path, value)
    } else {
        let pin = artifact(path.into())?;
        ensure!(
            fs::read(path)? == serde_json::to_vec(value)?,
            "retained fold/candidate input changed frozen rows"
        );
        Ok(pin)
    }
}
pub struct FoldInput {
    pub file: Artifact,
    pub labels: Vec<String>,
}
pub struct FitAdapter {
    pub write_inputs: bool,
    pub runtime: Runtime,
    pub journal: PathBuf,
    pub selection: SelectionPlan,
    pub folds: Vec<FoldInput>,
}
impl FitAdapter {
    pub fn prepare(
        dataset: &DevelopmentData,
        cut: &DevelopmentCut,
        selection: SelectionPlan,
        runtime: Runtime,
        journal: &Path,
        write_inputs: bool,
    ) -> Result<Self> {
        let rows = cut
            .train
            .rows
            .iter()
            .map(|r| (r.id.as_str(), r))
            .collect::<BTreeMap<_, _>>();
        let mut folds = Vec::new();
        for fold in 0..5 {
            let (fit_ids, predict_ids) = selection.inner_rows(fold)?;
            let fit = fit_ids
                .iter()
                .map(|r| {
                    rows.get(r.as_str())
                        .copied()
                        .context("inner fit row outside outer training closure")
                })
                .collect::<Result<Vec<_>>>()?;
            let predict = predict_ids
                .iter()
                .map(|r| {
                    rows.get(r.as_str())
                        .copied()
                        .context("inner validation row outside outer training closure")
                })
                .collect::<Result<Vec<_>>>()?;
            let input = fold_input(dataset, &fit, &predict)?;
            let hash = digest(&input)?;
            let file = input_artifact(
                write_inputs,
                &journal.join("asm-fold-inputs").join(format!("{hash}.json")),
                &input,
            )?;
            folds.push(FoldInput {
                file,
                labels: predict.iter().map(|r| r.target.clone()).collect(),
            });
        }
        Ok(Self {
            write_inputs,
            runtime,
            journal: journal.into(),
            selection,
            folds,
        })
    }
    pub fn job(&self, candidate: &Candidate, request: &EvaluationRequest) -> Result<WorkerJob> {
        let expected = self
            .selection
            .selection_requests(&candidate.program_sha256, &candidate.id_sha256)?;
        ensure!(
            expected
                .get(request.fold as usize)
                .is_some_and(|r| r.identity().ok() == request.identity().ok()),
            "candidate changed exact inner fold request"
        );
        let fold = self
            .folds
            .get(request.fold as usize)
            .context("inner fold outside plan")?;
        let input = SynthesizedFitSpec {
            schema: "jope.synthesized-fit-spec/v1".into(),
            fold_path: fold.file.path.clone(),
            fold_sha256: fold.file.sha256.clone(),
            recipe: candidate.recipe.clone(),
            scheduling_seed: request.seed,
        };
        let input = input_artifact(
            self.write_inputs,
            &self
                .journal
                .join("inputs")
                .join(format!("{}.json", request.identity()?)),
            &input,
        )?;
        synthesized_job(
            &self.runtime,
            &self.journal,
            request.clone(),
            input,
            vec![fold.file.clone()],
        )
    }
    pub fn score(
        &self,
        candidate: &Candidate,
        outcomes: &[WorkerOutcome],
        selection: &SelectionPlan,
    ) -> Result<Feedback> {
        ensure!(
            selection.identity()? == self.selection.identity()? && outcomes.len() == 5,
            "OOF output closure differs"
        );
        let requests =
            selection.selection_requests(&candidate.program_sha256, &candidate.id_sha256)?;
        let mut losses = Vec::new();
        let mut fitted = Vec::new();
        let mut starts = Vec::new();
        for (fold, (request, outcome)) in requests.iter().zip(outcomes).enumerate() {
            let mut job = self.job(candidate, request)?;
            // The pool attaches the original arm to this exact command.
            let retained: serde_json::Value = serde_json::from_slice(&fs::read(
                self.journal
                    .join("started")
                    .join(format!("{}.json", outcome.job_sha256)),
            )?)?;
            let actual: WorkerJob = serde_json::from_value(retained["job"].clone())?;
            job.arm = actual.arm.clone();
            ensure!(
                job.identity()? == actual.identity()?,
                "OOF started command differs"
            );
            super::receipt_timing::completed_fit(&self.journal, &job, outcome)?;
            let result = checked_output(&job, outcome, &candidate.recipe, request.seed)?;
            let measured = super::score::score_predictions(
                request.task,
                &request.classes,
                &self.folds[fold].labels,
                &result.predictions,
            )?;
            losses.push(request.metric.loss(measured.value)?);
            starts.push(result.physical_fit_starts);
            fitted.push(result.artifact_sha256);
        }
        let oof = OofEvidence::for_candidate(
            selection,
            &candidate.program_sha256,
            &candidate.id_sha256,
            starts,
            fitted,
        )?;
        let loss = losses.iter().sum::<f64>() / losses.len() as f64;
        ensure!(loss.is_finite(), "nonfinite train-side selection loss");
        Ok(Feedback {
            candidate_sha256: candidate.id_sha256.clone(),
            status: EvaluationStatus::Completed,
            loss: Some(loss),
            physical_fit_starts: oof.total_fit_starts(),
            worker_receipt_sha256: Some(digest(&outcomes)?),
            replay_evidence_sha256: Some(jope_pipeline::neural_search::fold_replay_evidence(
                outcomes,
            )?),
            fitted_artifact_sha256: Some(oof.fitted_bundle_sha256()?),
            reason: None,
            oof: Some(oof),
            retryable_inventory: false,
        })
    }
}

pub fn fold_input(
    dataset: &DevelopmentData,
    fit: &[&PreparedRow],
    predict: &[&PreparedRow],
) -> Result<SynthesizedFoldInput> {
    let labels = dataset
        .dataset
        .classes
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i))
        .collect::<BTreeMap<_, _>>();
    let targets = fit
        .iter()
        .map(|r| {
            if dataset.dataset.task == Task::Regression {
                let value: f64 = r.target.parse()?;
                ensure!(value.is_finite(), "nonfinite training target");
                Ok(value)
            } else {
                Ok(*labels
                    .get(r.target.as_str())
                    .context("training class outside frozen vocabulary")? as f64)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(SynthesizedFoldInput {
        task: dataset.dataset.task,
        schema: dataset.schema.clone(),
        classes: dataset.dataset.classes.clone(),
        train: super::data::raw_table(
            &dataset.dataset.features,
            &fit.iter().map(|r| r.values.clone()).collect::<Vec<_>>(),
        )?,
        targets,
        predict: super::data::raw_table(
            &dataset.dataset.features,
            &predict.iter().map(|r| r.values.clone()).collect::<Vec<_>>(),
        )?,
    })
}

pub fn synthesized_job(
    runtime: &Runtime,
    journal: &Path,
    request: EvaluationRequest,
    input: Artifact,
    extra: Vec<Artifact>,
) -> Result<WorkerJob> {
    let directory = journal.join("jobs").join(request.identity()?);
    let mut args = runtime.argument_prefix.clone();
    args.extend([
        "synthesized-fit-worker".into(),
        "--input".into(),
        input.path.display().to_string(),
        "--output".into(),
        directory.join("output.json").display().to_string(),
    ]);
    let mut inputs = runtime.inputs.clone();
    inputs.push(input);
    inputs.extend(extra);
    let job = WorkerJob {
        request,
        command: WorkerCommand {
            program: runtime.program.clone(),
            args,
            inputs,
            environment: runtime.environment.clone(),
        },
        directory,
        outputs: vec!["output.json".into()],
        arm: None,
    };
    job.identity()?;
    Ok(job)
}
pub fn checked_output(
    job: &WorkerJob,
    outcome: &WorkerOutcome,
    recipe: &jope_pipeline::synthesized_recipe::SynthesizedRecipe,
    seed: u64,
) -> Result<SynthesizedWorkerOutput> {
    ensure!(
        outcome.command_status == EvaluationStatus::Completed
            && outcome.exit_code == Some(0)
            && outcome.reason.is_none()
            && outcome.request_sha256 == job.request.identity()?
            && outcome.job_sha256 == job.identity()?,
        "unsuccessful/stale synthesized command"
    );
    outcome.verify_outputs()?;
    let path = job.directory.join("output.json");
    ensure!(
        outcome.outputs.len() == 1 && outcome.outputs[0].path == path,
        "synthesized output inventory differs"
    );
    let result: SynthesizedWorkerOutput = serde_json::from_slice(&fs::read(&path)?)?;
    let input = job
        .command
        .args
        .windows(2)
        .find(|p| p[0] == "--input")
        .context("input argument absent")?;
    let input = job
        .command
        .inputs
        .iter()
        .find(|a| a.path == Path::new(&input[1]))
        .context("input artifact absent")?;
    ensure!(
        result.schema == "jope.panel-synthesized-worker-output/v1"
            && result.input_sha256 == input.sha256
            && result.executable_recipe_sha256 == recipe.executable_sha256
            && result.physical_fit_starts == 1
            && result.artifact.scheduling_seed == seed
            && result.artifact.task == job.request.task
            && result.artifact.classes == job.request.classes
            && digest(&result.artifact.recipe)? == digest(recipe)?,
        "synthesized output changed original recipe, class order or fit count"
    );
    result.artifact.validate()?;
    ensure!(
        result.artifact_sha256 == sha256_hex(&result.artifact.bytes()?),
        "complete fitted predictor hash differs"
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inner_assignment_uses_groups_and_is_independent_of_targets() -> Result<()> {
        let mut rows = (0..30)
            .map(|i| PreparedRow {
                id: sha256_hex(format!("r{i}").as_bytes()),
                source_index: i,
                group: sha256_hex(format!("g{}", i / 2).as_bytes()),
                fold: 1,
                values: vec![Some(i.to_string())],
                target: (i % 2).to_string(),
            })
            .collect::<Vec<_>>();
        let before = inner_folds(&rows)?;
        for row in &mut rows {
            row.target = "renamed".into();
        }
        assert_eq!(before, inner_folds(&rows)?);
        let mut groups = BTreeMap::new();
        for (i, fold) in before.iter().enumerate() {
            for id in fold {
                let group = &rows.iter().find(|r| &r.id == id).unwrap().group;
                assert!(groups.insert(group, i).is_none_or(|previous| previous == i));
            }
        }
        assert_eq!(
            before
                .iter()
                .flatten()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            rows.len()
        );
        Ok(())
    }
}
