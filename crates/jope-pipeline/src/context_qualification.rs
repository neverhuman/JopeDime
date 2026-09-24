//! Parent controller for eight bounded native-reference encoding jobs.
//! These Diagnostic jobs score reference coordinates, never fitted predictors,
//! benchmark quality outcomes, teaching records or heldout labels.
use crate::{
    cpu_context_bridge::ExecutionProof,
    execution::{Artifact, ExecutionPool, WorkerCommand, WorkerJob, WorkerOutcome},
    experiment::{
        EvaluationRequest, EvaluationStage, EvaluationStatus, Experiment, Metric, REQUEST_SCHEMA,
    },
    sha256_hex,
};
use anyhow::{ensure, Context, Result};
use jope_policy::context_producer::{
    self,
    worker::{WorkerInvocation, WorkerMode},
    EncoderFiles,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
};
mod materialize;
pub use materialize::*;

pub const QUALIFICATION_PROTOCOL: &str = "jope.cpu-context-proof/v1";
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderAssets {
    pub configuration: Artifact,
    pub files: Vec<Artifact>,
}
/// Hash and inventory every model/metadata/notice byte without tensor loading.
pub fn prepare_assets(files: &EncoderFiles, path: &Path) -> Result<EncoderAssets> {
    let inventory = context_producer::worker::asset_inventory(files)?
        .into_iter()
        .map(|a| Artifact {
            path: a.path,
            sha256: a.sha256,
        })
        .collect();
    Ok(EncoderAssets {
        configuration: write_immutable(path, &serde_json::to_vec(files)?)?,
        files: inventory,
    })
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationPlan {
    pub schema: String,
    pub experiment: Experiment,
    pub references: MaterializedReferences,
    pub program: Artifact,
    pub assets: EncoderAssets,
    pub journal: PathBuf,
    pub jobs: Vec<WorkerJob>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletedQualification {
    pub qualification: Artifact,
    pub reference_manifest: Artifact,
    pub case_outputs: Vec<Artifact>,
    pub case_executions: Vec<ExecutionProof>,
}

impl CompletedQualification {
    /// Re-admit retained evidence without running workers or opening a window.
    pub fn verify(&self, source: &str, program: &Artifact) -> Result<()> {
        ensure!(
            self.case_outputs.len() == 8 && self.case_executions.len() == 8,
            "qualification requires eight executed cases"
        );
        program.verify()?;
        let manifest = read_artifact(&self.reference_manifest, 128 << 20)?;
        let cases = context_producer::case_inputs(&manifest, &self.reference_manifest.sha256)?;
        let mut outputs = Vec::new();
        for ((case, output), proof) in cases
            .iter()
            .zip(&self.case_outputs)
            .zip(&self.case_executions)
        {
            ensure!(
                proof.experiment.source_sha == source
                    && proof.job.command.program.path == program.path
                    && proof.job.command.program.sha256 == program.sha256,
                "qualification source or worker differs"
            );
            let input_sha = sha256_hex(case);
            let inputs = proof
                .job
                .command
                .inputs
                .iter()
                .filter(|a| a.sha256 == input_sha)
                .collect::<Vec<_>>();
            ensure!(
                inputs.len() == 1,
                "qualification case input absent or duplicated"
            );
            proof.attest(inputs[0], output)?;
            outputs.push(read_artifact(output, 2 << 20)?);
        }
        let rebuilt = context_producer::aggregate_qualification(
            &manifest,
            &self.reference_manifest.sha256,
            outputs,
            source,
            &program.sha256,
        )?;
        let retained: serde_json::Value =
            serde_json::from_slice(&read_artifact(&self.qualification, 32 << 20)?)?;
        ensure!(
            serde_json::to_value(rebuilt)? == retained,
            "qualification differs from actual case outputs"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationRun {
    pub schema: String,
    pub plan_sha256: String,
    pub outcomes: Vec<WorkerOutcome>,
    pub completed: Option<CompletedQualification>,
    pub validation_error: Option<String>,
    pub fitted_predictors: usize,
}

/// Prepare an immutable plan without admitting numerical execution. Existing
/// materialization and planning may resume only with identical input bytes.
pub fn plan_qualification(
    experiment: &Experiment,
    references: MaterializedReferences,
    program: Artifact,
    assets: EncoderAssets,
    journal: &Path,
) -> Result<QualificationPlan> {
    physical_directory(journal)?;
    experiment.validate()?;
    ensure!(
        experiment.protocol == QUALIFICATION_PROTOCOL
            && experiment.resources.gpu_tasks == 0
            && experiment.resources.per_fit_seconds <= 30
            && experiment.resources.threads_per_worker == 1,
        "qualification requires explicit CPU proof protocol and original bounded resources"
    );
    ensure!(
        experiment.panel_sha256 == references.manifest.sha256
            && experiment.numerical_policy == context_producer::verification_policy_sha256(),
        "qualification experiment differs from frozen reference/policy"
    );
    program.verify()?;
    assets.configuration.verify()?;
    let files: EncoderFiles =
        serde_json::from_slice(&read_artifact(&assets.configuration, 1 << 20)?)?;
    let expected = context_producer::worker::asset_inventory(&files)?
        .into_iter()
        .map(|a| Artifact {
            path: a.path,
            sha256: a.sha256,
        })
        .collect::<Vec<_>>();
    ensure!(
        serde_json::to_vec(&expected)? == serde_json::to_vec(&assets.files)?,
        "model closure differs from governed inventory"
    );
    let jobs = make_jobs(experiment, &references, &program, &assets, journal)?;
    let plan = QualificationPlan {
        schema: QUALIFICATION_PROTOCOL.into(),
        experiment: experiment.clone(),
        references,
        program,
        assets,
        journal: journal.into(),
        jobs,
    };
    write_immutable(
        &journal.join("context-qualification-plan.json"),
        &serde_json::to_vec(&plan)?,
    )?;
    Ok(plan)
}
fn make_jobs(
    experiment: &Experiment,
    references: &MaterializedReferences,
    program: &Artifact,
    assets: &EncoderAssets,
    journal: &Path,
) -> Result<Vec<WorkerJob>> {
    let bytes = read_artifact(&references.manifest, 128 << 20)?;
    let expected_inputs = context_producer::case_inputs(&bytes, &references.manifest.sha256)?;
    ensure!(
        references.cases.len() == 8 && references.job_identities.len() == 8,
        "incomplete qualification case inventory"
    );
    let manifest: context_producer::ReferenceManifest = serde_json::from_slice(&bytes)?;
    let mut jobs = Vec::new();
    for (index, expected_input) in expected_inputs.iter().enumerate() {
        let input = &references.cases[index];
        ensure!(
            read_artifact(input, 128 << 20)? == *expected_input,
            "case payload differs from predeclared manifest order"
        );
        let identity = &references.job_identities[index];
        let reference = &manifest.cases[index];
        let native: serde_json::Value = serde_json::from_slice(&reference.context_receipt)?;
        ensure!(
            identity.fitting_rows == reference.prepared_rows.len()
                && identity.fold as u64
                    == native["cut"].as_u64().context("native reference cut")?
                && native["prepared_sha256"] == identity.split_sha256
                && native["selected_training_rows_sha256"] == identity.fitting_rows_sha256,
            "qualification diagnostic identity differs from actual reference rows"
        );
        let reference_coordinates = reference.components["hyperion"]
            .vector
            .iter()
            .chain(&reference.components["foundation"].vector)
            .chain(&reference.components["target"].vector)
            .chain(&reference.context_vector)
            .copied()
            .collect::<Vec<_>>();
        ensure!(
            reference_coordinates.len() == 5120 * 4,
            "native diagnostic coordinate count changed"
        );
        let request = EvaluationRequest {
            schema: REQUEST_SCHEMA.into(),
            experiment_sha256: experiment.identity()?,
            dataset_sha256: identity.dataset_sha256.clone(),
            split_sha256: identity.split_sha256.clone(),
            fitting_rows_sha256: identity.fitting_rows_sha256.clone(),
            scoring_rows_sha256: sha256_hex(&reference_coordinates),
            fitting_rows: identity.fitting_rows,
            scoring_rows: 5120,
            recipe_sha256: sha256_hex(&serde_json::to_vec(&(
                QUALIFICATION_PROTOCOL,
                &program.sha256,
                &assets.configuration.sha256,
                &input.sha256,
            ))?),
            evaluator_sha256: context_producer::verification_policy_sha256(),
            checkpoint_sha256: Some(context_producer::TRANSFORMS_SHA256.into()),
            task: jope_datasets::catalog::Task::Regression,
            metric: Metric::Rmse,
            classes: vec![],
            fold: identity.fold,
            stage: EvaluationStage::Diagnostic,
            seed: experiment.seed,
            numerical_policy: context_producer::verification_policy_sha256(),
        };
        let directory = journal.join("jobs").join(request.identity()?);
        let invocation = WorkerInvocation {
            mode: WorkerMode::QualifyCase,
            input: policy_artifact(input),
            assets: policy_artifact(&assets.configuration),
            source_sha40: experiment.source_sha.clone(),
            producer_sha256: program.sha256.clone(),
            output: directory.join("case-output.json"),
        };
        let mut inputs = vec![input.clone(), assets.configuration.clone()];
        inputs.extend(assets.files.clone());
        let job = WorkerJob {
            request,
            command: WorkerCommand {
                program: program.clone(),
                args: invocation.arguments(),
                inputs,
                environment: BTreeMap::new(),
            },
            directory,
            outputs: vec!["case-output.json".into()],
            arm: None,
        };
        job.identity()?;
        jobs.push(job);
    }
    Ok(jobs)
}

pub fn run_qualification(
    pool: &mut ExecutionPool,
    plan: &QualificationPlan,
    cancellation: &AtomicBool,
) -> Result<QualificationRun> {
    // Runtime verification belongs to the original execution allowance too.
    // Installation and frozen plan preparation happen before this boundary.
    pool.admit()?;
    let persisted = read_artifact(
        &Artifact {
            path: plan.journal.join("context-qualification-plan.json"),
            sha256: sha256_hex(&serde_json::to_vec(plan)?),
        },
        32 << 20,
    )?;
    ensure!(
        persisted == serde_json::to_vec(plan)?,
        "qualification plan changed"
    );
    let expected = make_jobs(
        &plan.experiment,
        &plan.references,
        &plan.program,
        &plan.assets,
        &plan.journal,
    )?;
    ensure!(
        serde_json::to_vec(&expected)? == serde_json::to_vec(&plan.jobs)?,
        "qualification commands differ from frozen plan"
    );
    let outcomes = pool.run(plan.jobs.clone(), cancellation)?;
    ensure!(
        outcomes.len() == 8,
        "pool returned incomplete qualification outcomes"
    );
    let mut report = QualificationRun {
        schema: "jope.cpu-context-proof-result/v1".into(),
        plan_sha256: sha256_hex(&persisted),
        outcomes,
        completed: None,
        validation_error: None,
        fitted_predictors: 0,
    };
    if report
        .outcomes
        .iter()
        .all(|o| o.command_status == EvaluationStatus::Completed)
    {
        match finish_qualification(plan, &report.outcomes) {
            Ok(completed) => report.completed = Some(completed),
            Err(error) => report.validation_error = Some(format!("{error:#}")),
        }
    } else {
        report.validation_error=Some("native parity remains unavailable: one or more bounded case jobs did not complete; inspect all8 outcomes".into());
    }
    // Failure evidence is immutable too. A resume does not overwrite a prior
    // result or reset the pool window. Different attempts retain distinct hashes.
    let bytes = serde_json::to_vec(&report)?;
    write_immutable(
        &plan.journal.join(format!(
            "context-qualification-result-{}.json",
            sha256_hex(&bytes)
        )),
        &bytes,
    )?;
    Ok(report)
}
fn finish_qualification(
    plan: &QualificationPlan,
    outcomes: &[WorkerOutcome],
) -> Result<CompletedQualification> {
    let window = artifact(&plan.journal.join("window.json"))?;
    let mut outputs = Vec::new();
    let mut output_bytes = Vec::new();
    let mut proofs = Vec::new();
    for (index, (job, outcome)) in plan.jobs.iter().zip(outcomes).enumerate() {
        ensure!(
            outcome.job_sha256 == job.identity()?
                && outcome.request_sha256 == job.request.identity()?,
            "stale qualification outcome"
        );
        outcome.verify_outputs()?;
        let output = outcome
            .outputs
            .iter()
            .find(|a| a.path == job.directory.join("case-output.json"))
            .context("case output absent")?
            .clone();
        let proof = ExecutionProof {
            job: job.clone(),
            outcome: artifact(
                &plan
                    .journal
                    .join("finished")
                    .join(format!("{}.json", job.identity()?)),
            )?,
            started: artifact(
                &plan
                    .journal
                    .join("started")
                    .join(format!("{}.json", job.identity()?)),
            )?,
            experiment: plan.experiment.clone(),
            window: window.clone(),
            arm_window: None,
        };
        let attestation = proof.attest(&plan.references.cases[index], &output)?;
        write_immutable(
            &job.directory.join("execution-attestation.json"),
            &serde_json::to_vec(&attestation)?,
        )?;
        output_bytes.push(read_artifact(&output, 2 << 20)?);
        outputs.push(output);
        proofs.push(proof);
    }
    let manifest = read_artifact(&plan.references.manifest, 128 << 20)?;
    let qualification = context_producer::aggregate_qualification(
        &manifest,
        &plan.references.manifest.sha256,
        output_bytes,
        &plan.experiment.source_sha,
        &plan.program.sha256,
    )?;
    let file = write_immutable(
        &plan.journal.join("context-qualification.json"),
        &serde_json::to_vec(&qualification)?,
    )?;
    Ok(CompletedQualification {
        qualification: file,
        reference_manifest: plan.references.manifest.clone(),
        case_outputs: outputs,
        case_executions: proofs,
    })
}

/// New artifacts are private; existing identical bytes permit honest resume.
/// A changed artifact at the same path fails instead of overwriting evidence.
pub(crate) fn write_immutable(path: &Path, bytes: &[u8]) -> Result<Artifact> {
    if path.exists() {
        let expected = Artifact {
            path: path.into(),
            sha256: sha256_hex(bytes),
        };
        expected.verify()?;
        return Ok(expected);
    }
    context_producer::worker::write_new(path, bytes)?;
    Ok(Artifact {
        path: path.into(),
        sha256: sha256_hex(bytes),
    })
}
fn artifact(path: &Path) -> Result<Artifact> {
    ensure!(
        path.is_absolute() && path.canonicalize()? == path && fs::symlink_metadata(path)?.is_file(),
        "proof path must be physical regular file"
    );
    let a = Artifact {
        path: path.into(),
        sha256: crate::sha256_file(path)?,
    };
    a.verify()?;
    Ok(a)
}
fn policy_artifact(a: &Artifact) -> context_producer::AssetFile {
    context_producer::AssetFile {
        path: a.path.clone(),
        sha256: a.sha256.clone(),
    }
}
fn read_artifact(a: &Artifact, maximum: u64) -> Result<Vec<u8>> {
    policy_artifact(a).read(maximum)
}
fn physical_directory(path: &Path) -> Result<()> {
    ensure!(
        path.is_absolute() && path.canonicalize()? == path && fs::symlink_metadata(path)?.is_dir(),
        "private custody directory must be physical and absolute"
    );
    Ok(())
}
