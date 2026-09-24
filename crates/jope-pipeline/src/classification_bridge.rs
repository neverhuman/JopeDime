//! Parent-only experimental classification admission. Original process receipts
//! and all eight native component proofs are mandatory. No serialized boolean
//! creates the private verification token consumed by experimental training.
use crate::{
    context_qualification::{CompletedQualification, EncoderAssets},
    cpu_context_bridge::ExecutionProof,
    execution::{ArmBudget, Artifact, WorkerCommand, WorkerJob},
    experiment::{EvaluationRequest, EvaluationStage, Experiment},
    search_context::{classification_context, ExperimentalClassificationContext, TrainingIdentity},
    sha256_hex,
};
use anyhow::{ensure, Context, Result};
use jope_policy::context_producer::{
    self,
    classification::{ClassificationArtifact, ClassificationInput, ClassificationTask},
    worker::{WorkerInvocation, WorkerMode},
    ComponentQualificationPins, EncoderFiles,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassificationFiles {
    pub program: Artifact,
    pub source_sha40: String,
    pub input: Artifact,
    pub artifact: Artifact,
    pub encoder_configuration: Artifact,
    pub qualification: CompletedQualification,
    pub execution: ExecutionProof,
}
/// Constructor is private; parent admission cannot be replaced by deserializing
/// policy_compatible, passed, or a caller-selected numerical context vector.
pub struct VerifiedExperimentalContext {
    input: ClassificationInput,
    context: ExperimentalClassificationContext,
    training: TrainingIdentity,
    source_sha40: String,
    producer_sha256: String,
    artifact_sha256: String,
    qualification_sha256: String,
}
impl VerifiedExperimentalContext {
    pub fn input(&self) -> &ClassificationInput {
        &self.input
    }
    pub fn context(&self) -> &ExperimentalClassificationContext {
        &self.context
    }
    pub fn values(&self) -> &[f32] {
        self.context.values()
    }
    pub fn sha256(&self) -> &str {
        self.context.sha256()
    }
    pub fn training(&self) -> &TrainingIdentity {
        &self.training
    }
    pub fn source_sha40(&self) -> &str {
        &self.source_sha40
    }
    pub fn producer_sha256(&self) -> &str {
        &self.producer_sha256
    }
    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }
    pub fn qualification_sha256(&self) -> &str {
        &self.qualification_sha256
    }
}
fn bytes(file: &Artifact, maximum: u64) -> Result<Vec<u8>> {
    ensure!(
        fs::metadata(&file.path)?.len() <= maximum,
        "classification evidence exceeds bound"
    );
    file.verify()?;
    let value = fs::read(&file.path)?;
    ensure!(
        value.len() as u64 <= maximum && sha256_hex(&value) == file.sha256,
        "classification evidence changed while reading"
    );
    Ok(value)
}
fn same(a: &Artifact, b: &Artifact) -> bool {
    a.path == b.path && a.sha256 == b.sha256
}
fn policy_artifact(a: &Artifact) -> context_producer::AssetFile {
    context_producer::AssetFile {
        path: a.path.clone(),
        sha256: a.sha256.clone(),
    }
}
fn closure(input: &Artifact, configuration: &Artifact) -> Result<Vec<Artifact>> {
    let files: EncoderFiles = serde_json::from_slice(&bytes(configuration, 1 << 20)?)?;
    let mut values = vec![input.clone(), configuration.clone()];
    values.extend(
        context_producer::worker::asset_inventory(&files)?
            .into_iter()
            .map(|a| Artifact {
                path: a.path,
                sha256: a.sha256,
            }),
    );
    Ok(values)
}
fn task(input: &ClassificationInput) -> jope_datasets::catalog::Task {
    match input.task {
        ClassificationTask::Binary => jope_datasets::catalog::Task::Binary,
        ClassificationTask::Multiclass => jope_datasets::catalog::Task::Multiclass,
    }
}
/// Call after beginning the original arm. Preparation and closure hashing consume
/// its existing allowance; this function never starts a fresh execution window.
#[allow(clippy::too_many_arguments)]
pub fn build_job(
    experiment: &Experiment,
    request: EvaluationRequest,
    program: Artifact,
    input: Artifact,
    assets: &EncoderAssets,
    journal: &Path,
    arm: Option<ArmBudget>,
) -> Result<WorkerJob> {
    experiment.validate()?;
    request.validate()?;
    let original: ClassificationInput = serde_json::from_slice(&bytes(&input, 128 << 20)?)?;
    original.prepare()?;
    ensure!(
        request.stage == EvaluationStage::Diagnostic
            && request.experiment_sha256 == experiment.identity()?
            && request.task == task(&original)
            && request.classes == original.classes
            && request.dataset_sha256 == original.identity.dataset_sha256
            && request.split_sha256 == original.split_sha256
            && request.fitting_rows_sha256 == original.identity.fitting_rows_sha256
            && request.fitting_rows == original.rows.len()
            && request.fold == original.identity.cut,
        "classification job differs from exact training input"
    );
    let inputs = closure(&input, &assets.configuration)?;
    ensure!(
        serde_json::to_vec(&inputs[2..])? == serde_json::to_vec(&assets.files)?,
        "classification encoder asset inventory changed"
    );
    let directory = journal.join("jobs").join(request.identity()?);
    let invocation = WorkerInvocation {
        mode: WorkerMode::ProduceClassification,
        input: policy_artifact(&input),
        assets: policy_artifact(&assets.configuration),
        source_sha40: experiment.source_sha.clone(),
        producer_sha256: program.sha256.clone(),
        output: directory.join("output.json"),
    };
    let job = WorkerJob {
        request,
        command: WorkerCommand {
            program,
            args: invocation.arguments(),
            inputs,
            environment: BTreeMap::new(),
        },
        directory,
        outputs: vec!["output.json".into()],
        arm,
    };
    job.identity()?;
    Ok(job)
}

pub fn verify_classification(
    files: &ClassificationFiles,
    expected_input: &ClassificationInput,
) -> Result<VerifiedExperimentalContext> {
    expected_input.prepare()?;
    let input_bytes = bytes(&files.input, 128 << 20)?;
    ensure!(
        input_bytes == serde_json::to_vec(expected_input)?,
        "classification input differs from independently reconstructed original training rows"
    );
    let artifact_bytes = bytes(&files.artifact, 128 << 20)?;
    let artifact: ClassificationArtifact = serde_json::from_slice(&artifact_bytes)?;
    artifact.validate_for_input(expected_input)?;
    let proof = &files.execution;
    ensure!(
        same(&proof.job.command.program, &files.program)
            && proof.experiment.source_sha == files.source_sha40
            && artifact.source_sha40 == files.source_sha40
            && artifact.producer_sha256 == files.program.sha256,
        "classification execution source/program changed"
    );
    let request = &proof.job.request;
    ensure!(
        request.stage == EvaluationStage::Diagnostic
            && request.task == task(expected_input)
            && request.classes == expected_input.classes
            && request.dataset_sha256 == expected_input.identity.dataset_sha256
            && request.split_sha256 == expected_input.split_sha256
            && request.fitting_rows_sha256 == expected_input.identity.fitting_rows_sha256
            && request.fitting_rows == expected_input.rows.len()
            && request.fold == expected_input.identity.cut,
        "classification execution changed training selection/task"
    );
    let invocation = WorkerInvocation {
        mode: WorkerMode::ProduceClassification,
        input: policy_artifact(&files.input),
        assets: policy_artifact(&files.encoder_configuration),
        source_sha40: files.source_sha40.clone(),
        producer_sha256: files.program.sha256.clone(),
        output: files.artifact.path.clone(),
    };
    ensure!(
        proof.job.command.args == invocation.arguments()
            && files.artifact.path == proof.job.directory.join("output.json")
            && proof.job.outputs == vec![std::path::PathBuf::from("output.json")]
            && proof.job.command.environment.is_empty(),
        "classification worker invocation/outputs changed"
    );
    let expected_closure = closure(&files.input, &files.encoder_configuration)?;
    ensure!(
        serde_json::to_vec(&proof.job.command.inputs)? == serde_json::to_vec(&expected_closure)?,
        "classification worker has a different input closure"
    );
    let _actual_execution = proof.attest(&files.input, &files.artifact)?;
    let q = &files.qualification;
    ensure!(
        q.case_outputs.len() == 8 && q.case_executions.len() == 8,
        "classification requires all8 original native component qualification jobs"
    );
    let manifest = bytes(&q.reference_manifest, 128 << 20)?;
    let qualification = bytes(&q.qualification, 16 << 20)?;
    let cases = context_producer::case_inputs(&manifest, &q.reference_manifest.sha256)?;
    let mut executions = Vec::new();
    for (i, case) in q.case_executions.iter().enumerate() {
        ensure!(
            same(&case.job.command.program, &files.program)
                && case.experiment.source_sha == files.source_sha40
                && case.job.request.stage == EvaluationStage::Diagnostic,
            "classification component qualification source/program differs"
        );
        let case_invocation = WorkerInvocation::parse(&case.job.command.args)?;
        ensure!(
            matches!(case_invocation.mode, WorkerMode::QualifyCase)
                && case_invocation.source_sha40 == files.source_sha40
                && case_invocation.producer_sha256 == files.program.sha256
                && case_invocation.assets.sha256 == files.encoder_configuration.sha256
                && case_invocation.arguments() == case.job.command.args
                && case.job.command.environment.is_empty()
                && case.job.outputs == vec![std::path::PathBuf::from("case-output.json")]
                && case_invocation.output == case.job.directory.join("case-output.json"),
            "classification component qualification is another worker mode/assets"
        );
        let case_sha = sha256_hex(&cases[i]);
        let input = case
            .job
            .command
            .inputs
            .iter()
            .find(|a| a.sha256 == case_sha && a.path == case_invocation.input.path)
            .context("qualification original case input absent")?;
        ensure!(
            bytes(input, 128 << 20)? == cases[i]
                && case_invocation.input.sha256 == case_sha
                && case_invocation.output == q.case_outputs[i].path,
            "qualification input/output byte closure changed"
        );
        executions.push(serde_json::to_vec(
            &case.attest(input, &q.case_outputs[i])?,
        )?);
    }
    let pins = ComponentQualificationPins {
        source_sha40: files.source_sha40.clone(),
        producer_sha256: files.program.sha256.clone(),
        qualification_sha256: q.qualification.sha256.clone(),
        reference_manifest_sha256: q.reference_manifest.sha256.clone(),
        execution_sha256: executions.iter().map(|b| sha256_hex(b)).collect(),
    };
    let qualified = context_producer::verify_component_qualification(
        &qualification,
        &manifest,
        &executions,
        &pins,
    )?;
    ensure!(
        qualified.encoder() == &artifact.reference_encoder,
        "classification feature weights differ from qualified components"
    );
    let training = TrainingIdentity {
        prepared_sha256: expected_input.identity.prepared_sha256.clone(),
        training_sha256: expected_input.identity.training_sha256.clone(),
        selected_training_rows_sha256: expected_input.identity.fitting_rows_sha256.clone(),
        preprocessing_sha256: artifact.preprocessing_sha256.clone(),
        cut: expected_input.identity.cut as usize,
    };
    let context = classification_context(
        training.clone(),
        &expected_input
            .rows
            .iter()
            .map(|r| r.id.clone())
            .collect::<Vec<_>>(),
        &artifact.hyperion,
        &artifact.projected_foundation,
        &artifact.row_encodings,
        &expected_input
            .rows
            .iter()
            .map(|r| r.label.clone())
            .collect::<Vec<_>>(),
    )?;
    Ok(VerifiedExperimentalContext {
        input: expected_input.clone(),
        context,
        training,
        source_sha40: files.source_sha40.clone(),
        producer_sha256: files.program.sha256.clone(),
        artifact_sha256: files.artifact.sha256.clone(),
        qualification_sha256: q.qualification.sha256.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use context_producer::{Feature, FeatureKind, TrainingRow};
    #[test]
    fn experimental_context_has_bitwise_label_renaming_invariance_for_both_tasks() {
        for (task, k) in [
            (ClassificationTask::Binary, 2),
            (ClassificationTask::Multiclass, 3),
        ] {
            let rows = (0..12)
                .map(|i| TrainingRow {
                    id: format!("id{i}"),
                    values: vec![Some(i.to_string())],
                    label: format!("class{}", i % k),
                })
                .collect::<Vec<_>>();
            let mut input = ClassificationInput {
                schema: context_producer::classification::CLASSIFICATION_INPUT_SCHEMA.into(),
                task,
                classes: (0..k).map(|i| format!("class{i}")).collect(),
                split_sha256: "c".repeat(64),
                identity: context_producer::TrainingIdentity {
                    dataset_sha256: "a".repeat(64),
                    prepared_sha256: "b".repeat(64),
                    training_sha256: context_producer::training_rows_sha256(&rows).unwrap(),
                    fitting_rows_sha256: sha256_hex(
                        &serde_json::to_vec(&rows.iter().map(|r| &r.id).collect::<Vec<_>>())
                            .unwrap(),
                    ),
                    cut: 0,
                },
                features: vec![Feature {
                    name: "x".into(),
                    kind: FeatureKind::Numeric,
                }],
                rows,
            };
            let original = input.prepare().unwrap();
            // Explicit hand-authored component fixture, never an admitted model output.
            let row_vectors = (0..12)
                .map(|i| {
                    (0..128)
                        .map(|j| (i as f32 - j as f32) / 128.)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let assemble = |input: &ClassificationInput| {
                let p = input.prepare().unwrap();
                classification_context(
                    TrainingIdentity {
                        prepared_sha256: input.identity.prepared_sha256.clone(),
                        training_sha256: input.identity.training_sha256.clone(),
                        selected_training_rows_sha256: input.identity.fitting_rows_sha256.clone(),
                        preprocessing_sha256: sha256_hex(
                            &serde_json::to_vec(&p.preprocessing).unwrap(),
                        ),
                        cut: 0,
                    },
                    &input.rows.iter().map(|r| r.id.clone()).collect::<Vec<_>>(),
                    &vec![0.25; 512],
                    &vec![-0.5; 1024],
                    &row_vectors,
                    &input
                        .rows
                        .iter()
                        .map(|r| r.label.clone())
                        .collect::<Vec<_>>(),
                )
                .unwrap()
            };
            let before = assemble(&input);
            for row in &mut input.rows {
                row.label = format!("renamed:{}", row.label);
            }
            for class in &mut input.classes {
                *class = format!("renamed:{class}");
            }
            input.identity.training_sha256 =
                context_producer::training_rows_sha256(&input.rows).unwrap();
            let after = assemble(&input);
            assert_eq!(
                serde_json::to_vec(&original.rows).unwrap(),
                serde_json::to_vec(&input.prepare().unwrap().rows).unwrap()
            );
            assert_eq!(
                before
                    .values()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                after
                    .values()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(before.sha256(), after.sha256());
            let metadata = serde_json::to_value(after).unwrap();
            assert_eq!(metadata["policy_compatible"], false);
            assert_eq!(
                metadata["lineage"],
                crate::search_context::CLASSIFICATION_LINEAGE
            );
        }
    }
}
