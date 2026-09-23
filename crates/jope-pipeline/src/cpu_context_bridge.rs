//! Actual ExecutionPool receipt admission for the separately qualified CPU
//! encoder. Pure policy receipt validation never substitutes for these checks.
use crate::{
    execution::{Artifact, WorkerJob, WorkerOutcome},
    experiment::{EvaluationStatus, ExecutionWindow, Experiment},
    search_context::TrainingIdentity,
    sha256_hex,
};
use anyhow::{ensure, Context, Result};
use jope_policy::{
    context_producer::{self, ExecutionAttestation, VerifiedCpuContext},
    producer::QualifiedContextBundle,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionProof {
    pub job: WorkerJob,
    pub outcome: Artifact,
    pub started: Artifact,
    pub experiment: Experiment,
    pub window: Artifact,
    pub arm_window: Option<Artifact>,
}
impl ExecutionProof {
    pub fn attest(&self, input: &Artifact, output: &Artifact) -> Result<ExecutionAttestation> {
        self.outcome.verify()?;
        self.started.verify()?;
        self.window.verify()?;
        let id = self.job.identity()?;
        let journal = self
            .job
            .directory
            .parent()
            .and_then(|p| p.parent())
            .context("job is outside journal/jobs")?;
        ensure!(
            self.job.directory == journal.join("jobs").join(self.job.request.identity()?)
                && self.outcome.path == journal.join("finished").join(format!("{id}.json"))
                && self.started.path == journal.join("started").join(format!("{id}.json"))
                && self.window.path == journal.join("window.json"),
            "proof files do not occupy actual pool journal locations"
        );
        let persisted_experiment: Experiment =
            serde_json::from_slice(&std::fs::read(journal.join("experiment.json"))?)?;
        ensure!(
            persisted_experiment.identity()? == self.experiment.identity()?,
            "execution proof experiment differs from actual journal"
        );
        ensure!(
            !journal.join("cleanup").join(format!("{id}.json")).exists(),
            "context fit cleanup remains pending"
        );
        let outcome: WorkerOutcome = serde_json::from_slice(&std::fs::read(&self.outcome.path)?)?;
        let window: ExecutionWindow = serde_json::from_slice(&std::fs::read(&self.window.path)?)?;
        let started: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&self.started.path)?)?;
        ensure!(
            started["job_sha256"] == id && started["job"] == serde_json::to_value(&self.job)?,
            "started proof changed job identity"
        );
        let start = started["started_unix_ms"]
            .as_u64()
            .context("started time absent")?;
        let deadline = started["fit_deadline_unix_ms"]
            .as_u64()
            .context("fit deadline absent")?;
        ensure!(
            deadline >= start
                && deadline
                    <= start.saturating_add(self.experiment.resources.per_fit_seconds * 1000)
                && outcome
                    .consumption
                    .gate_released_unix_ms
                    .is_some_and(|v| v >= start)
                && outcome
                    .consumption
                    .observed_exit_unix_ms
                    .is_some_and(|v| v <= deadline),
            "context escaped persisted fit deadline"
        );
        let group = std::path::Path::new(
            started["cgroup"]
                .as_str()
                .context("started cgroup absent")?,
        );
        ensure!(
            group
                .file_name()
                .is_some_and(|n| n == format!("fit-{id}").as_str())
                && group.parent().and_then(|p| p.parent())
                    == Some(std::path::Path::new("/sys/fs/cgroup/system.slice")),
            "context cgroup path changed"
        );
        if group.exists() {
            let events = std::fs::read_to_string(group.join("cgroup.events"))?;
            ensure!(
                events
                    .lines()
                    .any(|line| line.split_whitespace().eq(["populated", "0"])),
                "context cgroup has surviving descendants"
            );
        }
        let mut result = execution_attestation(
            &self.job,
            &outcome,
            &self.experiment,
            &window,
            input,
            output,
        )?;
        if let Some(arm) = &self.job.arm {
            let file = self
                .arm_window
                .as_ref()
                .context("bounded arm receipt absent")?;
            file.verify()?;
            ensure!(
                file.path == journal.join("arms").join(format!("{}.json", arm.id_sha256)),
                "arm proof is outside actual journal"
            );
            let value: serde_json::Value = serde_json::from_slice(&std::fs::read(&file.path)?)?;
            let start = value["started_unix_ms"]
                .as_u64()
                .context("arm start absent")?;
            let deadline = value["deadline_unix_ms"]
                .as_u64()
                .context("arm deadline absent")?;
            ensure!(
                value["id_sha256"] == arm.id_sha256
                    && value["seconds"] == arm.seconds
                    && start.checked_add(arm.seconds * 1000) == Some(deadline),
                "arm receipt changed original allowance"
            );
            ensure!(
                outcome
                    .consumption
                    .gate_released_unix_ms
                    .is_some_and(|v| v >= start)
                    && outcome
                        .consumption
                        .observed_exit_unix_ms
                        .is_some_and(|v| v <= deadline),
                "encoder escaped original arm window"
            );
        }
        result.supervisor_receipt_sha256 = self.outcome.sha256.clone();
        Ok(result)
    }
}
/// Call only with the actual request, immutable pool result and its original
/// window. This helper verifies successful bounded execution, never an output's
/// self-reported origin or a boolean copied from a context artifact.
pub fn execution_attestation(
    job: &WorkerJob,
    outcome: &WorkerOutcome,
    experiment: &Experiment,
    window: &ExecutionWindow,
    input: &Artifact,
    output: &Artifact,
) -> Result<ExecutionAttestation> {
    experiment.validate()?;
    window.validate(experiment)?;
    ensure!(
        experiment.resources.per_fit_seconds <= 30 && experiment.resources.threads_per_worker == 1,
        "context job allowance differs from pool fit limit"
    );
    ensure!(
        job.request.experiment_sha256 == experiment.identity()?
            && outcome.job_sha256 == job.identity()?
            && outcome.request_sha256 == job.request.identity()?
            && outcome.command_status == EvaluationStatus::Completed
            && outcome.exit_code == Some(0)
            && outcome.signal.is_none()
            && outcome.reason.is_none(),
        "context execution identity/status changed"
    );
    job.command.program.verify()?;
    for artifact in &job.command.inputs {
        artifact.verify()?;
    }
    outcome.verify_outputs()?;
    input.verify()?;
    output.verify()?;
    ensure!(
        job.command
            .inputs
            .iter()
            .any(|a| a.path == input.path && a.sha256 == input.sha256)
            && outcome
                .outputs
                .iter()
                .any(|a| a.path == output.path && a.sha256 == output.sha256)
            && job
                .outputs
                .iter()
                .any(|p| job.directory.join(p) == output.path),
        "context execution does not bind expected input/output files"
    );
    let started = outcome
        .consumption
        .gate_released_unix_ms
        .context("context gate time unknown")?;
    let exited = outcome
        .consumption
        .observed_exit_unix_ms
        .context("context exit time unknown")?;
    let wall = outcome
        .consumption
        .wall_seconds
        .context("context wall time unknown")?;
    let cpu = outcome
        .consumption
        .cpu_seconds
        .context("context CPU time unknown")?;
    ensure!(
        started >= window.started_unix_ms
            && exited >= started
            && exited <= window.deadline_unix_ms - experiment.resources.reporting_seconds * 1000
            && wall.is_finite()
            && wall >= 0.
            && wall <= experiment.resources.per_fit_seconds as f64
            && cpu.is_finite()
            && cpu >= 0.,
        "context execution escaped original window or lacks finite accounting"
    );
    ensure!(
        outcome
            .consumption
            .cgroup_peak_memory_bytes
            .is_some_and(|n| n > 0 && n <= experiment.resources.memory_bytes),
        "context cgroup memory accounting absent/outside limit"
    );
    Ok(ExecutionAttestation {
        schema: "jope.cpu-context-execution/controller-v1".into(),
        supervisor_receipt_sha256: sha256_hex(&serde_json::to_vec(outcome)?),
        program_sha256: job.command.program.sha256.clone(),
        input_sha256: input.sha256.clone(),
        output_sha256: output.sha256.clone(),
        completed: true,
        cpu_limit: 1,
        memory_max: experiment.resources.memory_bytes,
        swap_max: experiment.resources.swap_bytes,
        threads: 1,
        elapsed_seconds: wall,
        deadline_seconds: experiment.resources.per_fit_seconds as f64,
    })
}

pub struct CpuContextFiles {
    pub vector: Artifact,
    pub artifact: Artifact,
    pub frozen_transforms: Artifact,
    pub bundle: Artifact,
    pub context_execution: ExecutionProof,
    pub case_executions: Vec<ExecutionProof>,
    pub training: TrainingIdentity,
}
/// These expected identities come from the frozen caller input plan or training
/// index. None is chosen from the context bundle being admitted.
pub struct AdmissionExpectation {
    pub source_sha40: String,
    pub producer_sha256: String,
    pub input_sha256: String,
    pub qualification_sha256: String,
    pub reference_manifest_sha256: String,
    pub dataset_sha256: String,
    pub training: TrainingIdentity,
}
pub fn load_qualified(
    files: &CpuContextFiles,
    expected: &AdmissionExpectation,
) -> Result<VerifiedCpuContext> {
    files.bundle.verify()?;
    files.artifact.verify()?;
    files.vector.verify()?;
    files.frozen_transforms.verify()?;
    ensure!(
        files.frozen_transforms.sha256 == context_producer::TRANSFORMS_SHA256
            && files.training == expected.training,
        "CPU context transform/training closure changed"
    );
    let bundle: QualifiedContextBundle =
        serde_json::from_slice(&std::fs::read(&files.bundle.path)?)?;
    let pins = &bundle.pins;
    ensure!(
        pins.source_sha256 == expected.source_sha40
            && pins.producer_sha256 == expected.producer_sha256
            && pins.input_sha256 == expected.input_sha256
            && pins.qualification_sha256 == expected.qualification_sha256
            && pins.reference_manifest_sha256 == expected.reference_manifest_sha256
            && pins.artifact_sha256 == files.artifact.sha256
            && pins.training.dataset_sha256 == expected.dataset_sha256
            && pins.training.prepared_sha256 == expected.training.prepared_sha256
            && pins.training.training_sha256 == expected.training.training_sha256
            && pins.training.fitting_rows_sha256 == expected.training.selected_training_rows_sha256
            && pins.training.cut as usize == expected.training.cut
            && pins.preprocessing_sha256 == expected.training.preprocessing_sha256,
        "CPU context bundle chose different input/qualification/source pins"
    );
    ensure!(
        bundle.artifact == std::fs::read(&files.artifact.path)?,
        "CPU context bundle differs from actual produced artifact"
    );
    let input = files
        .context_execution
        .job
        .command
        .inputs
        .iter()
        .find(|a| a.sha256 == expected.input_sha256)
        .context("encoder input absent from actual job")?;
    let context_attestation = files.context_execution.attest(input, &files.artifact)?;
    ensure!(
        serde_json::to_vec(&context_attestation)? == bundle.context_execution,
        "context normalized attestation differs from actual pool receipt"
    );
    ensure!(
        files.case_executions.len() == 8 && bundle.case_executions.len() == 8,
        "missing actual qualification executions"
    );
    let inputs = context_producer::case_inputs(
        &bundle.reference_manifest,
        &expected.reference_manifest_sha256,
    )?;
    let qualification: context_producer::Qualification =
        serde_json::from_slice(&bundle.qualification)?;
    for (i, proof) in files.case_executions.iter().enumerate() {
        ensure!(
            proof.job.command.program.sha256 == expected.producer_sha256,
            "qualification executable changed"
        );
        let input_sha = sha256_hex(&inputs[i]);
        let input = proof
            .job
            .command
            .inputs
            .iter()
            .find(|a| a.sha256 == input_sha)
            .context("native case absent from actual qualification job")?;
        proof.outcome.verify()?;
        let outcome: WorkerOutcome = serde_json::from_slice(&std::fs::read(&proof.outcome.path)?)?;
        let output_sha = sha256_hex(
            qualification
                .case_outputs
                .get(i)
                .context("missing produced case")?,
        );
        let output = outcome
            .outputs
            .iter()
            .find(|a| a.sha256 == output_sha)
            .context("actual qualification case output absent")?;
        let attestation = proof.attest(input, output)?;
        ensure!(
            serde_json::to_vec(&attestation)? == bundle.case_executions[i],
            "case normalized attestation differs from actual pool receipt"
        );
    }
    let verified = bundle.verify()?;
    let vector = verified
        .context()
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<_>>();
    ensure!(
        sha256_hex(&vector) == files.vector.sha256 && vector == std::fs::read(&files.vector.path)?,
        "policy vector differs from qualified CPU output"
    );
    Ok(verified)
}
