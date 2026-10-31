//! Controller bridge for the distinct experimental classifier proposal worker.
//! It shares the admitted pool, original arm and fitted OOF adapter with ASM.
use super::{
    data::DevelopmentData,
    optimize::{self, Configuration},
    optimize_fit::{artifact, digest, immutable},
};
use anyhow::{ensure, Context, Result};
use jope_datasets::catalog::Task;
use jope_pipeline::{
    classification_bridge::{self, ClassificationFiles},
    context_qualification::CompletedQualification,
    cpu_context_bridge::{AdmissionExpectation, CpuContextFiles, ExecutionProof},
    execution::{Artifact, WorkerCommand, WorkerJob},
    experiment::{EvaluationStatus, Experiment},
    experimental_producer::{self, PythonRuntime, Sampling, VerifiedProposalContext},
    neural_search::{
        self, Backend, ContextCost, PairInputPlan, ProposalPlan, SearchReceipt, Variation,
    },
    search_context::TrainingIdentity,
};
use jope_policy::context_producer::{
    classification::{ClassificationInput, ClassificationTask},
    TrainingInput,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

pub const SCHEMA: &str = "jope.experimental-python-configuration/v1";
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentalConfiguration {
    pub schema: String,
    pub task: Task,
    pub runtime: PythonRuntime,
    pub export_manifest: Artifact,
    pub checkpoint: Artifact,
    pub vocabulary: Artifact,
    pub canonicalizer: Artifact,
    pub canonicalizer_build: Artifact,
    pub canonicalizer_features: Vec<String>,
    pub sampling: Sampling,
    pub generic_draws: usize,
}
impl ExperimentalConfiguration {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SCHEMA
                && matches!(
                    self.task,
                    Task::Regression | Task::Binary | Task::Multiclass
                ),
            "experimental configuration requires explicit supported task"
        );
        self.sampling.validate()?;
        ensure!(
            (80..=4096 - self.sampling.count).contains(&self.generic_draws),
            "experimental generic bank allowance outside bounds"
        );
        for artifact in self.dependencies() {
            ensure!(
                artifact.path.is_absolute() && jope_pipeline::experiment::sha256(&artifact.sha256),
                "invalid experimental dependency pin"
            );
        }
        Ok(())
    }
    pub fn verify(&self, source_sha40: &str, source_manifest: &Artifact) -> Result<()> {
        ensure!(
            self.schema == SCHEMA
                && matches!(
                    self.task,
                    Task::Regression | Task::Binary | Task::Multiclass
                ),
            "experimental configuration requires explicit supported task"
        );
        self.sampling.validate()?;
        ensure!(
            (80..=4096 - self.sampling.count).contains(&self.generic_draws),
            "experimental bank allowance outside bounds"
        );
        self.runtime.validate()?;
        for a in [
            &self.export_manifest,
            &self.checkpoint,
            &self.vocabulary,
            &self.canonicalizer,
            &self.canonicalizer_build,
        ] {
            a.verify()?;
        }
        source_manifest.verify()?;
        let build: super::worker_build::BuildReceipt =
            experimental_producer::read(&self.canonicalizer_build)?;
        ensure!(
            build.schema == "jope.asm-cargo-build/v1"
                && build.exit_code == 0
                && build.source_sha40 == source_sha40
                && build.source_manifest_sha256 == source_manifest.sha256
                && build.started_unix_ms > 0
                && build.finished_unix_ms >= build.started_unix_ms
                && Path::new(&build.working_directory).is_absolute()
                && build.command.iter().any(|a| a == "build")
                && build
                    .command
                    .iter()
                    .any(|a| a.starts_with("--message-format=json")),
            "native canonicalizer lacks a matching successful source build"
        );
        build.cargo_messages.verify()?;
        let built = build
            .executables
            .iter()
            .filter(|e| e.target_name == "jope" && e.sha256 == self.canonicalizer.sha256)
            .collect::<Vec<_>>();
        ensure!(
            built.len() == 1,
            "canonicalizer differs from actual built CLI bytes"
        );
        let features = self
            .canonicalizer_features
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        ensure!(
            features.len() == self.canonicalizer_features.len(),
            "duplicate canonicalizer feature pin"
        );
        let mut matching = 0;
        let mut completed = false;
        for line in fs::read_to_string(&build.cargo_messages.path)?.lines() {
            let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            if value["reason"] == "build-finished" && value["success"] == true {
                completed = true;
            }
            if value["reason"] == "compiler-artifact"
                && value["target"]["name"] == "jope"
                && value["executable"].as_str() == built[0].path.to_str()
            {
                ensure!(
                    serde_json::from_value::<BTreeSet<String>>(value["features"].clone())?
                        == features,
                    "canonicalizer build features changed"
                );
                matching += 1;
            }
        }
        ensure!(
            completed && matching == 1,
            "canonicalizer build capture incomplete"
        );
        Ok(())
    }
    pub fn dependencies(&self) -> Vec<&Artifact> {
        let mut a = self.runtime.dependencies();
        a.extend([
            &self.export_manifest,
            &self.checkpoint,
            &self.vocabulary,
            &self.canonicalizer,
            &self.canonicalizer_build,
        ]);
        a
    }
}

/// Untagged serialization preserves existing regression input bytes/hashes.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ContextInput {
    Accepted(TrainingInput),
    Experimental(ClassificationInput),
}
impl ContextInput {
    pub fn accepted(&self) -> Result<&TrainingInput> {
        match self {
            Self::Accepted(input) => Ok(input),
            Self::Experimental(_) => {
                anyhow::bail!("classification input cannot enter regression producer")
            }
        }
    }
    pub fn experimental(&self) -> Result<&ClassificationInput> {
        match self {
            Self::Experimental(input) => Ok(input),
            Self::Accepted(_) => anyhow::bail!("regression input cannot enter classifier producer"),
        }
    }
}
impl From<TrainingInput> for ContextInput {
    fn from(value: TrainingInput) -> Self {
        Self::Accepted(value)
    }
}

pub fn bind_pair<V: Variation>(
    configuration: &Configuration,
    experimental: &ExperimentalConfiguration,
    dataset: &DevelopmentData,
    input: TrainingInput,
    mut pair: PairInputPlan,
    variation: &mut V,
) -> Result<(ContextInput, PairInputPlan)> {
    ensure!(
        experimental.task == dataset.dataset.task,
        "experimental checkpoint task differs from dataset"
    );
    let (context_input, preprocessing_sha256) = if dataset.dataset.task == Task::Regression {
        let preprocessing = digest(&input.prepare()?.preprocessing)?;
        (ContextInput::Accepted(input), preprocessing)
    } else {
        let class_input = ClassificationInput {
            schema: "jope.cpu-classification-training/v1".into(),
            split_sha256: dataset.dataset.split.sha256.clone(),
            task: match dataset.dataset.task {
                Task::Binary => ClassificationTask::Binary,
                Task::Multiclass => ClassificationTask::Multiclass,
                _ => anyhow::bail!("supported tabular task required"),
            },
            classes: dataset.dataset.classes.clone(),
            identity: input.identity,
            features: input.features,
            rows: input.rows,
        };
        let preprocessing = class_input.preprocessing_sha256()?;
        (ContextInput::Experimental(class_input), preprocessing)
    };
    let bank = experimental_producer::generic_bank(
        dataset.dataset.task,
        pair.random_seed.wrapping_add(1),
        experimental.generic_draws,
        variation,
    )?;
    let request = experimental_producer::RequestPlan {
        schema: experimental_producer::PLAN_SCHEMA.into(),
        task: dataset.dataset.task,
        worker_source_sha256: configuration.source_manifest.sha256.clone(),
        runtime: experimental.runtime.clone(),
        export_manifest: experimental.export_manifest.clone(),
        checkpoint: experimental.checkpoint.clone(),
        vocabulary: experimental.vocabulary.clone(),
        canonicalizer: experimental.canonicalizer.clone(),
        canonicalizer_source_sha40: configuration.source_sha40.clone(),
        training: TrainingIdentity {
            prepared_sha256: pair.prepared_sha256.clone(),
            training_sha256: pair.training_sha256.clone(),
            selected_training_rows_sha256: pair.selection.outer().fitting_rows_sha256.clone(),
            preprocessing_sha256: preprocessing_sha256.clone(),
            cut: pair.cut,
        },
        data_schema: dataset.schema.clone(),
        neural_seed: pair.random_seed,
        sampling: experimental.sampling.clone(),
        generic_bank_sha256: digest(&bank)?,
        generic_seed: bank.seed,
        generic_draws: experimental.generic_draws,
        generic_policy: bank.policy,
    };
    request.validate()?;
    pair.preprocessing_sha256 = preprocessing_sha256;
    pair.vocabulary_sha256 = experimental.vocabulary.sha256.clone();
    pair.installation.policy_program_sha256 = experimental.runtime.program.sha256.clone();
    pair.installation.checkpoint_receipt_sha256 = experimental.export_manifest.sha256.clone();
    pair.proposal = ProposalPlan::Experimental(Box::new(request));
    pair.encoding.input_sha256 = digest(&context_input)?;
    if dataset.dataset.task != Task::Regression {
        pair.encoding.backend = "asm.experimental-classification-neutral-foundation-cpu/v1".into();
    }
    Ok((context_input, pair))
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
pub enum AdmissionContext {
    Classification(ClassificationFiles),
    Regression(CpuContextFiles),
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Admission {
    pub schema: String,
    pub pair: PairInputPlan,
    pub context_files: AdmissionContext,
    pub worker: experimental_producer::Worker,
}
fn expected_plan(pair: &PairInputPlan) -> Result<&experimental_producer::RequestPlan> {
    match &pair.proposal {
        ProposalPlan::Experimental(value) => Ok(value),
        _ => anyhow::bail!("experimental input plan required"),
    }
}
fn qualification(config: &Configuration) -> Result<&CompletedQualification> {
    config.qualification.as_ref().context(
        "native component qualification absent; classifier feature producer remains unqualified",
    )
}

fn regression_expected(
    config: &Configuration,
    pair: &PairInputPlan,
    input: &TrainingInput,
) -> Result<AdmissionExpectation> {
    let q = qualification(config)?;
    input.prepare()?;
    ensure!(
        expected_plan(pair)?.task == Task::Regression
            && digest(input)? == pair.encoding.input_sha256,
        "experimental regression context input changed"
    );
    Ok(AdmissionExpectation {
        source_sha40: config.source_sha40.clone(),
        producer_sha256: config.encoder.program.sha256.clone(),
        input_sha256: pair.encoding.input_sha256.clone(),
        qualification_sha256: q.qualification.sha256.clone(),
        reference_manifest_sha256: q.reference_manifest.sha256.clone(),
        dataset_sha256: input.identity.dataset_sha256.clone(),
        training: expected_plan(pair)?.training.clone(),
    })
}
fn verify_context(
    config: &Configuration,
    pair: &PairInputPlan,
    input: &ContextInput,
    files: &AdmissionContext,
) -> Result<VerifiedProposalContext> {
    match (input, files) {
        (ContextInput::Accepted(input), AdmissionContext::Regression(files)) => {
            VerifiedProposalContext::regression(files, &regression_expected(config, pair, input)?)
        }
        (ContextInput::Experimental(input), AdmissionContext::Classification(files)) => {
            ensure!(
                files.source_sha40 == config.source_sha40
                    && experimental_producer::same(&files.program, &config.encoder.program)?
                    && experimental_producer::same(
                        &files.encoder_configuration,
                        &config.encoder_assets.configuration
                    )?
                    && experimental_producer::same(&files.qualification, qualification(config)?)?,
                "retained classifier producer chose different source/assets/qualification"
            );
            VerifiedProposalContext::classification(classification_bridge::verify_classification(
                files, input,
            )?)
        }
        _ => anyhow::bail!("experimental context kind differs from original input task"),
    }
}

/// Prepare a verified proposal worker and context from an admitted pair.
///
/// Keep these provenance inputs explicit: each identifies an independently
/// verified part of proposal admission and worker execution.
#[allow(clippy::too_many_arguments)]
pub fn prepare<B: Backend, V: Variation>(
    backend: &mut B,
    pair: &PairInputPlan,
    variation: &mut V,
    configuration: &Configuration,
    input: &ContextInput,
    input_file: &Artifact,
    experiment: &Experiment,
    journal: &Path,
) -> Result<(experimental_producer::Worker, VerifiedProposalContext)> {
    let request_plan = expected_plan(pair)?;
    let experimental = configuration
        .experimental
        .get(request_plan.task.as_str())
        .context("experimental configuration missing")?;
    experimental.verify(&configuration.source_sha40, &configuration.source_manifest)?;
    ensure!(
        configuration.encoder.unavailable.is_none(),
        "classification component worker unavailable"
    );
    let qualification = qualification(configuration)?;
    let worker_builds = configuration.verify_worker_builds()?;
    ensure!(
        configuration.encoder.environment.is_empty(),
        "classification worker requires its explicit empty environment"
    );
    let (context_files, outcome) = match input {
        ContextInput::Accepted(input) => {
            ensure!(
                request_plan.task == Task::Regression,
                "regression context cannot enter a classification proposal"
            );
            let (files, outcome) = optimize::prepare_regression_context(
                backend,
                pair,
                variation,
                configuration,
                input,
                input_file,
                experiment,
                journal,
                &worker_builds,
            )?;
            (AdmissionContext::Regression(files), outcome)
        }
        ContextInput::Experimental(_input) => {
            let request = optimize::diagnostic(
                pair,
                "asm.experimental-classification-components/v1",
                &input_file.sha256,
            )?;
            let job = classification_bridge::build_job(
                experiment,
                request,
                configuration.encoder.program.clone(),
                input_file.clone(),
                &configuration.encoder_assets,
                journal,
                Some(pair.arm(true)?),
            )?;
            // The classifier worker has its own exact feature-only closure. Scheduling,
            // cancellation and resource accounting still use the same original pool arm.
            let outcome = backend.produce(job.clone())?;
            ensure!(
                outcome.job_sha256 == job.identity()?
                    && outcome.request_sha256 == job.request.identity()?,
                "classification execution identity changed"
            );
            let inventory = serde_json::to_value(&outcome)?;
            backend.persist(
                &pair.arm(true)?,
                &format!("encoding-inventory-{}", digest(&inventory)?),
                &inventory,
            )?;
            ensure!(
                outcome.command_status == EvaluationStatus::Completed,
                "experimental encoding did not complete: {:?}: {:?}",
                outcome.command_status,
                outcome.reason
            );
            let output = outcome
                .outputs
                .iter()
                .find(|a| a.path == job.directory.join("output.json"))
                .context("classification component output absent")?
                .clone();
            let files = ClassificationFiles {
                program: configuration.encoder.program.clone(),
                source_sha40: configuration.source_sha40.clone(),
                input: input_file.clone(),
                artifact: output,
                encoder_configuration: configuration.encoder_assets.configuration.clone(),
                qualification: qualification.clone(),
                execution: optimize::proof(experiment, journal, job)?,
            };
            (AdmissionContext::Classification(files), outcome)
        }
    };
    let context = verify_context(configuration, pair, input, &context_files)?;
    ensure!(
        context.training() == &request_plan.training,
        "produced classification context changed pair training closure"
    );
    let root = journal.join("asm-contexts").join(pair.arm(true)?.id_sha256);
    let context_artifact = immutable(&root.join("experimental-context.json"), context.artifact())?;
    let bank = experimental_producer::generic_bank(
        request_plan.task,
        request_plan.generic_seed,
        request_plan.generic_draws,
        variation,
    )?;
    let bank_file = immutable(&root.join("generic-bank.json"), &bank)?;
    let producer_request = request_plan.request(context_artifact, bank_file)?;
    let request_file = immutable(&root.join("experimental-request.json"), &producer_request)?;
    let request = optimize::diagnostic(
        pair,
        "asm.experimental-python-proposals/v1",
        &request_file.sha256,
    )?;
    let directory = journal.join("jobs").join(request.identity()?);
    let output_name = PathBuf::from("proposals.json");
    let mut dependencies = request_plan
        .dependencies()
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    dependencies.extend(experimental_producer::export_metadata(request_plan)?);
    dependencies.extend([
        request_file.clone(),
        producer_request.context.clone(),
        producer_request.generic_bank.clone(),
        configuration.source_manifest.clone(),
        experimental.canonicalizer_build.clone(),
    ]);
    let worker = experimental_producer::Worker {
        job: WorkerJob {
            request,
            command: WorkerCommand {
                program: request_plan.runtime.program.clone(),
                args: request_plan
                    .runtime
                    .argv(&request_file, &directory.join(&output_name))?,
                inputs: dependencies,
                environment: request_plan.runtime.environment.clone(),
            },
            directory,
            outputs: [
                "proposals.json",
                "canonicalization-input.json",
                "canonicalization-output.json",
                "canonicalizer.stdout",
                "canonicalizer.stderr",
            ]
            .iter()
            .map(PathBuf::from)
            .collect(),
            arm: Some(pair.arm(true)?),
        },
        request: producer_request,
        request_artifact: request_file,
        source_manifest: configuration.source_manifest.clone(),
        context_cost: ContextCost {
            cached: false,
            production_receipt: match &context_files {
                AdmissionContext::Classification(files) => files.execution.outcome.clone(),
                AdmissionContext::Regression(files) => files.context_execution.outcome.clone(),
            },
            original_wall_seconds: outcome.consumption.wall_seconds,
            original_cpu_seconds: outcome.consumption.cpu_seconds,
            original_gpu_seconds: Some(0.),
            amortization_uses: Some(1),
        },
        output_name,
    };
    experimental_producer::verify_inputs(request_plan, &worker, &context, variation)?;
    immutable(
        &root.join("experimental-admission.json"),
        &Admission {
            schema: "jope.experimental-proposal-admission/v1".into(),
            pair: pair.clone(),
            context_files,
            worker: worker.clone(),
        },
    )?;
    Ok((worker, context))
}

/// Read-only replay rederives context and proposal admission from independently
/// reconstructed panel training rows plus the original pool files. No clocks reset.
pub fn verify_report<V: Variation>(
    config: &Configuration,
    input: &ContextInput,
    search: &optimize::DatasetSearch,
    receipt: &SearchReceipt,
    journal: &Path,
    variation: &mut V,
) -> Result<()> {
    let pair = &search.pair;
    let expected = expected_plan(pair)?;
    let directory = journal.join("asm-contexts").join(pair.arm(true)?.id_sha256);
    let record_file = artifact(directory.join("experimental-admission.json"))?;
    let admission: Admission = experimental_producer::read(&record_file)?;
    ensure!(
        admission.schema == "jope.experimental-proposal-admission/v1"
            && experimental_producer::same(&admission.pair, pair)?,
        "experimental admission changed original pair"
    );
    config
        .experimental
        .get(expected.task.as_str())
        .context("experimental configuration absent")?
        .verify(&config.source_sha40, &config.source_manifest)?;
    let context = verify_context(config, pair, input, &admission.context_files)?;
    let execution = search
        .receipt
        .neural
        .as_ref()
        .context("experimental neural execution absent")?;
    let job = &admission.worker.job;
    let proof = ExecutionProof {
        job: job.clone(),
        outcome: artifact(
            journal
                .join("finished")
                .join(format!("{}.json", job.identity()?)),
        )?,
        started: artifact(
            journal
                .join("started")
                .join(format!("{}.json", job.identity()?)),
        )?,
        experiment: serde_json::from_slice(&fs::read(journal.join("experiment.json"))?)?,
        window: artifact(journal.join("window.json"))?,
        arm_window: Some(artifact(
            journal
                .join("arms")
                .join(format!("{}.json", pair.arm(true)?.id_sha256)),
        )?),
    };
    let actual: jope_pipeline::execution::WorkerOutcome =
        experimental_producer::read(&proof.outcome)?;
    ensure!(
        experimental_producer::same(&actual, &execution.proposal_execution)?,
        "experimental proposal outcome differs from actual pool"
    );
    let admitted =
        experimental_producer::admit(pair, &admission.worker, &context, &proof, variation)?;
    let Some(neural_search::ProposalEvidence::Experimental(evidence)) = receipt.proposal.as_ref()
    else {
        anyhow::bail!("experimental arm lost its distinct evidence")
    };
    ensure!(
        experimental_producer::same(evidence, admitted.evidence())?,
        "experimental proposal evidence changed"
    );
    ensure!(
        receipt.observations.len() >= 80,
        "missing experimental generation-zero fits"
    );
    for (slot, (observed, seed)) in receipt.observations[..80]
        .iter()
        .zip(admitted.candidates())
        .enumerate()
    {
        observed.candidate.validate_initial(&receipt.arm)?;
        let origin = match seed.origin.as_str() {
            "neural_policy" => neural_search::Origin::NeuralPolicy,
            "neural_rejection_fallback" => neural_search::Origin::NeuralRejectionFallback,
            "random_exploration" => neural_search::Origin::RandomExploration,
            _ => anyhow::bail!("invalid experimental origin"),
        };
        ensure!(
            observed.candidate.initial_slot == Some(slot)
                && observed.candidate.genome == seed.recipe.original_proposal
                && experimental_producer::same(&observed.candidate.recipe, &seed.recipe)?
                && observed.candidate.origin == origin
                && observed.candidate.rank_log_weight == seed.rank_log_weight,
            "experimental generation zero differs from admitted actual proposals"
        );
    }
    Ok(())
}
