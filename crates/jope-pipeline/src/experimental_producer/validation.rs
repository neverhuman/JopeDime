mod candidates;
pub(super) use candidates::check_candidates;

use super::{
    digest, hash, read, same, AdmittedPopulation, Artifact, CandidateClaim, Claims, Evidence,
    ExecutionProof, GenericBank, Request, RequestPlan, SynthesizedRecipe, Variation,
    VerifiedProposalContext, Worker, LINEAGE, NUMERICS, OUTPUT_SCHEMA, SELECTION,
};
use anyhow::{ensure, Context, Result};
use feat_math::{
    advanced_v3::{validate_v3_genome, Objective, TaskSpec},
    GenomeIr,
};
use jope_datasets::catalog::Task;
use serde::Deserialize;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportManifest {
    schema: String,
    lineage: String,
    serving_eligible: bool,
    artifacts: Vec<ExportFile>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportFile {
    path: PathBuf,
    sha256: String,
}
/// Only the model/vocabulary plus plan and verification metadata are read here.
/// Corpus roots and other categories' checkpoints never enter the proposal job.
pub fn export_metadata(plan: &RequestPlan) -> Result<Vec<Artifact>> {
    let manifest: ExportManifest = read(&plan.export_manifest)?;
    ensure!(
        manifest.schema == "jope.experimental-training-export/v1"
            && manifest.lineage == "asm.experimental-python-policy-v3/v1"
            && !manifest.serving_eligible,
        "not an experimental export"
    );
    let root = plan
        .export_manifest
        .path
        .parent()
        .context("export parent absent")?;
    let mut files = BTreeMap::new();
    for file in manifest.artifacts {
        ensure!(
            !file.path.as_os_str().is_empty()
                && file
                    .path
                    .components()
                    .all(|c| matches!(c, std::path::Component::Normal(_)))
                && hash(&file.sha256),
            "invalid export entry"
        );
        ensure!(
            files
                .insert(
                    file.path.clone(),
                    Artifact {
                        path: root.join(file.path),
                        sha256: file.sha256
                    }
                )
                .is_none(),
            "duplicate export path"
        );
    }
    ensure!(
        same(
            files
                .get(&PathBuf::from(format!("{}/policy.pt", plan.task.as_str())))
                .context("export checkpoint absent")?,
            &plan.checkpoint
        )?,
        "checkpoint differs from export"
    );
    ensure!(
        files.iter().any(|(p, a)| p
            .to_string_lossy()
            .starts_with(&format!("{}/vocabulary.", plan.task.as_str()))
            && same(a, &plan.vocabulary).unwrap_or(false)),
        "vocabulary differs from export"
    );
    let training = files
        .get(Path::new("plan.json"))
        .context("training plan absent")?
        .clone();
    let verification = files
        .get(Path::new("verification.json"))
        .context("training verification absent")?
        .clone();
    let value: serde_json::Value = read(&training)?;
    let proof: serde_json::Value = read(&verification)?;
    ensure!(
        value["schema"] == "jope.training-plan/v1"
            && value["lineage"] == "asm.experimental-python-policy-v3/v1"
            && proof["schema"] == "jope.training-verification/v1"
            && proof["lineage"] == value["lineage"]
            && proof["plan_sha256"] == value["plan_sha256"]
            && proof["completed_plan"] == true
            && value["source_sha"] == plan.canonicalizer_source_sha40,
        "incomplete/mismatched training verification"
    );
    ensure!(
        value["category_inputs"][plan.task.as_str()]["inputs"]["vocabulary"]["sha256"]
            == plan.vocabulary.sha256,
        "trained vocabulary changed"
    );
    let records = proof["tasks"]
        .as_array()
        .context("training tasks absent")?
        .iter()
        .filter(|r| r["task"] == plan.task.as_str())
        .collect::<Vec<_>>();
    ensure!(
        records.len() == 1
            && records[0]["planned_passes"]
                .as_u64()
                .is_some_and(|n| n >= 2)
            && records[0]["completed_passes"].as_u64() >= records[0]["planned_passes"].as_u64(),
        "missing optimizer continuation proof"
    );
    let mut metadata = vec![training, verification];
    if plan.task == Task::Regression {
        ensure!(
            value["category_inputs"]["regression"]["data_kind"] == "admitted",
            "experimental regression consumption requires admitted real training"
        );
        let encoder = files
            .iter()
            .find(|(path, _)| path.to_string_lossy().starts_with("regression/encoder."))
            .map(|(_, file)| file.clone())
            .context("trained regression encoder absent")?;
        ensure!(
            value["category_inputs"]["regression"]["inputs"]["encoder"]["sha256"] == encoder.sha256,
            "trained regression encoder identity changed"
        );
        metadata.push(encoder);
    }
    Ok(metadata)
}

pub fn verify_inputs<V: Variation>(
    plan: &RequestPlan,
    worker: &Worker,
    context: &VerifiedProposalContext,
    variation: &mut V,
) -> Result<GenericBank> {
    plan.validate()?;
    context.require_task(plan.task)?;
    plan.runtime.validate()?;
    worker.context_cost.validate()?;
    ensure!(
        same(context.training(), &plan.training)?,
        "qualified context training rows differ"
    );
    ensure!(
        variation.generic_policy() == plan.generic_policy,
        "experimental generic policy changed"
    );
    let expected_bank = generic_bank(plan.task, plan.generic_seed, plan.generic_draws, variation)?;
    ensure!(
        digest(&expected_bank)? == plan.generic_bank_sha256,
        "generic bank differs from deterministic draw policy"
    );
    let bank: GenericBank = read(&worker.request.generic_bank)?;
    ensure!(
        same(&bank, &expected_bank)?,
        "worker bank changed original seeds/programs"
    );
    let expected = plan.request(
        worker.request.context.clone(),
        worker.request.generic_bank.clone(),
    )?;
    ensure!(
        same(&worker.request, &expected)?,
        "experimental worker request differs from frozen plan"
    );
    let actual: Request = read(&worker.request_artifact)?;
    ensure!(
        same(&actual, &worker.request)?,
        "request artifact differs from declared request"
    );
    let actual_context: serde_json::Value = read(&worker.request.context)?;
    ensure!(
        actual_context == *context.artifact(),
        "context artifact differs from independently qualified native context"
    );
    ensure!(
        worker.source_manifest.sha256 == plan.worker_source_sha256,
        "experimental source manifest changed"
    );
    ensure!(
        same(&worker.job.command.program, &plan.runtime.program)?
            && worker.job.command.args
                == plan.runtime.argv(
                    &worker.request_artifact,
                    &worker.job.directory.join(&worker.output_name)
                )?
            && worker.job.command.environment == plan.runtime.environment,
        "Python invocation differs from frozen physical interpreter/source environment"
    );
    ensure!(
        worker.job.request.task == plan.task
            && worker.job.request.fitting_rows_sha256
                == plan.training.selected_training_rows_sha256
            && worker.job.request.fold as usize == plan.training.cut,
        "proposal diagnostic request changes task/training rows"
    );
    let metadata = export_metadata(plan)?;
    if plan.task == Task::Regression {
        let trained_encoder: jope_policy::context_producer::EncoderIdentity = read(&metadata[2])?;
        ensure!(
            context.regression_encoder.as_ref() == Some(&trained_encoder),
            "proposal encoder differs from the trained regression context encoder"
        );
    }
    let mut required = plan.dependencies();
    required.extend([
        &worker.request_artifact,
        &worker.request.context,
        &worker.request.generic_bank,
        &worker.source_manifest,
    ]);
    required.extend(metadata.iter());
    for artifact in required {
        artifact.verify()?;
        ensure!(
            worker
                .job
                .command
                .inputs
                .iter()
                .any(|a| same(a, artifact).unwrap_or(false)),
            "experimental dependency absent from pinned job closure"
        );
    }
    ensure!(
        worker.output_name == Path::new("proposals.json")
            && [
                "proposals.json",
                "canonicalization-input.json",
                "canonicalization-output.json",
                "canonicalizer.stdout",
                "canonicalizer.stderr"
            ]
            .iter()
            .all(|p| worker.job.outputs.contains(&PathBuf::from(p))),
        "experimental output inventory incomplete"
    );
    Ok(bank)
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Canonicalization {
    pub(super) schema: String,
    pub(super) input_sha256: String,
    pub(super) results: Vec<CanonicalResult>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CanonicalResult {
    pub(super) index: usize,
    pub(super) original_bytes_sha256: String,
    pub(super) recipe: Option<SynthesizedRecipe>,
    pub(super) error: Option<String>,
}
fn task_spec(task: Task) -> Result<TaskSpec> {
    match task {
        Task::Regression => TaskSpec::regression(Objective::Rmse),
        Task::Binary => TaskSpec::binary(Objective::RocAuc),
        Task::Multiclass => TaskSpec::multiclass(Objective::MacroF1),
        _ => anyhow::bail!("experimental tabular task required"),
    }
}
pub(super) fn validate_recipe(recipe: &SynthesizedRecipe, plan: &RequestPlan) -> Result<()> {
    recipe.validate()?;
    validate_v3_genome(
        &GenomeIr::parse(&recipe.executable_canonical)?,
        &task_spec(plan.task)?,
        &plan.data_schema,
    )?;
    Ok(())
}
/// Structural claims checking is private and cannot create admission. The sole
/// public constructor below also requires qualified context and actual pool proof.
fn check_claims(
    plan: &RequestPlan,
    worker: &Worker,
    context: &VerifiedProposalContext,
    bank: &GenericBank,
    claims: &Claims,
    canonicalization: &Canonicalization,
) -> Result<()> {
    ensure!(
        claims.schema == OUTPUT_SCHEMA
            && claims.lineage == LINEAGE
            && !claims.serving_eligible
            && claims.backend == "torch-cpu"
            && claims.numerical_policy == NUMERICS
            && claims.selection_policy == SELECTION,
        "experimental output lineage/numerics changed"
    );
    ensure!(
        claims.request_sha256 == worker.request_artifact.sha256
            && hash(&claims.request_identity_sha256)
            && claims.task == plan.task
            && same(&claims.training, &plan.training)?
            && claims.checkpoint_sha256 == plan.checkpoint.sha256
            && claims.export_manifest_sha256 == plan.export_manifest.sha256
            && claims.vocabulary_sha256 == plan.vocabulary.sha256
            && claims.context_artifact_sha256 == worker.request.context.sha256
            && claims.context_sha256 == context.sha256()
            && claims.context_lineage == context.lineage
            && claims.runtime_sha256 == plan.runtime.runtime_sha256
            && claims.neural_seed == plan.neural_seed
            && same(&claims.sampling, &plan.sampling)?
            && claims.generic_bank_sha256 == plan.generic_bank_sha256
            && claims.generic_policy == plan.generic_policy
            && claims.generic_seed == plan.generic_seed
            && same(&claims.canonicalizer, &plan.canonicalizer)?
            && claims.canonicalizer_source_sha40 == plan.canonicalizer_source_sha40,
        "experimental output input binding changed"
    );
    for value in [
        &claims.export_verification_sha256,
        &claims.training_proof_sha256,
        &claims.parameter_sha256,
        &claims.canonicalization_sha256,
    ] {
        ensure!(hash(value), "invalid experimental proof digest");
    }
    let metadata = export_metadata(plan)?;
    ensure!(
        claims.export_verification_sha256 == metadata[1].sha256,
        "output training proof differs from export"
    );
    let verification: serde_json::Value = read(&metadata[1])?;
    let record = verification["tasks"]
        .as_array()
        .context("export tasks")?
        .iter()
        .find(|r| r["task"] == plan.task.as_str())
        .context("export task")?;
    ensure!(
        record["parameter_sha256"] == claims.parameter_sha256,
        "actual checkpoint parameter proof differs from export"
    );
    let training: serde_json::Value = read(&metadata[0])?;
    ensure!(
        training["category_inputs"][plan.task.as_str()]["data_kind"]
            == claims.checkpoint_training_data_kind
            && ["synthetic_fixture", "admitted"]
                .contains(&claims.checkpoint_training_data_kind.as_str()),
        "checkpoint training data provenance changed"
    );
    check_candidates(
        plan,
        &claims.candidates,
        claims.neural_candidates,
        bank,
        canonicalization,
    )
}

pub fn admit<V: Variation>(
    pair: &crate::neural_search::PairInputPlan,
    worker: &Worker,
    context: &VerifiedProposalContext,
    proof: &ExecutionProof,
    variation: &mut V,
) -> Result<AdmittedPopulation> {
    let crate::neural_search::ProposalPlan::Experimental(plan) = &pair.proposal else {
        anyhow::bail!("accepted plan cannot enter experimental admission")
    };
    let arm = pair.arm(true)?;
    ensure!(
        worker
            .job
            .arm
            .as_ref()
            .is_some_and(|a| a.id_sha256 == arm.id_sha256 && a.seconds == arm.seconds),
        "experimental execution changed original arm"
    );
    let diagnostic = crate::neural_search::diagnostic_request(
        pair,
        "asm.experimental-python-proposals/v1",
        &worker.request_artifact.sha256,
    )?;
    ensure!(
        same(&worker.job.request, &diagnostic)?,
        "experimental diagnostic request changed comparison identity"
    );
    let bank = verify_inputs(plan, worker, context, variation)?;
    ensure!(
        same(&proof.job, &worker.job)?,
        "proposal execution proof changed pinned job"
    );
    let output_path = worker.job.directory.join(&worker.output_name);
    let outcome: crate::execution::WorkerOutcome = read(&proof.outcome)?;
    let output = outcome
        .outputs
        .iter()
        .find(|a| a.path == output_path)
        .context("proposal output absent from actual execution")?;
    let attestation = proof.attest(&worker.request_artifact, output)?;
    ensure!(
        attestation.program_sha256 == plan.runtime.program.sha256
            && attestation.threads == 1
            && attestation.cpu_limit == 1,
        "proposal process differs from bounded CPU policy"
    );
    let claims: Claims = read(output)?;
    let canonical_output = outcome
        .outputs
        .iter()
        .find(|a| a.path == worker.job.directory.join("canonicalization-output.json"))
        .context("native batch output absent")?;
    let canonical_input = outcome
        .outputs
        .iter()
        .find(|a| a.path == worker.job.directory.join("canonicalization-input.json"))
        .context("native batch input absent")?;
    let canonicalization: Canonicalization = read(canonical_output)?;
    ensure!(
        canonicalization.schema == "jope.experimental-canonicalization-response/v1"
            && canonicalization.input_sha256 == canonical_input.sha256
            && canonical_output.sha256 == claims.canonicalization_sha256,
        "native canonicalization output binding changed"
    );
    check_claims(plan, worker, context, &bank, &claims, &canonicalization)?;
    let evidence = Evidence {
        schema: "jope.experimental-proposal-evidence/v1".into(),
        pair_sha256: digest(pair)?,
        request: worker.request.clone(),
        context_cost: worker.context_cost.clone(),
        producer_job_sha256: outcome.job_sha256,
        producer_output_sha256: output.sha256.clone(),
        population_sha256: digest(&claims.candidates)?,
        context_sha256: context.sha256().into(),
        checkpoint_sha256: plan.checkpoint.sha256.clone(),
        ranking_backend: NUMERICS.into(),
        export_verification_sha256: claims.export_verification_sha256,
        training_proof_sha256: claims.training_proof_sha256,
        parameter_sha256: claims.parameter_sha256,
        checkpoint_training_data_kind: claims.checkpoint_training_data_kind,
        execution_proof: proof.clone(),
    };
    Ok(AdmittedPopulation {
        candidates: claims.candidates,
        evidence,
    })
}
