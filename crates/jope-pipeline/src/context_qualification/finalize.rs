use super::*;
use std::fs;

pub(super) fn finish_qualification(
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
pub(super) fn policy_artifact(a: &Artifact) -> context_producer::AssetFile {
    context_producer::AssetFile {
        path: a.path.clone(),
        sha256: a.sha256.clone(),
    }
}
pub(super) fn read_artifact(a: &Artifact, maximum: u64) -> Result<Vec<u8>> {
    policy_artifact(a).read(maximum)
}
pub(super) fn physical_directory(path: &Path) -> Result<()> {
    ensure!(
        path.is_absolute() && path.canonicalize()? == path && fs::symlink_metadata(path)?.is_dir(),
        "private custody directory must be physical and absolute"
    );
    Ok(())
}
