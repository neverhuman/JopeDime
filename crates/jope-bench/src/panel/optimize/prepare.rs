use super::*;

/// Shared native regression producer; this returns qualified context files only.
/// Accepted and experimental policy consumers retain separate admission routes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_regression_context<B: Backend, V: Variation>(
    backend: &mut B,
    plan: &PairInputPlan,
    variation: &V,
    config: &Configuration,
    input: &TrainingInput,
    input_file: &Artifact,
    experiment: &Experiment,
    journal: &Path,
    worker_builds: &[Artifact],
) -> Result<(CpuContextFiles, WorkerOutcome)> {
    let qualification = config
        .qualification
        .as_ref()
        .context("native regression qualification absent")?;
    let request = diagnostic(plan, "asm.fresh-training-context/v1", &input_file.sha256)?;
    let directory = journal.join("jobs").join(request.identity()?);
    let invocation = context_producer::worker::WorkerInvocation {
        mode: context_producer::worker::WorkerMode::Produce,
        input: context_producer::AssetFile {
            path: input_file.path.clone(),
            sha256: input_file.sha256.clone(),
        },
        assets: context_producer::AssetFile {
            path: config.encoder_assets.configuration.path.clone(),
            sha256: config.encoder_assets.configuration.sha256.clone(),
        },
        source_sha40: config.source_sha40.clone(),
        producer_sha256: config.encoder.program.sha256.clone(),
        output: directory.join("output.json"),
    };
    let mut inputs = config.encoder.inputs.clone();
    inputs.extend_from_slice(worker_builds);
    inputs.push(config.release_manifest.clone());
    inputs.extend(config.encoder_assets.files.clone());
    inputs.extend([
        input_file.clone(),
        config.encoder_assets.configuration.clone(),
        config.source_manifest.clone(),
    ]);
    let job = WorkerJob {
        request,
        command: WorkerCommand {
            program: config.encoder.program.clone(),
            args: invocation.arguments(),
            inputs,
            environment: config.encoder.environment.clone(),
        },
        directory,
        outputs: vec!["output.json".into()],
        arm: Some(plan.arm(true)?),
    };
    let outcome = neural_search::encode_for_pair(backend, plan, variation, job.clone())?;
    ensure!(
        outcome.command_status == EvaluationStatus::Completed,
        "fresh encoding did not complete: {:?}: {:?}",
        outcome.command_status,
        outcome.reason
    );
    let output = outcome
        .outputs
        .iter()
        .find(|a| a.path == job.directory.join("output.json"))
        .context("fresh context output absent")?
        .clone();
    let context_proof = proof(experiment, journal, job)?;
    let context_execution = serde_json::to_vec(&context_proof.attest(input_file, &output)?)?;
    let manifest = fs::read(&qualification.reference_manifest.path)?;
    // The same case-execution attestations the producer registration a teaching freeze
    // preregisters pins (`CompletedQualification::producer_registration`).
    let executions = qualification.case_execution_attestations()?;
    let bundle = QualifiedContextBundle {
        artifact: fs::read(&output.path)?,
        qualification: fs::read(&qualification.qualification.path)?,
        reference_manifest: manifest,
        pins: context_producer::AdmissionPins {
            source_sha256: config.source_sha40.clone(),
            producer_sha256: config.encoder.program.sha256.clone(),
            artifact_sha256: output.sha256.clone(),
            qualification_sha256: qualification.qualification.sha256.clone(),
            reference_manifest_sha256: qualification.reference_manifest.sha256.clone(),
            context_execution_sha256: jope_pipeline::sha256_hex(&context_execution),
            qualification_execution_sha256: executions
                .iter()
                .map(|b| jope_pipeline::sha256_hex(b))
                .collect(),
            verification_policy_sha256: context_producer::verification_policy_sha256(),
            training: input.identity.clone(),
            input_sha256: input_file.sha256.clone(),
            preprocessing_sha256: plan.preprocessing_sha256.clone(),
        },
        context_execution,
        case_executions: executions,
    };
    let admitted = bundle.verify()?;
    let context_directory = journal.join("asm-contexts").join(plan.arm(true)?.id_sha256);
    let vector = immutable_bytes(
        &context_directory.join("context.f32"),
        &admitted
            .context()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let bundle_file = immutable(&context_directory.join("bundle.json"), &bundle)?;
    let files = CpuContextFiles {
        vector,
        artifact: output,
        frozen_transforms: config.frozen_transforms.clone(),
        bundle: bundle_file,
        context_execution: context_proof,
        case_executions: qualification.case_executions.clone(),
        training: TrainingIdentity {
            prepared_sha256: plan.prepared_sha256.clone(),
            training_sha256: plan.training_sha256.clone(),
            selected_training_rows_sha256: plan.selection.outer().fitting_rows_sha256.clone(),
            preprocessing_sha256: plan.preprocessing_sha256.clone(),
            cut: plan.cut,
        },
    };
    // Durable input to train prepare-request; callers need not reconstruct opaque proof pins.
    immutable(&context_directory.join("context-files.json"), &files)?;
    // W3-T02: the run also hands the qualified context off to teacher admission. Teaching
    // admission retains this record with the run's pool receipts (`teaching::admit_dataset`).
    let transforms =
        context_producer::FrozenTransforms::from_bytes(&config.frozen_transforms.read_verified()?)?;
    let handoff = admitted.handoff(
        &admitted.replay_assembly(&transforms)?,
        context_producer::ContextConsumer::TeacherAdmission,
    )?;
    immutable_bytes(
        &context_directory.join(TEACHER_HANDOFF),
        &serde_json::to_vec(&handoff)?,
    )?;
    Ok((files, outcome))
}
