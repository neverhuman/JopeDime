use super::*;

/// The context the W2 producer job wrote, decoded as the producer decodes it
/// (`jope_policy::context_producer::verify_qualified`): the retained, digest-verified artifact
/// bytes parsed as `CpuContextArtifact`, held to the envelope and context layout
/// `CpuContextArtifact::validate` checks, and produced under the handoff's release and input
/// scope. Returns the context's little-endian f32 bytes.
fn produced_context(state: &State, handoff: &HandoffRecord) -> Result<Vec<u8>> {
    let bytes = retained(state, &handoff.artifact_sha256)?;
    ensure!(
        bytes.len() <= MAX_ARTIFACT_BYTES,
        "ContextHandoffMismatch: the producer artifact exceeds its bound"
    );
    let artifact: ProducedArtifact = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "ContextHandoffMismatch: the producer artifact {} is not a {PRODUCED_ARTIFACT_SCHEMA} \
             context artifact",
            handoff.artifact_sha256
        )
    })?;
    ensure!(
        artifact.schema == PRODUCED_ARTIFACT_SCHEMA
            && artifact.backend == PRODUCED_ARTIFACT_BACKEND
            && artifact.qualification == "unqualified"
            && !artifact.final_test_targets_used,
        "ContextHandoffMismatch: the producer artifact is not the W2 CPU context envelope"
    );
    let (release, scope) = (&handoff.release, &handoff.scope);
    ensure!(
        artifact.source_sha256 == release.source_sha256
            && artifact.producer_sha256 == release.producer_sha256
            && artifact.encoder == release.encoder
            && artifact.input_sha256 == scope.input_sha256
            && artifact.training == scope.training
            && artifact.selected_training_rows_sha256 == scope.training.fitting_rows_sha256
            && artifact.preprocessing_sha256 == scope.preprocessing_sha256
            && artifact.prepared_rows_sha256 == scope.prepared_rows_sha256,
        "ContextHandoffMismatch: the producer artifact was produced under another release or \
         input scope than the handoff names"
    );
    ensure!(
        artifact.vector.len() == CONTEXT_WIDTH
            && artifact
                .vector
                .iter()
                .all(|v| v.is_finite() && v.abs() <= CONTEXT_CLIP),
        "ContextHandoffMismatch: the producer artifact's context is not a finite, clipped \
         {CONTEXT_WIDTH}-f32 vector"
    );
    Ok(artifact
        .vector
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect())
}

/// Teacher admission consumes the W2 handoff instead of trusting the receipt's own claims. The
/// handoff must:
/// - exist, under a freeze that preregistered a W2 producer (otherwise the teaching context is
///   quarantined, never upgraded in place);
/// - be handed off to teacher admission, under exactly the preregistered release identity,
///   transformation chain and W2 qualification (producer authentication);
/// - hand off exactly the teacher's context bytes (`context_sha256` is the sha256 of the
///   receipt's retained output, the context's little-endian bytes), scoped to the comparison's
///   dataset;
/// - name no revoked evidence (W12-T05);
/// - have its artifact produced by a completed execution of the preregistered producer
///   executable inside the teacher's own authorized experiment and window, proven from retained
///   pool receipts ([`verify_execution`]);
/// - hand off exactly the context that artifact contains, decoded as the W2 producer decodes it
///   ([`produced_context`]): the consumed vector is the artifact's vector byte for byte, and
///   `context_sha256` is its digest;
/// - name the controller's admission pins for this run: `admission_pins_sha256` is rebuilt from
///   the preregistered release, W2 qualification and case-execution pins, the artifact, its input
///   scope and the controller's attestation of the retained context execution.
///
/// Returns the handoff digest the qualification records.
pub(super) fn verify_handoff(
    state: &State,
    policy: &FrozenPolicy,
    plan: &ContextPlan,
    record: &HwmRecord,
    context: &ContextEvidence,
    revocations: &Revocations,
    teaching: &Teaching<'_>,
) -> Result<String> {
    let (producer, case_execution_pins) = preregistered_producer(plan)?;
    let evidence = context.handoff.as_ref().with_context(|| {
        quarantined(
            "MissingContextHandoff: the submission carries no W2 context handoff to teacher \
             admission; a self-declared encoder receipt is not a produced context",
        )
    })?;
    let ex = &evidence.execution;
    revocations.refuse(
        "the W2 context handoff",
        [
            evidence.handoff_sha256.as_str(),
            ex.job_sha256.as_str(),
            ex.outcome_sha256.as_str(),
            ex.experiment_sha256.as_str(),
            ex.window_sha256.as_str(),
        ],
    )?;
    let handoff = parse_handoff(&retained(state, &evidence.handoff_sha256)?)?;
    ensure!(
        handoff.schema == HANDOFF_SCHEMA,
        "ContextHandoffMismatch: handoff schema {:?}",
        handoff.schema
    );
    ensure!(
        handoff.consumer == TEACHER_ADMISSION,
        "ContextHandoffMismatch: the context was handed off for consumer {:?}, not \
         {TEACHER_ADMISSION}",
        handoff.consumer
    );
    ensure!(
        handoff.release == producer.release,
        "ProducerNotAuthenticated: the handoff's release identity (encoder closure, source, \
         producer executable, verification policy, reference manifest) is not the one the freeze \
         preregistered"
    );
    ensure!(
        handoff.qualification_sha256 == producer.qualification_sha256,
        "ProducerNotAuthenticated: the handoff names W2 qualification {}, not the preregistered {}",
        handoff.qualification_sha256,
        producer.qualification_sha256
    );
    ensure!(
        handoff.transformations == producer.transformations,
        "ContextHandoffMismatch: the handoff's transformation chain is not the preregistered chain"
    );
    ensure!(
        handoff.context_sha256 == context.output_sha256,
        "ContextHandoffMismatch: the teacher's context {} is not the handed-off context {}",
        context.output_sha256,
        handoff.context_sha256
    );
    let t = &handoff.scope.training;
    ensure!(
        t.dataset_sha256 == record.identity.dataset_hash,
        "ContextHandoffMismatch: the handoff's input scope is dataset {}, not the comparison's {}",
        t.dataset_sha256,
        record.identity.dataset_hash
    );
    for digest in [
        &t.prepared_sha256,
        &t.training_sha256,
        &t.fitting_rows_sha256,
        &handoff.scope.input_sha256,
        &handoff.scope.preprocessing_sha256,
        &handoff.scope.prepared_rows_sha256,
        &handoff.artifact_sha256,
        &handoff.admission_pins_sha256,
    ] {
        ensure!(
            is_hex64(digest),
            "ContextHandoffMismatch: invalid handoff digest"
        );
    }
    revocations.refuse(
        "the W2 context handoff",
        [
            handoff.artifact_sha256.as_str(),
            handoff.qualification_sha256.as_str(),
        ],
    )?;
    let context_execution_sha256 = verify_execution(state, policy, &handoff, ex, record, teaching)?;
    let produced = produced_context(state, &handoff)?;
    let produced_sha256 = sha256_hex(&produced);
    ensure!(
        produced == retained(state, &context.output_sha256)?
            && produced_sha256 == handoff.context_sha256,
        "ContextHandoffMismatch: the handed-off context {} is not the context the producer \
         artifact contains ({produced_sha256})",
        handoff.context_sha256
    );
    let r = &producer.release;
    let pins = AdmissionPinsRecord {
        source_sha256: &r.source_sha256,
        producer_sha256: &r.producer_sha256,
        artifact_sha256: &handoff.artifact_sha256,
        qualification_sha256: &producer.qualification_sha256,
        reference_manifest_sha256: &r.reference_manifest_sha256,
        context_execution_sha256: &context_execution_sha256,
        qualification_execution_sha256: case_execution_pins,
        verification_policy_sha256: &r.verification_policy_sha256,
        training: &handoff.scope.training,
        input_sha256: &handoff.scope.input_sha256,
        preprocessing_sha256: &handoff.scope.preprocessing_sha256,
    };
    let pins_sha256 = sha256_hex(&serde_json::to_vec(&pins)?);
    ensure!(
        handoff.admission_pins_sha256 == pins_sha256,
        "ProducerNotAuthenticated: the handoff's admission pins {} are not the controller's \
         admission pins {pins_sha256} for this run (the preregistered release, W2 qualification \
         and case-execution pins, the producer artifact, its input scope and the retained \
         context execution)",
        handoff.admission_pins_sha256
    );
    Ok(evidence.handoff_sha256.clone())
}

/// The W2 execution proof, re-verified from retained pool receipts: the checks
/// `jope_pipeline::cpu_context_bridge::execution_attestation` makes, on retained bytes instead of
/// live files. The preregistered producer executable ran the job to completion, inside its
/// original window (which opened after the freeze, and is the teacher's own authorized
/// experiment and window), under the pool's per-fit memory cap, on the handoff's input, and wrote
/// the handoff's artifact.
///
/// Returns the sha256 of the controller's attestation of this execution (the admission pins'
/// `context_execution_sha256`), derived from the verified receipts as the controller derives it.
fn verify_execution(
    state: &State,
    policy: &FrozenPolicy,
    handoff: &HandoffRecord,
    ex: &ContextExecution,
    record: &HwmRecord,
    teaching: &Teaching<'_>,
) -> Result<String> {
    const RECEIPTS: &str = "UnprovenContextExecution: a retained pool receipt of the context job \
                            is missing or unreadable";
    let job: WorkerJob = blob(state, &ex.job_sha256).context(RECEIPTS)?;
    let outcome: WorkerOutcome = blob(state, &ex.outcome_sha256).context(RECEIPTS)?;
    let experiment: Experiment = blob(state, &ex.experiment_sha256).context(RECEIPTS)?;
    let window: ExecutionWindow = blob(state, &ex.window_sha256).context(RECEIPTS)?;
    ensure!(
        job.command.program.sha256 == handoff.release.producer_sha256,
        "ProducerNotAuthenticated: the context job ran executable {}, not the preregistered \
         producer {}",
        job.command.program.sha256,
        handoff.release.producer_sha256
    );
    experiment.validate()?;
    window.validate(&experiment)?;
    ensure!(
        experiment.resources.per_fit_seconds <= 30 && experiment.resources.threads_per_worker == 1,
        "UnprovenContextExecution: the context job allowance differs from the pool fit limit"
    );
    ensure!(
        window.started_unix_ms >= policy.created_at_unix * 1000,
        "UnprovenContextExecution: the context was produced before the freeze that preregistered \
         its producer"
    );
    // PR #114 review round 1: the context was produced inside the teacher's own authorized
    // execution, the experiment and original window `qualify` holds to the freeze, not in any
    // self-consistent later window. With the timing check below (exit by the window's deadline
    // less its reporting allowance), a context produced after the teaching deadline never teaches.
    let (context_experiment, teacher_experiment) =
        (experiment.identity()?, teaching.experiment.identity()?);
    let authorized = teaching.window;
    ensure!(
        context_experiment == teacher_experiment
            && window.experiment_sha256 == authorized.experiment_sha256
            && window.started_unix_ms == authorized.started_unix_ms
            && window.deadline_unix_ms == authorized.deadline_unix_ms,
        "UnprovenContextExecution: the context was produced in experiment {context_experiment} \
         over [{}, {}] ms, not in the teacher's authorized experiment {teacher_experiment} and \
         window [{}, {}] ms",
        window.started_unix_ms,
        window.deadline_unix_ms,
        authorized.started_unix_ms,
        authorized.deadline_unix_ms
    );
    let job_sha = job.identity()?;
    ensure!(
        job.request.experiment_sha256 == experiment.identity()?
            && job.request.dataset_sha256 == record.identity.dataset_hash,
        "UnprovenContextExecution: the context job belongs to another experiment or dataset"
    );
    ensure!(
        outcome.schema == EXECUTION_SCHEMA
            && outcome.job_sha256 == job_sha
            && outcome.request_sha256 == job.request.identity()?
            && outcome.command_status == EvaluationStatus::Completed
            && outcome.exit_code == Some(0)
            && outcome.signal.is_none()
            && outcome.reason.is_none(),
        "UnprovenContextExecution: the context job did not complete cleanly under its own identity"
    );
    ensure!(
        job.command
            .inputs
            .iter()
            .any(|a| a.sha256 == handoff.scope.input_sha256),
        "UnprovenContextExecution: the handoff's input is not an input of the context job"
    );
    ensure!(
        outcome.outputs.iter().any(|a| {
            a.sha256 == handoff.artifact_sha256
                && job.outputs.iter().any(|p| job.directory.join(p) == a.path)
        }),
        "UnprovenContextExecution: the handoff's artifact is not a declared output of the context \
         job"
    );
    let c = &outcome.consumption;
    let (Some(started), Some(exited), Some(wall), Some(cpu)) = (
        c.gate_released_unix_ms,
        c.observed_exit_unix_ms,
        c.wall_seconds,
        c.cpu_seconds,
    ) else {
        bail!("UnprovenContextExecution: the context job's timing or CPU accounting is unknown");
    };
    ensure!(
        started >= window.started_unix_ms
            && exited >= started
            && exited
                <= window
                    .deadline_unix_ms
                    .saturating_sub(experiment.resources.reporting_seconds * 1000)
            && wall.is_finite()
            && wall >= 0.
            && wall <= experiment.resources.per_fit_seconds as f64
            && cpu.is_finite()
            && cpu >= 0.,
        "UnprovenContextExecution: the context job escaped its original window or lacks finite \
         accounting"
    );
    let allocation = jope_pipeline::execution::fit_memory_allocation(&experiment)?;
    ensure!(
        c.fit_memory_limit_bytes == Some(allocation.per_fit_bytes)
            && c.cgroup_peak_memory_bytes
                .is_some_and(|n| n > 0 && n <= allocation.per_fit_bytes),
        "UnprovenContextExecution: the context job's memory cap is absent or changed, or its peak \
         is outside the allocation"
    );
    // The supervisor receipt is the retained outcome file (`ExecutionProof::attest` pins the
    // file's digest), the input and output are the handoff's, verified above.
    let attestation = ExecutionAttestationRecord {
        schema: EXECUTION_ATTESTATION_SCHEMA,
        supervisor_receipt_sha256: &ex.outcome_sha256,
        program_sha256: &job.command.program.sha256,
        input_sha256: &handoff.scope.input_sha256,
        output_sha256: &handoff.artifact_sha256,
        completed: true,
        cpu_limit: 1,
        memory_max: allocation.per_fit_bytes,
        swap_max: experiment.resources.swap_bytes,
        threads: 1,
        elapsed_seconds: wall,
        deadline_seconds: experiment.resources.per_fit_seconds as f64,
    };
    Ok(sha256_hex(&serde_json::to_vec(&attestation)?))
}
