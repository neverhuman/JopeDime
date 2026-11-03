use super::*;
impl HandoffParts {
    /// Recompute the supervisor receipt's job identity after the job changed, then the
    /// controller's admission pins.
    pub(super) fn reseal(&mut self) {
        self.outcome.job_sha256 = self.job.identity().unwrap();
        self.outcome.request_sha256 = self.job.request.identity().unwrap();
        self.seal();
    }

    /// The supervisor receipt's bytes, as a submission retains them.
    pub(super) fn outcome_bytes(&self) -> Vec<u8> {
        canonical_json(&self.outcome).unwrap()
    }

    /// The job wrote `bytes` instead: its supervisor receipt, the handoff and the controller's
    /// admission pins all name them.
    pub(super) fn with_artifact(&mut self, bytes: Vec<u8>) {
        let sha = sha256_hex(&bytes);
        self.outcome.outputs[0].sha256 = sha.clone();
        self.handoff["artifact_sha256"] = sha.into();
        self.artifact = bytes;
        self.seal();
    }

    /// Set the handoff's `admission_pins_sha256` to the pins the controller issues for this
    /// run (`optimize::prepare_regression_context`): the release and W2 qualification the
    /// handoff names, the case-execution pins, this artifact and input scope, and the
    /// controller's attestation of the retained context execution.
    pub(super) fn seal(&mut self) {
        let h = &self.handoff;
        let text = |v: &serde_json::Value| v.as_str().unwrap().to_owned();
        let resources = &self.experiment.resources;
        let attestation = FixtureAttestation {
            schema: "jope.cpu-context-execution/controller-v1",
            supervisor_receipt_sha256: sha256_hex(&self.outcome_bytes()),
            program_sha256: self.job.command.program.sha256.clone(),
            input_sha256: text(&h["scope"]["input_sha256"]),
            output_sha256: text(&h["artifact_sha256"]),
            completed: true,
            cpu_limit: 1,
            memory_max: jope_pipeline::execution::fit_memory_allocation(&self.experiment)
                .unwrap()
                .per_fit_bytes,
            swap_max: resources.swap_bytes,
            threads: 1,
            elapsed_seconds: self.outcome.consumption.wall_seconds.unwrap(),
            deadline_seconds: resources.per_fit_seconds as f64,
        };
        let pins = FixturePins {
            source_sha256: text(&h["release"]["source_sha256"]),
            producer_sha256: text(&h["release"]["producer_sha256"]),
            artifact_sha256: text(&h["artifact_sha256"]),
            qualification_sha256: text(&h["qualification_sha256"]),
            reference_manifest_sha256: text(&h["release"]["reference_manifest_sha256"]),
            context_execution_sha256: sha256_hex(&serde_json::to_vec(&attestation).unwrap()),
            qualification_execution_sha256: self.case_execution_pins.clone(),
            verification_policy_sha256: text(&h["release"]["verification_policy_sha256"]),
            training: serde_json::from_value(h["scope"]["training"].clone()).unwrap(),
            input_sha256: text(&h["scope"]["input_sha256"]),
            preprocessing_sha256: text(&h["scope"]["preprocessing_sha256"]),
        };
        self.handoff["admission_pins_sha256"] =
            sha256_hex(&serde_json::to_vec(&pins).unwrap()).into();
    }
}

/// The handoff the preregistered producer makes of `context` (its little-endian f32 bytes)
/// for the comparison's dataset, with a completed execution inside `window`: the artifact the
/// job wrote is a `jope.cpu-context/v1` artifact holding exactly that context, and the
/// handoff names the controller's admission pins. Nothing is executed and nothing is
/// retained.
pub(super) fn handoff_parts(
    state: &State,
    context: &[u8],
    comparison: &HwmIdentity,
    experiment: &Experiment,
    window: &ExecutionWindow,
) -> HandoffParts {
    let plan = producer_plan();
    let directory = state.root().join("fixture-context-worker");
    let program = Artifact {
        path: directory.join("producer"),
        sha256: plan.release.producer_sha256.clone(),
    };
    let input = Artifact {
        path: directory.join("input.json"),
        sha256: sha256_hex(b"fixture context input"),
    };
    let training = json!({
        "dataset_sha256": comparison.dataset_hash,
        "prepared_sha256": sha256_hex(b"fixture prepared"),
        "training_sha256": sha256_hex(b"fixture training"),
        "fitting_rows_sha256": sha256_hex(b"fixture fitting rows"),
        "cut": 0
    });
    let vector = context
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    // `jope_policy::context_producer::CpuContextArtifact`, as the W2 producer job writes it.
    let artifact_bytes = serde_json::to_vec(&json!({
        "schema": "jope.cpu-context/v1",
        "backend": "Rust/Candle 0.11 CPU FP32; frozen accepted transforms; \
                    native-component qualification required",
        "qualification": "unqualified",
        "source_sha256": plan.release.source_sha256,
        "producer_sha256": plan.release.producer_sha256,
        "input_sha256": input.sha256,
        "training": training,
        "selected_training_rows_sha256": training["fitting_rows_sha256"],
        "original_labels_sha256": sha256_hex(b"fixture original labels"),
        "preprocessing_sha256": sha256_hex(b"fixture preprocessing"),
        "prepared_rows_sha256": sha256_hex(b"fixture prepared rows"),
        "preprocessing": {"fixture": "train-only preprocessing"},
        "encoder": plan.release.encoder,
        "components": {"fixture": "encoder products"},
        "vector": vector,
        "final_test_targets_used": false
    }))
    .unwrap();
    let artifact = Artifact {
        path: directory.join("artifact.json"),
        sha256: sha256_hex(&artifact_bytes),
    };
    let request = EvaluationRequest {
        schema: REQUEST_SCHEMA.into(),
        experiment_sha256: experiment.identity().unwrap(),
        dataset_sha256: comparison.dataset_hash.clone(),
        split_sha256: comparison.cut_id.clone(),
        fitting_rows_sha256: sha256_hex(b"fixture context fitting rows"),
        scoring_rows_sha256: sha256_hex(b"fixture context scoring rows"),
        fitting_rows: 10,
        scoring_rows: 1,
        recipe_sha256: plan.release.producer_sha256.clone(),
        evaluator_sha256: comparison.evaluator_sha256.clone(),
        checkpoint_sha256: None,
        task: Task::Regression,
        metric: jope_pipeline::experiment::Metric::Rmse,
        classes: vec![],
        fold: 0,
        stage: EvaluationStage::Diagnostic,
        seed: 7,
        numerical_policy: NUMERICAL_POLICY.into(),
    };
    let job = WorkerJob {
        request: request.clone(),
        command: WorkerCommand {
            program,
            args: vec![],
            inputs: vec![input.clone()],
            environment: BTreeMap::new(),
        },
        directory,
        outputs: vec!["artifact.json".into()],
        arm: None,
    };
    let allocation = jope_pipeline::execution::fit_memory_allocation(experiment).unwrap();
    let outcome = WorkerOutcome {
        schema: EXECUTION_SCHEMA.into(),
        job_sha256: job.identity().unwrap(),
        request_sha256: request.identity().unwrap(),
        command_status: EvaluationStatus::Completed,
        reason: None,
        exit_code: Some(0),
        signal: None,
        outputs: vec![artifact.clone()],
        stdout: None,
        stderr: None,
        consumption: ObservedConsumption {
            fit_memory_limit_bytes: Some(allocation.per_fit_bytes),
            gate_released_unix_ms: Some(window.started_unix_ms),
            observed_exit_unix_ms: Some(window.started_unix_ms + 10),
            wall_seconds: Some(0.01),
            cpu_seconds: Some(0.01),
            cgroup_peak_memory_bytes: Some(4096),
            accounting_reason: "synthetic fixture, nothing executed".into(),
            ..Default::default()
        },
    };
    let handoff = json!({
        "schema": HANDOFF_SCHEMA,
        "consumer": TEACHER_ADMISSION,
        "release": plan.release,
        "transformations": plan.transformations,
        "scope": {
            "training": training,
            "input_sha256": input.sha256,
            "preprocessing_sha256": sha256_hex(b"fixture preprocessing"),
            "prepared_rows_sha256": sha256_hex(b"fixture prepared rows")
        },
        "artifact_sha256": artifact.sha256,
        "qualification_sha256": plan.qualification_sha256,
        "admission_pins_sha256": "",
        "context_sha256": sha256_hex(context)
    });
    let mut parts = HandoffParts {
        handoff,
        job,
        outcome,
        artifact: artifact_bytes,
        case_execution_pins: fixture_case_execution_pins(),
        experiment: experiment.clone(),
    };
    parts.seal();
    parts
}

/// Retain `parts` as a submission would carry them.
pub(super) fn retain_handoff(
    state: &State,
    parts: &HandoffParts,
    experiment: &Experiment,
    window: &ExecutionWindow,
) -> HandoffEvidence {
    let put = |bytes: Vec<u8>| state.put_blob(&bytes).unwrap();
    put(parts.artifact.clone());
    HandoffEvidence {
        handoff_sha256: put(serde_json::to_vec(&parts.handoff).unwrap()),
        execution: ContextExecution {
            job_sha256: put(canonical_json(&parts.job).unwrap()),
            outcome_sha256: put(parts.outcome_bytes()),
            experiment_sha256: put(canonical_json(experiment).unwrap()),
            window_sha256: put(canonical_json(window).unwrap()),
        },
    }
}

/// The retained context bytes of `context`.
pub(super) fn context_bytes(state: &State, context: &ContextEvidence) -> Vec<u8> {
    state.read_blob(&context.output_sha256).unwrap().unwrap()
}

/// The preregistered producer's handoff of `context`, executed inside `window`.
pub(super) fn handed_off(
    state: &State,
    mut context: ContextEvidence,
    comparison: &HwmIdentity,
    experiment: &Experiment,
    window: &ExecutionWindow,
) -> ContextEvidence {
    let parts = handoff_parts(
        state,
        &context_bytes(state, &context),
        comparison,
        experiment,
        window,
    );
    context.handoff = Some(retain_handoff(state, &parts, experiment, window));
    context
}

/// Producer-side scope, built from the fixture rows rather than the receiver's derivation.
pub(super) fn context_scope(comparison: &HwmIdentity, rows: &[TargetRow]) -> Scope {
    Scope {
        comparison: comparison.clone(),
        source_family: context_plan().source_family,
        outer_fold: None,
        inner_fold: None,
        fitting: rows
            .iter()
            .map(|row| Member {
                row: row.id.clone(),
                group: row.group.clone(),
            })
            .collect(),
        forbidden: vec![],
    }
}

/// `width` finite f32 values inside the W2 frozen clip (|v| <= 32), little-endian.
pub(super) fn context_output(width: usize, offset: f32) -> Vec<u8> {
    (0..width)
        .flat_map(|index| (offset + index as f32 / 128.).to_le_bytes())
        .collect()
}

/// Retain one synthetic producer's receipt, fitted state and output; nothing is encoded.
pub(super) fn produce_context(
    state: &State,
    scope: Scope,
    encoder: Encoder,
    parent_receipts: Vec<String>,
    output: &[u8],
) -> ContextEvidence {
    let capability = EncoderCapability::bind(
        scope,
        encoder,
        Inputs {
            features_sha256: sha256_hex(b"fixture projected features"),
            labels_sha256: Some(sha256_hex(b"fixture projected labels")),
            parent_receipts,
        },
    )
    .unwrap();
    let fitted = format!("synthetic fitted context state/{}", output.len()).into_bytes();
    let receipt = capability.receipt(&fitted, output).unwrap();
    ContextEvidence {
        receipt_sha256: state.put_blob(&canonical_json(&receipt).unwrap()).unwrap(),
        fitted_state_sha256: state.put_blob(&fitted).unwrap(),
        output_sha256: state.put_blob(output).unwrap(),
        handoff: None,
    }
}

pub(super) fn valid_context(
    state: &State,
    comparison: &HwmIdentity,
    rows: &[TargetRow],
) -> ContextEvidence {
    let parent = produce_context(
        state,
        context_scope(comparison, rows),
        Encoder {
            product: Product::DatasetSignature512,
            ..context_plan().encoder
        },
        vec![],
        &context_output(512, 0.25),
    );
    produce_context(
        state,
        context_scope(comparison, rows),
        context_plan().encoder,
        vec![parent.receipt_sha256],
        &context_output(2048, 0.5),
    )
}
