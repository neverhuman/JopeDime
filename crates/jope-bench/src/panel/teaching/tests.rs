use super::*;
#[test]
fn transient_admission_failure_is_inventory_not_terminal() {
    let mut report = Report {
        schema: "synthetic".into(),
        experiment_sha256: "a".repeat(64),
        dataset: "fixture".into(),
        registry_sha256: "b".repeat(64),
        original_label_role: "test".into(),
        effective_label_role: "development".into(),
        baseline_worker_fits: 0,
        registration_unavailable: None,
        entries: vec![Admission {
            method: "asm.generic".into(),
            candidate_sha256: "c".repeat(64),
            record: None,
            qualified: false,
            reason: Some("synthetic missing retained input".into()),
            qualification: None,
        }],
    };
    assert!(!terminal_report(&report));
    // Only the dependency inventory frozen before admission is permanent.
    report.registration_unavailable = Some("fewer than five training groups".into());
    assert!(terminal_report(&report));
}

fn sha(bytes: &[u8]) -> String {
    jope_pipeline::sha256_hex(bytes)
}

/// A configuration whose files are never opened unless its W2 qualification is loaded.
fn configuration(qualification: Option<Value>, unavailable: Option<&str>) -> Configuration {
    let artifact = |name: &str| json!({"path": format!("/nonexistent/{name}"), "sha256": sha(name.as_bytes())});
    let runtime = |name: &str| {
        json!({"program": artifact(name), "argument_prefix": [], "inputs": [],
            "package_files": [], "environment": {}, "version": "fixture",
            "unavailable": unavailable})
    };
    serde_json::from_value(json!({
        "schema": super::super::optimize::SCHEMA, "source_sha40": "a".repeat(40),
        "release_manifest": artifact("release"), "source_manifest": artifact("source"),
        "checkpoint_receipt": artifact("checkpoint"), "policy_weights": artifact("policy"),
        "vocabulary": artifact("vocabulary"), "frozen_transforms": artifact("transforms"),
        "policy": runtime("policy worker"), "encoder": runtime("W2 context producer"),
        "encoder_assets": {"configuration": artifact("encoder assets"), "files": []},
        "qualification": qualification, "seconds": 60, "max_evaluations": 80,
        "sampling_rounds": 1, "sampling_batch": 8, "sampling_temperature": 1.0,
        "dataset_wave_width": 1, "scheduling_policy": "fixture", "inner_fold_policy": "fixture"
    }))
    .unwrap()
}

/// The producer a W2 qualification of `config`'s producer registers, converted as
/// `producer_plan` converts it: field for field into the preregistration teacher admission
/// parses.
fn registered(config: &Configuration) -> ProducerPlan {
    let qualification = context_producer::Qualification {
        schema: "jope.cpu-context-qualification/v1".into(),
        source_sha256: config.source_sha40.clone(),
        producer_sha256: config.encoder.program.sha256.clone(),
        verification_policy_sha256: context_producer::verification_policy_sha256(),
        reference_manifest_sha256: sha(b"reference manifest"),
        encoder: context_producer::EncoderIdentity::accepted(),
        case_outputs: vec![b"case output".to_vec(); 8],
    };
    let bytes = serde_json::to_vec(&qualification).unwrap();
    let case_executions = (0..8)
        .map(|case| sha(format!("case execution {case}").as_bytes()))
        .collect();
    let registration = context_producer::ProducerRegistration::from_qualification(
        &bytes,
        &sha(&bytes),
        case_executions,
    )
    .unwrap();
    serde_json::from_value(serde_json::to_value(&registration).unwrap()).unwrap()
}

/// W3-T02: a regression dataset's freeze preregisters the W2 producer exactly as its
/// qualification registers it, under the producer's own encoder identity. Every other case
/// preregisters no context and records why: no qualification, one its loader refuses, an
/// unavailable producer runtime, or a classification dataset.
#[test]
fn a_context_is_preregistered_only_from_a_registered_w2_producer() {
    let no_context = |config: &Configuration, task: Task, needle: &str| {
        let (plan, context) = teaching_context(config, task, &producer_plan(config));
        assert!(plan.is_none(), "{needle}");
        match context {
            TeachingContext::NoContext { reason } => {
                assert!(reason.contains(needle), "{needle}: {reason}")
            }
            other => panic!("{needle}: {other:?}"),
        }
    };
    no_context(
        &configuration(None, None),
        Task::Regression,
        "no completed W2 CPU context qualification is configured",
    );
    let unverifiable = json!({"qualification": {"path": "/nonexistent/q", "sha256": sha(b"q")},
        "reference_manifest": {"path": "/nonexistent/r", "sha256": sha(b"r")},
        "case_outputs": [], "case_executions": []});
    no_context(
        &configuration(Some(unverifiable.clone()), None),
        Task::Regression,
        "qualification requires eight executed cases",
    );
    no_context(
        &configuration(Some(unverifiable), Some("withheld")),
        Task::Regression,
        "runtime is unavailable: withheld",
    );

    let config = configuration(None, None);
    let producer = registered(&config);
    for task in [Task::Binary, Task::Multiclass] {
        let (plan, context) = teaching_context(&config, task, &Ok(producer.clone()));
        assert!(plan.is_none());
        assert!(
            matches!(&context, TeachingContext::NoContext { reason }
                if reason.contains("regression-only")),
            "{context:?}"
        );
    }
    let (plan, context) = teaching_context(&config, Task::Regression, &Ok(producer.clone()));
    let plan = plan.unwrap();
    assert_eq!(plan.producer.as_ref(), Some(&producer));
    // The registration's case-execution pins reach the preregistration: teacher admission
    // rebuilds every handoff's admission pins from them.
    assert_eq!(
        producer
            .qualification_execution_sha256
            .as_ref()
            .map(Vec::len),
        Some(8)
    );
    assert_eq!(plan.source_family, context_producer::SCHEMA);
    assert_eq!(
        plan.encoder,
        Encoder {
            artifact_sha256: config.encoder.program.sha256.clone(),
            configuration_sha256: config.encoder_assets.configuration.sha256.clone(),
            seed: 0,
            product: Product::TeachingContext2048,
        }
    );
    // The registered release is the producer executable the configuration runs.
    assert_eq!(
        producer.release.producer_sha256,
        config.encoder.program.sha256
    );
    assert_eq!(
        context,
        TeachingContext::Producer {
            qualification_sha256: producer.qualification_sha256.clone()
        }
    );
}

/// Registrations record the teaching context either way; a registration written before
/// producer preregistration keeps its bytes.
#[test]
fn registrations_record_the_teaching_context_and_old_ones_keep_their_bytes() {
    let old = json!({
        "state": "frozen",
        "comparison": HwmIdentity {
            task: "regression".into(),
            dataset_hash: "d".repeat(64),
            cut_id: "c".repeat(64),
            evaluator_sha256: "e".repeat(64),
            vocabulary_sha256: "a".repeat(64),
            protocol: "fixture".into(),
            comparison: None,
        },
        "partition": InnerPartition {
            schema: "fixture".into(),
            dataset_sha256: "d".repeat(64),
            original_split_sha256: "c".repeat(64),
            development_sha256: "b".repeat(64),
            outer_fold: 0,
            fitting_rows: vec!["r1".into()],
            hold_rows: vec!["r2".into()],
            fitting_groups: vec!["g1".into()],
            hold_groups: vec!["g2".into()],
            inner_validation_rows: vec![vec!["r1".into()]],
            assignment_policy: "fixture".into(),
            evaluation_seed: 7,
            request_seed_policy: "fixture".into(),
            executor_program_sha256: "f".repeat(64),
            runtime_inventory_sha256: "f".repeat(64),
            fit_randomness_policy: "fixture".into(),
            raw_input_policy: "fixture".into(),
        },
        "freeze": FreezeReceipt {
            comparison_sha256: "1".repeat(64),
            freeze_sha256: "2".repeat(64),
        },
    });
    let parsed: Registration = serde_json::from_value(old.clone()).unwrap();
    assert!(matches!(
        &parsed,
        Registration::Frozen { context: None, .. }
    ));
    assert_eq!(serde_json::to_value(&parsed).unwrap(), old);
    for (context, key) in [
        (
            TeachingContext::Producer {
                qualification_sha256: "3".repeat(64),
            },
            "producer",
        ),
        (
            TeachingContext::NoContext {
                reason: "fixture".into(),
            },
            "no_context",
        ),
    ] {
        let mut value = old.clone();
        value["context"] = serde_json::to_value(&context).unwrap();
        assert!(value["context"].get(key).is_some(), "{value}");
        let parsed: Registration = serde_json::from_value(value.clone()).unwrap();
        assert!(matches!(&parsed, Registration::Frozen { context: Some(c), .. } if *c == context));
        assert_eq!(serde_json::to_value(&parsed).unwrap(), value);
    }
}

/// A run that emitted no handoff yields no context; the submission then carries none.
#[test]
fn a_run_without_a_teacher_admission_handoff_yields_no_context() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp/teaching-handoff");
    fs::create_dir_all(&root).unwrap();
    let directory = tempfile::tempdir_in(fs::canonicalize(&root).unwrap()).unwrap();
    let state = State::at(&directory.path().join("state"));
    let freeze = FreezeReceipt {
        comparison_sha256: "1".repeat(64),
        freeze_sha256: "2".repeat(64),
    };
    let run = directory.path().join("asm-contexts");
    assert!(handed_off_context(&state, &freeze, &run).unwrap().is_none());
    // A handoff whose run files are absent is an error (retryable inventory), not a context.
    fs::create_dir_all(&run).unwrap();
    fs::write(run.join(super::super::optimize::TEACHER_HANDOFF), b"{}").unwrap();
    assert!(handed_off_context(&state, &freeze, &run).is_err());
}
