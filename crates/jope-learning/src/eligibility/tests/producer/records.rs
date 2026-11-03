use super::*;

/// `AdmissionPins`, as the handoff's `admission_pins_sha256` hashes them, with placeholder values.
pub(super) const ADMISSION_PINS_LAYOUT: &str = r#"{"source_sha256":"s","producer_sha256":"p","artifact_sha256":"a","qualification_sha256":"q","reference_manifest_sha256":"m","context_execution_sha256":"x","qualification_execution_sha256":["c0","c1"],"verification_policy_sha256":"v","training":{"dataset_sha256":"d","prepared_sha256":"r","training_sha256":"t","fitting_rows_sha256":"f","cut":2},"input_sha256":"i","preprocessing_sha256":"e"}"#;
/// `ExecutionAttestation`, as the pins' `context_execution_sha256` hashes it.
pub(super) const EXECUTION_ATTESTATION_LAYOUT: &str = r#"{"schema":"jope.cpu-context-execution/controller-v1","supervisor_receipt_sha256":"o","program_sha256":"p","input_sha256":"i","output_sha256":"a","completed":true,"cpu_limit":1,"memory_max":4096,"swap_max":0,"threads":1,"elapsed_seconds":0.25,"deadline_seconds":30.0}"#;
/// `CpuContextArtifact`'s fields, sorted.
pub(super) const ARTIFACT_FIELDS: [&str; 16] = [
    "backend",
    "components",
    "encoder",
    "final_test_targets_used",
    "input_sha256",
    "original_labels_sha256",
    "prepared_rows_sha256",
    "preprocessing",
    "preprocessing_sha256",
    "producer_sha256",
    "qualification",
    "schema",
    "selected_training_rows_sha256",
    "source_sha256",
    "training",
    "vector",
];

/// The W2 records teacher admission rebuilds (the admission pins and the controller's execution
/// attestation) and decodes (the producer artifact), pinned byte for byte with placeholder values.
/// `jope_policy`'s `handoff_tests::the_records_teacher_admission_rebuilds_are_pinned` pins the
/// same bytes for `AdmissionPins`, `ExecutionAttestation` and `CpuContextArtifact`, so a change on
/// either side fails a test on that side.
#[test]
fn the_w2_records_teacher_admission_rebuilds_are_pinned_field_for_field() {
    let training = context::HandoffTraining {
        dataset_sha256: "d".into(),
        prepared_sha256: "r".into(),
        training_sha256: "t".into(),
        fitting_rows_sha256: "f".into(),
        cut: 2,
    };
    let cases = ["c0".to_owned(), "c1".to_owned()];
    let pins = context::AdmissionPinsRecord {
        source_sha256: "s",
        producer_sha256: "p",
        artifact_sha256: "a",
        qualification_sha256: "q",
        reference_manifest_sha256: "m",
        context_execution_sha256: "x",
        qualification_execution_sha256: &cases,
        verification_policy_sha256: "v",
        training: &training,
        input_sha256: "i",
        preprocessing_sha256: "e",
    };
    assert_eq!(serde_json::to_string(&pins).unwrap(), ADMISSION_PINS_LAYOUT);
    let attestation = context::ExecutionAttestationRecord {
        schema: context::EXECUTION_ATTESTATION_SCHEMA,
        supervisor_receipt_sha256: "o",
        program_sha256: "p",
        input_sha256: "i",
        output_sha256: "a",
        completed: true,
        cpu_limit: 1,
        memory_max: 4096,
        swap_max: 0,
        threads: 1,
        elapsed_seconds: 0.25,
        deadline_seconds: 30.0,
    };
    assert_eq!(
        serde_json::to_string(&attestation).unwrap(),
        EXECUTION_ATTESTATION_LAYOUT
    );
    // The producer artifact: exactly `CpuContextArtifact`'s fields, under W2's schema and backend.
    let f = fixture_with(Task::Regression, 5, Some(preregistered()));
    let run = producer_run(&f, &teaching_context(), &f.window, |_| {});
    let artifact: context::ProducedArtifact = serde_json::from_slice(&run.artifact).unwrap();
    let value = serde_json::to_value(&artifact).unwrap();
    let mut keys = value
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    keys.sort();
    assert_eq!(keys, ARTIFACT_FIELDS);
    assert_eq!(context::PRODUCED_ARTIFACT_SCHEMA, "jope.cpu-context/v1");
    assert_eq!(
        context::PRODUCED_ARTIFACT_BACKEND,
        "Rust/Candle 0.11 CPU FP32; frozen accepted transforms; native-component qualification \
         required"
    );
    // The consumed context is the artifact's vector, as little-endian f32 bytes.
    assert_eq!(
        artifact
            .vector
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
        run.context
    );
}

/// Where no context is used the live driver freezes with no context plan; a freeze made before
/// producer preregistration has an encoder but no producer. Either way nothing is issued, nothing
/// is retained, and teacher admission refuses with the existing message.
#[test]
fn without_a_preregistered_producer_nothing_is_issued_and_no_teacher_is_admitted() {
    for (plan, expected) in [
        (
            None,
            "MissingEncoderProvenance: freeze preregistered no teaching context encoder",
        ),
        (
            Some(ContextPlan {
                producer: None,
                ..context_plan()
            }),
            "PreHandoffProvenance",
        ),
    ] {
        let f = fixture_with(Task::Regression, 5, plan.clone());
        let run = producer_run(&f, &teaching_context(), &f.window, |_| {});
        let error = format!(
            "{:#}",
            retain_handed_off_context(&f.state, &f.evidence.freeze, &run).unwrap_err()
        );
        assert!(
            error.contains(expected) && error.contains("new, fully provenanced qualification"),
            "{error}"
        );
        // Nothing of this run was retained (its job is the fixture's own context job).
        for bytes in [&run.handoff, &run.outcome, &run.artifact, &run.context] {
            assert_eq!(f.state.read_blob(&sha256_hex(bytes)).unwrap(), None);
        }
        assert_eq!(authority(&f.state), nothing());
        // What the live driver then submits: the teacher with no context.
        let mut evidence = f.evidence.clone();
        evidence.context = None;
        assert_refused(
            &f,
            &evidence,
            if plan.is_none() {
                expected
            } else {
                "MissingEncoderProvenance: submission carries no context receipt"
            },
            "no issued context",
        );
    }
}

/// Owner decision (d): a qualification stored before the handoff existed stays quarantined. The
/// product path's handed-off evidence for the same teacher does not upgrade it in place.
#[test]
fn a_pre_handoff_qualification_is_not_upgraded_by_the_product_path() {
    let f = fixture_with(Task::Regression, 5, Some(preregistered()));
    let run = producer_run(&f, &teaching_context(), &f.window, |_| {});
    let evidence = submission(&f, &run);
    let qualified = qualify(&f.state, &f.record, &evidence).unwrap();
    assert_eq!(ingest(&f).accepted, 1);
    // The same teacher's qualification as stored before the handoff existed.
    let mut old = qualified;
    old.context_handoff_sha256 = None;
    let mut old_evidence = evidence.clone();
    old_evidence.context.as_mut().unwrap().handoff = None;
    let bytes = canonical_json(&StoredQualification {
        evidence: old_evidence,
        qualification: old,
    })
    .unwrap();
    let path = qualification_path(&f.state, &f.record);
    fs::write(&path, &bytes).unwrap();
    let manifest = freeze_teaching(&f.state).unwrap();
    assert!(manifest.entries.is_empty());
    assert!(
        manifest.exclusions[0].1.contains("MissingContextHandoff"),
        "{:?}",
        manifest.exclusions
    );
    let before = authority(&f.state);
    let error = format!("{:#}", qualify(&f.state, &f.record, &evidence).unwrap_err());
    assert!(error.contains("RequalificationRefused"), "{error}");
    assert_eq!(fs::read(&path).unwrap(), bytes);
    assert_eq!(before, authority(&f.state));
    assert!(freeze_teaching(&f.state).unwrap().entries.is_empty());
}

/// The live driver freezes the inner route: the issued receipt is fitted on the inner partition's
/// fitting rows with its outer hold forbidden, and it verifies for the inner comparison only.
#[test]
fn the_issued_context_is_bound_to_the_inner_scope_the_live_driver_freezes() {
    let f = fixture_with(Task::Regression, 5, Some(preregistered()));
    let (plan, partition, freeze, _) = inner_fixture(&f);
    let policy = load_freeze(&f.state, &freeze).unwrap();
    let window = ExecutionWindow {
        experiment_sha256: f.window.experiment_sha256.clone(),
        started_unix_ms: policy.created_at_unix * 1000,
        deadline_unix_ms: policy.created_at_unix * 1000 + 60_000,
    };
    let run = producer_run(&f, &teaching_context(), &window, |_| {});
    let context = retain_handed_off_context(&f.state, &freeze, &run).unwrap();
    let scope = context::scope(&f.state, &policy, policy.context.as_ref().unwrap()).unwrap();
    assert_eq!(scope.outer_fold, Some(partition.outer_fold));
    assert_eq!(
        scope.forbidden.iter().map(|m| &m.row).collect::<Vec<_>>(),
        partition.hold_rows.iter().collect::<Vec<_>>()
    );
    let receipt: crate::encoder_receipt::EncoderReceipt =
        blob(&f.state, &context.receipt_sha256).unwrap();
    assert_eq!(receipt.scope_sha256, canonical_sha256(&scope).unwrap());

    let mut record = f.record.clone();
    record.identity = plan.comparison.clone();
    record.identity_sha256 = plan.comparison.sha256().unwrap();
    let mut evidence = f.evidence.clone();
    evidence.freeze = freeze;
    evidence.context = Some(context.clone());
    // The inner comparison's teacher is measured in the window that opened with its freeze.
    let teaching = context::Teaching {
        experiment: &f.experiment,
        window: &window,
    };
    let verify = |evidence: &OofEvidence| {
        context::verify(
            &f.state,
            &policy,
            &record,
            evidence,
            &Default::default(),
            &teaching,
        )
    };
    assert_eq!(
        verify(&evidence).unwrap(),
        (
            context.receipt_sha256.clone(),
            context.handoff.unwrap().handoff_sha256
        )
    );
    // The whole-development freeze's issue of the same run is not the inner comparison's context.
    evidence.context = Some(retain_handed_off_context(&f.state, &f.evidence.freeze, &run).unwrap());
    assert!(verify(&evidence).is_err());
}
