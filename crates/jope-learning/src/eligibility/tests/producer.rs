//! W3-T02, producer side: a real teacher is admissible through the product path.
//!
//! - The freeze preregisters the W2 producer exactly as its W2 qualification registers it
//!   (`jope_policy::context_producer::ProducerRegistration`, parsed field for field as
//!   `ProducerPlan`; the W2 side pins that record and shows it is the producer that hands off to
//!   teacher admission, in
//!   `the_preregistered_producer_is_the_producer_that_hands_off_to_teacher_admission`).
//! - The producer run hands its context off to teacher admission.
//! - `retain_handed_off_context`, which the live `jope-bench` driver calls, retains that run where
//!   teacher admission looks for it and issues the context's receipt under the freeze's scope.
//! - `qualify` admits the teacher and records the handoff; `hwm::ingest`, `freeze_teaching` and
//!   `revalidate_qualification` agree.
//!
//! The producer run is a CPU fixture: synthetic pool receipts and a synthetic context, as the live
//! driver retains them. Nothing is encoded or executed, and this is not a live W2 run.
use super::provenance::{assert_refused, authority, Authority};
use super::*;

/// The registration a W2 qualification yields, as `ProducerRegistration` serializes it.
fn registration() -> serde_json::Value {
    let plan = producer_plan();
    json!({
        "release": plan.release,
        "transformations": plan.transformations,
        "qualification_sha256": plan.qualification_sha256,
        "qualification_execution_sha256": plan.qualification_execution_sha256,
    })
}

/// The freeze's context plan as the live driver builds it: the preregistered context encoder and
/// the W2 producer parsed field for field from its registration.
fn preregistered() -> ContextPlan {
    ContextPlan {
        producer: Some(serde_json::from_value(registration()).unwrap()),
        ..context_plan()
    }
}

/// The context the producer run hands off; distinct from the fixture's own context, so nothing
/// of this run is retained before the product path retains it.
fn teaching_context() -> Vec<u8> {
    context_output(2048, 0.625)
}

/// The W2 producer run of `context` for `f`'s dataset inside `window`, as the live driver retains
/// it: the handoff record the run emitted for teacher admission, the context job and its
/// supervisor receipt, the experiment and window, the artifact the job wrote and the context's
/// bytes. `tamper` changes the run first. Nothing is retained.
fn producer_run(
    f: &Fixture,
    context: &[u8],
    window: &ExecutionWindow,
    tamper: impl FnOnce(&mut HandoffParts),
) -> ContextRun {
    let mut parts = handoff_parts(&f.state, context, &f.plan.comparison, &f.experiment, window);
    tamper(&mut parts);
    ContextRun {
        handoff: serde_json::to_vec(&parts.handoff).unwrap(),
        job: canonical_json(&parts.job).unwrap(),
        outcome: parts.outcome_bytes(),
        experiment: canonical_json(&f.experiment).unwrap(),
        window: serde_json::to_vec(window).unwrap(),
        artifact: parts.artifact,
        context: context.to_vec(),
    }
}

/// `f`'s teacher, submitted with the context the product path issued for `run`.
fn submission(f: &Fixture, run: &ContextRun) -> OofEvidence {
    OofEvidence {
        context: Some(retain_handed_off_context(&f.state, &f.evidence.freeze, run).unwrap()),
        ..f.evidence.clone()
    }
}

fn ingest(f: &Fixture) -> hwm::IngestReport {
    hwm::ingest(
        &f.state,
        None,
        vec![Candidate {
            record: f.record.clone(),
            blob: candidate_blob(""),
            chain_from_source: false,
        }],
        json!({"kind":"w3-t02-producer"}),
    )
    .unwrap()
}

fn nothing() -> Authority {
    Authority {
        qualified: vec![],
        pointers: vec![],
        accepted: vec![],
        teaching: vec![],
    }
}

/// Positive control: freeze with the preregistered producer, run it, hand its context off,
/// retain the run and admit the teacher. The qualification records the handoff.
#[test]
fn a_teacher_is_admitted_through_the_product_path() {
    let f = fixture_with(Task::Regression, 5, Some(preregistered()));
    assert_eq!(authority(&f.state), nothing());
    let run = producer_run(&f, &teaching_context(), &f.window, |_| {});
    let evidence = submission(&f, &run);
    // Issuing the context retains evidence and decides nothing.
    assert_eq!(authority(&f.state), nothing());
    let context = evidence.context.clone().unwrap();
    let handoff = context.handoff.clone().unwrap();
    // Everything teacher admission reads is retained under its digest.
    for (sha, bytes) in [
        (&handoff.handoff_sha256, &run.handoff),
        (&handoff.execution.job_sha256, &run.job),
        (&handoff.execution.outcome_sha256, &run.outcome),
        (&handoff.execution.experiment_sha256, &run.experiment),
        (&handoff.execution.window_sha256, &run.window),
        (&context.fitted_state_sha256, &run.artifact),
        (&context.output_sha256, &run.context),
    ] {
        assert_eq!(f.state.read_blob(sha).unwrap().as_ref(), Some(bytes));
    }
    // The receipt is the teaching context's, bound to the handoff's own input closure.
    let receipt: crate::encoder_receipt::EncoderReceipt =
        blob(&f.state, &context.receipt_sha256).unwrap();
    let handed: serde_json::Value = serde_json::from_slice(&run.handoff).unwrap();
    assert_eq!(receipt.encoder, context_plan().encoder);
    assert_eq!(
        receipt.inputs.features_sha256,
        handed["scope"]["input_sha256"]
    );
    assert_eq!(
        receipt.inputs.labels_sha256.as_deref(),
        handed["scope"]["training"]["training_sha256"].as_str()
    );
    assert!(receipt.inputs.parent_receipts.is_empty());
    assert_eq!(receipt.output_sha256, sha256_hex(&run.context));

    let qualification = qualify(&f.state, &f.record, &evidence).unwrap();
    assert_eq!(
        qualification.context_handoff_sha256.as_deref(),
        Some(sha256_hex(&run.handoff).as_str())
    );
    assert_eq!(
        qualification.context_receipt_sha256.as_deref(),
        Some(context.receipt_sha256.as_str())
    );
    let report = ingest(&f);
    assert_eq!((report.accepted, report.rejected), (1, 0));
    let after = authority(&f.state);
    assert_eq!(after.qualified.len(), 1);
    assert_eq!(
        after.teaching,
        vec![f.record.provenance.record_sha256.clone()]
    );
    assert_eq!(
        freeze_teaching(&f.state).unwrap().entries,
        vec![qualification.clone()]
    );
    assert_eq!(
        revalidate_qualification(&f.state, &f.record).unwrap(),
        qualification
    );
    // Issuing the same run again is idempotent.
    assert_eq!(
        retain_handed_off_context(&f.state, &f.evidence.freeze, &run).unwrap(),
        context
    );
}

/// Every refusal goes through `qualify` and `hwm::ingest`, is inventoried and changes no
/// authority: a producer the freeze did not preregister, another W2 qualification, a handoff for
/// another consumer, a substituted context, a run outside its window and missing pool receipts.
#[test]
fn the_product_path_refuses_an_unauthenticated_or_unproven_context() {
    type Tamper = fn(&mut HandoffParts);
    let cases: [(&str, &str, Tamper); 4] = [
        (
            "a run of a producer the freeze did not preregister",
            "ProducerNotAuthenticated",
            |p| {
                let other = sha256_hex(b"another W2 producer executable");
                p.handoff["release"]["producer_sha256"] = other.clone().into();
                p.job.command.program.sha256 = other;
                p.reseal();
            },
        ),
        (
            "a handoff under another W2 qualification",
            "ProducerNotAuthenticated",
            |p| p.handoff["qualification_sha256"] = sha256_hex(b"another W2 qualification").into(),
        ),
        (
            "a handoff for another consumer",
            "handed off for consumer",
            |p| p.handoff["consumer"] = "accepted_policy_proposal".into(),
        ),
        (
            "a run that exited after its original window",
            "escaped its original window",
            |p| {
                p.outcome.consumption.observed_exit_unix_ms = p
                    .outcome
                    .consumption
                    .observed_exit_unix_ms
                    .map(|exit| exit + 3_600_000)
            },
        ),
    ];
    for (label, expected, tamper) in cases {
        let f = fixture_with(Task::Regression, 5, Some(preregistered()));
        let run = producer_run(&f, &teaching_context(), &f.window, tamper);
        assert_refused(&f, &submission(&f, &run), expected, label);
    }

    // A substituted context: the teacher claims other context bytes than the producer handed off.
    let f = fixture_with(Task::Regression, 5, Some(preregistered()));
    let mut run = producer_run(&f, &teaching_context(), &f.window, |_| {});
    run.context = context_output(2048, 0.75);
    assert_refused(
        &f,
        &submission(&f, &run),
        "is not the handed-off context",
        "substituted context",
    );

    // A run in a window that opened before the freeze preregistered its producer.
    let f = fixture_with(Task::Regression, 5, Some(preregistered()));
    let early = ExecutionWindow {
        experiment_sha256: f.window.experiment_sha256.clone(),
        started_unix_ms: f.window.started_unix_ms - 120_000,
        deadline_unix_ms: f.window.deadline_unix_ms - 120_000,
    };
    let run = producer_run(&f, &teaching_context(), &early, |_| {});
    assert_refused(
        &f,
        &submission(&f, &run),
        "before the freeze",
        "run before the freeze",
    );

    // Missing pool receipts: the run was issued, then a retained receipt or the artifact is gone.
    type Retained = fn(&ContextEvidence) -> String;
    let missing: [(&str, &str, Retained); 3] = [
        (
            "the supervisor outcome",
            "a retained pool receipt of the context job is missing",
            |c| c.handoff.clone().unwrap().execution.outcome_sha256,
        ),
        (
            "the context job",
            "a retained pool receipt of the context job is missing",
            |c| c.handoff.clone().unwrap().execution.job_sha256,
        ),
        (
            "the artifact the job wrote",
            "retained context artifact is missing",
            |c| c.fitted_state_sha256.clone(),
        ),
    ];
    for (label, expected, retained) in missing {
        let f = fixture_with(Task::Regression, 5, Some(preregistered()));
        let run = producer_run(&f, &teaching_context(), &f.window, |_| {});
        let evidence = submission(&f, &run);
        fs::remove_file(
            f.state
                .blob_path(&retained(evidence.context.as_ref().unwrap())),
        )
        .unwrap();
        assert_refused(&f, &evidence, expected, label);
    }
}

/// A producer run whose artifact the job wrote as `change` edits it: the job's supervisor receipt,
/// the handoff and the controller's admission pins all name the edited bytes.
fn with_changed_artifact(f: &Fixture, change: impl FnOnce(&mut serde_json::Value)) -> ContextRun {
    producer_run(f, &teaching_context(), &f.window, |p| {
        let mut artifact: serde_json::Value = serde_json::from_slice(&p.artifact).unwrap();
        change(&mut artifact);
        p.with_artifact(serde_json::to_vec(&artifact).unwrap());
    })
}

/// PR #114 review round 1, P1 (the reviewer's probe): the producer artifact, the context job and
/// its supervisor outcome stay byte-identical, and only the submitted 2048-f32 context (finite,
/// clipped values) and the handoff's context digest are substituted. The artifact still holds the
/// original context, so the substitute is refused and no authority changes.
#[test]
fn a_substituted_context_with_a_matching_handoff_digest_is_refused() {
    let f = fixture_with(Task::Regression, 5, Some(preregistered()));
    let original = producer_run(&f, &teaching_context(), &f.window, |_| {});
    let mut run = original.clone();
    run.context = (0..2048).flat_map(|_| 1.0f32.to_le_bytes()).collect();
    let mut handoff: serde_json::Value = serde_json::from_slice(&run.handoff).unwrap();
    handoff["context_sha256"] = json!(sha256_hex(&run.context));
    run.handoff = serde_json::to_vec(&handoff).unwrap();
    assert_eq!(
        (&run.artifact, &run.job, &run.outcome),
        (&original.artifact, &original.job, &original.outcome)
    );
    assert_refused(
        &f,
        &submission(&f, &run),
        "is not the context the producer artifact contains",
        "a substituted context with a matching handoff digest",
    );
}

/// P1: an artifact the job wrote, and the handoff and the controller's pins name, that decodes to
/// another context than the one handed off (one value differs; still finite and clipped).
#[test]
fn an_artifact_that_decodes_to_another_context_is_refused() {
    let f = fixture_with(Task::Regression, 5, Some(preregistered()));
    let run = with_changed_artifact(&f, |a| a["vector"][0] = json!(0.25));
    assert_refused(
        &f,
        &submission(&f, &run),
        "is not the context the producer artifact contains",
        "an artifact of another context",
    );
}

/// P1: a truncated or corrupt producer artifact never yields a teaching context.
#[test]
fn a_truncated_or_corrupt_artifact_is_refused() {
    // Truncated: the job's receipt, the handoff and the pins name the first half of the artifact.
    let f = fixture_with(Task::Regression, 5, Some(preregistered()));
    let run = producer_run(&f, &teaching_context(), &f.window, |p| {
        let half = p.artifact[..p.artifact.len() / 2].to_vec();
        p.with_artifact(half);
    });
    assert_refused(
        &f,
        &submission(&f, &run),
        "is not a jope.cpu-context/v1 context artifact",
        "a truncated artifact",
    );

    type Change = fn(&mut serde_json::Value);
    let corrupt: [(&str, &str, Change); 6] = [
        (
            "an unknown artifact field",
            "is not a jope.cpu-context/v1 context artifact",
            |a| a["issued_by"] = json!("the candidate"),
        ),
        (
            "another artifact schema",
            "is not the W2 CPU context envelope",
            |a| a["schema"] = json!("jope.cpu-context/v0"),
        ),
        (
            "an artifact that used final test targets",
            "is not the W2 CPU context envelope",
            |a| a["final_test_targets_used"] = json!(true),
        ),
        (
            "an artifact of another input",
            "another release or input scope",
            |a| a["input_sha256"] = json!(sha256_hex(b"another input")),
        ),
        (
            "a context outside the frozen clip",
            "is not a finite, clipped 2048-f32 vector",
            |a| a["vector"][0] = json!(64.0),
        ),
        (
            "a short context",
            "is not a finite, clipped 2048-f32 vector",
            |a| {
                a["vector"].as_array_mut().unwrap().pop();
            },
        ),
    ];
    for (label, expected, change) in corrupt {
        let f = fixture_with(Task::Regression, 5, Some(preregistered()));
        let run = with_changed_artifact(&f, change);
        assert_refused(&f, &submission(&f, &run), expected, label);
    }

    // Corrupt on disk: the retained artifact no longer hashes to its content address.
    let f = fixture_with(Task::Regression, 5, Some(preregistered()));
    let run = producer_run(&f, &teaching_context(), &f.window, |_| {});
    let evidence = submission(&f, &run);
    let fitted = &evidence.context.as_ref().unwrap().fitted_state_sha256;
    fs::write(f.state.blob_path(fitted), b"{\"schema\":").unwrap();
    assert_refused(&f, &evidence, "is corrupt", "a corrupt retained artifact");
}

/// P1: the handoff's `admission_pins_sha256` must be the controller's pins for this run: the
/// release and W2 qualification the freeze preregistered, the W2 qualification's case-execution
/// pins, this run's artifact and input scope, and the controller's attestation of the retained
/// context execution. Pins the controller did not issue are refused.
#[test]
fn admission_pins_the_controller_did_not_issue_are_refused() {
    type Tamper = fn(&mut HandoffParts);
    let cases: [(&str, Tamper); 3] = [
        (
            "pins issued under other qualification case executions",
            |p| {
                p.case_execution_pins[0] = sha256_hex(b"another case execution");
                p.seal();
            },
        ),
        ("pins over another record of the context execution", |p| {
            p.outcome.consumption.cpu_seconds = Some(0.02)
        }),
        ("a digest that names no pins", |p| {
            p.handoff["admission_pins_sha256"] = sha256_hex(b"fixture admission pins").into()
        }),
    ];
    for (label, tamper) in cases {
        let f = fixture_with(Task::Regression, 5, Some(preregistered()));
        let run = producer_run(&f, &teaching_context(), &f.window, tamper);
        assert_refused(
            &f,
            &submission(&f, &run),
            "are not the controller's admission pins",
            label,
        );
    }

    // The freeze's registration preregistered other case-execution pins than the controller
    // issued this admission under.
    let mut other = preregistered();
    other
        .producer
        .as_mut()
        .unwrap()
        .qualification_execution_sha256 = Some(
        (0..8)
            .map(|case| sha256_hex(format!("another case execution {case}").as_bytes()))
            .collect(),
    );
    let f = fixture_with(Task::Regression, 5, Some(other));
    let run = producer_run(&f, &teaching_context(), &f.window, |_| {});
    assert_refused(
        &f,
        &submission(&f, &run),
        "are not the controller's admission pins",
        "a freeze that preregistered other case-execution pins",
    );

    // A registration that records no admission pins: nothing is issued or retained, and teacher
    // admission refuses the context with a clear, quarantining message.
    let mut unpinned = preregistered();
    unpinned
        .producer
        .as_mut()
        .unwrap()
        .qualification_execution_sha256 = None;
    let f = fixture_with(Task::Regression, 5, Some(unpinned));
    let run = producer_run(&f, &teaching_context(), &f.window, |_| {});
    let error = format!(
        "{:#}",
        retain_handed_off_context(&f.state, &f.evidence.freeze, &run).unwrap_err()
    );
    assert!(
        error.contains("records no controller admission pins")
            && error.contains("new, fully provenanced qualification"),
        "{error}"
    );
    for bytes in [&run.handoff, &run.outcome, &run.artifact, &run.context] {
        assert_eq!(f.state.read_blob(&sha256_hex(bytes)).unwrap(), None);
    }
    assert_eq!(authority(&f.state), nothing());
    assert_refused(
        &f,
        &f.evidence,
        "records no controller admission pins",
        "a registration without admission pins",
    );
}

/// `AdmissionPins`, as the handoff's `admission_pins_sha256` hashes them, with placeholder values.
const ADMISSION_PINS_LAYOUT: &str = r#"{"source_sha256":"s","producer_sha256":"p","artifact_sha256":"a","qualification_sha256":"q","reference_manifest_sha256":"m","context_execution_sha256":"x","qualification_execution_sha256":["c0","c1"],"verification_policy_sha256":"v","training":{"dataset_sha256":"d","prepared_sha256":"r","training_sha256":"t","fitting_rows_sha256":"f","cut":2},"input_sha256":"i","preprocessing_sha256":"e"}"#;
/// `ExecutionAttestation`, as the pins' `context_execution_sha256` hashes it.
const EXECUTION_ATTESTATION_LAYOUT: &str = r#"{"schema":"jope.cpu-context-execution/controller-v1","supervisor_receipt_sha256":"o","program_sha256":"p","input_sha256":"i","output_sha256":"a","completed":true,"cpu_limit":1,"memory_max":4096,"swap_max":0,"threads":1,"elapsed_seconds":0.25,"deadline_seconds":30.0}"#;
/// `CpuContextArtifact`'s fields, sorted.
const ARTIFACT_FIELDS: [&str; 16] = [
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
    assert_eq!(
        context::verify(&f.state, &policy, &record, &evidence, &Default::default()).unwrap(),
        (
            context.receipt_sha256.clone(),
            context.handoff.unwrap().handoff_sha256
        )
    );
    // The whole-development freeze's issue of the same run is not the inner comparison's context.
    evidence.context = Some(retain_handed_off_context(&f.state, &f.evidence.freeze, &run).unwrap());
    assert!(context::verify(&f.state, &policy, &record, &evidence, &Default::default()).is_err());
}
