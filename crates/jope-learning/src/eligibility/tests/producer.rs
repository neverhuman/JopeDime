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
    let mut parts = handoff_parts(
        &f.state,
        &sha256_hex(context),
        &f.plan.comparison,
        &f.experiment,
        window,
    );
    tamper(&mut parts);
    ContextRun {
        handoff: serde_json::to_vec(&parts.handoff).unwrap(),
        job: canonical_json(&parts.job).unwrap(),
        outcome: serde_json::to_vec(&parts.outcome).unwrap(),
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
