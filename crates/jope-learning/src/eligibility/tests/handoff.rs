//! W3-T02 and W2-T07 at the real teaching ingress: teacher admission consumes the W2 context
//! handoff (consumer, release identity, transformation chain, W2 qualification, input scope and
//! context digest) and the retained W2 execution proof, authenticated against the producer the
//! freeze preregistered, instead of trusting the encoder receipt's own claims. Every refusal goes
//! through `qualify` and `hwm::ingest` and leaves every authority-bearing state unchanged
//! (`assert_refused`); the positive control admits for every task. Synthetic fixtures only:
//! nothing is encoded or executed, and this is not a live W2 producer run.
use super::provenance::{assert_positive_control, assert_refused, authority};
use super::*;
use crate::revocation::testing;

/// A submission whose handoff was rebuilt from `f`'s context after `tamper`.
fn tampered(f: &Fixture, tamper: impl FnOnce(&mut HandoffParts)) -> OofEvidence {
    tampered_in(f, &f.window, tamper)
}

fn tampered_in(
    f: &Fixture,
    window: &ExecutionWindow,
    tamper: impl FnOnce(&mut HandoffParts),
) -> OofEvidence {
    let context = f.evidence.context.clone().unwrap();
    let mut parts = handoff_parts(
        &f.state,
        &context_bytes(&f.state, &context),
        &f.plan.comparison,
        &f.experiment,
        window,
    );
    tamper(&mut parts);
    let mut evidence = f.evidence.clone();
    evidence.context = Some(ContextEvidence {
        handoff: Some(retain_handoff(&f.state, &parts, &f.experiment, window)),
        ..context
    });
    evidence
}

#[test]
fn teacher_admission_consumes_an_authenticated_w2_handoff() {
    type Case = (&'static str, &'static str, fn(&mut HandoffParts));
    let cases: [Case; 13] = [
        (
            "handed off to the accepted policy proposal worker",
            "handed off for consumer",
            |p| p.handoff["consumer"] = "accepted_policy_proposal".into(),
        ),
        (
            "another producer executable in the release",
            "ProducerNotAuthenticated",
            |p| p.handoff["release"]["producer_sha256"] = sha256_hex(b"another producer").into(),
        ),
        ("another encoder closure", "ProducerNotAuthenticated", |p| {
            p.handoff["release"]["encoder"]["foundation_weights"] =
                sha256_hex(b"other weights").into()
        }),
        (
            "another W2 qualification",
            "ProducerNotAuthenticated",
            |p| p.handoff["qualification_sha256"] = sha256_hex(b"another qualification").into(),
        ),
        (
            "another transformation chain",
            "transformation chain",
            |p| p.handoff["transformations"][0] = "another.transformation/v1".into(),
        ),
        (
            "a handoff of another context",
            "is not the handed-off context",
            |p| p.handoff["context_sha256"] = sha256_hex(b"another context").into(),
        ),
        ("a handoff scoped to another dataset", "input scope", |p| {
            p.handoff["scope"]["training"]["dataset_sha256"] = sha256_hex(b"another dataset").into()
        }),
        (
            "an unknown handoff field",
            "not a jope.cpu-context-handoff/v1 record",
            |p| p.handoff["issued_by"] = "the candidate".into(),
        ),
        (
            "an execution of another executable",
            "ProducerNotAuthenticated",
            |p| {
                p.job.command.program.sha256 = sha256_hex(b"another executable");
                p.reseal();
            },
        ),
        (
            "a failed context execution",
            "UnprovenContextExecution",
            |p| p.outcome.exit_code = Some(1),
        ),
        (
            "an artifact the job never wrote",
            "UnprovenContextExecution",
            |p| p.handoff["artifact_sha256"] = sha256_hex(b"an artifact from elsewhere").into(),
        ),
        (
            "an input the job never read",
            "UnprovenContextExecution",
            |p| p.handoff["scope"]["input_sha256"] = sha256_hex(b"another input").into(),
        ),
        (
            "an execution without the pool memory cap",
            "UnprovenContextExecution",
            |p| p.outcome.consumption.fit_memory_limit_bytes = None,
        ),
    ];
    for (label, expected, tamper) in cases {
        let f = fixture(Task::Regression, 5);
        assert_refused(&f, &tampered(&f, tamper), expected, label);
    }

    // A context executed in a window that opened before the freeze preregistered its producer.
    let f = fixture(Task::Regression, 5);
    let early = ExecutionWindow {
        experiment_sha256: f.window.experiment_sha256.clone(),
        started_unix_ms: f.window.started_unix_ms - 120_000,
        deadline_unix_ms: f.window.deadline_unix_ms - 120_000,
    };
    assert_refused(
        &f,
        &tampered_in(&f, &early, |_| {}),
        "before the freeze",
        "context executed before the freeze",
    );

    // No handoff at all: a self-declared encoder receipt is not a produced context.
    let f = fixture(Task::Regression, 5);
    let mut evidence = f.evidence.clone();
    evidence.context.as_mut().unwrap().handoff = None;
    assert_refused(&f, &evidence, "MissingContextHandoff", "no handoff");
    assert!(
        format!("{:#}", qualify(&f.state, &f.record, &evidence).unwrap_err())
            .contains("new, fully provenanced qualification")
    );

    // A revoked handoff (W12-T05) refuses the teacher that consumed it.
    let f = fixture(Task::Regression, 5);
    let handoff = f
        .evidence
        .context
        .as_ref()
        .unwrap()
        .handoff
        .clone()
        .unwrap();
    let bytes = f.state.read_blob(&handoff.handoff_sha256).unwrap().unwrap();
    let repo = tempfile::tempdir().unwrap();
    testing::revoke(
        &f.state,
        repo.path(),
        &[("E-context-handoff", &bytes, &[])],
        &["E-context-handoff"],
    );
    assert_refused(&f, &f.evidence, "RevokedEvidence", "revoked handoff");

    // Positive control, for each task: the preregistered producer's handoff admits, and the
    // qualification records the handoff it was admitted under.
    for task in [Task::Regression, Task::Binary, Task::Multiclass] {
        let f = fixture(task, 5);
        assert_positive_control(&f);
        let qualification = revalidate_qualification(&f.state, &f.record).unwrap();
        assert_eq!(
            qualification.context_handoff_sha256,
            Some(
                f.evidence
                    .context
                    .as_ref()
                    .unwrap()
                    .handoff
                    .as_ref()
                    .unwrap()
                    .handoff_sha256
                    .clone()
            )
        );
    }
}

#[test]
fn a_freeze_without_a_preregistered_producer_stays_quarantined() {
    let plan = ContextPlan {
        producer: None,
        ..context_plan()
    };
    let f = fixture_with(Task::Regression, 5, Some(plan));
    assert_refused(
        &f,
        &f.evidence,
        "PreHandoffProvenance",
        "freeze without a producer",
    );
    let error = format!(
        "{:#}",
        qualify(&f.state, &f.record, &f.evidence).unwrap_err()
    );
    assert!(
        error.contains("new, fully provenanced qualification"),
        "{error}"
    );
}

/// Old formats keep their bytes and identities: a freeze plan with no producer, context evidence
/// with no handoff and a qualification with no handoff digest parse and re-serialize unchanged.
/// A stored pre-handoff qualification is quarantined (owner decision D4), and presenting a
/// handed-off context later cannot upgrade it in place.
#[test]
fn pre_handoff_formats_keep_their_meaning_and_stay_quarantined() {
    let old_plan = json!({"source_family": "family", "encoder": context_plan().encoder});
    let parsed: ContextPlan = serde_json::from_value(old_plan.clone()).unwrap();
    assert_eq!(parsed.producer, None);
    assert_eq!(serde_json::to_value(&parsed).unwrap(), old_plan);
    let old_context = json!({"receipt_sha256": "a".repeat(64),
        "fitted_state_sha256": "b".repeat(64), "output_sha256": "c".repeat(64)});
    let parsed: ContextEvidence = serde_json::from_value(old_context.clone()).unwrap();
    assert_eq!(parsed.handoff, None);
    assert_eq!(serde_json::to_value(&parsed).unwrap(), old_context);

    let f = fixture(Task::Regression, 5);
    let qualified = qualify(&f.state, &f.record, &f.evidence).unwrap();
    assert_eq!(ingest(&f).accepted, 1);
    let mut old = qualified.clone();
    old.context_handoff_sha256 = None;
    let mut evidence = f.evidence.clone();
    evidence.context.as_mut().unwrap().handoff = None;
    let bytes = canonical_json(&StoredQualification {
        evidence,
        qualification: old.clone(),
    })
    .unwrap();
    assert!(!String::from_utf8_lossy(&bytes).contains("handoff"));
    let path = qualification_path(&f.state, &f.record);
    fs::write(&path, &bytes).unwrap();
    let manifest = freeze_teaching(&f.state).unwrap();
    assert!(manifest.entries.is_empty());
    assert!(
        manifest.exclusions[0].1.contains("MissingContextHandoff")
            && manifest.exclusions[0]
                .1
                .contains("new, fully provenanced qualification"),
        "{:?}",
        manifest.exclusions
    );
    assert!(revalidate_qualification(&f.state, &f.record).is_err());
    assert!(revalidate_replay(&f.state, &[old]).is_err());
    // The handed-off evidence cannot upgrade the retained decision in place.
    let before = authority(&f.state);
    assert!(qualify(&f.state, &f.record, &f.evidence).is_err());
    assert_eq!(fs::read(&path).unwrap(), bytes);
    assert_eq!(before.qualified, authority(&f.state).qualified);
    assert!(freeze_teaching(&f.state).unwrap().entries.is_empty());
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
        json!({"kind":"w3-t02-handoff"}),
    )
    .unwrap()
}
