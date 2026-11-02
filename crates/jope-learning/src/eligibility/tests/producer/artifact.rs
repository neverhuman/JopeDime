use super::*;

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
