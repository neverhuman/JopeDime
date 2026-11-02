use super::*;

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
