use super::*;

#[test]
fn original_success_retains_model_zero_quality_and_adopts_once() {
    let fixture = Fixture::new(true);
    let (mut a, f, start, observed) = run_child(&fixture, "success");
    let sha = a
        .complete_observed(
            &f,
            observed,
            ExecutionOutcome::Success,
            crate::eval_registry::CompletionArtifacts {
                output: b"original output",
                model: Some(b"original model"),
                predictions: Some(b"original predictions"),
                metrics: Some(b"{\"rmse\":0.0}"),
            },
        )
        .unwrap();
    assert_eq!(
        a.completion(&sha).unwrap().outcome,
        ExecutionOutcome::Success
    );
    assert_eq!(a.model_for_completion(&sha).unwrap(), b"original model");
    let receipt = a.verify_committed_completion(&sha).unwrap();
    for (digest, expected) in [
        (
            receipt.output_sha256.as_str(),
            b"original output".as_slice(),
        ),
        (
            receipt.model_sha256.as_deref().unwrap(),
            b"original model".as_slice(),
        ),
        (
            receipt.predictions_sha256.as_deref().unwrap(),
            b"original predictions".as_slice(),
        ),
        (
            receipt.metrics_sha256.as_deref().unwrap(),
            b"{\"rmse\":0.0}".as_slice(),
        ),
    ] {
        assert_eq!(a.read_blob(digest).unwrap(), expected);
        assert_eq!(hash(expected), digest);
    }

    a.commit_original_completion(&f, &start, &sha).unwrap();
    assert_eq!(state(a.claim(&f, &request()).unwrap()), "completed");
    let model = a.completion(&sha).unwrap().model_sha256.unwrap();
    fs::remove_file(fixture.root.join("objects").join(model)).unwrap();
    assert!(a.model_for_completion(&sha).is_err());
    assert_eq!(state(a.claim(&f, &request()).unwrap()), "completed");
}

#[test]
fn failed_exit_signal_and_timeout_cannot_be_cleared_by_result_files() {
    for mode in ["failed", "signal", "timeout"] {
        let fixture = Fixture::new(true);
        let (mut a, f, _, observed) = run_child(&fixture, mode);
        if mode == "failed" {
            assert!(observed.exit_code.is_some_and(|code| code != 0));
            assert!(observed.signal.is_none());
        }
        let sha = a
            .complete_observed(
                &f,
                observed,
                ExecutionOutcome::Success,
                crate::eval_registry::CompletionArtifacts {
                    output: b"valid-looking result",
                    model: Some(b"model"),
                    predictions: Some(b"predictions"),
                    metrics: Some(b"{\"rmse\":0}"),
                },
            )
            .unwrap();
        let receipt = a.completion(&sha).unwrap();
        assert_ne!(receipt.outcome, ExecutionOutcome::Success);
        assert!(receipt.metrics_sha256.is_none());
        if mode == "timeout" {
            assert_eq!(receipt.outcome, ExecutionOutcome::Timeout);
        }
        assert_eq!(state(a.claim(&f, &request()).unwrap()), "terminal_failure");
    }
}

#[test]
fn preparation_delay_does_not_spend_execution_deadline() {
    let fixture = Fixture::new(true);
    let (mut a, f, start, observed) =
        run_child_with_preparation_delay(&fixture, "failed", Duration::from_secs(31));
    let prepared: u64 = fs::read_to_string(fixture._temp.path().join("start.ready"))
        .unwrap()
        .parse()
        .unwrap();
    assert!(start.authorized_monotonic_ns >= prepared);
    assert_eq!(
        start.deadline_monotonic_ns - start.authorized_monotonic_ns,
        30_000_000_000
    );
    let sha = a
        .complete_observed(
            &f,
            observed,
            ExecutionOutcome::Success,
            crate::eval_registry::CompletionArtifacts {
                output: b"valid-looking result",
                model: Some(b"model"),
                predictions: Some(b"predictions"),
                metrics: Some(b"{\"rmse\":0}"),
            },
        )
        .unwrap();
    let receipt = a.completion(&sha).unwrap();
    assert_ne!(receipt.outcome, ExecutionOutcome::Success);
    assert!(receipt.metrics_sha256.is_none());
    assert_eq!(state(a.claim(&f, &request()).unwrap()), "terminal_failure");
}

#[test]
fn preparation_failure_does_not_authorize_the_child() {
    let fixture = Fixture::new(true);
    let Err(failed) = std::panic::catch_unwind(|| {
        run_child_with_preparation_delay(&fixture, "prepare_failed", Duration::ZERO)
    }) else {
        panic!("preparation failure was accepted");
    };
    let message = failed
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| failed.downcast_ref::<&str>().copied());
    assert_eq!(
        message,
        Some("synthetic executor exited during preparation")
    );
    assert!(!fixture._temp.path().join("start.json").exists());
    assert!(!fixture._temp.path().join("start.ready").exists());
    assert_eq!(
        state(fixture.open().claim(&fit(), &request()).unwrap()),
        "claimed"
    );
}

#[test]
fn valid_sqlite_with_deleted_safety_rows_is_corruption() {
    for table in [
        "entries",
        "coverage",
        "exclusions",
        "scope_training",
        "numerical_guards",
    ] {
        let fixture = Fixture::new(true);
        let (mut a, f, _) = fixture.claim();
        a.import_exclusion(&HistoricalExclusion {
            scope: "synthetic".into(),
            canonical_program: Some("(pipe X (unrelated))".into()),
            training_sha256: None,
            evidence_sha256: hash(b"old"),
            reason: "unrelated historical program".into(),
        })
        .unwrap();
        let other = Connection::open(fixture.root.join("registry.sqlite")).unwrap();
        other.execute(&format!("DELETE FROM {table}"), []).unwrap();
        drop(other);
        assert!(
            a.claim(&f, &request()).is_err(),
            "live handle failed to detect {table} deletion"
        );
        drop(a);
        assert!(
            Authority::open(&fixture.root, &fixture.witness).is_err(),
            "restart failed to detect {table} deletion"
        );
    }
}

#[test]
fn new_binary_or_runtime_is_not_numerical_novelty() {
    let fixture = Fixture::new(true);
    let (mut authority, fit, _) = fixture.claim();
    for (exe, runtime) in [
        (
            hash(b"UI-only rebuild"),
            fit.identity.runtime_sha256.clone(),
        ),
        (
            fit.identity.executable_sha256.clone(),
            hash(b"new runtime label"),
        ),
    ] {
        let alias = VerifiedFit::from_verified_runtime(
            &fit.identity.canonical_program,
            crate::eval_registry::FitInputs {
                x: fit.x.clone(),
                y: fit.y.clone(),
                preprocessing: fit.preprocessing.clone(),
                feature_view: fit.feature_view.clone(),
            },
            fit.identity.fit_seed,
            exe,
            runtime,
            &contract(),
        )
        .unwrap();
        assert_ne!(alias.key, fit.key);
        assert!(state(authority.claim(&alias, &request()).unwrap()).contains("same numerical work"));
    }
}

#[test]
fn expired_start_and_execution_custody_remain_unavailable() {
    let fixture = Fixture::new(true);
    let (mut authority, fit, claim) = fixture.claim();
    let start = authority
        .authorize_with_deadline(
            &fit,
            &claim,
            &ProcessIdentity::current().unwrap(),
            Duration::from_millis(20),
        )
        .unwrap();
    // Synthetic custody isolates deadline behavior from expensive runtime hashes.
    // No numerical start is consumed or executed by this fixture.
    let custody = ExecutionCustody {
        start: start.clone(),
        lock: FileLock::acquire(&fixture.root.join("custody").join(fit.key()), true, true).unwrap(),
    };
    while monotonic_ns().unwrap() < start.deadline_monotonic_ns {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(custody
        .verify()
        .unwrap_err()
        .to_string()
        .contains("deadline expired"));
    assert!(authority
        .consume_start(&fit, &start)
        .err()
        .unwrap()
        .to_string()
        .contains("deadline expired"));
    assert_eq!(
        state(authority.lookup(&fit, &request()).unwrap()),
        "authorized"
    );
}

#[test]
fn synthetic_child_is_reaped_before_fixture_cleanup_on_panic() {
    let mut process = None;
    let mut path = None;
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let temp = tempfile::tempdir().unwrap();
        path = Some(temp.path().to_path_buf());
        let child = SyntheticChild(Command::new("/bin/sleep").arg("60").spawn().unwrap());
        process = Some(ProcessIdentity::read(child.0.id()).unwrap());
        panic!("injected synthetic readiness failure");
    }));
    assert!(outcome.is_err());
    let original = process.unwrap();
    assert_ne!(ProcessIdentity::read(original.pid).ok(), Some(original));
    assert!(!path.unwrap().exists());
}
