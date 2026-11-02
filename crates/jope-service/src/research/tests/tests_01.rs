use super::*;

#[test]
fn capture_preserves_policy_and_evaluates_the_selected_population_once() {
    for round in 0..3 {
        for all in [false, true] {
            let mut input = capture_input(round, all);
            input.policy.admitted.reverse();
            let expected = if all {
                (0..input.proposals.len()).collect()
            } else {
                input.policy.admitted.clone()
            };
            let before = serde_json::to_value(&input).unwrap();
            let mut capture = RoundCapture::new(input, usize::MAX).unwrap();
            let mut called = vec![];
            while capture
                .evaluate_next(|index, proposal| {
                    assert_eq!(proposal.program_sha256, format!("{index:064x}"));
                    called.push(index);
                    Ok(measured(index))
                })
                .unwrap()
            {}
            assert_eq!(called, expected);
            assert_eq!(capture.attempted_indices(), expected);
            assert_eq!(capture.outcomes().len(), expected.len());
            let output = capture.finish().unwrap();
            assert!(output.capture_complete);
            let after = serde_json::to_value(&output).unwrap();
            for key in [
                "identity",
                "round",
                "seed",
                "tokens",
                "mask",
                "revealed",
                "proposals",
                "source_indices",
                "policy",
                "evaluate_all_proposals",
            ] {
                assert_eq!(after[key], before[key], "changed {key}");
            }
            assert!(output.outcomes.iter().any(|o| matches!(
                o.measurement,
                Rung0Measurement::Failed {
                    code: WorkerFailure::NonFinite
                }
            )));
        }
    }
}

#[test]
fn capture_attempt_budget_includes_failures_and_preserves_sixteen_seed_inputs() {
    for budget in [0, 1, 3, 16, usize::MAX] {
        let mut input = capture_input(0, true);
        input.proposals = (0..16).map(program).collect();
        input.source_indices = (0..16).collect();
        input.policy.admitted = (0..16).collect();
        let mut capture = RoundCapture::new(input, budget).unwrap();
        while capture.evaluate_next(|i, _| Ok(measured(i))).unwrap() {}
        let output = capture.finish().unwrap();
        assert_eq!(output.proposals.len(), 16);
        assert_eq!(output.policy.admitted.len(), 16);
        assert_eq!(output.outcomes.len(), budget.min(16));
        assert_eq!(output.capture_complete, budget >= 16);
    }
    let mut capture = RoundCapture::new(capture_input(1, true), 8).unwrap();
    assert!(capture.evaluate_next(|i, _| Ok(measured(i))).unwrap());
    assert!(!capture.finish().unwrap().capture_complete);
}

#[test]
fn capture_refuses_invalid_inputs_and_retains_unknown_attempts_without_retry() {
    let mut observed = record(1, true);
    assert!(RoundCapture::new(observed.clone(), 8).is_err());
    observed.dispatched.clear();
    observed.outcomes.clear();
    observed.capture_complete = false;
    observed.tokens.clear();
    assert!(RoundCapture::new(observed, 8).is_err());
    for nonfinite in [false, true] {
        let mut capture = RoundCapture::new(capture_input(1, true), 8).unwrap();
        capture.evaluate_next(|i, _| Ok(measured(i))).unwrap();
        assert!(capture
            .evaluate_next(|_, _| {
                if nonfinite {
                    Ok(Rung0Measurement::Success {
                        surrogate: f64::NAN,
                        objective: 0.0,
                        gate: true,
                    })
                } else {
                    anyhow::bail!("unknown worker result")
                }
            })
            .is_err());
        assert_eq!(capture.attempted_indices(), [0, 1]);
        assert_eq!(capture.outcomes().len(), 1);
        assert!(capture
            .evaluate_next(|_, _| panic!("must not retry"))
            .is_err());
        assert!(capture.finish().is_err());
    }
    let mut capture = RoundCapture::new(capture_input(1, true), 8).unwrap();
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = capture.evaluate_next(|_, _| panic!("aborted evaluator"));
    }));
    assert!(unwind.is_err());
    assert_eq!(capture.attempted_indices(), [0]);
    assert!(capture.outcomes().is_empty());
    assert!(capture.finish().is_err());
}

#[test]
fn capture_measures_evaluator_wall_time() {
    let mut capture = RoundCapture::new(capture_input(0, false), 1).unwrap();
    capture
        .evaluate_next(|i, _| {
            std::thread::sleep(std::time::Duration::from_millis(3));
            Ok(measured(i))
        })
        .unwrap();
    assert!(capture.outcomes()[0].wall_ms >= 3);
}

#[cfg(unix)]
#[test]
fn captured_rounds_read_in_python_and_complete_all_guard_refuses_biased_or_partial_streams() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../python/src/jope/north_star/record_reader.py");
    for all in [false, true] {
        for (budget, dispatches, failures, incomplete_rounds) in [
            (0, 0, 0, 3),
            (1, 3, 3, 3),
            (3, 8, 3, 2),
            (16, if all { 26 } else { 10 }, 3, 0),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("capture.jsonl");
            let mut writer = RecordWriter::create(&path).unwrap();
            for round in 0..3 {
                let mut capture = RoundCapture::new(capture_input(round, all), budget).unwrap();
                while capture.evaluate_next(|i, _| Ok(measured(i))).unwrap() {}
                writer.append(&capture.finish().unwrap()).unwrap();
            }
            let output = Command::new("python3")
                .arg(&script)
                .arg(&path)
                .args([
                    "--allow-synthetic",
                    "--allow-incomplete",
                    "--detailed-counts",
                ])
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let counts: Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(
                counts,
                json!({
                    "rounds":3, "proposals":26, "admissions":10,
                    "dispatches":dispatches, "outcomes":dispatches,
                    "successes":dispatches-failures,
                    "failures":failures, "incomplete_rounds":incomplete_rounds
                })
            );
            let guarded = Command::new("python3")
                .arg(&script)
                .arg(&path)
                .args([
                    "--allow-synthetic",
                    "--allow-incomplete",
                    "--require-evaluate-all",
                ])
                .output()
                .unwrap();
            assert_eq!(guarded.status.success(), all && budget == 16);
            if !guarded.status.success() {
                assert!(String::from_utf8_lossy(&guarded.stderr)
                    .contains("complete all-proposal capture required"));
                assert!(guarded.stdout.is_empty(), "no partial success summary");
            }
        }
    }
}

#[cfg(unix)]
#[test]
fn rust_stream_reads_in_python_with_exact_inventory() {
    for all in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("control.jsonl");
        let mut writer = RecordWriter::create(&path).unwrap();
        for round in 0..3 {
            writer.append(&record(round, all)).unwrap();
        }
        let output = reader(&path, true);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let summary: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(
            summary,
            json!({
                "rounds":3, "proposals":26, "admissions":10,
                "dispatches":if all {26} else {10}, "outcomes":if all {26} else {10},
                "successes":if all {23} else {7}, "failures":3, "incomplete_rounds":0
            })
        );
        let script = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../python/src/jope/north_star/record_reader.py");
        let legacy = Command::new("python3")
            .arg(script)
            .arg(&path)
            .arg("--allow-synthetic")
            .output()
            .unwrap();
        assert!(legacy.status.success());
        assert_eq!(
            serde_json::from_slice::<Value>(&legacy.stdout).unwrap(),
            json!({"rounds":3,"proposals":26,"outcomes":if all {26} else {10},"failures":3,"incomplete_rounds":0})
        );
        assert!(
            !reader(&path, false).status.success(),
            "synthetic controls must not enter default reader"
        );
        let schema: Value = serde_json::from_str(SCHEMA).unwrap();
        assert_eq!(schema["properties"]["schema"]["const"], VERSION);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        assert!(
            RecordWriter::create(&path).is_err(),
            "never overwrite an existing stream"
        );
        if let Some(directory) = std::env::var_os("QLT03_QA_RECORDS") {
            let output = Path::new(&directory).join(if all {
                "evaluate-all.jsonl"
            } else {
                "admitted.jsonl"
            });
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(output)
                .unwrap();
            file.write_all(&fs::read(&path).unwrap()).unwrap();
            file.sync_all().unwrap();
        }
    }
}

#[cfg(unix)]
#[test]
fn rust_float32_boundary_encodings_read_in_python() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("float-boundaries.jsonl");
    let mut row = record(0, true);
    row.tokens[0][0] = f32::MAX;
    row.tokens[0][1] = -f32::MAX;
    row.tokens[0][2] = f32::MIN_POSITIVE;
    row.tokens[0][3] = f32::from_bits(1);
    row.proposals[0].feat412[0] = f32::MAX;
    row.proposals[0].proj16[0] = f64::MAX;
    RecordWriter::create(&path).unwrap().append(&row).unwrap();
    let output = reader(&path, true);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn missing_failure_or_unadmitted_label_cannot_be_accepted_as_complete() {
    let mut all = record(1, true);
    all.outcomes.remove(0); // A failed evaluation is still a required observation.
    assert!(all.validate().is_err());
    let mut admitted = record(1, false);
    admitted.outcomes.push(RecordedOutcome {
        proposal_index: 7,
        wall_ms: 1,
        measurement: Rung0Measurement::Failed {
            code: WorkerFailure::Timeout,
        },
    });
    assert!(admitted.validate().is_err());
    let mut duplicate = record(1, true);
    duplicate.outcomes[1].proposal_index = 0;
    assert!(duplicate.validate().is_err());
}

#[test]
fn malformed_slate_context_costs_and_partitions_are_refused() {
    engine_provider_requires_native_inputs();
    super::native_inputs::input_controls(record(0, false));
    let mut variants = vec![];
    let mut r = record(1, true);
    r.identity.partition = "validation".into();
    variants.push(r);
    let mut r = record(1, true);
    r.tokens[0].pop();
    variants.push(r);
    let mut r = record(1, true);
    r.proposals[0].feat412[0] = f32::NAN;
    variants.push(r);
    let mut r = record(1, true);
    r.proposals[1] = r.proposals[0].clone();
    variants.push(r);
    let mut r = record(1, true);
    r.revealed[0].surrogate = f64::INFINITY;
    variants.push(r);
    let mut r = record(2, true);
    r.policy.slate_costs.as_mut().unwrap().pop();
    variants.push(r);
    let mut r = record(2, true);
    r.policy.slate_costs.as_mut().unwrap()[0] = f64::NAN;
    variants.push(r);
    let mut r = record(1, true);
    r.policy.admitted[1] = 0;
    variants.push(r);
    let mut r = record(1, true);
    r.policy.slate_costs = Some(vec![0.0; 12]);
    variants.push(r);
    let mut r = record(0, true);
    r.policy.admitted.pop();
    variants.push(r);
    let mut r = record(1, true);
    r.identity.dope_weights_sha256 = "2".repeat(64);
    variants.push(r);
    for (i, r) in variants.iter().enumerate() {
        assert!(r.validate().is_err(), "negative case {i}");
    }
}

#[cfg(unix)]
#[test]
fn writer_rejects_replays_and_input_changes_without_writing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("control.jsonl");
    let mut writer = RecordWriter::create(&path).unwrap();
    writer.append(&record(0, true)).unwrap();
    let saved = fs::read(&path).unwrap();
    assert!(writer.append(&record(0, true)).is_err());
    assert!(writer.append(&record(2, true)).is_err());
    let mut variants = vec![];
    let mut r = record(1, true);
    r.identity.family_sha256 = "2".repeat(64);
    variants.push(r);
    let mut r = record(1, true);
    r.seed += 1;
    variants.push(r);
    let mut r = record(1, true);
    r.tokens[0][0] = 1.0;
    variants.push(r);
    variants.push(record(1, false));
    for r in variants {
        assert!(writer.append(&r).is_err());
        assert_eq!(fs::read(&path).unwrap(), saved);
    }
    writer.append(&record(1, true)).unwrap(); // Validation refusal does not poison a valid stream.
}

#[test]
fn failed_io_poisoning_prevents_a_second_append() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let mut writer = RecordWriter {
        file: File::open(file.path()).unwrap(),
        binding: None,
        next_round: 0,
        poisoned: false,
    };
    assert!(writer.append(&record(0, true)).is_err());
    assert!(writer.poisoned);
    assert!(writer
        .append(&record(0, true))
        .unwrap_err()
        .to_string()
        .contains("poisoned"));
}
