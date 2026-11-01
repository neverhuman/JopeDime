use super::*;
use crate::worker::WorkerFailure;
#[cfg(unix)]
use serde_json::{json, Value};
#[cfg(unix)]
use std::{fs, process::Command};

fn program(index: usize) -> RecordedProgram {
    RecordedProgram {
        program_sha256: format!("{index:064x}"),
        operator_family: "synthetic_control".into(),
        feat412: vec![0.0; 412],
        proj16: vec![0.0; 16],
    }
}

fn record(round: u64, all: bool) -> ResearchRound {
    let n = if round == 0 {
        2
    } else if round == 1 {
        8
    } else {
        16
    };
    let admitted: Vec<_> = (0..if round == 0 { n } else { 4 }).collect();
    ResearchRound {
        schema: VERSION.into(),
        identity: RecordIdentity {
            run_id: "synthetic-control".into(),
            task: RecordTask::Regression,
            evidence: RecordEvidence::SyntheticControl,
            partition: "train".into(),
            source_commit: "a".repeat(40),
            corpus_manifest_sha256: "b".repeat(64),
            dataset_sha256: "c".repeat(64),
            family_sha256: "d".repeat(64),
            problem_sha256: "e".repeat(64),
            bundle_sha256: "f".repeat(64),
            conditioning_sha256: "0".repeat(64),
            evaluator_sha256: "1".repeat(64),
            dope_weights_sha256: DOPE_SHA256.into(),
        },
        round,
        seed: 29,
        tokens: vec![vec![0.0; 32]; 10],
        mask: vec![true; 10],
        revealed: if round == 0 {
            vec![]
        } else {
            (100..104)
                .map(|i| RecordedReveal {
                    program: program(i),
                    surrogate: i as f64,
                    gate: i % 2 == 0,
                })
                .collect()
        },
        proposals: (0..n).map(program).collect(),
        source_indices: (0..n).map(|i| (i * 2) as u64).collect(),
        policy: BehaviorPolicy {
            kind: if round < 2 {
                RecordedPolicyKind::Heuristic
            } else {
                RecordedPolicyKind::Learned
            },
            explore: round >= 2,
            slate_costs: (round >= 2).then(|| vec![0.0; 4 + n]),
            admitted: admitted.clone(),
        },
        evaluate_all_proposals: all,
        dispatched: if all {
            (0..n).collect()
        } else {
            admitted.clone()
        },
        capture_complete: true,
        outcomes: (if all {
            (0..n).collect::<Vec<_>>()
        } else {
            admitted
        })
        .into_iter()
        .map(|i| RecordedOutcome {
            proposal_index: i,
            wall_ms: 2,
            measurement: if i == 0 {
                Rung0Measurement::Failed {
                    code: WorkerFailure::NonFinite,
                }
            } else {
                Rung0Measurement::Success {
                    surrogate: 1.0,
                    objective: -3.0,
                    gate: false,
                }
            },
        })
        .collect(),
    }
}

fn capture_input(round: u64, all: bool) -> ResearchRound {
    let mut input = record(round, all);
    input.dispatched.clear();
    input.outcomes.clear();
    input.capture_complete = false;
    input
}

fn measured(index: usize) -> Rung0Measurement {
    if index == 0 {
        Rung0Measurement::Failed {
            code: WorkerFailure::NonFinite,
        }
    } else {
        Rung0Measurement::Success {
            surrogate: index as f64,
            objective: -(index as f64),
            gate: index.is_multiple_of(2),
        }
    }
}

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
fn reader(path: &Path, synthetic: bool) -> std::process::Output {
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../python/src/jope/north_star/record_reader.py");
    let mut command = Command::new("python3");
    command.arg(script).arg(path).arg("--detailed-counts");
    if synthetic {
        command.arg("--allow-synthetic");
    }
    command
        .output()
        .expect("Python3 is required for QLT-03 cross-language QA")
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

#[cfg(unix)]
#[test]
fn budget_truncation_preserves_policy_and_is_not_complete_training_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("truncated.jsonl");
    let mut writer = RecordWriter::create(&path).unwrap();
    writer.append(&record(0, true)).unwrap();
    let mut truncated = record(1, true);
    truncated.dispatched.truncate(2);
    truncated.outcomes.truncate(2);
    truncated.capture_complete = false;
    assert_eq!(truncated.policy.admitted.len(), 4);
    writer.append(&truncated).unwrap();
    assert!(!reader(&path, true).status.success());
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../python/src/jope/north_star/record_reader.py");
    let output = Command::new("python3")
        .arg(script)
        .arg(&path)
        .args([
            "--allow-synthetic",
            "--allow-incomplete",
            "--detailed-counts",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let summary: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        summary,
        json!({
            "rounds":2, "proposals":10, "admissions":6, "dispatches":4,
            "outcomes":4, "successes":2, "failures":2, "incomplete_rounds":1
        })
    );
    truncated.capture_complete = true;
    assert!(truncated.validate().is_err());
}

#[cfg(unix)]
#[test]
fn python_rejects_malformed_or_truncated_streams() {
    let zero = serde_json::to_value(record(0, true)).unwrap();
    let one = serde_json::to_value(record(1, true)).unwrap();
    let line = |v: &Value| format!("{}\n", serde_json::to_string(v).unwrap());
    let mut cases = vec![
        String::new(),
        line(&zero).trim_end().into(),
        format!("{}{}", line(&zero), line(&zero)),
        line(&one),
    ];
    for partition in ["val", "sealed", "holdout", "test"] {
        let mut bad = zero.clone();
        bad["identity"]["partition"] = json!(partition);
        cases.push(line(&bad));
    }
    let mut bad = one.clone();
    bad["outcomes"].as_array_mut().unwrap().remove(0);
    cases.push(format!("{}{}", line(&zero), line(&bad)));
    let mut bad = one.clone();
    bad["seed"] = json!(30);
    cases.push(format!("{}{}", line(&zero), line(&bad)));
    let mut bad = zero.clone();
    bad["tokens"][0][0] = json!(1e39);
    cases.push(line(&bad));
    let mut bad = zero.clone();
    bad["seed"] = json!(false);
    cases.push(line(&bad));
    let mut bad = zero.clone();
    bad["extra"] = json!(0);
    cases.push(line(&bad));
    cases.push(line(&zero).replacen("\"schema\":", "\"seed\":29,\"schema\":", 1));
    cases.push(line(&zero).replacen("0.0", "NaN", 1));
    for (i, bytes) in cases.iter().enumerate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.jsonl");
        fs::write(&path, bytes).unwrap();
        let output = reader(&path, true);
        assert!(!output.status.success(), "Python negative case {i}");
        assert!(
            output.stdout.is_empty(),
            "no partial success summary: case {i}"
        );
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("encoded.jsonl");
    reject_non_utf8_jsonl(&path, &line(&zero), &line(&one));
}

#[cfg(unix)]
fn reject_non_utf8_jsonl(path: &Path, seed_line: &str, next_line: &str) {
    for valid in [seed_line.to_owned(), format!("{seed_line}{next_line}")] {
        fs::write(path, valid).unwrap();
        let output = reader(path, true);
        assert!(output.status.success(), "valid UTF-8 JSONL");
        assert!(!output.stdout.is_empty());
    }
    for suffix in [false, true] {
        let raw = if suffix { next_line } else { seed_line };
        assert!(raw.is_ascii(), "these byte-padding fixtures require ASCII");
        let mut bom = b"\xef\xbb\xbf".to_vec();
        bom.extend_from_slice(raw.as_bytes());
        let utf16: Vec<u8> = raw.bytes().flat_map(|byte| [0, byte]).collect();
        let utf32: Vec<u8> = raw.bytes().flat_map(|byte| [0, 0, 0, byte]).collect();
        for (encoding, invalid) in [
            ("UTF-8 BOM", bom),
            ("UTF-16 BE", utf16),
            ("UTF-32 BE", utf32),
        ] {
            let mut bytes = if suffix {
                seed_line.as_bytes().to_vec()
            } else {
                vec![]
            };
            bytes.extend(invalid);
            fs::write(path, bytes).unwrap();
            let output = reader(path, true);
            assert!(
                !output.status.success(),
                "accepted {encoding}, suffix={suffix}"
            );
            assert!(
                output.stdout.is_empty(),
                "partial success for {encoding}, suffix={suffix}"
            );
        }
    }
}

#[cfg(not(unix))]
#[test]
fn non_unix_private_file_creation_is_explicitly_unavailable() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("records.jsonl");
    match RecordWriter::create(&path) {
        Err(error) => assert!(error.to_string().contains("not qualified")),
        Ok(_) => panic!("unqualified private-file creation unexpectedly succeeded"),
    }
    assert!(!path.exists());
}

#[test]
fn private_capture_envelope_refuses_unknown_fields_and_oversized_cas_references() {
    use capture::{Binding, CaptureState};
    let home = tempfile::tempdir().unwrap();
    let store = crate::store::Store::open(home.path()).unwrap();
    let r = record(0, true);
    let binding = Binding {
        config: CaptureConfig {
            max_attempts: 16,
            max_rounds: 16,
            max_bytes: 1024 * 1024,
            evaluate_all_proposals: true,
            seed_protocol_sha256: "2".repeat(64),
        },
        identity: r.identity,
        seed: r.seed,
        tokens: r.tokens,
        mask: r.mask,
        matrix_sha256: "3".repeat(64),
    };
    let mut value = serde_json::to_value(&binding).unwrap();
    value["arbitrary_import"] = json!(true);
    let reference = store
        .publish_blob(std::io::Cursor::new(serde_json::to_vec(&value).unwrap()))
        .unwrap();
    assert!(CaptureState::read::<Binding>(&store, &reference).is_err());
    let mut too_large = reference;
    too_large.bytes = u64::MAX;
    assert!(CaptureState::read::<Binding>(&store, &too_large)
        .unwrap_err()
        .to_string()
        .contains("ceiling"));
}

#[test]
fn auxiliary_journal_requires_readable_results_and_matching_attempts() {
    use crate::worker::{Attempt, Limits, Request, Response};
    use capture::{AuxiliaryAttempt, AuxiliaryResult, Binding, CaptureState};

    let home = tempfile::tempdir().unwrap();
    let store = crate::store::Store::open(home.path()).unwrap();
    let mut row = record(0, true);
    row.identity.run_id = jope_api::ids::RunId::new().to_string();
    let binding = Binding {
        config: CaptureConfig {
            max_attempts: 16,
            max_rounds: 16,
            max_bytes: 1024 * 1024,
            evaluate_all_proposals: true,
            seed_protocol_sha256: "2".repeat(64),
        },
        identity: row.identity,
        seed: row.seed,
        tokens: row.tokens,
        mask: row.mask,
        matrix_sha256: "3".repeat(64),
    };
    let bytes = serde_json::to_vec(&binding).unwrap();
    let mut state = CaptureState {
        version: 1,
        binding: store.publish_blob(std::io::Cursor::new(&bytes)).unwrap(),
        config: binding.config.clone(),
        charged: 0,
        bytes: bytes.len() as u64,
        label_counter: 0,
        tainted: false,
        rounds: Default::default(),
        auxiliary: vec![],
    };
    let request = Request {
        attempt: Attempt {
            run: binding.identity.run_id.parse().unwrap(),
            round: 0,
            slot: 0,
            attempt: 0,
        },
        matrix_path: "synthetic.jmat".into(),
        fold_spec: serde_json::json!({}),
        dsl: "synthetic_control".into(),
        objective: serde_json::json!("rmse"),
        seed: 29,
        limits: Limits {
            wall_ms: 100,
            rss_bytes: 1024,
        },
    };
    let response = Response {
        attempt: request.attempt.clone(),
        scores: vec![1.0, 1.0, 1.0],
        fold_scores: vec![],
        predictions_digest: "4".repeat(64),
        wall_ms: 2,
        failure: None,
    };
    let request_ref = state.publish(&store, &request).unwrap();
    state.auxiliary.push(AuxiliaryAttempt {
        request: request_ref,
        result: None,
        model: None,
    });
    state.validate_storage(&store).unwrap();
    assert!(
        state.unknown(),
        "pending attempts remain available for inspection"
    );

    let valid = state
        .publish(
            &store,
            &AuxiliaryResult::Evaluation {
                response: response.clone(),
            },
        )
        .unwrap();
    state.auxiliary[0].result = Some(valid);
    state.validate_storage(&store).unwrap();
    assert!(!state.unknown());

    let unreadable = state.publish(&store, &serde_json::Value::Null).unwrap();
    state.auxiliary[0].result = Some(unreadable);
    assert!(
        state.validate_storage(&store).is_err(),
        "valid-hash null is not a known result"
    );

    let mut wrong_attempt = response.clone();
    wrong_attempt.attempt.attempt += 1;
    let reference = state
        .publish(
            &store,
            &AuxiliaryResult::Evaluation {
                response: wrong_attempt,
            },
        )
        .unwrap();
    state.auxiliary[0].result = Some(reference);
    assert!(
        state.validate_storage(&store).is_err(),
        "response must match its reservation"
    );

    for failure in [WorkerFailure::Timeout, WorkerFailure::StaleAttempt] {
        let reference = state
            .publish(
                &store,
                &AuxiliaryResult::Evaluation {
                    response: Response::failed(request.attempt.clone(), failure, 2),
                },
            )
            .unwrap();
        state.auxiliary[0].result = Some(reference);
        assert_eq!(
            state.validate_storage(&store).is_ok(),
            failure == WorkerFailure::Timeout
        );
    }

    for worker in [
        None,
        Some(Response::failed(
            request.attempt.clone(),
            WorkerFailure::Timeout,
            2,
        )),
    ] {
        let reference = state
            .publish(&store, &AuxiliaryResult::Failure { worker })
            .unwrap();
        state.auxiliary[0].result = Some(reference);
        state.validate_storage(&store).unwrap();
    }
    for worker in [
        response.clone(),
        Response::failed(request.attempt.clone(), WorkerFailure::StaleAttempt, 2),
    ] {
        let reference = state
            .publish(
                &store,
                &AuxiliaryResult::Failure {
                    worker: Some(worker),
                },
            )
            .unwrap();
        state.auxiliary[0].result = Some(reference);
        assert!(state.validate_storage(&store).is_err());
    }
    let mut invalid = response.clone();
    invalid.scores.clear();
    let reference = state
        .publish(&store, &AuxiliaryResult::Evaluation { response: invalid })
        .unwrap();
    state.auxiliary[0].result = Some(reference);
    assert!(state.validate_storage(&store).is_err());

    let fit = state.publish(&store, &AuxiliaryResult::Fit).unwrap();
    state.auxiliary[0].result = Some(fit);
    assert!(
        state.validate_storage(&store).is_err(),
        "successful fit requires its model root"
    );
    state.auxiliary[0].model = Some(state.publish(&store, &vec![1_u8, 2, 3]).unwrap());
    state.validate_storage(&store).unwrap();

    let mut foreign = request;
    foreign.attempt.run = jope_api::ids::RunId::new();
    state.auxiliary[0].request = state.publish(&store, &foreign).unwrap();
    assert!(
        state.validate_storage(&store).is_err(),
        "request must belong to the bound run"
    );
}

// Boundary-only control: never exports an Engine record or claims corpus evidence.
fn engine_provider_requires_native_inputs() {
    use super::capture::{Binding, CaptureProvider, SlateObservation};
    use crate::{
        discover::r#loop::Plan,
        worker::{Request, Response},
    };
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    struct Unbound(Arc<AtomicUsize>);
    impl CaptureProvider for Unbound {
        fn validate_binding(&self, _: &Binding, _: &Plan) -> Result<()> {
            Ok(())
        }
        fn observe(
            &self,
            _: u64,
            dsls: &[String],
            _: &[RecordedReveal],
        ) -> Result<SlateObservation> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(SlateObservation {
                programs: dsls
                    .iter()
                    .map(|dsl| RecordedProgram {
                        program_sha256: super::capture::digest(dsl.as_bytes()),
                        operator_family: "seed".into(),
                        feat412: vec![0.0; 412],
                        proj16: vec![0.0; 16],
                    })
                    .collect(),
                revealed_indices: vec![],
            })
        }
        fn measurement(&self, _: &Request, _: &Response) -> Result<Rung0Measurement> {
            anyhow::bail!("not executed")
        }
        fn label_seed(&self, _: u64, _: usize) -> Result<u64> {
            anyhow::bail!("not executed")
        }
    }
    let r = record(0, false);
    let calls = Arc::new(AtomicUsize::new(0));
    let mut identity = r.identity;
    identity.evidence = RecordEvidence::Engine;
    let context = CaptureContext {
        binding: Binding {
            config: CaptureConfig {
                max_attempts: 2,
                max_rounds: 1,
                max_bytes: 1_000_000,
                evaluate_all_proposals: false,
                seed_protocol_sha256: "2".repeat(64),
            },
            identity,
            seed: r.seed,
            tokens: r.tokens,
            mask: r.mask,
            matrix_sha256: "2".repeat(64),
        },
        provider: Arc::new(Unbound(calls.clone())),
    };
    assert!(
        context
            .observe(0, &["(pipe X (ridge))".into()], &[])
            .is_err(),
        "Engine provider accepted fabricated features without a native input closure"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "refuse before provider observation"
    );
}
