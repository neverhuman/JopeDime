use super::*;
use crate::worker::WorkerFailure;
use serde_json::{json, Value};
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

fn reader(path: &Path, synthetic: bool) -> std::process::Output {
    let script = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../python/src/jope/north_star/record_reader.py");
    let mut command = Command::new("python3");
    command.arg(script).arg(path);
    if synthetic {
        command.arg("--allow-synthetic");
    }
    command
        .output()
        .expect("Python3 is required for QLT-03 cross-language QA")
}

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
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
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
        .args(["--allow-synthetic", "--allow-incomplete"])
        .output()
        .unwrap();
    assert!(output.status.success());
    let summary: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(summary["incomplete_rounds"], 1);
    assert_eq!(summary["outcomes"], 4);
    truncated.capture_complete = true;
    assert!(truncated.validate().is_err());
}

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
        assert!(
            !reader(&path, true).status.success(),
            "Python negative case {i}"
        );
    }
}
