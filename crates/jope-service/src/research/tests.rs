use super::*;

#[path = "tests/tests_01.rs"]
mod tests_01;
#[path = "tests/tests_02.rs"]
mod tests_02;

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
        "Engine provider accepted forged features without a native input closure"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "refuse before provider observation"
    );
}
