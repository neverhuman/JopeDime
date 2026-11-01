use super::*;
use crate::{
    dataset::jmat,
    discover::r#loop::HeuristicAdmission,
    research::{
        capture::{digest, Binding, CaptureProvider, SlateObservation},
        *,
    },
    store::RecordKey,
    worker::{Request, Response, WorkerFailure},
};
use serde_json::{json, Value};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Control;
impl SearchSpace for Control {
    fn canonical(&self, dsl: &str) -> Result<String> {
        Ok(dsl.into())
    }
    fn seeds(&self) -> Result<Vec<String>> {
        Ok(vec!["control-0".into(), "control-1".into()])
    }
    fn propose(
        &self,
        _: &[String],
        _: usize,
        _: &mut rand_chacha::ChaCha12Rng,
    ) -> Result<Vec<String>> {
        anyhow::bail!("the bounded control has only one seed round")
    }
}
impl CaptureProvider for Control {
    fn validate_binding(&self, _: &Binding, _: &Plan) -> Result<()> {
        Ok(())
    }
    fn observe(&self, _: u64, dsls: &[String], _: &[RecordedReveal]) -> Result<SlateObservation> {
        Ok(SlateObservation {
            programs: dsls
                .iter()
                .map(|dsl| RecordedProgram {
                    program_sha256: digest(dsl.as_bytes()),
                    operator_family: "synthetic_control".into(),
                    feat412: vec![0.0; 412],
                    proj16: vec![0.0; 16],
                })
                .collect(),
            revealed_indices: vec![],
        })
    }
    fn measurement(&self, _: &Request, response: &Response) -> Result<Rung0Measurement> {
        Ok(match response.failure {
            Some(code) => Rung0Measurement::Failed { code },
            None => Rung0Measurement::Success {
                surrogate: response.scores[0],
                objective: response.scores[1],
                gate: false,
            },
        })
    }
    fn label_seed(&self, _: u64, index: usize) -> Result<u64> {
        Ok(100 + index as u64)
    }
}
struct ControlExecutor(AtomicUsize);
impl Executor for ControlExecutor {
    fn evaluate(&self, request: &Request) -> Response {
        self.0.fetch_add(1, Ordering::SeqCst);
        if request.dsl.ends_with('0') {
            Response::failed(request.attempt.clone(), WorkerFailure::NonFinite, 0)
        } else {
            Response {
                attempt: request.attempt.clone(),
                scores: vec![1.0, 2.0, 2.0],
                fold_scores: vec![],
                predictions_digest: digest(request.dsl.as_bytes()),
                wall_ms: 0,
                failure: None,
            }
        }
    }
    fn fitted_model(&self, _: &Request) -> Result<Vec<u8>> {
        Ok(b"synthetic-model".to_vec())
    }
}
fn fixture(all: bool) -> (tempfile::TempDir, Arc<Store>, Plan, Arc<CaptureContext>) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/research-run");
    fs::create_dir_all(&root).unwrap();
    let directory = tempfile::tempdir_in(fs::canonicalize(root).unwrap()).unwrap();
    let table =
        jope_runtime::table::parse_csv(b"x,target\n0,0\n1,1\n2,2\n3,3\n", &Default::default())
            .unwrap();
    let bytes = jmat::encode(&table).unwrap();
    let path = directory.path().join("control.jmat");
    fs::write(&path, &bytes).unwrap();
    let plan = Plan {
        run_id: jope_api::ids::RunId::new(),
        dataset_id: "ds_21fbfe8b53dbcd605d79486b".parse().unwrap(),
        problem_digest: digest(&bytes),
        bundle_digest: "b".repeat(64),
        binary_version: "control".into(),
        matrix_path: path,
        matrix_sha256: digest(&bytes),
        target: "target".into(),
        features: vec!["x".into()],
        objective: "rmse".into(),
        seed: 73,
        planned_workers: 8,
        effort: jope_api::Effort::Fast,
        evaluations: 2,
        wall_seconds: 60,
        evaluation_wall_ms: 1000,
        evaluation_rss_bytes: 1024 * 1024,
        groups: None,
        times: None,
        gap_rows: 0,
    };
    let binding = Binding {
        config: CaptureConfig {
            max_attempts: 2,
            max_rounds: 1,
            max_bytes: 1024 * 1024,
            evaluate_all_proposals: all,
            seed_protocol_sha256: digest(b"control-seeds"),
        },
        identity: RecordIdentity {
            run_id: plan.run_id.to_string(),
            task: RecordTask::Regression,
            evidence: RecordEvidence::SyntheticControl,
            partition: "train".into(),
            source_commit: "a".repeat(40),
            corpus_manifest_sha256: "c".repeat(64),
            dataset_sha256: plan.matrix_sha256.clone(),
            family_sha256: "d".repeat(64),
            problem_sha256: plan.problem_digest.clone(),
            bundle_sha256: plan.bundle_digest.clone(),
            conditioning_sha256: "e".repeat(64),
            evaluator_sha256: "f".repeat(64),
            dope_weights_sha256: DOPE_SHA256.into(),
        },
        seed: plan.seed,
        tokens: vec![vec![0.25; 32]; 10],
        mask: vec![true; 10],
        matrix_sha256: plan.matrix_sha256.clone(),
    };
    let context = Arc::new(CaptureContext::control(binding, Arc::new(Control)).unwrap());
    let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
    (directory, store, plan, context)
}
fn backend(
    executor: &ControlExecutor,
    workers: usize,
) -> RecordingBackend<'_, ControlExecutor, Control, HeuristicAdmission> {
    RecordingBackend {
        executor,
        space: &Control,
        admission: &HeuristicAdmission,
        live_workers: workers,
    }
}
fn reader(records: &Path, inventory: &Path, all: bool) -> std::process::Output {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../python/src/jope/north_star/record_reader.py");
    let mut command = Command::new("python3");
    command
        .arg(script)
        .arg(records)
        .args([
            "--allow-synthetic",
            "--detailed-counts",
            "--expected-inventory",
        ])
        .arg(inventory);
    if all {
        command.arg("--require-evaluate-all");
    }
    command.output().unwrap()
}

#[test]
fn committed_controls_match_python_bytes_identity_mode_and_every_count() {
    for all in [false, true] {
        for workers in [1, 8] {
            let (directory, store, plan, context) = fixture(all);
            let output = directory.path().join("records.jsonl");
            let executor = ControlExecutor(AtomicUsize::new(0));
            let inventory = record_inner(
                Arc::clone(&store),
                plan,
                context,
                backend(&executor, workers),
                &output,
                true,
            )
            .unwrap();
            assert_eq!(
                inventory.identity.evidence,
                RecordEvidence::SyntheticControl
            );
            assert_eq!(
                inventory.counts,
                RecordCounts {
                    rounds: 1,
                    proposals: 2,
                    admissions: 2,
                    dispatches: 2,
                    outcomes: 2,
                    successes: 1,
                    failures: 1,
                    incomplete_rounds: 0
                }
            );
            assert_eq!(
                inventory.records_sha256,
                digest(&fs::read(&output).unwrap())
            );
            let inventory_path = directory.path().join("inventory.json");
            let original = serde_json::to_value(&inventory).unwrap();
            fs::write(&inventory_path, serde_json::to_vec(&original).unwrap()).unwrap();
            let readback = reader(&output, &inventory_path, all);
            assert!(
                readback.status.success(),
                "{}",
                String::from_utf8_lossy(&readback.stderr)
            );
            assert_eq!(
                serde_json::from_slice::<Value>(&readback.stdout).unwrap(),
                serde_json::to_value(&inventory.counts).unwrap()
            );
            for field in ["count", "boolean", "identity", "mode", "hash", "unknown"] {
                let mut changed = original.clone();
                match field {
                    "count" => changed["counts"]["dispatches"] = json!(3),
                    "boolean" => changed["counts"]["failures"] = json!(true),
                    "identity" => changed["identity"]["source_commit"] = json!("0".repeat(40)),
                    "mode" => changed["evaluate_all_proposals"] = json!(!all),
                    "hash" => changed["records_sha256"] = json!("0".repeat(64)),
                    _ => changed["unreviewed"] = json!(true),
                }
                fs::write(&inventory_path, serde_json::to_vec(&changed).unwrap()).unwrap();
                let refusal = reader(&output, &inventory_path, all);
                assert!(!refusal.status.success(), "mutation {field} was accepted");
                assert!(
                    refusal.stdout.is_empty(),
                    "a refused inventory emitted a success summary"
                );
            }
            let mut oversized = vec![b; MAX_LINE_BYTES + 1];
            oversized.extend(serde_json::to_vec(&original).unwrap());
            fs::write(&inventory_path, oversized).unwrap();
            let refusal = reader(&output, &inventory_path, all);
            assert!(!refusal.status.success());
            assert!(refusal.stdout.is_empty());
            assert!(
                String::from_utf8_lossy(&refusal.stderr).contains("oversized expected inventory")
            );
        }
    }
}

#[test]
fn output_and_worker_refusals_precede_reservations_or_worker_calls() {
    for existing in [false, true] {
        for workers in [0, 9] {
            let (directory, store, plan, context) = fixture(false);
            let output = directory.path().join("records.jsonl");
            if existing {
                fs::write(&output, b"retained evidence").unwrap();
            }
            let executor = ControlExecutor(AtomicUsize::new(0));
            assert!(record_inner(
                Arc::clone(&store),
                plan.clone(),
                context,
                backend(&executor, workers),
                &output,
                true
            )
            .is_err());
            assert_eq!(executor.0.load(Ordering::SeqCst), 0);
            assert!(store.get(&RecordKey::Run(plan.run_id)).unwrap().is_none());
            if existing {
                assert_eq!(fs::read(output).unwrap(), b"retained evidence");
            } else {
                assert!(!output.exists());
            }
        }
    }
}

#[test]
fn public_runner_refuses_synthetic_context_before_work() {
    let (directory, store, plan, context) = fixture(false);
    let output = directory.path().join("records.jsonl");
    let executor = ControlExecutor(AtomicUsize::new(0));
    assert!(record_run(
        Arc::clone(&store),
        plan.clone(),
        context,
        backend(&executor, 1),
        &output
    )
    .is_err());
    assert_eq!(executor.0.load(Ordering::SeqCst), 0);
    assert!(store.get(&RecordKey::Run(plan.run_id)).unwrap().is_none());
    assert!(!output.exists());
}
