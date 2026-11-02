use super::*;

mod aborts;
mod basic;
mod fail_closed;
mod kill_recovery;

use super::*;
use crate::research::{
    capture::{Binding, CaptureConfig, CaptureProvider, SlateObservation},
    outbox::materialize_controls,
    *,
};

struct ClockScope(Option<Arc<std::sync::atomic::AtomicU64>>);
impl ClockScope {
    fn frozen() -> (Self, Arc<std::sync::atomic::AtomicU64>) {
        let clock = Arc::new(std::sync::atomic::AtomicU64::new(1_000_000));
        let previous = TEST_CLOCK.with(|current| current.replace(Some(Arc::clone(&clock))));
        (Self(previous), clock)
    }
}
impl Drop for ClockScope {
    fn drop(&mut self) {
        TEST_CLOCK.with(|clock| {
            clock.replace(self.0.take());
        });
    }
}

struct Space {
    seeds: usize,
    duplicate: bool,
}
impl SearchSpace for Space {
    fn canonical(&self, dsl: &str) -> Result<String> {
        Ok(dsl.trim().into())
    }
    fn seeds(&self) -> Result<Vec<String>> {
        Ok((0..self.seeds).map(|i| format!("seed-{i}")).collect())
    }
    fn propose(&self, _: &[String], count: usize, rng: &mut ChaCha12Rng) -> Result<Vec<String>> {
        let nonce = rng.next_u64();
        let mut values: Vec<_> = (0..count).map(|i| format!("p-{nonce}-{i}")).collect();
        if self.duplicate {
            values.insert(1, "seed-0".into());
            values.insert(3, format!(" {} ", values[0]));
        }
        Ok(values)
    }
}
const SPACE: Space = Space {
    seeds: 4,
    duplicate: false,
};

struct Provider {
    observations: AtomicUsize,
    bad_gate: bool,
}
impl CaptureProvider for Provider {
    fn validate_binding(&self, _: &Binding, _: &Plan) -> Result<()> {
        ensure!(!self.bad_gate, "control gate unavailable before dispatch");
        Ok(())
    }

    fn observe(
        &self,
        round: u64,
        dsls: &[String],
        _: &[RecordedReveal],
    ) -> Result<SlateObservation> {
        self.observations.fetch_add(1, Ordering::SeqCst);
        Ok(SlateObservation {
            programs: dsls
                .iter()
                .enumerate()
                .map(|(i, dsl)| RecordedProgram {
                    program_sha256: digest(dsl.as_bytes()),
                    operator_family: "synthetic_control".into(),
                    feat412: vec![i as f32; 412],
                    proj16: vec![1.000_000_000_000_01 + i as f64; 16],
                })
                .collect(),
            revealed_indices: if round == 0 { vec![] } else { (0..4).collect() },
        })
    }
    fn measurement(&self, request: &Request, response: &Response) -> Result<Rung0Measurement> {
        ensure!(!self.bad_gate, "control gate unavailable");
        Ok(Rung0Measurement::Success {
            surrogate: response.scores[0],
            objective: response.scores[1],
            gate: request.dsl.ends_with('0'),
        })
    }
    fn label_seed(&self, round: u64, index: usize) -> Result<u64> {
        Ok(9_000_000 + round * 16 + index as u64)
    }
}
fn context(
    p: &Plan,
    attempts: u64,
    all: bool,
    bad_gate: bool,
) -> (Arc<CaptureContext>, Arc<Provider>) {
    let provider = Arc::new(Provider {
        observations: AtomicUsize::new(0),
        bad_gate,
    });
    let binding = Binding {
        config: CaptureConfig {
            max_attempts: attempts,
            max_rounds: 16.min(attempts as usize),
            max_bytes: 8 * 1024 * 1024,
            evaluate_all_proposals: all,
            seed_protocol_sha256: digest(b"synthetic-control-extra-seeds-v1"),
        },
        identity: RecordIdentity {
            run_id: p.run_id.to_string(),
            task: RecordTask::Regression,
            evidence: RecordEvidence::SyntheticControl,
            partition: "train".into(),
            source_commit: "a".repeat(40),
            corpus_manifest_sha256: digest(b"control-corpus"),
            dataset_sha256: p.matrix_sha256.clone(),
            family_sha256: digest(b"control-family"),
            problem_sha256: p.problem_digest.clone(),
            bundle_sha256: p.bundle_digest.clone(),
            conditioning_sha256: digest(b"control-conditioning"),
            evaluator_sha256: digest(b"control-evaluator"),
            dope_weights_sha256: DOPE_SHA256.into(),
        },
        seed: p.seed,
        tokens: vec![vec![0.25; 32]; 10],
        mask: vec![true; 10],
        matrix_sha256: p.matrix_sha256.clone(),
    };
    (
        Arc::new(CaptureContext::control(binding, provider.clone()).unwrap()),
        provider,
    )
}
fn control_plan(home: &Path, budget: u64) -> Plan {
    let csv = format!(
        "x,target\n{}",
        (0..40)
            .map(|i| format!("{i},{}\n", i * 2))
            .collect::<String>()
    );
    let table =
        jope_runtime::table::parse_csv(csv.as_bytes(), &jope_runtime::limits::Limits::default())
            .unwrap();
    let bytes = crate::dataset::jmat::encode(&table).unwrap();
    let matrix_path = home.join("control.jmat");
    fs::write(&matrix_path, &bytes).unwrap();
    Plan {
        run_id: "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
        dataset_id: "ds_21fbfe8b53dbcd605d79486b".parse().unwrap(),
        problem_digest: digest(&bytes),
        bundle_digest: digest(b"synthetic-control-bundle"),
        binary_version: "synthetic-control-v1".into(),
        matrix_path,
        matrix_sha256: digest(&bytes),
        target: "target".into(),
        features: vec!["x".into()],
        objective: "rmse".into(),
        seed: 73,
        planned_workers: 8,
        effort: Effort::Fast,
        evaluations: budget,
        wall_seconds: 60,
        evaluation_wall_ms: 15000,
        evaluation_rss_bytes: 1024 * 1024,
        groups: None,
        times: None,
        gap_rows: 0,
    }
}
struct Evaluator {
    calls: Mutex<Vec<Request>>,
    store: Arc<Store>,
    fail_extra: bool,
    panic_at: Option<usize>,
}
impl Evaluator {
    fn new(store: Arc<Store>) -> Self {
        Self {
            calls: Mutex::new(vec![]),
            store,
            fail_extra: false,
            panic_at: None,
        }
    }
}
impl Executor for Evaluator {
    fn evaluate(&self, request: &Request) -> Response {
        let count = {
            let mut calls = self.calls.lock().unwrap();
            calls.push(request.clone());
            calls.len()
        };
        if let Some(row) = self
            .store
            .get(&RecordKey::Run(request.attempt.run.clone()))
            .unwrap()
        {
            if row.data["checkpoint"]["version"] == 3 {
                let state: CaptureState =
                    serde_json::from_value(row.data["checkpoint"]["capture"].clone()).unwrap();
                state.validate_storage(&self.store).unwrap();
                assert!(
                    state.roots().iter().all(|r| row.blobs.contains(r)),
                    "request ran without durable roots"
                );
                let reservations = state
                    .rounds
                    .values()
                    .map(|r| r.attempts.len())
                    .sum::<usize>();
                assert_eq!(
                    state.charged, reservations as u64,
                    "all reservations must precede execution"
                );
            }
        }
        assert_ne!(self.panic_at, Some(count), "control evaluator abort");
        let extra = (32..65536).contains(&request.attempt.slot);
        if extra && self.fail_extra {
            return Response::failed(request.attempt.clone(), WorkerFailure::Timeout, 7);
        }
        let loss = if extra {
            -10000.0
        } else if request.dsl.starts_with("seed") {
            100.0
        } else {
            10.0
        };
        Response {
            attempt: request.attempt.clone(),
            scores: vec![loss, loss, 777.0],
            fold_scores: vec![],
            predictions_digest: digest(request.dsl.as_bytes()),
            wall_ms: 7,
            failure: None,
        }
    }
    fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
        Ok(request.dsl.as_bytes().to_vec())
    }
}
fn records(d: &Discovery) -> Vec<ResearchRound> {
    d.checkpoint
        .capture
        .as_ref()
        .unwrap()
        .rounds
        .values()
        .filter_map(|r| r.committed.as_ref())
        .map(|r| CaptureState::read(&d.store, r).unwrap())
        .collect()
}
fn stored(d: &Discovery) -> Value {
    d.store
        .get(&RecordKey::Run(d.checkpoint.plan.run_id.clone()))
        .unwrap()
        .unwrap()
        .data["checkpoint"]
        .clone()
}
fn run_to_end(d: &mut Discovery, e: &Evaluator, workers: usize) {
    for _ in 0..16 {
        if d.completed().is_some() {
            return;
        }
        d.advance(&SPACE, &HeuristicAdmission, e, workers, usize::MAX)
            .unwrap();
    }
    panic!("control exceeded round bound");
}
#[test]
#[ignore = "own child controlled by sigkill_at_capture_boundaries_preserves_committed_prefix"]
fn capture_kill_child() {
    let home = PathBuf::from(std::env::var_os("QLT03_KILL_HOME").unwrap());
    let p = control_plan(&home, 40);
    let store = Arc::new(Store::open(home.join("store")).unwrap());
    let (c, _) = context(&p, 20, true, false);
    let mut d = Discovery::open_research(store.clone(), p.clone(), c).unwrap();
    let e = Evaluator::new(store.clone());
    run_to_end(&mut d, &e, 1);
    materialize_controls(&store, &p.run_id, &home.join("published.jsonl")).unwrap();
    panic!("requested process boundary was never reached");
}
