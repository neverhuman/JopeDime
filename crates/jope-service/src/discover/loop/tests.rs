use super::*;
use std::{
    io::{BufRead, BufReader, Write},
    process::{Command, Stdio},
    time::Duration,
};

const TRAIN: &[u8] =
    include_bytes!("../../../../../test-support/datasets/v1/d2-regression/train.csv");
const TRAIN_SHA: &str = "d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931";

fn plan(directory: &Path, rows: usize, effort: Effort, evaluations: u64) -> Plan {
    assert_eq!(digest(TRAIN), TRAIN_SHA);
    let mut headers: Vec<_> = (0..21).map(|j| format!("x{j}")).collect();
    headers.push("target".into());
    let mut csv = headers.join(",").into_bytes();
    csv.push(b'\n');
    csv.extend(TRAIN);
    let mut table =
        jope_runtime::table::parse_csv(&csv, &jope_runtime::limits::Limits::default()).unwrap();
    // Only the catalog TRAIN object is opened; the external TEST object is
    // never read. Workers receive only this fixed FIT subset of TRAIN.
    let (fit, _) = super::super::splits::random_split(
        table.rows.len(),
        table.rows.len() * 4 / 5,
        73,
        "dsc08/fit",
    )
    .unwrap();
    let source_rows: Vec<_> = fit.into_iter().take(rows).collect();
    table.rows = source_rows.iter().map(|&r| table.rows[r].clone()).collect();
    let bytes = crate::dataset::jmat::encode(&table).unwrap();
    let path = directory.join("fit.jmat");
    fs::write(&path, &bytes).unwrap();
    fs::write(
        directory.join("fit-source-rows.json"),
        serde_json::to_vec(&source_rows).unwrap(),
    )
    .unwrap();
    Plan {
        run_id: "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
        dataset_id: "ds_21fbfe8b53dbcd605d79486b".parse().unwrap(),
        problem_digest: digest(&bytes),
        bundle_digest: digest(b"DSC08 heuristic M1, no learned bundle"),
        binary_version: "dsc08-test-candidate-v1".into(),
        matrix_path: path,
        matrix_sha256: digest(&bytes),
        target: "target".into(),
        features: headers[..21].to_vec(),
        objective: "rmse".into(),
        seed: 73,
        planned_workers: 8,
        effort,
        evaluations,
        wall_seconds: 60,
        evaluation_wall_ms: 15_000,
        evaluation_rss_bytes: 2 * 1024 * 1024 * 1024,
        groups: None,
        times: None,
        gap_rows: 0,
    }
}

// Scheduler controls use an executor fixture, not ranker/quality evidence.
struct RoundSpace;
impl SearchSpace for RoundSpace {
    fn canonical(&self, dsl: &str) -> Result<String> {
        Ok(dsl.into())
    }
    fn seeds(&self) -> Result<Vec<String>> {
        Ok((0..4).map(|i| format!("seed-{i}")).collect())
    }
    fn propose(
        &self,
        parents: &[String],
        count: usize,
        rng: &mut ChaCha12Rng,
    ) -> Result<Vec<String>> {
        assert_eq!(parents[0], "seed-3", "parents retain user-objective order");
        let wave = rng.next_u64();
        Ok((0..count)
            .map(|i| format!("candidate-{wave}-{i}"))
            .collect())
    }
}
struct OpposingScores;
impl Executor for OpposingScores {
    fn evaluate(&self, request: &Request) -> Response {
        let seed = request
            .dsl
            .strip_prefix("seed-")
            .and_then(|s| s.parse::<u32>().ok());
        let (surrogate, objective) =
            seed.map_or((50.0, 5.0), |i| (f64::from(i), 20.0 - f64::from(i)));
        Response {
            attempt: request.attempt.clone(),
            scores: vec![surrogate, objective, objective],
            fold_scores: vec![],
            predictions_digest: digest(request.dsl.as_bytes()),
            wall_ms: 0,
            failure: None,
        }
    }
    fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
        Ok(request.dsl.as_bytes().to_vec())
    }
}
struct ContextAdmission {
    rounds: Mutex<Vec<u64>>,
    refuse: bool,
}
impl RoundAdmission for ContextAdmission {
    fn admit_round(
        &self,
        context: &RoundContext<'_>,
        proposals: Vec<String>,
        _: bool,
        _: &mut ChaCha12Rng,
    ) -> Result<Vec<String>> {
        self.rounds.lock().unwrap().push(context.round);
        ensure!(!self.refuse, "controlled admission refusal");
        assert_eq!(context.measured.len(), 4);
        assert_eq!(context.measured[0].dsl, "seed-3");
        assert_eq!(context.measured[0].algorithm_id, "alg_00000003");
        assert_eq!(context.measured[0].surrogate_fitness, 3.0);
        assert_eq!(context.plan.seed, 73);
        Ok(proposals.into_iter().skip(4).take(4).collect())
    }
}

mod admission_controls;
mod control;
#[cfg(feature = "research")]
mod durable_capture;
mod medium_real;
mod native_workers;
mod public_start;
mod recovery;
mod service_entry;
mod waves;

use admission_controls::finish;
use native_workers::{native_worker, pool, run_store, same};

#[test]
#[ignore = "supervised SIGKILL child; invoked by real_native_workers_preserve_projection_across_capacity_pause_and_sigkill"]
fn crash_parent() {
    let home = PathBuf::from(std::env::var_os("DSC08_KILL_HOME").unwrap());
    let p: Plan =
        serde_json::from_slice(&fs::read(std::env::var_os("DSC08_PLAN").unwrap()).unwrap())
            .unwrap();
    let binary = PathBuf::from(std::env::var_os("DSC08_WORKER").unwrap());
    let store = Arc::new(Store::open(home).unwrap());
    let mut d = Discovery::open(store, p).unwrap();
    let workers = pool(&binary, 8);
    d.advance(&NativeSpace, &HeuristicAdmission, &workers, 8, usize::MAX)
        .unwrap();
    d.advance(&NativeSpace, &HeuristicAdmission, &workers, 8, 1)
        .unwrap();
    struct Signal<'a>(&'a ProcessExecutor);
    impl Executor for Signal<'_> {
        fn evaluate(&self, request: &Request) -> Response {
            let _response = self.0.evaluate(request);
            println!("DSC08_NATIVE_RESULT_UNCOMMITTED");
            std::io::stdout().flush().unwrap();
            loop {
                std::thread::park_timeout(Duration::from_secs(1));
            }
        }
        fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
            self.0.fitted_model(request)
        }
    }
    d.advance(
        &NativeSpace,
        &HeuristicAdmission,
        &Signal(&workers),
        8,
        usize::MAX,
    )
    .unwrap();
}
