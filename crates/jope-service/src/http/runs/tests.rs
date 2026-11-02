use super::*;
use rand_chacha::rand_core::SeedableRng;

use crate::{
    dataset::register,
    engine::InProcessEngine,
    http::auth::{BearerToken, LocalAuth},
};
use axum::{
    body::{to_bytes, Body},
    http::{header, Request, StatusCode},
};
use tower::ServiceExt;

fn fixture() -> (tempfile::TempDir, InProcessEngine, RunSpec) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/int01");
    std::fs::create_dir_all(&root).unwrap();
    let directory = tempfile::tempdir_in(root).unwrap();
    let engine = InProcessEngine::open(directory.path()).unwrap();
    let raw = include_bytes!("../../../../test-support/datasets/v1/d2-regression/train.csv");
    assert_eq!(
        hash(raw),
        "d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931"
    );
    let mut names: Vec<_> = (0..21).map(|i| format!("x{i}")).collect();
    names.push("target".into());
    let mut csv = names.join(",");
    csv.push('\n');
    for line in std::str::from_utf8(raw).unwrap().lines().take(96) {
        csv.push_str(line);
        csv.push('\n');
    }
    let dataset = register::register(engine.store(), csv.as_bytes(), &Default::default()).unwrap();
    let spec=serde_json::from_value(json!({"dataset":dataset.id,"target":"target","task":"regression",
        "ignore_columns":[],"test_dataset":null,"split":{"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null},
        "objective":"rmse","effort":"fast","budget":{"evaluations":"4","wall_seconds":"30"},"landmarks":"none","guidance":"heuristic",
        "languages":["rust"],"require_all_languages":true,"device":"cpu","seed":"73","dry_run":true,"plan_digest":null})).unwrap();
    (directory, engine, spec)
}

fn continuation_backend_fixture() -> (
    tempfile::TempDir,
    InProcessEngine,
    ContinuationBackend,
    SearchPlan,
    crate::store::BlobRef,
) {
    let (directory, engine, spec) = fixture();
    let dataset: RegisteredDataset = serde_json::from_value(
        engine
            .store()
            .get(&RecordKey::Dataset(spec.dataset.clone()))
            .unwrap()
            .unwrap()
            .data,
    )
    .unwrap();
    let fit = dataset.matrix;
    let backend = BackendPlan::resolve(&spec, None, 1).unwrap();
    let plan = SearchPlan {
        run_id: "run_01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap(),
        dataset_id: spec.dataset.clone(),
        problem_digest: "a".repeat(64),
        bundle_digest: backend.bundle_digest,
        binary_version: "continuation-guard-test".into(),
        matrix_path: engine.store().verified_blob_path(&fit).unwrap(),
        matrix_sha256: hash(&engine.store().read_blob(&fit).unwrap()),
        target: spec.target.clone(),
        features: (0..21).map(|i| format!("x{i}")).collect(),
        objective: "rmse".into(),
        seed: spec.seed.0,
        planned_workers: 1,
        effort: spec.effort,
        evaluations: 4,
        wall_seconds: 30,
        evaluation_wall_ms: 1000,
        evaluation_rss_bytes: WORKER_RAM,
        groups: None,
        times: None,
        gap_rows: 0,
    };
    let context = ContinuationBackend {
        spec,
        placement: backend.placement,
        landmarks: None,
    };
    (directory, engine, context, plan, fit)
}

fn app(engine: &InProcessEngine) -> axum::Router {
    engine
        .router(LocalAuth::new(BearerToken::from_hex(&"ab".repeat(32)).unwrap(), 31234).unwrap())
        .unwrap()
}
fn request(spec: &RunSpec, key: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/runs")
        .header(header::HOST, "127.0.0.1:31234")
        .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
        .header(header::CONTENT_TYPE, "application/json")
        .header("idempotency-key", key)
        .body(Body::from(serde_json::to_vec(spec).unwrap()))
        .unwrap()
}
async fn body(response: Response) -> Value {
    serde_json::from_slice(&to_bytes(response.into_body(), 1 << 20).await.unwrap()).unwrap()
}

struct InlineQ1Native {
    calls: Arc<std::sync::atomic::AtomicUsize>,
}
impl crate::discover::r#loop::Executor for InlineQ1Native {
    fn evaluate(&self, request: &crate::worker::Request) -> crate::worker::Response {
        use crate::worker::Response;
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let matrix = jmat::Matrix::open(std::fs::read(&request.matrix_path).unwrap()).unwrap();
        match crate::worker::evaluate::native(request, &matrix) {
            Ok(scores) => Response {
                attempt: request.attempt.clone(),
                scores: scores.values,
                fold_scores: scores.fold_scores,
                predictions_digest: scores.predictions_digest,
                wall_ms: 0,
                failure: None,
            },
            Err(error) => Response::failed(request.attempt.clone(), error, 0),
        }
    }
    fn fitted_model(&self, request: &crate::worker::Request) -> anyhow::Result<Vec<u8>> {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/q1-native");
        std::fs::create_dir_all(&root)?;
        let directory = tempfile::tempdir_in(root)?;
        let output = directory.path().join("model.json");
        let mut request = request.clone();
        request.fold_spec["refit_output"] = json!(output);
        request.fold_spec["refit_max_bytes"] = json!(16 * 1024 * 1024);
        let response = self.evaluate(&request);
        anyhow::ensure!(response.failure.is_none(), "native fixture refit failed");
        let value: Value = serde_json::from_slice(&std::fs::read(output)?)?;
        let model = value["model_json"].as_str().unwrap();
        anyhow::ensure!(
            value["attempt"] == serde_json::to_value(&request.attempt)?
                && value["model_digest"] == hash(model.as_bytes())
                && value["predictions_digest"] == response.predictions_digest,
            "refit binding differs"
        );
        Ok(model.as_bytes().to_vec())
    }
}

mod continuation;
mod q1;
mod run;
