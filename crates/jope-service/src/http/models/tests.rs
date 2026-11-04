use super::*;

use crate::{config::Config, dataset::register, store::Store};
use axum::body::to_bytes;
use jope_api::decimal::DecimalU64;
use jope_runtime::{
    math_pipeline::{INPUT_PART, PIPELINE_PART},
    model_object::{Part, Producer, MATH_PIPELINE, SCHEMA},
};
fn open_state(path: &std::path::Path) -> DatasetState {
    DatasetState {
        store: Arc::new(Store::open(path).unwrap()),
        config: Config::from_sources(None, &BTreeMap::new(), &BTreeMap::new()).unwrap(),
        home: path.into(),
    }
}
fn fixture() -> (
    tempfile::TempDir,
    DatasetState,
    RunId,
    ArtifactId,
    PredictRequest,
) {
    let root =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/mcp05-model-tests");
    std::fs::create_dir_all(&root).unwrap();
    let dir = tempfile::tempdir_in(root).unwrap();
    let state = open_state(dir.path());
    let run = RunId::new();
    let mut input: Value = serde_json::from_slice(include_bytes!(
        "../../../../jope-runtime/tests/fixtures/math-pipeline/input.json"
    ))
    .unwrap();
    input["prediction_version"] = json!(2);
    let parts: BTreeMap<String, Vec<u8>> = BTreeMap::from([
        (INPUT_PART.into(), serde_json::to_vec(&input).unwrap()),
        (
            PIPELINE_PART.into(),
            include_bytes!("../../../../jope-runtime/tests/fixtures/math-pipeline/pipeline.json")
                .to_vec(),
        ),
    ]);
    let model = ModelObject::new(
        Manifest {
            schema: SCHEMA.into(),
            minimum_reader_version: 1,
            format: MATH_PIPELINE.into(),
            producer: Producer::default(),
            primary: PIPELINE_PART.into(),
            required_capabilities: vec![],
            files: parts
                .iter()
                .map(|(p, b)| Part::new(p, "application/json", b))
                .collect(),
            metadata: json!({"run_id":run}),
        },
        parts,
        MODEL_BYTES,
    )
    .unwrap();
    let id: ArtifactId = format!("art_{}", &model.id()[7..31]).parse().unwrap();
    let refs = std::iter::once(model.manifest_bytes())
        .chain(model.parts().values().map(Vec::as_slice))
        .map(|b| state.store.publish_blob(b).unwrap())
        .collect();
    let snapshot = RunSnapshot {
        run_id: run.clone(),
        revision: DecimalU64(1),
        state: RunState::Completed,
        phase: RunPhase::Finalizing,
        pending: None,
        effort: Effort::Fast,
        budget_used: BudgetUsed {
            evaluations: DecimalU64(1),
            wall_seconds: "0.1".into(),
        },
        budget_total: BudgetTotal {
            evaluations: DecimalU64(1),
            wall_seconds: DecimalU64(5),
        },
        best: Some(BestAlgorithm {
            algorithm_id: "alg_0001".parse().unwrap(),
            metric: "rmse".into(),
            cv_value: 0.25,
            model_artifact: Some(id.clone()),
        }),
        event_watermark: DecimalU64(0),
        warnings: vec![],
        completion_reason: Some(CompletionReason::BudgetEvaluations),
        evaluation_receipt: None,
    };
    state
        .store
        .transaction(|tx| {
            tx.put(
                &RecordKey::Artifact(id.clone()),
                &Record::metadata(
                    json!({"kind":"finalized_model","model_digest":model.id()}),
                    refs,
                ),
            )?;
            tx.put(
                &RecordKey::Run(run.clone()),
                &Record::run(
                    json!({"snapshot":snapshot,"checkpoint":{}}),
                    vec![],
                    RunState::Completed,
                    1,
                ),
            )
        })
        .unwrap();
    let input = model.math_pipeline().unwrap().input;
    let mut csv = input
        .columns
        .iter()
        .map(|c| c.name.clone())
        .chain(std::iter::once(input.target.name.clone()))
        .collect::<Vec<_>>()
        .join(",");
    csv.push('\n');
    for i in 0..4 {
        csv.push_str(
            &input
                .columns
                .iter()
                .map(|c| (c.min + i as f64).to_string())
                .chain(std::iter::once((i + 1).to_string()))
                .collect::<Vec<_>>()
                .join(","),
        );
        csv.push('\n');
    }
    let ds = register::register(&state.store, csv.as_bytes(), &Default::default()).unwrap();
    let request = PredictRequest {
        model_artifact: id.clone(),
        dataset: ds.id,
    };
    (dir, state, run, id, request)
}
fn identity<T: Serialize>(key: &str, operation: u8, target: &str, body: &T) -> RequestIdentity {
    RequestIdentity::new(
        "local-engine-owner",
        key,
        operation,
        target,
        None,
        &serde_json::to_value(body).unwrap(),
    )
    .unwrap()
}
fn export_request() -> ExportRequest {
    ExportRequest {
        algorithm_id: "alg_0001".parse().unwrap(),
        kind: ExportKind::Bundle,
        languages: vec![Language::Rust],
        require_all_languages: true,
        refit: Refit::Fit,
    }
}
async fn bytes(r: Response) -> Vec<u8> {
    to_bytes(r.into_body(), ARCHIVE_BYTES as usize)
        .await
        .unwrap()
        .to_vec()
}

mod export;
mod refit;
