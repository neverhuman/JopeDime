use super::*;

#[tokio::test]
async fn frozen_export_download_and_run_based_predict_use_one_model() {
    let (_dir, state, run, _, request) = fixture();
    let a = algorithm(
        &state,
        run.clone(),
        Some("alg_0001".parse().unwrap()),
        AlgorithmQuery::default(),
        false,
    )
    .unwrap();
    assert_eq!(a["algorithm_id"], "alg_0001");
    assert_eq!(a["model_artifact"], json!(request.model_artifact));
    let body = export_request();
    let target = format!("/v1/runs/{run}/exports");
    let r = export(
        &state,
        run.clone(),
        body.clone(),
        identity("export", 13, &target, &body),
    );
    assert_eq!(r.status(), 200);
    let manifest: ExportManifest = serde_json::from_slice(&bytes(r).await).unwrap();
    assert_eq!(manifest.parity_status, ParityStatus::Unavailable);
    assert!(manifest.parity_receipt.is_none());
    let downloaded = download(
        Extension(state.clone()),
        Ok(Path(manifest.artifact_id.to_string())),
    )
    .await;
    assert_eq!(downloaded.status(), 200);
    let digest = downloaded.headers()["x-jope-artifact-sha256"]
        .to_str()
        .unwrap()
        .to_owned();
    let archive = bytes(downloaded).await;
    assert_eq!(digest, hash(&archive));
    assert_eq!(
        manifest.model_digest,
        object(&state, &request.model_artifact).unwrap().0.id()
    );
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(archive)).unwrap();
    assert!(zip.by_name("model/manifest.json").is_ok());
    assert!(zip.by_name("src/rust/predictor.rs").is_ok());
    let exported = PredictRequest {
        model_artifact: manifest.artifact_id,
        dataset: request.dataset.clone(),
    };
    let prediction = predict(
        &state,
        exported.clone(),
        identity("predict-export", 15, "/v1/predictions", &exported),
    );
    assert_eq!(prediction.status(), 200);
    assert_eq!(prediction.headers()[header::CONTENT_TYPE], "text/csv");
    let output = bytes(prediction).await;
    let table = jope_runtime::table::parse_csv(&output, &Default::default()).unwrap();
    assert_eq!(table.rows.len(), 4);
    assert!(table
        .rows
        .iter()
        .all(|r| r[0].as_ref().unwrap().parse::<f64>().unwrap().is_finite()));
    let original = predict(
        &state,
        request.clone(),
        identity("predict-original", 15, "/v1/predictions", &request),
    );
    assert_eq!(bytes(original).await, output);
}
#[tokio::test]
async fn export_and_prediction_replay_after_store_reopen() {
    let (dir, state, run, _, request) = fixture();
    let body = export_request();
    let target = format!("/v1/runs/{run}/exports");
    let first = bytes(export(
        &state,
        run.clone(),
        body.clone(),
        identity("export", 13, &target, &body),
    ))
    .await;
    let predicted = bytes(predict(
        &state,
        request.clone(),
        identity("predict", 15, "/v1/predictions", &request),
    ))
    .await;
    drop(state);
    let reopened = open_state(dir.path());
    assert_eq!(
        bytes(export(
            &reopened,
            run,
            body.clone(),
            identity("export", 13, &target, &body)
        ))
        .await,
        first
    );
    assert_eq!(
        bytes(predict(
            &reopened,
            request.clone(),
            identity("predict", 15, "/v1/predictions", &request)
        ))
        .await,
        predicted
    );
    let mut changed = request;
    changed.dataset = "ds_00112233445566778899aabb".parse().unwrap();
    assert_eq!(
        predict(
            &reopened,
            changed.clone(),
            identity("predict", 15, "/v1/predictions", &changed)
        )
        .status(),
        409
    );
}
#[tokio::test]
async fn changed_export_body_conflicts_and_unsupported_language_has_no_receipt() {
    let (_dir, state, run, _, _) = fixture();
    let body = export_request();
    let target = format!("/v1/runs/{run}/exports");
    assert_eq!(
        export(
            &state,
            run.clone(),
            body.clone(),
            identity("same", 13, &target, &body)
        )
        .status(),
        200
    );
    let mut unsupported = body;
    unsupported.languages = vec![Language::Typescript];
    assert_eq!(
        export(
            &state,
            run.clone(),
            unsupported.clone(),
            identity("same", 13, &target, &unsupported)
        )
        .status(),
        409
    );
    assert_eq!(
        export(
            &state,
            run,
            unsupported.clone(),
            identity("unsupported", 13, &target, &unsupported)
        )
        .status(),
        422
    );
    assert!(state
        .store
        .get(&RecordKey::Command {
            principal: "local-engine-owner".into(),
            idempotency_key: "unsupported".into()
        })
        .unwrap()
        .is_none());
}
#[test]
fn unknown_algorithm_and_corrupt_model_are_not_false_successes() {
    let (_dir, state, run, id, _) = fixture();
    assert_eq!(
        champion(&state, &run, Some(&"alg_9999".parse().unwrap()))
            .unwrap_err()
            .code,
        ErrorCode::NotFound
    );
    let record = state
        .store
        .get(&RecordKey::Artifact(id.clone()))
        .unwrap()
        .unwrap();
    let path = state.store.verified_blob_path(&record.blobs[1]).unwrap();
    std::fs::write(path, b"corrupted fixture").unwrap();
    assert_eq!(
        object(&state, &id).unwrap_err().code,
        ErrorCode::ModelIntegrityFailed
    );
}
#[tokio::test]
async fn mutation_key_and_duplicate_json_refuse_before_effects() {
    for body in [
        r#"{"model_artifact":"art_00112233445566778899aabb","model_artifact":"art_ffeeddccbbaa998877665544","dataset":"ds_00112233445566778899aabb"}"#,
        r#"{}"#,
    ] {
        let req = Request::builder()
            .uri("/v1/predictions")
            .header("idempotency-key", "key")
            .body(Body::from(body))
            .unwrap();
        assert!(command::<PredictRequest>(req, 15).await.is_err());
    }
    let req = Request::builder()
        .uri("/v1/predictions")
        .body(Body::from("{}"))
        .unwrap();
    assert!(command::<PredictRequest>(req, 15).await.is_err());
}
#[tokio::test]
async fn measured_export_parity_rechecks_model_sources_mode_and_all_rows() {
    let (_dir, state, run, id, _) = fixture();
    let (model, original) = object(&state, &id).unwrap();
    let pipeline = model.math_pipeline().unwrap();
    let rows: Vec<Vec<Option<String>>> = (0..64)
        .map(|i| {
            pipeline
                .input
                .columns
                .iter()
                .map(|c| Some((c.min + (c.max - c.min) * (i as f64 / 63.0)).to_string()))
                .collect()
        })
        .collect();
    let borrowed = rows
        .iter()
        .map(|r| r.iter().map(Option::as_deref).collect())
        .collect::<Vec<_>>();
    let engine = pipeline.predict_cells(&borrowed).unwrap();
    // Controlled observations exercise receipt validation only. Actual toolchain
    // executions belong to the external qualification receipt, not this unit test.
    let receipt = bundle::MeasuredModelParity {
        schema: "jope.measured-model-parity/v1".into(),
        model_digest: model.id().into(),
        emitter_version: env!("CARGO_PKG_VERSION").into(),
        refit: Refit::Fit,
        source_sha256: BTreeMap::from([
            (
                "python".into(),
                hash(&model.export_source("python").unwrap()),
            ),
            ("rust".into(), hash(&model.export_source("rust").unwrap())),
        ]),
        toolchains: BTreeMap::from([
            ("python".into(), "controlled validator fixture".into()),
            ("rust".into(), "controlled validator fixture".into()),
        ]),
        rows,
        engine: engine.clone(),
        observed: BTreeMap::from([("python".into(), engine.clone()), ("rust".into(), engine)]),
    };
    for case in 0..5 {
        let mut bad = receipt.clone();
        match case {
            0 => bad.source_sha256.insert("python".into(), "0".repeat(64)),
            1 => {
                bad.observed.get_mut("rust").unwrap()[0] += 1.;
                None
            }
            2 => {
                bad.rows.pop();
                None
            }
            3 => {
                bad.refit = Refit::FitHoldout;
                None
            }
            _ => {
                bad.model_digest = "different-model".into();
                None
            }
        };
        assert!(bundle::record_model_parity(&state.store, &id, &model, &bad).is_err());
        assert_eq!(
            state
                .store
                .get(&RecordKey::Artifact(id.clone()))
                .unwrap()
                .unwrap(),
            original
        );
    }
    let parity_id = bundle::record_model_parity(&state.store, &id, &model, &receipt).unwrap();
    let mut body = export_request();
    body.languages = vec![Language::Python, Language::Rust];
    let target = format!("/v1/runs/{run}/exports");
    let response = export(
        &state,
        run,
        body.clone(),
        identity("measured-export", 13, &target, &body),
    );
    assert_eq!(response.status(), 200);
    let manifest: ExportManifest = serde_json::from_slice(&bytes(response).await).unwrap();
    assert_eq!(manifest.parity_status, ParityStatus::Passed);
    assert_eq!(manifest.parity_receipt, Some(parity_id.clone()));
    assert!(manifest.qualified_set_id.starts_with("measured-model:"));
    let downloaded = download(
        Extension(state.clone()),
        Ok(Path(manifest.artifact_id.to_string())),
    )
    .await;
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes(downloaded).await)).unwrap();
    assert!(archive.by_name("parity_receipt.json").is_ok());
    assert!(archive.by_name("src/python/predictor.py").is_ok());
    let proof = download(Extension(state), Ok(Path(parity_id.to_string()))).await;
    assert_eq!(proof.status(), 200);
}
