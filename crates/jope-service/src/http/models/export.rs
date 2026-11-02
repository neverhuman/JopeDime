use super::*;

pub(super) async fn command<T: DeserializeOwned + Serialize>(
    request: Request<Body>,
    operation: u8,
) -> Result<(T, RequestIdentity)> {
    let headers = request.headers().clone();
    let mut keys = headers.get_all("idempotency-key").iter();
    let Some(key) = keys.next().and_then(|v| v.to_str().ok()) else {
        return Err(bad());
    };
    if keys.next().is_some() {
        return Err(bad());
    }
    let target = request
        .uri()
        .path_and_query()
        .map(|p| p.as_str().to_owned())
        .ok_or_else(bad)?;
    let bytes = to_bytes(request.into_body(), 1024 * 1024)
        .await
        .map_err(|_| bad())?;
    let body: T = jope_runtime::strict_json::decode(&bytes).map_err(|_| bad())?;
    let value = serde_json::to_value(&body).map_err(|_| bad())?;
    let identity = RequestIdentity::new(
        "local-engine-owner",
        key,
        operation,
        &target,
        headers.get("if-match").and_then(|v| v.to_str().ok()),
        &value,
    )
    .map_err(|_| bad())?;
    Ok((body, identity))
}
pub(super) fn export(
    state: &DatasetState,
    run: RunId,
    request: ExportRequest,
    identity: RequestIdentity,
) -> Response {
    match state.store.idempotent_receipt(&identity) {
        Ok(Some(r)) => return r.into_response(),
        Err(e) => return e.into_response(),
        Ok(None) => {}
    }
    let result = (|| -> Result<(ArtifactId, Record, ExportManifest)> {
        let (model, original) = export_champion(state, &run, &request.algorithm_id, request.refit)?;
        let mut entries = BTreeMap::new();
        let mut languages = vec![];
        if request.kind != ExportKind::Source {
            entries.insert(
                "model/manifest.json".into(),
                model.manifest_bytes().to_vec(),
            );
            for (name, bytes) in model.parts() {
                entries.insert(format!("model/{name}"), bytes.clone());
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        if request.kind != ExportKind::Model {
            for language in &request.languages {
                let name = language_name(*language);
                if !seen.insert(name) {
                    return Err(bad());
                }
                match model.export_source(name) {
                    Ok(bytes) => {
                        let extension = match language {
                            Language::Rust => "rs",
                            Language::Python => "py",
                            _ => "mjs",
                        };
                        let path = format!("src/{name}/predictor.{extension}");
                        entries.insert(path.clone(), bytes);
                        languages.push(LanguageExport::Exported {
                            language: *language,
                            path,
                        });
                    }
                    Err(_) if request.require_all_languages => return Err(unavailable()),
                    Err(_) => languages.push(LanguageExport::Unsupported {
                        language: *language,
                        operations: vec!["requested_language".into()],
                    }),
                }
            }
        }
        if request.kind == ExportKind::Source && entries.is_empty() {
            return Err(unavailable());
        }
        entries.insert(
            "export.json".into(),
            serde_json::to_vec(&request).map_err(|_| integrity())?,
        );
        if entries.values().map(|v| v.len() as u64).sum::<u64>() > ARCHIVE_BYTES {
            return Err(bad());
        }
        let measured =
            bundle::verified_model_parity(&state.store, &original, &model, &request.languages)
                .map_err(|_| integrity())?;
        if let Some((_, _, bytes)) = &measured {
            entries.insert("parity_receipt.json".into(), bytes.clone());
        }
        let pipeline = model.math_pipeline().map_err(|_| unavailable())?;
        let retains = bundle::retains_training_data(&pipeline.state.model);
        for language in &languages {
            let LanguageExport::Exported { language, .. } = language else {
                continue;
            };
            let name = language_name(*language);
            let usage = match language {
                Language::Python => "Run `python3 predictor.py --self-check` to verify the embedded checks. Import `predict_rows` for prediction batches.",
                Language::Rust => "Run `rustc --edition=2021 --test predictor.rs -o predictor-self-check` and `./predictor-self-check` to verify the embedded checks. Include the source as a module and call `predict_cells` for prediction batches.",
                _ => "Import the fitted predictor module using the prediction input contract for the bundled model.",
            };
            let privacy = if retains {
                "\nPrivacy notice: this fitted model retains training rows in its learned state. Treat the bundle as containing training data.\n"
            } else {
                ""
            };
            entries.insert(
                format!("src/{name}/README.md"),
                format!("# {name} fitted predictor\n\n{usage}\n\nPass raw feature cells in the bundled input-schema order; do not include the target column. The predictor applies its fitted preprocessing. Model identity: {}. Refit mode: {}.\n{privacy}", model.id(), serde_json::to_string(&request.refit).map_err(|_| integrity())?).into_bytes(),
            );
        }
        let archive_manifest = BundleManifest {
            schema_version: bundle::BUNDLE_SCHEMA_VERSION,
            model_object_digest: hash(model.manifest_bytes()),
            retains_training_data: retains,
            entries: entries.iter().map(|(k, v)| (k.clone(), hash(v))).collect(),
        };
        entries.insert(
            "manifest.json".into(),
            serde_json::to_vec(&archive_manifest).map_err(|_| integrity())?,
        );
        let published = bundle::publish(
            &Bundle {
                entries,
                manifest: archive_manifest,
            },
            &mut StoreBundleWriter {
                store: &state.store,
            },
        )
        .map_err(|_| integrity())?;
        if published.size > ARCHIVE_BYTES {
            return Err(bad());
        }
        let archive = BlobRef {
            sha256: hex_digest(&published.sha256)?,
            bytes: published.size,
        };
        let id: ArtifactId = format!("art_{}", &published.sha256[..24])
            .parse()
            .map_err(|_| integrity())?;
        let manifest = ExportManifest {
            artifact_id: id.clone(),
            algorithm_id: request.algorithm_id,
            kind: request.kind,
            model_digest: model.id().into(),
            emitter_version: env!("CARGO_PKG_VERSION").into(),
            qualified_set_id: measured
                .as_ref()
                .map_or_else(|| "not-qualified".into(), |(_, id, _)| id.clone()),
            refit: request.refit,
            retains_training_data: retains,
            languages,
            parity_status: if measured.is_some() {
                ParityStatus::Passed
            } else {
                ParityStatus::Unavailable
            },
            parity_receipt: measured.map(|(id, _, _)| id),
        };
        let mut refs = original.blobs;
        refs.push(archive.clone());
        let record = Record::metadata(
            json!({"kind":"model_export","export_kind":request.kind,"model_digest":model.id(),"archive":archive}),
            refs,
        );
        Ok((id, record, manifest))
    })();
    let (id, record, manifest) = match result {
        Ok(v) => v,
        Err(e) => return response(&e),
    };
    match state.store.execute_idempotent(&identity, move |tx| {
        let key = RecordKey::Artifact(id);
        if let Some(old) = tx.get(&key)? {
            if old != record {
                return Err(crate::store::StoreError::Integrity(
                    "artifact identity collision",
                ));
            }
        } else {
            tx.put(&key, &record)?
        }
        CommandOutcome::json(200, serde_json::to_value(manifest)?)
    }) {
        Ok(v) => v.into_response(),
        Err(e) => e.into_response(),
    }
}
pub(super) fn hex_digest(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64 {
        return Err(integrity());
    }
    let mut bytes = [0; 32];
    for (i, slot) in bytes.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&value[2 * i..2 * i + 2], 16).map_err(|_| integrity())?
    }
    Ok(bytes)
}
pub(super) async fn export_model(
    Extension(state): Extension<DatasetState>,
    path: std::result::Result<Path<String>, axum::extract::rejection::PathRejection>,
    request: Request<Body>,
) -> Response {
    let Ok(Path(raw)) = path else {
        return response(&bad());
    };
    let Ok(run) = raw.parse::<RunId>() else {
        return response(&bad());
    };
    let (body, identity) = match command::<ExportRequest>(request, 13).await {
        Ok(v) => v,
        Err(e) => return response(&e),
    };
    match tokio::task::spawn_blocking(move || export(&state, run, body, identity)).await {
        Ok(v) => v,
        Err(_) => response(&integrity()),
    }
}
pub(super) fn csv(receipt: CommandReceipt) -> Response {
    match receipt.outcome.body().as_str() {
        Some(v) => ([(header::CONTENT_TYPE, "text/csv")], v.to_owned()).into_response(),
        None => response(&integrity()),
    }
}
pub(super) fn predict(
    state: &DatasetState,
    request: PredictRequest,
    identity: RequestIdentity,
) -> Response {
    match state.store.idempotent_receipt(&identity) {
        Ok(Some(r)) => return csv(r),
        Err(e) => return e.into_response(),
        Ok(None) => {}
    }
    let result = (|| -> Result<String> {
        let (model, _) = object(state, &request.model_artifact)?;
        let record = state
            .store
            .get(&RecordKey::Dataset(request.dataset.clone()))
            .map_err(|_| integrity())?
            .ok_or_else(missing)?;
        let dataset: RegisteredDataset =
            serde_json::from_value(record.data).map_err(|_| integrity())?;
        if dataset.id != request.dataset
            || !record.blobs.contains(&dataset.csv)
            || dataset.csv.bytes > MODEL_BYTES.min(state.config.limits.max_dataset_bytes.value)
        {
            return Err(bad());
        }
        let bytes = state
            .store
            .read_blob(&dataset.csv)
            .map_err(|_| integrity())?;
        let table =
            jope_runtime::table::parse_csv(&bytes, &Default::default()).map_err(|_| bad())?;
        let (values, _) = jope_runtime::budget::Budget::new(None, 1)
            .install(|| model.math_pipeline()?.predict_table(&table))
            .map_err(|_| unavailable())?
            .map_err(|_| bad())?;
        if values.len() != table.rows.len() || values.iter().any(|v| !v.is_finite()) {
            return Err(error(
                ErrorCode::NumericalFailure,
                "Frozen model produced invalid predictions.",
            ));
        }
        let mut out = String::from("prediction\n");
        for value in values {
            out.push_str(&value.to_string());
            out.push('\n');
        }
        if out.len() as u64 > MODEL_BYTES {
            return Err(bad());
        }
        Ok(out)
    })();
    let output = match result {
        Ok(v) => v,
        Err(e) => return response(&e),
    };
    match state.store.execute_idempotent(&identity, move |_| {
        CommandOutcome::json(200, Value::String(output))
    }) {
        Ok(v) => csv(v),
        Err(e) => e.into_response(),
    }
}
pub(super) async fn predict_model(
    Extension(state): Extension<DatasetState>,
    request: Request<Body>,
) -> Response {
    let (body, identity) = match command::<PredictRequest>(request, 15).await {
        Ok(v) => v,
        Err(e) => return response(&e),
    };
    match tokio::task::spawn_blocking(move || predict(&state, body, identity)).await {
        Ok(v) => v,
        Err(_) => response(&integrity()),
    }
}
pub(super) async fn download(
    Extension(state): Extension<DatasetState>,
    path: std::result::Result<Path<String>, axum::extract::rejection::PathRejection>,
) -> Response {
    let Ok(Path(raw)) = path else {
        return response(&bad());
    };
    let Ok(id) = raw.parse::<ArtifactId>() else {
        return response(&bad());
    };
    let result = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
        let record = state
            .store
            .get(&RecordKey::Artifact(id))
            .map_err(|_| integrity())?
            .ok_or_else(missing)?;
        if !matches!(
            record.data["kind"].as_str(),
            Some("model_export" | "model_parity")
        ) {
            return Err(missing());
        }
        let blob: BlobRef =
            serde_json::from_value(record.data["archive"].clone()).map_err(|_| integrity())?;
        if blob.bytes > ARCHIVE_BYTES || !record.blobs.contains(&blob) {
            return Err(integrity());
        }
        state.store.read_blob(&blob).map_err(|_| integrity())
    })
    .await;
    match result {
        Ok(Ok(bytes)) => {
            let digest = hash(&bytes);
            (
                [
                    (header::CONTENT_TYPE.as_str(), "application/octet-stream"),
                    ("x-jope-artifact-sha256", digest.as_str()),
                ],
                bytes,
            )
                .into_response()
        }
        Ok(Err(e)) => response(&e),
        Err(_) => response(&integrity()),
    }
}
