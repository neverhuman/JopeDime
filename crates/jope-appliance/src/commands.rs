use super::*;

pub(crate) fn summary_line(command: &str, receipt: &Receipt) {
    let method = receipt
        .method
        .as_ref()
        .map(|m| format!(" method={}", m.selected))
        .unwrap_or_default();
    let task = receipt
        .task
        .as_ref()
        .map(|t| match command {
            "predict" => format!(" task={}", t.task),
            _ => format!(" task={} rows={}", t.task, t.rows),
        })
        .unwrap_or_default();
    let objective = receipt
        .selection
        .as_ref()
        .filter(|s| s.fitness.is_some())
        .map(|s| {
            let calibrated = s
                .calibration
                .as_ref()
                .and_then(|c| c.output_transform)
                .map_or("", |t| {
                    if t.intercept == 0.0 && t.slope == 1.0 {
                        " floored"
                    } else {
                        " calibrated"
                    }
                });
            format!(" objective={}{calibrated}", s.metric)
        })
        .unwrap_or_default();
    let outputs: Vec<&str> = receipt.outputs.iter().map(|o| o.path.as_str()).collect();
    eprintln!(
        "jopedime {command}:{task}{method}{objective} device={} wrote {} in {:.2}s",
        receipt.device.selected,
        outputs.join(", "),
        receipt.timings.total_seconds
    );
}

pub(crate) fn run_fit(
    train: &TrainArgs,
    output: Option<&Path>,
    model_root: &Path,
    common: &CommonArgs,
) -> ExitCode {
    let mut run = Run::new("fit", common);
    let result = (|| -> anyhow::Result<()> {
        let landmark_inputs = train.landmarks.protected_inputs();
        let mut protected = vec![train.train.as_path()];
        protected.extend(landmark_inputs.iter().map(std::path::PathBuf::as_path));
        if let Some(output) = output {
            run.refuse_aliasing(output, &protected)?;
        } else {
            jope_runtime::output_safety::refuse_directory_aliasing(model_root, &protected)?;
            protected.push(model_root);
            if let Some(receipt) = &common.receipt {
                io::refuse_aliasing(receipt, &protected)?;
            }
            run.receipt_allowed = true;
        }
        landmarks::refuse_aliasing(train, common, &[output.unwrap_or(model_root)])?;
        let b = run.prepare(train.landmarks.enabled())?;
        let fitted = run.fit(train, &b)?;
        let started = Instant::now();
        if let Some(output) = output {
            let bytes = fitted.to_bytes()?;
            run.receipt
                .outputs
                .push(io::write_atomic("model", output, &bytes)?);
        } else {
            let object = jope_runtime::model_object::ModelObject::from_fitted(&fitted)?;
            let path = object.save(model_root, Limits::default().max_model_bytes)?;
            run.receipt.outputs.push(FileRecord {
                role: "model",
                path: path.display().to_string(),
                bytes: object.manifest_bytes().len() as u64,
                sha256: jope_runtime::sha256_hex(object.manifest_bytes()),
            });
        }
        run.receipt.timings.write_seconds += started.elapsed().as_secs_f64();
        Ok(())
    })();
    run.finish(result)
}

pub(crate) fn run_predict(
    model: &str,
    input: &Path,
    output: &Path,
    common: &CommonArgs,
) -> ExitCode {
    #[cfg(feature = "mcp")]
    if let Some((run_id, algorithm_id)) = model
        .split_once(':')
        .or_else(|| model.strip_prefix("run_").map(|_| (model, "")))
    {
        if run_id.parse::<jope_api::ids::RunId>().is_ok() {
            return cmd::dataset::report(run_predict_remote(
                run_id,
                (!algorithm_id.is_empty()).then_some(algorithm_id),
                input,
                output,
            ));
        }
    }
    let model = Path::new(model);
    let mut run = Run::new("predict", common);
    let result = (|| -> anyhow::Result<()> {
        run.refuse_aliasing(output, &[model, input])?;
        let b = run.prepare(false)?;
        let started = Instant::now();
        let limit = Limits::default().max_model_bytes;
        // Only the manifest's `format` and capabilities are read here; `ModelObject::read` and
        // `load_fitted` each verify every part, so the full read happens once, on the path that
        // uses it. A Platt-calibrated object predicts whole: `load_fitted` refuses it.
        let route = if model.is_dir() {
            object_route(model)?
        } else {
            ObjectRoute::Fitted
        };
        if route != ObjectRoute::Fitted {
            let object = jope_runtime::model_object::ModelObject::read(model, limit)?;
            return match route {
                ObjectRoute::MathPipeline => {
                    run.predict_math_pipeline(&object, model, input, output, started)
                }
                _ => run.predict_calibrated(&object, model, input, output, &b, started),
            };
        }
        let fitted = jope_runtime::model_object::load_fitted(model, limit)?;
        let identity_path = if model.is_dir() {
            model.join("manifest.json")
        } else {
            model.to_owned()
        };
        let bytes = jope_runtime::model_object::read_bounded(&identity_path, limit)?;
        run.receipt.inputs.push(FileRecord {
            role: "model",
            path: model.display().to_string(),
            bytes: bytes.len() as u64,
            sha256: jope_runtime::sha256_hex(&bytes),
        });
        run.receipt.timings.read_seconds += started.elapsed().as_secs_f64();
        run.receipt.method = Some(MethodRecord {
            requested: "model".into(),
            selected: fitted.header.method.id.clone(),
            recipe_sha256: fitted.header.recipe_sha256.clone(),
        });
        run.receipt.task = Some(TaskRecord {
            task: fitted.header.task.clone(),
            reason: "from the model file".into(),
            target: fitted.header.target.clone(),
            classes: fitted.header.class_order.len(),
            rows: fitted.header.training_procedure.rows,
            features: fitted.header.input_schema.len(),
        });
        run.predict(&fitted, input, output, &b)?;
        Ok(())
    })();
    run.finish(result)
}

#[cfg(feature = "mcp")]
pub(crate) fn run_predict_remote(
    run: &str,
    algorithm: Option<&str>,
    input: &Path,
    output: &Path,
) -> anyhow::Result<()> {
    use jope_api::{
        ids::{AlgorithmId, RunId},
        PredictRequest, RunSnapshot,
    };
    use jope_client::{files, Transport};
    use std::{io::Read, str::FromStr};
    let client = cmd::dataset::resolve_client()?;
    let run_id = RunId::from_str(run).map_err(|_| anyhow::anyhow!("invalid run id"))?;
    let snapshot: RunSnapshot = client.run_status(&run_id)?;
    let algorithm = match algorithm {
        Some(value) => {
            AlgorithmId::from_str(value).map_err(|_| anyhow::anyhow!("invalid algorithm id"))?
        }
        None => {
            snapshot
                .best
                .ok_or_else(|| anyhow::anyhow!("run has no best algorithm"))?
                .algorithm_id
        }
    };
    let record = client.algorithm_get(&run_id, &algorithm)?;
    let artifact = record
        .model_artifact
        .ok_or_else(|| anyhow::anyhow!("algorithm has no model artifact"))?;
    let register_key = cmd::common::resolve_idempotency_key(None);
    cmd::common::write_idempotency_key(&mut std::io::stderr().lock(), &register_key)?;
    let name = input
        .file_name()
        .map(|v| v.to_string_lossy().into_owned())
        .unwrap_or_else(|| "prediction.csv".into());
    let registered = files::upload_csv(&client, &name, std::fs::File::open(input)?, &register_key)?
        .json::<jope_api::DatasetProfile>()?;
    let predict_key = cmd::common::resolve_idempotency_key(None);
    cmd::common::write_idempotency_key(&mut std::io::stderr().lock(), &predict_key)?;
    let request = PredictRequest {
        model_artifact: artifact,
        dataset: registered.dataset_id,
    };
    let mut response = client.model_predict(&request, &predict_key)?;
    let mut bytes = Vec::new();
    response
        .body
        .by_ref()
        .take((1_u64 << 30) + 1)
        .read_to_end(&mut bytes)?;
    anyhow::ensure!(
        bytes.len() as u64 <= (1_u64 << 30),
        "prediction output exceeds 1 GiB"
    );
    if output == Path::new("-") {
        use std::io::Write;
        std::io::stdout().lock().write_all(&bytes)?;
    } else {
        io::write_atomic("predictions", output, &bytes)?;
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ObjectRoute {
    MathPipeline,
    Calibrated,
    Fitted,
}

fn object_route(object_dir: &Path) -> anyhow::Result<ObjectRoute> {
    use jope_runtime::model_object::{
        calibration_platt, read_bounded, MANIFEST, MATH_PIPELINE, MAX_MANIFEST_BYTES,
    };
    #[derive(serde::Deserialize)]
    struct Format {
        format: String,
        #[serde(default)]
        required_capabilities: Vec<String>,
    }
    let bytes = read_bounded(&object_dir.join(MANIFEST), MAX_MANIFEST_BYTES)?;
    let manifest: Format = serde_json::from_slice(&bytes)
        .map_err(|e| Failure::integrity(format!("model manifest: {e}")))?;
    Ok(if manifest.format == MATH_PIPELINE {
        ObjectRoute::MathPipeline
    } else if manifest
        .required_capabilities
        .iter()
        .any(|c| c == calibration_platt::CAPABILITY)
    {
        ObjectRoute::Calibrated
    } else {
        ObjectRoute::Fitted
    })
}

pub(crate) fn run_fit_predict(
    train: &TrainArgs,
    input: &Path,
    output: &Path,
    common: &CommonArgs,
) -> ExitCode {
    let mut run = Run::new("fit-predict", common);
    let result = (|| -> anyhow::Result<()> {
        let landmark_inputs = train.landmarks.protected_inputs();
        let mut protected = vec![train.train.as_path(), input];
        protected.extend(landmark_inputs.iter().map(std::path::PathBuf::as_path));
        run.refuse_aliasing(output, &protected)?;
        landmarks::refuse_aliasing(train, common, &[input, output])?;
        let b = run.prepare(train.landmarks.enabled())?;
        let fitted = run.fit(train, &b)?;
        run.predict(&fitted, input, output, &b)?;
        Ok(())
    })();
    run.finish(result)
}

pub(crate) fn report(result: anyhow::Result<()>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("jain: {error:#}");
            ExitCode::from(
                common::report_exit_code(&error).unwrap_or_else(|| kind_of(&error).exit_code()),
            )
        }
    }
}

pub(crate) fn load_fitness(text: &str) -> anyhow::Result<jope_runtime::fitness::Fitness> {
    let value = if let Some(path) = text.strip_prefix('@') {
        let mut file = std::fs::File::open(path)?;
        use std::io::Read;
        let mut bytes = Vec::new();
        file.by_ref()
            .take((jope_runtime::fitness::MAX_SPEC_BYTES + 1) as u64)
            .read_to_end(&mut bytes)?;
        String::from_utf8(bytes)?
    } else {
        text.to_owned()
    };
    jope_runtime::fitness::Fitness::parse(&value)
}

pub(crate) fn serve(mcp: bool) -> anyhow::Result<()> {
    use std::io::{BufRead, Read, Write};
    let package = bootstrap::open()?;
    let mut server = jope_runtime::service::Server::new(CreatedBy {
        source_sha40: bootstrap::source_sha40(),
        package_manifest_sha256: package
            .as_ref()
            .map(|p| bootstrap::summary(p).manifest_sha256),
        package_profile: package.as_ref().map(|p| p.manifest().profile.clone()),
    });
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    loop {
        let mut bytes = Vec::new();
        input
            .by_ref()
            .take((jope_runtime::service::MAX_MESSAGE_BYTES + 1) as u64)
            .read_until(b'\n', &mut bytes)?;
        if bytes.is_empty() {
            break;
        }
        let response = if bytes.len() > jope_runtime::service::MAX_MESSAGE_BYTES {
            // Skip the rest of the oversized line and answer it; the session goes on.
            if !bytes.ends_with(b"\n") {
                let mut rest = Vec::new();
                loop {
                    rest.clear();
                    let read = input
                        .by_ref()
                        .take(64 * 1024)
                        .read_until(b'\n', &mut rest)?;
                    if read == 0 || rest.ends_with(b"\n") {
                        break;
                    }
                }
            }
            Some(
                serde_json::json!({"jsonrpc":"2.0","id":null,"error":{"code":-32600,
                "message":format!("message larger than {} bytes",
                    jope_runtime::service::MAX_MESSAGE_BYTES)}}),
            )
        } else {
            match serde_json::from_slice(&bytes) {
                Ok(value) => server.dispatch(value, mcp),
                Err(error) => Some(
                    serde_json::json!({"jsonrpc":"2.0", "id":null,"error":{"code":-32700,"message":error.to_string()}}),
                ),
            }
        };
        if let Some(response) = response {
            serde_json::to_writer(&mut output, &response)?;
            writeln!(output)?;
            output.flush()?;
        }
    }
    Ok(())
}
