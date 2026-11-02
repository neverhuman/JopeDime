use super::*;

pub fn is_sha256(value: &Value) -> bool {
    value.as_str().is_some_and(|s| {
        s.len() == 64
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    })
}

/// References are checked against original bytes, never against caller-provided
/// presence flags. Missing/evicted evidence remains unavailable for admission.
pub fn artifact_bytes(root: &Path, reference: &Value) -> Result<Vec<u8>> {
    ensure!(
        is_sha256(&reference["sha256"]),
        "admission malformed artifact digest"
    );
    let path = resolve(
        root,
        Path::new(
            reference["path"]
                .as_str()
                .context("admission artifact path absent")?,
        ),
    )?;
    let bytes = std::fs::read(path)?;
    ensure!(
        hash(&bytes) == reference["sha256"],
        "admission artifact digest changed"
    );
    Ok(bytes)
}

pub fn artifact(root: &Path, reference: &Value) -> Result<Value> {
    Ok(serde_json::from_slice(&artifact_bytes(root, reference)?)?)
}

/// The supervisor retains these original artifacts. A stale result by itself
/// cannot turn an interrupted, failed, or unobserved execution into a label.
pub fn validate_completion(root: &Path, row: &Value) -> Result<()> {
    let completion = artifact(root, &row["clean_completion"])?;
    ensure!(
        completion["schema"] == COMPLETION_SCHEMA,
        "admission original completion absent"
    );
    let attempt = artifact(root, &completion["attempt"])?;
    for key in [
        "evaluation_identity_sha256",
        "program_sha256",
        "evaluator_sha256",
    ] {
        ensure!(
            is_sha256(&row[key]) && completion[key] == row[key] && attempt[key] == row[key],
            "admission attempt/completion identity conflict: {key}"
        );
    }
    let deadline = completion["deadline_seconds"]
        .as_f64()
        .context("admission completion deadline absent")?;
    ensure!(
        deadline.is_finite() && deadline > 0. && deadline <= 30.,
        "admission invalid deadline"
    );
    let exceeded = completion["deadline_exceeded"]
        .as_bool()
        .context("admission deadline outcome absent")?;
    let observed = completion["exit_observed"]
        .as_bool()
        .context("admission observed exit absent")?;
    ensure!(
        completion.get("exit_code").is_some()
            && completion.get("signal").is_some()
            && completion.get("runtime_seconds").is_some(),
        "admission incomplete process outcome"
    );
    let runtime = completion["runtime_seconds"].as_f64();
    ensure!(
        completion["runtime_seconds"].is_null()
            || runtime.is_some_and(|r| r.is_finite() && r >= 0.),
        "admission invalid measured runtime"
    );
    ensure!(
        row.get("runtime_seconds").is_some()
            && row["runtime_seconds"] == completion["runtime_seconds"],
        "admission original runtime conflict"
    );
    let outcome = completion["outcome"]
        .as_str()
        .context("admission execution outcome absent")?;
    let native_bytes = artifact_bytes(root, &completion["fit_completion"])?;
    crate::eval_registry::verify_completion_receipt(&native_bytes)?;
    let native: Value = serde_json::from_slice(&native_bytes)?;
    ensure!(
        row.get("numerical_runtime_seconds").is_some()
            && row["numerical_runtime_seconds"] == native["elapsed_seconds"],
        "admission original numerical runtime conflict"
    );
    for (native_key, row_key) in [
        ("program_sha256", "program_sha256"),
        ("executable_sha256", "evaluator_sha256"),
    ] {
        ensure!(
            is_sha256(&row[row_key]) && native[native_key] == row[row_key],
            "admission native execution identity conflict: {native_key}"
        );
    }
    if row.get("failure").is_some_and(Value::is_null) {
        ensure!(
            outcome == "success"
                && observed
                && completion["exit_code"] == 0
                && completion["signal"].is_null()
                && !exceeded
                && runtime.is_some_and(|r| r <= deadline),
            "admission execution did not complete cleanly"
        );
        ensure!(
            native["outcome"] == "success",
            "admission original model did not fit successfully"
        );
        ensure!(
            is_sha256(&native["model_sha256"])
                && native["model_sha256"] == completion["model"]["sha256"],
            "admission original model identity conflict"
        );
        artifact_bytes(root, &completion["model"])?;
        ensure!(
            artifact(root, &completion["output"])? == row["metrics"],
            "admission original output/metrics conflict"
        );
    } else {
        ensure!(
            row["failure"]
                .as_str()
                .is_some_and(|s| !s.trim().is_empty())
                && row["metrics"].is_null(),
            "admission failed execution carries quality or lacks explicit failure"
        );
        match outcome {
            "fit_failure" => ensure!(
                observed
                    && !exceeded
                    && completion["exit_code"].as_i64().is_some_and(|c| c != 0)
                    && completion["signal"].is_null()
                    && native["outcome"] == "deterministic_failure",
                "admission invalid deterministic failure"
            ),
            "timeout" => ensure!(exceeded, "admission timeout lacks censoring threshold"),
            "infrastructure_failure" | "unknown" => {}
            _ => anyhow::bail!("admission invalid failed outcome"),
        }
    }
    Ok(())
}

pub fn identity(row: &Value) -> Value {
    let keys = [
        "program_sha256",
        "family",
        "partition",
        "prepared_sha256",
        "cut",
        "training_sha256",
        "validation_sha256",
        "evaluator_sha256",
        "evaluator_protocol",
        "preprocessing_sha256",
        "target_scale",
        "metric_units",
        "vocabulary_sha256",
        "feature_view_sha256",
        "evaluation_seed",
        "resource_protocol",
    ];
    let mut value = serde_json::Map::new();
    for key in keys {
        value.insert(key.into(), row[key].clone());
    }
    Value::Object(value)
}

pub fn raw_view(training: &str, preprocessing: &str) -> String {
    hash(
        &serde_json::to_vec(&json!({"protocol":"raw_training_fold_v1",
        "training_sha256":training,"preprocessing_sha256":preprocessing}))
        .unwrap(),
    )
}

pub fn validate_receipt(search: &Value, row: &Value, rows: usize, expected: &Value) -> Result<()> {
    ensure!(
        expected["schema"] == POLICY_SCHEMA && is_sha256(&expected["evaluator_sha256"]),
        "admission frozen policy absent"
    );
    for key in [
        "evaluator_sha256",
        "prepared_sha256",
        "vocabulary_sha256",
        "observations_sha256",
    ] {
        ensure!(
            is_sha256(&search[key]),
            "admission malformed search identity: {key}"
        );
    }
    ensure!(
        search["schema"] == "parallel.search.v1"
            && search["completed_evaluations"] == rows
            && search["completed_at"].as_u64().is_some_and(|t| t > 0)
            && search["final_tests_opened"] == false,
        "admission incomplete or invalid search receipt"
    );
    ensure!(
        search["evaluator_sha256"] == expected["evaluator_sha256"]
            && search["evaluator_protocol"] == expected["evaluator_protocol"]
            && search["prepared_sha256"] == row["prepared_sha256"]
            && search["cut"] == row["cut"]
            && search["vocabulary_sha256"] == row["vocabulary_sha256"],
        "admission search/observation identity conflict"
    );
    Ok(())
}

/// `expected` is an experiment-frozen evaluator manifest, not a value supplied
/// by the observation. Historical observations missing fields are quarantined.
pub fn verify_inputs(root: &Path, context: &ContextRow, prepared: &Prepared) -> Result<String> {
    ensure!(
        matches!(context.partition.as_str(), "meta_train" | "meta_validation")
            && prepared.task == "regression",
        "admission sealed or unsupported task inputs"
    );
    ensure!(
        prepared.dataset == context.dataset
            && prepared.family == context.family
            && prepared.meta_partition == context.partition
            && prepared.clean_evaluation
            && !context.family.trim().is_empty(),
        "admission prepared/context provenance conflict"
    );
    let path = resolve(root, &context.prepared)?;
    // Route R8: the preparation is read once and bound by its digest. The
    // caller's parse (checked above) must be exactly that preparation, and the
    // views are resolved from it; `prepared.json` is never reopened.
    let bytes = read_verified(
        &path.join("prepared.json"),
        &context.prepared_sha256,
        "admission prepared artifact",
    )?;
    let bound: Prepared =
        serde_json::from_slice(&bytes).context("parse admission prepared artifact")?;
    ensure!(
        serde_json::to_value(&bound)? == serde_json::to_value(prepared)?,
        "admission prepared parse differs from the bound preparation bytes"
    );
    let mut training =
        crate::prepare::load_bound_view_in_root(root, &path, &bound, context.cut, true)?;
    let targets: Vec<_> = training
        .iter_mut()
        .map(|row| row.pop().context("admission target absent"))
        .collect::<Result<_>>()?;
    let digest = crate::eval_registry::ordered_training_sha256(&training, &targets)?;
    drop(training);
    crate::prepare::load_bound_view_in_root(root, &path, &bound, context.cut, false)?;
    Ok(digest)
}
