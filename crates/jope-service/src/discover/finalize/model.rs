use super::*;

pub(crate) fn put_frozen_model(
    tx: &mut crate::store::Write<'_>,
    key: RecordKey,
    record: Record,
) -> crate::store::Result<()> {
    if let Some(mut old) = tx.get(&key)? {
        if let Some(data) = old.data.as_object_mut() {
            data.remove("parity_receipt");
        }
        if old != record {
            return Err(StoreError::Integrity("finalized artifact collision"));
        }
        Ok(())
    } else {
        tx.put(&key, &record)
    }
}

pub(super) fn ids(part: &Partition) -> Vec<AuditRowId> {
    part.rows.clone()
}
pub(super) fn digest(bytes: &[u8]) -> String {
    crate::digest::sha256_hex(bytes)
}
pub(super) fn is_digest(value: &str) -> bool {
    crate::digest::is_sha256_hex(value)
}

pub(crate) fn load(part: &Partition, plan: &Plan, maximum: u64) -> Result<CsvTable> {
    let bytes = jope_runtime::model_object::read_bounded(&part.path, maximum)?;
    ensure!(digest(&bytes) == part.sha256, "partition digest mismatch");
    let matrix = Matrix::open(bytes)?;
    ensure!(
        matrix.rows() == part.rows.len(),
        "partition roster length differs from matrix"
    );
    let columns = plan
        .features
        .iter()
        .chain(std::iter::once(&plan.target))
        .map(|name| {
            matrix
                .columns()
                .iter()
                .position(|c| &c.name == name)
                .context("finalization column missing")
        })
        .collect::<Result<Vec<_>>>()?;
    let mut rows = Vec::with_capacity(matrix.rows());
    if let Ok(binary) = crate::objective::binary::BinaryObjective::parse_json(&plan.objective) {
        let feature_cols = &columns[..plan.features.len()];
        let target_col = columns[plan.features.len()];
        for row in 0..matrix.rows() {
            let mut parsed_row = Vec::with_capacity(columns.len());
            for &column in feature_cols {
                parsed_row.push(match matrix.cell(row, column) {
                    Some(JmatCell::Missing) => None,
                    Some(JmatCell::Numeric(v)) if v.is_finite() => Some(v.to_string()),
                    Some(JmatCell::Categorical(s)) => Some(s.to_owned()),
                    _ => bail!("binary finalization requires valid feature cells"),
                });
            }
            let cls = crate::discover::r#loop::binary_cell_class(
                matrix.cell(row, target_col),
                binary.class_order(),
            )?;
            parsed_row.push(Some(binary.class_order()[cls].to_owned()));
            rows.push(parsed_row);
        }
    } else {
        for row in 0..matrix.rows() {
            rows.push(columns.iter().map(|&column| match matrix.cell(row, column) {
                Some(JmatCell::Numeric(v)) if v.is_finite() => Ok(Some(v.to_string())),
                _ => bail!("M1 finalization requires complete finite numeric features and target; frozen math-pipeline cannot preserve mean-imputed missing values"),
            }).collect::<Result<Vec<_>>>()?);
        }
    }
    let headers = plan
        .features
        .iter()
        .chain(std::iter::once(&plan.target))
        .cloned()
        .collect();
    Ok(CsvTable {
        headers,
        rows,
        sha256: part.sha256.clone(),
        bytes: matrix.bytes().len() as u64,
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Envelope {
    pub(super) attempt: Attempt,
    pub(super) model_json: String,
    pub(super) model_digest: String,
    pub(super) predictions_digest: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeFit {
    version: u32,
    features: Vec<String>,
    dsl: String,
    min: Vec<f64>,
    max: Vec<f64>,
    fill: Vec<f64>,
    target_min: f64,
    target_scale: f64,
    model: feat_math::Model,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn refit(
    executor: &impl Executor,
    plan: &Plan,
    champion: &Champion,
    table: &CsvTable,
    matrix_path: &std::path::Path,
    matrix_sha: &str,
    kind: Refit,
    scratch: &std::path::Path,
    stage: u32,
    max_model: u64,
) -> Result<ModelObject> {
    let directory = scratch.join(format!("fit-{stage}"));
    fs::create_dir(&directory)?;
    let output = fs::canonicalize(directory)?.join("model.json");
    // The existing private refit mode fits every mapped row and ignores folds.
    // Keep its required two-fold envelope constant-size under the frame bound.
    let maximum = HANDOFF_MAX.min(plan.evaluation_rss_bytes).min(max_model);
    let request = Request {
        attempt: Attempt {
            run: plan.run_id.clone(),
            round: u64::MAX,
            slot: stage,
            attempt: 1,
        },
        matrix_path: matrix_path.into(),
        fold_spec: json!({"version":1,"matrix_sha256":matrix_sha,"target":plan.target,"features":plan.features,"folds":[{"repeat":0,"train":[0],"validation":[1]},{"repeat":0,"train":[1],"validation":[0]}],"refit_output":output,"refit_max_bytes":maximum}),
        dsl: champion.candidate.dsl.clone(),
        objective: json!(plan.objective),
        seed: plan.seed,
        limits: crate::worker::Limits {
            wall_ms: plan.evaluation_wall_ms,
            rss_bytes: plan.evaluation_rss_bytes,
        },
    };
    let response = executor.evaluate(&request);
    ensure!(
        response.attempt == request.attempt && response.valid() && response.failure.is_none(),
        "bounded finalization fit failed: {:?}",
        response.failure.unwrap_or(WorkerFailure::Protocol)
    );
    ensure!(
        response.scores.len() == 3,
        "finalization worker score shape changed"
    );
    let meta = fs::symlink_metadata(&output)?;
    ensure!(
        meta.is_file() && meta.len() <= maximum,
        "invalid private fit handoff"
    );
    let mut bytes = vec![];
    fs::File::open(&output)?
        .take(maximum + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= maximum,
        "private fit handoff exceeds bound"
    );
    let envelope: Envelope = serde_json::from_slice(&bytes)?;
    ensure!(
        envelope.attempt == request.attempt
            && envelope.predictions_digest == response.predictions_digest
            && digest(envelope.model_json.as_bytes()) == envelope.model_digest,
        "private fit fence or digest mismatch"
    );
    if let Ok(binary) = crate::objective::binary::BinaryObjective::parse_json(&plan.objective) {
        let state: crate::discover::r#loop::BinaryRefitState =
            serde_json::from_str(&envelope.model_json)?;
        ensure!(state.version == 1, "unsupported binary refit state version");
        let fitted = jope_runtime::fitted::FittedModel::from_bytes(&state.fitted_bytes)?;
        let fitted_features: Vec<String> = fitted
            .header
            .input_schema
            .iter()
            .map(|col| col.name.clone())
            .collect();
        let expected_classes = [
            binary.class_order()[0].to_owned(),
            binary.class_order()[1].to_owned(),
        ];
        ensure!(
            fitted_features == plan.features
                && fitted.header.target == plan.target
                && fitted.header.class_order.as_slice() == expected_classes.as_slice()
                && fitted.header.method.recipe == champion.candidate.dsl,
            "private binary model contract mismatch"
        );
        let mut object = ModelObject::from_fitted_bytes(state.fitted_bytes)?;
        if let Some(platt) = state.platt {
            object = object.with_platt_calibration(platt)?;
        }
        object.manifest().validate(max_model)?;
        object.verify_executable()?;
        let predictions = predict_binary(&object, table)?;
        let target_idx = table
            .headers
            .iter()
            .position(|c| c == &plan.target)
            .context("target missing")?;
        let mut measured = Vec::with_capacity(predictions.len() * 16);
        for (row, probs) in table.rows.iter().zip(predictions) {
            let label = row[target_idx].as_deref().context("target missing")?;
            let cls = binary
                .class_order()
                .iter()
                .position(|&c| c == label)
                .context("binary target outside class order")? as f64;
            measured.extend(cls.to_le_bytes());
            measured.extend(probs[1].to_le_bytes());
        }
        ensure!(
            digest(&measured) == response.predictions_digest,
            "frozen binary model object cannot exactly represent this native fit"
        );
        return Ok(object);
    }
    let native: NativeFit = serde_json::from_str(&envelope.model_json)?;
    ensure!(
        native.version == 1
            && native.features == plan.features
            && native.dsl == champion.candidate.dsl
            && native.fill.len() == plan.features.len()
            && native.fill.iter().all(|x| x.is_finite())
            && native.min.len() == plan.features.len()
            && native.max.len() == plan.features.len(),
        "private native model contract mismatch"
    );
    let input = InputContract::new(
        plan.features
            .iter()
            .enumerate()
            .map(|(i, name)| NumericColumn {
                name: name.clone(),
                min: native.min[i],
                max: native.max[i],
                constant: native.min[i] == native.max[i],
            })
            .collect(),
        (0..plan.features.len()).collect(),
        TargetScale {
            name: plan.target.clone(),
            min: native.target_min,
            scale: native.target_scale,
        },
    )?;
    let parts: BTreeMap<String, Vec<u8>> = BTreeMap::from([
        (
            PIPELINE_PART.into(),
            serde_json::to_vec(&PipelineState {
                dsl: native.dsl,
                model: native.model,
            })?,
        ),
        (INPUT_PART.into(), serde_json::to_vec(&input)?),
    ]);
    // Run provenance stays in FinalizationReceipt. Canonical predictor metadata
    // must be independent of the run identifier (the §5.6 ModelDigest contract).
    let object = ModelObject::new(
        Manifest {
            schema: SCHEMA.into(),
            minimum_reader_version: 1,
            format: MATH_PIPELINE.into(),
            producer: Producer::default(),
            primary: PIPELINE_PART.into(),
            required_capabilities: vec![],
            files: parts
                .iter()
                .map(|(path, data)| Part::new(path, "application/json", data))
                .collect(),
            metadata: json!({"refit":kind,"algorithm_id":champion.candidate.algorithm_id,"problem_digest":plan.problem_digest,"binary_version":plan.binary_version,"fitting":"selection_only","fit_stage":"champion_refit","prediction_version":2,"holdout_score_scope":"fit_only_model","training_rows":table.rows.len(),"missing_input_policy":"registered math-pipeline decoder: scaled minimum"}),
        },
        parts,
        max_model,
    )?;
    object.verify_executable()?;
    // Verify that frozen serialization preserves the actual bounded fit, including
    // target inversion. Training predictions never choose a new champion.
    let predictions = predict(&object, table)?;
    let target = table
        .headers
        .iter()
        .position(|c| c == &plan.target)
        .context("target missing")?;
    let mut measured = Vec::with_capacity(predictions.len() * 16);
    for (row, prediction) in table.rows.iter().zip(predictions) {
        let label: f64 = row[target].as_deref().context("target missing")?.parse()?;
        measured.extend(label.to_le_bytes());
        measured.extend(prediction.to_le_bytes());
    }
    ensure!(
        digest(&measured) == response.predictions_digest,
        "frozen export contract cannot exactly represent this native fit"
    );
    Ok(object)
}
pub(crate) fn predict(object: &ModelObject, table: &CsvTable) -> Result<Vec<f64>> {
    jope_runtime::budget::Budget::new(None, 1)
        .install(|| object.math_pipeline()?.predict_table(table).map(|x| x.0))?
}
pub(crate) fn score(
    object: &ModelObject,
    table: &CsvTable,
    target: &str,
    objective: &RegressionObjective,
) -> Result<f64> {
    let predictions = predict(object, table)?;
    let column = table
        .headers
        .iter()
        .position(|c| c == target)
        .context("target missing")?;
    let labels = table
        .rows
        .iter()
        .map(|r| {
            r[column]
                .as_deref()
                .context("target missing")?
                .parse::<f64>()
                .map_err(anyhow::Error::from)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(objective.evaluate(&labels, &predictions)?.value)
}

pub(crate) fn predict_binary(object: &ModelObject, table: &CsvTable) -> Result<Vec<[f64; 2]>> {
    use jope_runtime::methods::native::Predictions;
    let raw = if let Some(calibrated) = object.calibrated_input()? {
        calibrated.table(table)?.0
    } else {
        let fitted = object.fitted()?;
        jope_runtime::table::prediction_ignoring(
            table,
            fitted.schema(),
            &fitted.header.target,
            &fitted.header.auxiliary_columns,
        )?
        .0
    };
    let predicted = jope_runtime::api::predict_object(
        object,
        &raw,
        &jope_runtime::budget::Budget::new(None, 1),
    )?;
    let Predictions::Probabilities { values, .. } = predicted else {
        bail!("binary model object did not produce probability predictions");
    };
    ensure!(
        values.len() == table.rows.len(),
        "binary prediction row count mismatch"
    );
    values
        .into_iter()
        .map(|row| {
            ensure!(row.len() == 2, "binary probability width mismatch");
            Ok([row[0], row[1]])
        })
        .collect()
}

pub(crate) fn score_binary(
    object: &ModelObject,
    table: &CsvTable,
    target: &str,
    objective: &crate::objective::binary::BinaryObjective,
) -> Result<f64> {
    let probabilities = predict_binary(object, table)?;
    let column = table
        .headers
        .iter()
        .position(|c| c == target)
        .context("target missing")?;
    let labels: Vec<&str> = table
        .rows
        .iter()
        .map(|r| r[column].as_deref().context("target missing"))
        .collect::<Result<_>>()?;
    Ok(objective.evaluate(&labels, &probabilities)?.value)
}
pub(crate) fn publish(store: &Store, object: &ModelObject) -> Result<Vec<BlobRef>> {
    std::iter::once(object.manifest_bytes())
        .chain(object.parts().values().map(Vec::as_slice))
        .map(|bytes| store.publish_blob(bytes).map_err(anyhow::Error::from))
        .collect()
}
