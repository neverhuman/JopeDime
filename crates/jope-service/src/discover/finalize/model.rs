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

fn ids(part: &Partition) -> Vec<AuditRowId> {
    part.rows.clone()
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
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
    for row in 0..matrix.rows() {
        rows.push(columns.iter().map(|&column| match matrix.cell(row, column) {
            Some(JmatCell::Numeric(v)) if v.is_finite() => Ok(Some(v.to_string())),
            _ => bail!("M1 finalization requires complete finite numeric features and target; frozen math-pipeline cannot preserve mean-imputed missing values"),
        }).collect::<Result<Vec<_>>>()?);
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
struct Envelope {
    attempt: Attempt,
    model_json: String,
    model_digest: String,
    predictions_digest: String,
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
pub(crate) fn publish(store: &Store, object: &ModelObject) -> Result<Vec<BlobRef>> {
    std::iter::once(object.manifest_bytes())
        .chain(object.parts().values().map(Vec::as_slice))
        .map(|bytes| store.publish_blob(bytes).map_err(anyhow::Error::from))
        .collect()
}
