use super::*;

const MAX_ROWS: usize = 10_000;
const MAX_COLUMNS: usize = 1_024;
const MAX_CELLS: usize = 1_000_000;

pub(super) fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

/// Decode only the retained fitted state. Every caller, including private
/// contract fixtures, uses the same rejection contract; no fitting fallback.
pub(super) fn load_retained_model(bytes: &[u8]) -> Result<feat_math::Model> {
    serde_json::from_slice(bytes)
        .context("retained model format unavailable; refitting is forbidden")
}

fn monotonic_ns() -> Result<u64> {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `time` points to a live, writable timespec and has the exact
    // layout expected by clock_gettime; the return code gates its use below.
    ensure!(
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) } == 0,
        "query monotonic clock unavailable"
    );
    (time.tv_sec as u64)
        .checked_mul(1_000_000_000)
        .and_then(|n| n.checked_add(time.tv_nsec as u64))
        .context("query clock overflow")
}
fn vector_bits(values: &[f64]) -> Vec<u8> {
    let mut bytes = (values.len() as u64).to_le_bytes().to_vec();
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}
pub(super) fn matrix_sha256(rows: &[Vec<f64>]) -> String {
    let mut bytes = (rows.len() as u64).to_le_bytes().to_vec();
    for row in rows {
        bytes.extend_from_slice(&vector_bits(row));
    }
    hash(&bytes)
}
pub(super) fn receipt_digest(receipt: &QueryReceipt) -> Result<String> {
    let mut value = serde_json::to_value(receipt)?;
    value
        .as_object_mut()
        .context("query receipt object required")?
        .remove("receipt_sha256");
    Ok(hash(&serde_json::to_vec(&value)?))
}
pub(super) fn execution_query_key(
    input_key: &str,
    executable: &str,
    runtime: &str,
) -> Result<String> {
    Ok(hash(&serde_json::to_vec(
        &json!({"schema":"parallel.query_execution_identity.v1",
        "inputs_sha256":input_key,"executable_sha256":executable,"runtime_sha256":runtime,
        "scorer_source_sha256":hash(include_bytes!("query.rs")),
        "predictor_source_sha256":hash(include_bytes!("../../../feat-math/src/model.rs")),
        "preprocessing_source_sha256":hash(include_bytes!("../prepare.rs"))}),
    )?))
}

pub(super) struct PreparedQuery {
    pub(super) original: CompletionReceipt,
    pub(super) rows: Vec<Vec<f64>>,
    pub(super) preprocessing: Preprocessing,
    pub(super) query_key: String,
    pub(super) query_rows_sha256: String,
    pub(super) targets_sha256: String,
}
pub(super) fn prepare_query(
    completion: &[u8],
    model: &[u8],
    inputs: &QueryInputs,
) -> Result<PreparedQuery> {
    ensure!(
        inputs.schema == "parallel.query_inputs.v1",
        "unsupported query input schema"
    );
    inputs.contract.verify()?;
    ensure!(
        valid_sha(&inputs.completion_sha256)
            && valid_sha(&inputs.access_receipt_sha256)
            && !inputs.scope.trim().is_empty()
            && !inputs.task_id.trim().is_empty(),
        "query completion/access identity or provenance absent"
    );
    ensure!(
        hash(completion) == inputs.completion_sha256,
        "original query completion changed"
    );
    let original = verify_completion_receipt(completion)?;
    ensure!(
        original.outcome == ExecutionOutcome::Success,
        "unsuccessful original fit has no query quality"
    );
    ensure!(
        original.model_sha256.as_deref() == Some(hash(model).as_str()),
        "retained query model unavailable or changed"
    );
    ensure!(
        hash(&inputs.preprocessing) == original.fit_identity.preprocessing_sha256
            && hash(&inputs.feature_view) == original.fit_identity.feature_view_sha256,
        "query preprocessing or learned feature view differs from original fit"
    );
    // Identity-only preprocessing does not retain a provable feature width.
    // Compact views need their own qualified reader and are deliberately denied.
    let pp: Preprocessing = serde_json::from_slice(&inputs.preprocessing)
        .context("query needs retained training preprocessing and feature dimensions")?;
    ensure!(
        pp.schema == "parallel.preprocessing.v1"
            && pp.labels.is_empty()
            && !pp.columns.is_empty()
            && pp.columns.len() <= MAX_COLUMNS
            && pp.target_mean.is_finite()
            && pp.target_scale.is_finite()
            && pp.target_scale > 0.,
        "unsupported or malformed regression preprocessing"
    );
    for column in &pp.columns {
        ensure!(
            matches!(column.kind.as_str(), "numeric" | "categorical")
                && column.impute.is_finite()
                && column.mean.is_finite()
                && column.scale.is_finite()
                && column.scale > 0.
                && column.categories.windows(2).all(|pair| pair[0] < pair[1]),
            "malformed retained feature transformation"
        );
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct RawView {
        protocol: String,
        training_sha256: String,
        preprocessing_sha256: String,
    }
    let view: RawView = serde_json::from_slice(&inputs.feature_view)?;
    ensure!(
        view.protocol == "raw_training_fold_v1"
            && valid_sha(&view.training_sha256)
            && view.preprocessing_sha256 == original.fit_identity.preprocessing_sha256,
        "unsupported compact or conflicting raw feature view"
    );
    ensure!(
        !inputs.rows.is_empty()
            && inputs.rows.len() <= MAX_ROWS
            && inputs
                .rows
                .len()
                .checked_mul(pp.columns.len())
                .is_some_and(|n| n <= MAX_CELLS)
            && inputs.rows.len() == inputs.targets.len()
            && inputs.targets.iter().all(|v| v.is_finite()),
        "invalid or oversized query rows/targets"
    );
    let rows = inputs
        .rows
        .iter()
        .map(|row| pp.features(row))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        rows.iter().flatten().all(|v| v.is_finite()),
        "nonfinite transformed query input"
    );
    let query_rows_sha256 = hash(&serde_json::to_vec(&inputs.rows)?);
    let targets_sha256 = hash(&vector_bits(&inputs.targets));
    let query_key = hash(&serde_json::to_vec(&json!({
        "schema":"parallel.query_identity.v1", "model_sha256":original.model_sha256,
        "original_completion_sha256":inputs.completion_sha256,
        "query_rows_sha256":query_rows_sha256,"targets_sha256":targets_sha256,
        "transformed_query_sha256":matrix_sha256(&rows),
        "preprocessing_sha256":original.fit_identity.preprocessing_sha256,
        "feature_view_sha256":original.fit_identity.feature_view_sha256,"contract":inputs.contract,
    }))?);
    Ok(PreparedQuery {
        original,
        rows,
        preprocessing: pp,
        query_key,
        query_rows_sha256,
        targets_sha256,
    })
}

pub(super) fn metrics(targets: &[f64], predictions: &[f64]) -> Result<Vec<u8>> {
    ensure!(
        !targets.is_empty()
            && targets.len() == predictions.len()
            && targets.iter().chain(predictions).all(|v| v.is_finite()),
        "invalid query predictions or targets"
    );
    let mse = targets
        .iter()
        .zip(predictions)
        .map(|(y, p)| (y - p).powi(2))
        .sum::<f64>()
        / targets.len() as f64;
    ensure!(
        mse.is_finite() && mse >= 0.,
        "nonfinite query loss cannot create quality"
    );
    Ok(serde_json::to_vec(
        &json!({"schema":"parallel.query_metrics.v1",
        "task":"regression","metric":"rmse","units":"original_target",
        "outcome_policy":"strict_finite_v1","rows":targets.len(),"rmse":mse.sqrt()}),
    )?)
}
