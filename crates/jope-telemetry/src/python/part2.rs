// ---- registry ------------------------------------------------------------------------------

#[pyfunction]
fn metric_registry() -> Vec<(u16, &'static str, &'static str, &'static str, bool)> {
    crate::train::METRICS
        .iter()
        .map(|m| (m.id, m.name, m.unit, m.row, m.grouped))
        .collect()
}

#[pyfunction]
fn metric_id(name: &str) -> PyResult<u16> {
    crate::train::metric_by_name(name)
        .map(|m| m.id)
        .ok_or_else(|| PyValueError::new_err(format!("unknown telemetry metric {name}")))
}

/// Freeze a dataset-family split with the canonical jope-datasets implementation.
#[pyfunction]
#[pyo3(signature = (dataset_families, train_count, holdout_count, seed=0))]
fn freeze_source_family_split_py<'py>(
    py: Python<'py>,
    dataset_families: &Bound<'py, PyAny>,
    train_count: usize,
    holdout_count: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyAny>> {
    let families: BTreeMap<String, String> =
        serde_json::from_str(&to_json(dataset_families)?).map_err(value_error)?;
    let split = rust_freeze_source_family_split(&families, train_count, holdout_count, seed)
        .map_err(value_error)?;
    to_py(py, &split)
}

/// Derive the label-independent binary dataset source-family identity in Rust.
#[pyfunction]
fn derive_source_family(meta_path: &str) -> PyResult<String> {
    derive_binary_source_family(std::path::Path::new(meta_path)).map_err(value_error)
}

/// Build the canonical split-view and train-only binary target contracts in Rust.
#[pyfunction]
#[pyo3(signature = (dataset, source_family, outer_split, row_ids, train_sha256, training_targets, test_sha256=None))]
#[allow(clippy::too_many_arguments)]
fn build_binary_dataset_contracts<'py>(
    py: Python<'py>,
    dataset: &str,
    source_family: &str,
    outer_split: &str,
    row_ids: Vec<String>,
    train_sha256: &str,
    training_targets: Vec<f64>,
    test_sha256: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let outer_split = match outer_split {
        "train" => OuterSplit::Train,
        "holdout" => OuterSplit::Holdout,
        _ => return Err(value_error("outer_split must be 'train' or 'holdout'")),
    };
    let manifest = build_split_view_manifest(
        dataset,
        source_family,
        Task::Binary,
        outer_split,
        &row_ids,
        train_sha256,
        test_sha256,
    )
    .map_err(value_error)?;
    let summaries = build_binary_target_summaries(&training_targets, &row_ids, &manifest)
        .map_err(value_error)?;
    let manifest_sha256 = manifest.sha256().map_err(value_error)?;
    let result = serde_json::json!({
        "manifest": manifest.to_json(),
        "manifest_sha256": manifest_sha256,
        "target_summaries": summaries.iter().map(|summary| summary.to_json()).collect::<Vec<_>>(),
    });
    to_py(py, &result)
}
