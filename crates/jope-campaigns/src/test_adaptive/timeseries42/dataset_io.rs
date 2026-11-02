fn campaign_root(run_id: &str) -> PathBuf {
    Path::new(output_root_str()).join(run_id)
}

fn dataset_root(root: &Path, dataset: &str) -> PathBuf {
    root.join("datasets").join(dataset)
}

fn generation_root(root: &Path, generation: usize) -> PathBuf {
    root.join("generations")
        .join(format!("generation-{generation:06}"))
}

fn immutable_or_equal(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.exists() {
        if fs::read(path)? != bytes {
            bail!("immutable artifact collision at {}", path.display());
        }
        return Ok(());
    }
    immutable_write(path, bytes)
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<String> {
    let bytes = canonical_json(value)?;
    immutable_or_equal(path, &bytes)?;
    Ok(sha256_bytes(&bytes))
}

fn replace_outdated_artifact(path: &Path, bytes: &[u8], outdated_stem: &str) -> Result<String> {
    if path.exists() {
        if fs::read(path)? == bytes {
            return Ok(sha256_bytes(bytes));
        }
        let prior = fs::read(path)?;
        let prior_sha256 = sha256_bytes(&prior);
        let archive = path.with_file_name(format!(
            "{outdated_stem}.outdated-{}.v1.json",
            &prior_sha256[..16]
        ));
        immutable_or_equal(&archive, &prior)?;
        fs::remove_file(path).with_context(|| format!("remove outdated {}", path.display()))?;
    }
    immutable_write(path, bytes)?;
    Ok(sha256_bytes(bytes))
}

fn write_current_json<T: Serialize>(path: &Path, value: &T, outdated_stem: &str) -> Result<String> {
    let bytes = canonical_json(value)?;
    replace_outdated_artifact(path, &bytes, outdated_stem)
}

fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    serde_json::from_slice(&fs::read(path)?).with_context(|| format!("parse {}", path.display()))
}

fn finite_metric(value: &Value, key: &str) -> Option<f64> {
    value
        .get("metrics")
        .and_then(|metrics| metrics.get(key))
        .and_then(Value::as_f64)
        .filter(|score| score.is_finite())
}

fn string_array(value: &Value, key: &str) -> Result<Vec<String>> {
    let Some(array) = value.get(key) else {
        return Ok(Vec::new());
    };
    array
        .as_array()
        .with_context(|| format!("meta_json {key} must be an array"))?
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_string)
                .with_context(|| format!("meta_json {key} contains a non-string"))
        })
        .collect()
}

fn required_usize(value: &Value, key: &str) -> Result<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|raw| usize::try_from(raw).ok())
        .with_context(|| format!("meta_json missing numeric {key}"))
}

fn external_reports(dataset_dir: &Path, dataset_hash: &str) -> Result<Vec<ExternalReport>> {
    let reports = dataset_dir.join("reports");
    let mut paths = fs::read_dir(&reports)
        .with_context(|| format!("read {}", reports.display()))?
        .map(|entry| Ok(entry?.path()))
        .collect::<Result<Vec<_>>>()?;
    paths.sort();
    let mut accepted = Vec::new();
    for path in paths {
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let bytes = fs::read(&path)?;
        let value: Value = match serde_json::from_slice(&bytes) {
            Ok(value) => value,
            Err(_) => continue,
        };
        if value.get("task_type").and_then(Value::as_str) != Some("timeseries")
            || !matches!(
                value.get("status").and_then(Value::as_str),
                Some("completed" | "completed-valid")
            )
            || value.get("test_adaptive").and_then(Value::as_bool) == Some(true)
        {
            continue;
        }
        if value
            .get("dataset_hash")
            .and_then(Value::as_str)
            .is_some_and(|bound| bound != dataset_hash)
        {
            continue;
        }
        let model = value
            .get("model_name")
            .or_else(|| value.get("algorithm"))
            .and_then(Value::as_str)
            .unwrap_or_else(|| {
                path.file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("unknown")
            })
            .to_string();
        if model.to_ascii_lowercase().contains("jope") {
            continue;
        }
        let Some(mase) = finite_metric(&value, "mase") else {
            continue;
        };
        let Some(rmse) = finite_metric(&value, "rmse") else {
            continue;
        };
        accepted.push(ExternalReport {
            model,
            mase,
            rmse,
            report_path: path.to_string_lossy().into_owned(),
            report_sha256: sha256_bytes(&bytes),
        });
    }
    if accepted.is_empty() {
        bail!(
            "{} has no finite completed non-Jope external time-series report",
            dataset_dir.display()
        );
    }
    Ok(accepted)
}

fn official_leader(accepted: &[ExternalReport], dataset_dir: &Path) -> Result<OfficialLeader> {
    let best = accepted.iter().min_by(|left, right| {
        left.mase
            .total_cmp(&right.mase)
            .then(left.rmse.total_cmp(&right.rmse))
            .then(left.model.cmp(&right.model))
            .then(left.report_sha256.cmp(&right.report_sha256))
    });
    best.map(|report| OfficialLeader {
        model: report.model.clone(),
        mase: report.mase,
        rmse: report.rmse,
        report_path: report.report_path.clone(),
        report_sha256: report.report_sha256.clone(),
    })
    .with_context(|| {
        format!(
            "{} has no finite non-Jope official time-series MASE report",
            dataset_dir.display()
        )
    })
}

fn embedded_payload_sha256(value: &Value, key: &str) -> Result<String> {
    let embedded = value
        .get(key)
        .and_then(Value::as_str)
        .filter(|digest| digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .with_context(|| format!("missing or invalid {key}"))?
        .to_string();
    let mut unsigned = value.clone();
    unsigned
        .as_object_mut()
        .context("hashed payload must be an object")?
        .remove(key);
    if sha256_bytes(&serde_json::to_vec(&unsigned)?) != embedded {
        bail!("{key} digest drift");
    }
    Ok(embedded)
}

fn validate_coverage_final_receipt(
    path: &Path,
    campaign_id: &str,
    manifest_sha256: &str,
    task: &str,
    expected: usize,
) -> Result<String> {
    let value: Value = load_json(path)?;
    if value.get("schema").and_then(Value::as_str) != Some("coverage.final-receipt.v1")
        || value.get("campaign_id").and_then(Value::as_str) != Some(campaign_id)
        || value
            .get("campaign_manifest_sha256")
            .and_then(Value::as_str)
            != Some(manifest_sha256)
        || value.get("task").and_then(Value::as_str) != Some(task)
        || value.get("expected").and_then(Value::as_u64) != Some(expected as u64)
        || value.get("valid").and_then(Value::as_u64) != Some(expected as u64)
        || value.get("missing").and_then(Value::as_u64) != Some(0)
        || value.get("invalid").and_then(Value::as_u64) != Some(0)
        || value.get("complete").and_then(Value::as_bool) != Some(true)
        || value.get("reports").and_then(Value::as_array).map(Vec::len) != Some(expected)
    {
        bail!("invalid or incomplete coverage receipt {}", path.display());
    }
    let receipt_sha256 = embedded_payload_sha256(&value, "receipt_sha256")?;
    validate_receipt_report_files(&value, expected, true)?;
    Ok(receipt_sha256)
}

fn validate_timeseries_baseline_receipt() -> Result<String> {
    let value: Value = load_json(Path::new(ts_baseline_receipt_str()))?;
    if value.get("schema").and_then(Value::as_str)
        != Some("timeseries42.baseline-all21-final-receipt.v1")
        || value.get("parent_manifest_sha256").and_then(Value::as_str)
            != Some("75194fba4d85aefd9e065cd71bd95d3b9a9744cace74a7e2aa92859887ca8885")
        || value.get("repair_manifest_sha256").and_then(Value::as_str)
            != Some(TS_COVERAGE_MANIFEST_SHA256)
        || value.get("expected").and_then(Value::as_u64) != Some(882)
        || value.get("completed_numeric").and_then(Value::as_u64) != Some(877)
        || value.get("skipped_ineligible").and_then(Value::as_u64) != Some(5)
        || value.get("missing").and_then(Value::as_u64) != Some(0)
        || value.get("invalid").and_then(Value::as_u64) != Some(0)
        || value.get("errored").and_then(Value::as_u64) != Some(0)
        || value.get("complete").and_then(Value::as_bool) != Some(true)
        || value.get("reports").and_then(Value::as_array).map(Vec::len) != Some(882)
    {
        bail!("invalid exact 877+5 time-series baseline receipt");
    }
    let receipt_sha256 = embedded_payload_sha256(&value, "receipt_sha256")?;
    validate_receipt_report_files(&value, 882, false)?;
    Ok(receipt_sha256)
}

fn validate_coverage_binding() -> Result<CoverageBinding> {
    for (path, schema, campaign_id, expected_sha256) in [
        (
            mc_coverage_manifest_str(),
            "multiclass-union-runnable-v5",
            "mc-union-runnable272-v5",
            MC_COVERAGE_MANIFEST_SHA256,
        ),
        (
            ts_coverage_manifest_str(),
            "timeseries42-forecast-campaign-v10",
            "timeseries42-forecast3-v10",
            TS_COVERAGE_MANIFEST_SHA256,
        ),
    ] {
        let value: Value = load_json(Path::new(path))?;
        if value.get("schema_version").and_then(Value::as_str) != Some(schema)
            || value.get("campaign_id").and_then(Value::as_str) != Some(campaign_id)
            || embedded_payload_sha256(&value, "manifest_sha256")? != expected_sha256
        {
            bail!("coverage successor manifest drift at {path}");
        }
    }
    let multiclass_final_receipt_sha256 = validate_coverage_final_receipt(
        Path::new(mc_coverage_receipt_str()),
        "mc-union-runnable272-v5",
        MC_COVERAGE_MANIFEST_SHA256,
        "multiclass",
        11_696,
    )?;
    let timeseries_final_receipt_sha256 = validate_coverage_final_receipt(
        Path::new(ts_coverage_receipt_str()),
        "timeseries42-forecast3-v10",
        TS_COVERAGE_MANIFEST_SHA256,
        "timeseries",
        126,
    )?;
    let timeseries_baseline_receipt_sha256 = validate_timeseries_baseline_receipt()?;
    let set: Value = load_json(Path::new(coverage_comparator_set_str()))?;
    if set.get("schema").and_then(Value::as_str) != Some("test-adaptive-comparator-set.v2")
        || set.get("root").and_then(Value::as_str) != Some(coverage_comparator_root_str())
        || set
            .get("multiclass_coverage_receipt_sha256")
            .and_then(Value::as_str)
            != Some(multiclass_final_receipt_sha256.as_str())
        || set
            .get("timeseries_coverage_receipt_sha256")
            .and_then(Value::as_str)
            != Some(timeseries_final_receipt_sha256.as_str())
        || set
            .get("timeseries_baseline_receipt_sha256")
            .and_then(Value::as_str)
            != Some(timeseries_baseline_receipt_sha256.as_str())
        || set.get("lock_count").and_then(Value::as_u64) != Some(314)
        || set.get("locks").and_then(Value::as_array).map(Vec::len) != Some(314)
    {
        bail!("post-coverage comparator set identity drift");
    }
    let comparator_set_sha256 = embedded_payload_sha256(&set, "comparator_set_sha256")?;
    Ok(CoverageBinding {
        multiclass_manifest_sha256: MC_COVERAGE_MANIFEST_SHA256.to_string(),
        multiclass_final_receipt_sha256,
        timeseries_manifest_sha256: TS_COVERAGE_MANIFEST_SHA256.to_string(),
        timeseries_final_receipt_sha256,
        timeseries_baseline_receipt_sha256,
        comparator_root: coverage_comparator_root_str().to_string(),
        comparator_set_path: coverage_comparator_set_str().to_string(),
        comparator_set_sha256,
    })
}
