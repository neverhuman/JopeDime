use super::*;

pub(super) fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub(super) fn prediction_bytes(predictions: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(predictions.len() * 8);
    for prediction in predictions {
        bytes.extend_from_slice(&prediction.to_le_bytes());
    }
    bytes
}

pub(super) fn probability_bytes(probabilities: &[Vec<f64>]) -> Vec<u8> {
    let values = probabilities.iter().map(Vec::len).sum::<usize>();
    let mut bytes = Vec::with_capacity(values * 8);
    for row in probabilities {
        for probability in row {
            bytes.extend_from_slice(&probability.to_le_bytes());
        }
    }
    bytes
}

pub(super) fn balanced_accuracy(y: &[f64], predictions: &[f64]) -> f64 {
    let (mut true_positive, mut positive, mut true_negative, mut negative) =
        (0usize, 0usize, 0usize, 0usize);
    for (target, prediction) in y.iter().zip(predictions) {
        if *target == 1.0 {
            positive += 1;
            true_positive += usize::from(*prediction >= 0.5);
        } else {
            negative += 1;
            true_negative += usize::from(*prediction < 0.5);
        }
    }
    0.5 * (true_positive as f64 / positive as f64 + true_negative as f64 / negative as f64)
}

pub(super) fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

pub(super) fn short_hostname() -> String {
    Command::new("hostname")
        .arg("-s")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Numeric CSV with an optional first header row, LAST column = target. Mirrors jope-dime's
/// `data::read_csv_text_core(IndicatorMode::None)` exactly: trimmed fields,
/// `str::parse::<f64>`, non-finite features silently become 0.0, targets must
/// be finite, constant width enforced.
pub(super) fn read_numeric_csv(text: &str, source: &str) -> Result<Data> {
    let (mut x, mut y, mut features) = (Vec::new(), Vec::new(), None);
    let mut records_seen = 0usize;
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        records_seen += 1;
        let values = line
            .split(',')
            .map(|field| {
                field
                    .trim()
                    .parse::<f64>()
                    .with_context(|| format!("{source}:{} invalid numeric field", line_no + 1))
            })
            .collect::<Result<Vec<_>>>();
        let mut values = match values {
            Ok(values) => values,
            Err(_) if records_seen == 1 => continue,
            Err(error) => return Err(error),
        };
        if values.len() < 2 {
            bail!(
                "{source}:{} needs at least one feature and one target",
                line_no + 1
            );
        }
        let width = values.len() - 1;
        match features {
            Some(expected) if expected != width => {
                bail!(
                    "{source}:{} expected {expected} features, found {width}",
                    line_no + 1
                )
            }
            None => features = Some(width),
            _ => {}
        }
        let target = values.pop().expect("length checked");
        if !target.is_finite() {
            bail!("{source}:{} target is not finite", line_no + 1);
        }
        values.iter_mut().for_each(|v| {
            if !v.is_finite() {
                *v = 0.0;
            }
        });
        x.push(values);
        y.push(target);
    }
    let features = features.context("empty CSV")?;
    Ok(Data { x, y, features })
}

pub(super) fn load_pinned_csv(path: &str, pinned_sha256: &str, label: &str) -> Result<Data> {
    let bytes = fs::read(path).with_context(|| format!("read {label} CSV {path}"))?;
    let actual = sha256_hex(&bytes);
    if actual != pinned_sha256 {
        bail!("{label} CSV {path} sha256 {actual} does not match manifest pin {pinned_sha256}");
    }
    read_numeric_csv(std::str::from_utf8(&bytes)?, path)
}

pub(super) fn load_pinned_timeseries_csv(
    path: &str,
    pinned_sha256: &str,
    label: &str,
) -> Result<TimeSeriesTable> {
    let bytes = fs::read(path).with_context(|| format!("read {label} CSV {path}"))?;
    let actual = sha256_hex(&bytes);
    if actual != pinned_sha256 {
        bail!("{label} CSV {path} sha256 {actual} does not match manifest pin {pinned_sha256}");
    }
    read_timeseries_csv_bytes(&bytes, path)
}

pub(super) fn load_pinned_json(
    path: &str,
    pinned_sha256: &str,
    label: &str,
) -> Result<serde_json::Value> {
    let bytes = fs::read(path).with_context(|| format!("read {label} JSON {path}"))?;
    let actual = sha256_hex(&bytes);
    if actual != pinned_sha256 {
        bail!("{label} JSON {path} sha256 {actual} does not match manifest pin {pinned_sha256}");
    }
    serde_json::from_slice(&bytes).with_context(|| format!("parse {label} JSON {path}"))
}

/// Resolve the pinned reduction version from the shard schema. v1 manifests
/// must not carry a reduction pin (old binaries would ignore it silently);
/// v2 timeseries manifests must carry one; the pin is meaningless for other
/// tasks and is rejected there rather than ignored.
pub(super) fn resolve_reduction(shard: &Shard) -> Result<ReductionVersion> {
    match shard.schema.as_str() {
        SHARD_SCHEMA => {
            if shard.reduction.is_some() {
                bail!("reduction pin requires shard schema {SHARD_SCHEMA_V2}");
            }
            Ok(ReductionVersion::V1)
        }
        SHARD_SCHEMA_V2 => {
            if shard.task == "timeseries" {
                let spec = shard.reduction.as_ref().with_context(|| {
                    format!("{SHARD_SCHEMA_V2} timeseries shards must pin reduction.version")
                })?;
                ReductionVersion::parse(&spec.version)
            } else {
                if shard.reduction.is_some() {
                    bail!("reduction pins apply only to timeseries shards");
                }
                Ok(ReductionVersion::V1)
            }
        }
        other => bail!(
            "unsupported shard schema '{other}' (expected {SHARD_SCHEMA} or {SHARD_SCHEMA_V2})"
        ),
    }
}

pub(super) fn timeseries_manifest_spec(shard: &Shard) -> Result<TimeSeriesManifestSpec> {
    Ok(TimeSeriesManifestSpec {
        horizon: shard
            .horizon
            .context("timeseries manifest missing horizon")?,
        seasonality: shard
            .seasonality
            .context("timeseries manifest missing seasonality")?,
        frequency: shard
            .frequency
            .clone()
            .filter(|value| !value.trim().is_empty())
            .context("timeseries manifest missing frequency")?,
        target_is_final_column: shard
            .target_is_final_column
            .context("timeseries manifest missing target_is_final_column")?,
        known_future_covariates: shard.known_future_covariates.clone(),
        past_only_covariates: shard.past_only_covariates.clone(),
    })
}

pub(super) fn load_timeseries_shard_data(shard: &Shard) -> Result<TimeSeriesData> {
    let meta_json = shard
        .meta_json
        .as_deref()
        .context("timeseries manifest missing meta_json")?;
    let meta_sha256 = shard
        .meta_sha256
        .as_deref()
        .context("timeseries manifest missing meta_sha256")?;
    let train = load_pinned_timeseries_csv(&shard.train_csv, &shard.train_sha256, "train")?;
    let test = load_pinned_timeseries_csv(&shard.test_csv, &shard.test_sha256, "test")?;
    let meta = load_pinned_json(meta_json, meta_sha256, "meta")?;
    let spec = timeseries_manifest_spec(shard)?;
    validate_timeseries_data(train, test, &meta, &spec)
}

pub(super) fn binary_schema(features: usize) -> DataSchema {
    DataSchema {
        columns: (0..features)
            .map(|j| ColumnSpec::numeric(format!("c{j}")))
            .collect(),
        row_semantics: RowSemantics::Iid,
    }
}

/// Shortest-round-trip string table (mirrors binary_cv::raw_table).
pub(super) fn raw_table(features: usize, rows: &Rows) -> Result<RawTable> {
    let headers = (0..features).map(|j| format!("c{j}")).collect::<Vec<_>>();
    let cells = rows
        .iter()
        .map(|row| row.iter().map(|v| format!("{v:?}")).collect())
        .collect();
    RawTable::new(headers, cells)
}

pub(super) fn raw_table_named(headers: &[String], rows: &[Vec<f64>]) -> Result<RawTable> {
    let cells = rows
        .iter()
        .map(|row| row.iter().map(|v| format!("{v:?}")).collect())
        .collect();
    RawTable::new(headers.to_vec(), cells)
}

/// Per-task authority metric (R1/D2); must agree with
/// `jope_dime_regressor::task::TaskKind::primary_metric`.
pub(super) fn task_metric(task: &str) -> Option<&'static str> {
    Task::parse(task).map(Task::primary_metric)
}

pub(super) fn secondary_metric(task: &str) -> &'static str {
    match task {
        "binary" => "logloss",
        "timeseries" => "mase",
        "multiclass" => "accuracy",
        _ => "rmse",
    }
}

pub(super) fn is_head_like(op: &str) -> bool {
    is_head_op(op) || V3_EXTRA_HEADS.contains(&op)
}

pub(super) fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(super) fn is_strict_wave(value: &str) -> bool {
    value
        .strip_prefix("strict-wave-")
        .is_some_and(|suffix| suffix.len() == 6 && suffix.bytes().all(|byte| byte.is_ascii_digit()))
}

pub(super) fn is_multiclass_generation(value: &str) -> bool {
    value
        .strip_prefix("multiclass-generation-")
        .is_some_and(|suffix| suffix.len() == 3 && suffix.bytes().all(|byte| byte.is_ascii_digit()))
}

/// Resolve receipt labeling before loading any data. The default deliberately
/// preserves the legacy adaptive lane. Corrected authority is fail-closed.
/// Superseded generation-zero manifests are accepted only in their exact old
/// form so already-queued jobs can drain without receiving stronger claims.
pub(super) fn resolve_authority(shard: &Shard) -> Result<ResolvedAuthority> {
    let expected_metric =
        task_metric(&shard.task).with_context(|| format!("unsupported task '{}'", shard.task))?;
    let Some(authority) = &shard.label_authority else {
        return Ok(ResolvedAuthority {
            tier: "test_exact".to_string(),
            metric: expected_metric.to_string(),
            test_adaptive: true,
            benchmark_valid: false,
            selection_bias: Some("test-exact-reused".to_string()),
            generation: "adaptive".to_string(),
            authority_sha256: None,
            frozen_seed_sha256: None,
            model_checkpoint_sha256: None,
            hall_of_fame_sha256: None,
        });
    };
    let corrected = authority.score_tier == "test_exact"
        && authority.metric == expected_metric
        && authority.test_adaptive
        && !authority.benchmark_valid
        && authority.selection_bias.as_deref() == Some("test-exact-reused")
        && is_strict_wave(&authority.generation);
    let multiclass_test_selected = shard.task == "multiclass"
        && authority.score_tier == "test"
        && authority.metric == "macro_f1"
        && authority.test_adaptive
        && !authority.benchmark_valid
        && authority.selection_bias.as_deref() == Some("test-selected")
        && is_multiclass_generation(&authority.generation);
    let timeseries_test_selected = shard.task == "timeseries"
        && authority.score_tier == "test_exact"
        && authority.metric == "negative_mase"
        && authority.test_adaptive
        && !authority.benchmark_valid
        && authority.selection_bias.as_deref() == Some("test-selected")
        && is_strict_wave(&authority.generation);
    let legacy_superseded = authority.score_tier == "test_exact"
        && shard.task != "timeseries"
        && authority.metric == expected_metric
        && !authority.test_adaptive
        && authority.benchmark_valid
        && authority.selection_bias.is_none()
        && authority.generation == "zero";
    if !corrected && !multiclass_test_selected && !timeseries_test_selected && !legacy_superseded {
        bail!(
            "self-play authority requires either the strict-wave test_exact contract or the \
             multiclass score_tier=test, metric=macro_f1, test_adaptive=true, \
             benchmark_valid=false, selection_bias=test-selected, \
             generation=multiclass-generation-NNN contract or the timeseries \
             score_tier=test_exact, metric=negative_mase, test_adaptive=true, \
             benchmark_valid=false, selection_bias=test-selected, \
             generation=strict-wave-NNNNNN contract"
        );
    }
    for (name, value) in [
        ("frozen_seed_sha256", authority.frozen_seed_sha256.as_str()),
        (
            "model_checkpoint_sha256",
            authority.model_checkpoint_sha256.as_str(),
        ),
        (
            "hall_of_fame_sha256",
            authority.hall_of_fame_sha256.as_str(),
        ),
    ] {
        if !is_sha256(value) {
            bail!("label_authority.{name} must be a 64-character hexadecimal sha256");
        }
    }
    let freshness_fields_present = shard.checkpoint_epoch.is_some()
        || shard.checkpoint_sha256.is_some()
        || shard.catalog_sha256.is_some()
        || shard.protocol_sha256.is_some()
        || shard.generated_at_ms.is_some();
    if freshness_fields_present {
        let checkpoint = shard
            .checkpoint_sha256
            .as_deref()
            .context("fresh manifest is missing checkpoint_sha256")?;
        let epoch = shard
            .checkpoint_epoch
            .as_deref()
            .context("fresh manifest is missing checkpoint_epoch")?;
        let catalog = shard
            .catalog_sha256
            .as_deref()
            .context("fresh manifest is missing catalog_sha256")?;
        let protocol = shard
            .protocol_sha256
            .as_deref()
            .context("fresh manifest is missing protocol_sha256")?;
        if checkpoint != authority.model_checkpoint_sha256
            || epoch != format!("checkpoint-{checkpoint}")
            || !is_sha256(catalog)
            || !is_sha256(protocol)
            || shard.generated_at_ms.is_none()
        {
            bail!("fresh manifest checkpoint/catalog/protocol provenance mismatch");
        }
    }
    let authority_sha256 = sha256_hex(&serde_json::to_vec(authority)?);
    Ok(ResolvedAuthority {
        tier: authority.score_tier.clone(),
        metric: authority.metric.clone(),
        test_adaptive: authority.test_adaptive,
        benchmark_valid: authority.benchmark_valid,
        selection_bias: authority.selection_bias.clone(),
        generation: authority.generation.clone(),
        authority_sha256: Some(authority_sha256),
        frozen_seed_sha256: Some(authority.frozen_seed_sha256.clone()),
        model_checkpoint_sha256: Some(authority.model_checkpoint_sha256.clone()),
        hall_of_fame_sha256: Some(authority.hall_of_fame_sha256.clone()),
    })
}
