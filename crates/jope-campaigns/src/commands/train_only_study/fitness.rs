use super::*;

pub(super) fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub(super) fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hex::encode(hash.finalize()))
}

pub(super) fn canonicalize_json(value: Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut sorted = Map::new();
            let mut entries = object.into_iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            for (key, value) in entries {
                sorted.insert(key, canonicalize_json(value));
            }
            Value::Object(sorted)
        }
        Value::Array(values) => Value::Array(values.into_iter().map(canonicalize_json).collect()),
        other => other,
    }
}

pub(super) fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(&canonicalize_json(value.clone()))?)
}

pub(super) fn receipt_digest<T: Serialize>(receipt: &T) -> Result<String> {
    let value = serde_json::to_value(receipt)?;
    Ok(sha256_bytes(&canonical_json_bytes(&value)?))
}

pub(super) fn immutable_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("{} has no parent", path.display()))?;
    fs::create_dir_all(parent)?;
    if path.exists() {
        let present = fs::read(path)?;
        if present == bytes {
            return Ok(());
        }
        bail!(
            "immutable artifact already exists with different bytes: {}",
            path.display()
        );
    }
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("artifact"),
        std::process::id()
    ));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    Ok(())
}

pub(super) fn write_receipt<T: Serialize>(path: &Path, receipt: &T) -> Result<String> {
    let mut value = serde_json::to_value(receipt)?;
    let digest = sha256_bytes(&canonical_json_bytes(&value)?);
    value
        .as_object_mut()
        .context("receipt must serialize as an object")?
        .insert("receipt_sha256".to_string(), Value::String(digest.clone()));
    let mut bytes = canonical_json_bytes(&value)?;
    bytes.push(b'\n');
    immutable_write(path, &bytes)?;
    Ok(digest)
}

pub(super) fn read_receipt<T: DeserializeOwned>(path: &Path) -> Result<(T, String)> {
    let mut value: Value = serde_json::from_slice(
        &fs::read(path).with_context(|| format!("read receipt {}", path.display()))?,
    )?;
    let claimed = value
        .as_object_mut()
        .and_then(|object| object.remove("receipt_sha256"))
        .and_then(|value| value.as_str().map(str::to_owned))
        .context("receipt lacks receipt_sha256")?;
    let actual = sha256_bytes(&canonical_json_bytes(&value)?);
    if claimed != actual {
        bail!("receipt digest mismatch: {}", path.display());
    }
    Ok((serde_json::from_value(value)?, actual))
}

pub(super) fn validate_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{label} is not a SHA-256");
    }
    Ok(())
}

pub(super) fn study_dir(study_root: &Path, run_id: &str) -> Result<PathBuf> {
    if run_id.is_empty()
        || run_id == "."
        || run_id == ".."
        || !run_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        bail!("run id must contain only ASCII letters, digits, '-' or '_'");
    }
    Ok(study_root.join(run_id))
}

pub(super) fn semantic_identity(canonical_genome: &str) -> String {
    sha256_bytes(canonical_genome.as_bytes())
}

pub(super) fn genome_complexity(genome: &GenomeIr) -> usize {
    match genome {
        GenomeIr::Pipeline(pipeline) => pipeline.stages.len(),
        _ => 0,
    }
}

pub(super) fn genome_compatible(canonical_genome: &str, features: usize) -> bool {
    let Ok(genome) = GenomeIr::parse(canonical_genome) else {
        return false;
    };
    let GenomeIr::Pipeline(pipeline) = &genome else {
        return false;
    };
    if genome.canonical() != canonical_genome
        || pipeline.stages.is_empty()
        || pipeline.stages.len().saturating_sub(1) > MAX_TRANSFORMS
    {
        return false;
    }
    let head = &pipeline.stages.last().expect("nonempty").op;
    if matches!(head.as_str(), "logit_ridge" | "logistic") {
        return false;
    }
    // Expressions can address raw columns as cN, colN, or col(N).  Reject a
    // reference outside the dataset's raw schema.  Row summaries and `x` are
    // feature-count independent.
    for stage in &pipeline.stages {
        let Some(ParamValue::Str(expression)) = stage.params.get("expr") else {
            continue;
        };
        if expression_column_indices(expression)
            .into_iter()
            .any(|index| index >= features)
        {
            return false;
        }
    }
    true
}

pub(super) fn expression_column_indices(expression: &str) -> Vec<usize> {
    let bytes = expression.as_bytes();
    let mut result = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        let start = if bytes[index] == b'c' && bytes.get(index + 1).is_some_and(u8::is_ascii_digit)
        {
            Some(index + 1)
        } else if bytes[index..].starts_with(b"col(")
            && bytes.get(index + 4).is_some_and(u8::is_ascii_digit)
        {
            Some(index + 4)
        } else if bytes[index..].starts_with(b"col")
            && bytes.get(index + 3).is_some_and(u8::is_ascii_digit)
        {
            Some(index + 3)
        } else {
            None
        };
        if let Some(start) = start {
            let mut end = start;
            while bytes.get(end).is_some_and(u8::is_ascii_digit) {
                end += 1;
            }
            if let Ok(value) = expression[start..end].parse() {
                result.push(value);
            }
            index = end;
        } else {
            index += 1;
        }
    }
    result
}

pub(super) fn sotarena_metadata_path(sotarena_root: &Path, dataset_hash: &str) -> PathBuf {
    let legacy = sotarena_root.join(dataset_hash).join("meta.json");
    if legacy.is_file() {
        legacy
    } else {
        sotarena_root
            .join("splits/tabular-regression-v1")
            .join(dataset_hash)
            .join("metadata.json")
    }
}

pub(super) fn target_identity(sotarena_root: &Path, dataset_hash: &str) -> Result<String> {
    let meta_path = sotarena_metadata_path(sotarena_root, dataset_hash);
    let value: Value = serde_json::from_slice(
        &fs::read(&meta_path).with_context(|| format!("read {}", meta_path.display()))?,
    )?;
    let field = |key: &str| {
        value
            .get(key)
            .or_else(|| value.pointer(&format!("/acquisition/{key}")))
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_string()
    };
    let material = json!({
        "source_platform": field("source_platform"),
        "source_dataset_id": field("source_dataset_id"),
        "source_dataset_name": field("source_dataset_name"),
        "source_target_column_original": field("source_target_column_original"),
        "target_column": field("target_column"),
        "target_names": value.get("target_names").cloned().unwrap_or(Value::Null),
    });
    let canonical = canonical_json_bytes(&material)?;
    if canonical == b"{\"source_dataset_id\":\"\",\"source_dataset_name\":\"\",\"source_platform\":\"\",\"source_target_column_original\":\"\",\"target_column\":\"\",\"target_names\":null}" {
        Ok(format!("dataset:{dataset_hash}"))
    } else {
        Ok(format!("target:{}", sha256_bytes(&canonical)))
    }
}

pub(super) fn registry_source_family(
    sotarena_root: &Path,
    dataset_hash: &str,
) -> Result<Option<String>> {
    let meta_path = sotarena_metadata_path(sotarena_root, dataset_hash);
    if !meta_path.is_file() {
        return Ok(None);
    }
    let value: Value = serde_json::from_slice(&fs::read(&meta_path)?)?;
    Ok(value
        .get("source_family")
        .and_then(Value::as_str)
        .or_else(|| {
            value
                .pointer("/acquisition/source_dataset_name")
                .and_then(Value::as_str)
        })
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string))
}

pub(super) fn selection_key(dataset_hash: &str) -> String {
    let mut hash = Sha256::new();
    hash.update(b"checkpoint-assisted-transfer-v2\0");
    hash.update(STUDY_SEED.to_string().as_bytes());
    hash.update(b"\0");
    hash.update(dataset_hash.as_bytes());
    hex::encode(hash.finalize())
}

pub(super) fn metrics(truth: &[f64], predictions: &[f64]) -> Result<(f64, f64, f64)> {
    if truth.is_empty()
        || truth.len() != predictions.len()
        || truth
            .iter()
            .chain(predictions)
            .any(|value| !value.is_finite())
    {
        bail!("metrics require equal, non-empty, finite inputs");
    }
    let n = truth.len() as f64;
    let mean = truth.iter().sum::<f64>() / n;
    let mut squared_error = 0.0;
    let mut absolute_error = 0.0;
    let mut total_sum_squares = 0.0;
    for (&actual, &predicted) in truth.iter().zip(predictions) {
        squared_error += (actual - predicted).powi(2);
        absolute_error += (actual - predicted).abs();
        total_sum_squares += (actual - mean).powi(2);
    }
    let rmse = (squared_error / n).sqrt();
    let r2 = if total_sum_squares > 1e-24 {
        1.0 - squared_error / total_sum_squares
    } else {
        bail!("metrics target variance is zero");
    };
    let mae = absolute_error / n;
    if [rmse, r2, mae].iter().any(|value| !value.is_finite()) {
        bail!("metrics are non-finite");
    }
    Ok((rmse, r2, mae))
}

pub(crate) fn paired_robust_fitness(split_r2: &[f64]) -> Result<f64> {
    if split_r2.is_empty() || split_r2.iter().any(|value| !value.is_finite()) {
        bail!("paired robust fitness needs finite split scores");
    }
    let clipped = split_r2
        .iter()
        .map(|value| value.max(-1.0))
        .collect::<Vec<_>>();
    let mean = clipped.iter().sum::<f64>() / clipped.len() as f64;
    let population_sd = (clipped
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / clipped.len() as f64)
        .sqrt();
    Ok(mean - 0.25 * population_sd.min(1.0))
}

pub(super) fn rank_scored(left: &ScoredMember, right: &ScoredMember) -> Ordering {
    match (left.robust_fitness, right.robust_fitness) {
        (Some(left_fitness), Some(right_fitness)) => right_fitness
            .total_cmp(&left_fitness)
            .then_with(|| {
                right
                    .worst_split_r2
                    .unwrap()
                    .total_cmp(&left.worst_split_r2.unwrap())
            })
            .then_with(|| left.member.complexity.cmp(&right.member.complexity))
            .then_with(|| {
                left.member
                    .semantic_sha256
                    .cmp(&right.member.semantic_sha256)
            }),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => left
            .member
            .complexity
            .cmp(&right.member.complexity)
            .then_with(|| {
                left.member
                    .semantic_sha256
                    .cmp(&right.member.semantic_sha256)
            }),
    }
}
