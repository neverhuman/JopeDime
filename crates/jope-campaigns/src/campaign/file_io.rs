use super::*;

pub(super) fn genome_complexity(genome: &GenomeIr) -> usize {
    match genome {
        GenomeIr::Pipeline(pipeline) => pipeline.stages.len(),
        _ => 0,
    }
}

pub(super) fn genome_complexity_text(text: &str) -> usize {
    GenomeIr::parse(text).map_or(0, |genome| genome_complexity(&genome))
}

pub(super) fn head_family(genome: &GenomeIr) -> String {
    match genome {
        GenomeIr::Pipeline(pipeline) => pipeline
            .stages
            .last()
            .map_or_else(|| "ridge(default)".to_string(), |stage| stage.op.clone()),
        _ => "non_pipeline".to_string(),
    }
}

pub(super) fn head_family_text(text: &str) -> String {
    GenomeIr::parse(text).map_or_else(|_| "parse_error".to_string(), |genome| head_family(&genome))
}

pub(super) fn independent_metrics(truth: &[f64], predictions: &[f64]) -> Result<(f64, f64, f64)> {
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
        0.0
    };
    let mae = absolute_error / n;
    if !rmse.is_finite() || !r2.is_finite() || !mae.is_finite() {
        bail!("metrics are non-finite");
    }
    Ok((rmse, r2, mae))
}

pub(super) fn write_predictions_atomic(path: &Path, predictions: &[f64]) -> Result<()> {
    let parent = path.parent().context("prediction path has no parent")?;
    fs::create_dir_all(parent)?;
    let staged = parent.join(format!(".predictions.{}.stage", std::process::id()));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&staged)?;
    for prediction in predictions {
        writeln!(file, "{prediction:.17}")?;
    }
    file.flush()?;
    file.sync_all()?;
    if path.exists() {
        bail!("immutable prediction target already exists");
    }
    fs::rename(&staged, path)?;
    sync_directory(parent)
}

pub(super) fn read_predictions(path: &Path) -> Result<Vec<f64>> {
    let file = File::open(path)?;
    let mut predictions = Vec::new();
    for (line_number, line) in BufReader::new(file).lines().enumerate() {
        let value = line?
            .trim()
            .parse::<f64>()
            .with_context(|| format!("prediction line {} is invalid", line_number + 1))?;
        predictions.push(value);
    }
    Ok(predictions)
}

pub(super) fn require_fresh_artifact_root(path: &Path) -> Result<()> {
    if path.exists() {
        let mut entries = fs::read_dir(path)
            .with_context(|| format!("inspect exact-CV root {}", path.display()))?;
        if entries.next().transpose()?.is_some() {
            bail!("--exact-cv-dir must be absent or empty; campaign outputs are immutable");
        }
    }
    Ok(())
}

pub(super) fn immutable_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("immutable path has no parent")?;
    fs::create_dir_all(parent)?;
    if path.exists() {
        if fs::read(path)? == bytes {
            make_readonly(path)?;
            return Ok(());
        }
        bail!("immutable artifact collision at {}", path.display());
    }
    let staged = parent.join(format!(
        ".immutable-{}-{}.stage",
        std::process::id(),
        &sha256_bytes(bytes)[..16]
    ));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&staged)?;
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_all()?;
    match fs::hard_link(&staged, path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            if fs::read(path)? != bytes {
                let _ = fs::remove_file(&staged);
                bail!("immutable artifact collision at {}", path.display());
            }
        }
        Err(error) => {
            let _ = fs::remove_file(&staged);
            return Err(error).context("publish immutable artifact");
        }
    }
    fs::remove_file(&staged)?;
    make_readonly(path)?;
    sync_directory(parent)
}

pub(super) fn make_readonly(path: &Path) -> Result<()> {
    let mut permissions = fs::metadata(path)
        .with_context(|| format!("inspect immutable artifact {}", path.display()))?
        .permissions();
    permissions.set_readonly(true);
    fs::set_permissions(path, permissions)
        .with_context(|| format!("seal immutable artifact {}", path.display()))
}

pub(super) fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

pub(super) fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub(super) fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(super) fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("open {} for SHA-256", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

pub(super) fn validate_file_hash(path: &Path, expected: &str, label: &str) -> Result<()> {
    if sha256_file(path).with_context(|| format!("hash pinned {label}"))? != expected {
        bail!("{label} differs from pinned SHA-256 {expected}");
    }
    Ok(())
}

pub(super) fn validate_sha256(label: &str, value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{label} must be 64 lowercase hexadecimal characters");
    }
    Ok(())
}

pub(super) fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non-string panic")
}
