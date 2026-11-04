use super::*;

pub(crate) fn load_status(root: &Path) -> Result<Value> {
    let path = root.join(STATUS_RELATIVE);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(json!({})),
        Err(error) => {
            return Err(error).with_context(|| format!("read status receipt {}", path.display()))
        }
    };
    serde_json::from_slice(&bytes)
        .with_context(|| format!("parse status receipt {}", path.display()))
}

/// A target that fails today must still report when it last worked, otherwise
/// "no successes ever" and "no successes for six hours" look identical.
pub(crate) fn previous_success(previous: &Value, label: &str) -> u64 {
    previous
        .get("targets")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
        .iter()
        .find(|t| t.get("target").and_then(Value::as_str) == Some(label))
        .and_then(|t| t.get("last_success_ms").and_then(Value::as_u64))
        .unwrap_or(0)
}

pub(crate) fn write_status(root: &Path, report: &Value) -> Result<()> {
    let path = root.join(STATUS_RELATIVE);
    th::write_atomic(&path, &serde_json::to_vec_pretty(report)?)?;
    Ok(())
}

pub(crate) fn append_event(root: &Path, record: &Value) {
    let dir = replica_dir(root);
    if fs::create_dir_all(&dir).is_err() {
        return;
    }
    if let Ok(mut file) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("replica-events.v1.jsonl"))
    {
        let _ = writeln!(file, "{record}");
    }
}
