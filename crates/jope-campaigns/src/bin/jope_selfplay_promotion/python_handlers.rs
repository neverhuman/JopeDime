use super::*;

// ---------------------------------------------------------------------------
// Python delegation shim
// ---------------------------------------------------------------------------

pub(super) fn python_handler(name: &str, request: &Value, root: &Path) -> Result<Value> {
    let work = root.join("ops");
    fs::create_dir_all(&work)?;
    let request_path = work.join(format!(
        ".controller-shim-{}-{}.json",
        name,
        std::process::id()
    ));
    fs::write(&request_path, py_dumps(request, PyStyle::Default)?)?;
    let output = Command::new("/usr/bin/python3")
        .args(["-c", py_shim(), name, &request_path.to_string_lossy()])
        .current_dir(scripts_str())
        .env(
            "PYTHONPATH",
            format!(
                "{prime_src_str}:{scripts_str}",
                scripts_str = scripts_str(),
                prime_src_str = prime_src_str()
            ),
        )
        .output()
        .context("spawn python shim")?;
    let _ = fs::remove_file(&request_path);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!(
            "python handler {name} failed: {}",
            stderr
                .lines()
                .rev()
                .take(12)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join(" | ")
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let value: Value = serde_json::from_str(stdout.trim())
        .with_context(|| format!("python handler {name} returned invalid JSON"))?;
    if let Some(detail) = value.get("__pending__").and_then(Value::as_str) {
        return Err(pending(detail));
    }
    Ok(value)
}

pub(super) fn milestone_status(checkpoint_sha256: &str) -> Result<Value> {
    let output = Command::new("/usr/bin/python3")
        .args(["-c", py_shim(), "milestone_status", checkpoint_sha256])
        .current_dir(scripts_str())
        .env(
            "PYTHONPATH",
            format!(
                "{prime_src_str}:{scripts_str}",
                scripts_str = scripts_str(),
                prime_src_str = prime_src_str()
            ),
        )
        .output()
        .context("spawn milestone shim")?;
    if !output.status.success() {
        bail!(
            "milestone_status failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(serde_json::from_str(
        String::from_utf8_lossy(&output.stdout).trim(),
    )?)
}
