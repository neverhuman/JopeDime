use super::*;

pub(super) fn runtime_bundle() -> Result<PathBuf> {
    let executable = fs::canonicalize(std::env::current_exe()?)?;
    executable
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .context("canary executor is not running from a runtime bundle")
}

pub(super) fn toml_quote(value: &str) -> Result<String> {
    ensure!(
        !value.contains(['"', '\\', '\n', '\r']),
        "unsafe generated TOML value"
    );
    Ok(format!("\"{value}\""))
}

pub(super) fn append_input(
    lines: &mut Vec<String>,
    name: &str,
    source: &Path,
    target: &str,
) -> Result<()> {
    lines.extend([
        "[[input]]".to_owned(),
        format!("name = {}", toml_quote(name)?),
        format!("source = {}", toml_quote(&source.to_string_lossy())?),
        format!("target = {}", toml_quote(target)?),
        format!("sha256 = {}", toml_quote(&sha256_file(source)?)?),
        String::new(),
    ]);
    Ok(())
}

pub(super) fn write_immutable(path: &Path, body: &[u8]) -> Result<()> {
    if path.is_file() {
        ensure!(fs::read(path)? == body, "immutable canary artifact drift");
        return Ok(());
    }
    fs::create_dir_all(path.parent().context("immutable path has no parent")?)?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(path)?;
    file.write_all(body)?;
    file.sync_data()?;
    Ok(())
}

pub(super) fn canary_lane_ancestry(mode: CanaryMode, lane: &str) -> &'static str {
    match lane {
        "gp" => "none",
        "candidate" if mode == CanaryMode::Shadow => "shadow_dl",
        _ => "accepted_dl",
    }
}

pub(super) fn scq_jobs(args: &Args) -> Result<BTreeMap<String, String>> {
    let output = Command::new(&args.scq).args(["--json", "ls"]).output()?;
    ensure!(output.status.success(), "SCQ list failed");
    let value: Value = serde_json::from_slice(&output.stdout)?;
    Ok(value
        .pointer("/Jobs/jobs")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|row| {
            Some((
                row.get("id")?.as_str()?.to_owned(),
                row.get("state")?.as_str()?.to_owned(),
            ))
        })
        .collect())
}

pub(super) fn submit_spec(args: &Args, spec: &Path) -> Result<()> {
    let spec_text = spec.to_string_lossy();
    let check = Command::new(&args.scq)
        .args(["check", spec_text.as_ref()])
        .output()?;
    ensure!(
        check.status.success(),
        "SCQ check failed: {}",
        String::from_utf8_lossy(&check.stderr)
    );
    let submit = Command::new(&args.scq)
        .args(["submit", spec_text.as_ref()])
        .output()?;
    let message = String::from_utf8_lossy(if submit.stderr.is_empty() {
        &submit.stdout
    } else {
        &submit.stderr
    });
    ensure!(
        submit.status.success() || message.to_ascii_lowercase().contains("already exists"),
        "SCQ submit failed: {}",
        message.trim()
    );
    Ok(())
}

pub(super) fn recursive_find(path: &Path, name: &str, output: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            recursive_find(&path, name, output);
        } else if path.file_name().and_then(|value| value.to_str()) == Some(name) {
            output.push(path);
        }
    }
}

pub(super) fn fetch(args: &Args, job_id: &str, output: &Path) -> Result<()> {
    fs::create_dir_all(output)?;
    let result = Command::new(&args.scq)
        .args([
            "fetch",
            job_id,
            "--output",
            output.to_string_lossy().as_ref(),
        ])
        .output()?;
    ensure!(
        result.status.success(),
        "SCQ fetch failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(())
}

pub(super) fn proposal_job_id(intent_hash: &str, worker_sha256: &str) -> String {
    format!(
        "jope-canary-propose-{}-{}",
        &intent_hash[..12],
        &worker_sha256[..8]
    )
}
