use super::*;

pub(crate) fn reject_v1_quarantine_window(encodings_root: &Path) -> Result<()> {
    let quarantine_root = encodings_root.join("incident-quarantine");
    if !quarantine_root.exists() {
        return Ok(());
    }
    let index_path = quarantine_root.join("index.v1.json");
    let index: V1EncodingQuarantineV1 = serde_json::from_reader(std::io::BufReader::new(
        std::fs::File::open(&index_path).with_context(|| {
            format!(
                "V1 incident quarantine exists without readable frozen index {}",
                index_path.display()
            )
        })?,
    ))?;
    index.validate()?;
    bail!("legacy expansion driver is disabled throughout the Encoder V2 quarantine window")
}

pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub(crate) fn sh(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program).args(args).output()?;
    if !output.status.success() {
        bail!(
            "{program} {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr)
                .lines()
                .last()
                .unwrap_or("")
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

pub(crate) fn temporarily_lacks_launch_capacity(message: &str) -> bool {
    message.contains("dependency-unsatisfied")
        && (message.contains("no eligible launch-ready node currently")
            || message.contains("eligible_node launch-ready-capacity")
            // The capability smoke pins encoding to xbabe3. If only another
            // node is launch-ready, its expected test(1) is a placement
            // backoff rather than a missing encoder dependency.
            || (message.contains("smoke xbabe3 prime encode environment")
                && message.contains("exit 1")))
}

/// A capacity-only dependency-check failure must not strand breadth encoding.
/// Retry submission in case capacity opened between check and submit; permanent
/// dependency failures still fail closed.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct EncodeChunkSubmission {
    pub(crate) job_id: String,
    pub(crate) disposition: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) backoff_reason: Option<String>,
}

pub(crate) fn pending_breadth_admission(root: &Path) -> Result<Option<EncodeChunkSubmission>> {
    let path = root.join("ops/gpu-lane-admission/requests/breadth_encoder.json");
    if !path.is_file() {
        return Ok(None);
    }
    let value = read_json(&path)?;
    let job_id = value
        .get("immutable_job_identity")
        .and_then(Value::as_str)
        .context("breadth admission request lacks immutable job identity")?;
    Ok(Some(EncodeChunkSubmission {
        job_id: job_id.to_owned(),
        disposition: "gpu_admission_pending",
        backoff_reason: value
            .get("last_capacity_backoff")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| Some("waiting for xbabe3 to expose four core credits".to_owned())),
    }))
}

pub(crate) fn register_gpu_admission(spec: &Path, job_id: &str) -> Result<()> {
    let controller = bundled_binary("jope_gpu_lane_admission")?;
    let output = Command::new(controller)
        .args([
            "--register-role",
            "breadth-encoder",
            "--spec",
            spec.to_string_lossy().as_ref(),
            "--job-id",
            job_id,
        ])
        .output()
        .context("register breadth encoder GPU admission request")?;
    if !output.status.success() {
        bail!(
            "GPU admission request failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

pub(crate) fn scq_check_submit(spec: &Path, job_id: &str) -> Result<EncodeChunkSubmission> {
    let spec_text = spec.to_string_lossy();
    let check = Command::new(scq_str())
        .args(["check", spec_text.as_ref()])
        .output()
        .with_context(|| {
            format!(
                "run {scq_str} check {}",
                spec.display(),
                scq_str = scq_str()
            )
        })?;
    let check_message = String::from_utf8_lossy(if check.stderr.is_empty() {
        &check.stdout
    } else {
        &check.stderr
    });
    if !check.status.success() && temporarily_lacks_launch_capacity(&check_message) {
        let reason = format!("capacity backoff: {}", check_message.trim());
        register_gpu_admission(spec, job_id)?;
        return Ok(EncodeChunkSubmission {
            job_id: job_id.to_owned(),
            disposition: "gpu_admission_requested",
            backoff_reason: Some(reason),
        });
    }
    if !check.status.success() {
        bail!(
            "SCQ check rejected encoding chunk: {}",
            check_message.trim()
        );
    }

    let submit = Command::new(scq_str())
        .args(["submit", spec_text.as_ref()])
        .output()
        .with_context(|| {
            format!(
                "run {scq_str} submit {}",
                spec.display(),
                scq_str = scq_str()
            )
        })?;
    let submit_message = String::from_utf8_lossy(if submit.stderr.is_empty() {
        &submit.stdout
    } else {
        &submit.stderr
    });
    if submit.status.success()
        || submit_message
            .to_ascii_lowercase()
            .contains("already exists")
    {
        return Ok(EncodeChunkSubmission {
            job_id: job_id.to_owned(),
            disposition: "submitted",
            backoff_reason: None,
        });
    }
    if temporarily_lacks_launch_capacity(&submit_message) {
        let reason = format!("capacity backoff: {}", submit_message.trim());
        register_gpu_admission(spec, job_id)?;
        return Ok(EncodeChunkSubmission {
            job_id: job_id.to_owned(),
            disposition: "gpu_admission_requested",
            backoff_reason: Some(reason),
        });
    }
    bail!(
        "SCQ submit rejected encoding chunk: {}",
        submit_message.trim()
    )
}

pub(crate) fn runtime_bundle_root() -> Result<PathBuf> {
    let executable = std::fs::canonicalize(std::env::current_exe()?)?;
    let bin = executable
        .parent()
        .context("runtime executable has no parent")?;
    let runtime = bin.parent().context("runtime bin has no parent")?;
    let bundle = runtime
        .parent()
        .context("runtime directory has no parent")?;
    let manifest = read_json(&bundle.join("bundle.json"))?;
    if manifest.get("schema").and_then(Value::as_str) != Some("RuntimeBundle.v3") {
        bail!(
            "expansion driver is not executing from a verified RuntimeBundle.v3: {}",
            executable.display()
        );
    }
    Ok(bundle.to_path_buf())
}

pub(crate) fn bundled_binary(name: &str) -> Result<PathBuf> {
    let path = runtime_bundle_root()?.join("runtime/bin").join(name);
    if !path.is_file() {
        bail!("runtime bundle lacks required binary {}", path.display());
    }
    Ok(path)
}

pub(crate) fn bundled_encoding_asset(name: &str) -> Result<PathBuf> {
    let path = runtime_bundle_root()?.join("pytorch/encoding").join(name);
    if !path.is_file() {
        bail!("runtime bundle lacks encoding asset {}", path.display());
    }
    Ok(path)
}

pub(crate) fn publish_immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.is_file() {
        if std::fs::read(path)? != bytes {
            bail!("immutable content drift: {}", path.display());
        }
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    use std::io::Write;
    file.write_all(bytes)?;
    file.sync_data()?;
    Ok(())
}

pub(crate) fn normalized_runtime_bundle_paths(body: &str) -> String {
    const MARKER: &str = "/runtime-bundles/";
    let mut parts = body.split(MARKER);
    let mut normalized = parts.next().unwrap_or_default().to_owned();
    for part in parts {
        normalized.push_str(MARKER);
        let component_end = part.find('/').unwrap_or(part.len());
        let component = &part[..component_end];
        if component == "current"
            || (component.len() == 64 && component.bytes().all(|byte| byte.is_ascii_hexdigit()))
        {
            normalized.push_str("<bundle>");
        } else {
            normalized.push_str(component);
        }
        normalized.push_str(&part[component_end..]);
    }
    normalized
}

pub(crate) fn publish_equivalent_bundle_spec(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.is_file() {
        let existing = std::fs::read_to_string(path)?;
        let proposed = std::str::from_utf8(bytes).context("rendered SCQ spec is not UTF-8")?;
        if normalized_runtime_bundle_paths(&existing) != normalized_runtime_bundle_paths(proposed) {
            bail!("immutable content drift: {}", path.display());
        }
        return Ok(());
    }
    publish_immutable(path, bytes)
}

pub(crate) fn append_spec_input(
    lines: &mut Vec<String>,
    name: &str,
    source: &Path,
    target: &str,
) -> Result<()> {
    let source = source
        .to_str()
        .context("encoding input path is not UTF-8")?;
    if [name, source, target]
        .iter()
        .any(|value| value.contains(['"', '\\', '\n', '\r']))
    {
        bail!("encoding spec input contains unsafe TOML text");
    }
    lines.extend([
        "[[input]]".to_owned(),
        format!("name = \"{name}\""),
        format!("source = \"{source}\""),
        format!("target = \"{target}\""),
        format!("sha256 = \"{}\"", sha256_file(Path::new(source))?),
        String::new(),
    ]);
    Ok(())
}

pub(crate) fn read_json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_str(
        &std::fs::read_to_string(path).with_context(|| path.display().to_string())?,
    )?)
}

pub(crate) fn file_age_secs(path: &Path) -> Option<u64> {
    let modified = path.metadata().ok()?.modified().ok()?;
    SystemTime::now()
        .duration_since(modified)
        .ok()
        .map(|d| d.as_secs())
}

// ---------------------------------------------------------------------------
// Guardrails
// ---------------------------------------------------------------------------
