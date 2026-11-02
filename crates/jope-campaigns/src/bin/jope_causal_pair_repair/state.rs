use super::*;

pub(super) fn lane_name(lane: CausalLane) -> &'static str {
    match lane {
        CausalLane::NativeDlDirect => "native_dl_direct",
        CausalLane::MatchedGpComparator => "matched_gp_comparator",
    }
}

pub(super) fn entry_key(pair_id: &Sha256Digest, lane: CausalLane) -> String {
    format!("{}:{}", pair_id, lane_name(lane))
}

pub(super) fn load_queue(root: &Path) -> Result<RepairQueue> {
    let queue: RepairQueue = serde_json::from_value(read_json(
        &root.join("ops/causal-pair-repair-queue.v1.json"),
    )?)?;
    ensure!(
        queue.schema == QUEUE_SCHEMA,
        "pair repair queue schema drift"
    );
    Ok(queue)
}

pub(super) fn load_state(root: &Path) -> Result<RepairState> {
    let path = root.join("ops/causal-pair-repair-state.v1.json");
    if !path.is_file() {
        return Ok(RepairState::default());
    }
    let state: RepairState = serde_json::from_value(read_json(&path)?)?;
    ensure!(
        state.schema == STATE_SCHEMA,
        "pair repair state schema drift"
    );
    Ok(state)
}

pub(super) fn save_state(root: &Path, state: &mut RepairState) -> Result<()> {
    state.updated_at_ms = now_ms();
    atomic_json(
        &root.join("ops/causal-pair-repair-state.v1.json"),
        &serde_json::to_value(state)?,
    )
}

pub(super) fn scq_jobs(args: &Args) -> Result<BTreeMap<String, String>> {
    let output = Command::new(&args.scq).args(["--json", "ls"]).output()?;
    ensure!(
        output.status.success(),
        "SCQ list failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
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
    let spec = spec.to_string_lossy();
    let check = Command::new(&args.scq)
        .args(["check", spec.as_ref()])
        .output()?;
    ensure!(
        check.status.success(),
        "SCQ check failed: {}",
        String::from_utf8_lossy(&check.stderr)
    );
    let submit = Command::new(&args.scq)
        .args(["submit", spec.as_ref()])
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
