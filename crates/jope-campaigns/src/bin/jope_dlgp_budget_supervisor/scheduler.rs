use super::*;

pub(crate) fn load_plans(
    root: &Path,
    protocol_sha256: &str,
) -> Result<BTreeMap<String, TrajectoryPlan>> {
    let directory = root.join("plans");
    if !directory.is_dir() {
        return Ok(BTreeMap::new());
    }
    let mut plans = BTreeMap::new();
    for entry in fs::read_dir(directory)?.filter_map(|entry| entry.ok()) {
        if entry.path().extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let plan: TrajectoryPlan = serde_json::from_value(read_json(&entry.path())?)?;
        plan.validate(protocol_sha256)?;
        ensure!(
            plans
                .insert(plan.trajectory_identity.clone(), plan)
                .is_none(),
            "duplicate trajectory plan identity"
        );
    }
    Ok(plans)
}

pub(crate) fn load_progress(
    root: &Path,
    plans: &BTreeMap<String, TrajectoryPlan>,
    protocol_sha256: &str,
) -> Result<BTreeMap<String, TrajectoryProgress>> {
    let directory = root.join("receipts");
    if !directory.is_dir() {
        return Ok(BTreeMap::new());
    }
    let mut progress: BTreeMap<String, TrajectoryProgress> = BTreeMap::new();
    for entry in fs::read_dir(directory)?.filter_map(|entry| entry.ok()) {
        if entry.path().extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let value: TrajectoryProgress = serde_json::from_value(read_json(&entry.path())?)?;
        let plan = plans
            .get(&value.trajectory_identity)
            .context("trajectory receipt has no frozen plan")?;
        value.validate(plan, protocol_sha256)?;
        match progress.get(&value.trajectory_identity) {
            Some(existing) if existing.updated_at_ms == value.updated_at_ms => {
                ensure!(
                    serde_json::to_vec(existing)? == serde_json::to_vec(&value)?,
                    "duplicate trajectory completion conflicts"
                );
            }
            Some(existing) if existing.updated_at_ms > value.updated_at_ms => {}
            _ => {
                progress.insert(value.trajectory_identity.clone(), value);
            }
        }
    }
    Ok(progress)
}

pub(crate) fn scheduler_jobs(value: &Value) -> Vec<SchedulerJob> {
    value
        .pointer("/Jobs/jobs")
        .or_else(|| value.get("jobs"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|job| serde_json::from_value(job.clone()).ok())
        .collect()
}

pub(crate) fn launch_receipt_path(
    campaign_root: &Path,
    trajectory_identity: &str,
    attempt_number: u64,
) -> PathBuf {
    campaign_root
        .join("launches")
        .join(trajectory_identity)
        .join(format!("{attempt_number:08}.json"))
}

pub(crate) fn read_launch_receipt(
    campaign_root: &Path,
    trajectory_identity: &str,
    attempt_number: u64,
) -> Result<Option<CampaignLaunchReceipt>> {
    let path = launch_receipt_path(campaign_root, trajectory_identity, attempt_number);
    if !path.is_file() {
        return Ok(None);
    }
    let receipt: CampaignLaunchReceipt = serde_json::from_value(read_json(&path)?)?;
    receipt.validate()?;
    ensure!(
        receipt.trajectory_identity == trajectory_identity
            && receipt.attempt_number == attempt_number,
        "campaign launch receipt path/identity drift"
    );
    Ok(Some(receipt))
}

pub(crate) fn materialize_attempt_spec(
    campaign_root: &Path,
    plan: &TrajectoryPlan,
    attempt_number: u64,
    job_identity: &str,
) -> Result<PathBuf> {
    if attempt_number == 1 {
        ensure!(
            job_identity == plan.job_identity,
            "initial campaign attempt identity drift"
        );
        return Ok(plan.spec_path.clone());
    }
    let source = fs::read_to_string(&plan.spec_path)?;
    let mut in_job = false;
    let mut replacements = 0_u64;
    let mut rendered = String::with_capacity(source.len() + 32);
    for line in source.split_inclusive('\n') {
        let newline = if line.ends_with('\n') { "\n" } else { "" };
        let body = line.strip_suffix('\n').unwrap_or(line);
        let trimmed = body.trim();
        if trimmed.starts_with('[') {
            in_job = trimmed == "[job]";
        }
        if in_job {
            if let Some((key, value)) = trimmed.split_once('=') {
                if key.trim() == "id" {
                    ensure!(
                        value.trim() == serde_json::to_string(&plan.job_identity)?,
                        "frozen campaign spec original job identity drift"
                    );
                    let indentation = &body[..body.len().saturating_sub(body.trim_start().len())];
                    rendered.push_str(indentation);
                    rendered.push_str("id = ");
                    rendered.push_str(&serde_json::to_string(job_identity)?);
                    rendered.push_str(newline);
                    replacements += 1;
                    continue;
                }
            }
        }
        rendered.push_str(line);
    }
    ensure!(
        replacements == 1,
        "campaign spec must contain exactly one [job].id"
    );
    let path = campaign_root
        .join("scq-specs/recovery")
        .join(format!("{job_identity}.toml"));
    if path.is_file() {
        ensure!(
            fs::read_to_string(&path)? == rendered,
            "campaign retry spec replay differs"
        );
        return Ok(path);
    }
    fs::create_dir_all(path.parent().context("campaign retry spec parent")?)?;
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, rendered.as_bytes())?;
    fs::rename(&temporary, &path)?;
    Ok(path)
}

pub(crate) fn submit_attempt(
    args: &Args,
    campaign_root: &Path,
    plan: &TrajectoryPlan,
    payload_sha256: &str,
    attempt_number: u64,
    job_identity: &str,
) -> Result<CampaignLaunchReceipt> {
    if let Some(existing) =
        read_launch_receipt(campaign_root, &plan.trajectory_identity, attempt_number)?
    {
        ensure!(
            existing.scientific_payload_sha256 == payload_sha256
                && existing.job_identity == job_identity,
            "campaign launch replay changed payload or job identity"
        );
        return Ok(existing);
    }
    let spec = materialize_attempt_spec(campaign_root, plan, attempt_number, job_identity)?;
    let check = Command::new(&args.scq).arg("check").arg(&spec).output()?;
    ensure!(
        check.status.success(),
        "scq check failed for frozen trajectory {}: {}",
        plan.trajectory_identity,
        String::from_utf8_lossy(&check.stderr).trim()
    );
    let submit = Command::new(&args.scq).arg("submit").arg(&spec).output()?;
    ensure!(
        submit.status.success()
            || String::from_utf8_lossy(&submit.stderr).contains("already exists"),
        "submit failed for frozen trajectory {}: {}",
        plan.trajectory_identity,
        String::from_utf8_lossy(&submit.stderr).trim()
    );
    let receipt = CampaignLaunchReceipt::new(
        payload_sha256.to_owned(),
        plan.trajectory_identity.clone(),
        attempt_number,
        job_identity.to_owned(),
        sha256_file(&spec)?,
        now_ms(),
    )?;
    atomic_json(
        &launch_receipt_path(campaign_root, &plan.trajectory_identity, attempt_number),
        &serde_json::to_value(&receipt)?,
    )?;
    Ok(receipt)
}

pub(crate) fn terminal_class(state: &str) -> Option<DimeTerminalClassV1> {
    match state {
        "failed" | "cancelled" | "preempted" | "preempting" | "lost" | "vanished" => {
            Some(DimeTerminalClassV1::InfrastructureFailure)
        }
        "finished" | "completed" => Some(DimeTerminalClassV1::Completed),
        _ => None,
    }
}

pub(crate) fn next_attempt_identity(plan: &TrajectoryPlan, attempt_number: u64) -> String {
    format!("{}-infra{attempt_number:06}", plan.job_identity)
}
