use super::*;

pub(super) fn load_state(path: &Path) -> Result<DurableState> {
    if !path.is_file() {
        return Ok(DurableState::new());
    }
    let state: DurableState = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(
        state.schema == "JopeContinuousTrainerWatchdogState.v1",
        "unsupported trainer watchdog state schema"
    );
    ensure!(state.next_sequence > 0, "trainer sequence must be positive");
    Ok(state)
}

pub(super) fn save_state(path: &Path, state: &DurableState) -> Result<()> {
    atomic_json(path, &serde_json::to_value(state)?)
}

pub(super) fn load_label_watermark(path: &Path) -> Result<u64> {
    let value = read_json(path)?;
    let labels = value
        .get("exact_rows_total")
        .or_else(|| value.get("watermark"))
        .and_then(Value::as_u64)
        .with_context(|| format!("{} lacks a nonnegative label watermark", path.display()))?;
    ensure!(labels > 0, "label watermark is zero");
    Ok(labels)
}

pub(super) fn parse_jobs(value: &Value) -> Result<Vec<Value>> {
    value
        .get("Jobs")
        .and_then(|value| value.get("jobs"))
        .and_then(Value::as_array)
        .cloned()
        .context("SCQ ls JSON lacks Jobs.jobs")
}

pub(super) fn parse_priority(value: &Value) -> Result<f64> {
    value
        .get("FamilyPriorities")
        .and_then(|value| value.get("families"))
        .and_then(Value::as_array)
        .and_then(|rows| {
            rows.iter()
                .find(|row| row.get("family").and_then(Value::as_str) == Some(PROJECT))
        })
        .and_then(|row| row.get("effective_weight"))
        .and_then(Value::as_f64)
        .context("SCQ priority inventory lacks jope-selfplay-train")
}

pub(super) fn priority_is_ready(value: f64) -> bool {
    value.is_finite() && (value - REQUIRED_PRIORITY).abs() < 1e-9
}

pub(super) fn parse_trainer_job(value: &Value) -> Option<Result<TrainerJob>> {
    let id = value.get("id").and_then(Value::as_str)?;
    if !id.starts_with("jope-selfplay-train-") && !id.starts_with("jope-testonly-train-") {
        return None;
    }
    let parsed = parse_trainer_id(id);
    Some((|| {
        let (watermark, sequence) =
            parsed.with_context(|| format!("malformed trainer job id {id}"))?;
        Ok(TrainerJob {
            id: id.to_owned(),
            watermark,
            sequence,
            state: value
                .get("state")
                .and_then(Value::as_str)
                .context("trainer job lacks state")?
                .to_owned(),
            project: value
                .get("project")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            tier: value
                .get("tier")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            node: value.get("node").and_then(Value::as_str).map(str::to_owned),
            gpus: value.get("gpus").and_then(Value::as_u64).unwrap_or(0),
        })
    })())
}

pub(super) fn active_trainers(trainers: &[TrainerJob]) -> Result<Vec<&TrainerJob>> {
    let active = trainers
        .iter()
        .filter(|job| LIVE_STATES.contains(&job.state.as_str()))
        .collect::<Vec<_>>();
    ensure!(
        active.len() <= 1,
        "duplicate active trainers: {}",
        active
            .iter()
            .map(|job| job.id.as_str())
            .collect::<Vec<_>>()
            .join(",")
    );
    for job in &active {
        validate_active_trainer(job)?;
    }
    Ok(active)
}

pub(super) fn parse_trainer_id(id: &str) -> Option<(u64, u64)> {
    let suffix = id
        .strip_prefix("jope-selfplay-train-")
        .or_else(|| id.strip_prefix("jope-testonly-train-"))?;
    let mut fields = suffix.split('-');
    let watermark = fields.next()?.parse().ok()?;
    let sequence = fields
        .find_map(|field| field.strip_prefix('r').and_then(|row| row.parse().ok()))
        .unwrap_or(0);
    Some((watermark, sequence))
}

pub(super) fn validate_active_trainer(job: &TrainerJob) -> Result<()> {
    ensure!(
        matches!(job.project.as_str(), PROJECT | "jope-testonly"),
        "active trainer {} has unexpected project {}",
        job.id,
        job.project
    );
    ensure!(
        job.tier == "normal",
        "active trainer {} has unexpected tier {}",
        job.id,
        job.tier
    );
    ensure!(
        job.gpus == 1,
        "active trainer {} does not reserve exactly one GPU",
        job.id
    );
    if matches!(job.state.as_str(), "running" | "dispatching") {
        ensure!(
            job.node.as_deref() == Some("xbabe2"),
            "active trainer {} is not on xbabe2",
            job.id
        );
    }
    Ok(())
}
