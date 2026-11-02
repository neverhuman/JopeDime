use super::*;

pub(crate) fn ready_waves(
    waves_root: &Path,
    checkpoint_sha256: &str,
    checkpoint_epoch: &str,
    claimed: &BTreeSet<String>,
    limit: usize,
) -> Result<Vec<ReadyWave>> {
    if !waves_root.is_dir() {
        return Ok(Vec::new());
    }
    let mut directories = fs::read_dir(waves_root)?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.file_name().to_str()?.to_owned();
            valid_wave_id(&name).then_some((name, entry.path()))
        })
        .collect::<Vec<_>>();
    directories.sort_by(|left, right| right.0.cmp(&left.0));
    let mut waves = Vec::new();
    for (wave, directory) in directories {
        if waves.len() == limit {
            break;
        }
        let Some(ready) = validate_ready_wave(
            &wave,
            &directory,
            checkpoint_sha256,
            checkpoint_epoch,
            claimed,
        )?
        else {
            continue;
        };
        if ready.unclaimed_manifests > 0 {
            waves.push(ready);
        }
    }
    waves.sort_by(|left, right| left.wave.cmp(&right.wave));
    Ok(waves)
}

pub(crate) fn validate_ready_wave(
    wave: &str,
    directory: &Path,
    checkpoint_sha256: &str,
    checkpoint_epoch: &str,
    claimed: &BTreeSet<String>,
) -> Result<Option<ReadyWave>> {
    let summary_path = directory.join("summary.json");
    if !summary_path.is_file() {
        return Ok(None);
    }
    let summary: Value = serde_json::from_slice(&fs::read(&summary_path)?)
        .with_context(|| format!("parse {}", summary_path.display()))?;
    if summary.get("checkpoint_sha256").and_then(Value::as_str) != Some(checkpoint_sha256) {
        return Ok(None);
    }
    let realtime_schema = summary.get("schema").and_then(Value::as_str) == Some(WAVE_SCHEMA);
    let micro_wave_kind = summary.get("wave_kind").and_then(Value::as_str) == Some(MICRO_WAVE_KIND);
    if !realtime_schema && !micro_wave_kind {
        return Ok(None);
    }
    if !realtime_schema
        || !micro_wave_kind
        || summary.get("immutable").and_then(Value::as_bool) != Some(true)
        || summary.get("wave").and_then(Value::as_str) != Some(wave)
        || summary.get("checkpoint_epoch").and_then(Value::as_str) != Some(checkpoint_epoch)
    {
        bail!("{wave}: current-checkpoint immutable micro-wave summary invariants failed");
    }
    let entries = summary
        .get("manifests")
        .and_then(Value::as_array)
        .context("wave summary lacks manifests")?;
    if summary.get("manifest_count").and_then(Value::as_u64) != Some(entries.len() as u64) {
        bail!("{wave}: manifest count differs from immutable summary");
    }
    let mut hashes = BTreeSet::new();
    let mut paths = BTreeSet::new();
    let mut unclaimed = 0;
    for entry in entries {
        let digest = entry
            .get("sha256")
            .and_then(Value::as_str)
            .context("wave manifest receipt lacks SHA-256")?;
        validate_sha256(digest)?;
        let relative = entry
            .get("relative_path")
            .and_then(Value::as_str)
            .context("wave manifest receipt lacks relative path")?;
        let relative_path = Path::new(relative);
        if relative_path.is_absolute()
            || relative_path
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
            || relative_path.parent() != Some(Path::new("manifests"))
        {
            bail!("{wave}: manifest path escapes its immutable wave");
        }
        if !hashes.insert(digest.to_owned()) || !paths.insert(relative.to_owned()) {
            bail!("{wave}: duplicate manifest receipt");
        }
        if !directory.join(relative_path).is_file() {
            bail!("{wave}: immutable manifest is absent");
        }
        if !claimed.contains(digest) {
            unclaimed += 1;
        }
    }
    Ok(Some(ReadyWave {
        wave: wave.to_owned(),
        summary_sha256: sha256_file(&summary_path)?,
        manifests: entries.len(),
        unclaimed_manifests: unclaimed,
    }))
}

pub(crate) fn manifests_per_wave(max_candidates_per_manifest: usize) -> usize {
    DATASETS_PER_WAVE * CANDIDATES_PER_DATASET.div_ceil(max_candidates_per_manifest)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn supply_decision(
    queue_depth: usize,
    candidate_evaluation_work: usize,
    protected_candidate_evaluations: usize,
    current_ready_waves: usize,
    complete_ready_waves: usize,
    queue_floor: usize,
    queued_target: usize,
    candidate_evaluation_work_target: usize,
    queued_hard_cap: usize,
    candidate_reserve_min: usize,
    min_complete_reserve_waves: usize,
    max_overlap_waves: usize,
    max_builds_per_cycle: usize,
    max_candidates_per_manifest: usize,
) -> SupplyDecision {
    let estimated_manifests_per_wave = manifests_per_wave(max_candidates_per_manifest);
    let candidate_work_deficit =
        candidate_evaluation_work_target.saturating_sub(candidate_evaluation_work);
    let reserve_deficit = candidate_reserve_min.saturating_sub(protected_candidate_evaluations);
    let desired_candidate_evaluations = candidate_work_deficit.max(reserve_deficit);
    let waves_for_deficit = desired_candidate_evaluations.div_ceil(CANDIDATE_EVALUATIONS_PER_WAVE);
    let waves_for_complete_reserve =
        min_complete_reserve_waves.saturating_sub(complete_ready_waves);
    let overlap_slots = max_overlap_waves.saturating_sub(current_ready_waves);
    let maximum_candidate_work = queued_hard_cap
        .saturating_mul(max_candidates_per_manifest)
        .saturating_add(max_overlap_waves.saturating_mul(CANDIDATE_EVALUATIONS_PER_WAVE));
    let target_feasible = candidate_evaluation_work_target <= maximum_candidate_work;
    SupplyDecision {
        below_floor: queue_depth < queue_floor,
        overloaded: queue_depth > queued_target,
        candidate_evaluation_work,
        candidate_evaluation_work_target,
        estimated_manifests_per_wave,
        target_feasible,
        builds_planned: waves_for_deficit
            .max(waves_for_complete_reserve)
            .min(overlap_slots)
            .min(max_builds_per_cycle)
            * usize::from(target_feasible),
    }
}

pub(crate) fn next_wave_id(waves_root: &Path) -> Result<String> {
    let mut maximum = 0_u64;
    if waves_root.is_dir() {
        for entry in fs::read_dir(waves_root)? {
            let entry = entry?;
            if let Some(name) = entry.file_name().to_str() {
                maximum = maximum.max(parse_wave_id(name).unwrap_or(0));
            }
        }
        let pending = waves_root.join("pending");
        if pending.is_dir() {
            for entry in fs::read_dir(pending)? {
                let entry = entry?;
                let path = entry.path();
                let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
                    continue;
                };
                maximum = maximum.max(
                    stem.strip_suffix(".cohort")
                        .and_then(parse_wave_id)
                        .unwrap_or(0),
                );
            }
        }
    }
    if maximum >= 999_999 {
        bail!("strict wave identity space exhausted");
    }
    Ok(format!("strict-wave-{:06}", maximum + 1))
}

pub(crate) fn wave_to_build(waves_root: &Path) -> Result<String> {
    let pending = waves_root.join("pending");
    let mut unresolved = Vec::new();
    if pending.is_dir() {
        for entry in fs::read_dir(&pending)? {
            let entry = entry?;
            let path = entry.path();
            let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
                continue;
            };
            let Some(wave) = stem.strip_suffix(".cohort") else {
                continue;
            };
            if valid_wave_id(wave)
                && !waves_root.join(wave).is_dir()
                && !pending_wave_quarantined(waves_root, wave, &path)?
            {
                unresolved.push(wave.to_owned());
            }
        }
    }
    unresolved.sort();
    unresolved.dedup();
    match unresolved.as_slice() {
        [] => next_wave_id(waves_root),
        [wave, ..] => Ok(wave.clone()),
    }
}

pub(crate) fn pending_wave_quarantined(
    waves_root: &Path,
    wave: &str,
    cohort: &Path,
) -> Result<bool> {
    let receipt_path = waves_root
        .join("quarantine")
        .join(format!("{wave}.receipt.json"));
    if !receipt_path.is_file() {
        return Ok(false);
    }
    let receipt: Value = serde_json::from_slice(&fs::read(&receipt_path)?)?;
    if receipt.get("schema").and_then(Value::as_str) != Some("CohortSupplyQuarantineReceipt.v1")
        || receipt.get("wave").and_then(Value::as_str) != Some(wave)
        || receipt.get("operator_authorized").and_then(Value::as_bool) != Some(true)
        || receipt
            .get("reason")
            .and_then(Value::as_str)
            .filter(|reason| !reason.is_empty())
            .is_none()
    {
        bail!("{wave}: invalid cohort-supply quarantine authority");
    }
    let cohort_sha256 = receipt
        .get("cohort_sha256")
        .and_then(Value::as_str)
        .context("cohort-supply quarantine lacks cohort hash")?;
    validate_sha256(cohort_sha256)?;
    if sha256_file(cohort)? != cohort_sha256 {
        bail!("{wave}: quarantined cohort content drift");
    }
    let state_sha256 = receipt
        .get("supply_state_sha256")
        .and_then(Value::as_str)
        .context("cohort-supply quarantine lacks supply-state hash")?;
    validate_sha256(state_sha256)?;
    let root = waves_root
        .parent()
        .context("waves root has no flywheel parent")?;
    let state = supply_state_path(root, wave);
    if !state.is_file() || sha256_file(&state)? != state_sha256 {
        bail!("{wave}: quarantined supply-state content drift");
    }
    Ok(true)
}

pub(crate) fn valid_wave_id(value: &str) -> bool {
    parse_wave_id(value).is_some()
}

pub(crate) fn parse_wave_id(value: &str) -> Option<u64> {
    let digits = value.strip_prefix("strict-wave-")?;
    if digits.len() != 6 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

pub(crate) fn supply_state_v1_path(root: &Path, wave: &str) -> PathBuf {
    root.join("ops/cohort-supply-state-v1")
        .join(format!("{wave}.json"))
}

pub(crate) fn supply_state_path(root: &Path, wave: &str) -> PathBuf {
    root.join("ops/cohort-supply-state-v2")
        .join(format!("{wave}.json"))
}

pub(crate) fn persist_supply_state(root: &Path, state: &PersistedSupplyState) -> Result<()> {
    if state.schema != SUPPLY_STATE_V2_SCHEMA {
        bail!("only CohortSupplyState.v2 may be persisted by the active supply lane");
    }
    state.validate(&state.wave, &state.checkpoint_sha256)?;
    atomic_json(
        &supply_state_path(root, &state.wave),
        &serde_json::to_value(state)?,
    )
}

pub(crate) fn load_supply_state(
    root: &Path,
    wave: &str,
    checkpoint: &str,
) -> Result<Option<PersistedSupplyState>> {
    let v2_path = supply_state_path(root, wave);
    if v2_path.is_file() {
        let state: PersistedSupplyState = serde_json::from_slice(&fs::read(&v2_path)?)?;
        state.validate(wave, checkpoint)?;
        if state.schema != SUPPLY_STATE_V2_SCHEMA {
            bail!("v2 cohort supply path contains a non-v2 document");
        }
        return Ok(Some(state));
    }
    let v1_path = supply_state_v1_path(root, wave);
    if !v1_path.is_file() {
        return Ok(None);
    }
    let mut state: PersistedSupplyState = serde_json::from_slice(&fs::read(&v1_path)?)?;
    state.validate(wave, checkpoint)?;
    if state.schema != SUPPLY_STATE_V1_SCHEMA {
        bail!("legacy cohort supply path contains an unexpected schema");
    }
    state.schema = SUPPLY_STATE_V2_SCHEMA.to_owned();
    let job_identity = state
        .proposal_job_id
        .as_deref()
        .context("legacy cohort supply state lacks proposer identity")?;
    let spec = root
        .join("specs/cohort-propose")
        .join(format!("{job_identity}.toml"));
    if spec.is_file() {
        state
            .runtime_pinned_attempts
            .push(runtime_pinned_intent_from_spec(
                &spec,
                state.proposal_attempt,
            )?);
    } else if state.stage.ordinal() >= PersistedSupplyStage::SubmitCohortProposer.ordinal() {
        bail!("submitted legacy cohort supply state lacks its immutable proposer spec");
    }
    state.revision = state.revision.saturating_add(1);
    persist_supply_state(root, &state)?;
    Ok(Some(state))
}

pub(crate) fn mark_keeper_acknowledged(root: &Path, wave: &str, checkpoint: &str) -> Result<()> {
    let Some(mut state) = load_supply_state(root, wave, checkpoint)? else {
        return Ok(());
    };
    if state.stage == PersistedSupplyStage::KeeperAcknowledgment {
        return Ok(());
    }
    if state.stage != PersistedSupplyStage::BuildWave {
        bail!("{wave}: keeper acknowledged an incomplete supply state");
    }
    state.advance(PersistedSupplyStage::KeeperAcknowledgment)?;
    persist_supply_state(root, &state)
}

pub(crate) fn toml_quote(value: &str) -> Result<String> {
    if value.contains(['"', '\\', '\n', '\r']) {
        bail!("value is not safe for generated TOML");
    }
    Ok(format!("\"{value}\""))
}

pub(crate) fn toml_string_assignment(line: &str, name: &str) -> Option<String> {
    let value = line.trim().strip_prefix(&format!("{name} = "))?;
    serde_json::from_str::<String>(value).ok()
}
