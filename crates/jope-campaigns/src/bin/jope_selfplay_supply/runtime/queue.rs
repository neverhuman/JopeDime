use super::*;

pub(crate) fn load_queue_status(args: &Args) -> Result<QueueStatus> {
    let path = args.queue_status();
    let status: QueueStatus = serde_json::from_slice(&fs::read(&path)?)
        .with_context(|| format!("parse queue status {}", path.display()))?;
    if status.schema != "JopeSelfplayQueueKeeper.v3" {
        bail!("queue keeper status schema is not v3");
    }
    let maximum_age_ms = u128::from(args.queue_status_max_age_seconds) * 1_000;
    if now_ms().saturating_sub(status.generated_at_ms) > maximum_age_ms {
        bail!("queue keeper status is stale");
    }
    if status.queued.saturating_add(status.starting) != status.queued_starting {
        bail!("queue keeper queued/starting aggregate drift");
    }
    if status
        .reserve_manifests
        .saturating_add(status.admission_manifests)
        != status.unclaimed_current_checkpoint_manifests
    {
        bail!("queue keeper reserve/admission manifest aggregate drift");
    }
    if status.complete_reserve_waves != status.protected_reserve_waves.len() {
        bail!("queue keeper complete reserve-wave aggregate drift");
    }
    let useful_work = status
        .queued
        .saturating_add(status.starting)
        .saturating_add(status.dispatching)
        .saturating_add(status.validated_ready);
    if useful_work != status.useful_work {
        bail!("queue keeper useful-work aggregate drift");
    }
    if (status.useful_work == 0) != (status.candidate_evaluation_work == 0)
        || status.candidate_evaluation_work < status.useful_work as u64
    {
        bail!("queue keeper candidate-evaluation work aggregate drift");
    }
    if status
        .admission_waves
        .iter()
        .chain(status.protected_reserve_waves.iter())
        .any(|wave| wave.unclaimed_manifests == 0 || wave.candidate_evaluation_count == 0)
    {
        bail!("queue keeper wave lacks candidate-evaluation work");
    }
    let admission = status
        .admission_waves
        .iter()
        .map(|wave| wave.wave.as_str())
        .collect::<BTreeSet<_>>();
    if status
        .protected_reserve_waves
        .iter()
        .any(|wave| !wave.complete || admission.contains(wave.wave.as_str()))
    {
        bail!("queue keeper reserve/admission wave separation drift");
    }
    Ok(status)
}

pub(crate) fn load_pending_ack(root: &Path, checkpoint_sha256: &str) -> Result<Option<PendingAck>> {
    let path = root.join("ops/micro-wave-supply-v2.json");
    if !path.is_file() {
        return Ok(None);
    }
    let value: Value = serde_json::from_slice(&fs::read(&path)?)
        .with_context(|| format!("parse prior supply status {}", path.display()))?;
    if value.get("schema").and_then(Value::as_str) != Some("JopeMicroWaveSupplyStatus.v2")
        || value
            .get("active_checkpoint_sha256")
            .and_then(Value::as_str)
            != Some(checkpoint_sha256)
    {
        return Ok(None);
    }
    let Some(wave) = value.get("awaiting_keeper_ack").and_then(Value::as_str) else {
        return Ok(None);
    };
    if !valid_wave_id(wave) {
        bail!("prior supply status has an invalid acknowledgment wave");
    }
    let since_ms = value
        .get("awaiting_keeper_ack_since_ms")
        .and_then(Value::as_u64)
        .context("prior supply status lacks acknowledgment timestamp")?;
    Ok(Some(PendingAck {
        wave: wave.to_owned(),
        since_ms: u128::from(since_ms),
    }))
}

pub(crate) fn keeper_acknowledged(
    args: &Args,
    status: &QueueStatus,
    inputs: &CheckpointInputs,
    epoch: &str,
    claimed: &BTreeSet<String>,
    pending: &PendingAck,
) -> Result<bool> {
    if status.generated_at_ms <= pending.since_ms {
        return Ok(false);
    }
    if status
        .admission_waves
        .iter()
        .chain(status.protected_reserve_waves.iter())
        .any(|wave| wave.wave == pending.wave)
    {
        return Ok(true);
    }
    let directory = args.root.join("waves").join(&pending.wave);
    let Some(wave) = validate_ready_wave(
        &pending.wave,
        &directory,
        &inputs.checkpoint_sha256,
        epoch,
        claimed,
    )?
    else {
        bail!("keeper did not acknowledge the built current-checkpoint wave");
    };
    Ok(wave.unclaimed_manifests == 0)
}

pub(crate) fn validate_keeper_admission(
    status: &QueueStatus,
    ready: &[ReadyWave],
    waves_root: &Path,
    pending_wave: Option<&str>,
    max_overlap_waves: usize,
) -> Result<()> {
    if status
        .admission_waves
        .len()
        .saturating_add(status.protected_reserve_waves.len())
        > max_overlap_waves
    {
        bail!("keeper admission wave set exceeds the bounded overlap");
    }
    if status
        .admission_waves
        .windows(2)
        .any(|pair| pair[0].wave >= pair[1].wave)
    {
        bail!("keeper admission waves are not deterministic oldest-first");
    }
    let reported_unclaimed = status
        .admission_waves
        .iter()
        .chain(status.protected_reserve_waves.iter())
        .try_fold(0_usize, |total, wave| {
            total.checked_add(wave.unclaimed_manifests)
        })
        .context("keeper unclaimed manifest count overflow")?;
    if reported_unclaimed != status.unclaimed_current_checkpoint_manifests {
        bail!("keeper admission wave counts do not match its aggregate");
    }
    let ready_by_wave = ready
        .iter()
        .map(|wave| (wave.wave.as_str(), wave))
        .collect::<BTreeMap<_, _>>();
    let mut admitted = BTreeSet::new();
    for keeper_wave in status
        .admission_waves
        .iter()
        .chain(status.protected_reserve_waves.iter())
    {
        if !admitted.insert(keeper_wave.wave.as_str()) {
            bail!("keeper admission wave is duplicated");
        }
        if let Some(local) = ready_by_wave.get(keeper_wave.wave.as_str()) {
            if keeper_wave.summary_sha256 != local.summary_sha256
                || local.unclaimed_manifests > keeper_wave.unclaimed_manifests
            {
                bail!("keeper admission wave provenance or count drift");
            }
        } else {
            if !valid_wave_id(&keeper_wave.wave) {
                bail!("keeper admission wave identity is invalid");
            }
            let summary = waves_root.join(&keeper_wave.wave).join("summary.json");
            if !summary.is_file() || sha256_file(&summary)? != keeper_wave.summary_sha256 {
                bail!("keeper admitted a wave with no matching local immutable summary");
            }
        }
    }
    if status
        .protected_reserve_waves
        .iter()
        .any(|wave| !wave.complete || wave.manifest_count != wave.unclaimed_manifests)
    {
        bail!("keeper protected a partial reserve wave");
    }
    for wave in ready {
        if !admitted.contains(wave.wave.as_str()) && pending_wave != Some(wave.wave.as_str()) {
            bail!("current-checkpoint unclaimed wave is outside keeper admission");
        }
    }
    Ok(())
}
