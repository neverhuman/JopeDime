use super::*;

pub(crate) fn queue_report(records: &[JobRecord]) -> QueueReport {
    let mut counts = BTreeMap::new();
    let mut waves = BTreeMap::new();
    let mut generations = BTreeMap::new();
    for record in records {
        *counts
            .entry(record.project.clone())
            .or_insert_with(BTreeMap::new)
            .entry(record.state.clone())
            .or_insert_with(BTreeMap::new)
            .entry(record.epoch.clone())
            .or_insert(0) += 1;
        if record.project == SELFPLAY_PROJECT {
            *waves
                .entry(
                    record
                        .strict_wave
                        .clone()
                        .unwrap_or_else(|| "none".to_owned()),
                )
                .or_insert_with(BTreeMap::new)
                .entry(record.state.clone())
                .or_insert(0) += 1;
            *generations
                .entry(
                    record
                        .generation
                        .clone()
                        .unwrap_or_else(|| "none".to_owned()),
                )
                .or_insert_with(BTreeMap::new)
                .entry(record.state.clone())
                .or_insert(0) += 1;
        }
    }
    QueueReport {
        schema: "JopeQueueControlStatus.v1",
        generated_at_ms: now_ms(),
        counts_by_project_state_epoch: counts,
        selfplay_counts_by_wave_state: waves,
        selfplay_counts_by_generation_state: generations,
        newest_strict_wave: newest_wave(records),
        scheduler_jobs: records.len(),
        joined_claims: records
            .iter()
            .filter(|row| row.claim_state.is_some())
            .count(),
        valid_manifests: records
            .iter()
            .filter(|row| row.manifest_checked && row.manifest_valid)
            .count(),
        invalid_manifests: records
            .iter()
            .filter(|row| row.manifest_checked && !row.manifest_valid)
            .count(),
        unchecked_manifests: records
            .iter()
            .filter(|row| row.claim_state.is_some() && !row.manifest_checked)
            .count(),
        unclaimed_jobs: records
            .iter()
            .filter(|row| row.claim_state.is_none())
            .count(),
    }
}

pub(crate) fn newest_wave(records: &[JobRecord]) -> Option<String> {
    records
        .iter()
        .filter(|row| row.project == SELFPLAY_PROJECT && row.claim_state.is_some())
        .filter_map(|row| row.strict_wave.as_ref())
        .max_by_key(|wave| wave_number(wave))
        .cloned()
}

pub(crate) fn wave_number(wave: &str) -> (u64, &str) {
    (
        wave.rsplit('-')
            .next()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0),
        wave,
    )
}

pub(crate) fn hold_candidates(
    records: &[JobRecord],
    legacy: bool,
    superseded_waves: bool,
    newest_wave: Option<&str>,
) -> Vec<HoldCandidate> {
    let mut candidates = records
        .iter()
        .filter(|row| {
            row.manifest_valid
                && row.claim_state.as_deref() == Some("submitted")
                && is_holdable_state(&row.state)
                && if legacy {
                    row.project == LEGACY_PROJECT
                } else if superseded_waves {
                    row.project == SELFPLAY_PROJECT
                        && row.strict_wave.is_some()
                        && row.strict_wave.as_deref() != newest_wave
                } else {
                    false
                }
        })
        .filter_map(|row| {
            Some(HoldCandidate {
                job_id: row.job_id.clone(),
                project: row.project.clone(),
                observed_state: row.state.clone(),
                epoch: row.epoch.clone(),
                strict_wave: row.strict_wave.clone(),
                generation: row.generation.clone(),
                manifest: row.manifest.clone()?,
                manifest_sha256: row.manifest_sha256.clone()?,
            })
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.job_id.cmp(&right.job_id));
    candidates
}

pub(crate) fn retirement_candidates(
    records: &[JobRecord],
    legacy: bool,
    superseded_waves: bool,
    newest_wave: Option<&str>,
) -> Vec<HoldCandidate> {
    let mut candidates = records
        .iter()
        .filter(|row| {
            row.manifest_valid
                && row.claim_state.as_deref() == Some("submitted")
                && matches!(row.state.as_str(), "held" | "cancelled")
                && if legacy {
                    row.project == LEGACY_PROJECT
                } else if superseded_waves {
                    row.project == SELFPLAY_PROJECT
                        && row.strict_wave.is_some()
                        && row.strict_wave.as_deref() != newest_wave
                } else {
                    false
                }
        })
        .filter_map(|row| {
            Some(HoldCandidate {
                job_id: row.job_id.clone(),
                project: row.project.clone(),
                observed_state: row.state.clone(),
                epoch: row.epoch.clone(),
                strict_wave: row.strict_wave.clone(),
                generation: row.generation.clone(),
                manifest: row.manifest.clone()?,
                manifest_sha256: row.manifest_sha256.clone()?,
            })
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.job_id.cmp(&right.job_id));
    candidates
}

pub(crate) fn checkpoint_bridge_candidates(
    records: &[JobRecord],
    active_checkpoint: &str,
    limit: usize,
) -> Vec<HoldCandidate> {
    let mut candidates = records
        .iter()
        .filter(|row| {
            row.project == SELFPLAY_PROJECT
                && row.state == "held"
                && row.claim_state.as_deref() == Some("submitted")
                && row.manifest_valid
                && row.strict_wave.is_some()
                && row.checkpoint_sha256.as_deref() == Some(active_checkpoint)
        })
        .filter_map(|row| {
            Some(HoldCandidate {
                job_id: row.job_id.clone(),
                project: row.project.clone(),
                observed_state: row.state.clone(),
                epoch: row.epoch.clone(),
                strict_wave: row.strict_wave.clone(),
                generation: row.generation.clone(),
                manifest: row.manifest.clone()?,
                manifest_sha256: row.manifest_sha256.clone()?,
            })
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        wave_number(right.strict_wave.as_deref().unwrap_or_default())
            .cmp(&wave_number(
                left.strict_wave.as_deref().unwrap_or_default(),
            ))
            .then_with(|| left.job_id.cmp(&right.job_id))
    });
    candidates.truncate(limit);
    candidates
}

pub(crate) fn is_holdable_state(state: &str) -> bool {
    matches!(state, "queued" | "starting")
}

pub(crate) fn states_by_id(jobs: &[Value]) -> HashMap<String, (String, String)> {
    jobs.iter()
        .filter_map(|job| {
            let id = text_field(job, "id").or_else(|| text_field(job, "job_id"))?;
            Some((
                id,
                (
                    text_field(job, "project").unwrap_or_default(),
                    text_field(job, "state").unwrap_or_default(),
                ),
            ))
        })
        .collect()
}

pub(crate) fn text_field(value: &Value, field: &str) -> Option<String> {
    value.get(field).and_then(Value::as_str).map(str::to_owned)
}
