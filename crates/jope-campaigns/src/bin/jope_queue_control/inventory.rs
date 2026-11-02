use super::*;

pub(super) fn load_claims(path: &Path) -> Result<HashMap<String, Claim>> {
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .with_context(|| format!("open backlog {}", path.display()))?;
    connection.busy_timeout(std::time::Duration::from_secs(60))?;
    let mut statement = connection.prepare(
        "SELECT job_id,manifest_sha256,manifest,state FROM claims WHERE job_id IS NOT NULL",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            Claim {
                manifest_sha256: row.get(1)?,
                manifest: PathBuf::from(row.get::<_, String>(2)?),
                state: row.get(3)?,
            },
        ))
    })?;
    let mut claims = HashMap::new();
    for row in rows {
        let (job_id, claim) = row?;
        if claims.insert(job_id.clone(), claim).is_some() {
            bail!("duplicate backlog job_id {job_id}");
        }
    }
    Ok(claims)
}

pub(super) fn scheduler_jobs(scq: &Path, socket: &Path) -> Result<Vec<Value>> {
    let socket = socket
        .to_str()
        .context("SCQ socket path is not valid UTF-8")?;
    let output = scq_json(scq, &["ls", "--socket", socket])?;
    Ok(output
        .get("Jobs")
        .and_then(|value| value.get("jobs"))
        .or_else(|| output.get("jobs"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default())
}

pub(super) fn join_inventory(
    jobs: &[Value],
    claims: &HashMap<String, Claim>,
    mode: ValidationMode,
    workers: usize,
) -> Result<Vec<JobRecord>> {
    // Manifest validation is CPU-heavy (content and semantic SHA-256 checks).
    // Bound concurrency so it is fast without flooding the shared filesystem.
    ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|index| format!("queue-audit-{index}"))
        .build()?
        .install(|| {
            jobs.par_iter()
                .map(|job| join_job(job, claims, mode))
                .collect()
        })
}

pub(super) fn join_job(
    job: &Value,
    claims: &HashMap<String, Claim>,
    mode: ValidationMode,
) -> Result<JobRecord> {
    let job_id = text_field(job, "id")
        .or_else(|| text_field(job, "job_id"))
        .unwrap_or_default();
    let project = text_field(job, "project").unwrap_or_else(|| "unknown".to_owned());
    let state = text_field(job, "state").unwrap_or_else(|| "unknown".to_owned());
    let Some(claim) = claims.get(&job_id) else {
        return Ok(JobRecord {
            job_id,
            project,
            state,
            epoch: "unclaimed".to_owned(),
            strict_wave: None,
            generation: None,
            checkpoint_sha256: None,
            claim_state: None,
            manifest: None,
            manifest_sha256: None,
            manifest_checked: false,
            manifest_valid: false,
            invalid_reason: Some("no_matching_backlog_claim".to_owned()),
        });
    };

    let should_validate = claim.state == "submitted"
        && match mode {
            ValidationMode::Audit => false,
            ValidationMode::HoldLegacy => project == LEGACY_PROJECT && is_holdable_state(&state),
            ValidationMode::HoldSupersededWaves => {
                project == SELFPLAY_PROJECT && is_holdable_state(&state)
            }
            ValidationMode::RetireHeldLegacy => {
                project == LEGACY_PROJECT && matches!(state.as_str(), "held" | "cancelled")
            }
            ValidationMode::RetireHeldSupersededWaves => {
                project == SELFPLAY_PROJECT && matches!(state.as_str(), "held" | "cancelled")
            }
            ValidationMode::ReleaseCurrentCheckpointBridge => {
                project == SELFPLAY_PROJECT && state == "held"
            }
        };
    let validation = should_validate.then(|| validate_manifest(claim));
    let (manifest, invalid_reason) = match validation {
        Some(Ok(value)) => (Some(value), None),
        Some(Err(error)) => (None, Some(format!("{error:#}"))),
        None => (None, None),
    };
    let wave_hint = strict_wave_from_path(&claim.manifest);
    let generation_hint = wave_hint.clone();
    Ok(JobRecord {
        job_id,
        project,
        state,
        epoch: manifest.as_ref().map(manifest_epoch).unwrap_or_else(|| {
            if should_validate {
                "invalid-manifest".to_owned()
            } else if wave_hint.is_some() {
                "manifest-wave-unchecked".to_owned()
            } else {
                LEGACY_EPOCH.to_owned()
            }
        }),
        strict_wave: manifest.as_ref().and_then(strict_wave).or(wave_hint),
        generation: manifest.as_ref().and_then(generation).or(generation_hint),
        checkpoint_sha256: manifest.as_ref().and_then(manifest_checkpoint_binding),
        claim_state: Some(claim.state.clone()),
        manifest: Some(claim.manifest.to_string_lossy().into_owned()),
        manifest_sha256: Some(claim.manifest_sha256.clone()),
        manifest_checked: should_validate,
        manifest_valid: manifest.is_some(),
        invalid_reason,
    })
}

pub(super) fn validate_manifest(claim: &Claim) -> Result<Value> {
    if claim.manifest_sha256.len() != 64
        || !claim
            .manifest_sha256
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit())
    {
        bail!("claim manifest hash is not SHA-256");
    }
    if !claim.manifest.is_file() {
        bail!("claimed manifest does not exist");
    }
    let actual = sha256_file(&claim.manifest)?;
    if actual != claim.manifest_sha256 {
        bail!("claimed manifest hash drift");
    }
    let manifest: Value = serde_json::from_slice(&fs::read(&claim.manifest)?)
        .with_context(|| format!("parse manifest {}", claim.manifest.display()))?;
    if manifest.get("schema").and_then(Value::as_str) != Some(MANIFEST_SCHEMA) {
        bail!("wrong manifest schema");
    }
    if !manifest_freshness_scoped(&manifest) || protected_lane(&manifest) {
        bail!("manifest is outside freshness scope");
    }
    let task = manifest
        .get("task")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !matches!(task, "regression" | "binary" | "timeseries" | "multiclass") {
        bail!("unsupported task");
    }
    let candidates = manifest
        .get("candidates")
        .and_then(Value::as_array)
        .context("manifest candidates are absent")?;
    if candidates.is_empty() || candidates.len() > 100 {
        bail!("candidate count is outside 1..=100");
    }
    let mut semantics = BTreeSet::new();
    for candidate in candidates {
        let genome = candidate
            .get("canonical_genome")
            .and_then(Value::as_str)
            .context("candidate canonical genome is absent")?;
        let semantic = candidate
            .get("semantic_sha256")
            .and_then(Value::as_str)
            .context("candidate semantic hash is absent")?;
        if sha256_bytes(genome.as_bytes()) != semantic {
            bail!("candidate semantic hash drift");
        }
        if !semantics.insert(semantic) {
            bail!("duplicate candidate semantic");
        }
    }
    Ok(manifest)
}

pub(super) fn protected_lane(manifest: &Value) -> bool {
    let lane = manifest
        .get("evaluation_lane")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_ascii_lowercase();
    lane.contains("gate") || lane.contains("sealed") || lane.contains("canary")
}

pub(super) fn strict_wave(manifest: &Value) -> Option<String> {
    manifest
        .get("strict_wave")
        .and_then(Value::as_str)
        .or_else(|| {
            manifest
                .get("label_authority")
                .and_then(|value| value.get("strict_wave"))
                .and_then(Value::as_str)
        })
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

pub(super) fn generation(manifest: &Value) -> Option<String> {
    manifest
        .get("generation")
        .and_then(Value::as_str)
        .or_else(|| {
            manifest
                .get("label_authority")
                .and_then(|value| value.get("generation"))
                .and_then(Value::as_str)
        })
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

pub(super) fn manifest_checkpoint_binding(manifest: &Value) -> Option<String> {
    let candidates = manifest.get("candidates")?.as_array()?;
    let checkpoint = candidates
        .first()?
        .get("model_checkpoint_sha256")?
        .as_str()?;
    if !is_sha256(checkpoint)
        || !candidates.iter().all(|candidate| {
            candidate
                .get("model_checkpoint_sha256")
                .and_then(Value::as_str)
                == Some(checkpoint)
        })
        || manifest
            .get("label_authority")
            .and_then(|value| value.get("model_checkpoint_sha256"))
            .and_then(Value::as_str)
            != Some(checkpoint)
        || manifest
            .get("checkpoint_sha256")
            .and_then(Value::as_str)
            .is_some_and(|value| value != checkpoint)
    {
        return None;
    }
    Some(checkpoint.to_owned())
}

pub(super) fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub(super) fn strict_wave_from_path(path: &Path) -> Option<String> {
    path.ancestors()
        .filter_map(Path::file_name)
        .filter_map(|name| name.to_str())
        .find(|name| {
            name.strip_prefix("strict-wave-").is_some_and(|suffix| {
                !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
            })
        })
        .map(str::to_owned)
}

pub(super) fn queue_report(records: &[JobRecord]) -> QueueReport {
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

pub(super) fn newest_wave(records: &[JobRecord]) -> Option<String> {
    records
        .iter()
        .filter(|row| row.project == SELFPLAY_PROJECT && row.claim_state.is_some())
        .filter_map(|row| row.strict_wave.as_ref())
        .max_by_key(|wave| wave_number(wave))
        .cloned()
}

pub(super) fn wave_number(wave: &str) -> (u64, &str) {
    (
        wave.rsplit('-')
            .next()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0),
        wave,
    )
}

pub(super) fn hold_candidates(
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

pub(super) fn retirement_candidates(
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

pub(super) fn checkpoint_bridge_candidates(
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

pub(super) fn is_holdable_state(state: &str) -> bool {
    matches!(state, "queued" | "starting")
}

pub(super) fn states_by_id(jobs: &[Value]) -> HashMap<String, (String, String)> {
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

pub(super) fn text_field(value: &Value, field: &str) -> Option<String> {
    value.get(field).and_then(Value::as_str).map(str::to_owned)
}
