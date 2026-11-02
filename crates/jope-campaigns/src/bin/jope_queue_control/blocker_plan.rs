use super::*;

pub(crate) fn build_wave_blocker_plan(
    root: &Path,
    requested_wave_dir: &Path,
    pair_registry: &Path,
    claims: &HashMap<String, Claim>,
    scheduler: &HashMap<String, (String, String)>,
    receipt_dir: &Path,
    scq_events: &Path,
) -> Result<WaveBlockerPlan> {
    let wave_dir = fs::canonicalize(requested_wave_dir)
        .with_context(|| format!("canonicalize wave {}", requested_wave_dir.display()))?;
    let waves_root = fs::canonicalize(root.join("waves"))?;
    if wave_dir.parent() != Some(waves_root.as_path()) {
        bail!("blocker retirement wave must be a direct child of the configured waves root");
    }
    let wave = wave_dir
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| name.starts_with("strict-wave-"))
        .context("wave directory name is not strict-wave-*")?
        .to_owned();
    let summary_path = wave_dir.join("summary.json");
    let cohort_path = wave_dir.join("cohort.json");
    let summary_bytes = fs::read(&summary_path)?;
    let cohort_bytes = fs::read(&cohort_path)?;
    let summary: Value = serde_json::from_slice(&summary_bytes)?;
    let cohort: Value = serde_json::from_slice(&cohort_bytes)?;
    let summary_sha256 = sha256_bytes(&summary_bytes);
    let cohort_sha256 = sha256_bytes(&cohort_bytes);
    let checkpoint_sha256 = required_json_str(&summary, "checkpoint_sha256")?.to_owned();
    let checkpoint_epoch = required_json_str(&summary, "checkpoint_epoch")?.to_owned();
    let catalog_sha256 = required_json_str(&summary, "catalog_sha256")?;
    if !is_sha256(&checkpoint_sha256)
        || checkpoint_epoch != format!("checkpoint-{checkpoint_sha256}")
        || !is_sha256(catalog_sha256)
        || summary.get("immutable").and_then(Value::as_bool) != Some(true)
        // Any legal micro-wave shape: tasks.len() * 64. The exact count is
        // cross-checked against the frozen cohort's entry list below, so this
        // is a range gate, not the authority.
        || !matches!(
            summary.get("datasets").and_then(Value::as_u64),
            Some(128 | 192 | 256)
        )
        || summary
            .get("candidates_per_dataset")
            .and_then(Value::as_u64)
            != Some(256)
        || summary.get("cohort_sha256").and_then(Value::as_str) != Some(cohort_sha256.as_str())
    {
        bail!("wave summary invariants failed");
    }
    let wave_datasets = summary
        .get("datasets")
        .and_then(Value::as_u64)
        .expect("validated above") as usize;
    if cohort.get("schema").and_then(Value::as_str) != Some("JopeMicroCohort.v2")
        || cohort.get("wave").and_then(Value::as_str) != Some(wave.as_str())
        || cohort.get("checkpoint_sha256").and_then(Value::as_str)
            != Some(checkpoint_sha256.as_str())
        || cohort.get("checkpoint_epoch").and_then(Value::as_str) != Some(checkpoint_epoch.as_str())
        || cohort.get("catalog_sha256").and_then(Value::as_str) != Some(catalog_sha256)
        || cohort.get("immutable").and_then(Value::as_bool) != Some(true)
        || cohort
            .get("entries")
            .and_then(Value::as_array)
            .map(Vec::len)
            != Some(wave_datasets)
    {
        bail!("wave cohort invariants failed");
    }

    let entries = summary
        .get("manifests")
        .and_then(Value::as_array)
        .context("wave summary manifests are absent")?;
    // manifests = datasets * ceil(256 / max_candidates_per_manifest), derived
    // from the summary's own declared shard cap (the historical literal 384 was
    // 128 datasets at the never-used cap of 100 -- 128 * 3 shards).
    let max_shard = summary
        .get("max_candidates_per_manifest")
        .and_then(Value::as_u64)
        .unwrap_or(100)
        .max(1) as usize;
    let expected_manifests = wave_datasets * 256usize.div_ceil(max_shard);
    if summary.get("manifest_count").and_then(Value::as_u64) != Some(entries.len() as u64)
        || entries.len() != expected_manifests
    {
        bail!(
            "wave summary manifest count {} is not the expected {expected_manifests}",
            entries.len()
        );
    }
    let mut desired = BTreeSet::<(String, String, String, String)>::new();
    let mut current_manifest_hashes = BTreeSet::new();
    let mut seen_paths = BTreeSet::new();
    for entry in entries {
        let relative = required_json_str(entry, "relative_path")?;
        let expected_hash = required_json_str(entry, "sha256")?;
        let expected_task = required_json_str(entry, "task")?;
        let expected_dataset = required_json_str(entry, "dataset_hash")?;
        let expected_candidates = entry
            .get("candidate_count")
            .and_then(Value::as_u64)
            .context("summary candidate_count is absent")?
            as usize;
        if !is_sha256(expected_hash)
            || !seen_paths.insert(relative.to_owned())
            || !matches!(
                expected_task,
                "regression" | "binary" | "timeseries" | "multiclass"
            )
        {
            bail!("wave summary manifest entry is invalid or duplicated");
        }
        let manifest_path = wave_dir.join(relative);
        let canonical_manifest = fs::canonicalize(&manifest_path)?;
        if canonical_manifest.parent() != Some(wave_dir.join("manifests").as_path()) {
            bail!("wave manifest escapes the immutable manifests directory");
        }
        if sha256_file(&canonical_manifest)? != expected_hash {
            bail!("current wave manifest hash drift");
        }
        let claim = Claim {
            manifest_sha256: expected_hash.to_owned(),
            manifest: canonical_manifest,
            state: "submitted".to_owned(),
        };
        let manifest = validate_manifest(&claim)?;
        if strict_wave(&manifest).as_deref() != Some(wave.as_str())
            || manifest.get("checkpoint_sha256").and_then(Value::as_str)
                != Some(checkpoint_sha256.as_str())
            || manifest.get("checkpoint_epoch").and_then(Value::as_str)
                != Some(checkpoint_epoch.as_str())
            || manifest.get("catalog_sha256").and_then(Value::as_str) != Some(catalog_sha256)
            || manifest.get("task").and_then(Value::as_str) != Some(expected_task)
            || manifest.get("dataset_hash").and_then(Value::as_str) != Some(expected_dataset)
        {
            bail!("current wave manifest disagrees with its summary");
        }
        let protocol = required_json_str(&manifest, "protocol_sha256")?;
        if !is_sha256(protocol) {
            bail!("current wave protocol is not SHA-256");
        }
        let candidates = manifest
            .get("candidates")
            .and_then(Value::as_array)
            .context("current wave candidates are absent")?;
        if candidates.len() != expected_candidates {
            bail!("current wave candidate count disagrees with summary");
        }
        for candidate in candidates {
            let semantic = required_json_str(candidate, "semantic_sha256")?;
            if !desired.insert((
                expected_task.to_owned(),
                expected_dataset.to_owned(),
                semantic.to_owned(),
                protocol.to_owned(),
            )) {
                bail!("current wave desired pair identity is duplicated");
            }
        }
        current_manifest_hashes.insert(expected_hash.to_owned());
    }
    if !desired.len().is_multiple_of(256) || !matches!(desired.len() / 256, 128 | 192 | 256) {
        bail!(
            "current wave desired pair count {} is not a legal micro-wave total",
            desired.len()
        );
    }

    let connection = Connection::open_with_flags(
        pair_registry,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    connection.busy_timeout(Duration::from_secs(60))?;
    connection.execute_batch(
        "PRAGMA temp_store=MEMORY;
         CREATE TEMP TABLE desired_pairs(
           task TEXT NOT NULL,dataset_hash TEXT NOT NULL,
           semantic_sha256 TEXT NOT NULL,protocol_sha256 TEXT NOT NULL,
           PRIMARY KEY(task,dataset_hash,semantic_sha256,protocol_sha256)
         ) WITHOUT ROWID;",
    )?;
    {
        let mut insert = connection.prepare("INSERT INTO desired_pairs VALUES(?1,?2,?3,?4)")?;
        for (task, dataset, semantic, protocol) in &desired {
            insert.execute(params![task, dataset, semantic, protocol])?;
        }
    }
    let mut statement = connection.prepare(DESIRED_BLOCKER_QUERY)?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<String>>(2)?,
                row.get::<_, Option<String>>(3)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(statement);
    drop(connection);

    let held_proofs = load_held_retirement_proofs(receipt_dir)?;
    let blocker_job_ids = rows
        .iter()
        .filter_map(|(_state, _path, manifest, job)| {
            let manifest = manifest.as_ref()?;
            (!current_manifest_hashes.contains(manifest))
                .then(|| job.clone())
                .flatten()
        })
        .collect::<BTreeSet<_>>();
    let started_jobs = load_scq_start_evidence(scq_events, &blocker_job_ids)?;
    let mut current_wave_active_pairs = 0usize;
    let mut blocking_pairs_by_manifest = BTreeMap::<String, usize>::new();
    let mut candidates_by_manifest = BTreeMap::<String, HoldCandidate>::new();
    let mut bindings_by_manifest = BTreeMap::<String, (String, String)>::new();
    let mut seen_jobs = BTreeMap::<String, String>::new();
    let mut proof_errors = Vec::new();
    let mut exact_submitted_claims = 0usize;
    for (_state, manifest_path, manifest_sha256, job_id) in rows {
        let (Some(manifest_path), Some(manifest_sha256), Some(job_id)) =
            (manifest_path, manifest_sha256, job_id)
        else {
            continue;
        };
        if current_manifest_hashes.contains(&manifest_sha256) {
            current_wave_active_pairs += 1;
            continue;
        }
        *blocking_pairs_by_manifest
            .entry(manifest_sha256.clone())
            .or_insert(0) += 1;
        if let Some(existing_job) = seen_jobs.insert(job_id.clone(), manifest_sha256.clone()) {
            if existing_job != manifest_sha256 {
                bail!("one stale job is bound to multiple manifest hashes");
            }
        }
        if let Some((existing_job, existing_path)) = bindings_by_manifest.get(&manifest_sha256) {
            if existing_job != &job_id || existing_path != &manifest_path {
                bail!("stale manifest pair bindings disagree on job or path");
            }
            continue;
        }
        bindings_by_manifest.insert(
            manifest_sha256.clone(),
            (job_id.clone(), manifest_path.clone()),
        );
        let claim = claims
            .get(&job_id)
            .with_context(|| format!("blocking job {job_id} has no backlog claim"))?;
        if claim.manifest_sha256 != manifest_sha256
            || claim.manifest.to_string_lossy() != manifest_path
        {
            bail!(
                "blocking job {job_id} manifest binding does not exactly match \
                 its claim (hash_match={}, path_match={})",
                claim.manifest_sha256 == manifest_sha256,
                claim.manifest.to_string_lossy() == manifest_path
            );
        }
        if claim.state != "submitted" {
            proof_errors.push(format!("{}: claim_not_submitted:{}", job_id, claim.state));
            continue;
        }
        exact_submitted_claims += 1;
        let old_manifest = match validate_manifest(claim) {
            Ok(manifest) => manifest,
            Err(error) => {
                proof_errors.push(format!(
                    "{job_id}: blocking_manifest_provenance_invalid:{error:#}"
                ));
                continue;
            }
        };
        let Some(old_wave) = strict_wave(&old_manifest) else {
            proof_errors.push(format!("{job_id}: blocking_manifest_has_no_strict_wave"));
            continue;
        };
        if wave_number(&old_wave) >= wave_number(&wave) {
            proof_errors.push(format!(
                "{job_id}: blocking_manifest_is_current_or_future_wave:{old_wave}"
            ));
            continue;
        }
        if let Some(reason) = started_jobs.get(&job_id) {
            proof_errors.push(format!("{job_id}: {reason}; excluded from recycling"));
            continue;
        }
        let observed_state = match scheduler.get(&job_id) {
            Some((project, state)) if project == SELFPLAY_PROJECT && state == "cancelled" => {
                Some("cancelled")
            }
            Some((_project, state)) => {
                proof_errors.push(format!("{job_id}: excluded live SCQ state {state}"));
                None
            }
            None if held_proofs.contains(&(
                job_id.clone(),
                manifest_sha256.clone(),
                manifest_path.clone(),
            )) =>
            {
                Some("held")
            }
            None => {
                proof_errors.push(format!(
                    "{job_id}: pruned from SCQ without held/unstarted retirement proof"
                ));
                None
            }
        };
        let Some(observed_state) = observed_state else {
            continue;
        };
        candidates_by_manifest.insert(
            manifest_sha256.clone(),
            HoldCandidate {
                job_id,
                project: SELFPLAY_PROJECT.to_owned(),
                observed_state: observed_state.to_owned(),
                epoch: manifest_epoch(&old_manifest),
                strict_wave: Some(old_wave),
                generation: generation(&old_manifest),
                manifest: manifest_path,
                manifest_sha256,
            },
        );
    }
    let blocking_pairs = blocking_pairs_by_manifest.values().sum();
    let candidates = candidates_by_manifest.into_values().collect::<Vec<_>>();
    let blocking_manifests = bindings_by_manifest.len();
    Ok(WaveBlockerPlan {
        wave,
        wave_dir: wave_dir.to_string_lossy().into_owned(),
        summary_sha256,
        cohort_sha256,
        checkpoint_sha256,
        checkpoint_epoch,
        desired_pairs: desired.len(),
        current_wave_active_pairs,
        blocking_pairs,
        blocking_claims: blocking_manifests,
        exact_submitted_claims,
        blocking_manifests,
        proven_blocking_manifests: candidates.len(),
        unproven_blocking_manifests: blocking_manifests - candidates.len(),
        proof_errors,
        candidates,
    })
}

pub(crate) fn load_held_retirement_proofs(
    receipt_dir: &Path,
) -> Result<BTreeSet<(String, String, String)>> {
    let mut proofs = BTreeSet::new();
    if !receipt_dir.is_dir() {
        return Ok(proofs);
    }
    for entry in fs::read_dir(receipt_dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(expected_hash) = name
            .strip_prefix("preview-")
            .and_then(|value| value.strip_suffix(".json"))
        else {
            continue;
        };
        if !is_sha256(expected_hash) || sha256_file(&path)? != expected_hash {
            continue;
        }
        let value: Value = serde_json::from_slice(&fs::read(&path)?)?;
        if value.get("schema").and_then(Value::as_str) != Some("JopeQueueRetirementPreview.v1") {
            continue;
        }
        let Some(candidates) = value.get("candidates").and_then(Value::as_array) else {
            continue;
        };
        for candidate in candidates {
            let candidate: HoldCandidate = serde_json::from_value(candidate.clone())?;
            if candidate.project == SELFPLAY_PROJECT && candidate.observed_state == "held" {
                proofs.insert((
                    candidate.job_id,
                    candidate.manifest_sha256,
                    candidate.manifest,
                ));
            }
        }
    }
    Ok(proofs)
}

pub(crate) fn load_scq_start_evidence(
    event_log: &Path,
    blocker_jobs: &BTreeSet<String>,
) -> Result<BTreeMap<String, String>> {
    if blocker_jobs.is_empty() {
        return Ok(BTreeMap::new());
    }
    let file = File::open(event_log)
        .with_context(|| format!("open SCQ event history {}", event_log.display()))?;
    let mut evidence = BTreeMap::new();
    for (line_number, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if !line.contains("\"job\"") {
            continue;
        }
        let event: Value = serde_json::from_str(&line).with_context(|| {
            format!(
                "parse SCQ event history {}:{}",
                event_log.display(),
                line_number + 1
            )
        })?;
        let Some(job_id) = event.get("job").and_then(Value::as_str) else {
            continue;
        };
        if !blocker_jobs.contains(job_id) {
            continue;
        }
        let kind = event
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let class = event
            .get("class")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if kind == "exit" && class == "completed" {
            evidence.insert(
                job_id.to_owned(),
                "durable SCQ history records completion".to_owned(),
            );
        } else if matches!(kind, "launch" | "running") {
            evidence
                .entry(job_id.to_owned())
                .or_insert_with(|| "durable SCQ history records launch".to_owned());
        }
    }
    Ok(evidence)
}

pub(crate) fn revalidate_wave_blocker_plan(
    plan: &WaveBlockerPlan,
    scheduler: &HashMap<String, (String, String)>,
    claims: &HashMap<String, Claim>,
    receipt_dir: &Path,
    scq_events: &Path,
) -> Result<()> {
    let held_proofs = load_held_retirement_proofs(receipt_dir)?;
    let job_ids = plan
        .candidates
        .iter()
        .map(|candidate| candidate.job_id.clone())
        .collect::<BTreeSet<_>>();
    let started_jobs = load_scq_start_evidence(scq_events, &job_ids)?;
    for candidate in &plan.candidates {
        if let Some(reason) = started_jobs.get(&candidate.job_id) {
            bail!("blocker gained start evidence before transaction: {reason}");
        }
        let claim = claims
            .get(&candidate.job_id)
            .context("blocker claim disappeared before transaction")?;
        if !claim_matches_candidate(claim, candidate) {
            bail!("blocker claim changed before transaction");
        }
        let manifest = validate_manifest(claim)?;
        let old_wave = strict_wave(&manifest).context("blocker strict wave disappeared")?;
        if wave_number(&old_wave) >= wave_number(&plan.wave) {
            bail!("blocker provenance advanced before transaction");
        }
        match candidate.observed_state.as_str() {
            "cancelled"
                if scheduler
                    .get(&candidate.job_id)
                    .is_some_and(|(project, state)| {
                        project == SELFPLAY_PROJECT && state == "cancelled"
                    }) => {}
            "held"
                if !scheduler.contains_key(&candidate.job_id)
                    && held_proofs.contains(&(
                        candidate.job_id.clone(),
                        candidate.manifest_sha256.clone(),
                        candidate.manifest.clone(),
                    )) => {}
            _ => bail!("blocker SCQ state or held proof changed before transaction"),
        }
    }
    Ok(())
}

pub(crate) fn required_json_str<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value
        .get(field)
        .and_then(Value::as_str)
        .with_context(|| format!("{field} is absent"))
}
