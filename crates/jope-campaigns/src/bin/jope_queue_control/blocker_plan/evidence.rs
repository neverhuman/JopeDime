use super::*;

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
