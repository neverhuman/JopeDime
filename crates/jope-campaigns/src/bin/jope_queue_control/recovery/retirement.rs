use super::*;

pub(crate) fn recover_retirement_candidates(
    receipt_dir: &Path,
    action: &str,
    claims: &HashMap<String, Claim>,
    scheduler: &HashMap<String, (String, String)>,
) -> Result<Vec<HoldCandidate>> {
    if !receipt_dir.is_dir() {
        return Ok(Vec::new());
    }
    let expected_project = if action == "retire_held_legacy" {
        LEGACY_PROJECT
    } else {
        SELFPLAY_PROJECT
    };
    let mut recovered = BTreeMap::<String, HoldCandidate>::new();
    for entry in fs::read_dir(receipt_dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.starts_with("preview-") || !name.ends_with(".json") {
            continue;
        }
        let expected_hash = name
            .strip_prefix("preview-")
            .and_then(|value| value.strip_suffix(".json"))
            .unwrap_or_default();
        if expected_hash.len() != 64 || sha256_file(&path)? != expected_hash {
            continue;
        }
        let value: Value = serde_json::from_slice(&fs::read(&path)?)
            .with_context(|| format!("parse preview {}", path.display()))?;
        if value.get("schema").and_then(Value::as_str) != Some("JopeQueueRetirementPreview.v1")
            || value.get("action").and_then(Value::as_str) != Some(action)
        {
            continue;
        }
        let Some(rows) = value.get("candidates").and_then(Value::as_array) else {
            continue;
        };
        for row in rows {
            let candidate: HoldCandidate = serde_json::from_value(row.clone())?;
            // Present cancelled rows are already revalidated in the current
            // inventory. Prior previews are only needed after SCQ prunes a row.
            if candidate.project != expected_project || scheduler.contains_key(&candidate.job_id) {
                continue;
            }
            let Some(claim) = claims.get(&candidate.job_id) else {
                continue;
            };
            if !claim_matches_candidate(claim, &candidate) {
                continue;
            }
            recovered.insert(candidate.job_id.clone(), candidate);
        }
    }
    // Multiple durable previews intentionally repeat the same candidate. Hash
    // each immutable manifest once after deduplication, while still performing
    // a fresh validation immediately before any recovered mutation.
    for candidate in recovered.values() {
        let claim = claims
            .get(&candidate.job_id)
            .context("recovered retirement claim disappeared")?;
        validate_manifest(claim).with_context(|| format!("recover {}", candidate.job_id))?;
    }
    Ok(recovered.into_values().collect())
}

pub(crate) fn build_retired_binding_repair_plan(
    wave_dir: &Path,
    pair_registry: &Path,
    backlog: &Path,
    receipt_dir: &Path,
) -> Result<RetiredBindingRepairPlan> {
    let wave_dir = fs::canonicalize(wave_dir)?;
    let wave = wave_dir
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| name.starts_with("strict-wave-"))
        .context("repair wave directory name is invalid")?
        .to_owned();
    let mut audits = Vec::<(u64, PathBuf, String, Value)>::new();
    for entry in fs::read_dir(receipt_dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(expected_hash) = name
            .strip_prefix("blocker-audit-")
            .and_then(|value| value.strip_suffix(".json"))
        else {
            continue;
        };
        if !is_sha256(expected_hash) || sha256_file(&path)? != expected_hash {
            continue;
        }
        let value: Value = serde_json::from_slice(&fs::read(&path)?)?;
        if value.get("schema").and_then(Value::as_str) != Some("JopeWaveBlockerRetirementAudit.v1")
            || value.get("wave").and_then(Value::as_str) != Some(wave.as_str())
        {
            continue;
        }
        let expected_hash = expected_hash.to_owned();
        audits.push((
            value
                .get("created_at_ms")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            path,
            expected_hash,
            value,
        ));
    }
    audits.sort_by_key(|entry| entry.0);
    let (_, audit_path, audit_sha256, audit) = audits
        .pop()
        .context("no blocker retirement audit for wave")?;
    let source_preview_sha256 = required_json_str(&audit, "preview_sha256")?.to_owned();
    let source_preview_path = PathBuf::from(required_json_str(&audit, "preview_path")?);
    if !is_sha256(&source_preview_sha256)
        || sha256_file(&source_preview_path)? != source_preview_sha256
    {
        bail!("blocker retirement source preview hash drift");
    }
    let preview: Value = serde_json::from_slice(&fs::read(&source_preview_path)?)?;
    if preview.get("schema").and_then(Value::as_str) != Some("JopeWaveBlockerRetirementPreview.v1")
        || preview
            .get("plan")
            .and_then(|value| value.get("wave"))
            .and_then(Value::as_str)
            != Some(wave.as_str())
    {
        bail!("blocker retirement source preview schema or wave drift");
    }
    let results = audit
        .get("results")
        .and_then(Value::as_array)
        .context("blocker audit results are absent")?;
    let retired_jobs = results
        .iter()
        .map(|result| {
            if result.get("claim_updated").and_then(Value::as_bool) != Some(true)
                || result
                    .get("command_error")
                    .is_some_and(|value| !value.is_null())
                || result
                    .get("registry_error")
                    .is_some_and(|value| !value.is_null())
            {
                bail!("blocker audit contains an unsuccessful retirement");
            }
            Ok(required_json_str(result, "job_id")?.to_owned())
        })
        .collect::<Result<BTreeSet<_>>>()?;
    let all_candidates = preview
        .get("plan")
        .and_then(|value| value.get("candidates"))
        .and_then(Value::as_array)
        .context("blocker source candidates are absent")?;
    let mut candidates = all_candidates
        .iter()
        .map(|candidate| serde_json::from_value::<HoldCandidate>(candidate.clone()))
        .collect::<serde_json::Result<Vec<_>>>()?;
    candidates.retain(|candidate| retired_jobs.contains(&candidate.job_id));
    candidates.sort_by(|left, right| left.job_id.cmp(&right.job_id));
    if candidates.len() != retired_jobs.len() || candidates.is_empty() {
        bail!("blocker source audit/candidate identity mismatch");
    }

    let pair = Connection::open_with_flags(
        pair_registry,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    let owner_column = if sqlite_column_exists(&pair, "pairs", "reservation_owner")? {
        "reservation_owner"
    } else {
        "owner"
    };
    let backlog = Connection::open_with_flags(
        backlog,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    let retirement_events = {
        let mut statement = pair.prepare(
            "SELECT detail_json FROM events
             WHERE id > (SELECT COALESCE(MAX(id),0)-100000 FROM events)
               AND kind='superseded_unstarted'",
        )?;
        let details = statement
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        details
            .into_iter()
            .filter_map(|detail| serde_json::from_str::<Value>(&detail).ok())
            .filter_map(|detail| {
                Some((
                    detail.get("job_id")?.as_str()?.to_owned(),
                    detail.get("manifest_sha256")?.as_str()?.to_owned(),
                    detail.get("preview_sha256")?.as_str()?.to_owned(),
                ))
            })
            .collect::<BTreeSet<_>>()
    };
    let mut pair_count = 0usize;
    for candidate in &candidates {
        let claim: Option<(String, String, String)> = backlog
            .query_row(
                "SELECT state,manifest_sha256,manifest FROM claims WHERE job_id=?1",
                [&candidate.job_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()?;
        if claim.as_ref()
            != Some(&(
                "superseded_unstarted".to_owned(),
                candidate.manifest_sha256.clone(),
                candidate.manifest.clone(),
            ))
        {
            bail!("retired binding repair claim provenance mismatch");
        }
        let query = format!(
            "SELECT COUNT(*),
                    SUM(CASE WHEN state='available' AND {owner_column}='available'
                              AND receipt_sha256 IS NULL AND failure_class IS NULL
                              AND job_id=?2 AND manifest_path=?3
                             THEN 1 ELSE 0 END)
             FROM pairs INDEXED BY pair_manifest WHERE manifest_sha256=?1"
        );
        let (total, valid): (usize, Option<usize>) = pair.query_row(
            &query,
            params![
                candidate.manifest_sha256,
                candidate.job_id,
                candidate.manifest
            ],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        if total == 0 || valid.unwrap_or(0) != total {
            bail!("retired binding repair pair provenance mismatch");
        }
        if !retirement_events.contains(&(
            candidate.job_id.clone(),
            candidate.manifest_sha256.clone(),
            source_preview_sha256.clone(),
        )) {
            bail!("retired binding repair has no durable retirement event");
        }
        pair_count += total;
    }
    if audit.get("pair_updates_total").and_then(Value::as_u64) != Some(pair_count as u64) {
        bail!("retired binding repair pair count disagrees with source audit");
    }
    Ok(RetiredBindingRepairPlan {
        wave,
        source_audit_path: audit_path.to_string_lossy().into_owned(),
        source_audit_sha256: audit_sha256,
        source_preview_sha256,
        manifest_count: candidates.len(),
        pair_count,
        candidates,
    })
}

pub(crate) fn apply_retired_binding_repair(
    pair_registry: &Path,
    backlog: &Path,
    plan: &RetiredBindingRepairPlan,
    preview_sha256: &str,
) -> Result<usize> {
    let mut connection = Connection::open(pair_registry)?;
    connection.busy_timeout(Duration::from_secs(60))?;
    connection.execute(
        "ATTACH DATABASE ?1 AS backlog_db",
        [backlog.to_string_lossy().as_ref()],
    )?;
    let owner_column = if sqlite_column_exists(&connection, "pairs", "reservation_owner")? {
        "reservation_owner"
    } else {
        "owner"
    };
    let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
    transaction.execute_batch(
        "CREATE TEMP TABLE repair_manifests(
           manifest_sha256 TEXT PRIMARY KEY,job_id TEXT NOT NULL,
           manifest_path TEXT NOT NULL,detail_json TEXT NOT NULL
         ) WITHOUT ROWID;",
    )?;
    for candidate in &plan.candidates {
        let detail = serde_json::to_string(&json!({
            "job_id": candidate.job_id,
            "manifest_sha256": candidate.manifest_sha256,
            "source_audit_sha256": plan.source_audit_sha256,
            "repair_preview_sha256": preview_sha256,
        }))?;
        transaction.execute(
            "INSERT INTO repair_manifests VALUES(?1,?2,?3,?4)",
            params![
                candidate.manifest_sha256,
                candidate.job_id,
                candidate.manifest,
                detail
            ],
        )?;
    }
    let invalid_claims: usize = transaction.query_row(
        "SELECT COUNT(*) FROM repair_manifests r
         LEFT JOIN backlog_db.claims c
           ON c.job_id=r.job_id AND c.manifest_sha256=r.manifest_sha256
          AND c.manifest=r.manifest_path
         WHERE c.state IS NULL OR c.state!='superseded_unstarted'",
        [],
        |row| row.get(0),
    )?;
    if invalid_claims != 0 {
        bail!("retired binding repair claim state changed");
    }
    let update = format!(
        "UPDATE pairs INDEXED BY pair_manifest
         SET manifest_path=NULL,manifest_sha256=NULL,job_id=NULL,updated_at_ms=?1
         WHERE state='available' AND {owner_column}='available'
           AND receipt_sha256 IS NULL AND failure_class IS NULL
           AND manifest_sha256 IN(SELECT manifest_sha256 FROM repair_manifests)
           AND job_id=(SELECT r.job_id FROM repair_manifests r
                       WHERE r.manifest_sha256=pairs.manifest_sha256)
           AND manifest_path=(SELECT r.manifest_path FROM repair_manifests r
                              WHERE r.manifest_sha256=pairs.manifest_sha256)"
    );
    let updated = transaction.execute(&update, [now_ms() as i64])?;
    if updated != plan.pair_count {
        bail!("retired binding repair pair count changed before commit");
    }
    transaction.execute(
        "INSERT INTO events(kind,detail_json,created_at_ms)
         SELECT 'retired_binding_repair',detail_json,?1 FROM repair_manifests",
        [now_ms() as i64],
    )?;
    transaction.commit()?;
    Ok(updated)
}
