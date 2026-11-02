use super::*;

pub(super) fn recover_retirement_candidates(
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

pub(super) fn build_retired_binding_repair_plan(
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

pub(super) fn apply_retired_binding_repair(
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

#[allow(clippy::too_many_arguments)]
pub(super) fn build_missing_receipt_recovery_plan(
    root: &Path,
    wave_dir: &Path,
    pair_registry: &Path,
    backlog: &Path,
    scheduler: &HashMap<String, (String, String)>,
    receipt_dir: &Path,
    scq_events: &Path,
    runs_root: &Path,
    scq: &Path,
    socket: &Path,
) -> Result<MissingReceiptRecoveryPlan> {
    let claims = load_claims(backlog)?;
    // Reuse the strict blocker audit to validate the complete wave, current
    // checkpoint, immutable manifests, and exact pair/claim bindings.
    let _validated = build_wave_blocker_plan(
        root,
        wave_dir,
        pair_registry,
        &claims,
        scheduler,
        receipt_dir,
        scq_events,
    )?;
    let (wave, desired, current_manifest_hashes) = load_wave_desired_pairs(wave_dir)?;
    let pair = Connection::open_with_flags(
        pair_registry,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    pair.busy_timeout(Duration::from_secs(60))?;
    let mut lookup = pair.prepare(
        "SELECT state,manifest_path,manifest_sha256,job_id,receipt_sha256,failure_class
         FROM pairs
         WHERE task=?1 AND dataset_hash=?2 AND semantic_sha256=?3 AND protocol_sha256=?4",
    )?;
    let mut blockers = BTreeMap::<String, (String, String, Vec<RegistryPairIdentity>, bool)>::new();
    for identity in &desired {
        type PairRecoveryRow = (
            String,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
        );
        let row: Option<PairRecoveryRow> = lookup
            .query_row(
                params![
                    identity.task,
                    identity.dataset_hash,
                    identity.semantic_sha256,
                    identity.protocol_sha256
                ],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .optional()?;
        let Some((state, path, manifest, job, receipt, failure)) = row else {
            continue;
        };
        if !matches!(state.as_str(), "reserved" | "submitted" | "held") || receipt.is_some() {
            continue;
        }
        let (Some(path), Some(manifest), Some(job)) = (path, manifest, job) else {
            continue;
        };
        if current_manifest_hashes.contains(&manifest) {
            continue;
        }
        let entry = blockers
            .entry(manifest.clone())
            .or_insert_with(|| (job.clone(), path.clone(), Vec::new(), false));
        if entry.0 != job || entry.1 != path {
            bail!("missing-receipt blocker manifest binding is inconsistent");
        }
        entry.2.push(identity.clone());
        entry.3 |= failure.is_some();
    }
    drop(lookup);
    let mut manifest_pairs_lookup = pair.prepare(
        "SELECT task,dataset_hash,semantic_sha256,protocol_sha256,
                state,manifest_path,job_id,receipt_sha256,failure_class
         FROM pairs INDEXED BY pair_manifest
         WHERE manifest_sha256=?1
         ORDER BY task,dataset_hash,semantic_sha256,protocol_sha256",
    )?;

    let jobs = blockers
        .values()
        .map(|(job, _, _, _)| job.clone())
        .collect::<BTreeSet<_>>();
    let start_evidence = load_scq_start_evidence(scq_events, &jobs)?;
    let corpus = Connection::open_with_flags(
        root.join("corpus/strict-test-exact.sqlite"),
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    let mut reason_counts = BTreeMap::new();
    let mut exclusions = Vec::new();
    let mut candidates = Vec::new();
    for (manifest_sha256, (job_id, manifest_path, pairs, has_failure)) in blockers.iter() {
        let exclude = |reason: String,
                       reason_counts: &mut BTreeMap<String, usize>,
                       exclusions: &mut Vec<String>| {
            let class = reason.split(':').next().unwrap_or("unknown").to_owned();
            *reason_counts.entry(class).or_insert(0) += 1;
            exclusions.push(format!("{job_id}:{reason}"));
        };
        let Some(claim) = claims.get(job_id) else {
            bail!("missing-receipt blocker claim disappeared");
        };
        if claim.manifest_sha256 != *manifest_sha256
            || claim.manifest.to_string_lossy() != *manifest_path
        {
            bail!("missing-receipt blocker claim hash/path mismatch");
        }
        if !(claim.state == "submitted" || claim.state.starts_with("ingest_failed:")) {
            exclude(
                format!("claim_state_not_retryable:{}", claim.state),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        let old_manifest = match validate_manifest(claim) {
            Ok(manifest) => manifest,
            Err(error) => {
                exclude(
                    format!("manifest_provenance_invalid:{error:#}"),
                    &mut reason_counts,
                    &mut exclusions,
                );
                continue;
            }
        };
        if strict_wave(&old_manifest)
            .is_some_and(|old_wave| wave_number(&old_wave) >= wave_number(&wave))
        {
            exclude(
                "current_or_future_wave_binding".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        if *has_failure {
            exclude(
                "semantic_or_terminal_pair_failure".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        let Some(evidence) = start_evidence.get(job_id) else {
            exclude(
                "no_durable_launch_or_completion_evidence".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        };
        if scheduler.get(job_id).is_some_and(|(_, state)| {
            matches!(
                state.as_str(),
                "queued" | "starting" | "held" | "dispatching" | "running"
            )
        }) {
            exclude(
                "current_active_scheduler_state".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        let ingested: bool = corpus.query_row(
            "SELECT EXISTS(SELECT 1 FROM ingested_jobs WHERE job_id=?1)",
            [job_id],
            |row| row.get(0),
        )?;
        if ingested {
            exclude(
                "authoritative_ingested_receipt".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        let receipt_root = root.join("receipts").join(job_id);
        if tree_contains_named_file(&receipt_root, "SHARD_OK")? {
            exclude(
                "authoritative_local_receipt".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        if tree_contains_named_file(&receipt_root, "SHARD_FAIL.json")?
            || tree_contains_named_file(&runs_root.join(job_id), "SHARD_FAIL.json")?
        {
            exclude(
                "semantic_terminal_failure_output".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        if tree_contains_named_file(&runs_root.join(job_id), "SHARD_OK")? {
            exclude(
                "recoverable_run_receipt_exists".to_owned(),
                &mut reason_counts,
                &mut exclusions,
            );
            continue;
        }
        match probe_scq_output(scq, socket, job_id)? {
            OutputProbe::AuthoritativeReceipt => {
                exclude(
                    "recoverable_scq_receipt_exists".to_owned(),
                    &mut reason_counts,
                    &mut exclusions,
                );
                continue;
            }
            OutputProbe::SemanticFailure => {
                exclude(
                    "semantic_terminal_failure_output".to_owned(),
                    &mut reason_counts,
                    &mut exclusions,
                );
                continue;
            }
            OutputProbe::Unverified(error) => {
                exclude(
                    format!("output_unavailability_unverified:{error}"),
                    &mut reason_counts,
                    &mut exclusions,
                );
                continue;
            }
            OutputProbe::Unavailable(reason) => {
                let rows = manifest_pairs_lookup
                    .query_map([manifest_sha256], |row| {
                        Ok(ManifestPairBinding {
                            identity: RegistryPairIdentity {
                                task: row.get(0)?,
                                dataset_hash: row.get(1)?,
                                semantic_sha256: row.get(2)?,
                                protocol_sha256: row.get(3)?,
                            },
                            state: row.get(4)?,
                            manifest_path: row.get(5)?,
                            job_id: row.get(6)?,
                            receipt_sha256: row.get(7)?,
                            failure_class: row.get(8)?,
                        })
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                let Some(all_pairs) =
                    validate_missing_receipt_manifest_bindings(rows, manifest_path, job_id)
                else {
                    exclude(
                        "manifest_contains_authoritative_or_nonactive_pair".to_owned(),
                        &mut reason_counts,
                        &mut exclusions,
                    );
                    continue;
                };
                *reason_counts
                    .entry("eligible_infrastructure_retry_missing_receipt".to_owned())
                    .or_insert(0) += 1;
                candidates.push(MissingReceiptRecoveryCandidate {
                    job_id: job_id.clone(),
                    manifest: manifest_path.clone(),
                    manifest_sha256: manifest_sha256.clone(),
                    claim_state: claim.state.clone(),
                    scq_evidence: format!("{evidence}; {reason}"),
                    desired_pair_count: pairs.len(),
                    pair_count: all_pairs.len(),
                    pairs: all_pairs,
                });
            }
        }
    }
    drop(manifest_pairs_lookup);
    drop(pair);
    candidates.sort_by(|left, right| left.job_id.cmp(&right.job_id));
    exclusions.sort();
    let blocker_pairs = blockers.values().map(|entry| entry.2.len()).sum();
    let eligible_desired_pairs = candidates
        .iter()
        .map(|candidate| candidate.desired_pair_count)
        .sum();
    let eligible_total_pairs = candidates
        .iter()
        .map(|candidate| candidate.pair_count)
        .sum();
    Ok(MissingReceiptRecoveryPlan {
        wave,
        wave_dir: fs::canonicalize(wave_dir)?.to_string_lossy().into_owned(),
        desired_pairs: desired.len(),
        blocker_pairs,
        blocker_manifests: blockers.len(),
        eligible_desired_pairs,
        eligible_total_pairs,
        eligible_manifests: candidates.len(),
        reason_counts,
        exclusions,
        candidates,
    })
}

pub(super) fn validate_missing_receipt_manifest_bindings(
    rows: Vec<ManifestPairBinding>,
    manifest_path: &str,
    job_id: &str,
) -> Option<Vec<RegistryPairIdentity>> {
    if rows.is_empty()
        || rows.iter().any(|row| {
            !matches!(row.state.as_str(), "reserved" | "submitted" | "held")
                || row.manifest_path.as_deref() != Some(manifest_path)
                || row.job_id.as_deref() != Some(job_id)
                || row.receipt_sha256.is_some()
                || row.failure_class.is_some()
        })
    {
        return None;
    }
    Some(rows.into_iter().map(|row| row.identity).collect())
}

pub(super) fn load_wave_desired_pairs(
    wave_dir: &Path,
) -> Result<(String, BTreeSet<RegistryPairIdentity>, BTreeSet<String>)> {
    let wave_dir = fs::canonicalize(wave_dir)?;
    let wave = wave_dir
        .file_name()
        .and_then(|name| name.to_str())
        .context("wave directory name is invalid")?
        .to_owned();
    let summary: Value = serde_json::from_slice(&fs::read(wave_dir.join("summary.json"))?)?;
    let entries = summary
        .get("manifests")
        .and_then(Value::as_array)
        .context("wave summary manifests are absent")?;
    let mut desired = BTreeSet::new();
    let mut hashes = BTreeSet::new();
    for entry in entries {
        let path = wave_dir.join(required_json_str(entry, "relative_path")?);
        let expected_hash = required_json_str(entry, "sha256")?;
        if sha256_file(&path)? != expected_hash {
            bail!("wave desired manifest hash drift");
        }
        let manifest: Value = serde_json::from_slice(&fs::read(path)?)?;
        let task = required_json_str(&manifest, "task")?;
        let dataset = required_json_str(&manifest, "dataset_hash")?;
        let protocol = required_json_str(&manifest, "protocol_sha256")?;
        for candidate in manifest
            .get("candidates")
            .and_then(Value::as_array)
            .context("wave desired candidates are absent")?
        {
            desired.insert(RegistryPairIdentity {
                task: task.to_owned(),
                dataset_hash: dataset.to_owned(),
                semantic_sha256: required_json_str(candidate, "semantic_sha256")?.to_owned(),
                protocol_sha256: protocol.to_owned(),
            });
        }
        hashes.insert(expected_hash.to_owned());
    }
    if desired.len() % 256 != 0 || !matches!(desired.len() / 256, 128 | 192 | 256) {
        bail!(
            "wave desired identity count {} is not a legal micro-wave total",
            desired.len()
        );
    }
    Ok((wave, desired, hashes))
}

pub(super) enum OutputProbe {
    AuthoritativeReceipt,
    SemanticFailure,
    Unavailable(String),
    Unverified(String),
}

pub(super) fn probe_scq_output(scq: &Path, socket: &Path, job_id: &str) -> Result<OutputProbe> {
    let temporary = ProbeDirectory::create()?;
    let output = Command::new(scq)
        .arg("fetch")
        .arg(job_id)
        .arg("--output")
        .arg(&temporary.path)
        .arg("--socket")
        .arg(socket)
        .output()
        .with_context(|| format!("probe SCQ output for {job_id}"))?;
    if output.status.success() {
        if tree_contains_named_file(&temporary.path, "SHARD_OK")? {
            return Ok(OutputProbe::AuthoritativeReceipt);
        }
        if tree_contains_named_file(&temporary.path, "SHARD_FAIL.json")? {
            return Ok(OutputProbe::SemanticFailure);
        }
        return Ok(OutputProbe::Unavailable(
            "SCQ fetch returned no authoritative marker".to_owned(),
        ));
    }
    let error = String::from_utf8_lossy(if output.stderr.is_empty() {
        &output.stdout
    } else {
        &output.stderr
    })
    .trim()
    .to_owned();
    if error.contains("no such job")
        || error.contains("output is unavailable")
        || error.contains("no outputs")
    {
        Ok(OutputProbe::Unavailable(error))
    } else {
        Ok(OutputProbe::Unverified(error))
    }
}

pub(super) struct ProbeDirectory {
    path: PathBuf,
}

impl ProbeDirectory {
    fn create() -> Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "jope-scq-output-probe-{}-{}-{}",
            std::process::id(),
            now_ms(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path)?;
        Ok(Self { path })
    }
}

impl Drop for ProbeDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

pub(super) fn tree_contains_named_file(root: &Path, name: &str) -> Result<bool> {
    if !root.exists() {
        return Ok(false);
    }
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            bail!("output proof tree contains a symlink");
        }
        if kind.is_dir() {
            if tree_contains_named_file(&entry.path(), name)? {
                return Ok(true);
            }
        } else if entry.file_name() == name {
            return Ok(true);
        }
    }
    Ok(false)
}
