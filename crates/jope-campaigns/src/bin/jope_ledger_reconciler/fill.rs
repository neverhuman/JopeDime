use super::*;

/// Manifests the keeper will try to submit this cycle. These are exactly the
/// claims that `ready_claims`/`submit_claims` can pick up, so they are the only
/// ones whose missing attempts rows can abort a cycle.
pub(super) fn active_claim_manifests(
    backlog: &Connection,
    claim_states: &[String],
) -> Result<Vec<String>> {
    if claim_states.is_empty() {
        bail!("--claim-states must name at least one claim state");
    }
    if let Some(state) = claim_states
        .iter()
        .find(|state| !SUBMITTABLE_CLAIM_STATES.contains(&state.as_str()))
    {
        bail!("unsupported --claim-states value: {state}");
    }
    let placeholders = vec!["?"; claim_states.len()].join(",");
    let mut statement = backlog.prepare(&format!(
        "SELECT manifest_sha256 FROM claims WHERE state IN ({placeholders}) \
         ORDER BY manifest_sha256"
    ))?;
    let rows = statement.query_map(rusqlite::params_from_iter(claim_states), |row| {
        row.get::<_, String>(0)
    })?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

/// Collect the pairs of one manifest that are launch-ready in `pairs` but have
/// no live row in `attempts`. Read-only; the caller decides whether to write.
fn missing_attempts_for_manifest(
    registry: &Connection,
    manifest_sha256: &str,
    stats: &mut FillStats,
) -> Result<Vec<MissingAttempt>> {
    let mut pair_statement = registry.prepare_cached(
        "SELECT task,dataset_hash,semantic_sha256,protocol_sha256,state,reservation_owner,\
         manifest_path,manifest_sha256,receipt_sha256 FROM pairs WHERE manifest_sha256=?",
    )?;
    let mut active_statement = registry.prepare_cached(
        "SELECT COUNT(*) FROM attempts WHERE task=? AND dataset_hash=? AND semantic_sha256=? \
         AND protocol_sha256=? AND state IN ('reserved','submitted','infrastructure_retry')",
    )?;
    let mut evidence_statement = registry.prepare_cached(
        "SELECT COUNT(*) FROM attempts WHERE task=? AND dataset_hash=? AND semantic_sha256=? \
         AND protocol_sha256=? AND state='completed' AND manifest_sha256=?",
    )?;

    let rows = pair_statement.query_map([manifest_sha256], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
            row.get::<_, String>(5)?,
            row.get::<_, Option<String>>(6)?,
            row.get::<_, Option<String>>(7)?,
            row.get::<_, Option<String>>(8)?,
        ))
    })?;

    let mut missing = Vec::new();
    for row in rows {
        let (task, dataset_hash, semantic, protocol, state, owner, path, digest, receipt) = row?;
        stats.pairs_examined += 1;

        // Evidence already exists for this pair: leave it DONE.
        if receipt.is_some() {
            stats.skipped_evidence += 1;
            continue;
        }
        if !PAIR_LAUNCH_READY.contains(&state.as_str()) {
            stats.skipped_pair_not_launch_ready += 1;
            continue;
        }
        // The keeper matches attempts on manifest_sha256; a pair whose registry
        // row lost its manifest binding cannot be filled coherently.
        let Some(digest) = digest else {
            stats.skipped_pair_not_launch_ready += 1;
            continue;
        };
        if digest != manifest_sha256 {
            stats.skipped_pair_not_launch_ready += 1;
            continue;
        }

        let active: i64 = active_statement
            .query_row(params![task, dataset_hash, semantic, protocol], |row| {
                row.get(0)
            })?;
        if active > 0 {
            stats.skipped_already_active += 1;
            continue;
        }

        let completed: i64 = evidence_statement.query_row(
            params![task, dataset_hash, semantic, protocol, manifest_sha256],
            |row| row.get(0),
        )?;
        if completed > 0 {
            stats.skipped_evidence += 1;
            continue;
        }

        missing.push(MissingAttempt {
            task,
            dataset_hash,
            semantic_sha256: semantic,
            protocol_sha256: protocol,
            state,
            reservation_owner: owner,
            manifest_path: path,
            manifest_sha256: digest,
        });
    }
    Ok(missing)
}

pub(super) fn run_fill(
    backlog: &Connection,
    registry: &mut Connection,
    args: &Args,
    batch_id: &str,
    journal: &mut Journal,
) -> Result<FillStats> {
    let mut stats = FillStats::default();
    let manifests = active_claim_manifests(backlog, &args.claim_states)?;
    stats.candidate_manifests = manifests.len();

    let mut pending: Vec<MissingAttempt> = Vec::new();
    for manifest_sha256 in &manifests {
        if stats.inserted + pending.len() >= args.max_fill {
            stats.truncated_by_limit = true;
            break;
        }
        let missing = missing_attempts_for_manifest(registry, manifest_sha256, &mut stats)?;
        stats.manifests_completed += 1;
        pending.extend(missing);

        if pending.len() >= args.batch_rows {
            let written = flush_fill(registry, args, batch_id, journal, &mut pending)?;
            stats.inserted += written;
        }
    }
    if !pending.is_empty() {
        let written = flush_fill(registry, args, batch_id, journal, &mut pending)?;
        stats.inserted += written;
    }
    Ok(stats)
}

/// Write one bounded transaction of inserts. The journal is synced BEFORE the
/// commit so a crash can never leave an unrecorded row.
pub(super) fn flush_fill(
    registry: &mut Connection,
    args: &Args,
    batch_id: &str,
    journal: &mut Journal,
    pending: &mut Vec<MissingAttempt>,
) -> Result<usize> {
    if pending.is_empty() {
        return Ok(0);
    }
    if !args.apply {
        let count = pending.len();
        pending.clear();
        return Ok(count);
    }

    let detail = json!({ "reconciler": "wsA-three-ledger", "batch": batch_id }).to_string();
    let timestamp = now_ms() as i64;
    let mut inserted_ids: Vec<i64> = Vec::with_capacity(pending.len());

    let tx = registry.transaction_with_behavior(TransactionBehavior::Immediate)?;
    {
        let mut next_attempt = tx.prepare_cached(include_str!(
            "../../../sql/ledger_reconciler_next_attempt.sql"
        ))?;
        // Re-check inside the write transaction: another writer may have raced
        // us between the read scan and here. This keeps the lane idempotent.
        let mut recheck = tx.prepare_cached(
            "SELECT COUNT(*) FROM attempts WHERE task=? AND dataset_hash=? AND semantic_sha256=? \
             AND protocol_sha256=? AND state IN ('reserved','submitted','infrastructure_retry')",
        )?;
        let mut insert = tx.prepare_cached(
            "INSERT INTO attempts(task,dataset_hash,semantic_sha256,protocol_sha256,attempt_no,\
             state,reservation_owner,manifest_path,manifest_sha256,job_id,receipt_sha256,\
             failure_class,detail_json,created_at_ms,finalized_at_ms) \
             VALUES(?,?,?,?,?,?,?,?,?,NULL,NULL,NULL,?,?,NULL)",
        )?;

        for row in pending.iter() {
            let active: i64 = recheck.query_row(
                params![
                    row.task,
                    row.dataset_hash,
                    row.semantic_sha256,
                    row.protocol_sha256
                ],
                |value| value.get(0),
            )?;
            if active > 0 {
                continue;
            }
            let attempt_no: i64 = next_attempt.query_row(
                params![
                    row.task,
                    row.dataset_hash,
                    row.semantic_sha256,
                    row.protocol_sha256
                ],
                |value| value.get(0),
            )?;
            insert.execute(params![
                row.task,
                row.dataset_hash,
                row.semantic_sha256,
                row.protocol_sha256,
                attempt_no,
                row.state,
                row.reservation_owner,
                row.manifest_path,
                row.manifest_sha256,
                detail,
                timestamp
            ])?;
            inserted_ids.push(tx.last_insert_rowid());
        }
    }

    for id in &inserted_ids {
        journal.record(&json!({ "t": "attempt_insert", "id": id }))?;
    }
    journal.sync()?;
    tx.commit()?;

    pending.clear();
    Ok(inserted_ids.len())
}
